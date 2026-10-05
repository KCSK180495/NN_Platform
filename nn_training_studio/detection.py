"""Detection for NN Training Studio."""

from pathlib import Path
from datetime import datetime
import json
import numpy as np
import os
import platform
import shutil
import tempfile
import zipfile
from nn_training_studio.constants import (
    APP_SETTINGS_DIRECTORY,
    APP_VERSION,
    OBJECT_DETECTION_RUN_DIRECTORY,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from nn_training_studio.model_io import (
    safely_extract_zip,
)
from nn_training_studio.results import (
    _result_json_default,
)


def _object_detection_run_root():
    """Return the per-user folder used for YOLO runs and predictions."""
    system_name = platform.system().lower()
    if system_name == "windows":
        base_directory = Path(
            os.environ.get("LOCALAPPDATA", Path.home())
        )
    elif system_name == "darwin":
        base_directory = Path.home() / "Library" / "Application Support"
    else:
        base_directory = Path(
            os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
        )
    run_root = (
        base_directory / APP_SETTINGS_DIRECTORY / OBJECT_DETECTION_RUN_DIRECTORY
    )
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def _load_yaml_mapping(yaml_path):
    """Load a YOLO data YAML while keeping PyYAML an optional dependency."""
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(
            "Reading data.yaml requires PyYAML. Install object-detection "
            "support with: pip install ultralytics pyyaml pillow"
        ) from exc

    with open(yaml_path, "r", encoding="utf-8") as file:
        content = yaml.safe_load(file)
    if not isinstance(content, dict):
        raise ValueError("data.yaml must contain a YAML mapping.")
    return content


def _normalise_yolo_class_names(config):
    names = config.get("names")
    if isinstance(names, list):
        result = [str(value) for value in names]
    elif isinstance(names, dict):
        try:
            pairs = sorted(
                ((int(key), str(value)) for key, value in names.items()),
                key=lambda item: item[0],
            )
        except Exception as exc:
            raise ValueError(
                "YOLO class-name keys must be integer class IDs."
            ) from exc
        result = [name for _, name in pairs]
        expected = list(range(len(result)))
        actual = [index for index, _ in pairs]
        if actual != expected:
            raise ValueError(
                "YOLO class IDs in data.yaml must be consecutive from 0."
            )
    else:
        nc = config.get("nc")
        if nc is None:
            raise ValueError(
                "data.yaml must define 'names' or an integer 'nc'."
            )
        try:
            result = [f"Class {index}" for index in range(int(nc))]
        except Exception as exc:
            raise ValueError("'nc' in data.yaml must be an integer.") from exc
    if not result:
        raise ValueError("At least one object class is required.")
    return result


def _resolve_yolo_path(path_value, yaml_directory, dataset_root):
    candidate = Path(os.path.expandvars(os.path.expanduser(str(path_value))))
    if candidate.is_absolute():
        return candidate.resolve()
    root_candidate = (dataset_root / candidate).resolve()
    if root_candidate.exists():
        return root_candidate
    return (yaml_directory / candidate).resolve()


def _collect_yolo_split_images(split_value, yaml_directory, dataset_root):
    """Resolve a YOLO split entry containing a directory, text file, or list."""
    values = split_value if isinstance(split_value, list) else [split_value]
    images = []
    sources = []
    for value in values:
        source_path = _resolve_yolo_path(
            value,
            yaml_directory,
            dataset_root,
        )
        sources.append(str(source_path))
        if source_path.is_dir():
            images.extend(
                path.resolve()
                for path in source_path.rglob("*")
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            )
        elif source_path.is_file() and source_path.suffix.lower() == ".txt":
            for raw_line in source_path.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines():
                item = raw_line.strip()
                if not item:
                    continue
                image_path = Path(
                    os.path.expandvars(os.path.expanduser(item))
                )
                if not image_path.is_absolute():
                    from_text = (source_path.parent / image_path).resolve()
                    from_root = (dataset_root / image_path).resolve()
                    image_path = (
                        from_text if from_text.exists() else from_root
                    )
                images.append(image_path.resolve())
        elif source_path.is_file():
            if source_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                images.append(source_path.resolve())
        else:
            raise ValueError(
                f"YOLO split path does not exist: {source_path}"
            )
    unique_images = sorted(dict.fromkeys(images), key=lambda value: str(value))
    return unique_images, sources


def _candidate_yolo_label_path(image_path, dataset_root):
    """Map a conventional images/... path to labels/... with .txt suffix."""
    image_path = Path(image_path).resolve()
    parts = list(image_path.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        index = lowered.index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    try:
        relative = image_path.relative_to(dataset_root)
        relative_parts = list(relative.parts)
        if relative_parts and relative_parts[0].lower() in {
            "train",
            "val",
            "valid",
            "test",
        }:
            return (
                dataset_root / "labels" / Path(*relative_parts)
            ).with_suffix(".txt")
    except ValueError:
        pass
    return image_path.with_suffix(".txt")


def _parse_yolo_box_line(
    line,
    line_number,
    annotation_path,
    class_count,
):
    tokens = line.split()
    if len(tokens) != 5:
        raise ValueError(
            f"{annotation_path.name}:{line_number} has {len(tokens)} values; "
            "object detection requires: class x_center y_center width height."
        )
    try:
        class_value = float(tokens[0])
        class_id = int(class_value)
        coordinates = [float(value) for value in tokens[1:]]
    except Exception as exc:
        raise ValueError(
            f"{annotation_path.name}:{line_number} contains non-numeric values."
        ) from exc
    if class_value != class_id:
        raise ValueError(
            f"{annotation_path.name}:{line_number} class ID must be an integer."
        )
    if class_id < 0 or class_id >= class_count:
        raise ValueError(
            f"{annotation_path.name}:{line_number} class ID {class_id} is "
            f"outside 0–{class_count - 1}."
        )
    x_center, y_center, width, height = coordinates
    if not all(np.isfinite(coordinates)):
        raise ValueError(
            f"{annotation_path.name}:{line_number} contains NaN or infinity."
        )
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
        raise ValueError(
            f"{annotation_path.name}:{line_number} box centre must be "
            "normalised to 0–1."
        )
    if not (0 < width <= 1 and 0 < height <= 1):
        raise ValueError(
            f"{annotation_path.name}:{line_number} width and height must be "
            "greater than 0 and no more than 1."
        )
    tolerance = 1e-6
    if (
        x_center - width / 2 < -tolerance
        or x_center + width / 2 > 1 + tolerance
        or y_center - height / 2 < -tolerance
        or y_center + height / 2 > 1 + tolerance
    ):
        raise ValueError(
            f"{annotation_path.name}:{line_number} extends outside the image."
        )
    return {
        "class_id": class_id,
        "x_center": x_center,
        "y_center": y_center,
        "width": width,
        "height": height,
    }


def validate_yolo_dataset(data_yaml_path, maximum_error_examples=100):
    """
    Validate an Ultralytics object-detection dataset.

    Missing/empty label files are retained as valid background images, but they
    are reported separately so users can distinguish intended negatives from
    accidentally unlabelled photos.
    """
    yaml_path = Path(data_yaml_path).expanduser().resolve()
    if not yaml_path.is_file():
        raise ValueError("Select an existing YOLO data.yaml file.")
    config = _load_yaml_mapping(yaml_path)
    yaml_directory = yaml_path.parent
    configured_root = config.get("path", ".")
    dataset_root = _resolve_yolo_path(
        configured_root,
        yaml_directory,
        yaml_directory,
    )
    if not dataset_root.exists():
        raise ValueError(
            f"The dataset root defined by 'path' does not exist: {dataset_root}"
        )
    class_names = _normalise_yolo_class_names(config)
    if "train" not in config or "val" not in config:
        raise ValueError("data.yaml must define both 'train' and 'val' splits.")

    report = {
        "schema_version": 1,
        "validated_at": datetime.now().isoformat(timespec="seconds"),
        "data_yaml": str(yaml_path),
        "dataset_root": str(dataset_root),
        "class_names": class_names,
        "class_count": len(class_names),
        "splits": {},
        "total_images": 0,
        "total_instances": 0,
        "missing_label_files": 0,
        "empty_label_files": 0,
        "invalid_annotation_count": 0,
        "invalid_examples": [],
        "class_instance_counts": {
            str(index): 0 for index in range(len(class_names))
        },
    }
    records = []

    for split_name in ("train", "val", "test"):
        if split_name not in config or config.get(split_name) in (None, ""):
            continue
        image_paths, sources = _collect_yolo_split_images(
            config[split_name],
            yaml_directory,
            dataset_root,
        )
        if not image_paths:
            raise ValueError(
                f"The '{split_name}' split contains no supported images."
            )
        split_record = {
            "sources": sources,
            "image_count": len(image_paths),
            "labelled_images": 0,
            "background_images": 0,
            "missing_label_files": 0,
            "empty_label_files": 0,
            "instances": 0,
            "invalid_annotations": 0,
        }
        for image_path in image_paths:
            label_path = _candidate_yolo_label_path(
                image_path,
                dataset_root,
            )
            boxes = []
            errors = []
            label_state = "labelled"
            if not image_path.is_file():
                errors.append("Image file does not exist.")
            if not label_path.is_file():
                label_state = "missing"
                split_record["missing_label_files"] += 1
                report["missing_label_files"] += 1
            else:
                lines = [
                    line.strip()
                    for line in label_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    ).splitlines()
                    if line.strip()
                ]
                if not lines:
                    label_state = "empty"
                    split_record["empty_label_files"] += 1
                    report["empty_label_files"] += 1
                for line_number, line in enumerate(lines, start=1):
                    try:
                        box = _parse_yolo_box_line(
                            line,
                            line_number,
                            label_path,
                            len(class_names),
                        )
                        boxes.append(box)
                        report["class_instance_counts"][
                            str(box["class_id"])
                        ] += 1
                    except ValueError as exc:
                        errors.append(str(exc))
            if boxes:
                split_record["labelled_images"] += 1
            else:
                split_record["background_images"] += 1
            split_record["instances"] += len(boxes)
            split_record["invalid_annotations"] += len(errors)
            report["total_instances"] += len(boxes)
            report["invalid_annotation_count"] += len(errors)
            for error_message in errors:
                if len(report["invalid_examples"]) < maximum_error_examples:
                    report["invalid_examples"].append(
                        {
                            "split": split_name,
                            "image": str(image_path),
                            "label": str(label_path),
                            "error": error_message,
                        }
                    )
            records.append(
                {
                    "split": split_name,
                    "image_path": str(image_path),
                    "label_path": str(label_path),
                    "label_state": label_state,
                    "instance_count": len(boxes),
                    "boxes": boxes,
                    "errors": errors,
                }
            )
        report["splits"][split_name] = split_record
        report["total_images"] += len(image_paths)

    report["is_valid"] = report["invalid_annotation_count"] == 0
    report["warnings"] = []
    if report["missing_label_files"]:
        report["warnings"].append(
            "Missing label files are treated as background images. Confirm "
            "that these are intentional negative examples."
        )
    if report["empty_label_files"]:
        report["warnings"].append(
            "Empty label files are treated as background images."
        )
    missing_classes = [
        class_names[int(index)]
        for index, count in report["class_instance_counts"].items()
        if int(count) == 0
    ]
    if missing_classes:
        report["warnings"].append(
            "No annotated instances were found for: "
            + ", ".join(missing_classes)
        )
    return config, report, records


def format_yolo_validation_report(report):
    status = "VALID" if report.get("is_valid") else "INVALID"
    lines = [
        f"YOLO Dataset Validation: {status}",
        f"Configuration: {report.get('data_yaml')}",
        f"Dataset root: {report.get('dataset_root')}",
        f"Classes ({report.get('class_count')}): "
        + ", ".join(report.get("class_names", [])),
        "",
    ]
    for split_name, split in report.get("splits", {}).items():
        lines.extend(
            [
                f"{split_name.upper()}",
                f"  Images: {split.get('image_count', 0)}",
                f"  Labelled images: {split.get('labelled_images', 0)}",
                f"  Background images: {split.get('background_images', 0)}",
                f"  Object instances: {split.get('instances', 0)}",
                f"  Invalid annotations: "
                f"{split.get('invalid_annotations', 0)}",
            ]
        )
    lines.extend(["", "Instances per class:"])
    for index, class_name in enumerate(report.get("class_names", [])):
        count = report.get("class_instance_counts", {}).get(str(index), 0)
        lines.append(f"  {index}: {class_name} = {count}")
    if report.get("warnings"):
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {item}" for item in report["warnings"])
    if report.get("invalid_examples"):
        lines.append("")
        lines.append("Invalid annotation examples:")
        lines.extend(
            f"  - [{item['split']}] {item['error']} ({item['label']})"
            for item in report["invalid_examples"][:20]
        )
    return "\n".join(lines)


def create_yolo_preview_image(record, class_names, destination_path):
    """Draw one ground-truth annotation preview without OpenCV."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        raise RuntimeError(
            "Annotation preview requires Pillow: pip install pillow"
        ) from exc
    image_path = Path(record["image_path"])
    if not image_path.is_file():
        raise ValueError("The selected preview image no longer exists.")
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    palette = [
        "#00ff66",
        "#00b7ff",
        "#ffcc00",
        "#ff4d8d",
        "#c77dff",
        "#ff7b00",
    ]
    font = ImageFont.load_default()
    for box in record.get("boxes", []):
        class_id = int(box["class_id"])
        x_center = box["x_center"] * width
        y_center = box["y_center"] * height
        box_width = box["width"] * width
        box_height = box["height"] * height
        x1 = max(0, x_center - box_width / 2)
        y1 = max(0, y_center - box_height / 2)
        x2 = min(width - 1, x_center + box_width / 2)
        y2 = min(height - 1, y_center + box_height / 2)
        colour = palette[class_id % len(palette)]
        draw.rectangle((x1, y1, x2, y2), outline=colour, width=3)
        label = (
            class_names[class_id]
            if 0 <= class_id < len(class_names)
            else str(class_id)
        )
        text_box = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle(text_box, fill=colour)
        draw.text((x1, y1), label, fill="black", font=font)
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return str(destination)


def _find_detection_weights(run_directory, filename="best.pt"):
    run_path = Path(run_directory)
    candidates = list(run_path.rglob(filename))
    return str(candidates[0]) if candidates else None


def save_detection_result_bundle(
    save_path,
    run_directory,
    data_yaml_path,
    training_config,
    validation_report,
    inference_directories=None,
    custom_result_directories=None,
):
    """Save a complete, reloadable object-detection result ZIP."""
    run_path = Path(run_directory).resolve()
    if not run_path.is_dir():
        raise ValueError("The YOLO run directory does not exist.")
    best_path = _find_detection_weights(run_path, "best.pt")
    last_path = _find_detection_weights(run_path, "last.pt")
    manifest = {
        "schema_version": 1,
        "application_version": APP_VERSION,
        "package_type": "yolo_object_detection",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_yaml": "data.yaml",
        "best_weights": "weights/best.pt" if best_path else None,
        "last_weights": "weights/last.pt" if last_path else None,
        "training_config": "training_config.json",
        "dataset_validation": "dataset_validation.json",
        "class_names": validation_report.get("class_names", []),
        "ultralytics_required": True,
    }
    save_path = Path(save_path)
    temp_directory = Path(tempfile.mkdtemp(prefix="yolo_result_bundle_"))
    try:
        (temp_directory / "training_config.json").write_text(
            json.dumps(
                training_config,
                indent=2,
                default=_result_json_default,
            ),
            encoding="utf-8",
        )
        (temp_directory / "dataset_validation.json").write_text(
            json.dumps(
                validation_report,
                indent=2,
                default=_result_json_default,
            ),
            encoding="utf-8",
        )
        (temp_directory / "dataset_validation.txt").write_text(
            format_yolo_validation_report(validation_report),
            encoding="utf-8",
        )
        shutil.copy2(data_yaml_path, temp_directory / "data.yaml")
        (temp_directory / "detection_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        with zipfile.ZipFile(save_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for metadata_file in temp_directory.iterdir():
                archive.write(metadata_file, arcname=metadata_file.name)
            for result_file in sorted(run_path.rglob("*")):
                if not result_file.is_file():
                    continue
                if result_file.resolve() == save_path.resolve():
                    continue
                if best_path and result_file.resolve() == Path(best_path):
                    archive.write(result_file, arcname="weights/best.pt")
                elif last_path and result_file.resolve() == Path(last_path):
                    archive.write(result_file, arcname="weights/last.pt")
                else:
                    archive.write(
                        result_file,
                        arcname=str(
                            Path("training_results")
                            / result_file.relative_to(run_path)
                        ),
                    )
            for prediction_directory in inference_directories or []:
                prediction_path = Path(prediction_directory)
                if not prediction_path.is_dir():
                    continue
                for prediction_file in prediction_path.rglob("*"):
                    if prediction_file.is_file():
                        archive.write(
                            prediction_file,
                            arcname=str(
                                Path("predictions")
                                / prediction_path.name
                                / prediction_file.relative_to(prediction_path)
                            ),
                        )
            for index, custom_directory in enumerate(
                custom_result_directories or [],
                start=1,
            ):
                custom_path = Path(custom_directory)
                if not custom_path.is_dir():
                    continue
                for custom_file in custom_path.rglob("*"):
                    if custom_file.is_file():
                        archive.write(
                            custom_file,
                            arcname=str(
                                Path("custom_results")
                                / f"custom_result_{index:02d}"
                                / custom_file.relative_to(custom_path)
                            ),
                        )
        return manifest
    finally:
        shutil.rmtree(temp_directory, ignore_errors=True)


def load_detection_result_bundle(package_path):
    """Extract and validate an NN Studio object-detection result ZIP."""
    temp_directory = tempfile.mkdtemp(prefix="yolo_detection_package_")
    try:
        safely_extract_zip(package_path, temp_directory)
        root = Path(temp_directory)
        manifest_path = root / "detection_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(
                "This ZIP is not an NN Studio object-detection package."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("package_type") != "yolo_object_detection":
            raise ValueError("Unsupported detection package type.")
        best_relative = manifest.get("best_weights")
        last_relative = manifest.get("last_weights")
        best_path = root / best_relative if best_relative else None
        last_path = root / last_relative if last_relative else None
        weights_path = (
            best_path
            if best_path is not None and best_path.is_file()
            else last_path
        )
        if weights_path is None or not weights_path.is_file():
            raise ValueError("The detection package has no usable .pt weights.")
        config_path = root / "training_config.json"
        validation_path = root / "dataset_validation.json"
        return {
            "temp_directory": temp_directory,
            "manifest": manifest,
            "weights_path": str(weights_path),
            "data_yaml_path": str(root / "data.yaml"),
            "training_config": (
                json.loads(config_path.read_text(encoding="utf-8"))
                if config_path.is_file()
                else {}
            ),
            "validation_report": (
                json.loads(validation_path.read_text(encoding="utf-8"))
                if validation_path.is_file()
                else {}
            ),
        }
    except Exception:
        shutil.rmtree(temp_directory, ignore_errors=True)
        raise
