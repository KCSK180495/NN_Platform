"""Annotations for NN Training Studio."""

from pathlib import Path
from datetime import datetime
import hashlib
import json
import numpy as np
import pandas as pd
import shutil
from nn_training_studio.constants import (
    ANNOTATION_STATUS_APPROVED,
)


def _new_annotation_id(prefix="ann"):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"


def validate_signal_annotation(annotation, row_count):
    """Return a normalized interval/event annotation."""
    if not isinstance(annotation, dict):
        raise ValueError("Annotation must be a dictionary.")
    kind = annotation.get("kind")
    if kind not in {"interval", "event"}:
        raise ValueError("Signal annotation kind must be interval or event.")
    label = " ".join(str(annotation.get("label", "")).split()).strip()
    if not label:
        raise ValueError("Enter a label name first.")
    if int(row_count) <= 0:
        raise ValueError("The signal dataset is empty.")
    start = int(annotation.get("start", 0))
    end = int(annotation.get("end", start))
    if start > end:
        start, end = end, start
    if start < 0 or end >= int(row_count):
        raise ValueError(
            f"Annotation range must stay between row 0 and {int(row_count) - 1}."
        )
    if kind == "event":
        end = start
    confidence = float(annotation.get("confidence", 1.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Annotation confidence must be between 0 and 1.")
    return {
        "id": str(annotation.get("id") or _new_annotation_id()),
        "data_type": "signal",
        "kind": kind,
        "label": label,
        "start": start,
        "end": end,
        "source": str(annotation.get("source") or "Manual"),
        "confidence": confidence,
        "status": str(annotation.get("status") or ANNOTATION_STATUS_APPROVED),
        "notes": str(annotation.get("notes") or "")[:1000],
        "created_at": str(annotation.get("created_at") or datetime.now().isoformat()),
    }


def validate_box_annotation(annotation):
    """Validate normalized [0, 1] image bounding-box coordinates."""
    if not isinstance(annotation, dict):
        raise ValueError("Bounding-box annotation must be a dictionary.")
    label = " ".join(str(annotation.get("label", "")).split()).strip()
    if not label:
        raise ValueError("Enter a bounding-box class first.")
    image_path = str(annotation.get("image_path") or "").strip()
    if not image_path:
        raise ValueError("No image is selected for this bounding box.")
    coordinates = [float(annotation.get(key, 0.0)) for key in ("x", "y", "w", "h")]
    x, y, width, height = coordinates
    if width <= 0 or height <= 0:
        raise ValueError("Draw a bounding box with positive width and height.")
    if not all(0.0 <= value <= 1.0 for value in coordinates):
        raise ValueError("Bounding-box coordinates must be normalized to 0–1.")
    if x + width > 1.000001 or y + height > 1.000001:
        raise ValueError("The bounding box extends outside the image.")
    return {
        "id": str(annotation.get("id") or _new_annotation_id("box")),
        "data_type": "image",
        "kind": "box",
        "label": label,
        "image_path": image_path,
        "x": x,
        "y": y,
        "w": width,
        "h": height,
        "source": str(annotation.get("source") or "Manual"),
        "confidence": float(annotation.get("confidence", 1.0)),
        "status": str(annotation.get("status") or ANNOTATION_STATUS_APPROVED),
        "notes": str(annotation.get("notes") or "")[:1000],
        "created_at": str(annotation.get("created_at") or datetime.now().isoformat()),
    }


def write_yolo_annotation_dataset(output_root, annotations):
    """Write an approved, deterministic train/validation YOLO dataset."""
    boxes = [
        validate_box_annotation(item)
        for item in annotations
        if item.get("kind") == "box"
        and item.get("status") == ANNOTATION_STATUS_APPROVED
    ]
    if not boxes:
        raise ValueError("No approved bounding boxes are available.")
    by_image = {}
    for item in boxes:
        if not Path(item["image_path"]).is_file():
            raise ValueError(f"Annotated image is missing: {item['image_path']}")
        by_image.setdefault(item["image_path"], []).append(item)
    ordered_sources = sorted(
        by_image,
        key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
    )
    if len(ordered_sources) < 2:
        raise ValueError(
            "At least two annotated images are required to create separate train and validation splits."
        )
    root = Path(output_root)
    class_names = sorted({item["label"] for item in boxes})
    class_map = {name: index for index, name in enumerate(class_names)}
    validation_count = max(1, int(round(len(ordered_sources) * 0.20)))
    validation_sources = set(ordered_sources[:validation_count])
    split_counts = {"train": 0, "val": 0}
    for split in split_counts:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    for source in ordered_sources:
        items = by_image[source]
        split = "val" if source in validation_sources else "train"
        split_counts[split] += 1
        image_name = (
            f"{hashlib.sha256(source.encode()).hexdigest()[:8]}_"
            f"{Path(source).name}"
        )
        shutil.copy2(source, root / "images" / split / image_name)
        label_lines = []
        for item in items:
            x_center = item["x"] + item["w"] / 2.0
            y_center = item["y"] + item["h"] / 2.0
            label_lines.append(
                f"{class_map[item['label']]} {x_center:.8f} "
                f"{y_center:.8f} {item['w']:.8f} {item['h']:.8f}"
            )
        (root / "labels" / split / f"{Path(image_name).stem}.txt").write_text(
            "\n".join(label_lines) + "\n", encoding="utf-8"
        )
    yaml_lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "names:",
    ] + [
        f"  {index}: {json.dumps(name)}"
        for index, name in enumerate(class_names)
    ]
    (root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return {
        "root": str(root),
        "image_count": len(ordered_sources),
        "box_count": len(boxes),
        "class_names": class_names,
        "split_counts": split_counts,
    }


def build_signal_label_dataframe(dataframe, annotations, label_column="annotation_label"):
    """Apply reviewed interval labels without mutating the source dataframe."""
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise ValueError("Load a non-empty signal dataset first.")
    output = dataframe.copy()
    labels = pd.Series(pd.NA, index=output.index, dtype="object")
    sources = pd.Series(pd.NA, index=output.index, dtype="object")
    confidences = pd.Series(np.nan, index=output.index, dtype="float64")
    segment_ids = pd.Series(pd.NA, index=output.index, dtype="object")
    approved = [
        validate_signal_annotation(item, len(output))
        for item in annotations
        if item.get("data_type") == "signal"
        and item.get("kind") == "interval"
        and item.get("status") == ANNOTATION_STATUS_APPROVED
    ]
    # Later annotations intentionally take priority, matching the visible order.
    for item in approved:
        positions = range(item["start"], item["end"] + 1)
        index_values = output.index[list(positions)]
        labels.loc[index_values] = item["label"]
        sources.loc[index_values] = item["source"]
        confidences.loc[index_values] = item["confidence"]
        segment_ids.loc[index_values] = item["id"]
    output["annotation_original_row"] = np.arange(len(output), dtype=int)
    output[label_column] = labels
    output[f"{label_column}_source"] = sources
    output[f"{label_column}_confidence"] = confidences
    output["annotation_segment_id"] = segment_ids
    event_labels = pd.Series(pd.NA, index=output.index, dtype="object")
    event_sources = pd.Series(pd.NA, index=output.index, dtype="object")
    for item in annotations:
        if (
            item.get("data_type") == "signal"
            and item.get("kind") == "event"
            and item.get("status") == ANNOTATION_STATUS_APPROVED
        ):
            event = validate_signal_annotation(item, len(output))
            index_value = output.index[event["start"]]
            existing = event_labels.loc[index_value]
            event_labels.loc[index_value] = (
                f"{existing}; {event['label']}"
                if pd.notna(existing)
                else event["label"]
            )
            event_sources.loc[index_value] = event["source"]
    output["annotation_event"] = event_labels
    output["annotation_event_source"] = event_sources
    return output


def suggest_signal_anomaly_intervals(dataframe, columns, sensitivity=3.5):
    """Find conservative robust-z anomaly regions locally."""
    if dataframe is None or dataframe.empty:
        raise ValueError("Load a signal dataset first.")
    selected = [column for column in columns if column in dataframe.columns]
    if not selected:
        raise ValueError("Select at least one numeric signal column.")
    numeric = dataframe[selected].apply(pd.to_numeric, errors="coerce")
    scores = np.zeros(len(numeric), dtype=float)
    valid_columns = 0
    for column in selected:
        values = numeric[column].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if finite.sum() < 5:
            continue
        median = float(np.nanmedian(values))
        mad = float(np.nanmedian(np.abs(values - median)))
        scale = max(1.4826 * mad, float(np.nanstd(values)) * 0.05, 1e-12)
        column_score = np.abs(values - median) / scale
        column_score[~finite] = 0.0
        scores = np.maximum(scores, column_score)
        valid_columns += 1
    if not valid_columns:
        raise ValueError("Selected columns do not contain enough numeric values.")
    mask = scores >= float(sensitivity)
    raw_intervals = []
    start = None
    for index, active in enumerate(np.r_[mask, False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            end = index - 1
            confidence = min(0.99, 0.50 + float(np.max(scores[start:index])) / 20.0)
            raw_intervals.append({"start": start, "end": end, "confidence": confidence})
            start = None
    intervals = []
    for item in raw_intervals:
        if intervals and item["start"] - intervals[-1]["end"] <= 3:
            intervals[-1]["end"] = item["end"]
            intervals[-1]["confidence"] = max(
                intervals[-1]["confidence"], item["confidence"]
            )
        else:
            intervals.append(item)
    return intervals[:500]
