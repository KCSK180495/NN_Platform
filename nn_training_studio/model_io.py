"""Model io for NN Training Studio."""

from pathlib import Path
import joblib
import json
import keras
import os
import shutil
import tempfile
import zipfile
from nn_training_studio.constants import (
    DATA_MODE_IMAGE,
    DATA_MODE_TABULAR,
    TASK_CLASSIFICATION,
)


def safely_extract_zip(zip_path, destination):
    """Extract a ZIP package while blocking path-traversal entries."""
    destination_abs = os.path.abspath(destination)

    with zipfile.ZipFile(zip_path, "r") as zipf:
        for member in zipf.infolist():
            member_path = os.path.abspath(
                os.path.join(destination_abs, member.filename)
            )
            if not (
                member_path == destination_abs
                or member_path.startswith(destination_abs + os.sep)
            ):
                raise ValueError(
                    f"Unsafe path found in model package: {member.filename}"
                )

        zipf.extractall(destination_abs)


def load_saved_model_package(package_path):
    """
    Load the complete model package created by this application.

    The ZIP package is used instead of a standalone .keras file because
    evaluation also requires metadata, feature order, scaler, label encoder,
    filters, and window settings.
    """
    if not package_path.lower().endswith(".zip"):
        raise ValueError("Please select a ZIP model package saved by this app.")

    temp_dir = tempfile.mkdtemp(prefix="nn_model_package_")

    try:
        safely_extract_zip(package_path, temp_dir)

        model_path = os.path.join(temp_dir, "trained_model.keras")
        scaler_path = os.path.join(temp_dir, "scaler.pkl")
        encoder_path = os.path.join(temp_dir, "label_encoder.pkl")
        target_scaler_path = os.path.join(temp_dir, "target_scaler.pkl")
        metadata_path = os.path.join(temp_dir, "metadata.json")
        custom_filter_path = os.path.join(temp_dir, "custom_filter.py")
        custom_model_path = os.path.join(temp_dir, "custom_model.py")
        safe_filter_spec_path = os.path.join(
            temp_dir,
            "safe_filter_spec.json",
        )
        safe_model_spec_path = os.path.join(
            temp_dir,
            "safe_model_spec.json",
        )

        required_files = [model_path, scaler_path, metadata_path]
        missing_files = [
            os.path.basename(path)
            for path in required_files
            if not os.path.exists(path)
        ]
        if missing_files:
            raise ValueError(
                "The ZIP package is incomplete. Missing: "
                + ", ".join(missing_files)
            )

        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        scaler = joblib.load(scaler_path)
        task_type = metadata.get("task_type", TASK_CLASSIFICATION)
        label_encoder = None
        if os.path.exists(encoder_path):
            label_encoder = joblib.load(encoder_path)
        elif (
            task_type == TASK_CLASSIFICATION
            and metadata.get("data_mode", DATA_MODE_TABULAR)
            != DATA_MODE_IMAGE
        ):
            raise ValueError(
                "The classification package is missing label_encoder.pkl."
            )

        target_scaler = (
            joblib.load(target_scaler_path)
            if os.path.exists(target_scaler_path)
            else None
        )

        load_errors = []
        model = None

        try:
            model = keras.models.load_model(model_path)
        except Exception as exc:
            load_errors.append(str(exc))

        if model is None:
            try:
                model = keras.models.load_model(
                    model_path,
                    compile=False,
                    safe_mode=False
                )
            except TypeError:
                try:
                    model = keras.models.load_model(
                        model_path,
                        compile=False
                    )
                except Exception as exc:
                    load_errors.append(str(exc))
            except Exception as exc:
                load_errors.append(str(exc))

        if model is None:
            raise ValueError(
                "The Keras model could not be loaded. If the model uses custom "
                "layers, those layer classes must be available when loading.\n\n"
                + "\n".join(load_errors[-2:])
            )

        custom_filter_code = None
        if os.path.exists(custom_filter_path):
            custom_filter_code = Path(custom_filter_path).read_text(
                encoding="utf-8"
            )

        custom_model_code = None
        if os.path.exists(custom_model_path):
            custom_model_code = Path(custom_model_path).read_text(
                encoding="utf-8"
            )

        safe_filter_spec = None
        if os.path.exists(safe_filter_spec_path):
            with open(
                safe_filter_spec_path,
                "r",
                encoding="utf-8",
            ) as file:
                safe_filter_spec = json.load(file)

        safe_model_spec = None
        if os.path.exists(safe_model_spec_path):
            with open(
                safe_model_spec_path,
                "r",
                encoding="utf-8",
            ) as file:
                safe_model_spec = json.load(file)

        return {
            "package_path": package_path,
            "temp_dir": temp_dir,
            "model": model,
            "scaler": scaler,
            "label_encoder": label_encoder,
            "target_scaler": target_scaler,
            "metadata": metadata,
            "custom_filter_code": custom_filter_code,
            "custom_model_code": custom_model_code,
            "safe_filter_spec": safe_filter_spec,
            "safe_model_spec": safe_model_spec,
            "training_results_directory": (
                os.path.join(temp_dir, "results")
                if os.path.isdir(os.path.join(temp_dir, "results"))
                else None
            ),
        }

    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
