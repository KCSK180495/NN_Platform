"""Deployment for NN Training Studio."""

from pathlib import Path
import json
import zipfile
from nn_training_studio.constants import (
    APPLICATION_NAME,
    APP_VERSION,
    DATA_MODE_DETECTION,
    DATA_MODE_IMAGE,
    DATA_MODE_TABULAR,
    DETECTOR_INFERENCE_EXTENSIONS,
    DETECTOR_TRAINABLE_EXTENSIONS,
)


def detect_saved_model_type(path):
    """Identify Keras packages, detection packages, and direct detectors."""
    model_path = Path(path)
    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")
    if model_path.is_dir():
        return "detection_export"

    suffix = model_path.suffix.lower()
    if suffix == ".zip":
        try:
            with zipfile.ZipFile(model_path, "r") as archive:
                members = {
                    member.replace("\\", "/").lstrip("./")
                    for member in archive.namelist()
                }
        except zipfile.BadZipFile as exc:
            raise ValueError("The selected ZIP file is invalid.") from exc
        if any(
            member == "detection_manifest.json"
            or member.endswith("/detection_manifest.json")
            for member in members
        ):
            return "detection_package"
        if (
            any(
                member == "metadata.json"
                or member.endswith("/metadata.json")
                for member in members
            )
            and any(
                member == "trained_model.keras"
                or member.endswith("/trained_model.keras")
                for member in members
            )
        ):
            return "keras_package"
        raise ValueError(
            "This ZIP is not a recognized NN Studio model package."
        )
    if suffix in DETECTOR_TRAINABLE_EXTENSIONS:
        return "detection_trainable"
    if suffix in DETECTOR_INFERENCE_EXTENSIONS:
        return "detection_export"
    raise ValueError(
        "Unsupported model type. Select an NN Studio ZIP package or a "
        "supported YOLO detector file."
    )


def inspect_saved_model_descriptor(path):
    """
    Inspect a supported model without deserialising or executing it.

    The loading hub uses this lightweight descriptor to explain what the model
    can do before the user opens the corresponding application workspace.
    """
    model_path = Path(path).expanduser().resolve()
    model_type = detect_saved_model_type(model_path)
    descriptor = {
        "path": str(model_path),
        "name": model_path.name,
        "model_type": model_type,
        "display_type": "",
        "data_mode": "",
        "task_type": "",
        "trainable": False,
        "inference_ready": True,
        "capabilities": [],
        "details": {},
    }

    if model_type == "keras_package":
        metadata = {}
        with zipfile.ZipFile(model_path, "r") as archive:
            metadata_member = next(
                (
                    member
                    for member in archive.namelist()
                    if member.replace("\\", "/").lstrip("./")
                    .endswith("metadata.json")
                ),
                None,
            )
            if metadata_member:
                try:
                    metadata = json.loads(
                        archive.read(metadata_member).decode("utf-8")
                    )
                except Exception:
                    metadata = {}
        data_mode = metadata.get("data_mode", DATA_MODE_TABULAR)
        task_type = metadata.get("task_type", "Saved Keras model")
        descriptor.update(
            {
                "display_type": (
                    "Image-classification model package"
                    if data_mode == DATA_MODE_IMAGE
                    else "Signal / tabular model package"
                ),
                "data_mode": data_mode,
                "task_type": task_type,
                "trainable": False,
                "capabilities": [
                    "Run prediction on new compatible data",
                    "Evaluate when ground-truth labels are available",
                    "Preview predictions, metrics, and standard plots",
                    "Create manual or AI-assisted custom result plots",
                    "Export predictions and complete evaluation results",
                ],
                "details": {
                    "model_name": metadata.get("model_type")
                    or metadata.get("model_name"),
                    "input_shape": metadata.get("input_shape"),
                    "feature_columns": metadata.get("feature_cols")
                    or metadata.get("feature_columns"),
                    "target_columns": metadata.get("target_cols")
                    or metadata.get("target_columns"),
                    "class_names": metadata.get("class_names")
                    or metadata.get("class_values"),
                    "application_version": metadata.get(
                        "application_version"
                    ),
                },
            }
        )
        return descriptor

    if model_type == "detection_package":
        manifest = {}
        training_config = {}
        with zipfile.ZipFile(model_path, "r") as archive:
            normalised_members = {
                member.replace("\\", "/").lstrip("./"): member
                for member in archive.namelist()
            }
            manifest_name = next(
                (
                    original
                    for normalised, original in normalised_members.items()
                    if normalised.endswith("detection_manifest.json")
                ),
                None,
            )
            training_name = next(
                (
                    original
                    for normalised, original in normalised_members.items()
                    if normalised.endswith("training_config.json")
                ),
                None,
            )
            try:
                if manifest_name:
                    manifest = json.loads(
                        archive.read(manifest_name).decode("utf-8")
                    )
            except Exception:
                manifest = {}
            try:
                if training_name:
                    training_config = json.loads(
                        archive.read(training_name).decode("utf-8")
                    )
            except Exception:
                training_config = {}
        descriptor.update(
            {
                "display_type": "NN Studio object-detection package",
                "data_mode": DATA_MODE_DETECTION,
                "task_type": "YOLO object detection",
                "trainable": True,
                "capabilities": [
                    "Detect objects in an image, folder, video, or webcam",
                    "Preview annotated detections inside NN Studio",
                    "Evaluate on a labelled YOLO dataset",
                    "Resume or fine-tune from packaged .pt weights",
                    "Export detections, results, and deployment formats",
                ],
                "details": {
                    "created_at": manifest.get("created_at"),
                    "weights": (
                        manifest.get("best_weights")
                        or manifest.get("last_weights")
                    ),
                    "base_model": training_config.get("model"),
                    "image_size": training_config.get("imgsz"),
                    "class_names": manifest.get("class_names"),
                    "application_version": manifest.get(
                        "application_version"
                    ),
                },
            }
        )
        return descriptor

    suffix = model_path.suffix.lower()
    is_architecture = suffix in {".yaml", ".yml"}
    if model_type == "detection_trainable":
        display_type = (
            "YOLO architecture configuration"
            if is_architecture
            else "Trainable YOLO detector weights"
        )
        capabilities = [
            "Train or fine-tune on a compatible YOLO dataset",
            "Detect objects in an image, folder, video, or webcam",
            "Preview annotated detections inside NN Studio",
            "Evaluate and export detection results",
        ]
    else:
        display_type = "Exported inference-only object detector"
        capabilities = [
            "Detect objects in an image, folder, video, or webcam",
            "Preview annotated detections inside NN Studio",
            "Export annotated outputs and detection tables",
        ]
    descriptor.update(
        {
            "display_type": display_type,
            "data_mode": DATA_MODE_DETECTION,
            "task_type": "YOLO object detection",
            "trainable": model_type == "detection_trainable",
            "inference_ready": not is_architecture,
            "capabilities": capabilities,
            "details": {
                "format": (
                    "exported directory"
                    if model_path.is_dir()
                    else suffix.lstrip(".").upper()
                ),
                "architecture_only": is_architecture,
            },
        }
    )
    return descriptor


def deployment_required_inputs(descriptor):
    """Return the logical model inputs that must be mapped by an application."""
    if not descriptor:
        return []
    data_mode = descriptor.get("data_mode")
    if data_mode == DATA_MODE_DETECTION:
        return ["image_or_video_source"]
    if data_mode == DATA_MODE_IMAGE:
        return ["image_path"]
    details = descriptor.get("details") or {}
    features = (
        details.get("feature_columns")
        or details.get("input_columns")
        or []
    )
    if isinstance(features, str):
        features = [features]
    return [str(value) for value in features if str(value).strip()]


def deployment_recommended_targets(descriptor):
    """Choose deployment targets that match the inspected model data mode."""
    data_mode = (descriptor or {}).get("data_mode")
    common = [
        "Python Application",
        "Local REST API",
        "Docker REST API",
        "Windows Desktop / EXE",
        "Cloud API",
    ]
    if data_mode == DATA_MODE_DETECTION:
        return common + [
            "Camera / Object Detection",
            "Raspberry Pi / Edge Computer",
        ]
    if data_mode == DATA_MODE_IMAGE:
        return common + [
            "Camera / Object Detection",
            "Raspberry Pi / Edge Computer",
        ]
    return common + [
        "Raspberry Pi / Edge Computer",
        "Sensor / Serial Device",
        "MQTT / IoT",
        "Modbus TCP / PLC",
        "OPC UA",
    ]


def deployment_default_protocol(target):
    """Select a sensible communication protocol for a deployment target."""
    return {
        "Python Application": "Direct Python",
        "Local REST API": "HTTP / REST",
        "Docker REST API": "Docker",
        "Windows Desktop / EXE": "Direct Python",
        "Raspberry Pi / Edge Computer": "Direct Python",
        "Camera / Object Detection": "Camera / Video",
        "Sensor / Serial Device": "USB / Serial",
        "MQTT / IoT": "MQTT",
        "Modbus TCP / PLC": "Modbus TCP",
        "OPC UA": "OPC UA",
        "Cloud API": "HTTP / REST",
    }.get(target, "Direct Python")


def deployment_compatibility_report(
    descriptor,
    target,
    protocol,
    input_mappings,
    export_format,
):
    """Return deterministic compatibility evidence for the deployment page."""
    failures = []
    warnings = []
    passed = []
    if not descriptor:
        return {
            "status": "Not ready",
            "failures": ["No saved model has been selected."],
            "warnings": [],
            "passed": [],
        }

    model_path = Path(descriptor["path"])
    if not model_path.exists():
        failures.append("The selected model path no longer exists.")
    else:
        passed.append("Model artifact is available.")

    required_inputs = deployment_required_inputs(descriptor)
    mapping_by_model_input = {
        str(item.get("model_input", "")).strip(): str(
            item.get("application_input", "")
        ).strip()
        for item in input_mappings
    }
    missing = [
        name
        for name in required_inputs
        if not mapping_by_model_input.get(name)
    ]
    if missing:
        failures.append(
            "Application inputs are not mapped for: " + ", ".join(missing)
        )
    elif required_inputs:
        passed.append(f"All {len(required_inputs)} model inputs are mapped.")
    else:
        warnings.append(
            "The package does not list named inputs. Confirm the runtime input "
            "shape manually before deployment."
        )

    data_mode = descriptor.get("data_mode")
    if data_mode in {DATA_MODE_IMAGE, DATA_MODE_DETECTION} and protocol in {
        "USB / Serial",
        "Modbus TCP",
        "OPC UA",
    }:
        failures.append(
            f"{protocol} does not directly transport the image/video input "
            "required by this model."
        )
    if (
        data_mode == DATA_MODE_TABULAR
        and target == "Camera / Object Detection"
    ):
        failures.append(
            "A signal/tabular model cannot use the camera detection target."
        )
    if not descriptor.get("inference_ready", True):
        failures.append(
            "This artifact contains an architecture only. Train it or select "
            "weights before creating an inference deployment."
        )
    else:
        passed.append("The selected artifact is inference-ready.")

    if descriptor.get("trainable"):
        passed.append("The original artifact can be retained for fine-tuning.")
    else:
        warnings.append(
            "The selected artifact is intended for inference/evaluation; "
            "resuming training may not be available."
        )

    model_type = descriptor.get("model_type")
    if export_format != "Keep original model package":
        if model_type == "keras_package" and export_format in {
            "TensorRT",
            "TorchScript",
        }:
            warnings.append(
                f"{export_format} conversion is not performed automatically "
                "for this Keras package. Export it first in a compatible tool."
            )
        elif model_type.startswith("detection") and export_format in {
            "ONNX",
            "OpenVINO",
            "TFLite / LiteRT",
            "TensorRT",
            "TorchScript",
        }:
            warnings.append(
                f"The deployment package records the requested "
                f"{export_format} format. Use the detector workspace to export "
                "the converted model before final deployment."
            )
        else:
            warnings.append(
                "Format conversion is recorded as a deployment requirement; "
                "the original model is retained in this package."
            )
    else:
        passed.append("The original model and preprocessing remain together.")

    if target in {"Docker REST API", "Cloud API"}:
        warnings.append(
            "Confirm the host CPU/GPU, HTTPS authentication, storage, cost, "
            "and cancellation policy before production use."
        )
    if target == "Windows Desktop / EXE":
        warnings.append(
            "Build the executable on Windows and test it on a clean computer."
        )

    status = "Ready" if not failures else "Action required"
    if not failures and warnings:
        status = "Ready with warnings"
    return {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "passed": passed,
    }


def _deployment_python_runtime(descriptor):
    """Generate the shared, editable local inference runtime."""
    detection = descriptor.get("data_mode") == DATA_MODE_DETECTION
    if detection:
        return '''"""Generated by NN Training Studio. Review before production use."""
import json
import tempfile
import zipfile
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "deployment_config.json").read_text(encoding="utf-8"))
SOURCE = ROOT / CONFIG["model_artifact"]


def _resolve_weights():
    if SOURCE.suffix.lower() != ".zip":
        return SOURCE
    extract_dir = Path(tempfile.mkdtemp(prefix="nn_studio_detector_"))
    with zipfile.ZipFile(SOURCE, "r") as archive:
        archive.extractall(extract_dir)
    candidates = list(extract_dir.rglob("best.pt"))
    candidates += list(extract_dir.rglob("last.pt"))
    candidates += list(extract_dir.rglob("*.onnx"))
    if not candidates:
        raise RuntimeError("No deployable detector weights found in package.")
    return candidates[0]


MODEL = YOLO(str(_resolve_weights()))


def predict_source(source, confidence=0.25, iou=0.45, device=None):
    """Predict one image, folder, video, webcam index, or stream."""
    return MODEL.predict(
        source=source,
        conf=float(confidence),
        iou=float(iou),
        device=device,
        save=False,
    )


if __name__ == "__main__":
    sample = CONFIG.get("sample_source")
    if not sample:
        raise SystemExit("Set sample_source in deployment_config.json.")
    results = predict_source(sample)
    print(f"Processed {len(results)} result item(s).")
'''
    return '''"""Generated by NN Training Studio. Review before production use."""
import json
import tempfile
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "deployment_config.json").read_text(encoding="utf-8"))
SOURCE = ROOT / CONFIG["model_artifact"]
EXTRACTED = Path(tempfile.mkdtemp(prefix="nn_studio_model_"))
with zipfile.ZipFile(SOURCE, "r") as archive:
    archive.extractall(EXTRACTED)

MODEL = keras.models.load_model(EXTRACTED / "trained_model.keras", compile=False)
SCALER = joblib.load(EXTRACTED / "scaler.pkl")
TARGET_SCALER = (
    joblib.load(EXTRACTED / "target_scaler.pkl")
    if (EXTRACTED / "target_scaler.pkl").exists()
    else None
)
LABEL_ENCODER = (
    joblib.load(EXTRACTED / "label_encoder.pkl")
    if (EXTRACTED / "label_encoder.pkl").exists()
    else None
)


def _model_records(application_records):
    mappings = CONFIG["input_mappings"]
    return [
        {
            item["model_input"]: record[item["application_input"]]
            for item in mappings
        }
        for record in application_records
    ]


def predict_records(application_records):
    """Predict one record or a chronological list of signal records."""
    frame = pd.DataFrame(_model_records(application_records))
    ordered = [item["model_input"] for item in CONFIG["input_mappings"]]
    values = SCALER.transform(frame[ordered])
    expected = MODEL.input_shape
    if isinstance(expected, list):
        raise ValueError("Generated runtime supports one model input.")
    if len(expected) == 3:
        required = int(expected[1])
        if len(values) < required:
            raise ValueError(f"At least {required} chronological records are required.")
        values = values[-required:][np.newaxis, ...]
    predictions = np.asarray(MODEL.predict(values, verbose=0))
    if TARGET_SCALER is not None:
        predictions = TARGET_SCALER.inverse_transform(
            predictions.reshape(predictions.shape[0], -1)
        )
    if LABEL_ENCODER is not None:
        indices = np.argmax(predictions, axis=-1)
        return LABEL_ENCODER.inverse_transform(indices).tolist()
    return predictions.tolist()


if __name__ == "__main__":
    sample = CONFIG.get("sample_records") or []
    if not sample:
        raise SystemExit("Add sample_records to deployment_config.json.")
    print(json.dumps(predict_records(sample), indent=2))
'''


def generate_deployment_text_artifacts(
    descriptor,
    target,
    protocol,
    input_mappings,
    output_actions,
    export_format,
    model_artifact,
):
    """Generate editable integration source and supporting deployment files."""
    config = {
        "schema_version": 1,
        "created_by": f"{APPLICATION_NAME} {APP_VERSION}",
        "model_artifact": model_artifact,
        "model_type": descriptor.get("model_type"),
        "data_mode": descriptor.get("data_mode"),
        "task_type": descriptor.get("task_type"),
        "deployment_target": target,
        "communication_protocol": protocol,
        "requested_export_format": export_format,
        "input_mappings": input_mappings,
        "output_actions": output_actions,
        "sample_records": [],
        "sample_source": "",
    }
    files = {
        "app.py": _deployment_python_runtime(descriptor),
        "deployment_config.json": json.dumps(config, indent=2),
        "input_mapping.csv": (
            "application_input,model_input,processing\n"
            + "\n".join(
                ",".join(
                    [
                        str(item.get("application_input", "")),
                        str(item.get("model_input", "")),
                        str(item.get("processing", "")),
                    ]
                )
                for item in input_mappings
            )
            + "\n"
        ),
        "output_actions.json": json.dumps(output_actions, indent=2),
    }

    detection = descriptor.get("data_mode") == DATA_MODE_DETECTION
    requirements = (
        ["ultralytics>=8.3", "opencv-python>=4.9"]
        if detection
        else [
            "tensorflow>=2.16",
            "keras>=3.0",
            "numpy>=1.26",
            "pandas>=2.0",
            "joblib>=1.3",
            "scikit-learn>=1.4",
        ]
    )

    if target in {"Local REST API", "Docker REST API", "Cloud API"}:
        requirements += ["fastapi>=0.110", "uvicorn>=0.29"]
        if detection:
            prediction_call = (
                "results = predict_source(request.source, request.confidence, "
                "request.iou)\n"
                "    return {'result_items': len(results)}"
            )
            request_fields = (
                "    source: str\n"
                "    confidence: float = 0.25\n"
                "    iou: float = 0.45"
            )
            import_name = "predict_source"
        else:
            prediction_call = (
                "return {'predictions': predict_records(request.records)}"
            )
            request_fields = "    records: list[dict]"
            import_name = "predict_records"
        files["server.py"] = f'''"""Generated REST wrapper. Add authentication before internet exposure."""
from fastapi import FastAPI
from pydantic import BaseModel
from app import {import_name}

api = FastAPI(title="NN Studio Inference API")


class PredictionRequest(BaseModel):
{request_fields}


@api.get("/health")
def health():
    return {{"status": "ok"}}


@api.post("/predict")
def predict(request: PredictionRequest):
    {prediction_call}
'''

    if target == "Sensor / Serial Device":
        requirements.append("pyserial>=3.5")
        files["serial_integration.py"] = '''"""Read newline-delimited JSON records from a serial device."""
import json
import serial
from app import predict_records

PORT = "COM3"  # Change for the target computer.
BAUD_RATE = 115200

with serial.Serial(PORT, BAUD_RATE, timeout=1) as connection:
    while True:
        line = connection.readline().decode("utf-8", errors="ignore").strip()
        if not line:
            continue
        record = json.loads(line)
        print(predict_records([record]))
'''
    elif target == "MQTT / IoT":
        requirements.append("paho-mqtt>=2.0")
        files["mqtt_integration.py"] = '''"""MQTT inference example. Configure TLS and credentials for production."""
import json
import paho.mqtt.client as mqtt
from app import predict_records


def on_message(client, _userdata, message):
    record = json.loads(message.payload.decode("utf-8"))
    prediction = predict_records([record])
    client.publish("nn-studio/prediction", json.dumps(prediction))


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_message = on_message
client.connect("localhost", 1883)
client.subscribe("nn-studio/input")
client.loop_forever()
'''
    elif target == "Modbus TCP / PLC":
        requirements.append("pymodbus>=3.6")
        files["modbus_integration.py"] = '''"""Modbus TCP skeleton. Confirm register scaling and byte order."""
from pymodbus.client import ModbusTcpClient
from app import predict_records

client = ModbusTcpClient("192.168.0.10", port=502)
client.connect()
# Read configured registers, convert them into named application inputs,
# then call predict_records([record]). Do not guess PLC register mappings.
'''
    elif target == "OPC UA":
        requirements.append("opcua>=0.98")
        files["opcua_integration.py"] = '''"""OPC UA skeleton. Replace node IDs with verified server nodes."""
from opcua import Client
from app import predict_records

client = Client("opc.tcp://localhost:4840")
client.connect()
# Read verified nodes into one named record and call predict_records([record]).
'''

    files["requirements.txt"] = "\n".join(dict.fromkeys(requirements)) + "\n"

    if target == "Docker REST API":
        files["Dockerfile"] = '''FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "server:api", "--host", "0.0.0.0", "--port", "8000"]
'''
        files[".dockerignore"] = "__pycache__\n*.pyc\n.env\n"

    if target == "Windows Desktop / EXE":
        files["build_windows_exe.bat"] = (
            "py -m pip install -r requirements.txt pyinstaller\r\n"
            "pyinstaller --onedir --name NNStudioInference app.py\r\n"
        )

    files["README.md"] = f"""# NN Studio Deployment Package

Target: {target}

Protocol: {protocol}

Model: {descriptor.get('display_type')}

Task: {descriptor.get('task_type')}

Requested format: {export_format}

## Start

1. Review `deployment_config.json` and `input_mapping.csv`.
2. Create a Python 3.11 environment and install `requirements.txt`.
3. Test with sample data before connecting real equipment.
4. Run `app.py`, or `uvicorn server:api --host 127.0.0.1 --port 8000`
   for a generated REST target.
5. Add authentication and HTTPS before exposing any API to a network.

The original model artifact and its saved preprocessing are retained. No AI
provider key, cloud token, password, or raw training dataset is included.
Hardware register addresses, serial framing, OPC UA node IDs, and safety actions
must be verified on the real system before use.
"""
    return files
