"""Devices for NN Training Studio."""

from contextlib import nullcontext
import tensorflow as tf
from nn_training_studio.constants import (
    DEVICE_AUTO,
    DEVICE_CPU,
)


class DeviceManager:
    """Discover, validate, and resolve framework-specific training devices."""

    @staticmethod
    def _gpu_label(index, name):
        clean_name = str(name or "Compatible GPU").strip()
        return f"GPU {index} — {clean_name}"

    @classmethod
    def tensorflow_gpu_names(cls):
        names = []
        try:
            devices = tf.config.list_physical_devices("GPU")
        except Exception:
            devices = []
        for index, device in enumerate(devices):
            name = getattr(device, "name", f"GPU {index}")
            try:
                details = tf.config.experimental.get_device_details(device)
                name = details.get("device_name") or name
            except Exception:
                pass
            names.append(str(name))
        return names

    @classmethod
    def tensorflow_options(cls):
        return [
            DEVICE_AUTO,
            DEVICE_CPU,
            *[
                cls._gpu_label(index, name)
                for index, name in enumerate(cls.tensorflow_gpu_names())
            ],
        ]

    @classmethod
    def pytorch_gpu_names(cls):
        try:
            import torch

            if not torch.cuda.is_available():
                return []
            return [
                str(torch.cuda.get_device_name(index))
                for index in range(torch.cuda.device_count())
            ]
        except Exception:
            return []

    @classmethod
    def pytorch_options(cls):
        return [
            DEVICE_AUTO,
            DEVICE_CPU,
            *[
                cls._gpu_label(index, name)
                for index, name in enumerate(cls.pytorch_gpu_names())
            ],
        ]

    @staticmethod
    def _gpu_index(selection):
        value = str(selection or "").strip()
        if value.isdigit():
            return int(value)
        if value.lower().startswith("cuda:"):
            return int(value.split(":", 1)[1])
        if value.upper().startswith("GPU "):
            return int(value[4:].split()[0])
        return None

    @classmethod
    def resolve_tensorflow(cls, selection):
        requested = str(selection or DEVICE_AUTO).strip() or DEVICE_AUTO
        gpu_names = cls.tensorflow_gpu_names()
        gpu_index = cls._gpu_index(requested)
        if requested == DEVICE_CPU or requested.lower() == "cpu":
            resolved = "/CPU:0"
            actual_name = "CPU"
        elif requested == DEVICE_AUTO:
            if gpu_names:
                resolved = "/GPU:0"
                actual_name = gpu_names[0]
            else:
                resolved = "/CPU:0"
                actual_name = "CPU"
        elif gpu_index is not None:
            if gpu_index < 0 or gpu_index >= len(gpu_names):
                raise ValueError(
                    f"{requested} was selected, but TensorFlow detected "
                    f"{len(gpu_names)} compatible GPU(s)."
                )
            resolved = f"/GPU:{gpu_index}"
            actual_name = gpu_names[gpu_index]
        else:
            raise ValueError(f"Unsupported TensorFlow device: {requested}")
        return {
            "framework": "TensorFlow / Keras",
            "selected": requested,
            "backend_value": resolved,
            "actual_device": actual_name,
            "gpu_available": bool(gpu_names),
        }

    @classmethod
    def resolve_yolo(cls, selection):
        requested = str(selection or DEVICE_AUTO).strip() or DEVICE_AUTO
        gpu_names = cls.pytorch_gpu_names()
        gpu_index = cls._gpu_index(requested)
        if requested == DEVICE_CPU or requested.lower() == "cpu":
            backend_value = "cpu"
            actual_name = "CPU"
        elif requested == DEVICE_AUTO:
            backend_value = None
            actual_name = gpu_names[0] if gpu_names else "CPU"
        elif gpu_index is not None:
            if gpu_index < 0 or gpu_index >= len(gpu_names):
                raise ValueError(
                    f"{requested} was selected, but PyTorch detected "
                    f"{len(gpu_names)} compatible GPU(s)."
                )
            backend_value = str(gpu_index)
            actual_name = gpu_names[gpu_index]
        else:
            raise ValueError(f"Unsupported YOLO device: {requested}")
        return {
            "framework": "Ultralytics / PyTorch",
            "selected": requested,
            "backend_value": backend_value,
            "actual_device": actual_name,
            "gpu_available": bool(gpu_names),
        }

    @classmethod
    def yolo_label_from_backend(cls, value):
        if value is None or str(value).strip() == "":
            return DEVICE_AUTO
        if str(value).strip().lower() == "cpu":
            return DEVICE_CPU
        index = cls._gpu_index(value)
        names = cls.pytorch_gpu_names()
        if index is not None and 0 <= index < len(names):
            return cls._gpu_label(index, names[index])
        return DEVICE_AUTO

    @staticmethod
    def format_status(info):
        return (
            f"Selected: {info['selected']} | "
            f"Resolved: {info['actual_device']} "
            f"({info['backend_value'] or 'framework auto'})"
        )


def configure_tensorflow_memory_growth():
    """Avoid TensorFlow reserving all GPU memory when possible."""
    try:
        for gpu in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except (RuntimeError, ValueError):
                pass
    except Exception:
        pass


def call_on_tensorflow_device(device_selection, operation, *args, **kwargs):
    """Run one Keras operation on a validated CPU/GPU device."""
    device_info = DeviceManager.resolve_tensorflow(device_selection)
    scope = (
        tf.device(device_info["backend_value"])
        if device_info.get("backend_value")
        else nullcontext()
    )
    with scope:
        return operation(*args, **kwargs)
