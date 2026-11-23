"""Inspect signal packages without loading Keras or deserializing a scaler."""

from contextlib import contextmanager
from hashlib import file_digest
import json
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from .model_signal_filter import FILTER_METHOD, validate_signal_filter
from .pipeline_profile import Profile, positive_integer
from .sscn_scope import CONTRACT, needs_healthy_start, profile_inputs


def sha256(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def inspect_package(path, profile_path=None):
    path = Path(path).resolve()
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError("The model ZIP has duplicate entries.")
        models = set(names) & {"model.keras", "trained_model.keras"}
        if len(models) != 1 or "metadata.json" not in names:
            raise ValueError("Select a model ZIP with one Keras model and metadata.json at its root.")
        model_member = models.pop()
        metadata = json.loads(archive.read("metadata.json"))
    if not isinstance(metadata, dict):
        raise ValueError("Model metadata must be a JSON object.")
    studio = model_member == "trained_model.keras"
    if (metadata.get("task_type", "Classification") != "Classification"
            or metadata.get("data_mode", "Tabular / Signal CSV") != "Tabular / Signal CSV"):
        raise ValueError("NN Scope supports signal classification packages. Use the model's existing workspace for other tasks.")
    features = metadata.get("feature_cols") or metadata.get("feature_columns")
    classes = metadata.get("class_names") or metadata.get("class_values")
    for values, field in ((features, "features"), (classes, "class names")):
        if (not isinstance(values, list) or not values
                or any(not isinstance(value, str) or not value.strip() for value in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"Model metadata must list distinct, ordered {field}.")
    if len(classes) < 2:
        raise ValueError("Signal classification needs at least two class names.")
    dataset = metadata.get("dataset_analysis") or {}
    rate = float(metadata.get("sampling_frequency_hz") or dataset.get("sampling_frequency_hz") or 0)
    if not isfinite(rate) or rate <= 0:
        raise ValueError("The training sample rate is missing. Save a model with its actual sampling frequency.")
    window = 1 if metadata.get("use_window") is False else metadata.get("window_size")
    stride = 1 if metadata.get("use_window") is False else metadata.get("stride")
    positive_integer(window, "Model window")
    positive_integer(stride, "Model stride")
    scaler_type = metadata.get("scaler_type") or metadata.get("normalization") or "None"
    if scaler_type not in ("None", "StandardScaler", "MinMaxScaler", "RobustScaler"):
        raise ValueError(f"Unsupported fitted scaler: {scaler_type}.")
    metadata.update(feature_cols=features, class_names=classes, sampling_frequency_hz=rate,
                    window_size=window, stride=stride, scaler_type=scaler_type)
    if metadata.get("custom_filter_enabled") or metadata.get("safe_filter_spec") or "custom_filter.py" in names:
        raise ValueError("The model uses a custom training filter. NN Scope needs an equivalent supported preprocessing profile.")
    if studio and metadata.get("missing_value_method") not in (None, "", "Drop rows"):
        raise ValueError("The package does not provide the fitted missing-value transform NN Scope needs.")
    if any(metadata.get(key) for key in ("transform", "feature_transform", "input_representation")):
        raise ValueError("This input transform requires a dedicated NN Scope adapter.")
    if metadata.get("contract") not in (None, "", CONTRACT):
        raise ValueError(f"Unsupported signal contract: {metadata['contract']}.")

    # An external profile cannot replace a model-declared digest with a different recipe.
    profile = Profile.read(profile_path or path) if profile_path or "pipeline.json" in names else None
    if metadata.get("pipeline_sha256") and (profile is None or profile.digest != metadata["pipeline_sha256"]):
        raise ValueError("Load the preprocessing profile bound to this model package.")
    filter_method = metadata.get("filter_method") or "No filter"
    legacy_sscn = metadata.get("contract") == CONTRACT
    if profile is None and not legacy_sscn and filter_method != "No filter":
        if studio or filter_method != FILTER_METHOD or metadata.get("ai_filter_spec"):
            raise ValueError(f"The training filter '{filter_method}' has no matching NN Scope adapter.")
        validate_signal_filter(metadata.get("signal_filter"), rate)
    if profile is not None:
        profile.validate_metadata(metadata)
    declared_units = metadata.get("input_units") or {}
    if not isinstance(declared_units, dict) or any(not isinstance(unit, str) for unit in declared_units.values()):
        raise ValueError("Model input units must be a name-to-unit mapping.")
    info = {
        "metadata": metadata,
        "profile": profile,
        "package_path": str(path),
        "model_member": model_member,
        "scaler_member": ("scaler.pkl" if studio else "scaler.joblib"),
    }
    if scaler_type != "None" and info["scaler_member"] not in names:
        raise ValueError(f"Package is missing its fitted {scaler_type}.")
    info["inputs"] = profile_inputs(info) or {feature: declared_units.get(feature, "") for feature in features}
    info["healthy_start_required"] = needs_healthy_start(info)
    return info


@contextmanager
def load_package(path, profile_path=None, expected_sha256=None, expected_profile_digest=None):
    """Load a trusted local package. Read only the model and fitted scaler members."""
    if expected_sha256 and sha256(path) != expected_sha256:
        raise ValueError("Model ZIP does not match the saved session checksum.")
    info = inspect_package(path, profile_path)
    if expected_profile_digest and (not info["profile"] or info["profile"].digest != expected_profile_digest):
        raise ValueError("The preprocessing profile or its coefficients changed since configuration.")
    from keras.models import load_model
    import joblib

    with TemporaryDirectory(prefix="nn_scope_model_") as folder, ZipFile(path) as archive:
        model_path = Path(folder) / info["model_member"]
        with archive.open(info["model_member"]) as src, model_path.open("wb") as dst:
            from shutil import copyfileobj
            copyfileobj(src, dst)
        model = load_model(model_path, compile=False, safe_mode=True)
        metadata = info["metadata"]
        scaler = None
        if metadata["scaler_type"] != "None":
            with archive.open(info["scaler_member"]) as member:
                scaler = joblib.load(member)
            if (type(scaler).__name__ != metadata["scaler_type"]
                    or getattr(scaler, "n_features_in_", None) != len(metadata["feature_cols"])):
                raise ValueError("Fitted scaler does not match model metadata.")
            names = getattr(scaler, "feature_names_in_", metadata["feature_cols"])
            if list(names) != metadata["feature_cols"]:
                raise ValueError("Fitted scaler feature order differs from model metadata.")
        if info["profile"]:
            info["profile"].bind(metadata, model, scaler)
        else:
            size = len(metadata["feature_cols"])
            win = metadata["window_size"]
            output_widths = {len(metadata["class_names"])}
            if len(metadata["class_names"]) == 2:
                output_widths.add(1)
            if (model.input_shape not in ((None, win, size), (None, win * size))
                    or not isinstance(model.output_shape, tuple) or len(model.output_shape) != 2
                    or model.output_shape[-1] not in output_widths):
                raise ValueError("Model shape does not match a supported single-input signal classifier.")
        yield dict(info, model=model, scaler=scaler, package_sha256=sha256(path))
