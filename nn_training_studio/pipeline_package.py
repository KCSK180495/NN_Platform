"""Export and run a complete pipeline bundle using the same local engine."""

import argparse
from hashlib import sha256
from importlib import metadata as installed_metadata
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import pandas as pd

from .processing_pipeline import FITTED, RECORD, SAMPLE, Pipeline, integer, json_bytes, names, write_archive


def read_csv(path, record_column=None, label_column=None):
    """Keep recording and class identifiers such as 0003 as text."""
    text_columns = {column: str for column in (record_column, label_column) if column}
    frame = pd.read_csv(path, dtype=text_columns, keep_default_na=False)
    for column in text_columns.keys() & frame.columns:
        frame[column] = frame[column].mask(frame[column].str.strip().eq(""))
    return frame


def _model_files(pipeline, source):
    with ZipFile(source) as archive:
        members = archive.namelist()
        if len(members) != len(set(members)):
            raise ValueError("The model ZIP has duplicate members.")
        metadata = json.loads(archive.read("metadata.json"))
        if pipeline.legacy:
            if metadata.get("pipeline_sha256") != pipeline.digest:
                raise ValueError("Select the Scope model bound to this v1 recipe.")
        elif metadata.get("processing_pipeline_sha256") != pipeline.digest:
            raise ValueError("Train using this exact fitted pipeline before attaching a Studio model.")
        if metadata.get("task_type", "Classification") != "Classification":
            raise ValueError("Pipeline model export currently supports signal classifiers.")
        if (metadata.get("feature_cols") or metadata.get("feature_columns")) != pipeline.outputs:
            raise ValueError("Model inputs differ from the pipeline output order.")
        if metadata.get("sampling_frequency_hz") != pipeline.rate:
            raise ValueError("Model and pipeline sample rates differ.")
        classes = names(metadata.get("class_names"), "Model classes")
        if len(classes) < 2:
            raise ValueError("The model must have at least two output classes.")
        windowed = metadata.get("use_window", True)
        window = integer(metadata.get("window_size") if windowed else 1, "Window size")
        stride = integer(metadata.get("stride") if windowed else 1, "Stride")
        shape = [window, len(pipeline.outputs)] if windowed else [len(pipeline.outputs)]
        if metadata.get("input_shape") != shape:
            raise ValueError("The model input shape differs from the declared pipeline outputs.")
        if pipeline.legacy and (window != pipeline.recipe["window"] or stride != pipeline.recipe["stride"]):
            raise ValueError("Window or stride differs from the saved Scope profile.")
        scaler_type = metadata.get("scaler_type") or metadata.get("normalization") or "None"
        if scaler_type == "No normalization":
            scaler_type = "None"
        if scaler_type not in ("None", "StandardScaler", "MinMaxScaler", "RobustScaler"):
            raise ValueError(f"Unsupported model scaler: {scaler_type}.")
        original = "trained_model.keras" if "trained_model.keras" in members else "model.keras"
        files = {"model.keras": archive.read(original)}
        if scaler_type != "None":
            original_scaler = "scaler.pkl" if original == "trained_model.keras" else "scaler.joblib"
            files["scaler.joblib"] = archive.read(original_scaler)
        compact = {key: metadata[key] for key in (
                       "phase", "contract", "physical_hardware_fault_validation", "training_versions")
                   if key in metadata}
        compact.update(class_names=classes, feature_cols=pipeline.outputs, use_window=windowed,
                       window_size=window, stride=stride, scaler_type=scaler_type,
                       input_shape=shape, task_type="Classification")
        compact["model_files_sha256"] = {name: sha256(data).hexdigest() for name, data in files.items()}
        return compact, files


def export_pipeline(destination, pipeline, *, model_source=None):
    if Path(destination).suffix.lower() != ".zip":
        raise ValueError("Export to a .zip file.")
    if not pipeline.legacy and any(
        step["op"] in FITTED and f"step_{index}" not in pipeline.parameters
        for index, step in enumerate(pipeline.recipe["steps"])
    ):
        raise ValueError("Fit the recipe on training or calibration data before exporting it.")
    if model_source and Path(destination).resolve() == Path(model_source).resolve():
        raise ValueError("Choose a new filename for the exported bundle.")
    files = pipeline.files()
    metadata = {
        "pipeline_schema": pipeline.recipe["schema"],
        "pipeline_sha256": pipeline.digest,
        "sampling_frequency_hz": pipeline.rate,
        "input_units": pipeline.recipe["inputs"],
        "feature_cols": pipeline.outputs,
        "execution": "complete_record",
    }
    if model_source:
        model_meta, model_files = _model_files(pipeline, model_source)
        metadata.update(model_meta)
        files.update(model_files)
    root = Path(__file__).resolve().parent
    requirements = ["numpy", "pandas", "scipy", "scikit-learn"]
    if model_source:
        requirements.extend(("keras", "tensorflow", "joblib"))
    versions = {}
    for requirement in requirements:
        try:
            versions[requirement] = installed_metadata.version(requirement)
        except installed_metadata.PackageNotFoundError:
            continue
    extra_requirements = {requirement for step in pipeline.recipe["steps"] if step["op"] == "python"
                          for requirement in step["params"]["requirements"]}
    if extra_requirements:
        files["requirements-custom.txt"] = "-r requirements.txt\n" + "\n".join(sorted(extra_requirements)) + "\n"
    files.update({
        "metadata.json": json_bytes(metadata),
        "pipeline_runtime.json": json_bytes({"format": "nn_studio.pipeline_bundle/1",
                                             "schema": pipeline.recipe["schema"], "model": bool(model_source)}),
        "run_pipeline.py": (
            "from pathlib import Path\n"
            "from nn_training_studio.pipeline_package import main\n\n"
            "if __name__ == '__main__':\n"
            "    main(Path(__file__).resolve().parent)\n"
        ),
        "requirements.txt": "\n".join(requirements) + "\n",
        "environment.json": json_bytes(versions),
        "requirements-api.txt": "-r requirements.txt\nfastapi\nuvicorn\n",
        "serve_pipeline.py": (root / "pipeline_http.py").read_bytes(),
        "README.txt": (
            "Create a Python environment and install requirements.txt. Then run:\n"
            "  python run_pipeline.py input.csv features.csv --record-column Recording\n"
            "Omit --record-column for a single recording. Inputs must already be calibrated "
            "and sampled at the rate in metadata.json.\n"
            "Use an output ending in .npz for portable numeric features and alignment arrays.\n"
            "Pass --predict to emit class probabilities from an attached classifier. "
            "Pass --trust-artifacts only for a trusted serialized scaler.\n"
            "Review code/ before passing --trust-code. Install requirements-custom.txt if provided.\n"
            "For local HTTP access install requirements-api.txt and run python serve_pipeline.py. "
            "The server binds to localhost. Send {\"rows\": [{\"input_name\": 1.0}]} to "
            "/transform or /predict, with optional \"record_column\" to split records.\n"
        ),
        "nn_training_studio/processing_pipeline.py": (root / "processing_pipeline.py").read_bytes(),
        "nn_training_studio/pipeline_package.py": (root / "pipeline_package.py").read_bytes(),
        "nn_training_studio/__init__.py": "",
        "nn_training_studio/scope_runtime/__init__.py": "",
        "nn_training_studio/scope_runtime/pipeline_profile.py":
            (root / "scope_runtime" / "pipeline_profile.py").read_bytes(),
    })
    # Include the shared engine without importing the Studio GUI or training wizard.
    def verify(path):
        if Pipeline.read(path).digest != pipeline.digest:
            raise ValueError("The exported pipeline did not round-trip.")

    write_archive(destination, files, verify)
    return metadata


def save_output(path, frame):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() == ".npz":
        arrays = {}
        for name in frame:
            column = frame[name].to_numpy()
            arrays[name] = column.astype(str) if column.dtype.kind == "O" else column
        # Pass a stream so even a column named 'file' is a valid array key.
        buffer = BytesIO()
        np.savez_compressed(buffer, **{f"column_{index}": value
                                      for index, value in enumerate(arrays.values())},
                            columns=np.asarray(list(arrays)))
        path.write_bytes(buffer.getvalue())
    else:
        raise ValueError("Use a .csv or .npz output filename.")


def predict(pipeline, prepared, path, *, trust_artifacts=False):
    path = Path(path)
    if path.is_dir():
        manifest = json.loads((path / "metadata.json").read_bytes())
        files = {name: (path / name).read_bytes() for name in ("model.keras", "scaler.joblib")
                 if (path / name).exists()}
    else:
        with ZipFile(path) as archive:
            manifest = json.loads(archive.read("metadata.json"))
            files = {name: archive.read(name) for name in ("model.keras", "scaler.joblib") if name in archive.namelist()}
    if not files.get("model.keras") or manifest.get("pipeline_sha256") != pipeline.digest:
        raise ValueError("The selected bundle has no model bound to this fitted pipeline.")
    for name, digest in manifest.get("model_files_sha256", {}).items():
        if name not in files or sha256(files[name]).hexdigest() != digest:
            raise ValueError(f"Bundled model asset changed: {name}.")
    order = names(manifest.get("feature_cols"), "Model inputs")
    classes = names(manifest.get("class_names"), "Model classes")
    if order != pipeline.outputs:
        raise ValueError("Model feature order differs from the prepared features.")
    scaler_type = manifest.get("scaler_type", "None")
    scaler = None
    if scaler_type != "None":
        if not trust_artifacts:
            raise ValueError("This classifier has a serialized scaler. Enable artifact trust only for a trusted ZIP.")
        import joblib

        scaler = joblib.load(BytesIO(files["scaler.joblib"]))
        if (type(scaler).__name__ != scaler_type or scaler.n_features_in_ != len(order)
                or list(getattr(scaler, "feature_names_in_", order)) != order):
            raise ValueError("The model scaler differs from its metadata.")
    from keras.models import load_model

    with TemporaryDirectory(prefix="nn_pipeline_model_") as folder:
        model_path = Path(folder) / "model.keras"
        model_path.write_bytes(files["model.keras"])
        model = load_model(model_path, compile=False, safe_mode=True)
    window = integer(manifest["window_size"], "Model window")
    stride = integer(manifest["stride"], "Model stride")
    windowed = manifest.get("use_window", True)
    width = len(order)
    expected_shape = (None, window, width) if windowed else (None, width)
    model_widths = {len(classes), 1} if len(classes) == 2 else {len(classes)}
    if (model.input_shape != expected_shape or not isinstance(model.output_shape, tuple)
            or len(model.output_shape) != 2 or model.output_shape[-1] not in model_widths):
        raise ValueError("The classifier shape differs from the attached recipe or class list.")
    results = []
    for identity, record in prepared.groupby(RECORD, sort=False):
        x = record[order].to_numpy(dtype=float)
        if scaler is not None:
            x = scaler.transform(x)
        x = x.astype(np.float32)
        if not np.isfinite(x).all():
            raise ValueError(f"Invalid scaled input in recording {identity}.")
        starts = np.arange(0, len(x) - window + 1, stride)
        if not len(starts):
            raise ValueError(f"Recording {identity} is too short for the attached model.")
        for offset in range(0, len(starts), 128):
            rows = starts[offset:offset + 128]
            batch = x[rows[:, None] + np.arange(window)] if windowed else x[rows]
            scores = np.asarray(model.predict_on_batch(batch))
            if scores.shape == (len(rows), 1) and len(classes) == 2:
                scores = np.column_stack([1 - scores[:, 0], scores[:, 0]])
            if (scores.shape != (len(rows), len(classes)) or not np.isfinite(scores).all()
                    or np.any((scores < 0) | (scores > 1)) or not np.allclose(scores.sum(axis=1), 1, atol=.001)):
                raise ValueError("The attached classifier must return class probabilities.")
            positions = record[SAMPLE].to_numpy()
            block = pd.DataFrame(scores, columns=[f"prob_{label}" for label in classes])
            block.insert(0, "prediction", np.asarray(classes)[scores.argmax(axis=1)])
            block.insert(0, "end_sample", positions[rows + window - 1])
            block.insert(0, "start_sample", positions[rows])
            block.insert(0, RECORD, identity)
            results.append(block)
    return pd.concat(results, ignore_index=True)


def main(bundle=None):
    parser = argparse.ArgumentParser(description=__doc__)
    if bundle is None:
        parser.add_argument("bundle", type=Path, help="Pipeline ZIP or extracted directory")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--record-column")
    parser.add_argument("--label-column")
    parser.add_argument("--healthy", action="store_true", help="Confirm known healthy startup for Scope v1")
    parser.add_argument("--trust-code", action="store_true", help="Allow the bundle's custom Python")
    parser.add_argument("--trust-artifacts", action="store_true", help="Allow a serialized fitted model scaler")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()
    source = bundle or args.bundle
    if args.input.resolve() == args.output.resolve():
        parser.error("Choose a different file for the output.")
    pipeline = Pipeline.read(source)
    frame = read_csv(args.input, args.record_column, args.label_column)
    output = pipeline.transform(frame, args.record_column, args.label_column,
                                trust_code=args.trust_code, healthy=args.healthy)
    if args.predict:
        output = predict(pipeline, output, source, trust_artifacts=args.trust_artifacts)
    save_output(args.output, output)


if __name__ == "__main__":
    main()
