"""Versioned, record-based signal processing shared by Studio and exported runtimes."""

from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd
from scipy import signal

from .scope_runtime.pipeline_profile import FeatureStream, Profile


SCHEMA = "nn_studio.pipeline/2"
SCOPE_SCHEMA = "nn_scope.pipeline/1"
RECORD = "annotation_segment_id"
SAMPLE = "sample_index"
OPERATIONS = {
    "butterworth": {"kind": "lowpass", "cutoff_hz": 50, "order": 4, "mode": "causal"},
    "notch": {"frequency_hz": 50, "quality": 30, "mode": "causal"},
    "moving_average": {"samples": 5, "center": False},
    "median": {"samples": 5, "center": False},
    "ema": {"span": 5},
    "kalman": {"process_noise": 0.00001, "measurement_noise": 0.01},
    "rms": {"samples": 5, "center": False},
    "rolling_std": {"samples": 5, "center": False},
    "difference": {},
    "add": {},
    "subtract": {},
    "multiply": {},
    "divide": {},
    "sscn": {"warmup": 200, "pole_pairs": 4, "cycles": 1, "peak_floor": 0.00001,
             "references": []},
    "standard": {},
    "minmax": {},
    "robust": {},
    "python": {"file": "step.py", "entry": "transform", "interface": "columns", "requirements": []},
}
FITTED = {"standard", "minmax", "robust"}
FILTERS = {"butterworth", "notch"}
ROLLING = {"moving_average", "median", "rms", "rolling_std"}
ARITHMETIC = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply, "divide": np.divide}
SCALER_TYPES = {"standard": "StandardScaler", "minmax": "MinMaxScaler", "robust": "RobustScaler"}


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode("utf-8")


def number(value, label, minimum=0):
    if type(value) not in (float, int) or not np.isfinite(value) or value <= minimum:
        raise ValueError(f"{label} must be finite and greater than {minimum}.")
    return value


def integer(value, label, minimum=1, maximum=1_000_000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be an integer between {minimum} and {maximum}.")
    return value


def names(value, label):
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError(f"{label} must be a nonempty list of names.")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} contains duplicate names.")
    return value


def _validate_params(op, params, rate):
    if op in FILTERS:
        if params["mode"] not in ("causal", "zero_phase"):
            raise ValueError("Filter mode must be causal or zero_phase.")
        number(rate, "Sample rate")
    if op == "butterworth":
        integer(params["order"], "Butterworth order", maximum=10)
        kind = params["kind"]
        cutoff = params["cutoff_hz"]
        if kind in ("bandpass", "bandstop"):
            if not isinstance(cutoff, list) or len(cutoff) != 2:
                raise ValueError("Specify two cutoff frequencies for a band filter.")
            if not number(cutoff[0], "Lower cutoff") < number(cutoff[1], "Upper cutoff") < rate / 2:
                raise ValueError("Band cutoffs must increase and be below half the sample rate.")
        elif kind not in ("lowpass", "highpass") or number(cutoff, "Cutoff") >= rate / 2:
            raise ValueError("Use a supported filter kind with a cutoff below half the sample rate.")
    if op == "notch":
        if number(params["frequency_hz"], "Notch frequency") >= rate / 2:
            raise ValueError("Notch frequency must be below half the sample rate.")
        number(params["quality"], "Notch quality")
    if op in ROLLING:
        integer(params["samples"], "Window length")
        if type(params["center"]) is not bool:
            raise ValueError("center must be true or false.")
    if op == "ema":
        integer(params["span"], "EMA span")
    if op == "kalman":
        for key in params:
            number(params[key], key)
    if op == "sscn":
        number(rate, "Sample rate")
        integer(params["warmup"], "Warmup samples", minimum=0)
        integer(params["pole_pairs"], "Pole pairs")
        integer(params["cycles"], "Calibration cycles")
        number(params["peak_floor"], "Peak floor")
        refs = params["references"]
        if not isinstance(refs, list) or not refs:
            raise ValueError("SSCN needs measured healthy references with speed_rpm and peak_a.")
        speeds = []
        for ref in refs:
            if not isinstance(ref, dict) or set(ref) != {"speed_rpm", "peak_a"}:
                raise ValueError("Every healthy reference must contain speed_rpm and peak_a.")
            speeds.append(number(ref["speed_rpm"], "Reference RPM"))
            number(ref["peak_a"], "Reference peak", params["peak_floor"])
        if len(speeds) != len(set(speeds)):
            raise ValueError("Healthy reference RPMs must be unique.")
    if op == "python":
        if not isinstance(params["file"], str) or not re.fullmatch(r"[a-zA-Z_][a-zA-Z_0-9]*\.py", params["file"]):
            raise ValueError("Custom Python needs a simple .py filename.")
        if not isinstance(params["entry"], str) or not params["entry"].isidentifier():
            raise ValueError("Custom Python entry must be a function name.")
        if params["interface"] not in ("columns", "studio_filter"):
            raise ValueError("Use the columns or studio_filter Python interface.")
        requirements = params["requirements"]
        if not isinstance(requirements, list) or any(
            not isinstance(item, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*(==[A-Za-z0-9_.+-]+)?", item)
            for item in requirements
        ):
            raise ValueError("Use package names or exact package==version requirements for custom Python.")


def validate_recipe(recipe):
    """Normalize defaults and validate the full signal dependency graph before execution."""
    if not isinstance(recipe, dict) or recipe.get("schema") != SCHEMA:
        raise ValueError(f"Expected a {SCHEMA} recipe.")
    allowed = {"schema", "name", "sample_rate_hz", "inputs", "outputs", "steps"}
    if set(recipe) - allowed:
        raise ValueError("Unsupported recipe fields: " + ", ".join(sorted(set(recipe) - allowed)))
    name = recipe.get("name", "Processing pipeline")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Enter a pipeline name.")
    rate = recipe.get("sample_rate_hz")
    if rate is not None:
        number(rate, "Sample rate")
    inputs = recipe.get("inputs")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("Declare input column names and units, using '1' for dimensionless values.")
    names(list(inputs), "Input columns")
    if any(not isinstance(unit, str) or not unit.strip() for unit in inputs.values()):
        raise ValueError("Every input needs a unit.")
    if set(inputs) & {RECORD, SAMPLE}:
        raise ValueError("Recording ID and sample index are reserved for alignment.")
    steps = recipe.get("steps")
    if not isinstance(steps, list) or len(steps) > 100:
        raise ValueError("Provide an ordered list of at most 100 steps.")
    known = set(inputs)
    normalized = []
    for index, step in enumerate(steps, 1):
        if not isinstance(step, dict) or set(step) - {"op", "inputs", "outputs", "params"}:
            raise ValueError(f"Invalid step {index}.")
        op = step.get("op")
        if not isinstance(op, str) or op not in OPERATIONS:
            raise ValueError(f"Unsupported operation in step {index}: {op!r}.")
        source = names(step.get("inputs"), f"Step {index} inputs")
        target = names(step.get("outputs"), f"Step {index} outputs")
        if set(source) - known or set(target) & (known | {RECORD, SAMPLE}):
            raise ValueError(f"Step {index} must read existing columns and produce new names.")
        if op == "sscn":
            dimensions = (2, (1, 2))
        else:
            dimensions = (2 if op in ARITHMETIC else 1, (1,))
        if op != "python" and (len(source) != dimensions[0] or len(target) not in dimensions[1]):
            raise ValueError(f"Unexpected number of input or output columns in step {index}.")
        overrides = step.get("params", {})
        if not isinstance(overrides, dict) or set(overrides) - set(OPERATIONS[op]):
            raise ValueError(f"Unsupported parameter in step {index}.")
        params = {**OPERATIONS[op], **overrides}
        _validate_params(op, params, rate)
        if op == "python" and params["interface"] == "studio_filter" and len(source) != len(target):
            raise ValueError(f"Step {index} must declare one output per input for a Studio custom filter.")
        normalized.append(dict(op=op, inputs=source, outputs=target, params=params))
        known.update(target)
    outputs = names(recipe.get("outputs"), "Pipeline outputs")
    if set(outputs) - known:
        raise ValueError("Pipeline outputs must name available inputs or step results.")
    return dict(schema=SCHEMA, name=name, sample_rate_hz=rate, inputs=inputs,
                outputs=outputs, steps=normalized)


def _sos(op, params, rate):
    if op == "notch":
        return signal.tf2sos(*signal.iirnotch(params["frequency_hz"], params["quality"], fs=rate))
    return signal.butter(params["order"], params["cutoff_hz"], btype=params["kind"], fs=rate, output="sos")


def _fit_scaler(op, values):
    from sklearn import preprocessing

    scaler = getattr(preprocessing, SCALER_TYPES[op])().fit(values[:, None])
    if op == "minmax":
        return np.array([scaler.scale_[0], scaler.min_[0]])
    center = scaler.mean_ if op == "standard" else scaler.center_
    return np.array([center[0], scaler.scale_[0]])


def _process_step(op, columns, params, state, rate, code, input_names):
    x = columns[0]
    if op in FILTERS:
        if params["mode"] == "zero_phase":
            return (signal.sosfiltfilt(state, x),)
        return (signal.sosfilt(state, x, zi=signal.sosfilt_zi(state) * x[0])[0],)
    if op in ROLLING:
        series = pd.Series(x ** 2 if op == "rms" else x)
        window = series.rolling(params["samples"], center=params["center"], min_periods=1)
        if op == "median":
            values = window.median().to_numpy()
        elif op == "rolling_std":
            values = window.std(ddof=0).to_numpy()
        else:
            values = window.mean().to_numpy()
        return (np.sqrt(values) if op == "rms" else values,)
    if op == "ema":
        return (pd.Series(x).ewm(span=params["span"], adjust=False).mean().to_numpy(),)
    if op == "kalman":
        estimate, covariance = x[0], 1.0
        values = np.empty_like(x)
        for index, observation in enumerate(x):
            covariance += params["process_noise"]
            gain = covariance / (covariance + params["measurement_noise"])
            estimate += gain * (observation - estimate)
            covariance *= 1 - gain
            values[index] = estimate
        return (values,)
    if op == "difference":
        return (np.diff(x, prepend=x[0]),)
    if op in ARITHMETIC:
        if op == "divide" and np.any(columns[1] == 0):
            raise ValueError("Division by zero.")
        return (ARITHMETIC[op](*columns),)
    if op == "sscn":
        rpm = columns[1]
        start = params["warmup"]
        if start >= len(rpm) or np.any(rpm <= 0):
            raise ValueError("SSCN needs positive RPM and enough samples for warmup.")
        baseline = rpm[start]
        stop = start + int(np.ceil(params["cycles"] * rate * 60 / (params["pole_pairs"] * baseline)))
        if stop > len(rpm) or not np.allclose(rpm[start:stop], baseline, rtol=.001, atol=.01):
            raise ValueError("SSCN needs a complete electrical cycle at stable startup RPM.")
        matches = [ref for ref in params["references"]
                   if np.isclose(ref["speed_rpm"], baseline, rtol=.001, atol=.01)]
        if len(matches) != 1:
            raise ValueError(f"No unique saved healthy peak matches {baseline:g} RPM.")
        peak = matches[0]["peak_a"]
        normalized, peaks = x / peak, np.full(len(x), peak)
        normalized[:stop - 1] = peaks[:stop - 1] = np.nan
        return normalized, peaks
    if op in FITTED:
        return (x * state[0] + state[1] if op == "minmax" else (x - state[0]) / state[1],)
    function = code[params["file"]].get(params["entry"])
    if not callable(function):
        raise ValueError(f"{params['file']} must define the function {params['entry']}.")
    if params["interface"] == "studio_filter":
        frame = pd.DataFrame(dict(zip(input_names, columns)))
        result = function(frame.copy(), list(input_names))
        if (not isinstance(result, pd.DataFrame) or not result.index.equals(frame.index)
                or set(input_names) - set(result)):
            raise ValueError("A Studio custom filter must return its input columns in the original row order.")
        return tuple(result[name].to_numpy() for name in input_names)
    return function(columns, rate)


def _records(frame, record_column):
    if frame.empty:
        raise ValueError("The recording table is empty.")
    if not record_column:
        yield "record_1", frame
        return
    if record_column not in frame or frame[record_column].isna().any():
        raise ValueError("Recording IDs must be present on every row.")
    ids = frame[record_column].astype(str)
    cuts = np.flatnonzero(ids.to_numpy()[1:] != ids.to_numpy()[:-1]) + 1
    starts = np.r_[0, cuts]
    record_ids = ids.iloc[starts]
    if record_ids.duplicated().any():
        raise ValueError("Each recording must occupy consecutive rows.")
    for identity, start, stop in zip(record_ids, starts, np.r_[cuts, len(frame)]):
        yield identity, frame.iloc[start:stop]


@dataclass
class Pipeline:
    recipe: dict
    parameters: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    legacy: Profile | None = None

    @classmethod
    def build(cls, recipe, sources=None):
        recipe = validate_recipe(recipe)
        sources = sources or {}
        needed = {step["params"]["file"] for step in recipe["steps"] if step["op"] == "python"}
        if needed != set(sources):
            raise ValueError("Select the custom Python files named in the recipe: " + ", ".join(sorted(needed)))
        params = {f"step_{index}": _sos(step["op"], step["params"], recipe["sample_rate_hz"])
                  for index, step in enumerate(recipe["steps"]) if step["op"] in FILTERS}
        return cls(recipe, params, dict(sources))

    @property
    def outputs(self):
        return ([out["feature"] for out in self.legacy.spec["outputs"]] if self.legacy
                else self.recipe["outputs"])

    @property
    def rate(self):
        return self.recipe["analysis_hz"] if self.legacy else self.recipe["sample_rate_hz"]

    def _native_data(self, frame, record_column):
        inputs = list(self.recipe["inputs"])
        missing = set(inputs) - set(frame.columns)
        if missing:
            raise ValueError("Missing input columns: " + ", ".join(sorted(missing)))
        for name, record in _records(frame, record_column):
            data = {key: pd.to_numeric(record[key], errors="raise").to_numpy(dtype=float)
                    for key in inputs}
            if any(not np.isfinite(values).all() for values in data.values()):
                raise ValueError(f"Recording {name} contains missing or nonfinite inputs.")
            yield name, record, data

    def transform(self, frame, record_column=None, label_column=None, *, trust_code=False, healthy=False):
        return self._process(frame, record_column, label_column, trust_code, healthy, fit=False)

    def fit(self, training, record_column=None, *, trust_code=False):
        """Fit only the rows explicitly supplied as training or calibration data."""
        if self.legacy:
            raise ValueError("A Scope v1 profile already contains its preprocessing parameters.")
        if not any(step["op"] in FITTED for step in self.recipe["steps"]):
            raise ValueError("Add Standard, MinMax, or Robust scaling before fitting.")
        fitted = Pipeline(self.recipe, self.parameters.copy(), self.sources)
        fitted._process(training, record_column, None, trust_code, False, fit=True)
        return fitted

    def _process(self, frame, record_column, label_column, trust_code, healthy, fit):
        if label_column and (label_column not in frame or label_column in self.recipe["inputs"]
                             or label_column in (*self.outputs, RECORD, SAMPLE)):
            raise ValueError("Choose a label column distinct from the processed features.")
        if self.sources and not trust_code:
            raise ValueError("This recipe includes Python. Allow custom code only if you trust its source.")
        records = list(self._native_data(frame, record_column))
        if self.legacy:
            blocks = [FeatureStream(self.legacy, healthy).push_signals(0, data) for _, _, data in records]
        else:
            # Each recording has independent state, including custom module globals.
            modules = [{} for _ in records]
            for index, step in enumerate(self.recipe["steps"]):
                op, params = step["op"], step["params"]
                key = f"step_{index}"
                if op in FITTED:
                    if fit:
                        values = np.concatenate([data[step["inputs"][0]] for _, _, data in records])
                        values = values[np.isfinite(values)]
                        if not len(values):
                            raise ValueError("No usable training values for the fitted transform.")
                        self.parameters[key] = _fit_scaler(op, values)
                    if key not in self.parameters:
                        raise ValueError(f"Fit step {index + 1} on training or calibration data first.")
                for (_, _, data), code in zip(records, modules):
                    if op == "python" and params["file"] not in code:
                        namespace = {"__name__": f"pipeline_{params['file'][:-3]}", "np": np, "pd": pd}
                        exec(compile(self.sources[params["file"]], params["file"], "exec"), namespace)
                        code[params["file"]] = namespace
                    source = tuple(data[column].copy() for column in step["inputs"])
                    start = max((int(np.argmax(np.isfinite(x))) if np.isfinite(x).any() else len(x)) for x in source)
                    if any(not np.isfinite(x[start:]).all() for x in source):
                        raise ValueError(f"Step {index + 1} received a gap within a recording.")
                    if start == len(source[0]):
                        raise ValueError(f"No usable rows remain for step {index + 1}.")
                    result = _process_step(op, tuple(x[start:] for x in source), params,
                                           self.parameters.get(key), self.rate, code, step["inputs"])
                    if not isinstance(result, (tuple, list)) or (len(result) != len(step["outputs"]) and op != "sscn"):
                        raise ValueError(f"Step {index + 1} must return one array per declared output.")
                    for column, values in zip(step["outputs"], result):
                        values = np.asarray(values, dtype=float)
                        if values.shape != source[0][start:].shape or np.isinf(values).any():
                            raise ValueError(f"Step {index + 1} must return aligned numeric arrays.")
                        data[column] = np.r_[np.full(start, np.nan), values]
            blocks = []
            for _, _, data in records:
                values = np.column_stack([data[x] for x in self.outputs])
                usable = np.isfinite(values).all(axis=1)
                start = int(np.argmax(usable)) if usable.any() else len(values)
                if not usable[start:].all():
                    raise ValueError("The output has missing values within a recording.")
                blocks.append((start, values[start:]))
        result = []
        for (identity, record, _), (start, values) in zip(records, blocks):
            if not len(values):
                raise ValueError(f"Recording {identity} is too short for this pipeline.")
            block = pd.DataFrame(values, columns=self.outputs)
            block.insert(0, SAMPLE, np.arange(start, len(record)))
            block.insert(0, RECORD, identity)
            if label_column:
                block[label_column] = record[label_column].iloc[start:].to_numpy()
            result.append(block)
        return pd.concat(result, ignore_index=True)

    def files(self):
        if self.legacy:
            return dict(self.legacy.files)
        archive = BytesIO()
        np.savez_compressed(archive, **self.parameters)
        return {"pipeline.json": json_bytes(self.recipe), "parameters.npz": archive.getvalue(),
                **{f"code/{name}": source for name, source in self.sources.items()}}

    @property
    def digest(self):
        if self.legacy:
            return self.legacy.digest
        files = self.files()
        return sha256(b"".join(name.encode() + b"\0" + sha256(files[name]).digest()
                              for name in sorted(files))).hexdigest()

    @classmethod
    def read(cls, path):
        path = Path(path)
        if path.is_dir():
            recipe = json.loads((path / "pipeline.json").read_bytes())
            if isinstance(recipe, dict) and recipe.get("schema") == SCOPE_SCHEMA:
                return cls(recipe, legacy=Profile.read(path / "pipeline.json"))
            return cls._read_fitted(recipe, lambda name: (path / name).read_bytes(),
                                    (path / "metadata.json").exists())
        if path.suffix.lower() != ".zip":
            with path.open(encoding="utf-8-sig") as stream:
                recipe = json.load(stream)
            if isinstance(recipe, dict) and recipe.get("schema") == SCOPE_SCHEMA:
                return cls(recipe, legacy=Profile.read(path))
            sources = {step["params"]["file"]: (path.parent / "code" / step["params"]["file"]).read_bytes()
                       for step in validate_recipe(recipe)["steps"] if step["op"] == "python"}
            return cls.build(recipe, sources)
        with ZipFile(path) as archive:
            if len(archive.namelist()) != len(set(archive.namelist())):
                raise ValueError("The pipeline ZIP has duplicate members.")
            recipe = json.loads(archive.read("pipeline.json"))
            if isinstance(recipe, dict) and recipe.get("schema") == SCOPE_SCHEMA:
                return cls(recipe, legacy=Profile.read(path))
            return cls._read_fitted(recipe, archive.read, "metadata.json" in archive.namelist())

    @classmethod
    def _read_fitted(cls, recipe, read, has_metadata):
        sources = {step["params"]["file"]: read(f"code/{step['params']['file']}")
                   for step in validate_recipe(recipe)["steps"] if step["op"] == "python"}
        pipeline = cls.build(recipe, sources)
        with np.load(BytesIO(read("parameters.npz")), allow_pickle=False) as arrays:
            for index, step in enumerate(pipeline.recipe["steps"]):
                key = f"step_{index}"
                if step["op"] in FILTERS:
                    saved = arrays[key]
                    if saved.ndim != 2 or saved.shape[1] != 6 or not len(saved) or not np.isfinite(saved).all():
                        raise ValueError(f"Invalid saved filter coefficients for step {index + 1}.")
                    if not np.all(saved[:, 3] == 1) or any(np.max(abs(np.roots(row[3:]))) >= 1 for row in saved):
                        raise ValueError(f"Unstable saved filter for step {index + 1}.")
                    pipeline.parameters[key] = saved
                elif step["op"] in FITTED and key in arrays:
                    saved = arrays[key]
                    if saved.shape != (2,) or not np.isfinite(saved).all():
                        raise ValueError(f"Invalid fitted parameters for step {index + 1}.")
                    if saved[0 if step["op"] == "minmax" else 1] <= 0:
                        raise ValueError(f"Invalid scale for step {index + 1}.")
                    pipeline.parameters[key] = saved
        if has_metadata:
            expected = json.loads(read("metadata.json")).get("pipeline_sha256")
            if expected and pipeline.digest != expected:
                raise ValueError("The recipe, parameters, or Python files differ from the saved pipeline.")
        return pipeline


def write_archive(path, files, verify=None):
    destination = Path(path)
    with NamedTemporaryFile(dir=destination.parent, prefix=".pipeline-", suffix=".zip", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            for name, value in files.items():
                archive.writestr(name, value)
        if verify:
            verify(temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
