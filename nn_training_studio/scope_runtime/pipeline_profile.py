"""Version 1 preprocessing profiles. Each session owns one contiguous signal history."""
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import zipfile

import numpy as np
from scipy import signal


def positive_integer(value, field):
    if type(value) is not int or value < 1:
        raise ValueError(f"{field} must be a positive integer.")
    return value


@dataclass(frozen=True)
class Profile:
    spec: dict
    sos: np.ndarray
    digest: str
    files: dict

    @classmethod
    def read(cls, path):
        path = Path(path)
        archive = path.suffix.lower() == ".zip"
        if archive:
            with zipfile.ZipFile(path) as z:
                data = z.read("pipeline.json")
        else:
            data = path.read_bytes()
        files = {"pipeline.json": data}
        spec = json.loads(data)
        needs_filter = any(n["op"] == "sos.v1" for n in spec["steps"])
        if needs_filter:
            if archive:
                with zipfile.ZipFile(path) as z:
                    files["parameters.npz"] = z.read("parameters.npz")
            else:
                files["parameters.npz"] = path.with_name("parameters.npz").read_bytes()
        inputs = spec["inputs"]
        if (spec["schema"] != "nn_scope.pipeline/1" or not isinstance(inputs, dict) or not inputs
                or any(not isinstance(k, str) or not k or not isinstance(u, str) or not u.strip()
                       for k, u in inputs.items())
                or ("speed_rpm" in inputs and inputs["speed_rpm"].lower() != "rpm")):
            raise ValueError("Unsupported profile schema or input units.")
        for field in ("window", "stride", "analysis_hz", "history_samples"):
            positive_integer(spec[field], field)
        if spec["stride"] > spec["window"]:
            raise ValueError("Window stride exceeds the supported window length.")
        sos = None
        if needs_filter:
            with np.load(io.BytesIO(files["parameters.npz"]), allow_pickle=False) as arrays:
                sos = np.asarray(arrays["sos"], dtype=np.float64)
            if sos.ndim != 2 or sos.shape[1] != 6 or not len(sos) or not np.isfinite(sos).all() or not np.all(sos[:, 3] == 1):
                raise ValueError("Invalid saved SOS coefficients.")
            if any(np.max(abs(np.roots(section[3:]))) >= 1 for section in sos):
                raise ValueError("Filter poles must be inside the unit circle.")
        known = set(spec["inputs"])
        count_peak = 0
        for node in spec["steps"]:
            op = node["op"]
            required = {"sos.v1": ("input",), "healthy_cycle_peak.v1": ("input", "speed"),
                        "healthy_peak_by_rpm.v1": ("speed",),
                        "divide.v1": ("numerator", "denominator")}.get(op)
            if required is None or any(node[k] not in known for k in required) or node["out"] in known:
                raise ValueError(f"Invalid stage or signal dependency: {node}")
            if op == "sos.v1" and (node["array"] != "sos" or node["initial"] != "first_sample"):
                raise ValueError("Unsupported filter coefficients or initialization.")
            if op in ("healthy_cycle_peak.v1", "healthy_peak_by_rpm.v1"):
                count_peak += 1
                for k in ("warmup", "cycles", "pole_pairs"):
                    positive_integer(node[k], k)
                if not np.isfinite(node["peak_floor"]) or node["peak_floor"] <= 0 or count_peak > 1:
                    raise ValueError("Invalid healthy peak configuration.")
                if op == "healthy_peak_by_rpm.v1":
                    refs = node.get("references")
                    if node.get("phase") not in ("U", "V", "W") or not isinstance(refs, list) or not refs:
                        raise ValueError("Saved healthy references require a phase and a nonempty RPM table.")
                    table = np.array([[r["speed_rpm"], r["peak_a"]] for r in refs], dtype=float)
                    if (not np.isfinite(table).all() or np.any(table[:, 0] <= 0)
                            or np.any(table[:, 1] <= node["peak_floor"]) or np.any(np.diff(table[:, 0]) <= 0)):
                        raise ValueError("Saved healthy RPMs must be positive, unique and sorted; peaks must exceed the floor.")
            known.add(node["out"])
        if not spec["outputs"] or any(x["signal"] not in known for x in spec["outputs"]):
            raise ValueError("Unknown profile output.")
        digest = hashlib.sha256(b"".join(hashlib.sha256(files[n]).digest() for n in files)).hexdigest()
        return cls(spec, sos, digest, files)

    def save(self, path):
        path = Path(path)
        with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as f:
            temp = Path(f.name)
        try:
            with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
                for name, data in self.files.items():
                    z.writestr(name, data)
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)

    def validate_metadata(self, meta):
        spec = self.spec
        cols = [x["feature"] for x in spec["outputs"]]
        required = meta.get("pipeline_sha256")
        if required and required != self.digest:
            raise ValueError("The selected profile differs from the model's saved preprocessing profile.")
        for node in spec["steps"]:
            if node["op"] == "healthy_peak_by_rpm.v1" and node["phase"] != meta.get("phase"):
                raise ValueError("Saved healthy reference phase differs from the model phase.")
        if (cols != meta["feature_cols"] or spec["window"] != meta["window_size"]
                or spec["stride"] != meta["stride"] or spec["analysis_hz"] != meta["sampling_frequency_hz"]):
            raise ValueError("Profile feature order, window, stride or rate does not match the model.")

    def bind(self, meta, model, scaler):
        self.validate_metadata(meta)
        spec = self.spec
        width = len(meta["feature_cols"])
        if (model.input_shape != (None, spec["window"], width)
                or model.output_shape != (None, len(meta["class_names"]))
                or (scaler is not None and getattr(scaler, "n_features_in_", 0) != width)
                or (type(scaler).__name__ if scaler is not None else "None") != meta["scaler_type"]):
            raise ValueError("Profile features, window, rate, model shape or fitted scaler do not match.")


class FeatureStream:
    """Input blocks at the profile rate. Output contains only causally valid rows."""
    def __init__(self, profile, healthy):
        self.profile = profile
        self.offset = 0
        self.filters = {}
        self.peak = 0.
        self.baseline = None
        self.calibration_end = None
        self.reference_source = None
        self.reference_id = None
        self.valid_start = profile.spec["history_samples"] - 1
        if any(n["op"] == "healthy_cycle_peak.v1" for n in profile.spec["steps"]) and not healthy:
            raise ValueError("Confirm a known-healthy startup for this record in Diagnoses.")

    def push(self, start, current, speed):
        return self.push_signals(start, {"current_A": current, "speed_rpm": speed})

    def push_signals(self, start, inputs):
        if start != self.offset:
            raise ValueError("Signal discontinuity. Begin a new session and calibrate the new record.")
        values = {}
        length = None
        for name, unit in self.profile.spec["inputs"].items():
            raw = np.asarray(inputs[name])
            if name == "current_A":
                raw = raw.astype(np.float32)
            values[name] = raw.astype(np.float64)
            if length is None:
                length = len(raw)
            if raw.shape != (length,) or not length or not np.isfinite(values[name]).all():
                raise ValueError(f"Invalid or misaligned samples for {name}.")
            if unit.lower() == "rpm" and np.any(values[name] <= 0):
                raise ValueError("SSCN requires positive RPM.")
        end = start + length
        ready = True
        for n in self.profile.spec["steps"]:
            op, out = n["op"], n["out"]
            if op == "sos.v1":
                source = values[n["input"]]
                zi = self.filters.get(out)
                if zi is None:
                    zi = signal.sosfilt_zi(self.profile.sos) * source[0]
                values[out], self.filters[out] = signal.sosfilt(self.profile.sos, source, zi=zi)
            elif op in ("healthy_cycle_peak.v1", "healthy_peak_by_rpm.v1"):
                warm = n["warmup"]
                rpm = values[n["speed"]]
                if start <= warm < end:
                    self.baseline = float(rpm[warm-start])
                    count = int(np.ceil(n["cycles"] * self.profile.spec["analysis_hz"] * 60 /
                                        (n["pole_pairs"] * self.baseline)))
                    self.calibration_end = warm + count
                    self.valid_start = max(self.valid_start, self.calibration_end-1)
                    if op == "healthy_peak_by_rpm.v1":
                        matches = [r for r in n["references"]
                                   if np.isclose(r["speed_rpm"], self.baseline, rtol=.001, atol=.01)]
                        if len(matches) != 1:
                            available = ", ".join(f"{r['speed_rpm']:g}" for r in n["references"])
                            raise ValueError(f"No unique saved healthy {n['phase']} reference for {self.baseline:g} rpm. Available RPMs: {available}.")
                        self.peak = float(matches[0]["peak_a"])
                        self.reference_source = "saved healthy reference by RPM"
                        self.reference_id = matches[0].get("reference_id")
                    else:
                        self.reference_source = "confirmed healthy startup cycle"
                if self.calibration_end is not None:
                    lo, hi = max(start, warm)-start, min(end, self.calibration_end)-start
                    if lo < hi:
                        if not np.allclose(rpm[lo:hi], self.baseline, rtol=.001, atol=.01):
                            raise ValueError("Need one electrical cycle at stable startup speed for the reference.")
                        if op == "healthy_cycle_peak.v1":
                            self.peak = max(self.peak, float(np.max(np.abs(values[n["input"]][lo:hi]))))
                ready = self.calibration_end is not None and end >= self.calibration_end
                if ready and self.peak <= n["peak_floor"]:
                    raise ValueError("Healthy calibration peak is too small.")
                values[out] = np.full(length, self.peak)
            elif op == "divide.v1":
                if ready and np.any(values[n["denominator"]] == 0):
                    raise ValueError("Zero denominator in profile equation.")
                values[out] = values[n["numerator"]]/values[n["denominator"]] if ready else np.zeros(length)
        self.offset = end
        lo = min(end, max(start, self.valid_start)) if ready else end
        rows = np.column_stack([values[x["signal"]][lo-start:] for x in self.profile.spec["outputs"]])
        if not np.isfinite(rows).all():
            raise ValueError("Nonfinite engineered features.")
        return lo, rows.astype(np.float32)
