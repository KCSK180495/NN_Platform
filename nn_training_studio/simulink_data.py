"""File bridge from MATLAB/Simulink to signal training; no MATLAB engine needed."""
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_ROWS = 2_000_000
TIME_COLUMN = "Time_s"
RUN_COLUMN = "simulink_run_id"
ROW_COLUMN = "simulink_source_row"
SEGMENT_COLUMN = "annotation_segment_id"
SIMULINK_EVIDENCE_COLUMNS = frozenset({TIME_COLUMN, RUN_COLUMN, ROW_COLUMN, SEGMENT_COLUMN})
EXPORT_TOOLS = ("nn_export_simulink.m", "nn_run_and_export.m")


@dataclass
class SimulationDataset:
    frame: pd.DataFrame
    feature_columns: list
    target_column: str | None
    sample_rate_hz: float
    metadata: dict

    def summary(self):
        return (
            f"MATLAB / Simulink: {len(self.metadata['sources'])} run(s), "
            f"{len(self.frame):,} rows, {self.sample_rate_hz:g} Hz.\n"
            f"Suggested inputs: {', '.join(self.feature_columns)}\n"
            f"Target: {self.target_column or 'choose a target, or use an autoencoder'}\n"
            "Time/run identifiers are evidence, not suggested model inputs. "
            "Windows will not cross simulation-run boundaries. "
            "Review the training split policy before interpreting accuracy."
        )


def _metadata(payload):
    if not isinstance(payload, dict):
        raise ValueError("Simulation metadata must be a JSON object.")
    if payload.get("format") != "nnstudio-simulink" or payload.get("version") != 1:
        raise ValueError("Unsupported simulation export metadata; use nn_export_simulink.")
    return {key: payload[key] for key in (
        "model_name", "run_id", "units", "target_column", "interpolation",
        "sample_time_s", "created_utc", "signal_names",
    ) if key in payload}


def _read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        names = next(csv.reader(stream), [])
    if not names or any(not name.strip() for name in names) or len(set(names)) != len(names):
        raise ValueError("CSV column names must be nonempty and unique.")
    frame = pd.read_csv(path, nrows=MAX_ROWS + 1, encoding="utf-8-sig")
    sidecar = path.with_suffix(".simulation.json")
    metadata = {}
    if sidecar.is_file():
        if sidecar.stat().st_size > 1024 * 1024:
            raise ValueError("Simulation metadata exceeds 1 MiB.")
        metadata = _metadata(json.loads(sidecar.read_text(encoding="utf-8-sig")))
    return frame, metadata


def _read_mat(path):
    from scipy.io import loadmat

    try:
        payload = loadmat(path, simplify_cells=True)
    except NotImplementedError as exc:
        raise ValueError("MAT v7.3 is not supported. Re-export with nn_export_simulink "
                         "as CSV or a plain -v7 MAT file.") from exc
    except (ValueError, TypeError) as exc:
        raise ValueError("Cannot read this MAT file. Use nn_export_simulink to flatten "
                         "SimulationOutput/logsout objects to CSV or plain -v7 MAT.") from exc
    time = next((payload[k] for k in ("time", "Time_s", "tout", "t") if k in payload), None)
    data = payload.get("signals", payload.get("data"))
    if time is None or data is None or isinstance(data, dict):
        raise ValueError("MAT import needs numeric 'time' and 'signals' arrays. "
                         "Export logsout/SimulationOutput with nn_export_simulink first.")
    time = np.asarray(time).reshape(-1)
    if np.iscomplexobj(time):
        raise ValueError("MAT time must be real numeric seconds.")
    values = np.asarray(data)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[0] != len(time) or values.dtype.kind not in "biuf":
        raise ValueError("MAT signals must be a real samples-by-signals numeric matrix.")
    raw_names = payload.get("signal_names", [f"Signal_{i + 1}" for i in range(values.shape[1])])
    names = [str(n).strip() for n in np.atleast_1d(raw_names).reshape(-1)]
    if len(names) != values.shape[1] or len(set(names)) != len(names) or not all(names):
        raise ValueError("MAT signal_names must name each signal once.")
    if any(n in SIMULINK_EVIDENCE_COLUMNS for n in names):
        raise ValueError("MAT signal_names conflict with reserved time/run columns.")
    frame = pd.DataFrame(values, columns=names)
    frame.insert(0, TIME_COLUMN, time)
    labels = payload.get("labels", payload.get("label"))
    if labels is not None:
        if "Label" in frame:
            raise ValueError("MAT labels conflict with a signal named Label.")
        labels = np.atleast_1d(labels).reshape(-1)
        if len(labels) == 1:
            labels = np.repeat(labels, len(frame))
        if len(labels) != len(frame):
            raise ValueError("MAT labels must be scalar or have one value per sample.")
        frame["Label"] = labels
    metadata = _metadata(json.loads(str(payload["metadata_json"]))) if "metadata_json" in payload else {}
    return frame, metadata


def load_simulink_files(paths):
    """Import aligned CSV/plain MAT runs with their time and boundaries intact.

    Files need matching columns, units and a uniform sample interval. No
    interpolation, label invention, calibration or model fitting occurs here.
    """
    if not paths:
        raise ValueError("Select at least one Simulink CSV or MAT export.")
    frames, sources, seen_paths, seen_hashes = [], [], set(), set()
    reference_columns = reference_units = reference_rate = target = None
    for value in paths:
        path = Path(value).resolve()
        if path in seen_paths:
            raise ValueError("The same simulation file was selected twice.")
        seen_paths.add(path)
        if path.suffix.lower() not in (".csv", ".mat"):
            raise ValueError("Choose CSV or plain MAT exports from MATLAB/Simulink.")
        if path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Each simulation export must be no larger than 256 MiB.")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        checksum = digest.hexdigest()
        if checksum in seen_hashes:
            raise ValueError("Duplicate simulation export contents; select each run once.")
        seen_hashes.add(checksum)
        frame, metadata = _read_csv(path) if path.suffix.lower() == ".csv" else _read_mat(path)
        if len(frame) < 2 or sum(len(f) for f in frames) + len(frame) > MAX_ROWS:
            raise ValueError("Import requires two rows per run and at most two million rows total.")
        if any(c in frame for c in (RUN_COLUMN, ROW_COLUMN, SEGMENT_COLUMN)):
            raise ValueError("Export columns conflict with reserved simulation provenance names.")
        time_names = [c for c in frame if str(c).lower() in ("time_s", "time", "tout", "t")]
        if not time_names:
            raise ValueError("A numeric time column in seconds (Time_s, time, tout or t) is required.")
        if len(time_names) != 1:
            raise ValueError("The export has conflicting time columns; keep one numeric time column.")
        time_name = time_names[0]
        frame = frame.rename(columns={time_name: TIME_COLUMN})
        if frame.columns.duplicated().any():
            raise ValueError("The export has conflicting time columns.")
        time = pd.to_numeric(frame[TIME_COLUMN], errors="coerce").to_numpy(dtype=float)
        dt = np.diff(time)
        if not np.isfinite(time).all() or np.any(dt <= 0):
            raise ValueError("Simulation time must be finite and strictly increasing within each run.")
        interval = float(np.median(dt))
        if not np.allclose(dt, interval, rtol=1e-5, atol=max(1e-12, interval * 1e-8)):
            raise ValueError("Variable-step or gapped time data needs a uniform export. "
                             "Use fixed-step logging or nn_export_simulink(..., 'SampleTime', Ts).")
        rate = 1.0 / interval
        if not np.isfinite(rate):
            raise ValueError("Simulation sample rate must be finite.")
        target_name = metadata.get("target_column") or next(
            (c for c in frame if str(c).lower() in ("label", "target", "fault_label", "class")), None)
        if target_name is not None and (not isinstance(target_name, str) or target_name not in frame):
            raise ValueError("The target column declared by the export is missing.")
        if target_name and (frame[target_name].isna().any() or
                            frame[target_name].astype(str).str.strip().eq("").any()):
            raise ValueError("The simulation target/label has missing values; export confirmed targets first.")
        signal_names = [c for c in frame if c not in (TIME_COLUMN, target_name)]
        if not signal_names:
            raise ValueError("The simulation export has no input signals.")
        for column in signal_names:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Signal '{column}' needs finite numeric samples; clean/export it in MATLAB first.")
            frame[column] = values
        frame[TIME_COLUMN] = time
        units = metadata.get("units", {})
        if not isinstance(units, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in units.items()):
            raise ValueError("Signal units must be a name-to-unit JSON object.")
        units = {name: units.get(name, "") for name in signal_names}
        if reference_columns is None:
            reference_columns, reference_units, reference_rate, target = list(frame.columns), units, rate, target_name
        elif (set(frame.columns) != set(reference_columns) or units != reference_units
              or target_name != target or not np.isclose(rate, reference_rate, rtol=1e-5)):
            raise ValueError("All runs must have matching signal/target columns, units and sample rate.")
        frame = frame[reference_columns].copy()
        run_id = f"{path.stem}:{checksum[:12]}"
        frame[RUN_COLUMN] = run_id
        frame[ROW_COLUMN] = np.arange(len(frame))
        frame[SEGMENT_COLUMN] = run_id
        frames.append(frame)
        sources.append({"file": str(path), "sha256": checksum, "rows": len(frame),
                        "run_id": run_id, "export": metadata})
    combined = pd.concat(frames, ignore_index=True)
    features = [c for c in reference_columns if c not in (TIME_COLUMN, target)]
    return SimulationDataset(combined, features, target, reference_rate,
        {"format": "nnstudio-simulink-import", "version": 1, "sources": sources,
         "sample_rate_hz": reference_rate, "time_column": TIME_COLUMN,
         "feature_columns": features, "target_column": target, "units": reference_units})


def save_matlab_export_tools(directory):
    """Copy editable MATLAB helpers without overwriting the user's other code."""
    destination = Path(directory)
    source = Path(__file__).with_name("matlab")
    for name in EXPORT_TOOLS:
        target = destination / name
        if target.exists() and target.read_bytes() != (source / name).read_bytes():
            raise ValueError(f"{name} already exists and differs. Choose a different folder.")
    destination.mkdir(parents=True, exist_ok=True)
    for name in EXPORT_TOOLS:
        target = destination / name
        if not target.exists():
            shutil.copy2(source / name, target)
    return [str(destination / name) for name in EXPORT_TOOLS]
