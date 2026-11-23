"""Read the physical channels from one saved NN Scope capture."""
import json
from pathlib import Path
import numpy as np
from .scope_types import Channel, Capture


def capture_metadata(path):
    path = Path(path)
    if path.name.lower() not in {"native_samples.csv", "analysis_samples.csv"}:
        raise ValueError("For capture diagnosis, open native_samples.csv or analysis_samples.csv from a saved capture.")
    metadata = path.with_name("metadata.json")
    if not metadata.is_file():
        raise ValueError("Keep the capture's metadata.json beside the CSV for channel units and sampling rate.")
    saved = json.loads(metadata.read_text(encoding="utf-8-sig"))
    if saved.get("export_complete") is False:
        raise ValueError("This capture export was not completed.")
    return saved


def playback_capture(path, frame):
    meta = capture_metadata(path)
    native = Path(path).name.lower() == "native_samples.csv"
    rate = float(meta["native_sample_rate_hz" if native else "analysis_sample_rate_hz"])
    times = frame["Time_s"].to_numpy(dtype=float)
    if len(times) < 2 or not np.isfinite(rate) or rate <= 0 or not np.isfinite(times).all() or not np.allclose(np.diff(times), 1/rate, rtol=1e-4, atol=1e-12):
        raise ValueError("CSV time spacing must match the saved sample rate and describe one continuous capture.")
    channels = tuple(Channel(**row) for row in meta["channels"])
    names = [c.name for c in channels]
    if not channels or len(set(names)) != len(names):
        raise ValueError("Saved channels must have distinct names.")
    calibrated = [f"CH{c.number}_calibrated_{c.name}" for c in channels] if native else names
    values = frame[calibrated].to_numpy(dtype=float)
    scope = frame[[f"CH{c.number}_scope" for c in channels]].to_numpy(dtype=float) if native else values
    return Capture(meta["capture_id"], times, values, scope, rate, channels,
                   dict(meta, replay_analysis_only=not native))
