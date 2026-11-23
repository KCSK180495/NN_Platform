"""Convert saved Scope records to calibrated data for Studio preparation."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .csv_capture import playback_capture
from .package_io import sha256


CAPTURE_ID = "scope_capture_id"
EVIDENCE_COLUMNS = frozenset({"Time_s", CAPTURE_ID, "scope_source_row", "annotation_segment_id"})


@dataclass
class CaptureDataset:
    frame: pd.DataFrame
    features: list
    rate: float
    metadata: dict


def load_capture_dataset(paths):
    frames, records, ids = [], [], set()
    expected = rate = None
    for item in paths:
        path = Path(item)
        if path.name.lower() != "native_samples.csv":
            raise ValueError("Select native_samples.csv for each capture you want to prepare.")
        frame = pd.read_csv(path, float_precision="round_trip")
        capture = playback_capture(path, frame)
        if not np.isfinite(capture.values).all():
            raise ValueError(f"{path} contains nonfinite calibrated values.")
        channels = {channel.name: channel.unit for channel in capture.channels}
        if EVIDENCE_COLUMNS.intersection(channels):
            raise ValueError("A channel name conflicts with the capture bookkeeping columns.")
        if expected is None:
            expected, rate = channels, capture.sample_rate_hz
        if channels != expected or not np.isclose(rate, capture.sample_rate_hz, rtol=1e-8, atol=0):
            raise ValueError("Select captures with matching channel names, units and native sample rates.")
        identity = capture.metadata.get("source_capture_id") or capture.capture_id
        if identity in ids:
            raise ValueError("Two selected files represent the same source capture.")
        ids.add(identity)
        data = pd.DataFrame(capture.values, columns=list(channels)).reindex(columns=list(expected))
        data.insert(0, "Time_s", capture.time_s)
        data[CAPTURE_ID] = identity
        data["scope_source_row"] = np.arange(len(data))
        data["annotation_segment_id"] = identity
        frames.append(data)
        records.append(dict(capture_id=identity, file=str(path), sha256=sha256(path),
                            channels=[asdict(channel) for channel in capture.channels]))
    if not frames:
        raise ValueError("Select at least one saved capture.")
    return CaptureDataset(pd.concat(frames, ignore_index=True), list(expected), rate,
                          dict(sources=records, sample_rate_hz=rate, units=expected))
