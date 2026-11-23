"""Phase-local core4 features for the CSFDS SSCN capture contract."""
import numpy as np
from scipy import signal

CONTRACT = "CSFDS_PHASE_SSCN_V1"


def profile_inputs(bundle):
    if bundle.get("profile"):
        return bundle["profile"].spec["inputs"]
    if bundle["metadata"].get("contract") == CONTRACT:
        return {"current_A": "A", "speed_rpm": "rpm"}
    return {}


def needs_healthy_start(bundle):
    if bundle.get("profile"):
        return any(step["op"] == "healthy_cycle_peak.v1" for step in bundle["profile"].spec["steps"])
    return bundle["metadata"].get("contract") == CONTRACT


def resolve_inputs(required, channels, mapping, settings, manual_rpm):
    """Resolve calibrated channel indices and optional constant speed before reading samples."""
    available = {c.name: (index, c.unit.strip()) for index, c in enumerate(channels)}
    indices, issues = {}, []
    rpm = None
    for name, unit in required.items():
        if name == "speed_rpm":
            if settings.get("speed_source") == "User value":
                try:
                    rpm = float(manual_rpm)
                except (TypeError, ValueError):
                    rpm = None
                if rpm is None or not np.isfinite(rpm) or rpm <= 0:
                    issues.append("Enter a positive mechanical speed in rpm in Diagnoses.")
                continue
            if settings.get("speed_source") != "Channel":
                issues.append("Select a speed source in Diagnoses.")
                continue
            source = settings.get("speed_channel")
            if source in mapping.values():
                issues.append("Select a separate speed channel calibrated in rpm.")
                continue
            label = "speed channel"
        else:
            source, label = mapping.get(name), name
        if source not in available:
            issues.append(f"Map {label} to an enabled {unit} channel in Diagnoses (currently {source or 'unmapped'}).")
            continue
        index, actual = available[source]
        if actual.casefold() != unit.strip().casefold():
            issues.append(f"{label} on {source} requires {unit}. The channel unit is {actual or 'unset'}.")
            continue
        indices[name] = index
    return indices, rpm, issues


def capture_inputs(metadata):
    if metadata.get("pipeline_inputs"):
        return [name for name in metadata["pipeline_inputs"] if name != "speed_rpm"]
    if metadata.get("contract") == CONTRACT:
        return ["current_A"]
    return list(metadata.get("feature_cols") or [])


def prepare(capture, result, metadata, mapping, settings, manual_rpm):
    phase = metadata.get("phase")
    expected = ["speed_rpm", phase, f"{phase}_SSCN", f"SS_peak_{phase}"]
    if (phase not in ("U", "V", "W") or metadata.get("feature_mode") != "core4"
            or metadata.get("feature_cols") != expected):
        raise ValueError("This SSCN adapter requires a phase-local core4 package.")
    if not settings.get("healthy_startup"):
        raise ValueError("Confirm a known-healthy startup for this record in Diagnoses.")
    cfg = metadata["preprocessing"]
    fs = cfg["sample_rate_hz"]
    if not np.isclose(result.sample_rate_hz, fs, rtol=1e-6):
        raise ValueError("Analysis rate differs from the SSCN training rate.")
    indices, rpm, issues = resolve_inputs({"current_A": "A", "speed_rpm": "rpm"},
                                          capture.channels, mapping, settings, manual_rpm)
    if issues:
        raise ValueError(" ".join(issues))
    current_name = mapping.get("current_A")
    current = result.values[:, indices["current_A"]].astype(np.float32).astype(np.float64)
    if not len(current) or not np.isfinite(current).all():
        raise ValueError("Current must contain finite samples.")
    speed = result.values[:, indices["speed_rpm"]] if rpm is None else np.full(len(current), rpm)
    if not np.isfinite(speed).all() or np.any(speed <= 0):
        raise ValueError("SSCN requires finite, positive mechanical rpm.")
    sos = signal.butter(cfg["filter_order"], cfg["cutoff_hz"], fs=fs, output="sos")
    filtered, _ = signal.sosfilt(sos, current, zi=signal.sosfilt_zi(sos) * current[0])
    start = cfg["filter_warmup_samples"]
    if len(current) <= start:
        raise ValueError("Capture is shorter than the SSCN filter warmup.")
    baseline = float(speed[start])
    count = int(np.ceil(cfg["calibration_cycles"] * fs * 60 / (cfg["pole_pairs"] * baseline)))
    stop = start + count
    if stop > len(current) or not np.allclose(speed[start:stop], baseline, rtol=.001, atol=.01):
        raise ValueError("Capture one full healthy electrical cycle at stable startup speed.")
    peak = float(np.max(np.abs(filtered[start:stop])))
    if peak <= cfg["min_peak_a"]:
        raise ValueError("Healthy calibration peak is too small.")
    valid_start = max(stop - 1, start + cfg["statistics_window"] - 1)
    features = np.column_stack((speed, filtered, filtered/peak, np.full(len(speed), peak))).astype(np.float32)
    result.model_info.update(sensor_channel=current_name, speed_source=settings["speed_source"],
                             speed_channel=settings.get("speed_channel") if rpm is None else None,
                             calibration_speed_rpm=baseline, calibration_peak_a=peak,
                             calibration_start_sample=start, calibration_end_sample=stop-1,
                             valid_start_sample=valid_start, healthy_startup_confirmed=True,
                             effective_signal_filter=cfg)
    return features, valid_start
