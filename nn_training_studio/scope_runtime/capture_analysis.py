"""Pure capture processing and export; no Tk calls or implicit model training."""
from dataclasses import dataclass, field
from fractions import Fraction
from .sscn_scope import CONTRACT, prepare
from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
from .model_signal_filter import FILTER_METHOD,apply_model_signal_filter,validate_signal_filter


class _AnalysisCancelled(Exception):
    """Cooperative cancellation between model prediction batches."""


def _check_cancelled(cancel):
    if cancel is not None and cancel.is_set():
        raise _AnalysisCancelled()


def model_sample_rate(metadata):
    for key in ("sampling_frequency_hz", "sample_rate_hz", "sampling_rate_hz", "sample_rate", "fs"):
        if metadata.get(key) is not None:
            value = float(metadata[key])
            if np.isfinite(value) and value > 0:
                return value
    return None


def resample_capture(values, native_fs, target_fs):
    if not np.isfinite(target_fs) or target_fs <= 0:
        raise ValueError("Analysis sample rate must be positive and finite.")
    if target_fs > native_fs * (1 + 1e-8):
        raise ValueError(f"Scope captured at {native_fs:g} Hz, below requested {target_fs:g} Hz. "
                         "Increase scope sample rate; upsampling cannot recover missing information.")
    if np.isclose(target_fs, native_fs, rtol=1e-8, atol=0):
        return values.copy(), native_fs, 0
    ratio = Fraction(target_fs / native_fs).limit_denominator(100000)
    up, down = ratio.numerator, ratio.denominator
    actual_fs = native_fs * up / down
    if abs(actual_fs - target_fs) / target_fs > 1e-6 or max(up, down) > 100000:
        raise ValueError("Unsupported resampling ratio. Choose a simpler scope/analysis sample rate.")
    if len(values) * up / down > 1000000:
        raise ValueError("Analysis record exceeds one million points.")
    # Low-pass FIR is applied before decimation. Remove a conservative edge region
    # influenced by finite-record padding; do not invent samples across captures.
    output = resample_poly(values, up, down, axis=0, padtype="line")
    trim = int(np.ceil(10 * max(up, down) / down)) + 1
    if len(output) <= 2 * trim:
        raise ValueError("Capture is too short after anti-alias resampling. Increase timebase/record duration.")
    return output[trim:-trim], actual_fs, trim


@dataclass
class Analysis:
    time_s: np.ndarray
    values: np.ndarray
    sample_rate_hz: float
    predictions: list
    status: str
    inference_ms: float
    model_info: dict
    diagnosis_results: list = field(default_factory=list)


def analyse(capture, target_fs, bundle=None, mapping=None,
            scaler=None, cancel=None, operating_speed_rpm=None, sscn_settings=None):
    """Return display data even when model preprocessing/prediction is incompatible."""
    model_info = {}
    desired_fs = target_fs
    if bundle:
        metadata = bundle['metadata']
        model_info = {"metadata": metadata, "package_path": bundle.get('package_path'),
                      "package_sha256": bundle.get('package_sha256'),
                      "input_mapping": dict(mapping or {}),
                      "effective_scaler": type(scaler).__name__ if scaler is not None else "None"}
    # An acquisition can always be saved even if analysis resampling is impossible.
    try:
        if bundle:
            desired_fs = model_sample_rate(metadata) or target_fs
        values, fs, trim = resample_capture(capture.values, capture.sample_rate_hz, desired_fs)
    except Exception as exc:
        return Analysis(capture.time_s, capture.values, capture.sample_rate_hz, [],
                        f"ANALYSIS BLOCKED: {exc}", 0, model_info)
    times = capture.time_s[0] + (np.arange(len(values)) + trim) / fs
    result = Analysis(times, values, fs, [], "NO MODEL — acquisition only", 0, model_info)
    if not bundle:
        return result
    began = time.perf_counter()
    try:
        _check_cancelled(cancel)
        if model_sample_rate(metadata) is None:
            raise ValueError("Model metadata has no training sample rate. Add sampling_frequency_hz to metadata.json.")
        if metadata.get("pipeline_sha256") and bundle.get("profile") is None:
            raise ValueError("Load the preprocessing profile declared by this model in Diagnoses.")
        if bundle.get("profile") is not None:
            from .pipeline_inference import CaptureSession
            session = CaptureSession(capture, result, bundle, mapping or {}, sscn_settings or {},
                                     operating_speed_rpm, scaler)
            session.advance(len(values), cancel)
            result.inference_ms = (time.perf_counter() - began) * 1000
            return result
        features = list(metadata.get('feature_cols') or [])
        win, stride = int(metadata.get('window_size', 0)), int(metadata.get('stride', 1))
        if win < 1 or stride < 1:
            raise ValueError("Model window/stride metadata must be positive.")
        if metadata.get("contract") == CONTRACT:
            selected, model_filter_warmup = prepare(
                capture, result, metadata, mapping or {}, sscn_settings or {}, operating_speed_rpm)
        else:
            available = [c.name for c in capture.channels]
            if not features or any((mapping or {}).get(f) not in available for f in features):
                raise ValueError("Complete Input Mapping for every model feature.")
            if len({mapping[f] for f in features}) != len(features):
                raise ValueError("Each model feature must map to a different captured channel.")
            selected = values[:, [available.index(mapping[f]) for f in features]]
            if not np.isfinite(selected).all():
                raise ValueError('Non-finite captured input in mapped channels; check sensor data before filtering.')
            model_filter_warmup = 0
            if metadata.get('filter_method') == FILTER_METHOD:
                spec = validate_signal_filter(metadata.get('signal_filter'), fs)
                selected = apply_model_signal_filter(selected, fs, spec)
                model_filter_warmup = spec['warmup_samples']
                model_info['effective_signal_filter'] = spec
        selected = np.asarray(selected, dtype=float)
        expected = str(metadata.get('scaler_type') or 'None')
        if scaler is None and expected.lower() not in ('none', '', 'null'):
            raise ValueError(f"Missing fitted {expected}; load the training scaler.")
        if scaler is not None:
            if getattr(scaler, 'n_features_in_', len(features)) != len(features):
                raise ValueError("Scaler feature count does not match model inputs.")
            if expected in ('StandardScaler', 'MinMaxScaler') and type(scaler).__name__ != expected:
                raise ValueError(f"Model expects {expected}, received {type(scaler).__name__}.")
            names = getattr(scaler, 'feature_names_in_', None)
            if names is not None and list(names) != features:
                raise ValueError("Scaler feature names/order differ from model metadata.")
            selected = scaler.transform(pd.DataFrame(selected, columns=features) if names is not None else selected)
        if not np.isfinite(selected).all():
            raise ValueError("Non-finite preprocessed model input.")
        model = bundle['model']
        shape = model.input_shape
        if isinstance(shape, list) or len(shape) not in (2, 3):
            raise ValueError("This adapter supports one raw time-series input (2D flattened or 3D). "
                             "STFT/CWT, multi-input, and controller feature engineering need a dedicated adapter.")
        wanted = (win * len(features),) if len(shape) == 2 else (win, len(features))
        if any(a is not None and int(a) != b for a, b in zip(shape[1:], wanted)):
            raise ValueError(f"Model shape {shape} conflicts with window/features {wanted}.")
        warmup = model_filter_warmup
        starts = list(range(warmup, len(selected) - win + 1, stride))
        if not starts:
            raise ValueError(f"Need at least {win + warmup} analysis samples in one capture; got {len(selected)}. "
                             "Increase the captured duration. Separate captures are not joined.")
        classes = list(metadata.get('class_names') or [])
        for pos in range(0, len(starts), 64):
            _check_cancelled(cancel)
            batch_starts = starts[pos:pos + 64]
            x = np.stack([selected[s:s + win] for s in batch_starts]).astype(np.float32)
            if len(shape) == 2:
                x = x.reshape(len(x), -1)
            output = model.predict_on_batch(x)
            _check_cancelled(cancel)
            if isinstance(output, (list, dict)):
                raise ValueError("Multi-head model outputs require a dedicated output adapter.")
            output = np.asarray(output)
            if output.ndim == 1:
                output = output[:, None]
            if output.ndim != 2 or len(output) != len(x) or not np.isfinite(output).all():
                raise ValueError("Expected finite per-window predictions.")
            for start, prediction in zip(batch_starts, output):
                row = dict(capture_id=capture.capture_id, start_sample=start,
                           end_sample_exclusive=start + win, start_time_s=float(times[start]),
                           end_time_s=float(times[start + win - 1]))
                probs = np.array([1 - prediction[0], prediction[0]]) if len(prediction) == 1 else prediction
                if np.any(probs < 0) or np.any(probs > 1) or not np.isclose(probs.sum(), 1, atol=.01):
                    raise ValueError("Classifier output is not probabilities; specify a logits/output adapter.")
                idx = int(np.argmax(probs))
                if classes and len(classes) != len(probs):
                    raise ValueError("Class labels do not match classifier output size.")
                row.update(prediction=classes[idx], confidence=float(probs[idx]),
                           probabilities=probs.tolist(), task="Classification")
                result.predictions.append(row)
        result.status = f"ACTIVE — {len(starts)} windows in this capture"
    except _AnalysisCancelled:
        result.predictions = []
        result.status = 'CANCELLED - model analysis stopped; captured samples retained'
    except Exception as exc:
        result.predictions = []
        result.status = f"INFERENCE BLOCKED: {exc}"
    result.inference_ms = (time.perf_counter() - began) * 1000
    return result


def save_capture(directory, capture, result, *, recording_sequence=None):
    """One directory per capture: native values, resampled values, window results, metadata.

    The metadata file is written last as a completion marker. Partial writes stay visible.
    """
    order = "" if recording_sequence is None else f"{recording_sequence:06d}_"
    root = Path(directory) / f"capture_{order}{capture.capture_id}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "metadata.json").write_text('{"export_complete": false}', encoding='utf-8')
    if not capture.metadata.get("replay_analysis_only"):
        raw = pd.DataFrame({"Time_s": capture.time_s})
        for j, c in enumerate(capture.channels):
            raw[f"CH{c.number}_scope"] = capture.scope_values[:, j]
            raw[f"CH{c.number}_zero_corrected_scope"] = capture.scope_values[:, j] - c.sensor_zero
            raw[f"CH{c.number}_calibrated_{c.name}"] = capture.values[:, j]
        raw.to_csv(root / "native_samples.csv", index=False)
    display = pd.DataFrame(result.values, columns=[c.name for c in capture.channels])
    display.insert(0, 'Time_s', result.time_s)
    display.to_csv(root / "analysis_samples.csv", index=False)
    rows = []
    for row in result.predictions:
        row = dict(row)
        row['probabilities'] = json.dumps(row.get('probabilities', []))
        rows.append(row)
    columns = ['capture_id', 'start_sample', 'end_sample_exclusive',
               'start_time_s', 'end_time_s', 'prediction', 'confidence', 'probabilities', 'task']
    if result.diagnosis_results:
        columns += ['diagnosis_id', 'diagnosis_name', 'sample_rate_hz']
    pd.DataFrame(rows, columns=columns).to_csv(
        root / "predictions.csv", index=False)
    metadata = dict(capture.metadata, capture_id=capture.capture_id,
                    native_sample_rate_hz=(capture.metadata.get("native_sample_rate_hz")
                                           if capture.metadata.get("replay_analysis_only") else capture.sample_rate_hz),
                    analysis_sample_rate_hz=result.sample_rate_hz,
                    analysis_status=result.status, inference_ms=result.inference_ms,
                    model=result.model_info, export_complete=True)
    if recording_sequence is not None:
        metadata.update(recording_session=root.parent.name, recording_sequence=recording_sequence)
    if result.diagnosis_results:
        metadata['diagnoses'] = [{k: v for k, v in row.items() if k != 'predictions'}
                                 for row in result.diagnosis_results]
    temporary = root / 'metadata.tmp'
    temporary.write_text(json.dumps(metadata, indent=2, default=str), encoding='utf-8')
    temporary.replace(root / 'metadata.json')
    return str(root)
