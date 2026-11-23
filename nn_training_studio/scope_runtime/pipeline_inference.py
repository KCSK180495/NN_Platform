"""Incremental inference after conversion to the model sampling rate."""
import numpy as np
from .pipeline_profile import FeatureStream
from .sscn_scope import resolve_inputs, needs_healthy_start


class WindowSession:
    def __init__(self, bundle, scaler, healthy):
        self.profile = bundle["profile"]
        self.profile.bind(bundle["metadata"], bundle["model"], scaler)
        self.features = FeatureStream(self.profile, healthy)
        self.model = bundle["model"]
        self.scaler = scaler
        self.classes = bundle["metadata"]["class_names"]
        self.window = self.profile.spec["window"]
        self.stride = self.profile.spec["stride"]
        self.pending = np.empty((0, len(self.profile.spec["outputs"])), np.float32)
        self.next_window = None

    def push(self, start, current, rpm):
        return self.push_signals(start, {"current_A":current, "speed_rpm":rpm})

    def push_signals(self, start, inputs):
        first, values = self.features.push_signals(start, inputs)
        if not len(values):
            return []
        if self.next_window is None:
            self.next_window = first
        scaled = values if self.scaler is None else self.scaler.transform(values.astype(np.float64)).astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError("Model scaler produced nonfinite values.")
        self.pending = np.concatenate((self.pending, scaled))
        offsets = range(0, len(self.pending)-self.window+1, self.stride)
        output = []
        for pos in range(0, len(offsets), 64):
            batch = offsets[pos:pos+64]
            x = np.stack([self.pending[i:i+self.window] for i in batch])
            probs = np.asarray(self.model.predict_on_batch(x))
            if (probs.shape != (len(batch), len(self.classes)) or not np.isfinite(probs).all()
                    or np.any(probs < 0) or np.any(probs > 1) or not np.allclose(probs.sum(1),1,atol=.001)):
                raise ValueError("Profile classifier produced invalid probabilities.")
            output.extend((self.next_window+i, p) for i,p in zip(batch,probs))
        used = len(offsets)*self.stride
        self.next_window += used
        self.pending = self.pending[used:].copy()
        return output


class CaptureSession:
    def __init__(self, capture, result, bundle, mapping, settings, rpm, scaler):
        self.length = len(result.values)
        raw_inputs = bundle["profile"].spec["inputs"]
        indices, constant_rpm, issues = resolve_inputs(raw_inputs, capture.channels, mapping, settings, rpm)
        if issues:
            raise ValueError(" ".join(issues))
        self.inputs = {name: result.values[:, index] for name, index in indices.items()}
        if constant_rpm is not None:
            self.inputs["speed_rpm"] = np.broadcast_to(constant_rpm, (self.length,))
        current = mapping.get("current_A")
        self.result = result
        self.cid = capture.capture_id
        self.cursor = 0
        self.runner = WindowSession(bundle, scaler, settings.get("healthy_startup",False))
        source = settings.get("speed_source") if "speed_rpm" in raw_inputs else None
        speed_channel = settings.get("speed_channel") if source == "Channel" else None
        result.model_info.update(sensor_channel=current, speed_source=source, speed_channel=speed_channel,
                                 profile_name=self.runner.profile.spec["name"], profile_sha256=self.runner.profile.digest,
                                 healthy_startup_confirmed=bool(settings.get("healthy_startup")) and needs_healthy_start(bundle))
        result.status = "Waiting for warmup and a complete window"

    def advance(self, stop, cancel=None):
        r = self.result
        if not self.cursor <= stop <= self.length:
            raise ValueError("Invalid capture block boundary.")
        while self.cursor < stop:
            if cancel is not None and cancel.is_set():
                r.predictions.clear()
                r.status = "CANCELLED - analysis stopped"
                return
            end = min(stop, self.cursor+1024)
            outputs = self.runner.push_signals(self.cursor,{k:v[self.cursor:end] for k,v in self.inputs.items()})
            for start, prob in outputs:
                last = start+self.runner.window-1
                label = int(prob.argmax())
                r.predictions.append(dict(capture_id=self.cid,start_sample=start,end_sample_exclusive=last+1,
                    start_time_s=float(r.time_s[start]),end_time_s=float(r.time_s[last]),
                    prediction=self.runner.classes[label],confidence=float(prob[label]),
                    probabilities=prob.tolist(), task="Classification"))
            self.cursor = end
        state = self.runner.features
        r.model_info.update(valid_start_sample=state.valid_start)
        if state.reference_source is not None:
            r.model_info.update(healthy_reference_source=state.reference_source,
                               healthy_reference_id=state.reference_id)
        if state.calibration_end is not None:
            r.model_info.update(calibration_end_sample=state.calibration_end-1, calibration_speed_rpm=state.baseline)
            if state.offset >= state.calibration_end:
                r.model_info["calibration_peak_a"] = state.peak
        if r.predictions:
            r.status = f"ACTIVE - {len(r.predictions)} windows in this capture"
        elif stop == self.length:
            raise ValueError("Capture too short for calibration and a complete model window.")
