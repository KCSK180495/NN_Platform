"""Run independent diagnoses and own the instrument on one worker thread."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from threading import Event
from uuid import uuid4

import numpy as np
import pandas as pd

from .capture_analysis import analyse, save_capture
from .csv_capture import playback_capture
from .package_io import load_package, sha256
from .pipeline_inference import CaptureSession
from .rigol_dho814 import scan_usb
from .scope_registry import open_scope
from .scope_types import AcquisitionCancelled, AcquisitionTimeout, CaptureSettingsError, validate_channels
from .sscn_scope import resolve_inputs


def _diagnosis(capture, task, bundle, healthy):
    inputs = bundle["inputs"]
    mapping = task.get("mapping", {})
    units = {name: unit or task.get("input_units", {}).get(name, "")
             for name, unit in inputs.items()}
    if any(not unit.strip() for unit in units.values()):
        raise ValueError("Confirm a model unit for every input with undeclared units. Use '1' for dimensionless.")
    settings = dict(task.get("speed_settings", {}), healthy_startup=healthy)
    rpm = task.get("rpm")
    if "speed_rpm" in units and not bundle["profile"] and not bundle["healthy_start_required"]:
        # A raw classifier consumes ordinary physical channels. Its name alone
        # does not opt it into the SSCN manual RPM contract.
        settings.update(speed_source="Channel", speed_channel=mapping.get("speed_rpm"))
        mapping = {name: channel for name, channel in mapping.items() if name != "speed_rpm"}
    _, _, issues = resolve_inputs(units, capture.channels, mapping, settings, rpm)
    if issues:
        raise ValueError(" ".join(issues))
    if bundle["healthy_start_required"] and not healthy:
        raise ValueError("Confirm that this record starts healthy.")
    return dict(mapping=task.get("mapping", {}), operating_speed_rpm=rpm,
                sscn_settings=settings, scaler=bundle["scaler"])


class Playback:
    """Share the profile inference engine between recorded and instrument data."""

    def __init__(self, capture, tasks, healthy, cancel, stack):
        self.capture = capture
        self.display = analyse(capture, capture.sample_rate_hz)
        self.position = 0
        self.jobs = []
        timeline = {}
        bundles = {}
        for task in tasks:
            if cancel.is_set():
                raise AcquisitionCancelled()
            session = None
            result = replace(self.display, predictions=[], model_info={})
            try:
                key = (task["model"], task.get("profile"), task.get("model_sha256"), task.get("profile_digest"))
                if task.get("profile_sha256") and sha256(task["profile"]) != task["profile_sha256"]:
                    raise ValueError("Selected profile differs from the saved session checksum.")
                if key not in bundles:
                    bundles[key] = stack.enter_context(load_package(*key))
                bundle = bundles[key]
                args = _diagnosis(capture, task, bundle, healthy)
                rate = bundle["metadata"]["sampling_frequency_hz"]
                if rate not in timeline:
                    timeline[rate] = analyse(capture, rate)
                base = timeline[rate]
                if base.status.startswith("ANALYSIS BLOCKED"):
                    raise ValueError(base.status)
                if bundle["profile"]:
                    info = dict(metadata=bundle["metadata"], package_sha256=bundle["package_sha256"],
                                input_mapping=args["mapping"], input_units=task.get("input_units", {}))
                    result = replace(base, predictions=[], model_info=info)
                    session = CaptureSession(capture, result, bundle, args["mapping"],
                                             args["sscn_settings"], args["operating_speed_rpm"], args["scaler"])
                else:
                    result = analyse(capture, rate, bundle=bundle, cancel=cancel, **args)
                    result.model_info["input_units"] = task.get("input_units", {})
                phase = bundle["metadata"].get("phase")
                physical_phase = task.get("physical_phase")
                result.model_info.update(trained_phase=phase, physical_phase=physical_phase,
                                         cross_phase_comparison=bool(phase and physical_phase and phase != physical_phase))
            except Exception as exc:
                result.status = f"INFERENCE BLOCKED: {exc}"
                result.predictions.clear()
            self.jobs.append([task, session, result])

    def advance(self, position, cancel):
        if not self.position <= position <= len(self.capture.time_s):
            raise ValueError("Playback position must advance within one capture.")
        if cancel.is_set():
            raise AcquisitionCancelled()
        self.position = position
        time = self.capture.time_s[position - 1] if position else -np.inf
        diagnoses, predictions = [], []
        for job in self.jobs:
            task, session, result = job
            if session and position:
                stop = int(np.searchsorted(result.time_s, time, side="right"))
                try:
                    session.advance(stop, cancel)
                except Exception as exc:
                    result.status = f"INFERENCE BLOCKED: {exc}"
                    result.predictions.clear()
                    job[1] = None
            identity = dict(diagnosis_id=task["id"], diagnosis_name=task["name"])
            visible = [dict(row, **identity, sample_rate_hz=result.sample_rate_hz)
                       for row in result.predictions if row["end_time_s"] <= time]
            status = result.status
            if not session and status.startswith("ACTIVE") and position < len(self.capture.time_s):
                status = f"Playback · {len(visible)} completed windows"
            diagnoses.append(dict(**identity, predictions=visible, status=status, enabled=True,
                                  sample_rate_hz=result.sample_rate_hz, inference_ms=result.inference_ms,
                                  model_info=dict(result.model_info)))
            predictions.extend(visible)
        if cancel.is_set():
            raise AcquisitionCancelled()
        active = sum(bool(row["predictions"]) for row in diagnoses)
        status = f"{active}/{len(diagnoses)} models produced predictions" if diagnoses else "Acquisition only"
        return replace(self.display, predictions=predictions, diagnosis_results=diagnoses, status=status,
                       model_info={"mode": "nn_scope", "source": self.capture.metadata.get("replay_source", "instrument")})


class ScopeRecording:
    """Save each completed shot in acquisition order in a new session folder."""

    def __init__(self, parent):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder = Path(parent) / f"NN_Scope_{stamp}_{uuid4().hex[:8]}"
        self.folder.mkdir()
        self.saved = 0

    def write(self, capture, result):
        sequence = self.saved + 1
        save_capture(self.folder, capture, result, recording_sequence=sequence)
        self.saved = sequence


class ScopeWorker:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="NN Scope")
        self.cancel = Event()
        self.scope = None
        self.playback = None
        self.packages = ExitStack()

    def submit(self, operation, *args):
        return self.executor.submit(operation, *args)

    def scan(self, backend):
        return scan_usb(backend)

    def connect(self, resource, backend):
        self.disconnect()
        self.scope = open_scope(resource, backend=backend)
        return self.scope.identity, self.scope.capabilities

    def disconnect(self):
        if self.scope:
            scope, self.scope = self.scope, None
            scope.close()

    def capture(self, channels, settings, tasks, healthy):
        if not self.scope:
            raise ValueError("Connect an oscilloscope first.")
        self.clear_playback()
        capabilities = self.scope.capabilities
        validate_channels(channels, capabilities.analog_channels)
        options = dict(settings)
        if not capabilities.force_trigger:
            if options.pop("force_trigger", False):
                raise ValueError("Force trigger is only supported by the RIGOL DHO814 driver.")
        if options.get("setup") and not capabilities.software_setup:
            raise ValueError("Configure the capture on this oscilloscope's front panel.")
        try:
            capture = self.scope.capture(channels, self.cancel, **options)
        except (AcquisitionTimeout, AcquisitionCancelled, CaptureSettingsError):
            raise
        except Exception as exc:
            try:
                self.disconnect()
            except Exception:
                pass
            raise ConnectionError(f"Capture or transfer failed. Reconnect the instrument. {exc}") from exc
        if not tasks:
            # Finish handling a complete waveform even if Stop arrives after transfer.
            result = analyse(capture, capture.sample_rate_hz)
            return capture, replace(result, status="Acquisition only",
                                    model_info={"mode": "nn_scope", "source": "instrument"})
        with ExitStack() as stack:
            run = Playback(capture, tasks, healthy, self.cancel, stack)
            return capture, run.advance(len(capture.time_s), self.cancel)

    def capture_live(self, channels, settings, recording):
        capture, result = self.capture(channels, settings, [], False)
        if recording is not None:
            try:
                recording.write(capture, result)
            except Exception as exc:
                message = (f"Recording stopped. Could not save to {recording.folder}. {exc} "
                           "Use Save results to keep this waveform in another folder.")
                return capture, result, message
        return capture, result, None

    def open_playback(self, path, tasks, healthy):
        self.clear_playback()
        path = Path(path)
        source_sha = sha256(path)
        source = playback_capture(path, pd.read_csv(path, float_precision="round_trip"))
        if sha256(path) != source_sha:
            raise ValueError("The capture CSV changed as it was read. Open it again.")
        # Every replay export has its own identity and retains the source ID.
        capture = replace(source, capture_id=uuid4().hex,
                          metadata=dict(source.metadata, replay_source=path.name,
                                        source_capture_id=source.metadata.get("source_capture_id") or source.capture_id,
                                        replay_parent_capture_id=source.capture_id,
                                        source_csv_sha256=source_sha))
        try:
            self.playback = Playback(capture, tasks, healthy, self.cancel, self.packages)
            return capture, self.playback.advance(0, self.cancel)
        except Exception:
            self.clear_playback()
            raise

    def advance(self, position):
        return self.playback.advance(position, self.cancel)

    def clear_playback(self):
        self.playback = None
        self.packages.close()

    def close(self):
        self.cancel.set()
        self.submit(self._cleanup)
        self.executor.shutdown(wait=False)

    def _cleanup(self):
        try:
            self.clear_playback()
        finally:
            self.disconnect()
