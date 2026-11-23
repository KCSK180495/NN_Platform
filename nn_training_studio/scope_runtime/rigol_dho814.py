"""DHO814 SCPI acquisition. All calls must be made by one owning worker.

Based on RIGOL DHO800/900 Programming Guide, sections 3.27, 3.28, 4.3.
ASCII returns engineering values directly and avoids 8-bit BYTE quantisation.
No hardware connection is made on import. USB uses an installed VISA runtime.
"""
from dataclasses import asdict
from datetime import datetime, timezone
from threading import Lock
import time
import uuid
import numpy as np
from .channel_calibration import calibrate_channels, CALIBRATION_FORMULA
from .scope_types import (Channel, Capture, AcquisitionCancelled, AcquisitionTimeout,
                         CaptureSettingsError, validate_channels, ScopeCapabilities)


def parse_preamble(text):
    fields = [float(x) for x in text.strip().split(',')]
    if len(fields) != 10 or not np.all(np.isfinite(fields)):
        raise ValueError("Invalid DHO waveform preamble: expected ten finite values.")
    fmt, mode, points, count, dx, x0, xr, dy, y0, yr = fields
    if dx <= 0 or points < 1 or points != int(points):
        raise ValueError("Invalid waveform sample interval or point count.")
    return dict(format=int(fmt), mode=int(mode), points=int(points), count=count,
                dx=dx, x0=x0, xr=xr, dy=dy, y0=y0, yr=yr)


class _VisaManagerLease:
    """PyVISA shares a manager across scans and connections in the same process."""

    _lock = Lock()
    _users = {}

    def __init__(self, factory, backend):
        with self._lock:
            self._manager = factory(backend)
            self._users[self._manager] = self._users.get(self._manager, 0) + 1

    def open_resource(self, address):
        return self._manager.open_resource(address)

    def list_resources(self, query):
        return self._manager.list_resources(query)

    def close(self):
        with self._lock:
            manager, self._manager = self._manager, None
            if manager is None:
                return
            self._users[manager] -= 1
            if not self._users[manager]:
                del self._users[manager]
                manager.close()


def resource_manager(backend="@ivi"):
    try:
        import pyvisa
    except ImportError as exc:
        raise RuntimeError("Install PyVISA: python -m pip install pyvisa") from exc
    try:
        return _VisaManagerLease(pyvisa.ResourceManager, backend)
    except Exception as exc:
        raise RuntimeError(
            "Cannot open VISA backend. For Windows USB install NI-VISA (or a compatible "
            "VISA runtime) and use @ivi. For Ethernet, @py with pyvisa-py is also available. "
            f"Details: {exc}"
        ) from exc


def scan_usb(backend="@ivi"):
    rm = resource_manager(backend)
    try:
        return list(rm.list_resources("USB?*::INSTR"))
    finally:
        rm.close()


class DHO814:
    def __init__(self, resource, backend="@ivi", timeout_ms=5000, manager=None, instrument=None, identity=None):
        if not (resource.upper().startswith("USB") or resource.upper().startswith("TCPIP")):
            raise ValueError("Use a USBTMC or TCPIP VISA resource, not a serial COM port.")
        self.rm = manager if manager is not None else resource_manager(backend)
        self.inst = None
        self.resource = resource
        self.backend = backend
        try:
            self.inst = instrument if instrument is not None else self.rm.open_resource(resource)
            self.inst.timeout = timeout_ms
            self.inst.read_termination = "\n"
            self.inst.write_termination = "\n"
            self.inst.chunk_size = 1024 * 1024
            self.identity = identity if identity is not None else self.inst.query("*IDN?").strip()
            parts = self.identity.upper().split(',')
            if len(parts) < 2 or "RIGOL" not in parts[0] or parts[1].strip() != "DHO814":
                raise ValueError(f"Expected a RIGOL DHO814; received {self.identity!r}.")
            self.capabilities = ScopeCapabilities('rigol-dho814', 'RIGOL', 'DHO814', (1, 2, 3, 4),
                software_setup=True, memory_depths=(10000, 100000, 1000000),
                validation='USB capture checked on DHO814 firmware 00.01.04', force_trigger=True)
        except Exception:
            self.close()
            raise

    def close(self):
        try:
            if self.inst is not None:
                self.inst.close()
        finally:
            self.inst = None
            self.rm.close()

    def _check_errors(self):
        errors = []
        for _ in range(12):
            reply = self.inst.query(":SYST:ERR?").strip()
            if int(reply.split(',')[0]) == 0:
                break
            errors.append(reply)
        if errors:
            raise RuntimeError("Oscilloscope reported: " + "; ".join(errors))

    def _trigger_settings(self):
        settings = {'type': self.inst.query(':TRIG:MODE?').strip().upper()}
        if settings['type'] == 'EDGE':
            settings['source'] = self.inst.query(':TRIG:EDGE:SOUR?').strip()
            level = float(self.inst.query(':TRIG:EDGE:LEV?').strip())
            if not np.isfinite(level):
                raise ValueError('Invalid oscilloscope trigger level.')
            settings['level_scope_units'] = level
        return settings

    def _verify_stopped(self):
        """Confirm that the instrument is stopped and ready for another capture."""
        self.inst.write(':STOP')
        # DHO814 firmware 00.01.04 returns 0 to *OPC? even while stopped.
        # The trigger state is the acknowledgement we need before another shot.
        deadline = time.monotonic() + 1.0
        while True:
            state = self.inst.query(':TRIG:STAT?').strip().upper()
            if state == 'STOP':
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f'Oscilloscope did not stop. Last trigger state was {state!r}.')
            time.sleep(.05)
        if self.inst.query('*IDN?').strip() != self.identity.strip():
            raise RuntimeError('Oscilloscope identity changed during capture.')
        self._check_errors()

    def capture(self, channels, cancel, *, max_points=1000000, block_points=10000,
                trigger_timeout=10.0, setup=False, time_scale=0.01, memory_depth=100000,
                force_trigger=False):
        """Acquire once, wait for completion, read all selected channels while stopped.

        Leaves the instrument STOPPED. setup optionally sets only main timebase,
        memory depth and normal sample acquisition; vertical/trigger levels stay as set.
        Cancellation is cooperative, bounded by an outstanding VISA query timeout.
        """
        validate_channels(channels)
        if not (1 <= block_points <= 100000 and 1 <= max_points <= 25000000):
            raise CaptureSettingsError("Invalid transfer block size or maximum record size.")
        if not 0 < trigger_timeout <= 120:
            raise CaptureSettingsError("Trigger timeout must be between 0 and 120 seconds.")
        if setup and (not np.isfinite(time_scale) or not 1e-8 <= time_scale <= 1):
            raise CaptureSettingsError("Timebase must be 10 ns/div to 1 s/div for this application.")
        if setup and memory_depth not in (10000, 100000, 1000000):
            raise CaptureSettingsError("Select 10k, 100k or 1M memory depth.")
        def check_cancel():
            if cancel.is_set():
                raise AcquisitionCancelled()
        check_cancel()
        i = self.inst
        # Detect existing scope errors instead of silently clearing them.
        self._check_errors()
        i.write(":STOP")
        for c in channels:
            i.write(f":CHAN{c.number}:DISP ON")
        if setup:
            i.write(":TIM:MODE MAIN")
            i.write(f":TIM:SCAL {time_scale:.12g}")
            i.write(":ACQ:TYPE NORM")
            i.write(f":ACQ:MDEP {memory_depth}")
        self._check_errors()
        trigger_settings = self._trigger_settings()
        self._check_errors()
        force_sent = False
        armed = time.time()
        began = time.perf_counter()
        try:
            check_cancel()
            i.write(":SING")
            # Poll status directly: an *OPC? completion wait here can prevent
            # the force command and our trigger deadline from being reached.
            deadline = time.monotonic() + trigger_timeout
            while True:
                check_cancel()
                state = i.query(":TRIG:STAT?").strip().upper()
                if state == "STOP":
                    break
                if time.monotonic() >= deadline:
                    self._verify_stopped()
                    cause = 'Capture did not finish after force trigger.' if force_sent else 'No completed hardware trigger.'
                    source = trigger_settings.get('source', trigger_settings['type'])
                    advice = ('Check the timebase and trigger settings, then capture again.' if force_trigger else
                              'Check the trigger source and level, or select “Force trigger while waiting” '
                              'on Setup for a probe test.')
                    raise AcquisitionTimeout(
                        f'{cause} Trigger source {source}, last state {state}. Connection retained. {advice}')
                if force_trigger and not force_sent and state == 'WAIT':
                    check_cancel()
                    i.write(':TFOR')
                    force_sent = True
                cancel.wait(0.05)
            completed = time.time()
            # PREamble.points follows the selected transfer range, which may still
            # be the final block of the previous read. It is NOT the record length.
            depth = float(i.query(":ACQ:MDEP?").strip())
            if not np.isfinite(depth) or depth < 1 or depth != int(depth):
                raise ValueError("Scope did not report a valid acquired memory depth.")
            n = int(depth)
            if n > max_points:
                self._verify_stopped()
                raise CaptureSettingsError(
                    f"The scope recorded {n:,} points per channel. The limit is {max_points:,}. "
                    "Reduce the record depth on the scope or enable ‘Set timebase and depth on DHO814’ "
                    "in capture settings, then capture again.")
            columns, preambles, channel_settings, axis = [], {}, {}, None
            for c in channels:
                check_cancel()
                i.write(f":WAV:SOUR CHAN{c.number}")
                i.write(":WAV:MODE RAW")
                i.write(":WAV:FORM ASC")
                i.write(":WAV:STAR 1")
                i.write(f":WAV:STOP {n}")
                p = parse_preamble(i.query(":WAV:PRE?"))
                if p['format'] != 2 or p['mode'] != 2:
                    raise RuntimeError("Scope did not accept RAW / ASCII waveform mode.")
                if p['points'] != n:
                    raise ValueError(f"Scope exposed {p['points']:,} of {n:,} acquired points for "
                                     f"CH{c.number}; refusing to save a partial record.")
                offset = float(i.query(f":CHAN{c.number}:OFFS?").strip())
                if not np.isfinite(offset):
                    raise ValueError(f"Invalid oscilloscope vertical offset for CH{c.number}.")
                channel_settings[f"CH{c.number}"] = {"vertical_offset_v": offset}
                # xreference is a reference sample position, normally zero on DHO.
                t = (np.arange(n, dtype=float) - p['xr']) * p['dx'] + p['x0']
                if axis is not None and (len(t) != len(axis) or
                                        not np.allclose(t, axis, rtol=0, atol=p['dx'] * 1e-5)):
                    raise ValueError("Channels do not share the same capture time grid.")
                axis = t
                blocks = []
                for start in range(1, n + 1, block_points):
                    check_cancel()
                    end = min(n, start + block_points - 1)
                    # Extend the stop first when advancing, so start never exceeds the old stop.
                    i.write(f":WAV:STOP {end}")
                    i.write(f":WAV:STAR {start}")
                    try:
                        data = np.asarray(i.query_ascii_values(":WAV:DATA?", container=np.array), dtype=float)
                    except Exception as exc:
                        raise RuntimeError(f"CH{c.number} waveform read failed at samples "
                                           f"{start:,}–{end:,} of {n:,} (:WAV:DATA?). {exc}") from exc
                    if data.ndim != 1 or len(data) != end - start + 1 or not np.isfinite(data).all():
                        raise ValueError(f"Incomplete/invalid CH{c.number} waveform block {start}:{end}.")
                    blocks.append(data)
                columns.append(np.concatenate(blocks))
                preambles[f"CH{c.number}"] = p
            self._check_errors()
            raw = np.column_stack(columns)
            values = calibrate_channels(raw, channels)
            return Capture(uuid.uuid4().hex, axis, values, raw, 1 / p['dx'], tuple(channels), {
                "identity": self.identity, "resource": self.resource, "backend": self.backend,
                "driver_id": self.capabilities.driver_id, "device_vendor": "RIGOL", "device_model": "DHO814",
                "capabilities": asdict(self.capabilities),
                "format": "ASCII", "mode": "RAW", "preambles": preambles,
                "acquired_points_per_channel": n, "transferred_points_per_channel": len(axis),
                "full_record_transferred": True,
                "record_start_time_s": float(axis[0]), "record_end_time_s": float(axis[-1]),
                "channel_settings": channel_settings,
                "trigger_settings": trigger_settings,
                "trigger_policy": 'force_if_waiting' if force_trigger else 'hardware',
                "force_trigger_command_sent": force_sent,
                "trigger_timing_note": 'A software force command does not mark the physical fault time. '
                                       'A natural trigger may complete before the force command is needed.',
                "offset_note": "ASCII samples already contain the physical signal/DC level. "
                               "Scope vertical offset is recorded separately, not applied twice. "
                               "Sensor zero is measured only under a known zero-input condition.",
                "channels": [asdict(c) for c in channels],
                "host_armed_unix_s": armed, "host_complete_observed_unix_s": completed,
                "received_utc": datetime.now(timezone.utc).isoformat(),
                "capture_and_transfer_ms": (time.perf_counter() - began) * 1000,
                "timing_note": "Time_s is relative to the scope trigger. Host times are not hardware timestamps. "
                               "Separate captures may have gaps; never concatenate as continuous data.",
                "calibration": CALIBRATION_FORMULA + "; probe ratio is already applied by scope",
            })
        except (AcquisitionTimeout, CaptureSettingsError):
            # Communication and STOP state were verified before raising here.
            raise
        except AcquisitionCancelled:
            self._verify_stopped()
            raise
        except Exception:
            try:
                i.write(":STOP")
            except Exception:
                pass
            raise
