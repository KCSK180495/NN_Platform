"""Experimental family-specific drivers. USB/LAN VISA; no gap-free streaming.

Keysight: InfiniiVision 2000 X-Series Programmer's Guide, 9018-06893.
Tektronix: MSO/DPO4000B and MDO4000 Programmer Manual, 077051002.
Profiles match explicit model families; they are not blanket brand support.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import re
import time
import uuid
import numpy as np
from .channel_calibration import calibrate_channels, CALIBRATION_FORMULA
from .scope_types import Capture, ScopeCapabilities, AcquisitionCancelled, validate_channels


def normalized_identity(identity):
    parts = str(identity).strip().split(',')
    if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
        raise ValueError('Instrument returned an invalid *IDN? response.')
    return parts[0].strip().upper(), re.sub(r'[\s-]', '', parts[1].upper())


def keysight_match(identity):
    vendor, model = normalized_identity(identity)
    return vendor in ('KEYSIGHT TECHNOLOGIES', 'AGILENT TECHNOLOGIES') and bool(re.fullmatch(r'(DSO|MSO)X20[012][24]A', model))


def tektronix_match(identity):
    vendor, model = normalized_identity(identity)
    return vendor == 'TEKTRONIX' and bool(re.fullmatch(r'(?:DPO|MSO)(?:4014|4034|4054|4104)B|MDO(?:4014|4034|4054|4104)(?:3|6)?', model))


class BinaryScope:
    stop_command = ''
    vendor = ''
    driver_id = ''
    matcher = staticmethod(lambda _: False)

    def __init__(self, resource, backend='@ivi', timeout_ms=5000, manager=None, instrument=None):
        from .rigol_dho814 import resource_manager
        self.rm = manager if manager is not None else resource_manager(backend)
        self.inst = instrument
        self.resource, self.backend = resource, backend
        try:
            if self.inst is None:
                self.inst = self.rm.open_resource(resource)
            self.inst.timeout = timeout_ms
            self.inst.read_termination = '\n'
            self.inst.write_termination = '\n'
            self.inst.chunk_size = 1024 * 1024
            self.identity = self.inst.query('*IDN?').strip()
            if not self.matcher(self.identity):
                raise ValueError('Instrument does not match this driver family.')
            _, model = normalized_identity(self.identity)
            count = int(model[-2]) if self.vendor == 'Keysight' else 4
            self.capabilities = ScopeCapabilities(self.driver_id, self.vendor, self.identity.split(',')[1].strip(), tuple(range(1, count + 1)))
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

    def capture(self, channels, cancel, *, max_points=1000000, block_points=10000,
                trigger_timeout=10, setup=False, time_scale=.01, memory_depth=100000):
        validate_channels(channels, self.capabilities.analog_channels)
        if setup:
            raise ValueError('This experimental driver preserves instrument settings. Uncheck software setup and configure timebase/memory on the scope.')
        if not 1 <= max_points <= 25000000 or not 0 < trigger_timeout <= 120:
            raise ValueError('Invalid capture size limit or trigger timeout.')
        def check_cancel():
            if cancel.is_set():
                raise AcquisitionCancelled()
        check_cancel()
        start = time.perf_counter()
        armed = time.time()
        try:
            self.inst.write(self.stop_command)
            self._acquire(channels, cancel, trigger_timeout)
            check_cancel()
            completed = time.time()
            columns, preambles, axis = [], {}, None
            for channel in channels:
                check_cancel()
                values, t, preamble = self._read_channel(channel.number, max_points)
                if values.ndim != 1 or not len(values) or len(values) != len(t) or not np.isfinite(values).all():
                    raise ValueError(f'Incomplete/invalid CH{channel.number} waveform.')
                if axis is not None and (len(t) != len(axis) or not np.allclose(t, axis, rtol=0, atol=preamble['dx'] * 1e-5)):
                    raise ValueError('Channels do not share the same capture time grid.')
                axis = t
                columns.append(values)
                preambles[f'CH{channel.number}'] = preamble
            check_cancel()
            self._check_errors()
            raw = np.column_stack(columns)
            calibrated = calibrate_channels(raw, channels)
            return Capture(uuid.uuid4().hex, axis, calibrated, raw, 1 / preamble['dx'], tuple(channels), {
                'identity': self.identity, 'resource': self.resource, 'backend': self.backend,
                'driver_id': self.driver_id, 'device_vendor': self.vendor, 'device_model': self.capabilities.model,
                'capabilities': asdict(self.capabilities), 'format': 'signed 16-bit binary', 'mode': 'stopped record',
                'preambles': preambles, 'channels': [asdict(c) for c in channels],
                'host_armed_unix_s': armed, 'host_complete_observed_unix_s': completed,
                'received_utc': datetime.now(timezone.utc).isoformat(),
                'capture_and_transfer_ms': (time.perf_counter() - start) * 1000,
                'timing_note': 'One stopped capture. Separate captures have gaps; host observations are not trigger timestamps.',
                'validation': self.capabilities.validation, 'calibration': CALIBRATION_FORMULA,
            })
        except Exception:
            try:
                self.inst.write(self.stop_command)
            except Exception:
                pass
            raise

    def _count(self, number, max_points):
        value = float(number)
        if not np.isfinite(value) or value < 1 or value != int(value):
            raise ValueError('Invalid waveform point count.')
        if value > max_points:
            raise ValueError(f'Capture has {int(value):,} points/channel; limit is {max_points:,}. Reduce the scope record length or increase the limit.')
        return int(value)

    def _binary(self, query, count, big_endian):
        data = np.asarray(self.inst.query_binary_values(query, datatype='h', is_big_endian=big_endian,
            container=np.array, header_fmt='ieee', expect_termination=True), dtype=float)
        if data.ndim != 1 or len(data) != count:
            raise ValueError('Incomplete binary waveform transfer; record discarded.')
        return data


class Keysight2000X(BinaryScope):
    stop_command = ':STOP'
    vendor = 'Keysight'
    driver_id = 'keysight-infiniivision-2000x'
    matcher = staticmethod(keysight_match)

    def _check_errors(self):
        reply = self.inst.query(':SYST:ERR?').strip()
        if int(reply.split(',')[0]) != 0:
            raise RuntimeError('Keysight reported: ' + reply)

    def _acquire(self, channels, cancel, timeout):
        self._check_errors()
        if self.inst.query(':ACQ:TYPE?').strip().upper() not in ('NORM', 'NORMAL'):
            raise ValueError('Set Keysight acquisition type to Normal on the instrument.')
        if self.inst.query(':TIM:MODE?').strip().upper() != 'MAIN':
            raise ValueError('Set Keysight timebase mode to Main on the instrument.')
        for c in channels:
            self.inst.write(f':CHAN{c.number}:DISP ON')
        original_timeout = self.inst.timeout
        try:
            # DIGitize completes a new stopped acquisition. *OPC? can block until
            # trigger completion; cancellation is bounded by this VISA timeout.
            self.inst.timeout = int(timeout * 1000)
            self.inst.write(':DIG ' + ','.join(f'CHAN{c.number}' for c in channels))
            if self.inst.query('*OPC?').strip() != '1':
                raise RuntimeError('Keysight did not confirm digitize completion.')
        finally:
            self.inst.timeout = original_timeout

    def _read_channel(self, number, maximum):
        i = self.inst
        i.write(f':WAV:SOUR CHAN{number}')
        i.write(':WAV:FORM WORD')
        i.write(':WAV:BYT LSBF')
        i.write(':WAV:UNS 0')
        i.write(':WAV:POIN:MODE RAW')
        i.write(':WAV:POIN MAX')
        if i.query(':WAV:POIN:MODE?').strip().upper() != 'RAW':
            raise ValueError('Keysight did not accept RAW acquisition-record transfer.')
        fields = [float(v) for v in i.query(':WAV:PRE?').strip().split(',')]
        if len(fields) != 10 or not np.isfinite(fields).all():
            raise ValueError('Invalid Keysight waveform preamble.')
        fmt, mode, points, count, dx, x0, xr, dy, y0, yr = fields
        if fmt != 1 or mode != 0 or dx <= 0 or dy <= 0:
            raise ValueError('Keysight waveform is not a Normal WORD record with valid scaling.')
        n = self._count(points, maximum)
        # Verify selected maximum transfer contains the available native record.
        if n != self._count(i.query(':ACQ:POIN?'), maximum):
            raise ValueError('Keysight RAW transfer does not contain the full acquisition record.')
        if i.query(':WAV:BYT?').strip().upper() not in ('LSBF', 'LSBFIRST') or int(i.query(':WAV:UNS?')) != 0:
            raise ValueError('Keysight did not accept signed little-endian waveform encoding.')
        codes = self._binary(':WAV:DATA?', n, False)
        self._check_errors()
        return (codes - yr) * dy + y0, (np.arange(n) - xr) * dx + x0, dict(points=n, dx=dx, x0=x0, xr=xr, dy=dy, y0=y0, yr=yr)


class Tektronix4000(BinaryScope):
    stop_command = ':ACQ:STATE STOP'
    vendor = 'Tektronix'
    driver_id = 'tektronix-4000'
    matcher = staticmethod(tektronix_match)

    def _check_errors(self):
        # ALLEV? drains the Tek event queue; nonzero event codes reject the record.
        response = self.inst.query(':ALLEV?').strip()
        fields = response.split(',')
        if len(fields) < 2 or any(int(fields[index].strip().split(';')[-1]) != 0 for index in range(0, len(fields), 2)):
            raise RuntimeError('Tektronix reported an event: ' + response[:600])

    def _acquire(self, channels, cancel, timeout):
        self.inst.write(':HEADER OFF')
        self._check_errors()
        if self.inst.query(':ACQ:MODE?').strip().upper() != 'SAMPLE':
            raise ValueError('Set Tektronix acquisition mode to Sample on the instrument.')
        for c in channels:
            self.inst.write(f':SEL:CH{c.number} ON')
        self.inst.write(':ACQ:STOPA SEQ')
        self.inst.write(':ACQ:STATE RUN')
        deadline = time.monotonic() + timeout
        while self.inst.query(':ACQ:STATE?').strip() != '0':
            if cancel.is_set():
                raise AcquisitionCancelled()
            if time.monotonic() >= deadline:
                raise TimeoutError('No completed Tektronix acquisition; check trigger and record duration.')
            cancel.wait(.05)

    def _read_channel(self, number, maximum):
        i = self.inst
        i.write(f':DATA:SOUR CH{number}')
        n = self._count(i.query(':HOR:RECO?'), maximum)
        i.write(':DATA:ENC RIBINARY')
        i.write(':DATA:WIDTH 2')
        i.write(':DATA:START 1')
        i.write(f':DATA:STOP {n}')
        if i.query(':WFMO:PT_FMT?').strip().upper() != 'Y' or i.query(':WFMO:XUNIT?').strip().strip('"').lower() != 's':
            raise ValueError('Tektronix waveform must be a time-domain Y record.')
        if (int(i.query(':WFMO:BYT_NR?')) != 2 or i.query(':WFMO:BN_FMT?').strip().upper() != 'RI'
                or i.query(':WFMO:BYT_OR?').strip().upper() != 'MSB'):
            raise ValueError('Tektronix did not accept signed 16-bit big-endian encoding.')
        if self._count(i.query(':WFMO:NR_PT?'), maximum) != n:
            raise ValueError('Tektronix transfer does not contain the full acquisition record.')
        p = {key: float(i.query(query)) for key, query in dict(dx=':WFMO:XINCR?', x0=':WFMO:XZERO?',
            xr=':WFMO:PT_OFF?', dy=':WFMO:YMULT?', yr=':WFMO:YOFF?', y0=':WFMO:YZERO?').items()}
        if not np.isfinite(list(p.values())).all() or p['dx'] <= 0 or p['dy'] <= 0:
            raise ValueError('Invalid Tektronix waveform scaling.')
        codes = self._binary(':CURVE?', n, True)
        self._check_errors()
        return (codes - p['yr']) * p['dy'] + p['y0'], (np.arange(n) - p['xr']) * p['dx'] + p['x0'], dict(points=n, **p)
