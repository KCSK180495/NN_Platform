"""Fixed sensor-zero calibration, in the units returned by the scope."""
import numpy as np

CALIBRATION_FORMULA = 'physical_value = (scope_value - sensor_zero) * gain + offset'


def calibrate_channels(values, channels):
    raw = np.asarray(values, dtype=float)
    if raw.ndim != 2 or raw.shape[1] != len(channels):
        raise ValueError('Calibration channel count does not match waveform data.')
    zero = np.array([getattr(c, 'sensor_zero', 0.0) for c in channels], dtype=float)
    gain = np.array([c.gain for c in channels], dtype=float)
    offset = np.array([c.offset for c in channels], dtype=float)
    if not np.isfinite(np.concatenate((zero, gain, offset))).all() or (gain == 0).any():
        raise ValueError('Sensor zero, gain and offset must be finite; gain cannot be zero.')
    with np.errstate(over='ignore', invalid='ignore'):
        calibrated = (raw - zero) * gain + offset
    if not np.isfinite(calibrated).all():
        raise ValueError('Calibrated samples are not finite; check sensor zero/gain/offset.')
    return calibrated
