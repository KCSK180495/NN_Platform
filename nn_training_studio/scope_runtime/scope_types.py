"""Common oscilloscope contract. No manufacturer commands or UI dependencies."""
from dataclasses import dataclass
from typing import Protocol
import numpy as np


class AcquisitionCancelled(Exception):
    pass


class AcquisitionTimeout(TimeoutError):
    """Capture expired after the driver verified a stopped, usable connection."""


class CaptureSettingsError(ValueError):
    """Settings can be corrected on the current instrument session."""


class ScopeFailure(str):
    """Worker error text plus connection state, preserving the queue tuple API."""
    def __new__(cls, message, *, connection_lost=False, kind='error'):
        value = super().__new__(cls, message)
        value.connection_lost = connection_lost
        value.kind = kind
        return value


@dataclass(frozen=True)
class Channel:
    number: int
    name: str
    unit: str = 'V'
    gain: float = 1.0
    offset: float = 0.0
    sensor_zero: float = 0.0


@dataclass
class Capture:
    capture_id: str
    time_s: np.ndarray
    values: np.ndarray
    scope_values: np.ndarray
    sample_rate_hz: float
    channels: tuple
    metadata: dict


@dataclass(frozen=True)
class ScopeCapabilities:
    driver_id: str
    vendor: str
    model: str
    analog_channels: tuple
    software_setup: bool = False
    memory_depths: tuple = ()
    streaming: bool = False
    segmented_capture: bool = False
    validation: str = 'Experimental; offline-tested only'
    force_trigger: bool = False


class ScopeDriver(Protocol):
    identity: str
    capabilities: ScopeCapabilities
    def capture(self, channels, cancel, **settings) -> Capture: ...
    def close(self) -> None: ...


def validate_channels(channels, valid_numbers=(1, 2, 3, 4)):
    if not 1 <= len(channels) <= len(valid_numbers):
        raise ValueError(f'Select one to {len(valid_numbers)} available analog channels.')
    if len({c.number for c in channels}) != len(channels):
        raise ValueError('Duplicate oscilloscope channel.')
    names = [c.name for c in channels]
    if len(set(names)) != len(names) or any(not n.strip() for n in names):
        raise ValueError('Channel names must be nonempty and unique.')
    for c in channels:
        if c.number not in valid_numbers:
            raise ValueError(f'CH{c.number} is unavailable on this instrument.')
        if not np.isfinite(c.gain) or not np.isfinite(c.offset) or not np.isfinite(c.sensor_zero) or c.gain == 0:
            raise ValueError('Calibration gain must be finite/nonzero; offset must be finite.')
        if c.name in {'Time_s', 'Capture_ID', 'Sample_Index'}:
            raise ValueError(f'Reserved channel name: {c.name}')
