"""Explicit driver registration, VISA discovery and read-only identity matching.

To extend support, implement ScopeDriver and register a model-family matcher.
An unknown or ambiguously matched instrument never receives setup/acquire commands.
"""
from dataclasses import dataclass
from .rigol_dho814 import DHO814, resource_manager
from .scpi_scope_drivers import Keysight2000X, Tektronix4000, normalized_identity, keysight_match, tektronix_match


@dataclass(frozen=True)
class DriverRegistration:
    driver_id: str
    display_name: str
    matches: object
    factory: object


_drivers = []


def register_driver(registration):
    if not isinstance(registration, DriverRegistration) or not registration.driver_id or not callable(registration.matches) or not callable(registration.factory):
        raise ValueError('A driver needs an ID, identity matcher and factory.')
    if any(driver.driver_id == registration.driver_id for driver in _drivers):
        raise ValueError('Duplicate scope driver ID: ' + registration.driver_id)
    _drivers.append(registration)


def registered_drivers():
    return tuple(_drivers)


def identify_driver(identity, requested='auto'):
    normalized_identity(identity)
    candidates = [driver for driver in _drivers if driver.matches(identity)]
    if requested != 'auto':
        candidates = [driver for driver in candidates if driver.driver_id == requested]
    if len(candidates) != 1:
        vendor, model = normalized_identity(identity)
        reason = 'No matching driver' if not candidates else 'Ambiguous driver identity match'
        raise ValueError(f'{reason} for {vendor} {model}. Add/verify its model-family driver; no capture commands were sent.')
    return candidates[0]


def open_scope(resource, backend='@ivi', timeout_ms=5000, manager=None, driver_id='auto'):
    if not str(resource).upper().startswith(('USB', 'TCPIP')):
        raise ValueError('Choose a USBTMC or TCPIP VISA resource for the installed drivers.')
    rm = manager if manager is not None else resource_manager(backend)
    instrument = None
    try:
        instrument = rm.open_resource(resource)
        instrument.timeout = timeout_ms
        instrument.read_termination = '\n'
        instrument.write_termination = '\n'
        identity = instrument.query('*IDN?').strip()
        registration = identify_driver(identity, driver_id)
        options = {"identity": identity} if registration.driver_id == "rigol-dho814" else {}
        return registration.factory(resource=resource, backend=backend, timeout_ms=timeout_ms,
                                    manager=rm, instrument=instrument, **options)
    except Exception:
        try:
            if instrument is not None:
                instrument.close()
        except Exception:
            pass
        try:
            rm.close()
        except Exception:
            pass
        raise


register_driver(DriverRegistration('rigol-dho814', 'RIGOL DHO814 (existing)',
    lambda identity: 'RIGOL' in normalized_identity(identity)[0] and normalized_identity(identity)[1] == 'DHO814', DHO814))
register_driver(DriverRegistration('tektronix-4000', 'Tektronix MSO/DPO4000B, MDO4000 (experimental)', tektronix_match, Tektronix4000))
register_driver(DriverRegistration('keysight-infiniivision-2000x', 'Keysight/Agilent InfiniiVision 2000 X (experimental)', keysight_match, Keysight2000X))
