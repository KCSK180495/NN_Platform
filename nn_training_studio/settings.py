"""Settings for NN Training Studio."""

from pathlib import Path
import os
from nn_training_studio.constants import (
    APP_SETTINGS_DIRECTORY,
    APP_SETTINGS_FILENAME,
)


def _application_settings_path():
    """Return a per-user path for non-secret desktop preferences."""
    if os.name == "nt":
        base_directory = (
            os.getenv("LOCALAPPDATA")
            or os.getenv("APPDATA")
            or str(Path.home())
        )
    else:
        base_directory = (
            os.getenv("XDG_CONFIG_HOME")
            or str(Path.home() / ".config")
        )
    return (
        Path(base_directory)
        / APP_SETTINGS_DIRECTORY
        / APP_SETTINGS_FILENAME
    )
