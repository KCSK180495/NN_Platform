"""Branding for NN Training Studio."""

from pathlib import Path
import platform
import sys
import tkinter as tk
from nn_training_studio.constants import (
    BRAND_ASSET_FILENAMES,
    WINDOWS_APP_USER_MODEL_ID,
)


def _application_resource_directories():
    """Return ordered directories that may contain packaged brand assets."""
    candidates = []
    packaged_directory = getattr(sys, "_MEIPASS", None)
    if packaged_directory:
        candidates.append(Path(packaged_directory))
    try:
        candidates.append(Path(__file__).resolve().parent.parent)
    except NameError:
        pass
    candidates.append(Path.cwd())

    directories = []
    seen = set()
    for base_directory in candidates:
        for directory in (
            base_directory,
            base_directory / "assets",
            base_directory / "upload",
        ):
            try:
                resolved = directory.resolve()
            except Exception:
                resolved = directory
            marker = str(resolved)
            if marker not in seen:
                directories.append(resolved)
                seen.add(marker)
    return directories


def find_brand_asset(asset_kind):
    """Locate one canonical or legacy logo asset."""
    filenames = BRAND_ASSET_FILENAMES.get(asset_kind, ())
    for directory in _application_resource_directories():
        for filename in filenames:
            candidate = directory / filename
            if candidate.is_file():
                return candidate
    return None


def configure_windows_application_identity():
    """Give packaged and source runs one stable Windows taskbar identity."""
    if platform.system() != "Windows":
        return False
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            WINDOWS_APP_USER_MODEL_ID
        )
        return True
    except Exception:
        return False


def apply_window_branding(window, make_default=False):
    """Apply the Studio icon safely and retain Tk image references."""
    details = {
        "icon_path": None,
        "compact_path": None,
        "iconphoto_applied": False,
        "iconbitmap_applied": False,
    }

    compact_path = find_brand_asset("compact") or find_brand_asset("logo")
    if compact_path is not None:
        details["compact_path"] = str(compact_path)
        try:
            icon_photo = tk.PhotoImage(file=str(compact_path))
            window.iconphoto(bool(make_default), icon_photo)
            # Tk images disappear if Python releases the final reference.
            references = list(
                getattr(window, "_nn_studio_brand_image_refs", [])
            )
            references.append(icon_photo)
            window._nn_studio_brand_image_refs = references
            window._nn_studio_icon_photo = icon_photo
            details["iconphoto_applied"] = True
        except (tk.TclError, OSError):
            pass

    icon_path = find_brand_asset("icon")
    if icon_path is not None:
        details["icon_path"] = str(icon_path)
        if platform.system() == "Windows":
            try:
                window.iconbitmap(str(icon_path))
                details["iconbitmap_applied"] = True
            except tk.TclError:
                pass

    window._nn_studio_branding = details
    return details
