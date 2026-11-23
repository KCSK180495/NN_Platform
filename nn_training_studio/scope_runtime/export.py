"""Export the same NN Scope workspace and runtime used inside the platform."""

from importlib import metadata
from hashlib import sha256 as hash_bytes
from io import BytesIO
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from zipfile import ZIP_DEFLATED, ZipFile

from .package_io import sha256
from .pipeline_profile import Profile


SCHEMA = "nn_platform.scope_session/1"
REQUIREMENTS = """numpy>=1.26,<3
pandas>=2.1,<3
scipy>=1.11,<2
scikit-learn>=1.4,<2
joblib>=1.3,<2
tensorflow>=2.16,<3
keras>=3,<4
matplotlib>=3.8,<4
"""
LAUNCHER = '''import tkinter as tk
from pathlib import Path
from nn_training_studio.ui.deployment_scope import ScopeDeploymentWindow

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    window = ScopeDeploymentWindow(root, session_path=Path(__file__).with_name("scope_session.json"))
    window.bind("<Destroy>", lambda event: root.quit() if event.widget is window else None, add="+")
    root.mainloop()
    root.destroy()
'''


def load_session(path):
    path = Path(path)
    session = json.loads(path.read_text(encoding="utf-8"))
    if session.get("schema") != SCHEMA:
        raise ValueError("Unsupported NN Scope session format.")
    for task in session["tasks"]:
        for key in ("model", "profile"):
            if task.get(key):
                artifact = (path.parent / task[key]).resolve()
                if not artifact.is_relative_to(path.parent.resolve()):
                    raise ValueError("Session artifacts must be inside the deployment folder.")
                task[key] = str(artifact)
    return session


def export_session(path, configuration):
    import platform

    path = Path(path)
    runtime = Path(__file__).resolve().parent
    ui = runtime.parent / "ui"
    tracks = ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "tensorflow", "keras",
              "matplotlib", "pyvisa", "pyvisa-py", "protobuf")
    installed = {d.metadata["Name"].lower(): d.version for d in metadata.distributions()
                 if d.metadata["Name"].lower() in tracks}
    package_files = [*runtime.glob("*.py"), ui / "deployment_scope.py", ui / "scrollable.py"]
    session = dict(configuration, schema=SCHEMA, tasks=[])
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            archive.writestr("app.py", LAUNCHER)
            archive.writestr("requirements.txt", REQUIREMENTS)
            archive.writestr("requirements-instrument.txt", "-r requirements.txt\nPyVISA>=1.14,<2\nPyVISA-py>=0.7,<1\n")
            archive.writestr("environment.json", json.dumps(
                dict(python=platform.python_version(), platform=platform.platform(), packages=installed), indent=2))
            archive.writestr("README.txt",
                "Extract this deployment and run python app.py with Tk installed.\n"
                "Install requirements.txt for replay or requirements-instrument.txt for instrument use.\n"
                "environment.json records the exporter environment. Validate in the target environment.\n"
                "Use trusted model ZIPs. Legacy fitted scalers use serialized Python objects.\n"
                "Choose a saved capture with its metadata.json for replay, or connect the scope explicitly.\n"
                "Live view repeats waveform captures. Record saves them separately until Stop.\n"
                "Healthy startup confirmation is made separately for each record.\n")
            for source in package_files:
                archive.write(source, source.relative_to(runtime.parent.parent).as_posix())
            assets = {}
            for task in configuration["tasks"]:
                saved = dict(task)
                for key in ("model", "profile"):
                    if not task.get(key):
                        continue
                    asset = Path(task[key]).resolve()
                    checksum = f"{key}_sha256"
                    digest = sha256(asset)
                    if task.get(checksum) and task[checksum] != digest:
                        raise ValueError(f"{asset.name} differs from the configured checksum.")
                    profile = Profile.read(asset) if key == "profile" else None
                    if profile and task.get("profile_digest") and task["profile_digest"] != profile.digest:
                        raise ValueError("The profile or its coefficients changed since configuration.")
                    identity = (asset, key)
                    if identity not in assets:
                        member = f"assets/{len(assets) + 1}_{asset.name}"
                        if profile:
                            # A plain JSON profile can reference a neighboring parameters.npz.
                            # Bundle the exact validated recipe bytes and coefficients together.
                            member = str(Path(member).with_suffix(".zip").as_posix())
                            payload = BytesIO()
                            with ZipFile(payload, "w", ZIP_DEFLATED) as profile_zip:
                                for filename, data in profile.files.items():
                                    profile_zip.writestr(filename, data)
                            data = payload.getvalue()
                            archive.writestr(member, data)
                            digest = hash_bytes(data).hexdigest()
                        else:
                            archive.write(asset, member)
                            if digest != sha256(asset):
                                raise ValueError(f"{asset.name} changed during export. Retry with a stable file.")
                        assets[identity] = member, digest
                    saved[key], saved[checksum] = assets[identity]
                session["tasks"].append(saved)
            archive.writestr("scope_session.json", json.dumps(session, indent=2))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
