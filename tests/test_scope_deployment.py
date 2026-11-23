"""Headless checks for Scope routing, package integrity and capture provenance."""

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest
from zipfile import ZipFile

from nn_training_studio.constants import SCOPE_DEPLOY_TARGET
from nn_training_studio.deployment import deployment_recommended_targets, inspect_saved_model_descriptor
from nn_training_studio.scope_runtime.dataset import load_capture_dataset
from nn_training_studio.scope_runtime.export import export_session, load_session
from nn_training_studio.scope_runtime.package_io import inspect_package


class ScopeDeploymentTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.recipe = json.dumps(dict(
            schema="nn_scope.pipeline/1", name="Raw signal", inputs={"signal": "V"},
            analysis_hz=1000, window=8, stride=4, history_samples=1, steps=[],
            outputs=[dict(signal="signal", feature="signal")],
        )).encode()
        self.digest = sha256(sha256(self.recipe).digest()).hexdigest()
        self.metadata = dict(
            feature_cols=["signal"], class_names=["Low", "High"], sampling_frequency_hz=1000,
            window_size=8, stride=4, scaler_type="None", pipeline_sha256=self.digest,
        )

    def model_package(self, metadata, *, embedded=True):
        path = self.root / "classifier.zip"
        with ZipFile(path, "w") as archive:
            archive.writestr("model.keras", b"inspection must not deserialize this")
            archive.writestr("metadata.json", json.dumps(metadata))
            if embedded:
                archive.writestr("pipeline.json", self.recipe)
        return path

    def test_scope_archive_routes_to_scope_and_requires_the_bound_recipe(self):
        package = self.model_package(self.metadata)
        descriptor = inspect_saved_model_descriptor(package)
        self.assertEqual(descriptor["model_type"], "scope_package")
        self.assertEqual(deployment_recommended_targets(descriptor), [SCOPE_DEPLOY_TARGET])
        self.assertEqual(descriptor["details"]["feature_columns"], ["signal"])
        other = self.root / "profile.json"
        other.write_bytes(self.recipe + b"\n")
        with self.assertRaisesRegex(ValueError, "bound to this model"):
            inspect_package(package, other)

    def test_export_portable_external_profile(self):
        package = self.model_package(self.metadata, embedded=False)
        profile = self.root / "profile.json"
        profile.write_bytes(self.recipe)
        destination = self.root / "deployment.zip"
        task = dict(model=str(package), profile=str(profile), profile_digest=self.digest)
        export_session(destination, {"tasks": [task]})
        exported = self.root / "extracted"
        with ZipFile(destination) as archive:
            self.assertIn("app.py", archive.namelist())
            archive.extractall(exported)
        saved, = load_session(exported / "scope_session.json")["tasks"]
        report = inspect_package(saved["model"], saved["profile"])
        self.assertEqual(report["profile"].digest, self.digest)
        self.assertEqual(report["inputs"], {"signal": "V"})

    def test_capture_import_keeps_source_boundaries_and_excludes_predictions(self):
        files = []
        for identity in ("first", "second"):
            folder = self.root / identity
            folder.mkdir()
            path = folder / "native_samples.csv"
            path.write_text("Time_s,CH1_scope,CH1_calibrated_signal\n0,0,0\n.001,1,2\n", encoding="utf-8")
            (folder / "metadata.json").write_text(json.dumps({
                "capture_id": identity, "native_sample_rate_hz": 1000,
                "channels": [{"number": 1, "name": "signal", "unit": "A", "gain": 2}],
            }), encoding="utf-8")
            (folder / "predictions.csv").write_text("prediction\nUnverified fault\n", encoding="utf-8")
            files.append(path)
        dataset = load_capture_dataset(files)
        self.assertEqual(dataset.features, ["signal"])
        self.assertEqual(list(dataset.frame["annotation_segment_id"]), ["first", "first", "second", "second"])
        self.assertEqual(list(dataset.frame["signal"]), [0, 2, 0, 2])
        self.assertNotIn("prediction", dataset.frame)
        with self.assertRaisesRegex(ValueError, "same source capture"):
            load_capture_dataset([files[0], files[0]])
