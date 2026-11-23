"""Check the processing and deployment contracts on independent recordings."""

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from zipfile import ZipFile

import numpy as np
import pandas as pd

from nn_training_studio.constants import AI_PROVIDER_DEEPSEEK
from nn_training_studio.deployment import inspect_saved_model_descriptor
from nn_training_studio.pipeline_ai import draft_recipe
from nn_training_studio.pipeline_package import export_pipeline
from nn_training_studio.preprocessing import split_pipeline_records
from nn_training_studio.processing_pipeline import RECORD, SAMPLE, SCHEMA, Pipeline


class ProcessingPipelineTests(unittest.TestCase):
    def recipe(self, steps, outputs):
        return dict(schema=SCHEMA, name="Test features", sample_rate_hz=1000,
                    inputs={"current": "A"}, steps=steps, outputs=outputs)

    def test_fit_record_boundaries_and_standalone_round_trip(self):
        recipe = self.recipe([
            dict(op="difference", inputs=["current"], outputs=["change"]),
            dict(op="standard", inputs=["change"], outputs=["standardized"]),
        ], ["change", "standardized"])
        training = pd.DataFrame({"current": [1., 3., 5.]})
        new = pd.DataFrame({"current": [8., 10., 25., 29.], "record": ["01", "01", "02", "02"]})
        pipeline = Pipeline.build(recipe)
        with self.assertRaisesRegex(ValueError, "Fit step"):
            pipeline.transform(new, "record")
        fitted = pipeline.fit(training)
        expected = fitted.transform(new, "record")
        np.testing.assert_array_equal(expected["change"], [0., 2., 0., 4.])
        np.testing.assert_array_equal(expected[SAMPLE], [0, 1, 0, 1])
        np.testing.assert_allclose(expected["standardized"], (expected["change"] - 4 / 3) / np.std([0., 2., 2.]))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / "pipeline.zip"
            export_pipeline(package, fitted)
            self.assertEqual(inspect_saved_model_descriptor(package)["model_type"], "pipeline_package")
            with ZipFile(package) as archive:
                archive.extractall(root)
            new.to_csv(root / "input.csv", index=False)
            subprocess.run(
                [sys.executable, "run_pipeline.py", "input.csv", "features.csv", "--record-column", "record"],
                cwd=root, check=True, capture_output=True, text=True,
            )
            actual = pd.read_csv(root / "features.csv", dtype={RECORD: str})
            pd.testing.assert_frame_equal(expected, actual)

    def test_custom_filter_requires_trust_and_starts_fresh_per_record(self):
        recipe = self.recipe([
            dict(op="python", inputs=["current"], outputs=["adjusted"],
                 params={"file": "step.py", "entry": "custom_filter", "interface": "studio_filter"}),
        ], ["adjusted"])
        code = b"""
calls = 0
def custom_filter(df, selected_columns):
    global calls
    calls += 1
    df[selected_columns] += calls
    return df
"""
        pipeline = Pipeline.build(recipe, {"step.py": code})
        frame = pd.DataFrame({"current": [0., 1.], "record": ["one", "two"]})
        with self.assertRaisesRegex(ValueError, "trust"):
            pipeline.transform(frame, "record")
        np.testing.assert_array_equal(pipeline.transform(frame, "record", trust_code=True)["adjusted"], [1., 2.])

    def test_windows_restart_at_each_record_and_hold_out_groups(self):
        groups = np.repeat(["a", "b", "c", "d", "e", "f"], 7)
        X = np.arange(len(groups))[:, None]
        labels = np.repeat([0, 1, 0, 1, 0, 1], 7)
        windows, classes, train, validation, test = split_pipeline_records(
            X, labels, groups, window=4, stride=3, test_size=.2, validation_split=.2,
        )
        partitions = [set(windows[index, 0, 0] // 7) for index in (train, validation, test)]
        self.assertEqual(len(set.union(*partitions)), sum(map(len, partitions)))
        self.assertEqual(len(windows), 12)
        self.assertEqual(set(windows[:, 0, 0] % 7), {0, 3})
        self.assertEqual(set(classes[train]), {0, 1})

    def test_provider_response_is_validated_before_becoming_a_recipe(self):
        current = self.recipe([], ["current"])
        def transport(url, payload, headers, timeout):
            self.assertEqual(payload["response_format"], {"type": "json_object"})
            return {"choices": [{"message": {"content": json.dumps(current)}}]}
        provider = SimpleNamespace(api_key="mock-key", model="mock-model", base_url="https://example.com",
                                   timeout_seconds=10, transport=transport)
        draft_recipe(provider, AI_PROVIDER_DEEPSEEK, "Preserve current.", current["inputs"], 1000)
        current["steps"].append(dict(op="python", inputs=["current"], outputs=["transformed"]))
        with self.assertRaisesRegex(ValueError, "manual configuration"):
            draft_recipe(provider, AI_PROVIDER_DEEPSEEK, "Preserve current.", current["inputs"], 1000)
