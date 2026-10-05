"""Behavior tests for data processing without a display or TensorFlow."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from nn_training_studio.annotations import (
    validate_box_annotation,
    validate_signal_annotation,
)
from nn_training_studio.branding import _application_resource_directories
from nn_training_studio.constants import TASK_FORECASTING, TASK_REGRESSION
from nn_training_studio.filters import apply_safe_filter_spec
from nn_training_studio.image_data import split_image_records
from nn_training_studio.preprocessing import (
    create_supervised_windows,
    handle_missing_values_multi,
)
from nn_training_studio.templates import make_safe_filter_template


class DataProcessingTests(unittest.TestCase):
    def test_filter_template_preserves_targets_and_source(self):
        frame = pd.DataFrame({"signal": [1., 8., 2., 4., 3.], "label": [0, 1, 0, 1, 0]})
        original = frame.copy(deep=True)
        spec = make_safe_filter_template(["signal"])
        result = apply_safe_filter_spec(frame, spec, allowed_columns=["signal"])
        pd.testing.assert_frame_equal(frame, original)
        pd.testing.assert_series_equal(result["label"], frame["label"])
        expected = frame["signal"].rolling(5, min_periods=1, center=True).mean()
        pd.testing.assert_series_equal(result["signal"], expected)

    def test_filter_rejects_target_column(self):
        frame = pd.DataFrame({"signal": [1., 2.], "label": [0, 1]})
        with self.assertRaisesRegex(ValueError, "not permitted"):
            apply_safe_filter_spec(frame, make_safe_filter_template(["label"]),
                                   allowed_columns=["signal"])

    def test_missing_targets_are_dropped_instead_of_imputed(self):
        frame = pd.DataFrame({"x": [1., np.nan, 3.], "y": [4., 5., np.nan]})
        result = handle_missing_values_multi(frame, ["x"], ["y"], "Fill mean")
        self.assertEqual(result["x"].tolist(), [1., 2.])
        self.assertEqual(result["y"].tolist(), [4., 5.])
        self.assertTrue(pd.isna(frame.loc[1, "x"]))

    def test_forecast_targets_follow_the_input_window(self):
        values = np.arange(8).reshape(-1, 1)
        x, y = create_supervised_windows(values, values, 3, 1, TASK_FORECASTING, 2)
        np.testing.assert_array_equal(x[0].ravel(), [0, 1, 2])
        np.testing.assert_array_equal(y[0], [3, 4])
        self.assertEqual(x.shape, (4, 3, 1))

    def test_windows_do_not_cross_group_boundaries(self):
        values = np.arange(6).reshape(-1, 1)
        x, y = create_supervised_windows(values, values, 3, 1, TASK_REGRESSION,
                                         group_ids=["a"] * 3 + ["b"] * 3)
        np.testing.assert_array_equal(x[:, :, 0], [[0, 1, 2], [3, 4, 5]])
        np.testing.assert_array_equal(y[:, 0], [2, 5])

    def test_image_splits_are_disjoint_and_reproducible(self):
        records = pd.DataFrame([
            {"file_path": f"{label}/{i}.png", "class_index": label, "class_name": str(label)}
            for label in range(2) for i in range(12)
        ])
        splits = split_image_records(records, .2, .2)
        again = split_image_records(records, .2, .2)
        paths = [set(part["file_path"]) for part in splits]
        self.assertEqual(set.union(*paths), set(records["file_path"]))
        for i in range(3):
            self.assertEqual(set(splits[i]["class_index"]), {0, 1})
            pd.testing.assert_frame_equal(splits[i], again[i])
            for j in range(i):
                self.assertFalse(paths[i] & paths[j])


class AnnotationAndResourceTests(unittest.TestCase):
    def test_signal_annotation_normalizes_reversed_interval(self):
        result = validate_signal_annotation(
            {"kind": "interval", "label": "  Motor   fault ", "start": 5, "end": 2}, 10)
        self.assertEqual((result["start"], result["end"]), (2, 5))
        self.assertEqual(result["label"], "Motor fault")

    def test_out_of_range_signal_annotation_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_signal_annotation({"kind": "event", "label": "fault", "start": 10}, 10)

    def test_box_cannot_extend_outside_image(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_box_annotation({"label": "part", "image_path": "part.png",
                                     "x": .9, "y": .2, "w": .2, "h": .2})

    def test_asset_lookup_keeps_project_root_when_cwd_differs(self):
        with tempfile.TemporaryDirectory() as other:
            with patch("pathlib.Path.cwd", return_value=Path(other)):
                directories = _application_resource_directories()
        root = Path(__file__).resolve().parents[1]
        self.assertIn(root, directories)
        self.assertIn(root / "assets", directories)


if __name__ == "__main__":
    unittest.main()
