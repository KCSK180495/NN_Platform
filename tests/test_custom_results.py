"""Training result data, real plot rendering, AI contracts, and export lifecycle."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from nn_training_studio.constants import AI_PROVIDER_DEEPSEEK, TASK_CLASSIFICATION
from nn_training_studio.plotting import (
    build_result_context_schema, execute_custom_plot_code, request_ai_plot_recipe,
)
from nn_training_studio.result_customization import make_feature_sample, compute_tsne, extra_plot_code
from nn_training_studio.ui.results import CustomResultsStudioWindow
from nn_training_studio.ui.wizard_results import TrainingResultsMixin


class Value:
    def __init__(self, value):
        self.value = value
    def get(self, *args):
        return self.value


def training_owner():
    owner = TrainingResultsMixin()
    owner.clear_training_custom_results()
    owner.training_running = False
    owner.trained_model = object()
    owner.metadata = {'task_type': TASK_CLASSIFICATION, 'model_type': 'DNN', 'metrics': {'accuracy': .75}}
    owner.class_names = ['00', '01', 'unused']
    owner.y_test_result = np.array([0, 1, 0, 1])
    owner.y_pred_result = np.array([0, 1, 1, 1])
    owner.result_target_names = []
    owner.anomaly_scores = None
    owner.history = SimpleNamespace(history={'loss': [.8, .4], 'val_loss': [.9, .5], 'accuracy': [.6, .8]})
    owner.confusion_mat = np.array([[1, 1, 0], [0, 2, 0], [0, 0, 0]])
    owner.training_feature_sample = make_feature_sample(np.arange(12).reshape(4, 3), owner.y_test_result,
                                                       owner.y_pred_result, owner.class_names)
    owner.log_result = lambda message: None
    return owner


class CustomResultsTests(unittest.TestCase):
    def test_sample_is_bounded_reproducible_and_aligned(self):
        values = np.arange(1500 * 160).reshape(1500, 160)
        actual = np.arange(1500) % 2
        sample = make_feature_sample(values, actual, actual, ['a', 'b'])
        pd.testing.assert_frame_equal(sample, make_feature_sample(values, actual, actual, ['a', 'b']))
        self.assertEqual(sample.shape, (1000, 131))
        row_ids = sample.record_number.to_numpy() - 1
        np.testing.assert_array_equal(sample.feature_1, values[row_ids, 0])
        self.assertEqual(sample.actual_label.tolist(), [['a', 'b'][x % 2] for x in row_ids])

    def test_tsne_excludes_labels_and_preserves_index(self):
        frame = pd.DataFrame({'feature_1': [0, 1, 2, 3, 4], 'feature_2': [2, 1, 3, 1, 5],
                              'actual_label': ['a', 'b', 'a', 'b', 'a']}, index=[4, 8, 12, 16, 20])
        points = compute_tsne(frame)
        self.assertEqual(points.index.tolist(), frame.index.tolist())
        self.assertTrue(np.isfinite(points.to_numpy()).all())
        with self.assertRaises(ValueError):
            compute_tsne(frame, columns=['feature_1', 'actual_label'])
        with self.assertRaises(ValueError):
            compute_tsne(frame.head(2))
        with self.assertRaises(ValueError):
            compute_tsne(pd.DataFrame({'feature_1': [1, 1, 1]}))

    def test_training_context_uses_completed_metadata_and_copies_features(self):
        owner = training_owner()
        context = owner.get_training_results_context()
        self.assertEqual(set(context['tables']), {'results', 'training_history', 'confusion_matrix', 'embedding_features'})
        self.assertEqual(context['tables']['training_history'].epoch.tolist(), [1, 2])
        context['tables']['embedding_features'].iloc[0, 1] = -999
        self.assertNotEqual(owner.training_feature_sample.iloc[0, 1], -999)
        owner.training_running = True
        with self.assertRaises(ValueError):
            owner.get_training_results_context()

    def test_real_templates_render_png_and_svg_in_child_process(self):
        context = training_owner().get_training_results_context()
        for kind, table in [('Loss & Accuracy', 'training_history'),
                            ('Correlation Heatmap', 'embedding_features'), ('t-SNE', 'embedding_features')]:
            with self.subTest(kind=kind):
                code = extra_plot_code(kind, table, context['tables'][table])
                directory = execute_custom_plot_code(code, context)
                try:
                    self.assertGreater((Path(directory) / 'plot.png').stat().st_size, 1000)
                    self.assertGreater((Path(directory) / 'plot.svg').stat().st_size, 1000)
                finally:
                    shutil.rmtree(directory)

    def test_confusion_matrix_preserves_numeric_looking_labels_and_absent_classes(self):
        context = training_owner().get_training_results_context()
        fake = SimpleNamespace(table_var=Value('results'), x_column_var=Value('actual_label'),
                               y_column_var=Value('predicted_label'), plot_type_var=Value('Confusion Matrix'),
                               context=context, _selected_expression=lambda c: f'df[{c!r}]')
        code = CustomResultsStudioWindow.create_builtin_code(fake)
        # Contract check within the real child process after CSV deserialization.
        code = code.replace('    return fig',
                            "    assert int(matrix.to_numpy().sum()) == 4\n"
                            "    assert list(matrix.columns) == ['00', '01', 'unused']\n    return fig")
        directory = execute_custom_plot_code(code, context)
        shutil.rmtree(directory)

    def test_ai_receives_schema_and_tsne_helper_without_raw_rows(self):
        context = training_owner().get_training_results_context()
        schema = build_result_context_schema(context)
        code = extra_plot_code('t-SNE', 'embedding_features', context['tables']['embedding_features'])
        recipe = {'title': 't-SNE', 'plot_type': 't-SNE', 'explanation': 'An exploratory plot', 'code': code,
                  'table': 'embedding_features', 'required_columns': ['feature_1'], 'assumptions': []}
        response = {'choices': [{'message': {'content': json.dumps(recipe)}}]}
        settings = {'provider': AI_PROVIDER_DEEPSEEK, 'api_key': 'test-only', 'model': 'test-model',
                    'base_url': 'https://api.deepseek.com', 'timeout': 30}
        with patch('nn_training_studio.plotting._post_json_request', return_value=response) as post:
            received = request_ai_plot_recipe(settings, schema, 'Draw a t-SNE plot')
        self.assertEqual(received['code'], code)
        prompt = json.loads(post.call_args.args[1]['messages'][1]['content'])
        self.assertIn('compute_tsne', prompt['contract']['helpers'])
        self.assertEqual(prompt['result_context_schema']['tables']['results']['rows'], 4)
        self.assertNotIn('data', prompt['result_context_schema']['tables']['results'])
        self.assertNotIn('metadata', prompt['result_context_schema'])

    def test_attachments_export_and_reset_rejects_old_run(self):
        owner = training_owner()
        token = owner._custom_results_run_token
        with tempfile.TemporaryDirectory() as source, tempfile.TemporaryDirectory() as output:
            (Path(source) / 'plot.svg').write_text('<svg/>')
            owner.attach_training_custom_result(source, {}, token)
            (Path(output) / 'results_manifest.json').write_text('{"files": []}')
            paths = owner.export_training_custom_results(output, [])
            self.assertIn('custom_results/custom_result_01/plot.svg', paths)
            self.assertIn('embedding_features.csv', paths)
            manifest = json.loads((Path(output) / 'results_manifest.json').read_text())
            self.assertIn('custom_results/custom_result_01/plot.svg', manifest['files'])
            storage = Path(owner._training_custom_storage.name)
            owner.clear_training_custom_results()
            self.assertFalse(storage.exists())
            with self.assertRaises(ValueError):
                owner.attach_training_custom_result(source, {}, token)

    def test_export_uses_rendered_snapshot_after_editor_changes(self):
        context = training_owner().get_training_results_context()
        code = extra_plot_code('Loss & Accuracy', 'training_history', context['tables']['training_history'])
        directory = execute_custom_plot_code(code, context)
        recipe = {'table': 'training_history', 'code': code}
        fake = SimpleNamespace(current_output_directory=directory,
                               completed_plot_snapshot={'context': context, 'recipe': recipe, 'explanation': 'original'},
                               table_var=Value('embedding_features'), code_text=Value('changed code'))
        try:
            with tempfile.TemporaryDirectory() as destination:
                CustomResultsStudioWindow.materialize_custom_result(fake, destination)
                saved = json.loads((Path(destination) / 'plot_recipe.json').read_text())
                self.assertEqual(saved, recipe)
                self.assertEqual((Path(destination) / 'plot_code.py').read_text(), code)
                self.assertTrue((Path(destination) / 'context.json').exists())
                self.assertEqual(len(list(Path(destination).glob('table_*.csv'))), 4)
        finally:
            shutil.rmtree(directory)


if __name__ == '__main__':
    unittest.main()
