"""Training result contexts, custom plots, and complete-result attachments."""

from pathlib import Path
import json
import shutil
import tempfile
import pandas as pd
from tkinter import messagebox

from nn_training_studio.results import make_training_prediction_dataframe
from nn_training_studio.ui.results import CustomResultsStudioWindow


class TrainingResultsMixin:
    def clear_training_custom_results(self):
        self.training_feature_sample = None
        self.training_feature_source = ""
        self._custom_results_run_token = object()
        storage = getattr(self, "_training_custom_storage", None)
        if storage is not None:
            storage.cleanup()
        self._training_custom_storage = None

    def _cleanup_training_results_on_destroy(self, event):
        if event.widget is self:
            self.clear_training_custom_results()

    def get_training_results_context(self):
        if self.training_running or self.trained_model is None or not self.metadata:
            raise ValueError("Complete model training before customizing its results.")
        metadata = dict(self.metadata)
        predictions = make_training_prediction_dataframe(
            metadata['task_type'], self.y_test_result, self.y_pred_result,
            self.class_names, self.result_target_names, self.anomaly_scores)
        tables = {"results": predictions}
        history = pd.DataFrame(self.history.history if self.history is not None else {})
        if not history.empty:
            history.insert(0, 'epoch', range(1, len(history) + 1))
            tables['training_history'] = history
        if self.confusion_mat is not None:
            names = list(self.class_names or [])
            tables['confusion_matrix'] = pd.DataFrame(
                self.confusion_mat, index=names, columns=names).reset_index(names='actual_label')
        features = getattr(self, 'training_feature_sample', None)
        if features is not None and not features.empty:
            tables['embedding_features'] = features.copy()
        return {
            'title': 'Training Results — Custom Plots',
            'task_type': metadata['task_type'],
            'model_type': metadata.get('model_type', 'Keras model'),
            'tables': tables, 'metrics': metadata.get('metrics', {}),
            'metadata': metadata, 'class_names': self.class_names or [],
            'source': ('Held-out test results. ' + self.training_feature_source),
        }

    def open_training_results_studio(self):
        try:
            context = self.get_training_results_context()
        except ValueError as exc:
            messagebox.showwarning('Training Results', str(exc), parent=self)
            return
        run_token = self._custom_results_run_token
        window = CustomResultsStudioWindow(
            self, context_provider=lambda: context, settings_owner=self,
            attach_callback=lambda source, recipe: self.attach_training_custom_result(
                source, recipe, run_token))
        window.plot_type_var.set('Loss & Accuracy')
        if 'training_history' in context['tables']:
            window.table_var.set('training_history')
        window.refresh_column_choices()
        window.insert_builtin_template()
        window.notebook.select(window.design_tab)
        window.ai_prompt_text.delete('1.0', 'end')
        window.ai_prompt_text.insert('end',
            'Create a heatmap of feature correlations, or a t-SNE scatter plot of '
            'embedding_features colored by actual_label. Explain the patterns '
            'without treating this exploratory visualization as an accuracy measure.')

    def attach_training_custom_result(self, source, _recipe, run_token):
        if run_token is not self._custom_results_run_token or self.training_running:
            raise ValueError('These plots belong to an earlier run. Open the current training results.')
        if self._training_custom_storage is None:
            self._training_custom_storage = tempfile.TemporaryDirectory(prefix='nn_training_custom_')
        root = Path(self._training_custom_storage.name)
        destination = root / f'custom_result_{len(list(root.iterdir())) + 1:02d}'
        shutil.copytree(source, destination)
        self.log_result('Custom plot attached; it will be included in Save Complete Results.')

    def export_training_custom_results(self, output_directory, created_files):
        root = Path(output_directory)
        storage = getattr(self, '_training_custom_storage', None)
        if storage is not None:
            shutil.copytree(storage.name, root / 'custom_results', dirs_exist_ok=True)
        features = getattr(self, 'training_feature_sample', None)
        if features is not None and not features.empty:
            features.to_csv(root / 'embedding_features.csv', index=False)
            (root / 'embedding_features_readme.txt').write_text(
                self.training_feature_source + '\nAt most 1,000 held-out rows and 128 '
                'feature coordinates. record_number links to held-out predictions; '
                'labels are excluded from t-SNE inputs.\n', encoding='utf-8')
        files = sorted(path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file())
        manifest_path = root / 'results_manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['files'] = [name for name in files if name != 'results_manifest.json']
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return sorted(set(created_files) | set(files))
