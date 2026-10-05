"""Ui / wizard training workflow for NN Training Studio."""

from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from tkinter.scrolledtext import ScrolledText
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from datetime import datetime
from sklearn.metrics import f1_score
from tkinter import filedialog
import joblib
import json
from tkinter import messagebox
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import queue
import shutil
import tempfile
import tensorflow as tf
import threading
import tkinter as tk
import traceback
from tkinter import ttk
import zipfile
from nn_training_studio.checkpoints import (
    prepare_training_checkpoint_paths,
)
from nn_training_studio.constants import (
    APP_VERSION,
    CUSTOM_FILTER_MODE_EXPERT,
    DATA_MODE_IMAGE,
    DATA_MODE_TABULAR,
    DEVICE_AUTO,
    IMAGE_MODEL_EFFICIENTNET,
    IMAGE_MODEL_MOBILENET,
    MODEL_TYPE_SAFE_AI_LEGACY,
    MODEL_TYPE_SAFE_CUSTOM,
    SUPPORTED_IMAGE_EXTENSIONS,
    TASK_AUTOENCODER,
    TASK_CLASSIFICATION,
    TASK_FORECASTING,
    TASK_MULTI_OUTPUT,
    TASK_REGRESSION,
)
from nn_training_studio.devices import (
    DeviceManager,
    call_on_tensorflow_device,
)
from nn_training_studio.image_data import (
    make_image_tf_dataset,
    split_image_records,
)
from nn_training_studio.models import (
    build_image_classification_model,
    build_optimizer,
    build_selected_model,
    prediction_to_class,
    prepare_training_targets,
)
from nn_training_studio.preprocessing import (
    convert_features_to_numeric,
    create_supervised_windows,
    handle_missing_values_multi,
    inverse_transform_targets,
    prepare_scaled_training_data,
    regression_metrics,
    safe_train_test_split,
    summarize_training_preflight,
)
from nn_training_studio.results import (
    _result_json_default,
    make_training_prediction_dataframe,
    write_complete_result_files,
    zip_result_directory,
)
from nn_training_studio.training import (
    PersistentBestModelCheckpoint,
    TrainingProgressCallback,
    build_early_stopping_callbacks,
    load_best_checkpoint_model,
    validate_early_stopping_settings,
)


from nn_training_studio.result_customization import make_feature_sample


class TrainingWorkflowMixin:
    """TrainingWorkflow behavior for the main application."""

    def build_step_5(self):
        device_frame = ttk.LabelFrame(
            self.content_frame,
            text="Training Device",
        )
        device_frame.pack(fill=tk.X, pady=(5, 8))
        ttk.Label(device_frame, text="Use:").pack(
            side=tk.LEFT,
            padx=(8, 4),
            pady=7,
        )
        self.keras_training_device_combo = ttk.Combobox(
            device_frame,
            textvariable=self.training_device_var,
            values=DeviceManager.tensorflow_options(),
            state="readonly",
            width=38,
        )
        self.keras_training_device_combo.pack(
            side=tk.LEFT,
            padx=4,
            pady=7,
        )
        self.keras_training_device_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.update_training_device_status(),
        )
        ttk.Button(
            device_frame,
            text="Refresh Devices",
            command=self.refresh_training_devices,
        ).pack(side=tk.LEFT, padx=5, pady=7)
        ttk.Label(
            device_frame,
            textvariable=self.training_device_status_var,
            foreground="#245a85",
        ).pack(side=tk.LEFT, padx=10, pady=7)
        self.refresh_training_devices()

        button_frame = ttk.Frame(self.content_frame)
        button_frame.pack(fill=tk.X, pady=5)

        self.check_training_button = ttk.Button(
            button_frame,
            text="Check Training Setup",
            command=self.check_training_setup,
            state=tk.DISABLED if self.training_running else tk.NORMAL,
        )
        self.check_training_button.pack(side=tk.LEFT, padx=5)

        self.train_button = ttk.Button(
            button_frame,
            text="Start Training",
            command=self.start_training,
            state=tk.DISABLED if self.training_running else tk.NORMAL
        )
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.plot_button = ttk.Button(
            button_frame,
            text="Show Result Graphs",
            command=self.show_plots,
            state=tk.NORMAL if self.history is not None else tk.DISABLED
        )
        self.plot_button.pack(side=tk.LEFT, padx=5)

        self.save_button = ttk.Button(
            button_frame,
            text="Save Best Model Package",
            command=self.save_model_package,
            state=tk.NORMAL if self.trained_model is not None else tk.DISABLED
        )
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.save_results_button = ttk.Button(
            button_frame,
            text="Save Complete Results",
            command=self.save_training_results,
            state=tk.NORMAL if self.trained_model is not None else tk.DISABLED,
        )
        self.save_results_button.pack(side=tk.LEFT, padx=5)

        self.deploy_trained_button = ttk.Button(
            button_frame,
            text="Deploy & Integrate",
            command=self.open_last_saved_deployment,
            state=(
                tk.NORMAL
                if self.last_saved_model_package_path
                else tk.DISABLED
            ),
        )
        self.deploy_trained_button.pack(side=tk.LEFT, padx=5)

        custom_row = ttk.Frame(self.content_frame)
        custom_row.pack(fill=tk.X, pady=4)
        self.custom_training_results_button = ttk.Button(
            custom_row, text="Customize Results / AI Plots",
            command=self.open_training_results_studio,
            state=tk.NORMAL if self.trained_model is not None else tk.DISABLED)
        self.custom_training_results_button.pack(side=tk.LEFT, padx=5)
        ttk.Label(custom_row, text="Heatmaps, t-SNE, training curves, and editable AI plots").pack(side=tk.LEFT, padx=6)

        preflight_frame = ttk.Frame(self.content_frame)
        preflight_frame.pack(fill=tk.X, padx=5, pady=(2, 3))
        ttk.Label(
            preflight_frame,
            textvariable=self.training_preflight_status_var,
            foreground="#245a85",
            wraplength=1120,
        ).pack(anchor="w")

        progress_frame = ttk.LabelFrame(
            self.content_frame,
            text="Training Progress"
        )
        progress_frame.pack(fill=tk.X, pady=(5, 8))

        self.training_progress = ttk.Progressbar(
            progress_frame,
            orient=tk.HORIZONTAL,
            mode="determinate",
            maximum=max(1, int(self.epochs_var.get()))
        )
        self.training_progress.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=8,
            pady=8
        )

        self.training_status_label = ttk.Label(
            progress_frame,
            text="Ready"
        )
        self.training_status_label.pack(side=tk.RIGHT, padx=8)

        result_pane = ttk.PanedWindow(
            self.content_frame,
            orient=tk.VERTICAL
        )
        result_pane.pack(fill=tk.BOTH, expand=True, pady=5)

        epoch_frame = ttk.LabelFrame(
            result_pane,
            text="Live Epoch Metrics"
        )
        log_frame = ttk.LabelFrame(
            result_pane,
            text="Training Log and Evaluation"
        )
        result_pane.add(epoch_frame, weight=2)
        result_pane.add(log_frame, weight=3)

        columns = (
            "epoch",
            "loss",
            "metric",
            "val_loss",
            "val_metric"
        )
        self.epoch_tree = ttk.Treeview(
            epoch_frame,
            columns=columns,
            show="headings",
            height=9
        )

        metric_heading = (
            "Accuracy"
            if self.task_type_var.get() == TASK_CLASSIFICATION
            else "MAE"
        )
        headings = {
            "epoch": "Epoch",
            "loss": "Loss",
            "metric": metric_heading,
            "val_loss": "Validation Loss",
            "val_metric": "Validation " + metric_heading
        }
        widths = {
            "epoch": 80,
            "loss": 150,
            "metric": 150,
            "val_loss": 160,
            "val_metric": 170
        }

        for column in columns:
            self.epoch_tree.heading(column, text=headings[column])
            self.epoch_tree.column(
                column,
                width=widths[column],
                anchor=tk.CENTER
            )

        epoch_scroll = ttk.Scrollbar(
            epoch_frame,
            orient=tk.VERTICAL,
            command=self.epoch_tree.yview
        )
        self.epoch_tree.configure(yscrollcommand=epoch_scroll.set)
        self.epoch_tree.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=(5, 0),
            pady=5
        )
        epoch_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)

        self.result_text = ScrolledText(log_frame, wrap=tk.WORD)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.result_text.insert(tk.END, "Ready to train.\n\n")
        self.result_text.insert(tk.END, "Training summary:\n")
        self.result_text.insert(
            tk.END,
            f"Data mode: {self.data_mode_var.get()}\n"
        )
        self.result_text.insert(tk.END, f"Task: {self.task_type_var.get()}\n")
        self.result_text.insert(tk.END, f"Input columns: {self.feature_cols}\n")
        self.result_text.insert(tk.END, f"Target columns: {self.target_cols}\n")
        self.result_text.insert(tk.END, f"Built-in filter: {self.filter_method_var.get()}\n")
        self.result_text.insert(
            tk.END,
            f"Built-in filtered columns: {getattr(self, 'selected_filter_cols', [])}\n"
        )
        self.result_text.insert(
            tk.END,
            f"Custom filter enabled: {self.custom_filter_enabled_var.get()}\n"
        )
        self.result_text.insert(
            tk.END,
            f"Custom filter columns: {self.selected_custom_filter_cols}\n"
        )
        self.result_text.insert(
            tk.END,
            f"Preprocess: {self.missing_var.get()}, {self.scaler_var.get()}\n"
        )
        self.result_text.insert(tk.END, f"Model: {self.model_type_var.get()}\n")
        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            self.result_text.insert(
                tk.END,
                f"Image size: {self.image_height_var.get()} × "
                f"{self.image_width_var.get()}\n"
                f"Colour mode: {self.image_color_mode_var.get()}\n"
                f"Classes: {self.image_class_names}\n"
                f"Augmentation: {self.image_augmentation_var.get()}\n"
                f"ImageNet pretrained weights: "
                f"{self.image_pretrained_var.get()}\n"
            )
        if self.model_type_var.get() == "Custom Python Model":
            self.result_text.insert(
                tk.END,
                f"Custom model input: {self.custom_model_input_mode_var.get()}\n"
            )
        self.result_text.insert(tk.END, f"Output units: {self.output_units_var.get()}\n")
        self.result_text.insert(tk.END, f"Output activation: {self.output_activation_var.get()}\n")
        self.result_text.insert(tk.END, f"Loss function: {self.loss_var.get()}\n")
        self.result_text.insert(tk.END, f"Optimizer: {self.optimizer_var.get()}\n")
        self.result_text.insert(tk.END, f"Learning rate: {self.lr_var.get()}\n")
        self.result_text.insert(tk.END, f"Epochs: {self.epochs_var.get()}\n")
        self.result_text.insert(
            tk.END,
            f"Training device: {self.training_device_var.get()}\n",
        )
        self.result_text.insert(
            tk.END,
            "Automatic best-model checkpoint: Enabled (monitors val_loss)\n"
        )
        self.result_text.insert(
            tk.END,
            "Interrupted-training recovery: Enabled (saved every epoch)\n"
        )

        if self.history is not None:
            losses = self.history.history.get("loss", [])
            metric_key = (
                "accuracy"
                if self.task_type_var.get() == TASK_CLASSIFICATION
                else "mae"
            )
            metric_values = self.history.history.get(metric_key, [])
            val_losses = self.history.history.get("val_loss", [])
            val_metric_values = self.history.history.get(
                "val_" + metric_key,
                []
            )

            for index, loss in enumerate(losses):
                self.append_epoch_metrics(
                    index + 1,
                    loss,
                    metric_values[index] if index < len(metric_values) else None,
                    val_losses[index] if index < len(val_losses) else None,
                    (
                        val_metric_values[index]
                        if index < len(val_metric_values)
                        else None
                    )
                )


        if self._last_training_log and self.history is not None:
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert("1.0", self._last_training_log)
        if self.is_guided() and self.guided_detail_step is None:
            device_frame.pack_forget()
            for child in button_frame.winfo_children():
                child.pack_forget()
            for index, button in enumerate((self.check_training_button, self.train_button)):
                button_frame.columnconfigure(index, weight=1)
                button.grid(row=0, column=index, sticky="ew", padx=5, pady=4)
            self.train_button.configure(text="Start training")
            self.check_training_button.configure(text="Check my setup")
            result_pane.pack_forget()
            details_button = ttk.Button(self.content_frame, text="Show technical details")
            def toggle_details():
                if result_pane.winfo_manager():
                    result_pane.pack_forget()
                    details_button.configure(text="Show technical details")
                else:
                    result_pane.pack(fill=tk.BOTH, expand=True, pady=5)
                    details_button.configure(text="Hide technical details")
                self._on_content_frame_configure()
            details_button.configure(command=toggle_details)
            details_button.pack(anchor="w", padx=10, pady=8)

    def log_result(self, message):
        if not hasattr(self, "result_text") or not self.result_text.winfo_exists():
            return

        self.result_text.insert(tk.END, str(message) + "\n")
        self.result_text.see(tk.END)

    def refresh_training_devices(self):
        options = DeviceManager.tensorflow_options()
        if hasattr(self, "keras_training_device_combo"):
            self.keras_training_device_combo.config(values=options)
        if self.training_device_var.get() not in options:
            self.training_device_var.set(DEVICE_AUTO)
        self.update_training_device_status()

    def update_training_device_status(self):
        try:
            info = DeviceManager.resolve_tensorflow(
                self.training_device_var.get()
            )
            self.training_device_status_var.set(
                DeviceManager.format_status(info)
            )
        except Exception as exc:
            self.training_device_status_var.set(str(exc))

    def thread_log(self, message):
        self.ui_queue.put(("log", str(message)))

    def thread_epoch_update(
        self,
        epoch,
        loss,
        metric,
        val_loss,
        val_metric
    ):
        self.ui_queue.put((
            "epoch",
            (epoch, loss, metric, val_loss, val_metric)
        ))

    def process_ui_queue(self):
        try:
            while True:
                event_type, payload = self.ui_queue.get_nowait()

                if event_type == "log":
                    self.log_result(payload)
                elif event_type == "epoch":
                    self.append_epoch_metrics(*payload)
                elif event_type == "finished":
                    self.training_finished()
                elif event_type == "failed":
                    self.training_failed(payload)
                elif event_type == "ai_finished":
                    self.ai_recommendation_finished(payload)
                elif event_type == "ai_failed":
                    self.ai_recommendation_failed(payload)
                elif event_type == "ai_generation_finished":
                    self.ai_generation_finished(payload)
                elif event_type == "ai_generation_failed":
                    self.ai_generation_failed(payload)
                elif event_type == "component_ai_generation_finished":
                    try:
                        self.component_ai_generation_finished(payload)
                    except Exception as exc:
                        self.component_ai_generation_failed({
                            "component": payload.get("component", "model"),
                            "error": str(exc),
                        })
                elif event_type == "component_ai_generation_failed":
                    self.component_ai_generation_failed(payload)
                elif event_type == "filter_tool_ai_finished":
                    try:
                        self.filter_tool_ai_generation_finished(payload)
                    except Exception as exc:
                        self.filter_tool_ai_generation_failed(str(exc))
                elif event_type == "filter_tool_ai_failed":
                    self.filter_tool_ai_generation_failed(payload)
                elif event_type == "ai_connection_test_finished":
                    self.ai_connection_test_finished(payload)
                elif event_type == "ai_connection_test_failed":
                    self.ai_connection_test_failed(payload)
                elif event_type == "studio_guide_ai_finished":
                    window = payload.get("window")
                    try:
                        window_exists = bool(
                            window is not None and window.winfo_exists()
                        )
                    except tk.TclError:
                        window_exists = False
                    if window_exists:
                        window.finish_ai_request(
                            payload.get("answer"),
                            None,
                        )
                elif event_type == "studio_guide_ai_failed":
                    window = payload.get("window")
                    try:
                        window_exists = bool(
                            window is not None and window.winfo_exists()
                        )
                    except tk.TclError:
                        window_exists = False
                    if window_exists:
                        window.finish_ai_request(
                            None,
                            payload.get("error", "Unknown guide error."),
                            payload.get("fallback"),
                        )
        except queue.Empty:
            pass

        if self.winfo_exists():
            self.after(100, self.process_ui_queue)

    @staticmethod
    def _format_metric(value):
        if value is None:
            return "—"
        return f"{float(value):.6f}"

    def append_epoch_metrics(
        self,
        epoch,
        loss,
        metric,
        val_loss,
        val_metric
    ):
        if hasattr(self, "epoch_tree") and self.epoch_tree.winfo_exists():
            self.epoch_tree.insert(
                "",
                tk.END,
                values=(
                    epoch,
                    self._format_metric(loss),
                    self._format_metric(metric),
                    self._format_metric(val_loss),
                    self._format_metric(val_metric)
                )
            )
            children = self.epoch_tree.get_children()
            if children:
                self.epoch_tree.see(children[-1])

        if hasattr(self, "training_progress"):
            self.training_progress["value"] = epoch

        if hasattr(self, "training_status_label"):
            total_epochs = max(1, int(self.epochs_var.get()))
            self.training_status_label.config(
                text=f"Epoch {epoch} / {total_epochs}"
            )

    def collect_training_settings(self):
        device_info = DeviceManager.resolve_tensorflow(
            self.training_device_var.get()
        )
        return {
            "early_stopping": self.collect_early_stopping_settings(),
            "data_mode": self.data_mode_var.get(),
            "task_type": self.task_type_var.get(),
            "feature_cols": list(self.feature_cols),
            "label_col": self.label_col,
            "target_cols": list(self.target_cols),
            "missing_method": self.missing_var.get(),
            "scaler_option": self.scaler_var.get(),
            "target_scaler_option": self.target_scaler_var.get(),
            "model_type": self.model_type_var.get(),
            "custom_model_input_mode": self.custom_model_input_mode_var.get(),
            "custom_model_code": self.custom_model_code,
            "safe_model_spec": self.safe_model_spec,
            "custom_model_ai_prompt": (
                self.custom_model_ai_prompt_var.get().strip()
            ),
            "custom_model_ai_report": self.custom_model_ai_report,
            "window_size": int(self.window_size_var.get()),
            "stride": int(self.stride_var.get()),
            "forecast_horizon": int(self.forecast_horizon_var.get()),
            "anomaly_percentile": float(self.anomaly_percentile_var.get()),
            "epochs": int(self.epochs_var.get()),
            "batch_size": int(self.batch_size_var.get()),
            "training_device_selection": self.training_device_var.get(),
            "training_device_info": device_info,
            "validation_split": float(self.validation_split_var.get()) / 100,
            "hidden_activation": self.hidden_activation_var.get(),
            "dropout_rate": float(self.dropout_var.get()),
            "output_mode": self.output_mode_var.get(),
            "output_units": int(self.output_units_var.get()),
            "output_activation": self.output_activation_var.get(),
            "loss_name": self.loss_var.get(),
            "optimizer_name": self.optimizer_var.get(),
            "learning_rate": float(self.lr_var.get()),
            "test_size": float(self.test_size_var.get()) / 100,
            "image_directory": self.image_directory,
            "image_class_names": list(self.image_class_names),
            "image_height": int(self.image_height_var.get()),
            "image_width": int(self.image_width_var.get()),
            "image_color_mode": self.image_color_mode_var.get(),
            "image_augmentation": bool(
                self.image_augmentation_var.get()
            ),
            "image_pretrained_weights": bool(
                self.image_pretrained_var.get()
            ),
            "built_in_filter_method": self.filter_method_var.get(),
            "built_in_filter_columns": list(
                getattr(self, "selected_filter_cols", [])
            ),
            "moving_window": int(self.moving_window_var.get()),
            "ema_span": int(self.ema_span_var.get()),
            "median_window": int(self.median_window_var.get()),
            "kalman_q": float(self.kalman_q_var.get()),
            "kalman_r": float(self.kalman_r_var.get()),
            "custom_filter_enabled": bool(
                self.custom_filter_enabled_var.get()
            ),
            "custom_filter_mode": self.custom_filter_mode_var.get(),
            "custom_filter_columns": list(
                self.selected_custom_filter_cols
            ),
            "custom_filter_code": self.custom_filter_code,
            "safe_filter_spec": self.safe_filter_spec,
            "custom_filter_ai_prompt": (
                self.custom_filter_ai_prompt_var.get().strip()
            ),
            "custom_filter_ai_report": self.custom_filter_ai_report,
            "source_file": self.file_path,
            "dataset_profile": self.dataset_profile,
            "dataset_analysis_report": self.dataset_analysis_report,
            "ai_recommendation": self.ai_recommendation,
            "ai_recommendation_report": self.ai_recommendation_report,
            "ai_provider_summary": self.ai_provider_summary,
            "ai_generation": self.ai_generation,
            "ai_generation_report": self.ai_generation_report,
            "ai_generation_provider_summary": (
                self.ai_generation_provider_summary
            ),
        }

    def check_training_setup(self):
        """Run the same local checks used by Start Training, without fitting."""
        try:
            self._check_preparation_applied()
            self.validate_step_1()
            self.validate_step_2()
            self.validate_step_3()
            self.validate_step_4()
            settings = self.collect_training_settings()
            preflight = summarize_training_preflight(
                self.df.copy(), settings
            )
            prepared_label = (
                "windows" if preflight.get("uses_windows") else "samples"
            )
            status = (
                "Setup ready: "
                f"{preflight['prepared_samples']} prepared {prepared_label} → "
                f"{preflight['training_samples']} train, "
                f"{preflight['validation_samples']} validation, "
                f"{preflight['test_samples']} test."
            )
            self.training_preflight_status_var.set(status)
            messagebox.showinfo("Training Setup Ready", status)
        except Exception as exc:
            self.training_preflight_status_var.set(
                "Setup needs attention: " + str(exc)
            )
            messagebox.showerror("Training Setup Error", str(exc))

    def start_training(self):
        try:
            self._check_preparation_applied()
            self.validate_step_1()
            self.validate_step_2()
            self.validate_step_3()
            self.validate_step_4()
            settings = self.collect_training_settings()
            df_snapshot = self.df.copy()
            preflight = summarize_training_preflight(df_snapshot, settings)
        except Exception as exc:
            messagebox.showerror("Setting Error", str(exc))
            return

        self.training_running = True
        self.last_saved_model_package_path = None
        self.refresh_early_stopping_controls()
        self.ui_mode_combo.configure(state=tk.DISABLED)
        if hasattr(self, "guided_training_settings_frame"):
            for child in self.guided_training_settings_frame.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Button, ttk.Combobox)):
                    child.configure(state=tk.DISABLED)
        self.train_button.config(state=tk.DISABLED)
        self.check_training_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.save_results_button.config(state=tk.DISABLED)
        self.plot_button.config(state=tk.DISABLED)
        self.custom_training_results_button.config(state=tk.DISABLED)
        self.back_button.config(state=tk.DISABLED)
        self.next_button.config(state=tk.DISABLED)

        self.trained_model = None
        self.scaler = None
        self.label_encoder = None
        self.target_scaler = None
        self.history = None
        self.best_checkpoint_path = None
        self.training_run_id = None
        self.confusion_mat = None
        self.metadata = None
        self.y_test_result = None
        self.y_pred_result = None
        self.result_target_names = []
        self.anomaly_scores = None
        self.clear_training_custom_results()
        self.training_error_text = ""

        for item in self.epoch_tree.get_children():
            self.epoch_tree.delete(item)

        self.training_progress["maximum"] = max(1, settings["epochs"])
        self.training_progress["value"] = 0
        self.training_status_label.config(text="Starting training worker...")
        prepared_label = (
            "windows" if preflight.get("uses_windows") else "samples"
        )
        self.training_preflight_status_var.set(
            "Preflight passed: "
            f"{preflight['prepared_samples']} prepared {prepared_label} → "
            f"{preflight['training_samples']} train, "
            f"{preflight['validation_samples']} validation, "
            f"{preflight['test_samples']} test."
        )

        self.result_text.delete("1.0", tk.END)
        self.log_result("Training preflight passed.")
        self.log_result(self.training_preflight_status_var.get() + "\n")

        # Let Tk paint the disabled controls and progress state before model
        # construction starts. TensorFlow initialization can otherwise make a
        # successful click appear to do nothing for several seconds.
        self.after_idle(
            lambda: self.launch_training_worker(df_snapshot, settings)
        )

    def launch_training_worker(self, df_snapshot, settings):
        if not self.training_running:
            return
        self.training_status_label.config(text="Preparing data...")
        self.log_result("Training worker started.\n")
        self.training_thread = threading.Thread(
            target=self.training_worker,
            args=(df_snapshot, settings),
            daemon=True
        )
        self.training_thread.start()

    def training_worker(self, df, settings):
        if settings.get("data_mode") == DATA_MODE_IMAGE:
            self.training_worker_image(df, settings)
            return

        try:
            task_type = settings["task_type"]
            feature_cols = settings["feature_cols"]
            target_cols = settings["target_cols"]

            missing_method = settings["missing_method"]
            scaler_option = settings["scaler_option"]
            target_scaler_option = settings["target_scaler_option"]

            model_type = settings["model_type"]
            custom_model_input_mode = settings["custom_model_input_mode"]
            custom_model_code = settings["custom_model_code"]
            safe_model_spec = settings["safe_model_spec"]
            window_size = settings["window_size"]
            stride = settings["stride"]
            forecast_horizon = settings["forecast_horizon"]
            anomaly_percentile = settings["anomaly_percentile"]

            epochs = settings["epochs"]
            batch_size = settings["batch_size"]
            validation_split = settings["validation_split"]
            training_device = settings["training_device_selection"]
            device_info = settings["training_device_info"]
            self.thread_log(
                "\n" + DeviceManager.format_status(device_info)
            )

            hidden_activation = settings["hidden_activation"]
            dropout_rate = settings["dropout_rate"]

            output_mode = settings["output_mode"]
            output_activation = settings["output_activation"]
            loss_name = settings["loss_name"]

            optimizer_name = settings["optimizer_name"]
            learning_rate = settings["learning_rate"]

            test_size = settings["test_size"]

            checkpoint_paths = prepare_training_checkpoint_paths(
                df,
                settings,
            )
            best_model_path = checkpoint_paths["best_model_path"]
            best_state_path = checkpoint_paths["best_state_path"]
            recovery_directory = checkpoint_paths["recovery_directory"]
            recovery_pending = (
                recovery_directory.exists()
                and any(recovery_directory.iterdir())
            )
            self.training_run_id = checkpoint_paths["run_id"]
            self.best_checkpoint_path = str(best_model_path)

            self.thread_log(f"Task type: {task_type}")
            self.thread_log(f"Input columns: {feature_cols}")
            self.thread_log(f"Target columns: {target_cols}")
            self.thread_log(f"Model type: {model_type}")
            self.thread_log(f"Loss function: {loss_name}")
            self.thread_log(f"Optimizer: {optimizer_name}")
            self.thread_log(f"Output activation: {output_activation}")
            self.thread_log(
                "Automatic best-model checkpoint: enabled (val_loss)"
            )
            self.thread_log(
                "Interrupted-training recovery: enabled (every epoch)"
            )
            if recovery_pending:
                self.thread_log(
                    "An interrupted checkpoint was found. Training will "
                    "resume automatically from the latest saved epoch."
                )

            df = convert_features_to_numeric(df, feature_cols)

            if task_type in (
                TASK_REGRESSION,
                TASK_MULTI_OUTPUT,
                TASK_FORECASTING
            ):
                for col in target_cols:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = handle_missing_values_multi(
                df=df,
                feature_cols=feature_cols,
                target_cols=target_cols,
                method=missing_method
            )

            if df.empty:
                raise ValueError("No data left after missing value processing.")

            annotation_group_ids = (
                df["annotation_segment_id"].to_numpy()
                if "annotation_segment_id" in df.columns
                else None
            )
            X = df[feature_cols].to_numpy(dtype=float)
            y_source = None
            label_encoder = None
            class_names = None
            num_classes = None

            if task_type == TASK_CLASSIFICATION:
                label_encoder = LabelEncoder()
                y_source = label_encoder.fit_transform(
                    df[target_cols[0]].to_numpy()
                )
                class_names = [str(c) for c in label_encoder.classes_]
                num_classes = len(class_names)
                if num_classes < 2:
                    raise ValueError(
                        "The classification target must contain at least two classes."
                    )
            elif task_type in (
                TASK_REGRESSION,
                TASK_MULTI_OUTPUT,
                TASK_FORECASTING
            ):
                y_source = df[target_cols].to_numpy(dtype=float)

            self.thread_log(f"\nFinal X shape: {X.shape}")
            if y_source is not None:
                self.thread_log(f"Final target shape: {y_source.shape}")

            if output_mode == "Auto":
                if task_type == TASK_CLASSIFICATION:
                    output_units = num_classes
                elif task_type == TASK_FORECASTING:
                    output_units = len(target_cols) * forecast_horizon
                elif task_type == TASK_AUTOENCODER:
                    output_units = len(feature_cols)
                else:
                    output_units = len(target_cols)
            else:
                output_units = settings["output_units"]

            if task_type == TASK_CLASSIFICATION:
                self.thread_log(f"\nDetected classes: {class_names}")
                self.thread_log(f"Number of classes: {num_classes}")
            self.thread_log(f"Output units: {output_units}")

            use_window = (
                task_type == TASK_FORECASTING
                or model_type in [
                    "CNN",
                    "LSTM",
                    "CNN-LSTM",
                    "LSTM Autoencoder"
                ]
                or (
                    model_type in (
                        "Custom Python Model",
                        MODEL_TYPE_SAFE_CUSTOM,
                        MODEL_TYPE_SAFE_AI_LEGACY,
                    )
                    and custom_model_input_mode == "Window-based (3D)"
                )
            )

            if use_window:
                self.thread_log(f"\nCreating windows: window={window_size}, stride={stride}")

                if len(X) < window_size:
                    raise ValueError("Dataset is smaller than the selected window size.")

                X_model, y_model = create_supervised_windows(
                    X=X,
                    y=y_source,
                    window_size=window_size,
                    stride=stride,
                    task_type=task_type,
                    forecast_horizon=forecast_horizon,
                    group_ids=annotation_group_ids,
                )

                if len(X_model) == 0:
                    raise ValueError("No windows created. Please reduce window size or stride.")

                input_shape = (X_model.shape[1], X_model.shape[2])

            else:
                X_model = X
                if task_type == TASK_AUTOENCODER:
                    y_model = X.copy()
                else:
                    y_model = y_source
                input_shape = (X_model.shape[1],)

            self.thread_log(f"Model input shape: {input_shape}")

            X_train, X_test, y_train_raw, y_test_raw = safe_train_test_split(
                X_model,
                y_model,
                test_size,
                task_type=task_type
            )

            partitions = prepare_scaled_training_data(
                X_train, X_test, y_train_raw, y_test_raw,
                scaler_option, target_scaler_option, task_type,
                validation_split, len(target_cols),
            )
            X_train = partitions["X_train"]
            X_validation = partitions["X_validation"]
            X_test = partitions["X_test"]
            y_train_raw = partitions["y_train"]
            y_validation_raw = partitions["y_validation"]
            y_test_raw = partitions["y_test"]
            scaler = partitions["scaler"]
            target_scaler = partitions["target_scaler"]
            self.thread_log(
                f"Normalization fitted on training samples only: {scaler_option}; "
                f"target normalization: {target_scaler_option}"
            )

            if task_type == TASK_CLASSIFICATION:
                y_train = prepare_training_targets(
                    y=y_train_raw,
                    loss_name=loss_name,
                    output_units=output_units
                )
                y_validation = prepare_training_targets(
                    y=y_validation_raw,
                    loss_name=loss_name,
                    output_units=output_units,
                )
            else:
                y_train = y_train_raw
                y_validation = y_validation_raw

            self.thread_log(f"Training samples: {len(X_train)}")
            self.thread_log(f"Validation samples: {len(X_validation)}")
            self.thread_log(f"Testing samples: {len(X_test)}")

            model = call_on_tensorflow_device(
                training_device,
                build_selected_model,
                model_type=model_type,
                input_shape=input_shape,
                output_units=output_units,
                hidden_activation=hidden_activation,
                output_activation=output_activation,
                dropout_rate=dropout_rate,
                custom_model_code=custom_model_code,
                safe_model_spec=safe_model_spec,
                task_type=task_type,
            )

            optimizer = build_optimizer(
                optimizer_name=optimizer_name,
                learning_rate=learning_rate
            )

            call_on_tensorflow_device(
                training_device,
                model.compile,
                optimizer=optimizer,
                loss=loss_name,
                metrics=(
                    ["accuracy"]
                    if task_type == TASK_CLASSIFICATION
                    else ["mae"]
                ),
            )

            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))

            self.thread_log("\nModel Summary:")
            self.thread_log("\n".join(model_summary))

            checkpoint_callback = PersistentBestModelCheckpoint(
                filepath=best_model_path,
                state_path=best_state_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                mode="min",
                verbose=0,
            )
            recovery_callback = tf.keras.callbacks.BackupAndRestore(
                backup_dir=str(recovery_directory),
                save_freq="epoch",
                delete_checkpoint=True,
            )

            if checkpoint_callback.restored_previous_best:
                self.thread_log(
                    "Existing best checkpoint retained: "
                    f"epoch {checkpoint_callback.best_epoch}, "
                    f"val_loss={float(checkpoint_callback.best):.8f}"
                )

            callbacks = [
                TrainingProgressCallback(
                    self.thread_log,
                    self.thread_epoch_update,
                    metric_name=(
                        "accuracy"
                        if task_type == TASK_CLASSIFICATION
                        else "mae"
                    ),
                    metric_label=(
                        "accuracy"
                        if task_type == TASK_CLASSIFICATION
                        else "mae"
                    )
                ),
                checkpoint_callback,
                recovery_callback,
            ] + build_early_stopping_callbacks(settings, self.thread_log)

            self.thread_log("\nTraining neural network...")

            history = call_on_tensorflow_device(
                training_device,
                model.fit,
                X_train,
                y_train,
                validation_data=(X_validation, y_validation),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0,
            )

            if not best_model_path.exists():
                raise ValueError(
                    "Training did not create a best-model checkpoint. "
                    "Validation loss may be unavailable or non-finite."
                )

            self.thread_log(
                "\nReloading the best validation-loss checkpoint..."
            )
            model = call_on_tensorflow_device(
                training_device,
                load_best_checkpoint_model,
                best_model_path,
            )
            best_epoch = checkpoint_callback.best_epoch
            best_val_loss = float(checkpoint_callback.best)
            self.thread_log(
                f"Best checkpoint: epoch {best_epoch}, "
                f"val_loss={best_val_loss:.8f}"
            )

            self.thread_log("\nEvaluating best model...")

            raw_prediction = np.asarray(
                call_on_tensorflow_device(
                    training_device,
                    model.predict,
                    X_test,
                    verbose=0,
                )
            )

            cm = None
            report_text = None
            result_metrics = {}
            result_target_names = []
            actual_result = None
            predicted_result = None
            anomaly_threshold = None
            anomaly_scores = None

            if task_type == TASK_CLASSIFICATION:
                y_pred = prediction_to_class(
                    y_prob=raw_prediction,
                    output_units=output_units
                )
                accuracy = accuracy_score(y_test_raw, y_pred)
                f1 = f1_score(
                    y_test_raw,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
                labels = np.arange(num_classes)
                report_text = classification_report(
                    y_test_raw,
                    y_pred,
                    labels=labels,
                    target_names=class_names,
                    zero_division=0
                )
                cm = confusion_matrix(
                    y_test_raw,
                    y_pred,
                    labels=labels
                )
                result_metrics = {
                    "accuracy": float(accuracy),
                    "weighted_f1_score": float(f1)
                }
                actual_result = np.asarray(y_test_raw)
                predicted_result = np.asarray(y_pred)

            elif task_type == TASK_AUTOENCODER:
                predicted_result = inverse_transform_targets(
                    raw_prediction,
                    scaler
                )
                actual_result = inverse_transform_targets(
                    y_test_raw,
                    scaler
                )
                result_metrics = regression_metrics(
                    actual_result,
                    predicted_result
                )

                train_reconstruction = np.asarray(
                    call_on_tensorflow_device(
                        training_device,
                        model.predict,
                        X_train,
                        verbose=0,
                    )
                )
                reduction_axes = tuple(range(1, train_reconstruction.ndim))
                train_scores = np.mean(
                    np.abs(train_reconstruction - y_train_raw),
                    axis=reduction_axes
                )
                anomaly_threshold = float(
                    np.percentile(train_scores, anomaly_percentile)
                )
                anomaly_scores = np.mean(
                    np.abs(raw_prediction - y_test_raw),
                    axis=tuple(range(1, raw_prediction.ndim))
                )
                anomaly_count = int(
                    np.sum(anomaly_scores > anomaly_threshold)
                )
                result_metrics.update({
                    "anomaly_threshold_scaled_mae": anomaly_threshold,
                    "anomaly_count": anomaly_count,
                    "anomaly_rate": float(anomaly_count / len(anomaly_scores))
                })
                result_target_names = list(feature_cols)
                report_text = (
                    "Autoencoder reconstruction and anomaly evaluation\n"
                    f"MAE: {result_metrics['mae']:.8f}\n"
                    f"MSE: {result_metrics['mse']:.8f}\n"
                    f"RMSE: {result_metrics['rmse']:.8f}\n"
                    f"R²: {result_metrics['r2']:.8f}\n"
                    f"Anomaly threshold (scaled MAE, {anomaly_percentile:g}th "
                    f"percentile): {anomaly_threshold:.8f}\n"
                    f"Flagged test records: {anomaly_count}/{len(anomaly_scores)}"
                )

            else:
                predicted_result = inverse_transform_targets(
                    raw_prediction,
                    target_scaler
                )
                actual_result = inverse_transform_targets(
                    y_test_raw,
                    target_scaler
                )
                result_metrics = regression_metrics(
                    actual_result,
                    predicted_result
                )
                if task_type == TASK_FORECASTING:
                    result_target_names = [
                        f"{target}[t+{step}]"
                        for step in range(1, forecast_horizon + 1)
                        for target in target_cols
                    ]
                else:
                    result_target_names = list(target_cols)
                report_text = (
                    f"{task_type} evaluation\n"
                    f"MAE: {result_metrics['mae']:.8f}\n"
                    f"MSE: {result_metrics['mse']:.8f}\n"
                    f"RMSE: {result_metrics['rmse']:.8f}\n"
                    f"R²: {result_metrics['r2']:.8f}"
                )

            metadata = {
                "package_format_version": 7,
                "application_version": APP_VERSION,
                "data_mode": DATA_MODE_TABULAR,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "source_file": settings["source_file"],
                "dataset_analysis": settings["dataset_profile"],
                "ai_recommendation": settings["ai_recommendation"],
                "ai_provider_summary": settings["ai_provider_summary"],
                "ai_generation": settings["ai_generation"],
                "ai_generation_provider_summary": settings[
                    "ai_generation_provider_summary"
                ],

                "task_type": task_type,
                "feature_columns": feature_cols,
                "target_columns": target_cols,
                "label_column": (
                    target_cols[0]
                    if task_type == TASK_CLASSIFICATION
                    else None
                ),
                "class_names": class_names or [],

                "filter_method": settings["built_in_filter_method"],
                "filtered_columns": settings["built_in_filter_columns"],
                "moving_average_window": settings["moving_window"],
                "ema_span": settings["ema_span"],
                "median_window": settings["median_window"],
                "kalman_q": settings["kalman_q"],
                "kalman_r": settings["kalman_r"],

                "custom_filter_enabled": settings["custom_filter_enabled"],
                "custom_filter_mode": settings["custom_filter_mode"],
                "custom_filter_columns": settings["custom_filter_columns"],
                "safe_filter_spec": settings["safe_filter_spec"],
                "custom_filter_ai_prompt": settings[
                    "custom_filter_ai_prompt"
                ],
                "custom_filter_ai_report": settings[
                    "custom_filter_ai_report"
                ],

                "missing_value_method": missing_method,
                "normalization": scaler_option,
                "target_normalization": target_scaler_option,

                "model_type": model_type,
                "custom_model_input_mode": (
                    custom_model_input_mode
                    if model_type in (
                        "Custom Python Model",
                        MODEL_TYPE_SAFE_CUSTOM,
                        MODEL_TYPE_SAFE_AI_LEGACY,
                    )
                    else None
                ),
                "safe_model_spec": (
                    safe_model_spec
                    if model_type in (
                        MODEL_TYPE_SAFE_CUSTOM,
                        MODEL_TYPE_SAFE_AI_LEGACY,
                    )
                    else None
                ),
                "custom_model_ai_prompt": settings[
                    "custom_model_ai_prompt"
                ],
                "custom_model_ai_report": settings[
                    "custom_model_ai_report"
                ],
                "use_window": use_window,
                "window_size": window_size if use_window else None,
                "stride": stride if use_window else None,
                "forecast_horizon": (
                    forecast_horizon
                    if task_type == TASK_FORECASTING
                    else None
                ),
                "anomaly_percentile": (
                    anomaly_percentile
                    if task_type == TASK_AUTOENCODER
                    else None
                ),
                "anomaly_threshold": anomaly_threshold,

                "hidden_activation": hidden_activation,
                "dropout_rate": dropout_rate,
                "output_units": output_units,
                "output_activation": output_activation,
                "loss_function": loss_name,
                "optimizer": optimizer_name,
                "learning_rate": learning_rate,

                "epochs": epochs,
                "batch_size": batch_size,
                "execution_device": device_info,
                "validation_split": validation_split,
                "early_stopping": validate_early_stopping_settings(settings.get("early_stopping")),
                "test_size": test_size,
                "checkpointing": {
                    "automatic_best_model": True,
                    "monitor": "val_loss",
                    "mode": "min",
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_val_loss,
                    "complete_model_saved": True,
                    "optimizer_state_saved": True,
                    "interrupted_training_recovery": True,
                    "recovery_save_frequency": "epoch",
                    "training_run_id": checkpoint_paths["run_id"],
                },

                "input_shape": list(input_shape),
                "metrics": result_metrics
            }

            self.trained_model = model
            self.scaler = scaler
            self.target_scaler = target_scaler
            self.label_encoder = label_encoder
            self.metadata = metadata
            self.history = history
            self.confusion_mat = cm
            self.class_names = class_names or result_target_names
            self.y_test_result = actual_result
            self.y_pred_result = predicted_result
            self.result_target_names = result_target_names
            self.anomaly_scores = anomaly_scores
            self.training_feature_sample = make_feature_sample(
                X_test, actual_result if task_type == TASK_CLASSIFICATION else None,
                predicted_result if task_type == TASK_CLASSIFICATION else None, class_names)
            self.training_feature_source = (
                "Embedding uses model-input features after preprocessing; windows are flattened. "
                "Large feature sets retain up to 128 evenly spaced coordinates.")

            self.thread_log("\nTraining completed successfully.")
            self.thread_log(
                "The best validation-loss model is active and will be used "
                "for package export."
            )
            self.thread_log("=" * 60)
            if task_type == TASK_CLASSIFICATION:
                self.thread_log(
                    f"Accuracy: {result_metrics['accuracy'] * 100:.2f}%"
                )
                self.thread_log(
                    "Weighted F1-score: "
                    f"{result_metrics['weighted_f1_score'] * 100:.2f}%"
                )
            else:
                self.thread_log(f"MAE: {result_metrics['mae']:.8f}")
                self.thread_log(f"MSE: {result_metrics['mse']:.8f}")
                self.thread_log(f"RMSE: {result_metrics['rmse']:.8f}")
                self.thread_log(f"R²: {result_metrics['r2']:.8f}")
            self.thread_log("=" * 60)
            self.thread_log("\nEvaluation Report:")
            self.thread_log(report_text)
            if cm is not None:
                self.thread_log("\nConfusion Matrix:")
                self.thread_log(cm)

            self.ui_queue.put(("finished", None))

        except Exception:
            error_text = traceback.format_exc()
            self.thread_log("\nTraining failed.")
            self.thread_log(error_text)
            self.ui_queue.put(("failed", error_text))

    def training_worker_image(self, records_df, settings):
        """Train and evaluate an image classifier using streaming datasets."""
        try:
            class_names = list(settings["image_class_names"])
            image_height = int(settings["image_height"])
            image_width = int(settings["image_width"])
            color_mode = settings["image_color_mode"]
            channels = 1 if color_mode == "Grayscale" else 3
            model_type = settings["model_type"]
            epochs = int(settings["epochs"])
            batch_size = int(settings["batch_size"])
            training_device = settings["training_device_selection"]
            device_info = settings["training_device_info"]
            validation_split = float(settings["validation_split"])
            test_size = float(settings["test_size"])
            dropout_rate = float(settings["dropout_rate"])
            optimizer_name = settings["optimizer_name"]
            learning_rate = float(settings["learning_rate"])
            augmentation_enabled = bool(settings["image_augmentation"])
            pretrained_weights = bool(
                settings["image_pretrained_weights"]
            )
            output_units = len(class_names)

            if output_units < 2:
                raise ValueError(
                    "Image classification requires at least two classes."
                )
            required_record_columns = {
                "file_path",
                "class_name",
                "class_index",
            }
            missing_columns = sorted(
                required_record_columns - set(records_df.columns)
            )
            if missing_columns:
                raise ValueError(
                    "Image record data is missing: " + str(missing_columns)
                )

            checkpoint_paths = prepare_training_checkpoint_paths(
                records_df,
                settings,
            )
            best_model_path = checkpoint_paths["best_model_path"]
            best_state_path = checkpoint_paths["best_state_path"]
            recovery_directory = checkpoint_paths["recovery_directory"]
            recovery_pending = (
                recovery_directory.exists()
                and any(recovery_directory.iterdir())
            )
            self.training_run_id = checkpoint_paths["run_id"]
            self.best_checkpoint_path = str(best_model_path)

            self.thread_log(f"Data mode: {DATA_MODE_IMAGE}")
            self.thread_log(f"Image folder: {settings['image_directory']}")
            self.thread_log(f"Classes: {class_names}")
            self.thread_log(
                f"Image size: {image_height} × {image_width} × {channels}"
            )
            self.thread_log(f"Model type: {model_type}")
            self.thread_log(DeviceManager.format_status(device_info))
            self.thread_log(
                f"Training augmentation: {augmentation_enabled}"
            )
            self.thread_log(
                f"ImageNet pretrained weights: {pretrained_weights}"
            )
            self.thread_log(
                "Automatic best-model checkpoint: enabled (val_loss)"
            )
            self.thread_log(
                "Interrupted-training recovery: enabled (every epoch)"
            )
            if recovery_pending:
                self.thread_log(
                    "An interrupted checkpoint was found. Training will "
                    "resume automatically from the latest saved epoch."
                )

            train_records, validation_records, test_records = (
                split_image_records(
                    records_df,
                    test_fraction=test_size,
                    validation_fraction=validation_split,
                    random_seed=42,
                )
            )
            self.thread_log(
                "\nImage split: "
                f"train={len(train_records)}, "
                f"validation={len(validation_records)}, "
                f"test={len(test_records)}"
            )

            train_dataset = make_image_tf_dataset(
                train_records,
                image_height=image_height,
                image_width=image_width,
                color_mode=color_mode,
                batch_size=batch_size,
                shuffle=True,
            )
            validation_dataset = make_image_tf_dataset(
                validation_records,
                image_height=image_height,
                image_width=image_width,
                color_mode=color_mode,
                batch_size=batch_size,
                shuffle=False,
            )
            test_dataset = make_image_tf_dataset(
                test_records,
                image_height=image_height,
                image_width=image_width,
                color_mode=color_mode,
                batch_size=batch_size,
                shuffle=False,
            )

            try:
                model = call_on_tensorflow_device(
                    training_device,
                    build_image_classification_model,
                    model_type=model_type,
                    input_shape=(image_height, image_width, channels),
                    output_units=output_units,
                    dropout_rate=dropout_rate,
                    augmentation_enabled=augmentation_enabled,
                    pretrained_weights=pretrained_weights,
                )
            except Exception as exc:
                if (
                    pretrained_weights
                    and model_type in (
                        IMAGE_MODEL_MOBILENET,
                        IMAGE_MODEL_EFFICIENTNET,
                    )
                ):
                    raise ValueError(
                        "The transfer-learning model could not load ImageNet "
                        "weights. Check internet access, or disable pretrained "
                        f"weights and try again. Original error: {exc}"
                    ) from exc
                raise

            optimizer = build_optimizer(
                optimizer_name=optimizer_name,
                learning_rate=learning_rate,
            )
            call_on_tensorflow_device(
                training_device,
                model.compile,
                optimizer=optimizer,
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            model_summary = []
            model.summary(print_fn=model_summary.append)
            self.thread_log("\nModel Summary:")
            self.thread_log("\n".join(model_summary))

            checkpoint_callback = PersistentBestModelCheckpoint(
                filepath=best_model_path,
                state_path=best_state_path,
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=False,
                mode="min",
                verbose=0,
            )
            recovery_callback = tf.keras.callbacks.BackupAndRestore(
                backup_dir=str(recovery_directory),
                save_freq="epoch",
                delete_checkpoint=True,
            )
            callbacks = [
                TrainingProgressCallback(
                    self.thread_log,
                    self.thread_epoch_update,
                    metric_name="accuracy",
                    metric_label="accuracy",
                ),
                checkpoint_callback,
                recovery_callback,
            ] + build_early_stopping_callbacks(settings, self.thread_log)

            if checkpoint_callback.restored_previous_best:
                self.thread_log(
                    "Existing best checkpoint retained: "
                    f"epoch {checkpoint_callback.best_epoch}, "
                    f"val_loss={float(checkpoint_callback.best):.8f}"
                )

            self.thread_log("\nTraining image classifier...")
            history = call_on_tensorflow_device(
                training_device,
                model.fit,
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs,
                callbacks=callbacks,
                verbose=0,
            )

            if not best_model_path.exists():
                raise ValueError(
                    "Training did not create a best-model checkpoint. "
                    "Validation loss may be unavailable or non-finite."
                )

            self.thread_log(
                "\nReloading the best validation-loss checkpoint..."
            )
            model = call_on_tensorflow_device(
                training_device,
                load_best_checkpoint_model,
                best_model_path,
            )
            best_epoch = checkpoint_callback.best_epoch
            best_val_loss = float(checkpoint_callback.best)
            self.thread_log(
                f"Best checkpoint: epoch {best_epoch}, "
                f"val_loss={best_val_loss:.8f}"
            )

            raw_prediction = np.asarray(
                call_on_tensorflow_device(
                    training_device,
                    model.predict,
                    test_dataset,
                    verbose=0,
                )
            )
            predicted_labels = np.argmax(raw_prediction, axis=1)
            actual_labels = test_records[
                "class_index"
            ].astype(int).to_numpy()
            labels = np.arange(output_units)
            accuracy = accuracy_score(actual_labels, predicted_labels)
            weighted_f1 = f1_score(
                actual_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
            report_text = classification_report(
                actual_labels,
                predicted_labels,
                labels=labels,
                target_names=class_names,
                zero_division=0,
            )
            cm = confusion_matrix(
                actual_labels,
                predicted_labels,
                labels=labels,
            )
            result_metrics = {
                "accuracy": float(accuracy),
                "weighted_f1_score": float(weighted_f1),
                "training_images": int(len(train_records)),
                "validation_images": int(len(validation_records)),
                "test_images": int(len(test_records)),
            }

            metadata = {
                "package_format_version": 7,
                "application_version": APP_VERSION,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data_mode": DATA_MODE_IMAGE,
                "source_file": settings["source_file"],
                "source_directory": settings["image_directory"],
                "dataset_analysis": settings["dataset_profile"],
                "task_type": TASK_CLASSIFICATION,
                "feature_columns": [],
                "target_columns": ["class_name"],
                "label_column": "class_name",
                "class_names": class_names,
                "class_to_index": {
                    name: index
                    for index, name in enumerate(class_names)
                },
                "model_type": model_type,
                "image_height": image_height,
                "image_width": image_width,
                "image_channels": channels,
                "image_color_mode": color_mode,
                "image_extensions": sorted(SUPPORTED_IMAGE_EXTENSIONS),
                "image_augmentation": augmentation_enabled,
                "image_pretrained_weights": pretrained_weights,
                "normalization": "Stored inside image model",
                "output_units": output_units,
                "output_activation": "softmax",
                "loss_function": "sparse_categorical_crossentropy",
                "optimizer": optimizer_name,
                "learning_rate": learning_rate,
                "dropout_rate": dropout_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "execution_device": device_info,
                "validation_split": validation_split,
                "early_stopping": validate_early_stopping_settings(settings.get("early_stopping")),
                "test_size": test_size,
                "checkpointing": {
                    "automatic_best_model": True,
                    "monitor": "val_loss",
                    "mode": "min",
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_val_loss,
                    "complete_model_saved": True,
                    "optimizer_state_saved": True,
                    "interrupted_training_recovery": True,
                    "recovery_save_frequency": "epoch",
                    "training_run_id": checkpoint_paths["run_id"],
                },
                "input_shape": [image_height, image_width, channels],
                "metrics": result_metrics,
            }

            self.trained_model = model
            self.scaler = None
            self.target_scaler = None
            self.label_encoder = None
            self.metadata = metadata
            self.history = history
            self.confusion_mat = cm
            self.class_names = class_names
            self.y_test_result = actual_labels
            self.y_pred_result = predicted_labels
            self.result_target_names = []
            self.anomaly_scores = None
            self.training_feature_sample = make_feature_sample(
                raw_prediction, actual_labels, predicted_labels, class_names)
            self.training_feature_source = (
                "Image embedding uses predicted class probabilities, not image pixels or hidden-layer features.")

            self.thread_log("\nTraining completed successfully.")
            self.thread_log(
                "The best validation-loss image model is active and ready "
                "for package export."
            )
            self.thread_log("=" * 60)
            self.thread_log(f"Accuracy: {accuracy * 100:.2f}%")
            self.thread_log(
                f"Weighted F1-score: {weighted_f1 * 100:.2f}%"
            )
            self.thread_log("=" * 60)
            self.thread_log("\nEvaluation Report:")
            self.thread_log(report_text)
            self.thread_log("\nConfusion Matrix:")
            self.thread_log(cm)
            self.ui_queue.put(("finished", None))
        except Exception:
            error_text = traceback.format_exc()
            self.thread_log("\nImage training failed.")
            self.thread_log(error_text)
            self.ui_queue.put(("failed", error_text))

    def training_finished(self):
        self.training_running = False
        self.refresh_early_stopping_controls()
        self.training_thread = None
        self.ui_mode_combo.configure(state="readonly")
        if hasattr(self, "guided_training_settings_frame"):
            for child in self.guided_training_settings_frame.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Button)):
                    child.configure(state=tk.NORMAL)
        self.check_training_button.config(state=tk.NORMAL)
        self.train_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)
        self.save_results_button.config(state=tk.NORMAL)
        self.plot_button.config(state=tk.NORMAL)
        self.custom_training_results_button.config(state=tk.NORMAL)
        self.back_button.config(state=tk.NORMAL)
        self.next_button.config(state=tk.DISABLED, text="Finish")

        if hasattr(self, "training_status_label"):
            completed_epochs = len(
                self.history.history.get("loss", [])
            ) if self.history is not None else 0
            self.training_status_label.config(
                text=f"Completed after {completed_epochs} epoch(s)"
            )
        self.training_preflight_status_var.set(
            "Training completed. The best validation-loss checkpoint was "
            "reloaded for evaluation and saving."
        )

        checkpoint_info = (
            self.metadata.get("checkpointing", {})
            if isinstance(self.metadata, dict)
            else {}
        )
        messagebox.showinfo(
            "Training Completed",
            "Model training completed successfully.\n\n"
            f"Best epoch: {checkpoint_info.get('best_epoch', '—')}\n"
            "The best validation-loss model is ready to save."
        )

        self._update_guided_footer()

    def training_failed(self, error_text=None):
        self.training_running = False
        self.refresh_early_stopping_controls()
        self.training_thread = None
        self.ui_mode_combo.configure(state="readonly")
        if hasattr(self, "guided_training_settings_frame"):
            for child in self.guided_training_settings_frame.winfo_children():
                if isinstance(child, (ttk.Entry, ttk.Button)):
                    child.configure(state=tk.NORMAL)
        self.training_error_text = str(error_text or "Unknown training error.")
        self.check_training_button.config(state=tk.NORMAL)
        self.train_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED)
        self.save_results_button.config(state=tk.DISABLED)
        self.plot_button.config(state=tk.DISABLED)
        self.custom_training_results_button.config(state=tk.DISABLED)
        self.back_button.config(state=tk.NORMAL)
        self.next_button.config(state=tk.DISABLED, text="Finish")

        error_lines = [
            line.strip()
            for line in self.training_error_text.splitlines()
            if line.strip()
        ]
        error_summary = (
            error_lines[-1] if error_lines else "Unknown training error."
        )
        if len(error_summary) > 360:
            error_summary = error_summary[:357] + "..."

        if hasattr(self, "training_status_label"):
            self.training_status_label.config(text="Training failed")
        self.training_preflight_status_var.set(
            "Training stopped: " + error_summary
        )

        messagebox.showerror(
            "Training Failed",
            "Training stopped before a usable model was produced.\n\n"
            + error_summary
            + "\n\nThe complete traceback remains visible in the Training Log."
        )

        self._update_guided_footer()

    def show_plots(self):
        if self.history is None:
            messagebox.showwarning("No Result", "Please train a model first.")
            return

        task_type = self.task_type_var.get()
        metric_key = "accuracy" if task_type == TASK_CLASSIFICATION else "mae"
        metric_label = "Accuracy" if task_type == TASK_CLASSIFICATION else "MAE"

        plt.figure()
        if metric_key in self.history.history:
            plt.plot(
                self.history.history[metric_key],
                label=f"Training {metric_label}"
            )

        if "val_" + metric_key in self.history.history:
            plt.plot(
                self.history.history["val_" + metric_key],
                label=f"Validation {metric_label}"
            )

        plt.xlabel("Epoch")
        plt.ylabel(metric_label)
        plt.title(f"Training and Validation {metric_label}")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure()
        plt.plot(self.history.history["loss"], label="Training Loss")

        if "val_loss" in self.history.history:
            plt.plot(self.history.history["val_loss"], label="Validation Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.show()

        if task_type == TASK_CLASSIFICATION:
            cm = self.confusion_mat
            class_names = self.class_names
            if cm is None:
                return

            plt.figure()
            plt.imshow(cm)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("Actual Label")
            plt.colorbar()

            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks, class_names, rotation=45)
            plt.yticks(tick_marks, class_names)

            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    plt.text(j, i, cm[i, j], ha="center", va="center")

            plt.tight_layout()
            plt.show()
            return

        if self.y_test_result is not None and self.y_pred_result is not None:
            actual = np.asarray(self.y_test_result).reshape(
                len(self.y_test_result),
                -1
            )
            predicted = np.asarray(self.y_pred_result).reshape(
                len(self.y_pred_result),
                -1
            )
            series_count = min(4, actual.shape[1])
            plt.figure()
            sample_count = min(300, len(actual))
            for index in range(series_count):
                name = (
                    self.result_target_names[index]
                    if index < len(self.result_target_names)
                    else f"Output {index + 1}"
                )
                plt.plot(
                    actual[:sample_count, index],
                    label=f"Actual {name}"
                )
                plt.plot(
                    predicted[:sample_count, index],
                    linestyle="--",
                    label=f"Predicted {name}"
                )
            plt.xlabel("Test Sample")
            plt.ylabel("Value")
            plt.title("Actual vs Predicted")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        if task_type == TASK_AUTOENCODER and self.anomaly_scores is not None:
            plt.figure()
            plt.hist(self.anomaly_scores, bins=40, alpha=0.8)
            threshold = self.metadata.get("anomaly_threshold")
            if threshold is not None:
                plt.axvline(
                    threshold,
                    color="red",
                    linestyle="--",
                    label="Anomaly threshold"
                )
                plt.legend()
            plt.xlabel("Scaled Reconstruction MAE")
            plt.ylabel("Count")
            plt.title("Autoencoder Anomaly Scores")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

    def _write_current_training_results(self, output_directory):
        """Write the current held-out evaluation and training history."""
        if self.trained_model is None or self.metadata is None:
            raise ValueError("No completed training result is available.")

        task_type = self.metadata.get(
            "task_type",
            self.task_type_var.get(),
        )
        predictions_df = make_training_prediction_dataframe(
            task_type=task_type,
            actual_values=self.y_test_result,
            predicted_values=self.y_pred_result,
            class_names=self.class_names,
            target_names=self.result_target_names,
            anomaly_scores=self.anomaly_scores,
        )
        history_dict = (
            dict(self.history.history)
            if self.history is not None
            else {}
        )
        log_text = self._last_training_log
        if hasattr(self, "result_text") and self.result_text.winfo_exists():
            log_text = self.result_text.get("1.0", tk.END).strip()

        source_description = (
            self.metadata.get("source_directory")
            or self.metadata.get("source_file")
            or "Training dataset"
        )
        created_files = write_complete_result_files(
            output_directory,
            title="Model Training and Held-out Test Results",
            task_type=task_type,
            source_description=str(source_description),
            metadata=self.metadata,
            predictions_df=predictions_df,
            metrics=self.metadata.get("metrics", {}),
            report_text=log_text,
            confusion_mat=self.confusion_mat,
            class_names=self.class_names,
            history=history_dict,
            log_text=log_text,
            actual_values=self.y_test_result,
            predicted_values=self.y_pred_result,
            target_names=self.result_target_names,
            anomaly_scores=self.anomaly_scores,
        )

        return self.export_training_custom_results(output_directory, created_files)

    def save_training_results(self):
        """Save complete training/evaluation evidence without duplicating model."""
        if self.trained_model is None:
            messagebox.showwarning(
                "No Results",
                "Please train a model before saving its results.",
            )
            return

        source_stem = Path(
            self.metadata.get("source_file")
            or self.metadata.get("source_directory")
            or "trained_model"
        ).stem
        save_path = filedialog.asksaveasfilename(
            title="Save Complete Training Results",
            initialfile=f"{source_stem}_training_results.zip",
            defaultextension=".zip",
            filetypes=[("ZIP Result Bundle", "*.zip")],
        )
        if not save_path:
            return

        temp_dir = tempfile.mkdtemp(prefix="nn_training_results_")
        try:
            self._write_current_training_results(temp_dir)
            zip_result_directory(temp_dir, save_path)
            messagebox.showinfo(
                "Complete Results Saved",
                "Training history, held-out predictions, metrics, reports, "
                f"tables, and plots were saved successfully:\n{save_path}",
            )
        except Exception as exc:
            messagebox.showerror(
                "Save Error",
                f"Failed to save complete training results:\n{exc}",
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def open_last_saved_deployment(self):
        if (
            not self.last_saved_model_package_path
            or not Path(self.last_saved_model_package_path).is_file()
        ):
            messagebox.showinfo(
                "Save Model Package First",
                "Save the best model package first. Deploy & Integrate keeps "
                "the trained model, scaler, preprocessing, feature order, and "
                "metadata together.",
            )
            return
        self.open_deploy_integrate(
            initial_model_path=self.last_saved_model_package_path
        )

    def save_model_package(self):
        if self.trained_model is None:
            messagebox.showwarning("No Model", "Please train a model first.")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Best Model Package",
            defaultextension=".zip",
            filetypes=[("ZIP Package", "*.zip")]
        )

        if not save_path:
            return

        try:
            temp_dir = tempfile.mkdtemp()

            model_path = os.path.join(temp_dir, "trained_model.keras")
            scaler_path = os.path.join(temp_dir, "scaler.pkl")
            label_encoder_path = os.path.join(temp_dir, "label_encoder.pkl")
            target_scaler_path = os.path.join(temp_dir, "target_scaler.pkl")
            metadata_path = os.path.join(temp_dir, "metadata.json")
            analysis_json_path = os.path.join(
                temp_dir,
                "dataset_analysis.json"
            )
            analysis_report_path = os.path.join(
                temp_dir,
                "dataset_analysis.txt"
            )
            ai_recommendation_json_path = os.path.join(
                temp_dir,
                "ai_recommendation.json"
            )
            ai_recommendation_report_path = os.path.join(
                temp_dir,
                "ai_recommendation.txt"
            )
            ai_generation_json_path = os.path.join(
                temp_dir,
                "ai_generation.json",
            )
            ai_generation_report_path = os.path.join(
                temp_dir,
                "ai_generation.txt",
            )
            safe_filter_spec_path = os.path.join(
                temp_dir,
                "safe_filter_spec.json",
            )
            safe_model_spec_path = os.path.join(
                temp_dir,
                "safe_model_spec.json",
            )
            results_dir = os.path.join(temp_dir, "results")
            custom_filter_path = os.path.join(temp_dir, "custom_filter.py")
            custom_model_path = os.path.join(temp_dir, "custom_model.py")

            if (
                self.best_checkpoint_path
                and os.path.exists(self.best_checkpoint_path)
            ):
                shutil.copy2(self.best_checkpoint_path, model_path)
            else:
                # Backward-compatible fallback for models loaded from an
                # older package that has no Version 16 checkpoint path.
                self.trained_model.save(model_path)
            joblib.dump(self.scaler, scaler_path)
            if self.label_encoder is not None:
                joblib.dump(self.label_encoder, label_encoder_path)
            if self.target_scaler is not None:
                joblib.dump(self.target_scaler, target_scaler_path)

            included_result_files = self._write_current_training_results(
                results_dir
            )
            self.metadata["included_training_results"] = {
                "available": True,
                "directory": "results",
                "files": included_result_files,
            }
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(
                    self.metadata,
                    f,
                    indent=4,
                    default=_result_json_default,
                )

            if self.dataset_profile is not None:
                with open(
                    analysis_json_path,
                    "w",
                    encoding="utf-8"
                ) as file:
                    json.dump(self.dataset_profile, file, indent=4)
                with open(
                    analysis_report_path,
                    "w",
                    encoding="utf-8"
                ) as file:
                    file.write(self.dataset_analysis_report)

            if self.ai_recommendation is not None:
                ai_saved_record = {
                    "provider": self.ai_provider_summary,
                    "recommendation": self.ai_recommendation,
                }
                with open(
                    ai_recommendation_json_path,
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(ai_saved_record, file, indent=4)
                with open(
                    ai_recommendation_report_path,
                    "w",
                    encoding="utf-8",
                ) as file:
                    file.write(self.ai_recommendation_report)

            if self.ai_generation is not None:
                ai_generation_record = {
                    "provider": self.ai_generation_provider_summary,
                    "generation": self.ai_generation,
                }
                with open(
                    ai_generation_json_path,
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(ai_generation_record, file, indent=4)
                with open(
                    ai_generation_report_path,
                    "w",
                    encoding="utf-8",
                ) as file:
                    file.write(self.ai_generation_report)

            if (
                self.custom_filter_enabled_var.get()
                and self.custom_filter_mode_var.get()
                == CUSTOM_FILTER_MODE_EXPERT
            ):
                with open(custom_filter_path, "w", encoding="utf-8") as f:
                    f.write(self.custom_filter_code)

            if self.model_type_var.get() == "Custom Python Model":
                with open(custom_model_path, "w", encoding="utf-8") as f:
                    f.write(self.custom_model_code)

            if self.safe_filter_spec is not None:
                with open(
                    safe_filter_spec_path,
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(self.safe_filter_spec, file, indent=4)

            if self.safe_model_spec is not None:
                with open(
                    safe_model_spec_path,
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(self.safe_model_spec, file, indent=4)

            with zipfile.ZipFile(save_path, "w") as zipf:
                zipf.write(model_path, arcname="trained_model.keras")
                zipf.write(scaler_path, arcname="scaler.pkl")
                zipf.write(metadata_path, arcname="metadata.json")
                if self.dataset_profile is not None:
                    zipf.write(
                        analysis_json_path,
                        arcname="dataset_analysis.json"
                    )
                    zipf.write(
                        analysis_report_path,
                        arcname="dataset_analysis.txt"
                    )
                if self.ai_recommendation is not None:
                    zipf.write(
                        ai_recommendation_json_path,
                        arcname="ai_recommendation.json",
                    )
                    zipf.write(
                        ai_recommendation_report_path,
                        arcname="ai_recommendation.txt",
                    )
                if self.ai_generation is not None:
                    zipf.write(
                        ai_generation_json_path,
                        arcname="ai_generation.json",
                    )
                    zipf.write(
                        ai_generation_report_path,
                        arcname="ai_generation.txt",
                    )
                if self.label_encoder is not None:
                    zipf.write(
                        label_encoder_path,
                        arcname="label_encoder.pkl"
                    )
                if self.target_scaler is not None:
                    zipf.write(
                        target_scaler_path,
                        arcname="target_scaler.pkl"
                    )

                if (
                    self.custom_filter_enabled_var.get()
                    and self.custom_filter_mode_var.get()
                    == CUSTOM_FILTER_MODE_EXPERT
                ):
                    zipf.write(custom_filter_path, arcname="custom_filter.py")

                if self.model_type_var.get() == "Custom Python Model":
                    zipf.write(custom_model_path, arcname="custom_model.py")

                if self.safe_filter_spec is not None:
                    zipf.write(
                        safe_filter_spec_path,
                        arcname="safe_filter_spec.json",
                    )

                if self.safe_model_spec is not None:
                    zipf.write(
                        safe_model_spec_path,
                        arcname="safe_model_spec.json",
                    )

                for result_path in sorted(Path(results_dir).rglob("*")):
                    if result_path.is_file():
                        zipf.write(
                            result_path,
                            arcname=str(
                                Path("results")
                                / result_path.relative_to(results_dir)
                            ),
                        )

            self.last_saved_model_package_path = save_path
            if hasattr(self, "deploy_trained_button"):
                self.deploy_trained_button.config(state=tk.NORMAL)
            messagebox.showinfo(
                "Saved",
                "Best model package and complete training results were saved "
                f"successfully:\n{save_path}\n\n"
                "Deploy & Integrate is now available on this page."
            )

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save model package:\n{e}")
