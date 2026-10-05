"""Ui / evaluation for NN Training Studio."""

from pathlib import Path
from tkinter.scrolledtext import ScrolledText
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from datetime import datetime
from sklearn.metrics import f1_score
from tkinter import filedialog
from tkinter import messagebox
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import queue
import shutil
import tempfile
import threading
import tkinter as tk
import traceback
from tkinter import ttk
from nn_training_studio.constants import (
    CUSTOM_FILTER_MODE_SAFE_AI,
    CUSTOM_FILTER_MODE_SAFE_AI_LEGACY,
    DATA_MODE_IMAGE,
    DATA_MODE_TABULAR,
    DEVICE_AUTO,
    TASK_AUTOENCODER,
    TASK_CLASSIFICATION,
    TASK_FORECASTING,
    TASK_MULTI_OUTPUT,
    TASK_REGRESSION,
)
from nn_training_studio.deployment import (
    detect_saved_model_type,
)
from nn_training_studio.devices import (
    DeviceManager,
    call_on_tensorflow_device,
)
from nn_training_studio.filters import (
    apply_safe_filter_spec,
    apply_selected_filter,
    execute_custom_filter_code,
)
from nn_training_studio.image_data import (
    make_image_tf_dataset,
    scan_image_dataset,
)
from nn_training_studio.inference import (
    create_inference_windows,
    encode_external_labels,
    handle_missing_features_only,
    make_probability_column_name,
    validate_model_input_shape,
)
from nn_training_studio.model_io import (
    load_saved_model_package,
)
from nn_training_studio.models import (
    prediction_to_class,
)
from nn_training_studio.preprocessing import (
    convert_features_to_numeric,
    handle_missing_values,
    handle_missing_values_multi,
    inverse_transform_targets,
    regression_metrics,
)
from nn_training_studio.results import (
    write_complete_result_files,
    zip_result_directory,
)
from nn_training_studio.ui.detection import (
    ObjectDetectionWindow,
)
from nn_training_studio.ui.results import (
    CustomResultsStudioWindow,
)


class ModelEvaluationWindow(tk.Toplevel):
    """Independent workflow for loading and evaluating a saved model package."""

    NO_LABEL_TEXT = "<No label - prediction only>"

    def __init__(self, parent, initial_package_path=None):
        super().__init__(parent)

        self.title("Signal / Image Model Application")
        self.geometry("1220x800")
        self.minsize(1050, 700)

        self.package_data = None
        self.external_df = None
        self.external_csv_path = None
        self.external_image_directory = None
        self.prediction_df = None
        self.evaluation_report = None
        self.evaluation_cm = None
        self.evaluation_class_names = None
        self.evaluation_metrics = None
        self.custom_result_directories = []
        self.worker_queue = queue.Queue()
        self.evaluation_running = False

        self.eval_label_var = tk.StringVar(value=self.NO_LABEL_TEXT)
        self.apply_saved_filters_var = tk.BooleanVar(value=True)
        self.inference_device_var = tk.StringVar(value=DEVICE_AUTO)
        self.inference_device_status_var = tk.StringVar(value="")

        self.protocol("WM_DELETE_WINDOW", self.close_window)

        self.build_interface()
        self.refresh_inference_devices()
        self.after(100, self.process_worker_queue)
        if initial_package_path:
            self.after_idle(
                lambda: self.load_package_path(
                    initial_package_path,
                    confirm_trust=True,
                )
            )

    def build_interface(self):
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=12, pady=10)

        ttk.Label(
            header,
            text="Signal / Image Model Application",
            font=("Arial", 18, "bold")
        ).pack(anchor="w")

        ttk.Label(
            header,
            text=(
                "Load model → review requirements → load compatible new data "
                "→ predict or evaluate → preview, customize, and export results"
            )
        ).pack(anchor="w", pady=(3, 0))

        load_frame = ttk.LabelFrame(
            self,
            text="1. Select Model Package and Application Data",
        )
        load_frame.pack(fill=tk.X, padx=12, pady=5)

        package_row = ttk.Frame(load_frame)
        package_row.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(
            package_row,
            text="Load Model Package (.zip)",
            command=self.load_package
        ).pack(side=tk.LEFT)
        self.package_label = ttk.Label(package_row, text="No package loaded")
        self.package_label.pack(side=tk.LEFT, padx=10)
        self.export_original_results_button = ttk.Button(
            package_row,
            text="Export Original Training Results",
            command=self.export_original_training_results,
            state=tk.DISABLED,
        )
        self.export_original_results_button.pack(side=tk.RIGHT)

        data_row = ttk.Frame(load_frame)
        data_row.pack(fill=tk.X, padx=8, pady=6)
        self.load_evaluation_data_button = ttk.Button(
            data_row,
            text="Load Evaluation Data",
            command=self.load_external_data
        )
        self.load_evaluation_data_button.pack(side=tk.LEFT)
        self.data_label = ttk.Label(data_row, text="No evaluation dataset loaded")
        self.data_label.pack(side=tk.LEFT, padx=10)

        option_row = ttk.Frame(load_frame)
        option_row.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(
            option_row,
            text="Actual Label / Target Column:"
        ).pack(side=tk.LEFT)
        self.eval_label_combo = ttk.Combobox(
            option_row,
            textvariable=self.eval_label_var,
            values=[self.NO_LABEL_TEXT],
            state="readonly",
            width=34
        )
        self.eval_label_combo.pack(side=tk.LEFT, padx=(5, 18))

        ttk.Checkbutton(
            option_row,
            text="Apply the filters saved in the model package",
            variable=self.apply_saved_filters_var
        ).pack(side=tk.LEFT)

        device_row = ttk.Frame(load_frame)
        device_row.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Label(device_row, text="Inference device:").pack(side=tk.LEFT)
        self.inference_device_combo = ttk.Combobox(
            device_row,
            textvariable=self.inference_device_var,
            values=DeviceManager.tensorflow_options(),
            state="readonly",
            width=38,
        )
        self.inference_device_combo.pack(side=tk.LEFT, padx=(5, 8))
        self.inference_device_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.update_inference_device_status(),
        )
        ttk.Button(
            device_row,
            text="Refresh Devices",
            command=self.refresh_inference_devices,
        ).pack(side=tk.LEFT)
        ttk.Label(
            device_row,
            textvariable=self.inference_device_status_var,
            foreground="#245a85",
        ).pack(side=tk.LEFT, padx=12)

        info_frame = ttk.LabelFrame(
            self,
            text="2. Review Model Requirements and Compatibility",
        )
        info_frame.pack(fill=tk.X, padx=12, pady=5)
        self.package_info_text = ScrolledText(info_frame, height=9, wrap=tk.WORD)
        self.package_info_text.pack(fill=tk.X, padx=5, pady=5)
        self.package_info_text.insert(
            tk.END,
            "Load a model package to view its required input columns and preprocessing settings."
        )

        action_frame = ttk.Frame(self)
        action_frame.pack(fill=tk.X, padx=12, pady=8)

        self.evaluate_button = ttk.Button(
            action_frame,
            text="3. Run Evaluation / Prediction",
            command=self.start_evaluation,
            state=tk.DISABLED
        )
        self.evaluate_button.pack(side=tk.LEFT, padx=(0, 5))

        self.plot_eval_button = ttk.Button(
            action_frame,
            text="Show Confusion Matrix",
            command=self.show_evaluation_confusion_matrix,
            state=tk.DISABLED
        )
        self.plot_eval_button.pack(side=tk.LEFT, padx=5)

        self.export_prediction_button = ttk.Button(
            action_frame,
            text="Export Predictions CSV",
            command=self.export_predictions,
            state=tk.DISABLED
        )
        self.export_prediction_button.pack(side=tk.LEFT, padx=5)

        self.export_report_button = ttk.Button(
            action_frame,
            text="Save Evaluation Report",
            command=self.export_report,
            state=tk.DISABLED
        )
        self.export_report_button.pack(side=tk.LEFT, padx=5)

        self.save_complete_results_button = ttk.Button(
            action_frame,
            text="Save Complete Results",
            command=self.save_complete_results,
            state=tk.DISABLED,
        )
        self.save_complete_results_button.pack(side=tk.LEFT, padx=5)

        self.custom_results_button = ttk.Button(
            action_frame,
            text="Open Custom Results Studio",
            command=self.open_custom_results_studio,
            state=tk.DISABLED,
        )
        self.custom_results_button.pack(side=tk.LEFT, padx=5)

        self.evaluation_progress = ttk.Progressbar(
            action_frame,
            mode="indeterminate"
        )
        self.evaluation_progress.pack(
            side=tk.RIGHT,
            fill=tk.X,
            expand=True,
            padx=(20, 0)
        )

        pane = ttk.PanedWindow(self, orient=tk.VERTICAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

        preview_frame = ttk.LabelFrame(
            pane,
            text="4. Prediction Preview",
        )
        result_frame = ttk.LabelFrame(
            pane,
            text="5. Evaluation Log and Report",
        )
        pane.add(preview_frame, weight=2)
        pane.add(result_frame, weight=3)

        prediction_columns = (
            "position",
            "actual",
            "predicted",
            "confidence"
        )
        self.prediction_tree = ttk.Treeview(
            preview_frame,
            columns=prediction_columns,
            show="headings",
            height=8
        )
        headings = {
            "position": "Row / Window",
            "actual": "Actual",
            "predicted": "Predicted / Status",
            "confidence": "Confidence / Error"
        }
        widths = {
            "position": 220,
            "actual": 180,
            "predicted": 180,
            "confidence": 140
        }
        for column in prediction_columns:
            self.prediction_tree.heading(column, text=headings[column])
            self.prediction_tree.column(
                column,
                width=widths[column],
                anchor=tk.CENTER
            )

        tree_scroll = ttk.Scrollbar(
            preview_frame,
            orient=tk.VERTICAL,
            command=self.prediction_tree.yview
        )
        self.prediction_tree.configure(yscrollcommand=tree_scroll.set)
        self.prediction_tree.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=(5, 0),
            pady=5
        )
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 5), pady=5)

        self.eval_result_text = ScrolledText(result_frame, wrap=tk.WORD)
        self.eval_result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.eval_result_text.insert(tk.END, "Ready.\n")

    def log(self, message):
        self.eval_result_text.insert(tk.END, str(message) + "\n")
        self.eval_result_text.see(tk.END)

    def refresh_inference_devices(self):
        options = DeviceManager.tensorflow_options()
        if hasattr(self, "inference_device_combo"):
            self.inference_device_combo.config(values=options)
        if self.inference_device_var.get() not in options:
            self.inference_device_var.set(DEVICE_AUTO)
        self.update_inference_device_status()

    def update_inference_device_status(self):
        try:
            info = DeviceManager.resolve_tensorflow(
                self.inference_device_var.get()
            )
            self.inference_device_status_var.set(
                DeviceManager.format_status(info)
            )
        except Exception as exc:
            self.inference_device_status_var.set(str(exc))

    def close_window(self):
        if self.evaluation_running:
            close = messagebox.askyesno(
                "Evaluation Running",
                "Evaluation is still running. Close this window?",
                parent=self
            )
            if not close:
                return

        if self.package_data is not None:
            shutil.rmtree(
                self.package_data.get("temp_dir", ""),
                ignore_errors=True
            )
        for directory in self.custom_result_directories:
            shutil.rmtree(directory, ignore_errors=True)
        self.destroy()

    def reset_evaluation_results(self):
        self.prediction_df = None
        self.evaluation_report = None
        self.evaluation_cm = None
        self.evaluation_class_names = None
        self.evaluation_metrics = None

        for item in self.prediction_tree.get_children():
            self.prediction_tree.delete(item)

        self.plot_eval_button.config(state=tk.DISABLED)
        self.export_prediction_button.config(state=tk.DISABLED)
        self.export_report_button.config(state=tk.DISABLED)
        self.save_complete_results_button.config(state=tk.DISABLED)
        self.custom_results_button.config(state=tk.DISABLED)

    def load_package(self):
        package_path = filedialog.askopenfilename(
            parent=self,
            title="Select Saved Model Package",
            filetypes=[("ZIP Model Package", "*.zip")]
        )
        if not package_path:
            return
        try:
            model_type = detect_saved_model_type(package_path)
        except Exception as exc:
            messagebox.showerror(
                "Unsupported Model",
                str(exc),
                parent=self,
            )
            return
        if model_type != "keras_package":
            if not messagebox.askyesno(
                "Open Object Detection",
                "This is an object-detection model. Open it in the detector "
                "application workspace?",
                parent=self,
            ):
                return
            ObjectDetectionWindow(
                self.master,
                initial_model_path=package_path,
                application_mode=True,
            )
            return
        self.load_package_path(package_path, confirm_trust=True)

    def load_package_path(self, package_path, confirm_trust=True):
        if confirm_trust and not messagebox.askyesno(
            "Trusted Package",
            "Load this model package only if you created it or trust its source. "
            "Model packages contain serialized Python objects and may include "
            "custom Python filter code. Continue?",
            parent=self
        ):
            return

        try:
            self.config(cursor="watch")
            self.update_idletasks()

            old_package = self.package_data
            package_data = load_saved_model_package(package_path)
            self.package_data = package_data
            self.external_df = None
            self.external_csv_path = None
            self.external_image_directory = None
            self.data_label.config(text="No evaluation dataset loaded")

            if old_package is not None:
                shutil.rmtree(
                    old_package.get("temp_dir", ""),
                    ignore_errors=True
                )

            self.package_label.config(text=os.path.basename(package_path))
            self.export_original_results_button.config(
                state=(
                    tk.NORMAL
                    if package_data.get("training_results_directory")
                    else tk.DISABLED
                )
            )
            self.apply_saved_filters_var.set(True)
            self.reset_evaluation_results()
            self.update_package_information()
            self.update_label_choices()
            self.update_evaluate_button_state()
            data_mode = package_data["metadata"].get(
                "data_mode",
                DATA_MODE_TABULAR,
            )
            self.load_evaluation_data_button.config(
                text=(
                    "Load Evaluation Image Folder"
                    if data_mode == DATA_MODE_IMAGE
                    else "Load Evaluation CSV / Excel"
                )
            )

            messagebox.showinfo(
                "Model Loaded",
                "The model package and its preprocessing information were loaded successfully.",
                parent=self
            )

        except Exception as exc:
            messagebox.showerror(
                "Load Error",
                f"Failed to load the model package:\n{exc}",
                parent=self
            )
        finally:
            self.config(cursor="")

    def export_original_training_results(self):
        """Export the training evidence stored in a Version 22 model package."""
        if self.package_data is None:
            return
        results_directory = self.package_data.get(
            "training_results_directory"
        )
        if not results_directory or not os.path.isdir(results_directory):
            messagebox.showwarning(
                "No Stored Training Results",
                "This model package does not contain original training results.",
                parent=self,
            )
            return

        default_name = (
            Path(self.package_data["package_path"]).stem
            + "_original_training_results.zip"
        )
        save_path = filedialog.asksaveasfilename(
            parent=self,
            title="Export Original Training Results",
            initialfile=default_name,
            defaultextension=".zip",
            filetypes=[("ZIP Result Bundle", "*.zip")],
        )
        if not save_path:
            return

        try:
            zip_result_directory(results_directory, save_path)
            messagebox.showinfo(
                "Training Results Exported",
                f"Original training results were saved successfully:\n{save_path}",
                parent=self,
            )
        except Exception as exc:
            messagebox.showerror(
                "Export Error",
                f"Failed to export original training results:\n{exc}",
                parent=self,
            )

    def load_external_data(self):
        if (
            self.package_data is not None
            and self.package_data["metadata"].get("data_mode")
            == DATA_MODE_IMAGE
        ):
            self.load_external_image_folder()
        else:
            self.load_external_csv()

    def load_external_csv(self):
        csv_path = filedialog.askopenfilename(
            parent=self,
            title="Select Evaluation Dataset",
            filetypes=[
                ("Data Files", "*.csv *.xlsx *.xls"),
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx *.xls"),
            ],
        )
        if not csv_path:
            return

        try:
            suffix = Path(csv_path).suffix.lower()
            if suffix == ".csv":
                self.external_df = pd.read_csv(csv_path)
            elif suffix in (".xlsx", ".xls"):
                self.external_df = pd.read_excel(csv_path)
            else:
                raise ValueError("Choose a CSV, XLSX, or XLS file.")
            self.external_csv_path = csv_path
            self.external_image_directory = None
            self.data_label.config(
                text=(
                    f"{os.path.basename(csv_path)} | "
                    f"{len(self.external_df)} rows, "
                    f"{len(self.external_df.columns)} columns"
                )
            )
            self.reset_evaluation_results()
            self.update_label_choices()
            self.update_package_information()
            self.update_evaluate_button_state()

        except Exception as exc:
            messagebox.showerror(
                "CSV Load Error",
                f"Failed to load the evaluation CSV:\n{exc}",
                parent=self
            )

    def load_external_image_folder(self):
        directory = filedialog.askdirectory(
            parent=self,
            title=(
                "Select Evaluation Image Folder "
                "(direct subfolders are actual class names)"
            ),
        )
        if not directory:
            return
        try:
            records_df, _ = scan_image_dataset(
                directory,
                minimum_class_count=1,
                minimum_images_per_class=1,
            )
            self.external_df = records_df
            self.external_csv_path = None
            self.external_image_directory = str(Path(directory).resolve())
            self.data_label.config(
                text=(
                    f"{os.path.basename(directory)} | "
                    f"{len(records_df)} images, "
                    f"{records_df['class_name'].nunique()} folder label(s)"
                )
            )
            self.reset_evaluation_results()
            self.update_label_choices()
            self.update_package_information()
            self.update_evaluate_button_state()
        except Exception as exc:
            messagebox.showerror(
                "Image Folder Error",
                f"Failed to load the evaluation image folder:\n{exc}",
                parent=self,
            )

    def update_label_choices(self):
        if (
            self.package_data is not None
            and self.package_data["metadata"].get("data_mode")
            == DATA_MODE_IMAGE
        ):
            self.eval_label_combo.config(
                values=["Folder name (actual class)"],
                state=tk.DISABLED,
            )
            self.eval_label_var.set("Folder name (actual class)")
            return

        self.eval_label_combo.config(state="readonly")
        values = [self.NO_LABEL_TEXT]
        preferred_label = None

        if self.external_df is not None:
            values.extend(list(self.external_df.columns))

        if self.package_data is not None:
            metadata = self.package_data["metadata"]
            preferred_label = metadata.get("label_column")
            if not preferred_label:
                target_columns = metadata.get("target_columns", [])
                preferred_label = target_columns[0] if target_columns else None

        self.eval_label_combo.config(values=values)

        if (
            preferred_label
            and self.external_df is not None
            and preferred_label in self.external_df.columns
        ):
            self.eval_label_var.set(preferred_label)
        else:
            self.eval_label_var.set(self.NO_LABEL_TEXT)

    def update_evaluate_button_state(self):
        state = (
            tk.NORMAL
            if self.package_data is not None
            and self.external_df is not None
            and not self.evaluation_running
            else tk.DISABLED
        )
        self.evaluate_button.config(state=state)

    def update_package_information(self):
        self.package_info_text.delete("1.0", tk.END)

        if self.package_data is None:
            self.package_info_text.insert(
                tk.END,
                "Load a model package to view compatibility information."
            )
            return

        metadata = self.package_data["metadata"]
        data_mode = metadata.get("data_mode", DATA_MODE_TABULAR)
        included_results = metadata.get("included_training_results", {})
        results_message = (
            "- Original training results: Included in this package.\n"
            if included_results.get("available")
            else "- Original training results: Not included (older package).\n"
        )
        if data_mode == DATA_MODE_IMAGE:
            self.package_info_text.insert(
                tk.END,
                "Saved image model requirements:\n"
                f"- Task type: {metadata.get('task_type')}\n"
                f"- Model type: {metadata.get('model_type')}\n"
                f"- Image size: {metadata.get('image_height')} × "
                f"{metadata.get('image_width')}\n"
                f"- Colour mode: {metadata.get('image_color_mode')}\n"
                f"- Classes: {metadata.get('class_names', [])}\n"
                "- Preprocessing and normalization are stored inside the "
                "saved model.\n"
                + results_message
                + "\n"
            )
            if self.external_df is not None:
                package_classes = set(metadata.get("class_names", []))
                folder_classes = set(
                    self.external_df["class_name"].astype(str).unique()
                )
                unknown = sorted(folder_classes - package_classes)
                self.package_info_text.insert(
                    tk.END,
                    "Compatibility check:\n"
                    f"- Evaluation images: {len(self.external_df)}\n"
                    f"- Folder labels: {sorted(folder_classes)}\n"
                )
                if unknown:
                    self.package_info_text.insert(
                        tk.END,
                        "- Folder labels not present in the saved model will "
                        f"be predicted but excluded from metrics: {unknown}\n"
                    )
                else:
                    self.package_info_text.insert(
                        tk.END,
                        "- All folder labels match saved classes.\n"
                    )
            return

        feature_columns = metadata.get("feature_columns", [])
        label_column = metadata.get("label_column")
        target_columns = metadata.get(
            "target_columns",
            [label_column] if label_column else []
        )
        class_names = metadata.get("class_names", [])

        self.package_info_text.insert(tk.END, "Saved model requirements:\n")
        self.package_info_text.insert(
            tk.END,
            f"- Task type: {metadata.get('task_type', TASK_CLASSIFICATION)}\n"
        )
        self.package_info_text.insert(
            tk.END,
            f"- Model type: {metadata.get('model_type', 'Unknown')}\n"
        )
        self.package_info_text.insert(
            tk.END,
            f"- Required feature columns in exact order: {feature_columns}\n"
        )
        self.package_info_text.insert(
            tk.END,
            f"- Original target columns: {target_columns}\n"
        )
        self.package_info_text.insert(
            tk.END,
            f"- Classes: {class_names}\n"
        )
        self.package_info_text.insert(
            tk.END,
            f"- Normalization: {metadata.get('normalization', 'Unknown')}\n"
        )
        if metadata.get("target_normalization"):
            self.package_info_text.insert(
                tk.END,
                "- Target normalization: "
                f"{metadata.get('target_normalization')}\n"
            )
        self.package_info_text.insert(
            tk.END,
            f"- Built-in filter: {metadata.get('filter_method', 'No filter')}\n"
        )
        self.package_info_text.insert(
            tk.END,
            f"- Custom filter enabled: {metadata.get('custom_filter_enabled', False)}\n"
        )
        self.package_info_text.insert(
            tk.END,
            f"- Window-based input: {metadata.get('use_window', False)}\n"
        )
        self.package_info_text.insert(tk.END, results_message)
        if metadata.get("use_window", False):
            self.package_info_text.insert(
                tk.END,
                f"- Window size / stride: {metadata.get('window_size')} / "
                f"{metadata.get('stride')}\n"
            )

        if self.external_df is not None:
            missing_features = [
                col for col in feature_columns
                if col not in self.external_df.columns
            ]
            self.package_info_text.insert(tk.END, "\nCompatibility check:\n")
            if missing_features:
                self.package_info_text.insert(
                    tk.END,
                    f"- Missing required columns: {missing_features}\n"
                )
            else:
                self.package_info_text.insert(
                    tk.END,
                    "- All required feature columns are available.\n"
                )

            extra_columns = [
                col for col in self.external_df.columns
                if col not in feature_columns
            ]
            self.package_info_text.insert(
                tk.END,
                f"- Additional columns will be ignored except the selected label: "
                f"{extra_columns}\n"
            )

    def start_evaluation(self):
        if self.package_data is None or self.external_df is None:
            messagebox.showwarning(
                "Missing Input",
                "Load both a model package and compatible evaluation data "
                "first.",
                parent=self
            )
            return

        try:
            device_info = DeviceManager.resolve_tensorflow(
                self.inference_device_var.get()
            )
        except Exception as exc:
            messagebox.showerror(
                "Inference Device",
                str(exc),
                parent=self,
            )
            return
        self.log(DeviceManager.format_status(device_info))

        metadata = self.package_data["metadata"]
        if metadata.get("data_mode") == DATA_MODE_IMAGE:
            self.evaluation_running = True
            self.reset_evaluation_results()
            self.evaluate_button.config(state=tk.DISABLED)
            self.evaluation_progress.start(10)
            self.eval_result_text.delete("1.0", tk.END)
            self.log("Image evaluation started...")
            package_snapshot = dict(self.package_data)
            package_snapshot["inference_device_selection"] = (
                self.inference_device_var.get()
            )
            threading.Thread(
                target=self.evaluation_worker_image,
                args=(
                    package_snapshot,
                    self.external_df.copy(),
                ),
                daemon=True,
            ).start()
            return

        required_features = metadata.get("feature_columns", [])
        missing_features = [
            col for col in required_features
            if col not in self.external_df.columns
        ]
        if missing_features:
            messagebox.showerror(
                "Incompatible Dataset",
                "The evaluation dataset is missing required feature columns:\n"
                + str(missing_features),
                parent=self
            )
            return

        label_column = self.eval_label_var.get()
        if label_column == self.NO_LABEL_TEXT:
            label_column = None

        self.evaluation_running = True
        self.reset_evaluation_results()
        self.evaluate_button.config(state=tk.DISABLED)
        self.evaluation_progress.start(10)
        self.eval_result_text.delete("1.0", tk.END)
        self.log("Evaluation started...")

        package_snapshot = dict(self.package_data)
        package_snapshot["inference_device_selection"] = (
            self.inference_device_var.get()
        )
        dataframe_snapshot = self.external_df.copy()
        apply_filters = bool(self.apply_saved_filters_var.get())

        thread = threading.Thread(
            target=self.evaluation_worker,
            args=(
                package_snapshot,
                dataframe_snapshot,
                label_column,
                apply_filters
            ),
            daemon=True
        )
        thread.start()

    def worker_log(self, message):
        self.worker_queue.put(("log", str(message)))

    def evaluation_worker_image(self, package_data, image_records):
        """Run a saved image classifier on another class-organized folder."""
        try:
            metadata = package_data["metadata"]
            model = package_data["model"]
            class_names = list(metadata.get("class_names", []))
            if len(class_names) < 2:
                raise ValueError(
                    "The image package does not contain a valid class mapping."
                )

            image_height = int(metadata.get("image_height") or 224)
            image_width = int(metadata.get("image_width") or 224)
            color_mode = metadata.get("image_color_mode", "RGB")
            batch_size = int(metadata.get("batch_size") or 32)
            dataset = make_image_tf_dataset(
                image_records,
                image_height=image_height,
                image_width=image_width,
                color_mode=color_mode,
                batch_size=batch_size,
                shuffle=False,
            )
            self.worker_log(
                f"Prepared {len(image_records)} images at "
                f"{image_height} × {image_width} ({color_mode})."
            )
            device_selection = package_data.get(
                "inference_device_selection",
                DEVICE_AUTO,
            )
            self.worker_log(
                DeviceManager.format_status(
                    DeviceManager.resolve_tensorflow(device_selection)
                )
            )
            probabilities = np.asarray(
                call_on_tensorflow_device(
                    device_selection,
                    model.predict,
                    dataset,
                    verbose=0,
                )
            )
            if probabilities.ndim != 2:
                raise ValueError(
                    "The loaded image model returned an unexpected output "
                    f"shape: {probabilities.shape}"
                )
            if probabilities.shape[1] != len(class_names):
                raise ValueError(
                    "The model output count does not match the saved class "
                    "mapping."
                )

            predicted_indices = np.argmax(probabilities, axis=1)
            predicted_labels = np.asarray(class_names)[predicted_indices]
            confidence = np.max(probabilities, axis=1)
            actual_labels = image_records[
                "class_name"
            ].astype(str).to_numpy()
            class_to_index = {
                name: index for index, name in enumerate(class_names)
            }
            actual_indices = np.asarray(
                [
                    class_to_index.get(label, -1)
                    for label in actual_labels
                ],
                dtype=int,
            )
            known_mask = actual_indices >= 0

            result_data = {
                "record_number": np.arange(1, len(image_records) + 1),
                "source_file": image_records["file_path"].astype(str).to_numpy(),
                "actual_folder_label": actual_labels,
                "predicted_class_index": predicted_indices,
                "predicted_label": predicted_labels,
                "confidence": confidence,
            }
            for index, class_name in enumerate(class_names):
                result_data[
                    make_probability_column_name(index, class_name)
                ] = probabilities[:, index]

            cm = None
            metrics = None
            if np.any(known_mask):
                labels = np.arange(len(class_names))
                known_actual = actual_indices[known_mask]
                known_predicted = predicted_indices[known_mask]
                accuracy = accuracy_score(known_actual, known_predicted)
                weighted_f1 = f1_score(
                    known_actual,
                    known_predicted,
                    average="weighted",
                    zero_division=0,
                )
                report_text = classification_report(
                    known_actual,
                    known_predicted,
                    labels=labels,
                    target_names=class_names,
                    zero_division=0,
                )
                cm = confusion_matrix(
                    known_actual,
                    known_predicted,
                    labels=labels,
                )
                metrics = {
                    "accuracy": float(accuracy),
                    "weighted_f1": float(weighted_f1),
                    "evaluated_images": int(np.sum(known_mask)),
                    "prediction_only_images": int(np.sum(~known_mask)),
                }
                self.worker_log(
                    f"Accuracy: {accuracy * 100:.2f}% | "
                    f"Weighted F1: {weighted_f1 * 100:.2f}%"
                )
            else:
                report_text = (
                    "Image prediction completed. None of the evaluation "
                    "folder names matched saved class names, so accuracy and "
                    "F1 metrics were not calculated."
                )

            self.worker_log("\n" + report_text)
            prediction_df = pd.DataFrame(result_data)
            self.worker_queue.put(("finished", {
                "prediction_df": prediction_df,
                "positions": [
                    str(Path(path).name)
                    for path in image_records["file_path"]
                ],
                "actual_labels": actual_labels,
                "predicted_labels": predicted_labels,
                "confidence": confidence,
                "report_text": report_text,
                "confusion_matrix": cm,
                "class_names": class_names,
                "metrics": metrics,
            }))
        except Exception:
            self.worker_queue.put(("failed", traceback.format_exc()))

    def evaluation_worker_v7_classification_only(
        self,
        package_data,
        external_df,
        label_column,
        apply_filters
    ):
        try:
            metadata = package_data["metadata"]
            model = package_data["model"]
            scaler = package_data["scaler"]
            label_encoder = package_data["label_encoder"]
            custom_filter_code = package_data.get("custom_filter_code")
            safe_filter_spec = (
                package_data.get("safe_filter_spec")
                or metadata.get("safe_filter_spec")
            )

            feature_columns = list(metadata.get("feature_columns", []))
            if not feature_columns:
                raise ValueError("The package metadata has no feature columns.")

            working_df = external_df.copy()
            self.worker_log(
                f"Loaded external data: {len(working_df)} rows, "
                f"{len(working_df.columns)} columns"
            )
            self.worker_log(f"Required input columns: {feature_columns}")

            if apply_filters:
                filter_method = metadata.get("filter_method", "No filter")
                filter_columns = [
                    col for col in metadata.get("filtered_columns", [])
                    if col in working_df.columns
                ]

                self.worker_log(
                    f"Applying saved built-in filter: {filter_method} "
                    f"to {filter_columns}"
                )
                working_df = apply_selected_filter(
                    df=working_df,
                    selected_filter_cols=filter_columns,
                    filter_method=filter_method,
                    moving_window=int(metadata.get("moving_average_window", 5)),
                    ema_span=int(metadata.get("ema_span", 10)),
                    median_window=int(metadata.get("median_window", 5)),
                    kalman_q=float(metadata.get("kalman_q", 1e-5)),
                    kalman_r=float(metadata.get("kalman_r", 1e-2))
                )

                if metadata.get("custom_filter_enabled", False):
                    custom_columns = [
                        col for col in metadata.get("custom_filter_columns", [])
                        if col in working_df.columns
                    ]
                    self.worker_log(
                        f"Applying saved custom filter to {custom_columns}"
                    )
                    if (
                        metadata.get("custom_filter_mode") in (
                            CUSTOM_FILTER_MODE_SAFE_AI,
                            CUSTOM_FILTER_MODE_SAFE_AI_LEGACY,
                        )
                    ):
                        if not safe_filter_spec:
                            raise ValueError(
                                "The saved safe filter specification is missing."
                            )
                        analysis = metadata.get("dataset_analysis") or {}
                        working_df = apply_safe_filter_spec(
                            df=working_df,
                            filter_spec=safe_filter_spec,
                            sampling_frequency=analysis.get(
                                "sampling_frequency_hz"
                            ),
                            allowed_columns=feature_columns,
                        )
                    else:
                        if not custom_filter_code:
                            raise ValueError(
                                "The metadata requires a custom filter, but "
                                "custom_filter.py is missing from the package."
                            )
                        working_df = execute_custom_filter_code(
                            df=working_df,
                            selected_columns=custom_columns,
                            custom_code=custom_filter_code
                        )
            else:
                self.worker_log(
                    "Saved filters skipped. Use this only when the new CSV "
                    "already contains equivalently filtered signals."
                )

            working_df = convert_features_to_numeric(
                working_df,
                feature_columns
            )

            missing_method = metadata.get(
                "missing_value_method",
                "Drop rows"
            )

            if label_column is not None:
                if label_column not in working_df.columns:
                    raise ValueError(
                        f"Selected label column '{label_column}' does not exist."
                    )
                working_df = handle_missing_values(
                    df=working_df,
                    feature_cols=feature_columns,
                    label_col=label_column,
                    method=missing_method
                )
            else:
                working_df = handle_missing_features_only(
                    df=working_df,
                    feature_cols=feature_columns,
                    method=missing_method
                )

            if working_df.empty:
                raise ValueError(
                    "No rows remain after applying missing-value processing."
                )

            row_ids = working_df.index.to_numpy()
            X = working_df[feature_columns].to_numpy(dtype=float)

            y_encoded = None
            if label_column is not None:
                y_encoded = encode_external_labels(
                    working_df[label_column].to_numpy(),
                    label_encoder
                )

            if scaler is not None:
                self.worker_log("Applying the scaler fitted during training...")
                X = scaler.transform(X)
            else:
                self.worker_log("The saved package does not use normalization.")

            use_window = bool(metadata.get("use_window", False))

            if use_window:
                window_size = int(metadata.get("window_size"))
                stride = int(metadata.get("stride"))

                if len(X) < window_size:
                    raise ValueError(
                        f"The external dataset has {len(X)} usable rows, but "
                        f"the model requires a window size of {window_size}."
                    )

                X_model, y_model, start_rows, end_rows = create_inference_windows(
                    X=X,
                    y=y_encoded,
                    window_size=window_size,
                    stride=stride,
                    row_ids=row_ids
                )
                if len(X_model) == 0:
                    raise ValueError("No evaluation windows could be created.")

                positions = [
                    f"Window {index + 1}: rows {start}–{end}"
                    for index, (start, end) in enumerate(
                        zip(start_rows, end_rows)
                    )
                ]
            else:
                X_model = X
                y_model = y_encoded
                start_rows = row_ids
                end_rows = row_ids
                positions = [f"Row {row}" for row in row_ids]

            self.worker_log(f"Prepared model input shape: {X_model.shape}")
            validate_model_input_shape(model, X_model)

            self.worker_log("Running model prediction...")
            device_selection = package_data.get(
                "inference_device_selection",
                DEVICE_AUTO,
            )
            self.worker_log(
                DeviceManager.format_status(
                    DeviceManager.resolve_tensorflow(device_selection)
                )
            )
            y_probability_raw = np.asarray(
                call_on_tensorflow_device(
                    device_selection,
                    model.predict,
                    X_model,
                    verbose=0,
                )
            )

            output_units = int(metadata.get("output_units", 0))
            if output_units <= 0:
                output_units = (
                    1
                    if y_probability_raw.ndim == 1
                    else int(y_probability_raw.shape[-1])
                )

            y_pred = prediction_to_class(
                y_probability_raw,
                output_units
            )

            class_values = np.asarray(label_encoder.classes_)
            if np.any(y_pred < 0) or np.any(y_pred >= len(class_values)):
                raise ValueError(
                    "The model produced a class index outside the saved label range."
                )
            predicted_labels = class_values[y_pred]

            if output_units == 1:
                positive_probability = y_probability_raw.reshape(-1)
                probabilities = np.column_stack([
                    1.0 - positive_probability,
                    positive_probability
                ])
            else:
                probabilities = y_probability_raw

            if probabilities.ndim != 2:
                raise ValueError(
                    f"Unexpected prediction output shape: {probabilities.shape}"
                )

            confidence = np.max(probabilities, axis=1)

            result_data = {
                "record_number": np.arange(1, len(y_pred) + 1),
                "source_start_row": start_rows,
                "source_end_row": end_rows,
                "predicted_class_index": y_pred,
                "predicted_label": predicted_labels,
                "confidence": confidence,
            }

            actual_labels = None
            if y_model is not None:
                actual_labels = class_values[y_model]
                result_data["actual_class_index"] = y_model
                result_data["actual_label"] = actual_labels

            for class_index, class_name in enumerate(class_values):
                if class_index < probabilities.shape[1]:
                    result_data[
                        make_probability_column_name(class_index, class_name)
                    ] = probabilities[:, class_index]

            prediction_df = pd.DataFrame(result_data)

            metrics = None
            report_text = None
            cm = None
            class_names = [str(value) for value in class_values]

            if y_model is not None:
                labels = np.arange(len(class_names))
                accuracy = accuracy_score(y_model, y_pred)
                weighted_f1 = f1_score(
                    y_model,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
                report_text = classification_report(
                    y_model,
                    y_pred,
                    labels=labels,
                    target_names=class_names,
                    zero_division=0
                )
                cm = confusion_matrix(
                    y_model,
                    y_pred,
                    labels=labels
                )
                metrics = {
                    "accuracy": float(accuracy),
                    "weighted_f1": float(weighted_f1),
                    "number_of_records": int(len(y_pred)),
                }

                self.worker_log("\nExternal evaluation completed.")
                self.worker_log("=" * 60)
                self.worker_log(f"Accuracy: {accuracy * 100:.2f}%")
                self.worker_log(
                    f"Weighted F1-score: {weighted_f1 * 100:.2f}%"
                )
                self.worker_log("=" * 60)
                self.worker_log("\nClassification Report:")
                self.worker_log(report_text)
                self.worker_log("\nConfusion Matrix:")
                self.worker_log(cm)
            else:
                self.worker_log("\nPrediction completed without actual labels.")
                self.worker_log(
                    f"Generated {len(prediction_df)} prediction record(s)."
                )

            payload = {
                "prediction_df": prediction_df,
                "positions": positions,
                "actual_labels": actual_labels,
                "predicted_labels": predicted_labels,
                "confidence": confidence,
                "report_text": report_text,
                "confusion_matrix": cm,
                "class_names": class_names,
                "metrics": metrics,
            }
            self.worker_queue.put(("finished", payload))

        except Exception:
            self.worker_queue.put(("failed", traceback.format_exc()))

    def evaluation_worker(
        self,
        package_data,
        external_df,
        label_column,
        apply_filters
    ):
        """Run task-aware prediction and optional evaluation on a new CSV."""
        try:
            metadata = package_data["metadata"]
            task_type = metadata.get("task_type", TASK_CLASSIFICATION)
            model = package_data["model"]
            scaler = package_data["scaler"]
            target_scaler = package_data.get("target_scaler")
            label_encoder = package_data.get("label_encoder")
            custom_filter_code = package_data.get("custom_filter_code")
            safe_filter_spec = (
                package_data.get("safe_filter_spec")
                or metadata.get("safe_filter_spec")
            )

            feature_columns = list(metadata.get("feature_columns", []))
            saved_target_columns = list(metadata.get(
                "target_columns",
                [metadata.get("label_column")]
                if metadata.get("label_column")
                else []
            ))
            if not feature_columns:
                raise ValueError("The package metadata has no feature columns.")

            working_df = external_df.copy()
            self.worker_log(f"Task type: {task_type}")
            self.worker_log(f"Required inputs: {feature_columns}")

            if apply_filters:
                filter_method = metadata.get("filter_method", "No filter")
                filter_columns = [
                    col for col in metadata.get("filtered_columns", [])
                    if col in working_df.columns
                ]
                working_df = apply_selected_filter(
                    df=working_df,
                    selected_filter_cols=filter_columns,
                    filter_method=filter_method,
                    moving_window=int(metadata.get("moving_average_window", 5)),
                    ema_span=int(metadata.get("ema_span", 10)),
                    median_window=int(metadata.get("median_window", 5)),
                    kalman_q=float(metadata.get("kalman_q", 1e-5)),
                    kalman_r=float(metadata.get("kalman_r", 1e-2))
                )
                self.worker_log(
                    f"Applied saved built-in filter: {filter_method}"
                )

                if metadata.get("custom_filter_enabled", False):
                    custom_columns = [
                        col for col in metadata.get(
                            "custom_filter_columns",
                            []
                        )
                        if col in working_df.columns
                    ]
                    if (
                        metadata.get("custom_filter_mode") in (
                            CUSTOM_FILTER_MODE_SAFE_AI,
                            CUSTOM_FILTER_MODE_SAFE_AI_LEGACY,
                        )
                    ):
                        if not safe_filter_spec:
                            raise ValueError(
                                "The saved safe filter specification is missing."
                            )
                        analysis = metadata.get("dataset_analysis") or {}
                        working_df = apply_safe_filter_spec(
                            working_df,
                            safe_filter_spec,
                            sampling_frequency=analysis.get(
                                "sampling_frequency_hz"
                            ),
                            allowed_columns=feature_columns,
                        )
                        self.worker_log("Applied the saved safe AI filter.")
                    else:
                        if not custom_filter_code:
                            raise ValueError(
                                "The saved custom filter code is missing."
                            )
                        working_df = execute_custom_filter_code(
                            working_df,
                            custom_columns,
                            custom_filter_code
                        )
                        self.worker_log("Applied the saved custom filter.")

            working_df = convert_features_to_numeric(
                working_df,
                feature_columns
            )

            actual_target_columns = []
            if label_column is not None and task_type != TASK_AUTOENCODER:
                if task_type in (TASK_CLASSIFICATION, TASK_REGRESSION):
                    actual_target_columns = [label_column]
                else:
                    missing_saved_targets = [
                        col for col in saved_target_columns
                        if col not in working_df.columns
                    ]
                    if missing_saved_targets:
                        raise ValueError(
                            "Evaluation of all saved outputs requires these "
                            f"target columns: {missing_saved_targets}"
                        )
                    actual_target_columns = saved_target_columns

            if task_type in (
                TASK_REGRESSION,
                TASK_MULTI_OUTPUT,
                TASK_FORECASTING
            ):
                for col in actual_target_columns:
                    working_df[col] = pd.to_numeric(
                        working_df[col],
                        errors="coerce"
                    )

            missing_method = metadata.get(
                "missing_value_method",
                "Drop rows"
            )
            if actual_target_columns:
                working_df = handle_missing_values_multi(
                    working_df,
                    feature_columns,
                    actual_target_columns,
                    missing_method
                )
            else:
                working_df = handle_missing_features_only(
                    working_df,
                    feature_columns,
                    missing_method
                )
            if working_df.empty:
                raise ValueError(
                    "No usable rows remain after missing-value processing."
                )

            row_ids = working_df.index.to_numpy()
            X = working_df[feature_columns].to_numpy(dtype=float)
            if scaler is not None:
                X = scaler.transform(X)

            y_values = None
            if actual_target_columns:
                if task_type == TASK_CLASSIFICATION:
                    if label_encoder is None:
                        raise ValueError(
                            "Classification label encoder is missing."
                        )
                    y_values = encode_external_labels(
                        working_df[actual_target_columns[0]].to_numpy(),
                        label_encoder
                    )
                else:
                    y_values = working_df[
                        actual_target_columns
                    ].to_numpy(dtype=float)
                    if target_scaler is not None:
                        y_values = target_scaler.transform(y_values)

            use_window = bool(metadata.get("use_window", False))
            window_size = int(metadata.get("window_size") or 1)
            stride = int(metadata.get("stride") or 1)
            forecast_horizon = int(
                metadata.get("forecast_horizon") or 1
            )

            if use_window:
                if task_type == TASK_FORECASTING:
                    final_start = (
                        len(X) - window_size - forecast_horizon + 1
                    )
                else:
                    final_start = len(X) - window_size + 1
                if final_start <= 0:
                    raise ValueError(
                        "The external dataset is too short for the saved "
                        "window and forecast horizon."
                    )

                X_records = []
                y_records = []
                start_rows = []
                end_rows = []
                for start in range(0, final_start, stride):
                    end = start + window_size
                    X_window = X[start:end]
                    X_records.append(X_window)
                    start_rows.append(row_ids[start])
                    end_rows.append(row_ids[end - 1])

                    if task_type == TASK_AUTOENCODER:
                        y_records.append(X_window.copy())
                    elif y_values is not None:
                        if task_type == TASK_CLASSIFICATION:
                            y_records.append(
                                pd.Series(
                                    y_values[start:end]
                                ).mode().iloc[0]
                            )
                        elif task_type in (
                            TASK_REGRESSION,
                            TASK_MULTI_OUTPUT
                        ):
                            y_records.append(y_values[end - 1])
                        elif task_type == TASK_FORECASTING:
                            y_records.append(
                                y_values[
                                    end:end + forecast_horizon
                                ].reshape(-1)
                            )

                X_model = np.asarray(X_records)
                y_model = (
                    np.asarray(y_records)
                    if y_records
                    else None
                )
                start_rows = np.asarray(start_rows)
                end_rows = np.asarray(end_rows)
                positions = [
                    f"Window {i + 1}: rows {start}–{end}"
                    for i, (start, end) in enumerate(
                        zip(start_rows, end_rows)
                    )
                ]
            else:
                X_model = X
                y_model = X.copy() if task_type == TASK_AUTOENCODER else y_values
                start_rows = row_ids
                end_rows = row_ids
                positions = [f"Row {row}" for row in row_ids]

            validate_model_input_shape(model, X_model)
            self.worker_log(f"Prepared model input: {X_model.shape}")
            device_selection = package_data.get(
                "inference_device_selection",
                DEVICE_AUTO,
            )
            self.worker_log(
                DeviceManager.format_status(
                    DeviceManager.resolve_tensorflow(device_selection)
                )
            )
            raw_prediction = np.asarray(
                call_on_tensorflow_device(
                    device_selection,
                    model.predict,
                    X_model,
                    verbose=0,
                )
            )

            result_data = {
                "record_number": np.arange(1, len(X_model) + 1),
                "source_start_row": start_rows,
                "source_end_row": end_rows,
            }
            report_text = None
            cm = None
            metrics = None
            class_names = []

            if task_type == TASK_CLASSIFICATION:
                output_units = int(metadata.get("output_units") or 1)
                y_pred = prediction_to_class(
                    raw_prediction,
                    output_units
                )
                class_values = np.asarray(label_encoder.classes_)
                predicted_labels = class_values[y_pred]

                if output_units == 1:
                    positive_probability = raw_prediction.reshape(-1)
                    probabilities = np.column_stack([
                        1.0 - positive_probability,
                        positive_probability
                    ])
                else:
                    probabilities = raw_prediction
                confidence = np.max(probabilities, axis=1)

                result_data["predicted_class_index"] = y_pred
                result_data["predicted_label"] = predicted_labels
                result_data["confidence"] = confidence

                actual_labels = None
                if y_model is not None:
                    actual_labels = class_values[
                        np.asarray(y_model, dtype=int)
                    ]
                    result_data["actual_class_index"] = y_model
                    result_data["actual_label"] = actual_labels

                for index, class_name in enumerate(class_values):
                    if index < probabilities.shape[1]:
                        result_data[
                            make_probability_column_name(index, class_name)
                        ] = probabilities[:, index]

                class_names = [str(value) for value in class_values]
                if y_model is not None:
                    labels = np.arange(len(class_names))
                    accuracy = accuracy_score(y_model, y_pred)
                    weighted_f1 = f1_score(
                        y_model,
                        y_pred,
                        average="weighted",
                        zero_division=0
                    )
                    report_text = classification_report(
                        y_model,
                        y_pred,
                        labels=labels,
                        target_names=class_names,
                        zero_division=0
                    )
                    cm = confusion_matrix(y_model, y_pred, labels=labels)
                    metrics = {
                        "accuracy": float(accuracy),
                        "weighted_f1": float(weighted_f1),
                        "number_of_records": int(len(y_pred))
                    }

            else:
                if task_type == TASK_AUTOENCODER:
                    predicted_numeric = inverse_transform_targets(
                        raw_prediction,
                        scaler
                    )
                    actual_numeric = inverse_transform_targets(
                        y_model,
                        scaler
                    )
                    anomaly_scores = np.mean(
                        np.abs(raw_prediction - y_model),
                        axis=tuple(range(1, raw_prediction.ndim))
                    )
                    threshold = metadata.get("anomaly_threshold")
                    if threshold is None:
                        threshold = float(np.percentile(
                            anomaly_scores,
                            float(metadata.get("anomaly_percentile") or 95)
                        ))
                    is_anomaly = anomaly_scores > float(threshold)
                    result_data["anomaly_score"] = anomaly_scores
                    result_data["is_anomaly"] = is_anomaly
                    confidence = anomaly_scores
                    predicted_labels = np.where(
                        is_anomaly,
                        "Anomaly",
                        "Normal"
                    )
                    actual_labels = np.repeat("Input reconstruction", len(X_model))
                    metrics = regression_metrics(
                        actual_numeric,
                        predicted_numeric
                    )
                    metrics.update({
                        "anomaly_threshold_scaled_mae": float(threshold),
                        "anomaly_count": int(np.sum(is_anomaly)),
                        "anomaly_rate": float(np.mean(is_anomaly)),
                        "number_of_records": int(len(X_model))
                    })
                    report_text = (
                        "Autoencoder external reconstruction evaluation\n"
                        + "\n".join(
                            f"{key}: {value}"
                            for key, value in metrics.items()
                        )
                    )
                else:
                    predicted_numeric = inverse_transform_targets(
                        raw_prediction,
                        target_scaler
                    )
                    actual_numeric = (
                        inverse_transform_targets(y_model, target_scaler)
                        if y_model is not None
                        else None
                    )
                    if task_type == TASK_FORECASTING:
                        output_names = [
                            f"{target}[t+{step}]"
                            for step in range(1, forecast_horizon + 1)
                            for target in saved_target_columns
                        ]
                    else:
                        output_names = saved_target_columns

                    pred_2d = predicted_numeric.reshape(
                        len(predicted_numeric),
                        -1
                    )
                    for index in range(pred_2d.shape[1]):
                        name = (
                            output_names[index]
                            if index < len(output_names)
                            else f"output_{index + 1}"
                        )
                        safe_name = "".join(
                            char if char.isalnum() or char == "_" else "_"
                            for char in name
                        )
                        result_data[f"predicted_{safe_name}"] = pred_2d[:, index]

                    actual_labels = None
                    if actual_numeric is not None:
                        actual_2d = actual_numeric.reshape(
                            len(actual_numeric),
                            -1
                        )
                        sample_error = np.mean(
                            np.abs(actual_2d - pred_2d),
                            axis=1
                        )
                        confidence = sample_error
                        for index in range(actual_2d.shape[1]):
                            name = (
                                output_names[index]
                                if index < len(output_names)
                                else f"output_{index + 1}"
                            )
                            safe_name = "".join(
                                char if char.isalnum() or char == "_" else "_"
                                for char in name
                            )
                            result_data[
                                f"actual_{safe_name}"
                            ] = actual_2d[:, index]
                        metrics = regression_metrics(
                            actual_numeric,
                            predicted_numeric
                        )
                        metrics["number_of_records"] = int(len(pred_2d))
                        report_text = (
                            f"{task_type} external evaluation\n"
                            + "\n".join(
                                f"{key}: {value}"
                                for key, value in metrics.items()
                            )
                        )
                        actual_labels = [
                            np.array2string(row, precision=5)
                            for row in actual_2d
                        ]
                    else:
                        confidence = np.full(len(pred_2d), np.nan)
                        report_text = (
                            f"{task_type} prediction completed without "
                            "actual target values."
                        )

                    predicted_labels = [
                        np.array2string(row, precision=5)
                        for row in pred_2d
                    ]

            prediction_df = pd.DataFrame(result_data)
            self.worker_log("\n" + (report_text or "Prediction completed."))

            self.worker_queue.put(("finished", {
                "prediction_df": prediction_df,
                "positions": positions,
                "actual_labels": actual_labels,
                "predicted_labels": predicted_labels,
                "confidence": confidence,
                "report_text": report_text,
                "confusion_matrix": cm,
                "class_names": class_names,
                "metrics": metrics,
            }))

        except Exception:
            self.worker_queue.put(("failed", traceback.format_exc()))

    def process_worker_queue(self):
        try:
            while True:
                event_type, payload = self.worker_queue.get_nowait()
                if event_type == "log":
                    self.log(payload)
                elif event_type == "finished":
                    self.finish_evaluation(payload)
                elif event_type == "failed":
                    self.fail_evaluation(payload)
        except queue.Empty:
            pass

        if self.winfo_exists():
            self.after(100, self.process_worker_queue)

    def finish_evaluation(self, payload):
        self.evaluation_running = False
        self.evaluation_progress.stop()
        self.update_evaluate_button_state()

        self.prediction_df = payload["prediction_df"]
        self.evaluation_report = payload["report_text"]
        self.evaluation_cm = payload["confusion_matrix"]
        self.evaluation_class_names = payload["class_names"]
        self.evaluation_metrics = payload["metrics"]

        positions = payload["positions"]
        actual_labels = payload["actual_labels"]
        predicted_labels = payload["predicted_labels"]
        confidence = payload["confidence"]

        preview_limit = min(1000, len(predicted_labels))
        for index in range(preview_limit):
            actual = "—" if actual_labels is None else actual_labels[index]
            self.prediction_tree.insert(
                "",
                tk.END,
                values=(
                    positions[index],
                    actual,
                    predicted_labels[index],
                    f"{float(confidence[index]):.6f}"
                )
            )

        if len(predicted_labels) > preview_limit:
            self.log(
                f"\nPreview limited to the first {preview_limit} records. "
                "The exported CSV contains all predictions."
            )

        if self.evaluation_cm is not None:
            self.plot_eval_button.config(state=tk.NORMAL)
        if self.evaluation_report is not None:
            self.export_report_button.config(state=tk.NORMAL)

        self.export_prediction_button.config(state=tk.NORMAL)
        self.save_complete_results_button.config(state=tk.NORMAL)
        self.custom_results_button.config(state=tk.NORMAL)

        messagebox.showinfo(
            "Evaluation Completed",
            "The saved model was successfully applied to the new dataset.",
            parent=self
        )

    def fail_evaluation(self, error_text):
        self.evaluation_running = False
        self.evaluation_progress.stop()
        self.update_evaluate_button_state()
        self.log("\nEvaluation failed.\n")
        self.log(error_text)
        messagebox.showerror(
            "Evaluation Failed",
            "Evaluation failed. Check the report area for details.",
            parent=self
        )

    def get_custom_results_context(self):
        """Expose copied, model-aware result tables to the shared studio."""
        if self.prediction_df is None or self.package_data is None:
            raise ValueError(
                "Run evaluation or prediction before opening Custom Results."
            )
        metadata = dict(self.package_data.get("metadata") or {})
        tables = {"results": self.prediction_df.copy()}
        if self.external_df is not None:
            tables["input_data"] = self.external_df.copy()

        results_directory = self.package_data.get(
            "training_results_directory"
        )
        if results_directory and Path(results_directory).is_dir():
            history_candidates = list(
                Path(results_directory).rglob("training_history.csv")
            )
            if history_candidates:
                try:
                    tables["training_history"] = pd.read_csv(
                        history_candidates[0]
                    )
                except Exception:
                    pass
        if self.evaluation_cm is not None:
            class_names = list(self.evaluation_class_names or [])
            matrix = np.asarray(self.evaluation_cm)
            if len(class_names) != len(matrix):
                class_names = [str(index) for index in range(len(matrix))]
            tables["confusion_matrix"] = pd.DataFrame(
                matrix,
                index=class_names,
                columns=class_names,
            ).reset_index(names="actual_label")

        source = (
            self.external_image_directory
            or self.external_csv_path
            or "External evaluation data"
        )
        return {
            "title": "Loaded Model Evaluation Results",
            "task_type": metadata.get(
                "task_type",
                TASK_CLASSIFICATION,
            ),
            "model_type": metadata.get("model_type", "Keras model"),
            "tables": tables,
            "metrics": self.evaluation_metrics or {},
            "metadata": metadata,
            "class_names": self.evaluation_class_names or [],
            "source": source,
        }

    def attach_custom_result(self, source_directory, _recipe=None):
        """Keep an independent copy for the next complete-result export."""
        destination = tempfile.mkdtemp(prefix="nn_attached_custom_result_")
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(source_directory, destination)
        self.custom_result_directories.append(destination)
        self.log(
            "\nCustom visualization attached: "
            + Path(destination).name
        )

    def open_custom_results_studio(self):
        CustomResultsStudioWindow(
            self,
            context_provider=self.get_custom_results_context,
            settings_owner=self.master,
            attach_callback=self.attach_custom_result,
        )

    def show_evaluation_confusion_matrix(self):
        if self.evaluation_cm is None:
            messagebox.showwarning(
                "No Confusion Matrix",
                "Provide compatible actual labels and run evaluation first.",
                parent=self
            )
            return

        cm = self.evaluation_cm
        class_names = self.evaluation_class_names

        plt.figure()
        plt.imshow(cm)
        plt.title("External Dataset Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")
        plt.colorbar()

        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        for row in range(len(class_names)):
            for column in range(len(class_names)):
                plt.text(
                    column,
                    row,
                    cm[row, column],
                    ha="center",
                    va="center"
                )

        plt.tight_layout()
        plt.show()

    def export_predictions(self):
        if self.prediction_df is None:
            return

        save_path = filedialog.asksaveasfilename(
            parent=self,
            title="Export Model Predictions",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")]
        )
        if not save_path:
            return

        try:
            self.prediction_df.to_csv(save_path, index=False)
            messagebox.showinfo(
                "Predictions Exported",
                f"Predictions saved successfully:\n{save_path}",
                parent=self
            )
        except Exception as exc:
            messagebox.showerror(
                "Export Error",
                f"Failed to export predictions:\n{exc}",
                parent=self
            )

    def export_report(self):
        if self.evaluation_report is None:
            return

        save_path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Evaluation Report",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt")]
        )
        if not save_path:
            return

        try:
            metadata = dict(self.package_data["metadata"])
            metadata["evaluation_execution_device"] = (
                DeviceManager.resolve_tensorflow(
                    self.inference_device_var.get()
                )
            )
            evaluation_source = (
                self.external_image_directory
                or self.external_csv_path
                or "External evaluation data"
            )
            lines = [
                "External Dataset Model Evaluation",
                "=" * 60,
                f"Model package: {self.package_data['package_path']}",
                f"Evaluation dataset: {evaluation_source}",
                f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Task type: {metadata.get('task_type', TASK_CLASSIFICATION)}",
                f"Feature columns: {metadata.get('feature_columns', [])}",
                f"Target columns: {metadata.get('target_columns', [])}",
                f"Selected actual column: {self.eval_label_var.get()}",
                "Execution device: "
                + DeviceManager.format_status(
                    metadata["evaluation_execution_device"]
                ),
                "",
            ]

            if self.evaluation_metrics is not None:
                lines.append("Metrics")
                lines.append("-" * 60)
                for key, value in self.evaluation_metrics.items():
                    lines.append(f"{key}: {value}")
                lines.append("")

            lines.extend([
                "Evaluation Report",
                "-" * 60,
                self.evaluation_report,
            ])
            if self.evaluation_cm is not None:
                lines.extend([
                    "",
                    "Confusion Matrix",
                    "-" * 60,
                    str(self.evaluation_cm),
                ])

            with open(save_path, "w", encoding="utf-8") as file:
                file.write("\n".join(lines))

            messagebox.showinfo(
                "Report Saved",
                f"Evaluation report saved successfully:\n{save_path}",
                parent=self
            )
        except Exception as exc:
            messagebox.showerror(
                "Save Error",
                f"Failed to save evaluation report:\n{exc}",
                parent=self
            )

    def save_complete_results(self):
        """Save all predictions, metrics, reports, tables, and plots together."""
        if self.prediction_df is None or self.package_data is None:
            messagebox.showwarning(
                "No Results",
                "Run evaluation or prediction before saving complete results.",
                parent=self,
            )
            return

        default_name = (
            Path(self.package_data["package_path"]).stem
            + "_evaluation_results.zip"
        )
        save_path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Complete Evaluation Results",
            initialfile=default_name,
            defaultextension=".zip",
            filetypes=[("ZIP Result Bundle", "*.zip")],
        )
        if not save_path:
            return

        temp_dir = tempfile.mkdtemp(prefix="nn_evaluation_results_")
        try:
            metadata = dict(self.package_data["metadata"])
            metadata["evaluation_execution_device"] = (
                DeviceManager.resolve_tensorflow(
                    self.inference_device_var.get()
                )
            )
            source_path = (
                self.external_image_directory
                or self.external_csv_path
                or "External evaluation data"
            )
            write_complete_result_files(
                temp_dir,
                title="Loaded Model Evaluation Results",
                task_type=metadata.get(
                    "task_type",
                    TASK_CLASSIFICATION,
                ),
                source_description=str(source_path),
                metadata=metadata,
                predictions_df=self.prediction_df,
                metrics=self.evaluation_metrics,
                report_text=self.evaluation_report or "",
                confusion_mat=self.evaluation_cm,
                class_names=self.evaluation_class_names,
                log_text=self.eval_result_text.get("1.0", tk.END).strip(),
            )
            custom_root = Path(temp_dir) / "custom_results"
            for index, directory in enumerate(
                self.custom_result_directories,
                start=1,
            ):
                source = Path(directory)
                if source.is_dir():
                    shutil.copytree(
                        source,
                        custom_root / f"custom_result_{index:02d}",
                    )
            zip_result_directory(temp_dir, save_path)
            messagebox.showinfo(
                "Complete Results Saved",
                "Predictions, metrics, reports, tables, and available plots "
                f"were saved successfully:\n{save_path}",
                parent=self,
            )
        except Exception as exc:
            messagebox.showerror(
                "Save Error",
                f"Failed to save complete evaluation results:\n{exc}",
                parent=self,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
