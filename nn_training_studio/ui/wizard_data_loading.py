"""Ui / wizard data loading for NN Training Studio."""

from pathlib import Path
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog
from tkinter import messagebox
import os
import pandas as pd
import tkinter as tk
from tkinter import ttk
from nn_training_studio.constants import (
    CUSTOM_FILTER_MODE_EXPERT,
    DATA_MODE_IMAGE,
    DATA_MODE_TABULAR,
    IMAGE_MODEL_CNN,
    SUPPORTED_IMAGE_MODEL_TYPES,
    TASK_CLASSIFICATION,
)
from nn_training_studio.image_data import (
    build_image_dataset_profile,
    scan_image_dataset,
)
from nn_training_studio.simulink_data import save_matlab_export_tools
from nn_training_studio.ui.simulink_import import SimulinkImportDialog


class DataLoadingMixin:
    """DataLoading behavior for the main application."""

    def build_step_1(self):
        top_frame = ttk.Frame(self.content_frame)
        self.step1_top_frame = top_frame
        top_frame.pack(fill=tk.X, pady=10)
        top_frame.columnconfigure(1, weight=1)

        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            self.step1_load_button = ttk.Button(
                top_frame,
                text="Load Image Dataset Folder",
                command=self.load_image_folder,
            )
            mode_help = (
                "Choose one root folder whose direct subfolders are the class "
                "names. Image files may be nested below each class folder."
            )
        else:
            self.step1_load_button = ttk.Button(
                top_frame,
                text="Load CSV / Excel Dataset",
                command=self.load_csv,
            )
            mode_help = (
                "Choose a CSV or Excel table containing signal, tabular, or "
                "time-series samples."
            )

        self.file_label = ttk.Label(
            top_frame,
            text=(
                "No dataset loaded"
                if self.file_path is None
                else os.path.basename(self.file_path)
            ),
            wraplength=850,
        )
        self.step1_load_button.grid(row=0, column=0, sticky="w", padx=5)
        self.file_label.grid(row=0, column=1, sticky="ew", padx=10)
        if self.data_mode_var.get() != DATA_MODE_IMAGE:
            actions = ttk.Frame(top_frame)
            actions.grid(row=1, column=0, columnspan=2, sticky="w", padx=5, pady=(8, 0))
            ttk.Button(actions, text="Import MATLAB / Simulink", command=self.load_simulink_exports).pack(side=tk.LEFT)
            ttk.Button(actions, text="Save MATLAB export helpers...", command=self.save_simulink_export_helpers).pack(side=tk.LEFT, padx=8)
            ttk.Button(actions, text="Import NN Scope Captures", command=self.load_scope_captures).pack(side=tk.LEFT)

        info_frame = ttk.LabelFrame(self.content_frame, text="Dataset Preview")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.step1_text = ScrolledText(info_frame, wrap=tk.WORD)
        self.step1_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        if self.df is None:
            self.step1_text.insert(
                tk.END,
                mode_help
            )
        else:
            self.display_dataset_summary()
        self.after_idle(self._apply_responsive_layout)

    def load_csv(self):
        file_path = filedialog.askopenfilename(
            title="Select Signal / Tabular Dataset",
            filetypes=[
                ("Data Files", "*.csv *.xlsx *.xls"),
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx *.xls"),
            ],
        )

        if not file_path:
            return

        try:
            self.set_training_workspace(DATA_MODE_TABULAR)
            self.image_directory = None
            self.image_records = None
            self.image_class_names = []
            if self.model_type_var.get() in SUPPORTED_IMAGE_MODEL_TYPES:
                self.model_type_var.set("CNN-LSTM")
            if self.scaler_var.get() == "Image model preprocessing":
                self.scaler_var.set("MinMaxScaler")
            suffix = Path(file_path).suffix.lower()
            if suffix == ".csv":
                self.raw_df = pd.read_csv(file_path)
            elif suffix in (".xlsx", ".xls"):
                self.raw_df = pd.read_excel(file_path)
            else:
                raise ValueError("Choose a CSV, XLSX, or XLS file.")
            if self.raw_df.empty:
                raise ValueError("The selected dataset is empty.")
            self.simulink_import_metadata = None
            self.scope_import_metadata = None
            self.processing_pipeline_training = None
            self.sampling_frequency_var.set("")
            self.filtered_df = self.raw_df.copy()
            self.custom_filtered_df = self.filtered_df.copy()
            self.df = self.custom_filtered_df.copy()
            self.file_path = file_path

            # Reset results that belonged to a previously loaded dataset.
            self.trained_model = None
            self.scaler = None
            self.label_encoder = None
            self.target_scaler = None
            self.metadata = None
            self.history = None
            self.confusion_mat = None
            self.class_names = None
            self.y_test_result = None
            self.y_pred_result = None
            self.result_target_names = []
            self.anomaly_scores = None
            self.dataset_profile = None
            self.dataset_analysis_report = ""
            self.ai_recommendation = None
            self.ai_recommendation_report = ""
            self.ai_provider_summary = None
            self.ai_generation = None
            self.ai_generation_report = ""
            self.ai_generation_provider_summary = None
            self.safe_filter_generation = None
            self.safe_filter_recommendation = None
            self.safe_model_generation = None
            self.safe_model_recommendation = None
            self.custom_filter_enabled_var.set(False)
            self.custom_filter_mode_var.set(CUSTOM_FILTER_MODE_EXPERT)
            self.selected_custom_filter_cols = []
            self.safe_filter_spec = None
            self.safe_model_spec = None
            self.custom_filter_ai_report = ""
            self.custom_model_ai_report = ""

            self.clear_content()
            self.initialize_signal_column_defaults()

            self.clear_content()
            self._editor_drafts = {}
            self._pending_preparation_steps = set()
            self._last_training_log = ""
            self.last_saved_model_package_path = None
            self.project_draft_path = None
            self.guided_stage = 0
            self.guided_detail_step = None
            self.show_step()

        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load CSV file:\n{e}")

    def load_simulink_exports(self):
        if self._operation_busy():
            return
        paths = filedialog.askopenfilenames(
            parent=self, title="Select MATLAB / Simulink runs (matching columns and sample rate)",
            filetypes=[("Simulation exports", "*.csv *.mat"), ("CSV", "*.csv"), ("MAT (-v7 numeric arrays)", "*.mat")])
        if paths:
            SimulinkImportDialog(self, tuple(paths), self.apply_simulink_dataset)

    def load_scope_captures(self):
        from nn_training_studio.scope_runtime.dataset import load_capture_dataset

        if self._operation_busy():
            return
        paths = filedialog.askopenfilenames(parent=self, title="Select native_samples.csv capture files",
                                           filetypes=[("Saved capture CSV", "*.csv")])
        if paths:
            SimulinkImportDialog(self, tuple(paths), self.apply_scope_dataset,
                                 loader=load_capture_dataset, title="Import NN Scope Captures")

    def apply_scope_dataset(self, dataset):
        self.reset_training_project(DATA_MODE_TABULAR)
        self.raw_df = dataset.frame
        self.filtered_df = self.raw_df.copy()
        self.custom_filtered_df = self.raw_df.copy()
        self.df = self.raw_df.copy()
        self.file_path = dataset.metadata["sources"][0]["file"]
        self.scope_import_metadata = dataset.metadata
        self.sampling_frequency_var.set(f"{dataset.rate:.12g}")
        self.guided_data_order_var.set("Ordered measurements")
        self.feature_cols = dataset.features
        self.show_step()

    def apply_simulink_dataset(self, dataset):
        # Commit only after every selected file passed validation.
        self.reset_training_project(DATA_MODE_TABULAR)
        self.raw_df = dataset.frame.copy()
        self.filtered_df = self.raw_df.copy()
        self.custom_filtered_df = self.raw_df.copy()
        self.df = self.raw_df.copy()
        self.file_path = dataset.metadata["sources"][0]["file"]
        self.simulink_import_metadata = dataset.metadata
        self.sampling_frequency_var.set(f"{dataset.sample_rate_hz:.12g}")
        self.guided_data_order_var.set("Ordered measurements")
        self.feature_cols = list(dataset.feature_columns)
        self.target_cols = [dataset.target_column] if dataset.target_column else []
        self.label_col = dataset.target_column or ""
        self.label_var.set(self.label_col)
        if dataset.target_column:
            self.output_units_var.set(str(self.df[dataset.target_column].nunique()))
        self.show_step()

    def save_simulink_export_helpers(self):
        directory = filedialog.askdirectory(parent=self, title="Save MATLAB helpers beside your Simulink model")
        if not directory:
            return
        try:
            save_matlab_export_tools(directory)
            messagebox.showinfo("MATLAB helpers saved",
                "In MATLAB, add this folder to your path, then run:\n\n"
                "out = sim('your_model');\n"
                "nn_export_simulink(out, 'run.csv', 'Label', 'Normal');\n\n"
                "Or run and export together:\n"
                "nn_run_and_export('your_model', 'run.csv', 'Label', 'Normal');\n\n"
                "Enable signal logging in your model first. Select Import MATLAB / Simulink when the file is ready.", parent=self)
        except Exception as exc:
            messagebox.showerror("MATLAB helpers", str(exc), parent=self)

    def load_image_folder(self):
        directory = filedialog.askdirectory(
            title=(
                "Select Image Dataset Folder "
                "(one direct subfolder per class)"
            )
        )
        if not directory:
            return

        try:
            image_records, class_names = scan_image_dataset(directory)
            self.simulink_import_metadata = None
            self.scope_import_metadata = None
            self.processing_pipeline_training = None
            self.set_training_workspace(DATA_MODE_IMAGE)
            self.image_directory = str(Path(directory).resolve())
            self.image_records = image_records.copy()
            self.image_class_names = list(class_names)
            # A compact records table is retained for checkpoint fingerprinting
            # and summaries. Image pixels remain streamed from disk.
            self.raw_df = image_records.copy()
            self.filtered_df = image_records.copy()
            self.custom_filtered_df = image_records.copy()
            self.df = image_records.copy()
            self.file_path = self.image_directory
            self.dataset_profile = build_image_dataset_profile(
                image_records,
                class_names,
                self.image_directory,
            )
            self.dataset_analysis_report = ""
            self.ai_recommendation = None
            self.ai_recommendation_report = ""
            self.ai_provider_summary = None
            self.ai_generation = None
            self.ai_generation_report = ""
            self.ai_generation_provider_summary = None
            self.safe_filter_generation = None
            self.safe_filter_recommendation = None
            self.safe_model_generation = None
            self.safe_model_recommendation = None
            self.safe_filter_spec = None
            self.safe_model_spec = None
            self.custom_filter_enabled_var.set(False)
            self.selected_filter_cols = []
            self.selected_custom_filter_cols = []

            self.clear_content()
            self.task_type_var.set(TASK_CLASSIFICATION)
            self.feature_cols = ["image"]
            self.target_cols = ["class_name"]
            self.label_col = "class_name"
            self.label_var.set("class_name")
            self.model_type_var.set(IMAGE_MODEL_CNN)
            self.output_mode_var.set("Auto")
            self.output_units_var.set(str(len(class_names)))
            self.output_activation_var.set("softmax")
            self.loss_var.set("sparse_categorical_crossentropy")
            self.scaler_var.set("Image model preprocessing")

            self.trained_model = None
            self.scaler = None
            self.label_encoder = None
            self.target_scaler = None
            self.metadata = None
            self.history = None
            self.best_checkpoint_path = None
            self.training_run_id = None
            self.confusion_mat = None
            self.class_names = None
            self.y_test_result = None
            self.y_pred_result = None
            self.result_target_names = []
            self.anomaly_scores = None
            self.clear_content()
            self._editor_drafts = {}
            self._pending_preparation_steps = set()
            self._last_training_log = ""
            self.last_saved_model_package_path = None
            self.project_draft_path = None
            self.guided_stage = 0
            self.guided_detail_step = None
            self.show_step()
        except Exception as exc:
            messagebox.showerror(
                "Image Dataset Error",
                f"Failed to load the image dataset:\n{exc}",
            )

    def display_dataset_summary(self):
        self.step1_text.delete("1.0", tk.END)

        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            profile = self.dataset_profile or {}
            self.step1_text.insert(
                tk.END,
                "Image dataset loaded successfully.\n\n"
            )
            self.step1_text.insert(
                tk.END,
                f"Folder: {self.image_directory}\n"
            )
            self.step1_text.insert(
                tk.END,
                f"Photos: {profile.get('image_count', 0)}\n"
            )
            self.step1_text.insert(
                tk.END,
                f"Classes: {profile.get('class_count', 0)}\n\n"
            )
            self.step1_text.insert(tk.END, "Images per class:\n")
            for name in self.image_class_names:
                count = profile.get("class_counts", {}).get(name, 0)
                self.step1_text.insert(tk.END, f"- {name}: {count}\n")
            self.step1_text.insert(
                tk.END,
                "\nSupported files found:\n"
                f"{profile.get('extension_counts', {})}\n\n"
                "The images will be streamed from disk, resized during "
                "training, and split so every class is represented in the "
                "training, validation, and test sets."
            )
            return

        preview_df = self.raw_df if self.raw_df is not None else self.df

        self.step1_text.insert(tk.END, "Dataset loaded successfully.\n\n")
        self.step1_text.insert(tk.END, f"File: {self.file_path}\n")
        self.step1_text.insert(tk.END, f"Rows: {preview_df.shape[0]}\n")
        self.step1_text.insert(tk.END, f"Columns: {preview_df.shape[1]}\n")
        self.step1_text.insert(tk.END, f"Missing values: {int(preview_df.isna().sum().sum())}\n\n")
        simulation = getattr(self, "simulink_import_metadata", None)
        if simulation:
            self.step1_text.insert(tk.END,
                f"Simulink runs: {len(simulation['sources'])}; sample rate: {simulation['sample_rate_hz']:g} Hz.\n"
                "Time and run identifiers are excluded from suggested inputs. Windows stay inside a run.\n"
                "Select/confirm the target signals or labels before training.\n\n")
        scope = getattr(self, "scope_import_metadata", None)
        if scope:
            self.step1_text.insert(tk.END,
                f"NN Scope captures: {len(scope['sources'])}; {scope['sample_rate_hz']:g} Hz.\n"
                "Add confirmed targets for supervised training. Review the evaluation split before interpreting accuracy.\n\n")

        self.step1_text.insert(tk.END, "Column names:\n")
        self.step1_text.insert(tk.END, str(list(preview_df.columns)))
        self.step1_text.insert(tk.END, "\n\nDataset preview:\n")
        self.step1_text.insert(tk.END, preview_df.head(15).to_string())

    def validate_step_1(self):
        if self.df is None:
            raise ValueError("Please load a CSV or image dataset first.")
        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            if self.image_records is None or self.image_records.empty:
                raise ValueError("Please load a valid image dataset folder.")
            if len(self.image_class_names) < 2:
                raise ValueError(
                    "Image classification requires at least two classes."
                )
