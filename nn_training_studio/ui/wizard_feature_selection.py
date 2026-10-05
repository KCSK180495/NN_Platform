"""Ui / wizard feature selection for NN Training Studio."""

from tkinter.scrolledtext import ScrolledText
import pandas as pd
import tkinter as tk
from tkinter import ttk
from nn_training_studio.constants import (
    DATA_MODE_IMAGE,
    TASK_AUTOENCODER,
    TASK_CLASSIFICATION,
    TASK_FORECASTING,
    TASK_MULTI_OUTPUT,
    TASK_REGRESSION,
    TASK_TYPES,
)


class FeatureSelectionMixin:
    """FeatureSelection behavior for the main application."""

    def build_step_2(self):
        if self.df is None:
            ttk.Label(self.content_frame, text="Please load dataset first.").pack()
            return

        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            frame = ttk.LabelFrame(
                self.content_frame,
                text="Image Classification Task",
            )
            frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            summary = ScrolledText(frame, wrap=tk.WORD)
            summary.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            profile = self.dataset_profile or {}
            summary.insert(
                tk.END,
                "Task type: Image classification\n\n"
                "Input: decoded and resized image pixels\n"
                "Target: class folder name\n\n"
                f"Classes ({len(self.image_class_names)}):\n"
            )
            for class_name in self.image_class_names:
                summary.insert(
                    tk.END,
                    f"- {class_name}: "
                    f"{profile.get('class_counts', {}).get(class_name, 0)} "
                    "images\n"
                )
            summary.insert(
                tk.END,
                "\nFolder names define the labels. Rename or reorganize the "
                "folders before loading if a class name is incorrect.\n\n"
                "Version 18 supports whole-image classification. Object "
                "detection (for example YOLO bounding boxes) requires a "
                "separate annotation workflow and is not mixed into this "
                "classifier."
            )
            summary.config(state=tk.DISABLED)
            return

        task_frame = ttk.LabelFrame(self.content_frame, text="Task Type")
        task_frame.pack(fill=tk.X, padx=5, pady=(5, 8))

        ttk.Label(
            task_frame,
            text="Neural-network task:"
        ).pack(side=tk.LEFT, padx=(8, 5), pady=8)

        task_combo = ttk.Combobox(
            task_frame,
            textvariable=self.task_type_var,
            values=TASK_TYPES,
            state="readonly",
            width=34
        )
        task_combo.pack(side=tk.LEFT, padx=5, pady=8)
        task_combo.bind("<<ComboboxSelected>>", self.on_task_changed)

        self.task_hint_label = ttk.Label(
            task_frame,
            text="",
            wraplength=650
        )
        self.task_hint_label.pack(side=tk.LEFT, padx=12, pady=8)

        columns_frame = ttk.Frame(self.content_frame)
        columns_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.LabelFrame(columns_frame, text="Input Columns")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        middle_frame = ttk.LabelFrame(columns_frame, text="Target Columns")
        middle_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        right_frame = ttk.LabelFrame(self.content_frame, text="Selected Data Preview")
        right_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=(8, 5))

        ttk.Label(
            left_frame,
            text="Signals/features supplied to the model:",
            font=("Arial", 11, "bold")
        ).pack(anchor="w")

        self.feature_listbox = tk.Listbox(
            left_frame,
            selectmode=tk.MULTIPLE,
            height=10,
            exportselection=False
        )
        self.feature_listbox.pack(fill=tk.BOTH, expand=True, pady=5)

        for col in self.df.columns:
            self.feature_listbox.insert(tk.END, col)

        for i, col in enumerate(self.df.columns):
            if col in self.feature_cols:
                self.feature_listbox.selection_set(i)

        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            button_frame,
            text="Select Numeric Columns",
            command=self.select_numeric_features
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        ttk.Button(
            button_frame,
            text="Clear",
            command=self.clear_feature_selection
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        ttk.Label(
            middle_frame,
            text="Select one or more outputs:",
            font=("Arial", 11, "bold")
        ).pack(anchor="w")

        self.target_listbox = tk.Listbox(
            middle_frame,
            selectmode=tk.MULTIPLE,
            height=10,
            exportselection=False
        )
        self.target_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        for col in self.df.columns:
            self.target_listbox.insert(tk.END, col)
        for i, col in enumerate(self.df.columns):
            if col in self.target_cols:
                self.target_listbox.selection_set(i)
        self.target_listbox.bind(
            "<<ListboxSelect>>",
            lambda event: self.preview_selected_columns()
        )

        ttk.Button(
            middle_frame,
            text="Clear Targets",
            command=lambda: self.target_listbox.selection_clear(0, tk.END)
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            middle_frame,
            text="Preview Selected Columns",
            command=self.preview_selected_columns
        ).pack(fill=tk.X, pady=2)

        self.step2_text = ScrolledText(right_frame, wrap=tk.WORD)
        self.step2_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.update_task_column_controls()
        self.preview_selected_columns()

    def on_task_changed(self, event=None):
        self.apply_task_defaults()
        self.update_task_column_controls()
        self.preview_selected_columns()

    def apply_task_defaults(self):
        task_type = self.task_type_var.get()

        if task_type == TASK_CLASSIFICATION:
            self.model_type_var.set("CNN-LSTM")
            self.output_mode_var.set("Auto")
            self.output_activation_var.set("softmax")
            self.loss_var.set("sparse_categorical_crossentropy")
        elif task_type in (TASK_REGRESSION, TASK_MULTI_OUTPUT):
            self.model_type_var.set("DNN")
            self.output_mode_var.set("Auto")
            self.output_activation_var.set("linear")
            self.loss_var.set("mean_squared_error")
        elif task_type == TASK_FORECASTING:
            self.model_type_var.set("LSTM")
            self.output_mode_var.set("Auto")
            self.output_activation_var.set("linear")
            self.loss_var.set("mean_squared_error")
            self.stride_var.set("1")
        elif task_type == TASK_AUTOENCODER:
            self.model_type_var.set("Dense Autoencoder")
            self.output_mode_var.set("Auto")
            self.output_activation_var.set("linear")
            self.loss_var.set("mean_squared_error")

    def update_task_column_controls(self):
        if not hasattr(self, "target_listbox"):
            return

        task_type = self.task_type_var.get()
        hints = {
            TASK_CLASSIFICATION: "Choose exactly one categorical label column.",
            TASK_REGRESSION: "Choose exactly one numeric continuous target.",
            TASK_MULTI_OUTPUT: "Choose two or more numeric targets predicted together.",
            TASK_FORECASTING: (
                "Choose one or more numeric targets. Past input windows predict "
                "their future values."
            ),
            TASK_AUTOENCODER: (
                "No target column is required. The selected input columns are "
                "reconstructed and used for anomaly scoring."
            ),
        }
        self.task_hint_label.config(text=hints.get(task_type, ""))

        if task_type == TASK_AUTOENCODER:
            self.target_listbox.selection_clear(0, tk.END)
            self.target_listbox.config(state=tk.DISABLED)
        else:
            self.target_listbox.config(state=tk.NORMAL)

    def select_numeric_features(self):
        if self.df is None:
            return

        target_cols = set(self.get_selected_targets_from_listbox())

        self.feature_listbox.selection_clear(0, tk.END)

        for i, col in enumerate(self.df.columns):
            if col in target_cols:
                continue

            if "id" in col.lower():
                continue

            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.feature_listbox.selection_set(i)

    def clear_feature_selection(self):
        self.feature_listbox.selection_clear(0, tk.END)

    def get_selected_features_from_listbox(self):
        if not hasattr(self, "feature_listbox"):
            return list(self.feature_cols)
        selected_indices = self.feature_listbox.curselection()
        return [self.feature_listbox.get(i) for i in selected_indices]

    def get_selected_targets_from_listbox(self):
        if not hasattr(self, "target_listbox"):
            return list(self.target_cols)
        selected_indices = self.target_listbox.curselection()
        return [self.target_listbox.get(i) for i in selected_indices]

    def preview_selected_columns(self):
        feature_cols = self.get_selected_features_from_listbox()
        target_cols = (
            []
            if self.task_type_var.get() == TASK_AUTOENCODER
            else self.get_selected_targets_from_listbox()
        )

        self.step2_text.delete("1.0", tk.END)

        if len(feature_cols) == 0:
            self.step2_text.insert(tk.END, "No input columns selected.")
            return

        overlap = sorted(set(target_cols).intersection(feature_cols))
        if overlap:
            self.step2_text.insert(
                tk.END,
                "Error: target columns cannot also be input columns: "
                + str(overlap)
            )
            return

        preview_cols = list(dict.fromkeys(feature_cols + target_cols))

        self.step2_text.insert(
            tk.END,
            f"Task type:\n{self.task_type_var.get()}\n\n"
        )
        self.step2_text.insert(tk.END, "Selected input columns:\n")
        self.step2_text.insert(tk.END, str(feature_cols))
        self.step2_text.insert(tk.END, "\n\nSelected target columns:\n")
        self.step2_text.insert(
            tk.END,
            "Input reconstruction (no separate targets)"
            if not target_cols and self.task_type_var.get() == TASK_AUTOENCODER
            else str(target_cols)
        )
        self.step2_text.insert(tk.END, "\n\nPreview:\n")
        self.step2_text.insert(tk.END, self.df[preview_cols].head(15).to_string())

    def validate_step_2(self):
        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            if self.image_records is None or self.image_records.empty:
                raise ValueError("The image dataset is not available.")
            if len(self.image_class_names) < 2:
                raise ValueError(
                    "Image classification requires at least two classes."
                )
            self.task_type_var.set(TASK_CLASSIFICATION)
            self.feature_cols = ["image"]
            self.target_cols = ["class_name"]
            self.label_col = "class_name"
            self.label_var.set("class_name")
            self.output_mode_var.set("Auto")
            self.output_units_var.set(str(len(self.image_class_names)))
            self.output_activation_var.set("softmax")
            self.loss_var.set("sparse_categorical_crossentropy")
            return

        self.feature_cols = self.get_selected_features_from_listbox()
        task_type = self.task_type_var.get()
        self.target_cols = (
            []
            if task_type == TASK_AUTOENCODER
            else self.get_selected_targets_from_listbox()
        )
        self.label_col = self.target_cols[0] if self.target_cols else ""
        self.label_var.set(self.label_col)

        if len(self.feature_cols) == 0:
            raise ValueError("Please select at least one input column.")

        if task_type in (TASK_CLASSIFICATION, TASK_REGRESSION):
            if len(self.target_cols) != 1:
                raise ValueError(
                    f"{task_type} requires exactly one target column."
                )
        elif task_type == TASK_MULTI_OUTPUT:
            if len(self.target_cols) < 2:
                raise ValueError(
                    "Multi-output regression requires at least two target columns."
                )
        elif task_type == TASK_FORECASTING:
            if not self.target_cols:
                raise ValueError(
                    "Time-series forecasting requires at least one target column."
                )

        overlap = sorted(set(self.target_cols).intersection(self.feature_cols))
        if overlap:
            raise ValueError(
                "Target columns cannot also be input columns: " + str(overlap)
            )
        filtered_targets = sorted(
            set(self.target_cols).intersection(
                self.selected_custom_filter_cols
            )
        )
        if (
            self.custom_filter_enabled_var.get()
            and filtered_targets
        ):
            raise ValueError(
                "A target column cannot be modified by the custom filter: "
                + str(filtered_targets)
                + ". Return to the custom-filter page and remove it from the "
                "filter specification."
            )

        if task_type == TASK_CLASSIFICATION:
            estimated_classes = self.df[self.label_col].dropna().nunique()
            self.output_units_var.set(str(estimated_classes))
        elif task_type == TASK_FORECASTING:
            horizon = int(self.forecast_horizon_var.get())
            self.output_units_var.set(str(len(self.target_cols) * horizon))
        elif task_type == TASK_AUTOENCODER:
            self.output_units_var.set(str(len(self.feature_cols)))
        else:
            self.output_units_var.set(str(len(self.target_cols)))

        numeric_targets = (
            task_type in (
                TASK_REGRESSION,
                TASK_MULTI_OUTPUT,
                TASK_FORECASTING
            )
        )
        if numeric_targets:
            invalid = [
                col for col in self.target_cols
                if pd.to_numeric(self.df[col], errors="coerce").notna().sum() == 0
            ]
            if invalid:
                raise ValueError(
                    "These target columns contain no numeric values: "
                    + str(invalid)
                )
