"""Ui / wizard preprocessing for NN Training Studio."""

from tkinter.scrolledtext import ScrolledText
import tkinter as tk
from tkinter import ttk
from nn_training_studio.constants import (
    DATA_MODE_IMAGE,
    TASK_AUTOENCODER,
    TASK_FORECASTING,
    TASK_MULTI_OUTPUT,
    TASK_REGRESSION,
)


class PreprocessingMixin:
    """Preprocessing behavior for the main application."""

    def build_step_3(self):
        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            frame = ttk.LabelFrame(
                self.content_frame,
                text="Image Preprocessing",
            )
            frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            form = ttk.Frame(frame)
            form.pack(fill=tk.X, padx=20, pady=15)
            form.columnconfigure(1, weight=1)

            ttk.Label(
                form,
                text="Image Height (pixels):",
                font=("Arial", 11, "bold"),
            ).grid(row=0, column=0, sticky="w", pady=8)
            ttk.Entry(
                form,
                textvariable=self.image_height_var,
            ).grid(row=0, column=1, sticky="ew", padx=10, pady=8)

            ttk.Label(
                form,
                text="Image Width (pixels):",
                font=("Arial", 11, "bold"),
            ).grid(row=1, column=0, sticky="w", pady=8)
            ttk.Entry(
                form,
                textvariable=self.image_width_var,
            ).grid(row=1, column=1, sticky="ew", padx=10, pady=8)

            ttk.Label(
                form,
                text="Colour Mode:",
                font=("Arial", 11, "bold"),
            ).grid(row=2, column=0, sticky="w", pady=8)
            ttk.Combobox(
                form,
                textvariable=self.image_color_mode_var,
                values=["RGB", "Grayscale"],
                state="readonly",
            ).grid(row=2, column=1, sticky="ew", padx=10, pady=8)

            ttk.Checkbutton(
                form,
                text=(
                    "Enable safe training augmentation "
                    "(flip, rotation, zoom, contrast)"
                ),
                variable=self.image_augmentation_var,
            ).grid(
                row=3,
                column=0,
                columnspan=2,
                sticky="w",
                pady=10,
            )

            ttk.Label(
                form,
                text="Test Data Percentage:",
                font=("Arial", 11, "bold"),
            ).grid(row=4, column=0, sticky="w", pady=8)
            ttk.Entry(
                form,
                textvariable=self.test_size_var,
            ).grid(row=4, column=1, sticky="ew", padx=10, pady=8)

            ttk.Label(
                frame,
                text=(
                    "Images are streamed from disk, resized without changing "
                    "the source files, and normalized inside the saved model. "
                    "Augmentation is active only during training."
                ),
                wraplength=900,
                foreground="#245a85",
            ).pack(anchor="w", padx=20, pady=15)
            return

        frame = ttk.Frame(self.content_frame)
        frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        ttk.Label(
            frame,
            text="Missing Value Method:",
            font=("Arial", 11, "bold")
        ).pack(anchor="w")

        ttk.Combobox(
            frame,
            textvariable=self.missing_var,
            state="readonly",
            values=[
                "Drop rows",
                "Fill mean",
                "Fill median",
                "Forward/backward fill"
            ]
        ).pack(fill=tk.X, pady=5)

        ttk.Label(
            frame,
            text="Normalization Method:",
            font=("Arial", 11, "bold")
        ).pack(anchor="w", pady=(15, 0))

        ttk.Combobox(
            frame,
            textvariable=self.scaler_var,
            state="readonly",
            values=[
                "MinMaxScaler",
                "StandardScaler",
                "No normalization"
            ]
        ).pack(fill=tk.X, pady=5)

        if self.task_type_var.get() in (
            TASK_REGRESSION,
            TASK_MULTI_OUTPUT,
            TASK_FORECASTING
        ):
            ttk.Label(
                frame,
                text="Target Normalization Method:",
                font=("Arial", 11, "bold")
            ).pack(anchor="w", pady=(15, 0))

            ttk.Combobox(
                frame,
                textvariable=self.target_scaler_var,
                state="readonly",
                values=[
                    "StandardScaler",
                    "MinMaxScaler",
                    "No normalization"
                ]
            ).pack(fill=tk.X, pady=5)

        if self.task_type_var.get() == TASK_FORECASTING:
            ttk.Label(
                frame,
                text="Forecast Horizon (future rows):",
                font=("Arial", 11, "bold")
            ).pack(anchor="w", pady=(15, 0))
            ttk.Entry(
                frame,
                textvariable=self.forecast_horizon_var
            ).pack(fill=tk.X, pady=5)

        if self.task_type_var.get() == TASK_AUTOENCODER:
            ttk.Label(
                frame,
                text="Anomaly Threshold Percentile:",
                font=("Arial", 11, "bold")
            ).pack(anchor="w", pady=(15, 0))
            ttk.Entry(
                frame,
                textvariable=self.anomaly_percentile_var
            ).pack(fill=tk.X, pady=5)

        ttk.Label(
            frame,
            text="Test Data Percentage:",
            font=("Arial", 11, "bold")
        ).pack(anchor="w", pady=(15, 0))

        ttk.Entry(
            frame,
            textvariable=self.test_size_var
        ).pack(fill=tk.X, pady=5)

        summary = ScrolledText(frame, height=12, wrap=tk.WORD)
        summary.pack(fill=tk.BOTH, expand=True, pady=20)

        summary.insert(tk.END, "Current selected data:\n\n")
        summary.insert(tk.END, f"Task type: {self.task_type_var.get()}\n")
        summary.insert(tk.END, f"Input columns: {self.feature_cols}\n")
        summary.insert(tk.END, f"Target columns: {self.target_cols}\n\n")
        summary.insert(tk.END, "General recommendation:\n")
        summary.insert(tk.END, "- Missing value: Drop rows or Forward/backward fill\n")
        summary.insert(tk.END, "- Normalization: MinMaxScaler\n")
        summary.insert(tk.END, "- Test data percentage: 30\n")

    def validate_step_3(self):
        test_size = float(self.test_size_var.get())

        if test_size <= 0 or test_size >= 90:
            raise ValueError("Test data percentage must be between 1 and 89.")

        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            image_height = int(self.image_height_var.get())
            image_width = int(self.image_width_var.get())
            if image_height < 32 or image_width < 32:
                raise ValueError(
                    "Image height and width must each be at least 32 pixels."
                )
            if image_height > 2048 or image_width > 2048:
                raise ValueError(
                    "Image height and width must not exceed 2048 pixels."
                )
            if self.image_color_mode_var.get() not in ("RGB", "Grayscale"):
                raise ValueError("Choose RGB or Grayscale colour mode.")
            return

        if self.task_type_var.get() == TASK_FORECASTING:
            if int(self.forecast_horizon_var.get()) <= 0:
                raise ValueError("Forecast horizon must be larger than 0.")

        if self.task_type_var.get() == TASK_AUTOENCODER:
            percentile = float(self.anomaly_percentile_var.get())
            if percentile <= 50 or percentile >= 100:
                raise ValueError(
                    "Anomaly threshold percentile must be between 50 and 100."
                )
