"""Ui / wizard model settings for NN Training Studio."""

from tkinter.scrolledtext import ScrolledText
import json
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
from nn_training_studio.ai_validation import (
    validate_ai_generation,
)
from nn_training_studio.constants import (
    DATA_MODE_IMAGE,
    DEFAULT_CUSTOM_MODEL_CODE,
    IMAGE_MODEL_EFFICIENTNET,
    IMAGE_MODEL_MOBILENET,
    MODEL_TYPE_SAFE_AI_LEGACY,
    MODEL_TYPE_SAFE_CUSTOM,
    OFFICIAL_PROVIDER_SETTINGS,
    SUPPORTED_IMAGE_MODEL_TYPES,
    TASK_AUTOENCODER,
    TASK_CLASSIFICATION,
    TASK_FORECASTING,
)
from nn_training_studio.models import (
    build_custom_model_from_code,
    build_safe_model_from_spec,
)
from nn_training_studio.templates import (
    make_safe_filter_template,
    make_safe_model_template,
)
from nn_training_studio.training import (
    validate_early_stopping_settings,
)


class ModelSettingsMixin:
    """ModelSettings behavior for the main application."""

    def build_step_4(self):
        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            self.build_image_model_settings()
            return

        notebook = ttk.Notebook(self.content_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        basic_tab = ttk.Frame(notebook)
        advanced_tab = ttk.Frame(notebook)
        custom_model_tab = ttk.Frame(notebook)
        guide_tab = ttk.Frame(notebook)

        notebook.add(basic_tab, text="Basic Model Settings")
        notebook.add(advanced_tab, text="Output / Loss / Optimizer")
        notebook.add(custom_model_tab, text="Customize Model (AI / Python)")
        notebook.add(guide_tab, text="Guide")

        self.build_step_4_basic_tab(basic_tab)
        self.build_step_4_advanced_tab(advanced_tab)
        self.build_step_4_custom_model_tab(custom_model_tab)
        self.build_step_4_guide_tab(guide_tab)

    def collect_early_stopping_settings(self):
        return validate_early_stopping_settings({
            "enabled": self.early_stopping_enabled_var.get(),
            "patience": self.early_stopping_patience_var.get(),
            "min_delta": self.early_stopping_min_delta_var.get(),
            "start_from_epoch": self.early_stopping_warmup_var.get(),
        }, int(self.epochs_var.get()))

    def build_early_stopping_controls(self, parent):
        panel = ttk.LabelFrame(parent, text="Early stopping", padding=8)
        panel.columnconfigure(1, weight=1)
        toggle = ttk.Checkbutton(
            panel, text="Enable early stopping", variable=self.early_stopping_enabled_var,
            command=self.refresh_early_stopping_controls,
        )
        toggle.grid(row=0, column=0, columnspan=2, sticky="w")
        entries = []
        for row, (label, variable) in enumerate((
            ("Patience (epochs without improvement)", self.early_stopping_patience_var),
            ("Minimum improvement in validation loss", self.early_stopping_min_delta_var),
            ("Warm-up epochs (skip monitoring)", self.early_stopping_warmup_var),
        ), start=1):
            ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=3)
            entry = ttk.Entry(panel, textvariable=variable, width=12)
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            entries.append(entry)
        ttk.Label(
            panel, wraplength=500,
            text="Monitors validation loss (lower is better). Patience 0 stops at the first "
                 "non-improving monitored epoch. Warm-up 0 monitors from epoch 1. "
                 "Epochs remains the maximum; the best checkpoint is kept even when disabled.",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self._early_stopping_panels.append((panel, toggle, entries))
        self.refresh_early_stopping_controls()
        return panel

    def refresh_early_stopping_controls(self):
        active = []
        for panel, toggle, entries in self._early_stopping_panels:
            if not panel.winfo_exists():
                continue
            active.append((panel, toggle, entries))
            toggle.configure(state=tk.DISABLED if self.training_running else tk.NORMAL)
            state = (tk.NORMAL if self.early_stopping_enabled_var.get()
                     and not self.training_running else tk.DISABLED)
            for entry in entries:
                entry.configure(state=state)
        self._early_stopping_panels = active

    def build_image_model_settings(self):
        notebook = ttk.Notebook(self.content_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        settings_tab = ttk.Frame(notebook)
        guide_tab = ttk.Frame(notebook)
        notebook.add(settings_tab, text="Image Model Settings")
        notebook.add(guide_tab, text="Guide")

        frame = ttk.Frame(settings_tab)
        frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        frame.columnconfigure(1, weight=1)

        ttk.Label(
            frame,
            text="Model Type:",
            font=("Arial", 10, "bold"),
        ).grid(row=0, column=0, sticky="w", pady=8)
        model_combo = ttk.Combobox(
            frame,
            textvariable=self.model_type_var,
            values=SUPPORTED_IMAGE_MODEL_TYPES,
            state="readonly",
        )
        model_combo.grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        model_combo.bind("<<ComboboxSelected>>", self.on_image_model_changed)

        ttk.Checkbutton(
            frame,
            text=(
                "Use ImageNet pretrained weights for transfer-learning models "
                "(first use may download weights)"
            ),
            variable=self.image_pretrained_var,
        ).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
            pady=8,
        )

        fields = [
            ("Epochs:", self.epochs_var),
            ("Batch Size:", self.batch_size_var),
            ("Validation Split (% of non-test images):", self.validation_split_var),
            ("Dropout Rate:", self.dropout_var),
            ("Learning Rate:", self.lr_var),
        ]
        for row_index, (label_text, variable) in enumerate(fields, start=2):
            ttk.Label(
                frame,
                text=label_text,
                font=("Arial", 10, "bold"),
            ).grid(row=row_index, column=0, sticky="w", pady=8)
            ttk.Entry(
                frame,
                textvariable=variable,
            ).grid(
                row=row_index,
                column=1,
                sticky="ew",
                padx=8,
                pady=8,
            )

        ttk.Label(
            frame,
            text="Optimizer:",
            font=("Arial", 10, "bold"),
        ).grid(row=7, column=0, sticky="w", pady=8)
        ttk.Combobox(
            frame,
            textvariable=self.optimizer_var,
            values=["Adam", "SGD", "RMSprop", "Nadam"],
            state="readonly",
        ).grid(row=7, column=1, sticky="ew", padx=8, pady=8)

        ttk.Label(
            frame,
            text=(
                f"Output: {len(self.image_class_names)} softmax units\n"
                "Loss: sparse categorical cross-entropy\n"
                f"Input: {self.image_height_var.get()} × "
                f"{self.image_width_var.get()} "
                f"{self.image_color_mode_var.get()}"
            ),
            wraplength=850,
            foreground="#245a85",
        ).grid(
            row=8,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(15, 5),
        )

        self.build_early_stopping_controls(frame).grid(
            row=9, column=0, columnspan=2, sticky="ew", pady=8
        )

        guide = ScrolledText(guide_tab, wrap=tk.WORD)
        guide.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        guide.insert(
            tk.END,
            "IMAGE MODEL GUIDE\n\n"
            "Image CNN\n"
            "- Best first choice when working offline or starting with a "
            "simple baseline.\n"
            "- Trains all layers from the current dataset.\n\n"
            "MobileNetV2 Transfer Learning\n"
            "- Smaller and usually faster than EfficientNetB0.\n"
            "- A practical choice for deployment on modest hardware.\n\n"
            "EfficientNetB0 Transfer Learning\n"
            "- Often provides a strong accuracy/size balance.\n"
            "- Usually needs more memory and computation than MobileNetV2.\n\n"
            "For transfer learning, RGB mode and pretrained ImageNet weights "
            "are recommended. Disable pretrained weights only when internet "
            "access is unavailable and no cached weights exist."
        )
        guide.config(state=tk.DISABLED)

    def on_image_model_changed(self, event=None):
        if self.model_type_var.get() in (
            IMAGE_MODEL_MOBILENET,
            IMAGE_MODEL_EFFICIENTNET,
        ):
            self.lr_var.set("0.0001")
            self.image_color_mode_var.set("RGB")
        else:
            self.lr_var.set("0.001")

    def build_step_4_basic_tab(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        form = ttk.Frame(frame)
        form.pack(fill=tk.X)

        ttk.Label(form, text="Model Type:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", pady=8)
        ttk.Combobox(
            form,
            textvariable=self.model_type_var,
            state="readonly",
            values=[
                "DNN",
                "CNN",
                "LSTM",
                "CNN-LSTM",
                "Dense Autoencoder",
                "LSTM Autoencoder",
                MODEL_TYPE_SAFE_CUSTOM,
                "Custom Python Model"
            ],
            width=35
        ).grid(row=0, column=1, sticky="ew", pady=8)

        ttk.Label(form, text="Window / Sequence Size:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(form, textvariable=self.window_size_var, width=38).grid(row=1, column=1, sticky="ew", pady=8)

        ttk.Label(form, text="Stride:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="w", pady=8)
        ttk.Entry(form, textvariable=self.stride_var, width=38).grid(row=2, column=1, sticky="ew", pady=8)

        ttk.Label(form, text="Epochs:", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky="w", pady=8)
        ttk.Entry(form, textvariable=self.epochs_var, width=38).grid(row=3, column=1, sticky="ew", pady=8)

        ttk.Label(form, text="Batch Size:", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky="w", pady=8)
        ttk.Entry(form, textvariable=self.batch_size_var, width=38).grid(row=4, column=1, sticky="ew", pady=8)

        ttk.Label(form, text="Validation Split Percentage:", font=("Arial", 10, "bold")).grid(row=5, column=0, sticky="w", pady=8)
        ttk.Entry(form, textvariable=self.validation_split_var, width=38).grid(row=5, column=1, sticky="ew", pady=8)

        ttk.Label(form, text="Hidden Activation:", font=("Arial", 10, "bold")).grid(row=6, column=0, sticky="w", pady=8)
        ttk.Combobox(
            form,
            textvariable=self.hidden_activation_var,
            state="readonly",
            values=["relu", "tanh", "sigmoid", "elu", "selu"],
            width=35
        ).grid(row=6, column=1, sticky="ew", pady=8)

        ttk.Label(form, text="Dropout Rate:", font=("Arial", 10, "bold")).grid(row=7, column=0, sticky="w", pady=8)
        ttk.Entry(form, textvariable=self.dropout_var, width=38).grid(row=7, column=1, sticky="ew", pady=8)

        self.build_early_stopping_controls(form).grid(
            row=8, column=0, columnspan=2, sticky="ew", pady=8
        )

        form.columnconfigure(1, weight=1)

        note = ScrolledText(frame, height=8, wrap=tk.WORD)
        note.pack(fill=tk.BOTH, expand=True, pady=20)

        note.insert(tk.END, "Basic model guide:\n\n")
        note.insert(tk.END, "DNN: suitable for simple row-based data.\n")
        note.insert(tk.END, "CNN: suitable for signal/window data.\n")
        note.insert(tk.END, "LSTM: suitable for sequence data.\n")
        note.insert(tk.END, "CNN-LSTM: recommended for motor current/fault diagnosis data.\n")
        note.insert(tk.END, "Dense Autoencoder: row reconstruction and anomaly scoring.\n")
        note.insert(tk.END, "LSTM Autoencoder: sequence reconstruction and anomaly scoring.\n")
        note.insert(
            tk.END,
            "Safe Custom Model: editable allowlisted architecture created "
            "manually or generated by AI.\n",
        )
        note.insert(tk.END, "Custom Python Model: define your own Keras architecture in the custom model tab.\n\n")
        note.insert(tk.END, f"Current task: {self.task_type_var.get()}\n\n")
        note.insert(tk.END, "Recommended first setting:\n")
        note.insert(tk.END, f"- Model Type: {self.model_type_var.get()}\n")
        note.insert(tk.END, "- Window Size: 200\n")
        note.insert(tk.END, "- Stride: 50\n")
        note.insert(tk.END, "- Epochs: 30\n")
        note.insert(tk.END, "- Batch Size: 32\n")
        note.insert(tk.END, "- Validation Split: 20\n")
        note.insert(tk.END, "- Hidden Activation: relu\n")
        note.insert(tk.END, "- Dropout Rate: 0.3\n")

    def build_step_4_advanced_tab(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)

        form = ttk.Frame(frame)
        form.pack(fill=tk.X)

        ttk.Label(form, text="Output Units Mode:", font=("Arial", 10, "bold")).grid(row=0, column=0, sticky="w", pady=8)
        ttk.Combobox(
            form,
            textvariable=self.output_mode_var,
            state="readonly",
            values=["Auto", "Custom"],
            width=35
        ).grid(row=0, column=1, sticky="ew", pady=8)

        ttk.Label(form, text="Output Units:", font=("Arial", 10, "bold")).grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(form, textvariable=self.output_units_var, width=38).grid(row=1, column=1, sticky="ew", pady=8)

        ttk.Label(form, text="Output Activation:", font=("Arial", 10, "bold")).grid(row=2, column=0, sticky="w", pady=8)
        ttk.Combobox(
            form,
            textvariable=self.output_activation_var,
            state="readonly",
            values=["softmax", "sigmoid", "linear", "tanh"],
            width=35
        ).grid(row=2, column=1, sticky="ew", pady=8)

        ttk.Label(form, text="Loss Function:", font=("Arial", 10, "bold")).grid(row=3, column=0, sticky="w", pady=8)
        ttk.Combobox(
            form,
            textvariable=self.loss_var,
            state="readonly",
            values=[
                "sparse_categorical_crossentropy",
                "categorical_crossentropy",
                "binary_crossentropy",
                "mean_squared_error",
                "mean_absolute_error"
            ],
            width=35
        ).grid(row=3, column=1, sticky="ew", pady=8)

        ttk.Label(form, text="Optimizer:", font=("Arial", 10, "bold")).grid(row=4, column=0, sticky="w", pady=8)
        ttk.Combobox(
            form,
            textvariable=self.optimizer_var,
            state="readonly",
            values=["Adam", "SGD", "RMSprop", "Nadam"],
            width=35
        ).grid(row=4, column=1, sticky="ew", pady=8)

        ttk.Label(form, text="Learning Rate:", font=("Arial", 10, "bold")).grid(row=5, column=0, sticky="w", pady=8)
        ttk.Entry(form, textvariable=self.lr_var, width=38).grid(row=5, column=1, sticky="ew", pady=8)

        form.columnconfigure(1, weight=1)

        button_frame = ttk.Frame(frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(
            button_frame,
            text="Use Recommended Settings for Current Task",
            command=self.set_task_recommended_settings
        ).pack(side=tk.LEFT, padx=5)

        if self.task_type_var.get() == TASK_CLASSIFICATION:
            ttk.Button(
                button_frame,
                text="Use Recommended Binary Settings",
                command=self.set_binary_settings
            ).pack(side=tk.LEFT, padx=5)

        note = ScrolledText(frame, height=10, wrap=tk.WORD)
        note.pack(fill=tk.BOTH, expand=True, pady=15)

        note.insert(tk.END, "Advanced setting guide:\n\n")
        note.insert(tk.END, f"Task: {self.task_type_var.get()}\n")
        note.insert(tk.END, f"Input columns: {self.feature_cols}\n")
        note.insert(tk.END, f"Target columns: {self.target_cols}\n\n")
        if self.task_type_var.get() == TASK_CLASSIFICATION:
            estimated_classes = self.df[self.label_col].dropna().nunique()
            note.insert(tk.END, f"Detected classes: {estimated_classes}\n")
            note.insert(tk.END, "Use softmax + sparse categorical cross-entropy.\n")
        elif self.task_type_var.get() == TASK_AUTOENCODER:
            note.insert(
                tk.END,
                "The output is the reconstructed input. Use linear output and "
                "mean-squared error.\n"
            )
        else:
            note.insert(
                tk.END,
                "Continuous targets use linear output and mean-squared error. "
                "MAE, MSE, RMSE, and R² are reported in original units.\n"
            )
        note.insert(tk.END, "\nOptimizer: Adam\nLearning rate: 0.001\n")

    def build_step_4_custom_model_tab(self, parent):
        outer = ttk.Frame(parent)
        outer.pack(fill=tk.BOTH, expand=True, padx=15, pady=12)

        ai_designer = ttk.LabelFrame(
            outer,
            text="AI Neural-Network Designer",
        )
        ai_designer.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(
            ai_designer,
            text=(
                "Describe the architecture, accuracy/speed trade-off, preferred "
                "layers, model-size limit, or overfitting constraints:"
            ),
            wraplength=1000,
        ).pack(anchor="w", padx=8, pady=(6, 2))
        ttk.Entry(
            ai_designer,
            textvariable=self.custom_model_ai_prompt_var,
        ).pack(fill=tk.X, padx=8, pady=3)
        ai_model_action_row = ttk.Frame(ai_designer)
        ai_model_action_row.pack(fill=tk.X, padx=6, pady=(2, 7))
        ai_model_action_row.columnconfigure(0, weight=1)
        self.custom_model_ai_button = ttk.Button(
            ai_model_action_row,
            text="Generate Customized NN Model with AI",
            command=self.start_custom_model_ai_generation,
            state=(
                tk.NORMAL
                if self.ai_provider_var.get() in OFFICIAL_PROVIDER_SETTINGS
                else tk.DISABLED
            ),
        )
        self.custom_model_ai_button.grid(
            row=0,
            column=0,
            sticky="w",
            padx=2,
            pady=2,
        )
        ttk.Label(
            ai_model_action_row,
            text=(
                f"Provider: {self.ai_provider_var.get()} "
                f"| Model: {self.ai_model_var.get() or 'not configured'}"
            ),
            foreground="#245a85",
            wraplength=980,
        ).grid(
            row=1,
            column=0,
            sticky="w",
            padx=4,
            pady=(2, 0),
        )
        self.custom_model_ai_status_label = ttk.Label(
            ai_model_action_row,
            text=(
                "Describe the model, then generate it from the current task "
                "and columns. An analysis recommendation is not required."
            ),
            foreground="#7a4b00",
            wraplength=980,
        )
        self.custom_model_ai_status_label.grid(
            row=2,
            column=0,
            sticky="w",
            padx=4,
            pady=(1, 2),
        )

        safe_frame = ttk.LabelFrame(
            outer,
            text="Safe Custom Model Specification (Manual / AI)",
        )
        safe_frame.pack(fill=tk.BOTH, expand=False, pady=(0, 8))
        safe_summary = (
            json.dumps(self.safe_model_spec, indent=2)
            if self.safe_model_spec is not None
            else "No safe custom model has been created yet."
        )
        self.safe_model_summary_text = ScrolledText(
            safe_frame,
            wrap=tk.NONE,
            height=5,
            font=("Consolas", 9),
        )
        self.safe_model_summary_text.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=8,
            pady=6,
        )
        self.safe_model_summary_text.insert(tk.END, safe_summary)
        ttk.Button(
            safe_frame,
            text="Validate / Apply Edited Safe JSON",
            command=self.apply_edited_safe_model_spec,
            state=(
                tk.NORMAL
                if self.safe_model_spec is not None
                else tk.DISABLED
            ),
        ).pack(side=tk.RIGHT, padx=8, pady=6)

        top_row = ttk.Frame(outer)
        top_row.pack(fill=tk.X)

        ttk.Label(
            top_row,
            text="Custom model input format:",
            font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Combobox(
            top_row,
            textvariable=self.custom_model_input_mode_var,
            state="readonly",
            values=["Window-based (3D)", "Row-based (2D)"],
            width=25
        ).pack(side=tk.LEFT)

        ttk.Label(
            outer,
            text=(
                "Safe Custom Model uses the reviewed declarative specification "
                "above and does not require AI. Custom Python Model uses the "
                "expert code editor below and runs locally with your Python "
                "permissions."
            ),
            foreground="dark red",
            wraplength=900
        ).pack(anchor="w", pady=(8, 5))

        ttk.Label(
            outer,
            text=(
                "Required function: build_custom_model(input_shape, output_units, "
                "hidden_activation, output_activation, dropout_rate)"
            ),
            font=("Arial", 10, "bold")
        ).pack(anchor="w", pady=(5, 2))

        button_row = ttk.Frame(outer)
        button_row.pack(fill=tk.X, pady=5)

        ttk.Button(
            button_row,
            text="Create Safe Model Template",
            command=self.load_safe_model_template
        ).pack(side=tk.LEFT, padx=3)

        ttk.Button(
            button_row,
            text="Load Expert Python Template",
            command=self.load_custom_model_template
        ).pack(side=tk.LEFT, padx=3)

        ttk.Button(
            button_row,
            text="Validate Custom Model",
            command=self.validate_custom_model_preview
        ).pack(side=tk.LEFT, padx=3)

        self.custom_model_code_text = ScrolledText(
            outer,
            wrap=tk.NONE,
            height=9,
            font=("Consolas", 10)
        )
        self.custom_model_code_text.pack(
            fill=tk.BOTH,
            expand=True,
            pady=5
        )
        self.custom_model_code_text.insert(tk.END, self.custom_model_code)

        self.custom_model_preview_text = ScrolledText(
            outer,
            wrap=tk.WORD,
            height=6
        )
        self.custom_model_preview_text.pack(fill=tk.X, pady=(5, 0))
        self.custom_model_preview_text.insert(
            tk.END,
            "The model summary will appear here after validation."
        )

    def start_custom_model_ai_generation(self):
        if not self.feature_cols:
            messagebox.showwarning(
                "AI Neural-Network Designer",
                "Confirm the input columns before generating a model.",
            )
            return
        self.start_component_ai_generation(
            component="model",
            customization_prompt=self.custom_model_ai_prompt_var.get(),
            input_columns=list(self.feature_cols),
        )

    def get_custom_model_code(self):
        if hasattr(self, "custom_model_code_text"):
            return self.custom_model_code_text.get("1.0", tk.END).strip()
        return self.custom_model_code.strip()

    def validate_safe_model_for_current_settings(self, model_spec):
        """Validate safe model JSON against current task/columns, without AI."""
        recommendation = self.build_current_generation_recommendation(
            input_columns=list(self.feature_cols),
            component="model",
        )
        generation = {
            "summary": (
                "Safe model validated from the user's current task and columns."
            ),
            "filter_spec": make_safe_filter_template([]),
            "model_spec": model_spec,
        }
        generation = validate_ai_generation(
            generation,
            self.dataset_profile,
            recommendation,
        )
        self.safe_model_generation = generation
        self.safe_model_recommendation = recommendation
        self.safe_model_spec = generation["model_spec"]
        return self.safe_model_spec

    def load_safe_model_template(self):
        """Start an editable allowlisted model from current user selections."""
        try:
            if not self.feature_cols:
                raise ValueError(
                    "Select and confirm at least one input column first."
                )
            recommendation = self.build_current_generation_recommendation(
                input_columns=list(self.feature_cols),
                component="model",
            )
            model_spec = make_safe_model_template(
                self.custom_model_input_mode_var.get(),
                self.task_type_var.get(),
            )
            generation = {
                "summary": (
                    "User-created safe neural-network template. No AI "
                    "recommendation or API request was used."
                ),
                "filter_spec": make_safe_filter_template([]),
                "model_spec": model_spec,
            }
            generation = validate_ai_generation(
                generation,
                self.dataset_profile,
                recommendation,
            )
            self.safe_model_generation = generation
            self.safe_model_recommendation = recommendation
            self.safe_model_spec = generation["model_spec"]
            self.model_type_var.set(MODEL_TYPE_SAFE_CUSTOM)
            if hasattr(self, "safe_model_summary_text"):
                self.safe_model_summary_text.delete("1.0", tk.END)
                self.safe_model_summary_text.insert(
                    tk.END,
                    json.dumps(self.safe_model_spec, indent=2),
                )
            if hasattr(self, "custom_model_ai_status_label"):
                self.custom_model_ai_status_label.config(
                    text=(
                        "Manual safe template created. Edit the layers, then "
                        "validate the model summary."
                    )
                )
            self.validate_safe_model_preview()
        except Exception as exc:
            messagebox.showerror("Safe Model Template Error", str(exc))

    def load_custom_model_template(self):
        self.custom_model_code = DEFAULT_CUSTOM_MODEL_CODE
        if hasattr(self, "custom_model_code_text"):
            self.custom_model_code_text.delete("1.0", tk.END)
            self.custom_model_code_text.insert(tk.END, self.custom_model_code)

    def validate_custom_model_preview(self):
        try:
            custom_code = self.get_custom_model_code()
            input_mode = self.custom_model_input_mode_var.get()
            feature_count = max(1, len(self.feature_cols))
            window_size = max(2, int(self.window_size_var.get()))

            if input_mode == "Window-based (3D)":
                input_shape = (window_size, feature_count)
            else:
                input_shape = (feature_count,)

            output_units = (
                self.get_auto_output_units()
                if self.output_mode_var.get() == "Auto"
                else int(self.output_units_var.get())
            )

            model = build_custom_model_from_code(
                custom_code=custom_code,
                input_shape=input_shape,
                output_units=output_units,
                hidden_activation=self.hidden_activation_var.get(),
                output_activation=self.output_activation_var.get(),
                dropout_rate=float(self.dropout_var.get())
            )

            summary_lines = []
            model.summary(print_fn=summary_lines.append)

            self.custom_model_code = custom_code
            self.custom_model_preview_text.delete("1.0", tk.END)
            self.custom_model_preview_text.insert(
                tk.END,
                "Custom model validated successfully.\n\n"
                f"Test input shape: {input_shape}\n\n"
                + "\n".join(summary_lines)
            )

            messagebox.showinfo(
                "Custom Model Valid",
                "The custom Keras model was validated successfully."
            )

        except Exception as exc:
            messagebox.showerror("Custom Model Error", str(exc))

    def apply_edited_safe_model_spec(self):
        try:
            if hasattr(self, "safe_model_summary_text"):
                text_value = self.safe_model_summary_text.get(
                    "1.0", tk.END
                ).strip()
                if not text_value:
                    raise ValueError(
                        "The safe model specification is empty."
                    )
                try:
                    edited_model_spec = json.loads(text_value)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"The edited safe model is not valid JSON: {exc}"
                    ) from exc
            else:
                edited_model_spec = self.safe_model_spec

            if edited_model_spec is None:
                raise ValueError(
                    "Create a safe model template or generate one with AI first."
                )
            self.safe_model_spec = self.validate_safe_model_for_current_settings(
                edited_model_spec
            )
            self.custom_model_input_mode_var.set(
                self.safe_model_spec["input_mode"]
            )
            self.model_type_var.set(MODEL_TYPE_SAFE_CUSTOM)
            self.validate_safe_model_preview()
        except Exception as exc:
            messagebox.showerror("Safe Custom Model Edit Error", str(exc))

    def validate_safe_model_preview(self):
        try:
            if self.safe_model_spec is None:
                raise ValueError(
                    "Create a safe model template or generate one with AI first."
                )
            self.safe_model_spec = self.validate_safe_model_for_current_settings(
                self.safe_model_spec
            )
            feature_count = max(1, len(self.feature_cols))
            window_size = max(2, int(self.window_size_var.get()))
            input_shape = (
                (window_size, feature_count)
                if self.safe_model_spec["input_mode"] == "Window-based (3D)"
                else (feature_count,)
            )
            output_units = (
                self.get_auto_output_units()
                if self.output_mode_var.get() == "Auto"
                else int(self.output_units_var.get())
            )
            model = build_safe_model_from_spec(
                model_spec=self.safe_model_spec,
                input_shape=input_shape,
                output_units=output_units,
                output_activation=self.output_activation_var.get(),
                task_type=self.task_type_var.get(),
            )
            summary_lines = []
            model.summary(print_fn=summary_lines.append)
            self.custom_model_preview_text.delete("1.0", tk.END)
            self.custom_model_preview_text.insert(
                tk.END,
                "Safe custom model validated successfully.\n\n"
                f"Test input shape: {input_shape}\n\n"
                + "\n".join(summary_lines),
            )
            messagebox.showinfo(
                "Safe Custom Model Valid",
                "The allowlisted model was built and shape-validated locally.",
            )
        except Exception as exc:
            messagebox.showerror("Safe Custom Model Error", str(exc))

    def build_step_4_guide_tab(self, parent):
        guide = ScrolledText(parent, wrap=tk.WORD)
        guide.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        guide.insert(tk.END, "Current dataset information:\n\n")
        guide.insert(tk.END, f"Task: {self.task_type_var.get()}\n")
        guide.insert(tk.END, f"Input columns: {self.feature_cols}\n")
        guide.insert(tk.END, f"Target columns: {self.target_cols}\n\n")

        guide.insert(tk.END, "Model setting:\n")
        guide.insert(tk.END, f"- Model: {self.model_type_var.get()}\n")
        guide.insert(tk.END, "- Window Size: 200\n")
        guide.insert(tk.END, "- Stride: 50\n")
        guide.insert(tk.END, "- Epochs: 30\n")
        guide.insert(tk.END, "- Batch Size: 32\n")
        guide.insert(tk.END, "- Validation Split: 20\n\n")

        guide.insert(tk.END, "Output/loss setting:\n")
        guide.insert(tk.END, "- Output Units Mode: Auto\n")
        guide.insert(
            tk.END,
            f"- Output Activation: {self.output_activation_var.get()}\n"
        )
        guide.insert(tk.END, f"- Loss Function: {self.loss_var.get()}\n")
        guide.insert(tk.END, "- Optimizer: Adam\n")
        guide.insert(tk.END, "- Learning Rate: 0.001\n\n")

        guide.insert(tk.END, "Evaluation:\n")
        if self.task_type_var.get() == TASK_CLASSIFICATION:
            guide.insert(
                tk.END,
                "Accuracy, weighted F1, classification report, and confusion matrix.\n"
            )
        elif self.task_type_var.get() == TASK_AUTOENCODER:
            guide.insert(
                tk.END,
                "Reconstruction error and percentile-based anomaly threshold.\n"
            )
        else:
            guide.insert(tk.END, "MAE, MSE, RMSE, R², and actual-vs-predicted plots.\n")

    def get_auto_output_units(self):
        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            return len(self.image_class_names)
        task_type = self.task_type_var.get()
        if task_type == TASK_CLASSIFICATION:
            return int(self.df[self.label_col].dropna().nunique())
        if task_type == TASK_FORECASTING:
            return len(self.target_cols) * int(self.forecast_horizon_var.get())
        if task_type == TASK_AUTOENCODER:
            return len(self.feature_cols)
        return len(self.target_cols)

    def set_task_recommended_settings(self):
        self.apply_task_defaults()
        self.output_units_var.set(str(self.get_auto_output_units()))
        self.optimizer_var.set("Adam")
        self.lr_var.set("0.001")
        messagebox.showinfo(
            "Updated",
            f"Recommended {self.task_type_var.get()} settings applied."
        )

    def set_multiclass_settings(self):
        estimated_classes = self.df[self.label_col].dropna().nunique()

        self.output_mode_var.set("Auto")
        self.output_units_var.set(str(estimated_classes))
        self.output_activation_var.set("softmax")
        self.loss_var.set("sparse_categorical_crossentropy")
        self.optimizer_var.set("Adam")
        self.lr_var.set("0.001")

        messagebox.showinfo(
            "Updated",
            "Recommended multi-class settings applied."
        )

    def set_binary_settings(self):
        self.output_mode_var.set("Custom")
        self.output_units_var.set("1")
        self.output_activation_var.set("sigmoid")
        self.loss_var.set("binary_crossentropy")
        self.optimizer_var.set("Adam")
        self.lr_var.set("0.001")

        messagebox.showinfo(
            "Updated",
            "Recommended binary settings applied."
        )

    def validate_step_4(self):
        self.collect_early_stopping_settings()
        model_type = self.model_type_var.get()
        task_type = self.task_type_var.get()

        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            if model_type not in SUPPORTED_IMAGE_MODEL_TYPES:
                raise ValueError("Choose a supported image classification model.")
            if task_type != TASK_CLASSIFICATION:
                raise ValueError(
                    "Photo training currently supports image classification."
                )
            image_height = int(self.image_height_var.get())
            image_width = int(self.image_width_var.get())
            if image_height < 32 or image_width < 32:
                raise ValueError(
                    "Image height and width must each be at least 32 pixels."
                )
            if (
                model_type in (
                    IMAGE_MODEL_MOBILENET,
                    IMAGE_MODEL_EFFICIENTNET,
                )
                and self.image_color_mode_var.get() != "RGB"
            ):
                raise ValueError(
                    "Transfer-learning image models require RGB colour mode."
                )
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            validation_split = float(self.validation_split_var.get()) / 100
            dropout_rate = float(self.dropout_var.get())
            learning_rate = float(self.lr_var.get())
            if epochs <= 0:
                raise ValueError("Epochs must be larger than 0.")
            if batch_size <= 0:
                raise ValueError("Batch size must be larger than 0.")
            if not 0.0 < validation_split < 0.8:
                raise ValueError(
                    "Validation split must be between 1% and 79%."
                )
            if not 0.0 <= dropout_rate < 1.0:
                raise ValueError(
                    "Dropout rate must be between 0 and 0.99."
                )
            if learning_rate <= 0:
                raise ValueError("Learning rate must be larger than 0.")
            if len(self.image_class_names) < 2:
                raise ValueError(
                    "Image classification requires at least two classes."
                )
            self.output_units_var.set(str(len(self.image_class_names)))
            self.output_activation_var.set("softmax")
            self.loss_var.set("sparse_categorical_crossentropy")
            return

        window_size = int(self.window_size_var.get())
        stride = int(self.stride_var.get())
        epochs = int(self.epochs_var.get())
        batch_size = int(self.batch_size_var.get())

        validation_split_percent = float(self.validation_split_var.get())
        validation_split = validation_split_percent / 100

        dropout_rate = float(self.dropout_var.get())
        learning_rate = float(self.lr_var.get())

        output_mode = self.output_mode_var.get()
        output_activation = self.output_activation_var.get()
        loss_name = self.loss_var.get()

        if output_mode == "Auto":
            output_units = self.get_auto_output_units()
            self.output_units_var.set(str(output_units))
        else:
            output_units = int(self.output_units_var.get())

        custom_model_uses_windows = (
            model_type in (
                "Custom Python Model",
                MODEL_TYPE_SAFE_CUSTOM,
                MODEL_TYPE_SAFE_AI_LEGACY,
            )
            and self.custom_model_input_mode_var.get() == "Window-based (3D)"
        )

        window_models = [
            "CNN",
            "LSTM",
            "CNN-LSTM",
            "LSTM Autoencoder"
        ]
        if (
            model_type in window_models
            or custom_model_uses_windows
            or task_type == TASK_FORECASTING
        ):
            if window_size <= 1:
                raise ValueError("Window size must be larger than 1.")

            if stride <= 0:
                raise ValueError("Stride must be larger than 0.")

        if task_type == TASK_AUTOENCODER:
            if model_type not in (
                "Dense Autoencoder",
                "LSTM Autoencoder",
                MODEL_TYPE_SAFE_CUSTOM,
                MODEL_TYPE_SAFE_AI_LEGACY,
                "Custom Python Model"
            ):
                raise ValueError(
                    "For the autoencoder task, choose Dense Autoencoder, "
                    "LSTM Autoencoder, or a compatible Custom Python Model."
                )
        elif model_type in ("Dense Autoencoder", "LSTM Autoencoder"):
            raise ValueError(
                "Autoencoder model architectures can only be used with the "
                "Autoencoder / Anomaly Detection task."
            )

        if model_type == "Custom Python Model":
            custom_code = self.get_custom_model_code()
            if not custom_code:
                raise ValueError("Custom model code is empty.")

            self.custom_model_code = custom_code

            test_input_shape = (
                (window_size, max(1, len(self.feature_cols)))
                if custom_model_uses_windows
                else (max(1, len(self.feature_cols)),)
            )

            build_custom_model_from_code(
                custom_code=custom_code,
                input_shape=test_input_shape,
                output_units=output_units,
                hidden_activation=self.hidden_activation_var.get(),
                output_activation=output_activation,
                dropout_rate=dropout_rate
            )

        if model_type in (MODEL_TYPE_SAFE_CUSTOM, MODEL_TYPE_SAFE_AI_LEGACY):
            if self.safe_model_spec is None:
                raise ValueError(
                    "Create a safe model template or generate one with AI in "
                    "the Customize Model tab first."
                )
            if hasattr(self, "safe_model_summary_text"):
                try:
                    edited_model_spec = json.loads(
                        self.safe_model_summary_text.get(
                            "1.0", tk.END
                        ).strip()
                    )
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"The edited safe model is not valid JSON: {exc}"
                    ) from exc
            else:
                edited_model_spec = self.safe_model_spec
            self.safe_model_spec = self.validate_safe_model_for_current_settings(
                edited_model_spec
            )
            self.custom_model_input_mode_var.set(
                self.safe_model_spec["input_mode"]
            )
            safe_model_uses_windows = (
                self.safe_model_spec["input_mode"] == "Window-based (3D)"
            )
            if safe_model_uses_windows and window_size <= 1:
                raise ValueError("Window size must be larger than 1.")
            if safe_model_uses_windows and stride <= 0:
                raise ValueError("Stride must be larger than 0.")
            safe_input_shape = (
                (window_size, max(1, len(self.feature_cols)))
                if safe_model_uses_windows
                else (max(1, len(self.feature_cols)),)
            )
            build_safe_model_from_spec(
                model_spec=self.safe_model_spec,
                input_shape=safe_input_shape,
                output_units=output_units,
                output_activation=output_activation,
                task_type=task_type,
            )

        if epochs <= 0:
            raise ValueError("Epochs must be larger than 0.")

        if batch_size <= 0:
            raise ValueError("Batch size must be larger than 0.")

        if validation_split <= 0 or validation_split >= 0.8:
            raise ValueError("Validation split must be between 1% and 79%.")

        if dropout_rate < 0 or dropout_rate >= 1:
            raise ValueError("Dropout rate must be between 0 and 0.99.")

        if learning_rate <= 0:
            raise ValueError("Learning rate must be larger than 0.")

        if output_units <= 0:
            raise ValueError("Output units must be larger than 0.")

        if task_type == TASK_CLASSIFICATION:
            estimated_classes = int(
                self.df[self.label_col].dropna().nunique()
            )
            if estimated_classes < 2:
                raise ValueError(
                    "Classification requires at least two distinct classes."
                )

            if loss_name == "binary_crossentropy":
                if estimated_classes != 2 or output_units != 1:
                    raise ValueError(
                        "Binary cross-entropy requires exactly two classes "
                        "and one output unit."
                    )
                if output_activation != "sigmoid":
                    raise ValueError(
                        "Binary cross-entropy requires sigmoid output activation."
                    )
            else:
                if loss_name not in (
                    "sparse_categorical_crossentropy",
                    "categorical_crossentropy"
                ):
                    raise ValueError(
                        "Classification requires a categorical or binary "
                        "cross-entropy loss."
                    )
                if output_units != estimated_classes:
                    raise ValueError(
                        "Classification output units must equal the number "
                        f"of classes ({estimated_classes})."
                    )
                if output_activation != "softmax":
                    raise ValueError(
                        "Multi-class classification requires softmax activation."
                    )
        else:
            if loss_name not in (
                "mean_squared_error",
                "mean_absolute_error"
            ):
                raise ValueError(
                    f"{task_type} requires mean_squared_error or "
                    "mean_absolute_error."
                )
            if output_activation != "linear":
                raise ValueError(
                    f"{task_type} should use linear output activation."
                )

            expected_units = self.get_auto_output_units()
            if (
                task_type != TASK_AUTOENCODER
                and output_units != expected_units
            ):
                raise ValueError(
                    f"Auto output units for {task_type} must be "
                    f"{expected_units}."
                )
