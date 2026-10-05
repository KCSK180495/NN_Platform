"""Ui / wizard filter workspace for NN Training Studio."""

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from pathlib import Path
from tkinter.scrolledtext import ScrolledText
from datetime import datetime
from tkinter import filedialog
import json
from tkinter import messagebox
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import threading
import tkinter as tk
from tkinter import ttk
from nn_training_studio.ai_providers import (
    format_ai_generation_report,
)
from nn_training_studio.ai_validation import (
    validate_ai_generation,
    validate_ai_recommendation,
)
from nn_training_studio.analysis import (
    analyze_dataset_profile,
    format_dataset_analysis_report,
)
from nn_training_studio.constants import (
    CUSTOM_FILTER_MODE_EXPERT,
    CUSTOM_FILTER_MODE_SAFE_AI,
    DATA_MODE_TABULAR,
    DEFAULT_CUSTOM_FILTER_CODE,
    FILTER_EXPORT_APPEND,
    FILTER_EXPORT_MODES,
    FILTER_EXPORT_REPLACE,
    FILTER_EXPORT_SELECTED,
    FILTER_PRESET_SCHEMA_VERSION,
    OFFICIAL_PROVIDER_SETTINGS,
    SUPPORTED_FILTER_METHODS,
    TASK_AUTOENCODER,
    TASK_CLASSIFICATION,
    TASK_TYPES,
    WORKSPACE_FILTER,
)
from nn_training_studio.filters import (
    apply_safe_filter_spec,
    apply_selected_filter,
    compare_filter_result,
    execute_custom_filter_code,
)
from nn_training_studio.templates import (
    make_safe_filter_template,
    make_safe_model_template,
)


class FilterWorkspaceMixin:
    """FilterWorkspace behavior for the main application."""

    def show_filter_workspace(self):
        if self.training_running or self.ai_request_running:
            messagebox.showinfo(
                "Operation Running",
                "Wait for the active operation to finish before opening the "
                "filter workspace.",
            )
            return
        self.current_view = WORKSPACE_FILTER
        self.clear_content()
        self.title_label.config(text="Filter & Export Data")
        self.progress_label.config(
            text=(
                "Load → select columns → configure → preview → export or "
                "continue to signal training"
            )
        )
        self.set_workflow_navigation_visible(False)
        self.home_button.config(state=tk.NORMAL)
        self.settings_button.config(state=tk.NORMAL)
        self.build_filter_workspace()

    def build_filter_workspace(self):
        toolbar = ttk.Frame(self.content_frame)
        toolbar.pack(fill=tk.X, padx=8, pady=(8, 5))
        ttk.Button(
            toolbar,
            text="Load CSV / Excel",
            command=self.load_filter_tool_data,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            toolbar,
            text="Load Filter Preset",
            command=self.load_filter_tool_preset,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            toolbar,
            text="Save Filter Preset",
            command=self.save_filter_tool_preset,
        ).pack(side=tk.LEFT, padx=4)
        source_text = (
            os.path.basename(self.filter_tool_source_path)
            if self.filter_tool_source_path
            else "No dataset loaded"
        )
        self.filter_tool_file_label = ttk.Label(
            toolbar,
            text=source_text,
        )
        self.filter_tool_file_label.pack(side=tk.LEFT, padx=12)

        # Keep the most important standalone-filter actions above the
        # resizable panes. On shorter displays, widgets placed below an
        # expanding PanedWindow can otherwise be pushed outside the window.
        quick_actions = ttk.Frame(self.content_frame)
        quick_actions.pack(fill=tk.X, padx=8, pady=(0, 5))
        ttk.Button(
            quick_actions,
            text="Preview / Apply Filter",
            command=self.preview_filter_tool_result,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            quick_actions,
            text="Save Filtered Dataset...",
            command=self.save_filter_tool_data,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            quick_actions,
            text="Continue to Signal Training →",
            command=self.continue_filter_to_training,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(
            quick_actions,
            textvariable=self.filter_tool_export_status_var,
            foreground="#245a85",
        ).pack(side=tk.RIGHT, padx=8)

        pane = ttk.PanedWindow(self.content_frame, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=5)
        controls = ttk.LabelFrame(pane, text="Filter Pipeline")
        preview = ttk.LabelFrame(pane, text="Before / After Preview")
        pane.add(controls, weight=2)
        pane.add(preview, weight=3)

        columns_frame = ttk.LabelFrame(
            controls,
            text="1. Numeric Columns to Filter",
        )
        columns_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.filter_tool_columns_listbox = tk.Listbox(
            columns_frame,
            selectmode=tk.MULTIPLE,
            exportselection=False,
            height=7,
        )
        self.filter_tool_columns_listbox.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=(6, 0),
            pady=6,
        )
        column_scroll = ttk.Scrollbar(
            columns_frame,
            orient=tk.VERTICAL,
            command=self.filter_tool_columns_listbox.yview,
        )
        column_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 6), pady=6)
        self.filter_tool_columns_listbox.configure(
            yscrollcommand=column_scroll.set
        )
        if self.filter_tool_source_df is not None:
            for column in self.filter_tool_source_df.select_dtypes(
                include=[np.number]
            ).columns:
                self.filter_tool_columns_listbox.insert(tk.END, column)

        column_actions = ttk.Frame(controls)
        column_actions.pack(fill=tk.X, padx=8, pady=(0, 5))
        ttk.Button(
            column_actions,
            text="Select All Numeric",
            command=lambda: self.filter_tool_columns_listbox.selection_set(
                0, tk.END
            ),
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            column_actions,
            text="Clear Selection",
            command=lambda: self.filter_tool_columns_listbox.selection_clear(
                0, tk.END
            ),
        ).pack(side=tk.LEFT, padx=3)

        filter_tabs = ttk.Notebook(controls)
        filter_tabs.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        built_in_tab = ttk.Frame(filter_tabs)
        ai_tab = ttk.Frame(filter_tabs)
        expert_tab = ttk.Frame(filter_tabs)
        export_tab = ttk.Frame(filter_tabs)
        filter_tabs.add(built_in_tab, text="Built-in")
        filter_tabs.add(ai_tab, text="AI / Safe JSON")
        filter_tabs.add(expert_tab, text="Expert Python")
        filter_tabs.add(export_tab, text="Export")

        settings = ttk.LabelFrame(built_in_tab, text="2. Built-in Filter")
        settings.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        settings.columnconfigure(1, weight=1)
        ttk.Label(settings, text="Method:").grid(
            row=0, column=0, sticky="w", padx=7, pady=5
        )
        ttk.Combobox(
            settings,
            textvariable=self.filter_tool_method_var,
            values=SUPPORTED_FILTER_METHODS,
            state="readonly",
        ).grid(row=0, column=1, sticky="ew", padx=7, pady=5)

        parameter_rows = [
            ("Moving-average window:", self.filter_tool_moving_window_var),
            ("EMA span:", self.filter_tool_ema_span_var),
            ("Median window:", self.filter_tool_median_window_var),
            ("Kalman process noise Q:", self.filter_tool_kalman_q_var),
            ("Kalman measurement noise R:", self.filter_tool_kalman_r_var),
        ]
        for row_index, (label, variable) in enumerate(parameter_rows, start=1):
            ttk.Label(settings, text=label).grid(
                row=row_index,
                column=0,
                sticky="w",
                padx=7,
                pady=3,
            )
            ttk.Entry(settings, textvariable=variable).grid(
                row=row_index,
                column=1,
                sticky="ew",
                padx=7,
                pady=3,
            )

        ai_designer = ttk.LabelFrame(
            ai_tab,
            text="3. AI-Assisted Safe Filter",
        )
        ai_designer.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        ttk.Label(
            ai_designer,
            text=(
                "The AI receives a compact statistical profile, selected "
                "column names, and your instructions—not raw dataset rows. "
                "Its declarative JSON can only use validated filter operations."
            ),
            wraplength=540,
            foreground="#245a85",
        ).pack(anchor="w", padx=7, pady=(6, 3))
        sampling_row = ttk.Frame(ai_designer)
        sampling_row.pack(fill=tk.X, padx=7, pady=3)
        ttk.Label(
            sampling_row,
            text="Sampling frequency (Hz, optional):",
        ).pack(side=tk.LEFT)
        ttk.Entry(
            sampling_row,
            textvariable=self.filter_tool_sampling_frequency_var,
            width=16,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Label(
            sampling_row,
            text="Required for Butterworth and notch filters.",
            foreground="#7a4b00",
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(
            ai_designer,
            text="Describe the noise and features that must be preserved:",
        ).pack(anchor="w", padx=7, pady=(5, 2))
        ttk.Entry(
            ai_designer,
            textvariable=self.filter_tool_ai_prompt_var,
        ).pack(fill=tk.X, padx=7, pady=3)
        ai_action_row = ttk.Frame(ai_designer)
        ai_action_row.pack(fill=tk.X, padx=5, pady=4)
        self.filter_tool_ai_button = ttk.Button(
            ai_action_row,
            text="Generate Filter with AI",
            command=self.start_filter_tool_ai_generation,
            state=(
                tk.NORMAL
                if self.ai_provider_var.get() in OFFICIAL_PROVIDER_SETTINGS
                else tk.DISABLED
            ),
        )
        self.filter_tool_ai_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(
            ai_action_row,
            text="Create Manual Safe Template",
            command=self.create_filter_tool_safe_template,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            ai_action_row,
            text="Validate JSON & Preview",
            command=self.preview_filter_tool_result,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            ai_action_row,
            text="View Generated Details",
            command=self.show_filter_tool_ai_details,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(
            ai_designer,
            text="Enable this safe JSON pipeline after the built-in filter",
            variable=self.filter_tool_safe_enabled_var,
        ).pack(anchor="w", padx=7, pady=(2, 3))
        self.filter_tool_ai_status_label = ttk.Label(
            ai_designer,
            text=self.filter_tool_ai_status,
            wraplength=540,
            foreground="#7a4b00",
        )
        self.filter_tool_ai_status_label.pack(
            anchor="w",
            padx=7,
            pady=(1, 4),
        )
        self.filter_tool_safe_json_text = ScrolledText(
            ai_designer,
            height=9,
            wrap=tk.NONE,
            font=("Consolas", 9),
        )
        self.filter_tool_safe_json_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=7,
            pady=(0, 7),
        )
        self.filter_tool_safe_json_text.insert(
            tk.END,
            json.dumps(
                self.filter_tool_safe_filter_spec
                or {
                    "status": (
                        "Generate a filter with AI or create a manual safe "
                        "template."
                    )
                },
                indent=2,
            ),
        )

        custom = ttk.LabelFrame(
            expert_tab,
            text="4. Optional Expert Python Filter",
        )
        custom.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        ttk.Label(
            custom,
            text=(
                "Expert mode executes local Python with your application "
                "permissions. Use only code you trust. For normal use, prefer "
                "the validated AI / Safe JSON tab."
            ),
            wraplength=540,
            foreground="dark red",
        ).pack(anchor="w", padx=7, pady=(6, 2))
        ttk.Checkbutton(
            custom,
            text="Run custom_filter(df, selected_columns) after built-in filter",
            variable=self.filter_tool_custom_enabled_var,
        ).pack(anchor="w", padx=7, pady=4)
        self.filter_tool_custom_code_text = ScrolledText(
            custom,
            height=8,
            wrap=tk.NONE,
            font=("Consolas", 9),
        )
        self.filter_tool_custom_code_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=7,
            pady=(0, 7),
        )
        self.filter_tool_custom_code_text.insert(
            tk.END,
            self.filter_tool_custom_code,
        )

        export_settings = ttk.LabelFrame(
            export_tab,
            text="5. Export Layout",
        )
        export_settings.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)
        ttk.Combobox(
            export_settings,
            textvariable=self.filter_tool_export_mode_var,
            values=FILTER_EXPORT_MODES,
            state="readonly",
        ).pack(fill=tk.X, padx=7, pady=5)
        suffix_row = ttk.Frame(export_settings)
        suffix_row.pack(fill=tk.X, padx=7, pady=3)
        ttk.Label(suffix_row, text="New-column suffix:").pack(side=tk.LEFT)
        ttk.Entry(
            suffix_row,
            textvariable=self.filter_tool_suffix_var,
            width=18,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(
            export_settings,
            text="Save a JSON processing report beside the dataset",
            variable=self.filter_tool_save_report_var,
        ).pack(anchor="w", padx=7, pady=(3, 6))
        ttk.Button(
            export_settings,
            text="Apply Filter and Save Dataset...",
            command=self.save_filter_tool_data,
        ).pack(anchor="w", padx=7, pady=(5, 4))
        ttk.Label(
            export_settings,
            textvariable=self.filter_tool_export_status_var,
            wraplength=520,
            foreground="#245a85",
        ).pack(anchor="w", padx=7, pady=(2, 7))

        action_row = ttk.Frame(preview)
        action_row.pack(fill=tk.X, padx=7, pady=7)
        ttk.Button(
            action_row,
            text="Preview / Apply",
            command=self.preview_filter_tool_result,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            action_row,
            text="Plot Selected Columns",
            command=self.plot_filter_tool_result,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            action_row,
            text="Reset Result",
            command=self.reset_filter_tool_result,
        ).pack(side=tk.LEFT, padx=3)

        preview_notebook = ttk.Notebook(preview)
        preview_notebook.pack(fill=tk.BOTH, expand=True, padx=7, pady=5)
        data_preview_tab = ttk.Frame(preview_notebook)
        signal_plot_tab = ttk.Frame(preview_notebook)
        preview_notebook.add(data_preview_tab, text="Data & Metrics")
        preview_notebook.add(signal_plot_tab, text="Signal Plot")
        self.filter_tool_preview_notebook = preview_notebook

        self.filter_tool_preview_text = ScrolledText(
            data_preview_tab,
            wrap=tk.NONE,
        )
        self.filter_tool_preview_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=3,
            pady=3,
        )
        plot_controls = ttk.Frame(signal_plot_tab)
        plot_controls.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(plot_controls, text="Start row:").pack(side=tk.LEFT)
        ttk.Entry(
            plot_controls,
            textvariable=self.filter_tool_plot_start_var,
            width=9,
        ).pack(side=tk.LEFT, padx=(3, 7))
        ttk.Label(plot_controls, text="End row:").pack(side=tk.LEFT)
        ttk.Entry(
            plot_controls,
            textvariable=self.filter_tool_plot_end_var,
            width=9,
        ).pack(side=tk.LEFT, padx=(3, 7))
        ttk.Label(plot_controls, text="Max points:").pack(side=tk.LEFT)
        ttk.Entry(
            plot_controls,
            textvariable=self.filter_tool_plot_max_points_var,
            width=9,
        ).pack(side=tk.LEFT, padx=(3, 7))
        ttk.Combobox(
            plot_controls,
            textvariable=self.filter_tool_plot_mode_var,
            values=["Overlay", "Difference"],
            state="readonly",
            width=12,
        ).pack(side=tk.LEFT, padx=3)
        x_axis_values = ["Sample index"]
        if self.filter_tool_source_df is not None:
            x_axis_values.extend(
                str(column)
                for column in self.filter_tool_source_df.columns
            )
        ttk.Label(plot_controls, text="X axis:").pack(
            side=tk.LEFT, padx=(7, 2)
        )
        ttk.Combobox(
            plot_controls,
            textvariable=self.filter_tool_x_axis_var,
            values=x_axis_values,
            state="readonly",
            width=16,
        ).pack(side=tk.LEFT, padx=3)
        self.filter_tool_plot_host = ttk.Frame(signal_plot_tab)
        self.filter_tool_plot_host.pack(
            fill=tk.BOTH,
            expand=True,
            padx=3,
            pady=3,
        )
        self.filter_tool_plot_placeholder = ttk.Label(
            self.filter_tool_plot_host,
            text=(
                "The signal comparison will appear here after Preview / Apply.\n"
                "Select at least one numeric column first."
            ),
            foreground="#666666",
            justify=tk.CENTER,
        )
        self.filter_tool_plot_placeholder.pack(expand=True, padx=20, pady=20)
        if self.filter_tool_source_df is None:
            self.filter_tool_preview_text.insert(
                tk.END,
                "Load a CSV or Excel file to begin.\n\n"
                "The original file is never overwritten automatically.",
            )
        else:
            self.update_filter_tool_preview(
                self.filter_tool_source_df,
                "Source dataset loaded. Configure a filter and choose Preview.",
            )

    def load_filter_tool_data(self):
        path = filedialog.askopenfilename(
            title="Select Dataset to Filter",
            filetypes=[
                ("Data Files", "*.csv *.xlsx *.xls"),
                ("CSV Files", "*.csv"),
                ("Excel Files", "*.xlsx *.xls"),
            ],
        )
        if not path:
            return
        try:
            suffix = Path(path).suffix.lower()
            if suffix == ".csv":
                dataframe = pd.read_csv(path)
            elif suffix in (".xlsx", ".xls"):
                dataframe = pd.read_excel(path)
            else:
                raise ValueError("Choose a CSV, XLSX, or XLS file.")
            if dataframe.empty:
                raise ValueError("The selected dataset is empty.")
            if not list(dataframe.select_dtypes(include=[np.number]).columns):
                raise ValueError(
                    "The selected dataset has no numeric columns to filter."
                )
            self.filter_tool_source_df = dataframe
            self.filter_tool_result_df = None
            self.filter_tool_source_path = str(Path(path).resolve())
            self.filter_tool_safe_filter_spec = None
            self.filter_tool_safe_enabled_var.set(False)
            self.filter_tool_ai_report = ""
            self.filter_tool_ai_provider_summary = None
            self.filter_tool_ai_status = (
                "Dataset loaded. Select columns, describe the required "
                "filter, then generate or create a safe template."
            )
            self.filter_tool_export_status_var.set(
                "Dataset loaded; no filtered export saved yet."
            )
            self.filter_tool_plot_start_var.set("0")
            self.filter_tool_plot_end_var.set("")
            self.filter_tool_x_axis_var.set("Sample index")
            self.show_filter_workspace()
        except Exception as exc:
            messagebox.showerror("Dataset Load Error", str(exc))

    def get_filter_tool_selected_columns(self):
        if not hasattr(self, "filter_tool_columns_listbox"):
            return []
        return [
            self.filter_tool_columns_listbox.get(index)
            for index in self.filter_tool_columns_listbox.curselection()
        ]

    def get_filter_tool_sampling_frequency(self):
        value = self.filter_tool_sampling_frequency_var.get().strip()
        if not value:
            return None
        sampling_frequency = float(value)
        if sampling_frequency <= 0:
            raise ValueError("Sampling frequency must be greater than zero.")
        return sampling_frequency

    def build_filter_tool_generation_context(self, selected_columns):
        """Build an isolated, filter-only AI context for standalone datasets."""
        if self.filter_tool_source_df is None:
            raise ValueError("Load a CSV or Excel dataset first.")
        selected_columns = list(dict.fromkeys(selected_columns))
        if not selected_columns:
            raise ValueError(
                "Select the numeric signal columns that the AI may filter."
            )
        profile = analyze_dataset_profile(
            self.filter_tool_source_df,
            sampling_frequency=self.get_filter_tool_sampling_frequency(),
        )
        invalid_columns = [
            column
            for column in selected_columns
            if column not in profile["numeric_columns"]
        ]
        if invalid_columns:
            raise ValueError(
                "AI filters may use numeric columns only: "
                + ", ".join(invalid_columns)
            )
        profile_by_name = {
            item["name"]: item for item in profile["column_profiles"]
        }
        protected_columns = []
        for column in selected_columns:
            column_profile = profile_by_name[column]
            column_lower = str(column).lower()
            if (
                column_profile.get("is_time_like")
                or column_profile.get("is_id_like")
                or any(
                    token in column_lower
                    for token in ("label", "target", "class")
                )
            ):
                protected_columns.append(column)
        if protected_columns:
            raise ValueError(
                "Do not filter time, ID, label, target, or class columns: "
                + ", ".join(protected_columns)
            )

        task = {
            "task_type": TASK_AUTOENCODER,
            "confidence": 1.0,
            "input_columns": selected_columns,
            "target_columns": [],
            "reasons": [
                "Standalone filtering uses the selected signal columns only."
            ],
            "warnings": [],
            "alternatives": [
                task_type
                for task_type in TASK_TYPES
                if task_type != TASK_AUTOENCODER
            ][:3],
        }
        filter_plan = {
            "method": "No filter",
            "columns": [],
            "moving_average_window": 5,
            "ema_span": 10,
            "median_window": 5,
            "kalman_q": 0.00001,
            "kalman_r": 0.01,
            "confidence": 1.0,
            "reasons": [
                "The user requested a customized standalone filter."
            ],
            "risks": [
                "Filtering may suppress transients, harmonics, or fault content."
            ],
            "validation_checks": [
                "Compare plots, correlation, RMSE, and standard deviation."
            ],
        }
        model_plan = {
            "model_type": "LSTM Autoencoder",
            "window_size": 200,
            "stride": 50,
            "epochs": 30,
            "batch_size": 32,
            "validation_split_percent": 20.0,
            "hidden_activation": "relu",
            "dropout_rate": 0.2,
            "output_activation": "linear",
            "loss_function": "mean_squared_error",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "feature_scaling": "StandardScaler",
            "target_scaling": "No normalization",
            "confidence": 1.0,
            "reasons": [
                "A minimal companion model context is required by the schema."
            ],
            "warnings": [],
        }
        recommendation = validate_ai_recommendation(
            {
                "summary": (
                    "Standalone filter context built from the current dataset "
                    "profile and the user's selected columns."
                ),
                "task": task,
                "filter": filter_plan,
                "model": model_plan,
                "future_generation": {
                    "custom_filter_would_help": True,
                    "custom_filter_reason": (
                        "The user requested a customized filter."
                    ),
                    "custom_model_would_help": False,
                    "custom_model_reason": (
                        "No model is needed in the standalone filter workspace."
                    ),
                },
            },
            profile,
        )
        return profile, recommendation

    def get_filter_tool_safe_spec_from_editor(self):
        if not hasattr(self, "filter_tool_safe_json_text"):
            return self.filter_tool_safe_filter_spec
        editor_value = self.filter_tool_safe_json_text.get(
            "1.0", tk.END
        ).strip()
        if not editor_value:
            raise ValueError("The safe filter JSON is empty.")
        try:
            filter_spec = json.loads(editor_value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"The safe filter specification is not valid JSON: {exc}"
            ) from exc
        if not isinstance(filter_spec, dict):
            raise ValueError("The safe filter specification must be an object.")
        return filter_spec

    def validate_filter_tool_safe_spec(self, selected_columns):
        filter_spec = self.get_filter_tool_safe_spec_from_editor()
        if filter_spec is None:
            raise ValueError(
                "Generate a filter with AI or create a manual safe template."
            )
        profile, recommendation = self.build_filter_tool_generation_context(
            selected_columns
        )
        generation = validate_ai_generation(
            {
                "summary": (
                    "Standalone safe filter validated from the editable JSON."
                ),
                "filter_spec": filter_spec,
                "model_spec": make_safe_model_template(
                    "Window-based (3D)",
                    TASK_AUTOENCODER,
                ),
            },
            profile,
            recommendation,
        )
        self.filter_tool_safe_filter_spec = generation["filter_spec"]
        return self.filter_tool_safe_filter_spec, profile, recommendation

    def create_filter_tool_safe_template(self):
        try:
            selected_columns = self.get_filter_tool_selected_columns()
            profile, recommendation = (
                self.build_filter_tool_generation_context(selected_columns)
            )
            generation = validate_ai_generation(
                {
                    "summary": (
                        "Manual safe filter template created without an API "
                        "request."
                    ),
                    "filter_spec": make_safe_filter_template(selected_columns),
                    "model_spec": make_safe_model_template(
                        "Window-based (3D)",
                        TASK_AUTOENCODER,
                    ),
                },
                profile,
                recommendation,
            )
            self.filter_tool_safe_filter_spec = generation["filter_spec"]
            self.filter_tool_safe_enabled_var.set(True)
            if hasattr(self, "filter_tool_safe_json_text"):
                self.filter_tool_safe_json_text.delete("1.0", tk.END)
                self.filter_tool_safe_json_text.insert(
                    tk.END,
                    json.dumps(self.filter_tool_safe_filter_spec, indent=2),
                )
            self.filter_tool_ai_status = (
                "Manual safe template created. Edit its operations and "
                "parameters, then validate and preview."
            )
            if hasattr(self, "filter_tool_ai_status_label"):
                self.filter_tool_ai_status_label.config(
                    text=self.filter_tool_ai_status
                )
        except Exception as exc:
            messagebox.showerror("Safe Filter Template Error", str(exc))

    def start_filter_tool_ai_generation(self):
        try:
            if self.ai_request_running:
                raise ValueError("Wait for the active AI request to finish.")
            selected_columns = self.get_filter_tool_selected_columns()
            profile, recommendation = (
                self.build_filter_tool_generation_context(selected_columns)
            )
            prompt = self.filter_tool_ai_prompt_var.get().strip()
            if not prompt:
                raise ValueError(
                    "Describe the required filter before requesting AI help."
                )
            provider_result = self.get_component_ai_provider()
            if provider_result is None:
                return
            provider, provider_name, model_name, base_url = provider_result
            complete_prompt = (
                "Generate a customized FILTER only for these confirmed "
                f"columns: {selected_columns}. The filter_spec is the required "
                "deliverable. Return a minimal compatible model_spec only "
                "because the shared schema requires it.\n\n"
                "USER FILTER REQUIREMENTS:\n"
                + prompt
            )
            self.ai_request_running = True
            self.filter_tool_ai_status = (
                "Analysing the compact profile and generating a validated "
                "safe filter..."
            )
            if hasattr(self, "filter_tool_ai_status_label"):
                self.filter_tool_ai_status_label.config(
                    text=self.filter_tool_ai_status
                )
            if hasattr(self, "filter_tool_ai_button"):
                self.filter_tool_ai_button.config(state=tk.DISABLED)
            self.home_button.config(state=tk.DISABLED)
            self.settings_button.config(state=tk.DISABLED)
            threading.Thread(
                target=self.filter_tool_ai_generation_worker,
                args=(
                    provider,
                    json.loads(json.dumps(profile)),
                    json.loads(json.dumps(recommendation)),
                    complete_prompt,
                    provider_name,
                    model_name,
                    base_url,
                ),
                daemon=True,
            ).start()
        except Exception as exc:
            messagebox.showerror("AI Filter Designer", str(exc))

    def filter_tool_ai_generation_worker(
        self,
        provider,
        profile,
        recommendation,
        prompt,
        provider_name,
        model_name,
        base_url,
    ):
        try:
            generation = provider.generate(
                profile,
                recommendation,
                user_goal=prompt,
            )
            self.ui_queue.put((
                "filter_tool_ai_finished",
                {
                    "generation": generation,
                    "profile": profile,
                    "recommendation": recommendation,
                    "report": format_ai_generation_report(
                        generation,
                        provider_name,
                        model_name,
                    ),
                    "provider_summary": {
                        "provider": provider_name,
                        "model": model_name,
                        "base_url": base_url,
                        "profile_only": True,
                        "generated_python": False,
                        "declarative_safe_builders": True,
                        "component": "standalone_filter",
                    },
                },
            ))
        except Exception as exc:
            self.ui_queue.put(("filter_tool_ai_failed", str(exc)))

    def filter_tool_ai_generation_finished(self, payload):
        self.ai_request_running = False
        generation = validate_ai_generation(
            json.loads(json.dumps(payload["generation"])),
            payload["profile"],
            payload["recommendation"],
        )
        self.filter_tool_safe_filter_spec = generation["filter_spec"]
        self.filter_tool_safe_enabled_var.set(bool(
            self.filter_tool_safe_filter_spec["pipelines"]
        ))
        self.filter_tool_ai_report = payload["report"]
        self.filter_tool_ai_provider_summary = payload["provider_summary"]
        provider_name = payload["provider_summary"]["provider"]
        self.ai_connection_state = "connected"
        self.ai_connection_message = (
            f"{provider_name} connected successfully."
        )
        self.ai_last_connection_check = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.ai_last_connection_details = (
            "A filter-generation request completed successfully."
        )
        if hasattr(self, "filter_tool_safe_json_text"):
            self.filter_tool_safe_json_text.delete("1.0", tk.END)
            self.filter_tool_safe_json_text.insert(
                tk.END,
                json.dumps(self.filter_tool_safe_filter_spec, indent=2),
            )
        self.filter_tool_ai_status = (
            "AI filter generated and validated. Review the JSON, preview the "
            "plots and preservation metrics, then export."
        )
        if hasattr(self, "filter_tool_ai_status_label"):
            self.filter_tool_ai_status_label.config(
                text=self.filter_tool_ai_status
            )
        if hasattr(self, "filter_tool_ai_button"):
            self.filter_tool_ai_button.config(state=tk.NORMAL)
        self.home_button.config(state=tk.NORMAL)
        self.settings_button.config(state=tk.NORMAL)
        self.update_provider_status_display()
        if self.filter_tool_safe_filter_spec["pipelines"]:
            self.preview_filter_tool_result()
        else:
            messagebox.showinfo(
                "AI Filter Result",
                "The AI recommended no additional filtering. The original "
                "signals remain unchanged.",
            )
        # Show every AI result—including a recommendation to leave the data
        # unchanged—in a resizable window. The embedded editor remains
        # available for quick edits, but it can be short on smaller displays.
        details_window = getattr(
            self,
            "filter_tool_ai_details_window",
            None,
        )
        if details_window is not None and details_window.winfo_exists():
            details_window.destroy()
        self.filter_tool_ai_details_window = None
        self.show_filter_tool_ai_details()

    def show_filter_tool_ai_details(self):
        """Show the full generated JSON and AI explanation in a large editor."""
        existing = getattr(self, "filter_tool_ai_details_window", None)
        if existing is not None and existing.winfo_exists():
            existing.deiconify()
            existing.lift()
            existing.focus_force()
            return

        details_window = tk.Toplevel(self)
        self.filter_tool_ai_details_window = details_window
        details_window.title("AI Filter Details")
        details_window.geometry("920x680")
        details_window.minsize(720, 500)
        details_window.transient(self)

        heading = ttk.Frame(details_window)
        heading.pack(fill=tk.X, padx=12, pady=(10, 5))
        ttk.Label(
            heading,
            text="Generated AI Filter",
            font=("Arial", 14, "bold"),
        ).pack(anchor="w")
        selected_columns = self.get_filter_tool_selected_columns()
        ttk.Label(
            heading,
            text=(
                "Selected columns: "
                + (", ".join(selected_columns) if selected_columns else "None")
                + "\nEdit the safe JSON below, then validate it before use."
            ),
            foreground="#245a85",
        ).pack(anchor="w", pady=(3, 0))

        details_notebook = ttk.Notebook(details_window)
        details_notebook.pack(
            fill=tk.BOTH,
            expand=True,
            padx=12,
            pady=6,
        )
        json_tab = ttk.Frame(details_notebook)
        explanation_tab = ttk.Frame(details_notebook)
        details_notebook.add(json_tab, text="Filter JSON")
        details_notebook.add(explanation_tab, text="AI Explanation")

        json_editor = ScrolledText(
            json_tab,
            wrap=tk.NONE,
            font=("Consolas", 10),
            undo=True,
        )
        json_editor.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        if hasattr(self, "filter_tool_safe_json_text"):
            json_value = self.filter_tool_safe_json_text.get(
                "1.0", tk.END
            ).strip()
        else:
            json_value = json.dumps(
                self.filter_tool_safe_filter_spec
                or {
                    "status": (
                        "Generate a filter with AI or create a manual safe "
                        "template."
                    )
                },
                indent=2,
            )
        json_editor.insert(tk.END, json_value)

        explanation = ScrolledText(
            explanation_tab,
            wrap=tk.WORD,
            font=("Consolas", 10),
        )
        explanation.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        provider_text = json.dumps(
            self.filter_tool_ai_provider_summary,
            indent=2,
            default=str,
        ) if self.filter_tool_ai_provider_summary else "Not available"
        explanation.insert(
            tk.END,
            "STATUS\n"
            + self.filter_tool_ai_status
            + "\n\nPROVIDER\n"
            + provider_text
            + "\n\nGENERATION REPORT\n"
            + (
                self.filter_tool_ai_report
                or "No AI explanation is available yet."
            ),
        )
        explanation.config(state=tk.DISABLED)

        status_var = tk.StringVar(
            value="Review the JSON, then choose Apply Edits or Validate & Preview."
        )
        ttk.Label(
            details_window,
            textvariable=status_var,
            wraplength=860,
            foreground="#7a4b00",
        ).pack(fill=tk.X, padx=14, pady=(0, 4))

        def apply_detail_edits(preview=False):
            try:
                editor_value = json_editor.get("1.0", tk.END).strip()
                parsed = json.loads(editor_value)
                if not isinstance(parsed, dict):
                    raise ValueError(
                        "The safe filter specification must be a JSON object."
                    )
                if not hasattr(self, "filter_tool_safe_json_text"):
                    raise ValueError(
                        "Reopen the Filter & Export workspace and try again."
                    )
                self.filter_tool_safe_json_text.delete("1.0", tk.END)
                self.filter_tool_safe_json_text.insert(
                    tk.END,
                    json.dumps(parsed, indent=2),
                )
                selected = self.get_filter_tool_selected_columns()
                validated, _, _ = self.validate_filter_tool_safe_spec(selected)
                self.filter_tool_safe_enabled_var.set(
                    bool(validated.get("pipelines"))
                )
                normalized_json = json.dumps(validated, indent=2)
                self.filter_tool_safe_json_text.delete("1.0", tk.END)
                self.filter_tool_safe_json_text.insert(
                    tk.END,
                    normalized_json,
                )
                json_editor.delete("1.0", tk.END)
                json_editor.insert(tk.END, normalized_json)
                status_var.set(
                    "JSON edits applied and validated successfully."
                )
                if preview:
                    self.preview_filter_tool_result()
            except Exception as exc:
                status_var.set(f"Validation failed: {exc}")
                messagebox.showerror(
                    "AI Filter JSON Error",
                    str(exc),
                    parent=details_window,
                )

        buttons = ttk.Frame(details_window)
        buttons.pack(fill=tk.X, padx=12, pady=(2, 10))
        ttk.Button(
            buttons,
            text="Apply JSON Edits",
            command=apply_detail_edits,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            buttons,
            text="Validate & Preview",
            command=lambda: apply_detail_edits(preview=True),
        ).pack(side=tk.LEFT, padx=3)

        def close_details():
            self.filter_tool_ai_details_window = None
            details_window.destroy()

        ttk.Button(
            buttons,
            text="Close",
            command=close_details,
        ).pack(side=tk.RIGHT, padx=3)
        details_window.protocol("WM_DELETE_WINDOW", close_details)

    def filter_tool_ai_generation_failed(self, error_message):
        self.ai_request_running = False
        self.filter_tool_ai_status = "AI filter generation failed."
        if hasattr(self, "filter_tool_ai_status_label"):
            self.filter_tool_ai_status_label.config(
                text=self.filter_tool_ai_status
            )
        if hasattr(self, "filter_tool_ai_button"):
            self.filter_tool_ai_button.config(state=tk.NORMAL)
        self.home_button.config(state=tk.NORMAL)
        self.settings_button.config(state=tk.NORMAL)
        messagebox.showerror("AI Filter Generation Error", error_message)

    def apply_filter_tool_pipeline(self):
        if self.filter_tool_source_df is None:
            raise ValueError("Load a CSV or Excel dataset first.")
        selected_columns = self.get_filter_tool_selected_columns()
        method = self.filter_tool_method_var.get()
        safe_enabled = bool(self.filter_tool_safe_enabled_var.get())
        custom_enabled = bool(self.filter_tool_custom_enabled_var.get())
        if safe_enabled and custom_enabled:
            raise ValueError(
                "Choose either the AI / Safe JSON pipeline or Expert Python "
                "for the optional custom stage. Stacking both would make the "
                "saved inference package ambiguous."
            )
        if method != "No filter" and not selected_columns:
            raise ValueError(
                "Select at least one numeric column for the built-in filter."
            )
        if safe_enabled and not selected_columns:
            raise ValueError(
                "Select the columns that the safe filter is allowed to modify."
            )
        if custom_enabled and not selected_columns:
            raise ValueError(
                "Select at least one column for the custom filter."
            )

        result = apply_selected_filter(
            self.filter_tool_source_df,
            selected_columns,
            method,
            int(self.filter_tool_moving_window_var.get()),
            int(self.filter_tool_ema_span_var.get()),
            int(self.filter_tool_median_window_var.get()),
            float(self.filter_tool_kalman_q_var.get()),
            float(self.filter_tool_kalman_r_var.get()),
        )
        processed_columns = list(selected_columns)
        if safe_enabled:
            safe_spec, profile, _ = self.validate_filter_tool_safe_spec(
                selected_columns
            )
            result = apply_safe_filter_spec(
                result,
                safe_spec,
                sampling_frequency=profile.get("sampling_frequency_hz"),
                allowed_columns=selected_columns,
            )
            for pipeline in safe_spec["pipelines"]:
                column = pipeline["column"]
                if column not in processed_columns:
                    processed_columns.append(column)
        if custom_enabled:
            custom_code = self.filter_tool_custom_code_text.get(
                "1.0", tk.END
            ).strip()
            result = execute_custom_filter_code(
                result,
                selected_columns,
                custom_code,
            )
            self.filter_tool_custom_code = custom_code
        self.filter_tool_result_df = result
        return result, processed_columns

    def update_filter_tool_preview(self, dataframe, message):
        if not hasattr(self, "filter_tool_preview_text"):
            return
        self.filter_tool_preview_text.delete("1.0", tk.END)
        self.filter_tool_preview_text.insert(
            tk.END,
            message
            + "\n\n"
            + f"Shape: {dataframe.shape[0]} rows × "
            + f"{dataframe.shape[1]} columns\n\n"
            + dataframe.head(20).to_string(index=False),
        )

    def preview_filter_tool_result(self):
        try:
            result, selected_columns = self.apply_filter_tool_pipeline()
            metrics_text = ""
            if selected_columns:
                metrics = compare_filter_result(
                    self.filter_tool_source_df,
                    result,
                    selected_columns,
                )
                metrics_text = "\n\nPreservation checks:\n" + json.dumps(
                    metrics,
                    indent=2,
                    default=str,
                )
            self.update_filter_tool_preview(
                result,
                "Filter pipeline applied successfully.\n"
                f"Method: {self.filter_tool_method_var.get()}\n"
                f"Columns: {selected_columns}"
                + metrics_text,
            )
            if selected_columns:
                # Preview and plotting are one user action. Rendering after the
                # current callback lets Tk finish notebook geometry first.
                self.after_idle(
                    lambda plot_result=result.copy(), plot_columns=list(
                        selected_columns
                    ): self.plot_filter_tool_result(
                        result=plot_result,
                        selected_columns=plot_columns,
                    )
                )
        except Exception as exc:
            if hasattr(self, "filter_tool_preview_text"):
                self.filter_tool_preview_text.delete("1.0", tk.END)
                self.filter_tool_preview_text.insert(
                    tk.END,
                    "Filter preview failed.\n\n" + str(exc),
                )
            messagebox.showerror("Filter Preview Error", str(exc))

    def reset_filter_tool_result(self):
        self.filter_tool_result_df = None
        if self.filter_tool_source_df is not None:
            self.update_filter_tool_preview(
                self.filter_tool_source_df,
                "Result reset. The source dataset is unchanged.",
            )
        if hasattr(self, "filter_tool_plot_host"):
            for child in self.filter_tool_plot_host.winfo_children():
                child.destroy()

    def plot_filter_tool_result(self, result=None, selected_columns=None):
        try:
            if result is None or selected_columns is None:
                result, selected_columns = self.apply_filter_tool_pipeline()
            if not selected_columns:
                raise ValueError("Select at least one numeric column to plot.")
            if not hasattr(self, "filter_tool_plot_host"):
                raise ValueError("Reopen the filter workspace and try again.")
            row_count = len(result)
            start_value = self.filter_tool_plot_start_var.get().strip()
            end_value = self.filter_tool_plot_end_var.get().strip()
            start_row = int(start_value or 0)
            end_row = int(end_value) if end_value else row_count
            max_points = int(
                self.filter_tool_plot_max_points_var.get().strip() or 5000
            )
            if start_row < 0 or start_row >= row_count:
                raise ValueError(
                    f"Start row must be between 0 and {row_count - 1}."
                )
            if end_row <= start_row or end_row > row_count:
                raise ValueError(
                    f"End row must be greater than {start_row} and no more "
                    f"than {row_count}."
                )
            if max_points < 100:
                raise ValueError("Max points must be at least 100.")
            step = max(
                1,
                int(np.ceil((end_row - start_row) / max_points)),
            )
            row_positions = np.arange(start_row, end_row, step)
            x_axis_name = self.filter_tool_x_axis_var.get()
            if (
                x_axis_name != "Sample index"
                and x_axis_name in self.filter_tool_source_df.columns
            ):
                x_values = self.filter_tool_source_df.iloc[
                    row_positions
                ][x_axis_name].to_numpy()
                x_label = x_axis_name
            else:
                x_values = row_positions
                x_label = "Sample index"

            if vars(self).get("filter_tool_plot_figure") is not None:
                plt.close(self.filter_tool_plot_figure)
            for child in self.filter_tool_plot_host.winfo_children():
                child.destroy()
            columns_to_plot = selected_columns[:4]
            figure, axes = plt.subplots(
                len(columns_to_plot),
                1,
                figsize=(9.0, max(3.4, 2.6 * len(columns_to_plot))),
                squeeze=False,
                sharex=True,
            )
            plot_mode = self.filter_tool_plot_mode_var.get()
            for row_index, column in enumerate(columns_to_plot):
                axis = axes[row_index][0]
                before = pd.to_numeric(
                    self.filter_tool_source_df.iloc[row_positions][column],
                    errors="coerce",
                ).to_numpy(dtype=float)
                after = pd.to_numeric(
                    result.iloc[row_positions][column],
                    errors="coerce",
                ).to_numpy(dtype=float)
                if plot_mode == "Difference":
                    axis.plot(
                        x_values,
                        after - before,
                        label="After − Before",
                        color="#c23b22",
                        linewidth=1.0,
                    )
                    axis.axhline(0.0, color="#555555", linewidth=0.7)
                else:
                    axis.plot(
                        x_values,
                        before,
                        label="Before",
                        color="#777777",
                        alpha=0.72,
                        linewidth=0.9,
                    )
                    axis.plot(
                        x_values,
                        after,
                        label="After",
                        color="#1769aa",
                        linewidth=1.15,
                    )
                axis.set_title(str(column), loc="left", fontsize=10)
                axis.set_ylabel("Amplitude")
                axis.legend(loc="upper right")
                axis.grid(alpha=0.25)
            axes[-1][0].set_xlabel(x_label)
            figure.suptitle(
                (
                    f"Filter comparison: rows {start_row}–{end_row - 1}"
                    + (
                        f" (displaying every {step}th point)"
                        if step > 1
                        else ""
                    )
                ),
                fontsize=11,
            )
            figure.tight_layout()
            canvas = FigureCanvasTkAgg(
                figure,
                master=self.filter_tool_plot_host,
            )
            canvas.draw()
            toolbar = NavigationToolbar2Tk(
                canvas,
                self.filter_tool_plot_host,
                pack_toolbar=False,
            )
            toolbar.update()
            toolbar.pack(side=tk.TOP, fill=tk.X)
            canvas.get_tk_widget().pack(
                side=tk.TOP,
                fill=tk.BOTH,
                expand=True,
            )
            self.filter_tool_plot_canvas = canvas
            self.filter_tool_plot_figure = figure
            self.filter_tool_plot_toolbar = toolbar
            if hasattr(self, "filter_tool_preview_notebook"):
                self.filter_tool_preview_notebook.select(1)
            self.filter_tool_plot_host.update_idletasks()
            canvas.draw_idle()
            self.after_idle(
                lambda current_canvas=canvas: self.redraw_embedded_canvas(
                    current_canvas
                )
            )
        except Exception as exc:
            self.show_embedded_plot_message(
                "filter_tool_plot_host",
                "Signal plot could not be rendered:\n" + str(exc),
            )
            messagebox.showerror("Filter Plot Error", str(exc))

    def build_filter_tool_export_dataframe(self, selected_columns):
        if self.filter_tool_result_df is None:
            raise ValueError("Preview or apply the filter before exporting.")
        mode = self.filter_tool_export_mode_var.get()
        if mode == FILTER_EXPORT_REPLACE:
            return self.filter_tool_result_df.copy()
        if not selected_columns:
            raise ValueError(
                "Select at least one filtered column for this export layout."
            )
        if mode == FILTER_EXPORT_SELECTED:
            return self.filter_tool_result_df[selected_columns].copy()
        if mode == FILTER_EXPORT_APPEND:
            suffix = self.filter_tool_suffix_var.get().strip()
            if not suffix:
                raise ValueError("Enter a suffix for appended filtered columns.")
            exported = self.filter_tool_source_df.copy()
            collisions = [
                column + suffix
                for column in selected_columns
                if column + suffix in exported.columns
            ]
            if collisions:
                raise ValueError(
                    "These appended column names already exist: "
                    + str(collisions)
                )
            for column in selected_columns:
                exported[column + suffix] = self.filter_tool_result_df[column]
            return exported
        raise ValueError("Choose a supported export layout.")

    def filter_tool_processing_report(self, selected_columns):
        return {
            "schema_version": FILTER_PRESET_SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_file": self.filter_tool_source_path,
            "source_rows": int(len(self.filter_tool_source_df)),
            "source_columns": list(self.filter_tool_source_df.columns),
            "selected_columns": list(selected_columns),
            "built_in_filter": {
                "method": self.filter_tool_method_var.get(),
                "moving_average_window": int(
                    self.filter_tool_moving_window_var.get()
                ),
                "ema_span": int(self.filter_tool_ema_span_var.get()),
                "median_window": int(
                    self.filter_tool_median_window_var.get()
                ),
                "kalman_q": float(self.filter_tool_kalman_q_var.get()),
                "kalman_r": float(self.filter_tool_kalman_r_var.get()),
            },
            "sampling_frequency_hz": (
                self.get_filter_tool_sampling_frequency()
            ),
            "safe_filter_enabled": bool(
                self.filter_tool_safe_enabled_var.get()
            ),
            "safe_filter_spec": (
                self.filter_tool_safe_filter_spec
                if self.filter_tool_safe_enabled_var.get()
                else None
            ),
            "ai_filter_request": (
                self.filter_tool_ai_prompt_var.get().strip()
            ),
            "ai_filter_provider": self.filter_tool_ai_provider_summary,
            "ai_filter_report": self.filter_tool_ai_report,
            "custom_filter_enabled": bool(
                self.filter_tool_custom_enabled_var.get()
            ),
            "custom_filter_code": (
                self.filter_tool_custom_code
                if self.filter_tool_custom_enabled_var.get()
                else ""
            ),
            "export_mode": self.filter_tool_export_mode_var.get(),
            "append_suffix": self.filter_tool_suffix_var.get(),
        }

    def save_filter_tool_data(self):
        try:
            _, selected_columns = self.apply_filter_tool_pipeline()
            exported = self.build_filter_tool_export_dataframe(
                selected_columns
            )
            path = filedialog.asksaveasfilename(
                title="Save Filtered Dataset",
                defaultextension=".csv",
                filetypes=[
                    ("CSV File", "*.csv"),
                    ("Excel Workbook", "*.xlsx"),
                ],
            )
            if not path:
                return
            suffix = Path(path).suffix.lower()
            if suffix == ".csv":
                exported.to_csv(path, index=False)
            elif suffix == ".xlsx":
                exported.to_excel(path, index=False)
            else:
                raise ValueError("Save the filtered data as CSV or XLSX.")

            report_path = None
            if self.filter_tool_save_report_var.get():
                report_path = str(Path(path).with_suffix("")) + (
                    "_processing_report.json"
                )
                with open(report_path, "w", encoding="utf-8") as report_file:
                    json.dump(
                        self.filter_tool_processing_report(selected_columns),
                        report_file,
                        indent=2,
                    )
            message = f"Filtered dataset saved:\n{path}"
            if report_path:
                message += f"\n\nProcessing report:\n{report_path}"
            self.filter_tool_export_status_var.set(
                f"Saved: {Path(path).name}"
            )
            messagebox.showinfo("Dataset Saved", message)
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc))

    def save_filter_tool_preset(self):
        try:
            if self.filter_tool_source_df is None:
                raise ValueError("Load a dataset before saving a preset.")
            selected_columns = self.get_filter_tool_selected_columns()
            if hasattr(self, "filter_tool_custom_code_text"):
                self.filter_tool_custom_code = (
                    self.filter_tool_custom_code_text.get(
                        "1.0", tk.END
                    ).strip()
                )
            preset = self.filter_tool_processing_report(selected_columns)
            preset["preset_type"] = "NNTrainingStudio.FilterPreset"
            path = filedialog.asksaveasfilename(
                title="Save Filter Preset",
                defaultextension=".json",
                filetypes=[("JSON Filter Preset", "*.json")],
            )
            if not path:
                return
            with open(path, "w", encoding="utf-8") as preset_file:
                json.dump(preset, preset_file, indent=2)
            messagebox.showinfo("Preset Saved", f"Filter preset saved:\n{path}")
        except Exception as exc:
            messagebox.showerror("Preset Save Error", str(exc))

    def load_filter_tool_preset(self):
        path = filedialog.askopenfilename(
            title="Load Filter Preset",
            filetypes=[("JSON Filter Preset", "*.json")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as preset_file:
                preset = json.load(preset_file)
            if preset.get("preset_type") != "NNTrainingStudio.FilterPreset":
                raise ValueError(
                    "This JSON file is not an NN Training Studio filter preset."
                )
            if int(preset.get("schema_version", -1)) != (
                FILTER_PRESET_SCHEMA_VERSION
            ):
                raise ValueError("The filter preset version is not supported.")
            settings = preset.get("built_in_filter", {})
            method = settings.get("method", "No filter")
            if method not in SUPPORTED_FILTER_METHODS:
                raise ValueError(f"Unsupported filter method: {method}")
            self.filter_tool_method_var.set(method)
            self.filter_tool_moving_window_var.set(
                str(settings.get("moving_average_window", 5))
            )
            self.filter_tool_ema_span_var.set(str(settings.get("ema_span", 10)))
            self.filter_tool_median_window_var.set(
                str(settings.get("median_window", 5))
            )
            self.filter_tool_kalman_q_var.set(
                str(settings.get("kalman_q", 0.00001))
            )
            self.filter_tool_kalman_r_var.set(
                str(settings.get("kalman_r", 0.01))
            )
            sampling_frequency = preset.get("sampling_frequency_hz")
            self.filter_tool_sampling_frequency_var.set(
                "" if sampling_frequency is None else str(sampling_frequency)
            )
            safe_filter_spec = preset.get("safe_filter_spec")
            if safe_filter_spec is not None and not isinstance(
                safe_filter_spec, dict
            ):
                raise ValueError(
                    "The preset safe_filter_spec must be a JSON object."
                )
            self.filter_tool_safe_filter_spec = safe_filter_spec
            self.filter_tool_safe_enabled_var.set(bool(
                preset.get("safe_filter_enabled", False)
                and safe_filter_spec is not None
            ))
            self.filter_tool_ai_prompt_var.set(
                str(
                    preset.get("ai_filter_request")
                    or self.filter_tool_ai_prompt_var.get()
                )
            )
            self.filter_tool_ai_provider_summary = preset.get(
                "ai_filter_provider"
            )
            self.filter_tool_ai_report = str(
                preset.get("ai_filter_report") or ""
            )
            self.filter_tool_custom_enabled_var.set(
                bool(preset.get("custom_filter_enabled", False))
            )
            self.filter_tool_custom_code = (
                preset.get("custom_filter_code")
                or DEFAULT_CUSTOM_FILTER_CODE
            )
            export_mode = preset.get(
                "export_mode",
                FILTER_EXPORT_REPLACE,
            )
            if export_mode in FILTER_EXPORT_MODES:
                self.filter_tool_export_mode_var.set(export_mode)
            self.filter_tool_suffix_var.set(
                str(preset.get("append_suffix", "_filtered"))
            )
            if self.current_view == WORKSPACE_FILTER:
                self.build_filter_workspace_after_preset(
                    preset.get("selected_columns", [])
                )
            messagebox.showinfo(
                "Preset Loaded",
                "The filter settings were loaded. Missing dataset columns are "
                "left unselected.",
            )
        except Exception as exc:
            messagebox.showerror("Preset Load Error", str(exc))

    def build_filter_workspace_after_preset(self, selected_columns):
        self.show_filter_workspace()
        available = {
            self.filter_tool_columns_listbox.get(index): index
            for index in range(self.filter_tool_columns_listbox.size())
        }
        for column in selected_columns:
            if column in available:
                self.filter_tool_columns_listbox.selection_set(
                    available[column]
                )

    def continue_filter_to_training(self):
        try:
            result, selected_columns = self.apply_filter_tool_pipeline()
            source_df = self.filter_tool_source_df.copy()
            built_in_df = apply_selected_filter(
                source_df,
                selected_columns,
                self.filter_tool_method_var.get(),
                int(self.filter_tool_moving_window_var.get()),
                int(self.filter_tool_ema_span_var.get()),
                int(self.filter_tool_median_window_var.get()),
                float(self.filter_tool_kalman_q_var.get()),
                float(self.filter_tool_kalman_r_var.get()),
            )
            self.reset_training_project(DATA_MODE_TABULAR)
            self.raw_df = source_df
            self.filtered_df = built_in_df
            self.custom_filtered_df = result.copy()
            self.df = result.copy()
            self.file_path = self.filter_tool_source_path
            standalone_sampling_frequency = (
                self.get_filter_tool_sampling_frequency()
            )
            self.sampling_frequency_var.set(
                ""
                if standalone_sampling_frequency is None
                else str(standalone_sampling_frequency)
            )
            self.dataset_profile = analyze_dataset_profile(
                source_df,
                sampling_frequency=standalone_sampling_frequency,
            )
            self.dataset_analysis_report = format_dataset_analysis_report(
                self.dataset_profile
            )
            self.filter_method_var.set(self.filter_tool_method_var.get())
            self.moving_window_var.set(
                self.filter_tool_moving_window_var.get()
            )
            self.ema_span_var.set(self.filter_tool_ema_span_var.get())
            self.median_window_var.set(
                self.filter_tool_median_window_var.get()
            )
            self.kalman_q_var.set(self.filter_tool_kalman_q_var.get())
            self.kalman_r_var.set(self.filter_tool_kalman_r_var.get())
            self.selected_filter_cols = list(selected_columns)
            custom_enabled = bool(
                self.filter_tool_custom_enabled_var.get()
            )
            safe_enabled = bool(
                self.filter_tool_safe_enabled_var.get()
            )
            self.custom_filter_enabled_var.set(
                safe_enabled or custom_enabled
            )
            self.custom_filter_mode_var.set(
                CUSTOM_FILTER_MODE_SAFE_AI
                if safe_enabled
                else CUSTOM_FILTER_MODE_EXPERT
            )
            self.custom_filter_code = self.filter_tool_custom_code
            self.selected_custom_filter_cols = (
                list(selected_columns)
                if safe_enabled or custom_enabled
                else []
            )
            self.safe_filter_spec = (
                json.loads(json.dumps(self.filter_tool_safe_filter_spec))
                if safe_enabled
                else None
            )
            self.initialize_signal_column_defaults()
            # Filtering is already complete. Continue at task/input selection;
            # the settings are retained for package inference and are not
            # applied a second time to the in-memory training dataframe.
            self.current_step = 4
            self.show_step()
        except Exception as exc:
            messagebox.showerror("Continue to Training Error", str(exc))

    def initialize_signal_column_defaults(self):
        """Choose safe initial feature and target columns for a signal table."""
        columns = list(self.df.columns)
        possible_labels = [
            "label_id",
            "label",
            "fault_label",
            "class",
            "target",
            "y",
        ]
        selected_label = next(
            (name for name in possible_labels if name in columns),
            columns[-1] if columns else None,
        )
        if selected_label is None:
            raise ValueError("The dataset has no columns.")
        self.label_var.set(selected_label)
        self.label_col = selected_label
        self.target_cols = [selected_label]
        self.feature_cols = [
            column
            for column in self.df.select_dtypes(include=[np.number]).columns
            if column != selected_label and "id" not in column.lower()
        ]
        if self.task_type_var.get() == TASK_CLASSIFICATION:
            estimated_classes = self.df[selected_label].dropna().nunique()
            self.output_units_var.set(str(max(1, estimated_classes)))
        else:
            self.output_units_var.set("1")
