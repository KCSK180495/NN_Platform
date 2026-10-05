"""Ui / deploy for NN Training Studio."""

from pathlib import Path
from tkinter.scrolledtext import ScrolledText
import ast
from datetime import datetime
from tkinter import filedialog
import hashlib
import json
from tkinter import messagebox
import pandas as pd
import queue
import shutil
import tempfile
import threading
import tkinter as tk
from tkinter import ttk
from urllib.parse import urlparse
from nn_training_studio.ai_transport import (
    _validate_provider_base_url,
)
from nn_training_studio.branding import (
    apply_window_branding,
)
from nn_training_studio.constants import (
    AI_API_PURPOSES,
    AI_CODE_MAX_SELECTED_FILES,
    AI_IMPLEMENTATION_APPLICATION_TYPES,
    AI_IMPLEMENTATION_DATA_SOURCES,
    AI_IMPLEMENTATION_RESULTS,
    APPLICATION_NAME,
    APP_VERSION,
    DATA_MODE_DETECTION,
    DATA_MODE_TABULAR,
    DEPLOY_EXPORT_FORMATS,
    DEPLOY_PROTOCOLS,
    DEPLOY_TARGETS,
    INTEGRATION_MODES,
    INTEGRATION_MODE_HYBRID,
    INTEGRATION_MODE_LOCAL_MODEL,
    INTEGRATION_MODE_PROVIDER_API,
    OFFICIAL_PROVIDER_SETTINGS,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from nn_training_studio.deployment import (
    deployment_compatibility_report,
    deployment_default_protocol,
    deployment_recommended_targets,
    deployment_required_inputs,
    generate_deployment_text_artifacts,
    inspect_saved_model_descriptor,
)
from nn_training_studio.integration import (
    build_ai_code_integration_diff,
    build_beginner_integration_request,
    build_complete_ai_package_inventory,
    collect_ai_integration_sources,
    copy_complete_ai_project,
    format_ai_project_summary,
    request_ai_code_integration,
    scan_ai_integration_files,
    scan_ai_integration_project,
    summarise_ai_integration_project,
    validate_ai_integrated_source_files,
    verify_complete_ai_package_zip,
)
from nn_training_studio.provider_artifacts import (
    generate_ai_provider_api_artifacts,
    provider_api_compatibility_report,
)
from nn_training_studio.results import (
    _result_json_default,
    zip_result_directory,
)
from nn_training_studio.ui.detection import (
    ObjectDetectionWindow,
)
from nn_training_studio.ui.evaluation import (
    ModelEvaluationWindow,
)


class DeployIntegrateWindow(tk.Toplevel):
    """Guided workspace for turning a saved model into an application package."""

    def __init__(self, parent, initial_model_path=None, initial_goal=None):
        super().__init__(parent)
        self.settings_owner = parent
        apply_window_branding(self)
        self.title(f"{APPLICATION_NAME} {APP_VERSION} — Deploy & Integrate")
        self.geometry("1240x860")
        self.minsize(980, 680)

        self.descriptor = None
        self.selected_model_path = None
        self.sample_path = None
        self.generated_files = {}
        self.current_generated_preview_name = None
        self.ai_code_project_root = None
        self.ai_code_scan = None
        self.ai_code_sources = []
        self.ai_code_result = None
        self.ai_code_validation = None
        self.ai_code_running = False
        self.integration_mode_var = tk.StringVar(
            value=INTEGRATION_MODE_LOCAL_MODEL
        )
        self.ai_api_purpose_var = tk.StringVar(value=AI_API_PURPOSES[0])
        self.provider_integration_status_var = tk.StringVar(value="")
        self.deploy_step_hint_var = tk.StringVar(value="")
        self.model_path_var = tk.StringVar(value="No model selected")
        self.model_summary_var = tk.StringVar(
            value="Select a model to begin the deployment workflow."
        )
        self.target_var = tk.StringVar(value=DEPLOY_TARGETS[0])
        self.protocol_var = tk.StringVar(value=DEPLOY_PROTOCOLS[0])
        self.export_format_var = tk.StringVar(
            value=DEPLOY_EXPORT_FORMATS[0]
        )
        self.mapping_source_var = tk.StringVar(value="")
        self.mapping_processing_var = tk.StringVar(
            value="Saved package preprocessing"
        )
        self.output_action_var = tk.StringVar(value="")
        self.compatibility_status_var = tk.StringVar(
            value="Compatibility has not been checked."
        )
        self.sample_status_var = tk.StringVar(
            value="No sample source selected."
        )
        self.package_status_var = tk.StringVar(
            value="No deployment package generated yet."
        )
        self.ai_code_project_var = tk.StringVar(
            value="No application project selected."
        )
        self.ai_code_project_summary_var = tk.StringVar(
            value="Select a folder or individual code files for a private local summary."
        )
        self.ai_code_application_type_var = tk.StringVar(
            value=AI_IMPLEMENTATION_APPLICATION_TYPES[0]
        )
        self.ai_code_result_type_var = tk.StringVar(
            value=AI_IMPLEMENTATION_RESULTS[0]
        )
        self.ai_code_data_source_var = tk.StringVar(
            value=AI_IMPLEMENTATION_DATA_SOURCES[0]
        )
        self.ai_code_goal_var = tk.StringVar(value="")
        self.ai_code_approval_var = tk.BooleanVar(value=False)
        self.ai_code_status_var = tk.StringVar(
            value="Select an application project and review the files to share."
        )
        self.ai_code_result_file_var = tk.StringVar(value="Integration plan")

        self.build_interface()
        if initial_goal:
            self.ai_code_goal_text.delete("1.0", tk.END)
            self.ai_code_goal_text.insert("1.0", str(initial_goal).strip())
            self.build_beginner_ai_code_request()
            self.notebook.select(self.project_tab)
        if initial_model_path:
            self.after_idle(
                lambda: self.inspect_model_path(initial_model_path)
            )

    def build_interface(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        header.columnconfigure(0, weight=1)
        ttk.Label(
            header,
            text="Deploy & Integrate Model",
            font=("Arial", 19, "bold"),
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text=(
                "Import application → choose model/API → validate → output code "
                "→ continue coding with AI assistance"
            ),
            foreground="#245a85",
        ).grid(row=1, column=0, sticky="w", pady=(3, 0))
        ttk.Label(
            header,
            text=APP_VERSION,
            font=("Arial", 10, "bold"),
            foreground="#245a85",
        ).grid(row=0, column=1, rowspan=2, sticky="ne")

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=14,
            pady=(0, 8),
        )
        self.project_tab = ttk.Frame(self.notebook)
        self.integration_tab = ttk.Frame(self.notebook)
        self.test_tab = ttk.Frame(self.notebook)
        self.output_tab = ttk.Frame(self.notebook)
        self.ai_code_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.project_tab, text="1. Project & Goal")
        self.notebook.add(self.integration_tab, text="2. Model / AI API")
        self.notebook.add(self.test_tab, text="3. Test & Validate")
        self.notebook.add(self.output_tab, text="4. Output Code")
        self.notebook.add(self.ai_code_tab, text="5. AI Coding (Optional)")

        self.integration_notebook = ttk.Notebook(self.integration_tab)
        self.integration_notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.model_tab = ttk.Frame(self.integration_notebook)
        self.mapping_tab = ttk.Frame(self.integration_notebook)
        self.integration_notebook.add(self.model_tab, text="Integration Choice")
        self.integration_notebook.add(self.mapping_tab, text="Input / Output Mapping")

        self.output_notebook = ttk.Notebook(self.output_tab)
        self.output_notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.code_tab = ttk.Frame(self.output_notebook)
        self.package_tab = ttk.Frame(self.output_notebook)
        self.output_notebook.add(self.code_tab, text="Generated Code")
        self.output_notebook.add(self.package_tab, text="Package & Export")

        self.build_project_tab()
        self.build_model_tab()
        self.build_mapping_tab()
        self.build_test_tab()
        self.build_code_tab()
        self.build_package_tab()
        self.build_ai_code_tab()

        footer = ttk.Frame(self)
        footer.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 12))
        footer.columnconfigure(0, weight=1)
        self.deploy_step_hint_label = ttk.Label(
            footer,
            textvariable=self.deploy_step_hint_var,
            foreground="#666666",
            wraplength=720,
        )
        self.deploy_step_hint_label.grid(row=0, column=0, sticky="w")
        self.deploy_back_button = ttk.Button(
            footer,
            text="← Back",
            command=lambda: self.move_deploy_step(-1),
            width=13,
        )
        self.deploy_back_button.grid(row=0, column=1, padx=4)
        self.deploy_next_button = ttk.Button(
            footer,
            text="Next →",
            command=lambda: self.move_deploy_step(1),
            width=13,
        )
        self.deploy_next_button.grid(row=0, column=2, padx=4)
        ttk.Button(
            footer,
            text="Close",
            command=self.destroy,
            width=11,
        ).grid(row=0, column=3, padx=(4, 0))
        self.notebook.bind("<<NotebookTabChanged>>", self.refresh_deploy_step)
        self.bind("<Alt-Left>", lambda _event: self.move_deploy_step(-1))
        self.bind("<Alt-Right>", lambda _event: self.move_deploy_step(1))
        self.after_idle(self.refresh_deploy_step)

    def move_deploy_step(self, change):
        tabs = self.notebook.tabs()
        try:
            current = self.notebook.index(self.notebook.select())
        except tk.TclError:
            current = 0
        target = max(0, min(len(tabs) - 1, current + int(change)))
        self.notebook.select(target)

    def refresh_deploy_step(self, _event=None):
        try:
            index = self.notebook.index(self.notebook.select())
        except tk.TclError:
            index = 0
        hints = [
            "Select the existing project or code files, then describe the application in plain language.",
            "Choose a trained model, an AI provider API, or both; then confirm the target and mappings.",
            "Run compatibility checks. Validate sample input when a local model is selected.",
            "Generate, review, copy, save, or package the integration output.",
            "Approve only the selected source files, then let the assistant propose complete reviewed changes.",
        ]
        self.deploy_step_hint_var.set(f"Step {index + 1} of 5 — {hints[index]}")
        self.deploy_back_button.config(
            state=tk.DISABLED if index == 0 else tk.NORMAL
        )
        self.deploy_next_button.config(
            state=tk.DISABLED if index == 4 else tk.NORMAL
        )

    def build_project_tab(self):
        intro = ttk.LabelFrame(
            self.project_tab,
            text="Start with the Existing Application",
        )
        intro.pack(fill=tk.X, padx=10, pady=(10, 5))
        ttk.Label(
            intro,
            text=(
                "Select a project folder or only the code files you want NN Studio "
                "to analyse. Then describe the result you want as if you were "
                "chatting with a developer. Technical terminology is optional."
            ),
            wraplength=1080,
            foreground="#245a85",
        ).pack(anchor="w", padx=10, pady=(8, 4))
        ttk.Label(
            intro,
            text=(
                "The scan is local. Secret/configuration files, credentials, model "
                "weights, datasets, dependencies, and repository metadata are blocked."
            ),
            wraplength=1080,
            foreground="#666666",
        ).pack(anchor="w", padx=10, pady=(0, 8))

        project_row = ttk.Frame(self.project_tab)
        project_row.pack(fill=tk.X, padx=10, pady=4)
        ttk.Button(
            project_row,
            text="Select Project Folder",
            command=self.select_ai_code_project,
        ).pack(side=tk.LEFT)
        ttk.Button(
            project_row,
            text="Select Individual Code Files",
            command=self.select_ai_code_files,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            project_row,
            text="Select Recommended Files",
            command=self.select_recommended_ai_code_files,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            project_row,
            text="Clear Sharing",
            command=self.clear_ai_code_file_selection,
        ).pack(side=tk.LEFT, padx=5)
        ttk.Label(
            project_row,
            textvariable=self.ai_code_project_var,
            foreground="#245a85",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        ttk.Label(
            self.project_tab,
            textvariable=self.ai_code_project_summary_var,
            wraplength=1160,
            justify=tk.LEFT,
            foreground="#486070",
        ).pack(fill=tk.X, padx=17, pady=(0, 3))

        middle = ttk.PanedWindow(self.project_tab, orient=tk.HORIZONTAL)
        middle.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        selection = ttk.LabelFrame(middle, text="Files Available for Review")
        conversation = ttk.LabelFrame(middle, text="Describe Your Application")
        middle.add(selection, weight=3)
        middle.add(conversation, weight=2)

        columns = ("share", "path", "language", "size")
        self.ai_code_tree = ttk.Treeview(
            selection,
            columns=columns,
            show="headings",
            height=12,
        )
        for column, heading, width in (
            ("share", "Share", 62),
            ("path", "Project File", 300),
            ("language", "Type", 65),
            ("size", "Bytes", 75),
        ):
            self.ai_code_tree.heading(column, text=heading)
            self.ai_code_tree.column(column, width=width, anchor=tk.W)
        tree_scroll = ttk.Scrollbar(
            selection,
            orient=tk.VERTICAL,
            command=self.ai_code_tree.yview,
        )
        self.ai_code_tree.configure(yscrollcommand=tree_scroll.set)
        self.ai_code_tree.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
            padx=(7, 0),
            pady=7,
        )
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 7), pady=7)
        self.ai_code_tree.bind("<Double-1>", self.toggle_ai_code_file)

        choices = ttk.Frame(conversation)
        choices.pack(fill=tk.X, padx=7, pady=(7, 3))
        for label, variable, values in (
            ("Application", self.ai_code_application_type_var, AI_IMPLEMENTATION_APPLICATION_TYPES),
            ("Output", self.ai_code_result_type_var, AI_IMPLEMENTATION_RESULTS),
            ("Data", self.ai_code_data_source_var, AI_IMPLEMENTATION_DATA_SOURCES),
        ):
            ttk.Label(choices, text=label + ":").pack(anchor="w")
            ttk.Combobox(
                choices,
                textvariable=variable,
                values=values,
                state="readonly",
            ).pack(fill=tk.X, pady=(0, 4))
        ttk.Label(
            conversation,
            text="Your message:",
            font=("Arial", 9, "bold"),
        ).pack(anchor="w", padx=7, pady=(3, 0))
        self.ai_code_goal_text = ScrolledText(
            conversation,
            height=5,
            wrap=tk.WORD,
        )
        self.ai_code_goal_text.pack(fill=tk.BOTH, expand=True, padx=7, pady=4)
        self.ai_code_goal_text.insert(
            "1.0",
            "Example: Add the fault prediction and confidence to my website. "
            "I do not know which file should call the model.",
        )
        ttk.Button(
            conversation,
            text="Plan My Integration",
            command=self.build_beginner_ai_code_request,
        ).pack(fill=tk.X, padx=7, pady=(3, 7))

    def build_model_tab(self):
        mode_frame = ttk.LabelFrame(
            self.model_tab,
            text="Choose What the Application Will Use",
        )
        mode_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        mode_frame.columnconfigure(1, weight=1)
        ttk.Label(mode_frame, text="Integration path:").grid(
            row=0, column=0, sticky="w", padx=(8, 4), pady=7
        )
        self.integration_mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.integration_mode_var,
            values=INTEGRATION_MODES,
            state="readonly",
            width=34,
        )
        self.integration_mode_combo.grid(
            row=0, column=1, sticky="ew", padx=(0, 10), pady=7
        )
        self.integration_mode_combo.bind(
            "<<ComboboxSelected>>",
            self.on_integration_mode_changed,
        )
        ttk.Label(mode_frame, text="AI API purpose:").grid(
            row=0, column=2, sticky="w", padx=(0, 4), pady=7
        )
        self.ai_api_purpose_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.ai_api_purpose_var,
            values=AI_API_PURPOSES,
            state="readonly",
            width=35,
        )
        self.ai_api_purpose_combo.grid(
            row=0, column=3, sticky="ew", padx=(0, 8), pady=7
        )
        self.ai_api_purpose_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.invalidate_generated_files(),
        )
        self.provider_integration_label = ttk.Label(
            mode_frame,
            textvariable=self.provider_integration_status_var,
            foreground="#245a85",
            wraplength=820,
        )
        self.provider_integration_label.grid(
            row=1, column=0, columnspan=3, sticky="w", padx=8, pady=(0, 7)
        )
        ttk.Button(
            mode_frame,
            text="Open Provider Settings",
            command=self.open_provider_settings,
        ).grid(row=1, column=3, sticky="e", padx=8, pady=(0, 7))

        model_bar = ttk.LabelFrame(self.model_tab, text="Optional Local Model")
        model_bar.pack(fill=tk.X, padx=10, pady=5)
        self.select_model_button = ttk.Button(
            model_bar,
            text="Select Model / Package",
            command=self.select_model,
        )
        self.select_model_button.pack(side=tk.LEFT, padx=8, pady=8)
        self.select_model_directory_button = ttk.Button(
            model_bar,
            text="Select Exported Model Folder",
            command=self.select_model_directory,
        )
        self.select_model_directory_button.pack(side=tk.LEFT, padx=(0, 8), pady=8)
        ttk.Label(
            model_bar,
            textvariable=self.model_path_var,
            foreground="#245a85",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

        summary_frame = ttk.LabelFrame(
            self.model_tab,
            text="Model Overview",
        )
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.model_summary_text = ScrolledText(
            summary_frame,
            height=12,
            wrap=tk.WORD,
        )
        self.model_summary_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=7,
            pady=7,
        )
        self._set_text(
            self.model_summary_text,
            "Select a saved model package, YOLO weight, or exported detector.",
        )

        settings = ttk.LabelFrame(
            self.model_tab,
            text="Deployment Choice",
        )
        settings.pack(fill=tk.X, padx=10, pady=(0, 10))
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(3, weight=1)
        ttk.Label(settings, text="Target:").grid(
            row=0, column=0, sticky="w", padx=(8, 4), pady=8
        )
        self.target_combo = ttk.Combobox(
            settings,
            textvariable=self.target_var,
            values=DEPLOY_TARGETS,
            state="readonly",
            width=31,
        )
        self.target_combo.grid(
            row=0, column=1, sticky="ew", padx=(0, 12), pady=8
        )
        self.target_combo.bind(
            "<<ComboboxSelected>>",
            self.on_target_changed,
        )
        ttk.Label(settings, text="Communication:").grid(
            row=0, column=2, sticky="w", padx=(0, 4), pady=8
        )
        self.protocol_combo = ttk.Combobox(
            settings,
            textvariable=self.protocol_var,
            values=DEPLOY_PROTOCOLS,
            state="readonly",
            width=24,
        )
        self.protocol_combo.grid(
            row=0, column=3, sticky="ew", padx=(0, 8), pady=8
        )
        self.protocol_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.invalidate_generated_files(),
        )
        ttk.Label(settings, text="Model format:").grid(
            row=1, column=0, sticky="w", padx=(8, 4), pady=(0, 8)
        )
        self.export_combo = ttk.Combobox(
            settings,
            textvariable=self.export_format_var,
            values=DEPLOY_EXPORT_FORMATS,
            state="readonly",
            width=31,
        )
        self.export_combo.grid(
            row=1, column=1, sticky="ew", padx=(0, 12), pady=(0, 8)
        )
        self.export_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.invalidate_generated_files(),
        )
        ttk.Label(
            settings,
            text=(
                "Format conversion is never assumed. The original artifact "
                "is retained until a converted model is explicitly supplied."
            ),
            foreground="#666666",
            wraplength=520,
        ).grid(
            row=1,
            column=2,
            columnspan=2,
            sticky="w",
            padx=(0, 8),
            pady=(0, 8),
        )
        self.refresh_provider_integration_status()
        self.on_integration_mode_changed()

    def build_mapping_tab(self):
        mapping_frame = ttk.LabelFrame(
            self.mapping_tab,
            text="Application Input → Model Input",
        )
        mapping_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=(10, 5),
        )
        columns = ("application_input", "model_input", "processing")
        self.input_mapping_tree = ttk.Treeview(
            mapping_frame,
            columns=columns,
            show="headings",
            height=7,
        )
        for column, heading, width in (
            ("application_input", "Application / Device Input", 290),
            ("model_input", "Required Model Input", 240),
            ("processing", "Processing", 360),
        ):
            self.input_mapping_tree.heading(column, text=heading)
            self.input_mapping_tree.column(
                column,
                width=width,
                anchor=tk.W,
            )
        self.input_mapping_tree.pack(
            fill=tk.BOTH,
            expand=True,
            padx=7,
            pady=(7, 4),
        )
        self.input_mapping_tree.bind(
            "<<TreeviewSelect>>",
            self.load_selected_mapping,
        )
        mapping_edit = ttk.Frame(mapping_frame)
        mapping_edit.pack(fill=tk.X, padx=7, pady=(2, 7))
        ttk.Label(mapping_edit, text="Application input:").pack(side=tk.LEFT)
        ttk.Entry(
            mapping_edit,
            textvariable=self.mapping_source_var,
            width=28,
        ).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Label(mapping_edit, text="Processing:").pack(side=tk.LEFT)
        ttk.Entry(
            mapping_edit,
            textvariable=self.mapping_processing_var,
            width=38,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 10))
        ttk.Button(
            mapping_edit,
            text="Update Selected Mapping",
            command=self.update_selected_mapping,
        ).pack(side=tk.RIGHT)

        output_frame = ttk.LabelFrame(
            self.mapping_tab,
            text="Model Output → Application Action",
        )
        output_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=(5, 10),
        )
        output_columns = ("model_output", "application_action")
        self.output_mapping_tree = ttk.Treeview(
            output_frame,
            columns=output_columns,
            show="headings",
            height=6,
        )
        self.output_mapping_tree.heading(
            "model_output", text="Model Output / Class"
        )
        self.output_mapping_tree.heading(
            "application_action", text="Application Action"
        )
        self.output_mapping_tree.column(
            "model_output", width=330, anchor=tk.W
        )
        self.output_mapping_tree.column(
            "application_action", width=570, anchor=tk.W
        )
        self.output_mapping_tree.pack(
            fill=tk.BOTH,
            expand=True,
            padx=7,
            pady=(7, 4),
        )
        self.output_mapping_tree.bind(
            "<<TreeviewSelect>>",
            self.load_selected_output,
        )
        output_edit = ttk.Frame(output_frame)
        output_edit.pack(fill=tk.X, padx=7, pady=(2, 7))
        ttk.Label(output_edit, text="Action:").pack(side=tk.LEFT)
        ttk.Entry(
            output_edit,
            textvariable=self.output_action_var,
        ).pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=(4, 10),
        )
        ttk.Button(
            output_edit,
            text="Update Selected Action",
            command=self.update_selected_output,
        ).pack(side=tk.RIGHT)

    def build_test_tab(self):
        compatibility = ttk.LabelFrame(
            self.test_tab,
            text="Deployment Compatibility Checker",
        )
        compatibility.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=(10, 5),
        )
        action_row = ttk.Frame(compatibility)
        action_row.pack(fill=tk.X, padx=7, pady=7)
        ttk.Button(
            action_row,
            text="Run Compatibility Check",
            command=self.run_compatibility_check,
        ).pack(side=tk.LEFT)
        ttk.Button(
            action_row,
            text="Test Configured AI Provider",
            command=self.test_configured_provider,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Label(
            action_row,
            textvariable=self.compatibility_status_var,
            foreground="#245a85",
        ).pack(side=tk.LEFT, padx=12)
        self.compatibility_text = ScrolledText(
            compatibility,
            height=12,
            wrap=tk.WORD,
        )
        self.compatibility_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=7,
            pady=(0, 7),
        )
        self._set_text(
            self.compatibility_text,
            "Select a model and run the compatibility check.",
        )

        sample_frame = ttk.LabelFrame(
            self.test_tab,
            text="Sample / Live Test (for a loaded local model)",
        )
        sample_frame.pack(fill=tk.X, padx=10, pady=(5, 10))
        ttk.Button(
            sample_frame,
            text="Select Sample File",
            command=self.select_sample_file,
        ).pack(side=tk.LEFT, padx=(7, 4), pady=8)
        ttk.Button(
            sample_frame,
            text="Select Sample Folder",
            command=self.select_sample_folder,
        ).pack(side=tk.LEFT, padx=4, pady=8)
        ttk.Button(
            sample_frame,
            text="Validate Sample Input",
            command=self.validate_sample_input,
        ).pack(side=tk.LEFT, padx=4, pady=8)
        ttk.Button(
            sample_frame,
            text="Open Actual Prediction / Live Test",
            command=self.open_actual_test,
        ).pack(side=tk.LEFT, padx=4, pady=8)
        ttk.Label(
            sample_frame,
            textvariable=self.sample_status_var,
            foreground="#666666",
            wraplength=420,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=8)

    def build_code_tab(self):
        toolbar = ttk.Frame(self.code_tab)
        toolbar.pack(fill=tk.X, padx=10, pady=(10, 5))
        ttk.Button(
            toolbar,
            text="Generate Integration Code",
            command=self.generate_code,
        ).pack(side=tk.LEFT)
        ttk.Button(
            toolbar,
            text="Validate Python Syntax",
            command=self.validate_generated_code,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            toolbar,
            text="Save Current Code",
            command=self.save_current_code,
        ).pack(side=tk.LEFT, padx=6)
        self.generated_file_var = tk.StringVar(value="app.py")
        self.generated_file_combo = ttk.Combobox(
            toolbar,
            textvariable=self.generated_file_var,
            values=["app.py"],
            state="readonly",
            width=30,
        )
        self.generated_file_combo.pack(side=tk.RIGHT)
        self.generated_file_combo.bind(
            "<<ComboboxSelected>>",
            self.show_selected_generated_file,
        )
        ttk.Label(toolbar, text="Preview file:").pack(
            side=tk.RIGHT, padx=(6, 4)
        )

        self.integration_code_text = ScrolledText(
            self.code_tab,
            wrap=tk.NONE,
            font=("Consolas", 10),
            undo=True,
        )
        self.integration_code_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=(0, 10),
        )
        self._set_text(
            self.integration_code_text,
            "# Generate code after selecting a model and confirming mappings.",
        )

    def build_ai_code_tab(self):
        intro = ttk.LabelFrame(
            self.ai_code_tab,
            text="Optional: Continue Coding with the AI Assistant",
        )
        intro.pack(fill=tk.X, padx=10, pady=(10, 5))
        ttk.Label(
            intro,
            text=(
                "The assistant uses the project and message from Step 1 plus the "
                "integration contract from Step 2. It returns complete changed/new "
                "files, a diff, validation steps, and a copy-and-paste guide. "
                "Step 4 remains available without this provider-assisted stage."
            ),
            wraplength=1080,
            foreground="#245a85",
        ).pack(anchor="w", padx=10, pady=(8, 4))
        ttk.Label(
            intro,
            text=(
                "Nothing is executed or written into the original project. Review "
                "every proposed change before creating a verified complete project "
                "folder and ZIP."
            ),
            wraplength=1080,
            foreground="#666666",
        ).pack(anchor="w", padx=10, pady=(0, 8))

        approval = ttk.LabelFrame(self.ai_code_tab, text="Review & Approval")
        approval.pack(fill=tk.X, padx=10, pady=(5, 4))
        ttk.Label(
            approval,
            textvariable=self.ai_code_project_var,
            foreground="#245a85",
        ).pack(anchor="w", padx=7, pady=(6, 2))
        ttk.Checkbutton(
            approval,
            text=(
                "I reviewed the files marked Share in Step 1 and approve sending "
                "their source text and the integration contract to the selected provider."
            ),
            variable=self.ai_code_approval_var,
        ).pack(anchor="w", padx=7, pady=(2, 6))

        actions = ttk.Frame(self.ai_code_tab)
        actions.pack(fill=tk.X, padx=10, pady=(2, 5))
        self.ai_code_generate_button = ttk.Button(
            actions,
            text="Analyse, Plan & Implement",
            command=self.start_ai_code_integration,
        )
        self.ai_code_generate_button.pack(side=tk.LEFT)
        ttk.Button(
            actions,
            text="Validate Proposed Files",
            command=self.validate_ai_code_result,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            actions,
            text="Create Complete Updated Code Package",
            command=self.export_ai_integrated_project,
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            actions,
            text="Return to Project & Goal",
            command=lambda: self.notebook.select(self.project_tab),
        ).pack(side=tk.RIGHT)
        ttk.Label(
            self.ai_code_tab,
            textvariable=self.ai_code_status_var,
            foreground="#245a85",
            wraplength=1120,
        ).pack(fill=tk.X, padx=12, pady=(0, 4))

        result = ttk.LabelFrame(
            self.ai_code_tab,
            text="Review Complete Implementation",
        )
        result.pack(fill=tk.BOTH, expand=True, padx=10, pady=(3, 10))
        result_toolbar = ttk.Frame(result)
        result_toolbar.pack(fill=tk.X, padx=7, pady=(7, 4))
        ttk.Label(result_toolbar, text="View:").pack(side=tk.LEFT)
        self.ai_code_result_combo = ttk.Combobox(
            result_toolbar,
            textvariable=self.ai_code_result_file_var,
            values=["Integration plan"],
            state="readonly",
            width=36,
        )
        self.ai_code_result_combo.pack(side=tk.LEFT, padx=5)
        self.ai_code_result_combo.bind(
            "<<ComboboxSelected>>",
            self.show_ai_code_result_view,
        )
        ttk.Button(
            result_toolbar,
            text="Save Edited File",
            command=self.save_ai_code_result_edit,
        ).pack(side=tk.RIGHT)
        ttk.Button(
            result_toolbar,
            text="Copy View",
            command=self.copy_ai_code_result_view,
        ).pack(side=tk.RIGHT, padx=5)
        self.ai_code_result_text = ScrolledText(
            result,
            wrap=tk.NONE,
            font=("Consolas", 9),
            undo=False,
        )
        self.ai_code_result_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=7,
            pady=(0, 7),
        )
        self._set_text(
            self.ai_code_result_text,
            "No AI integration has been generated.\n\n"
            "Complete Step 1 and Step 2, approve the selected files above, then "
            "choose Analyse, Plan & Implement. Nothing runs automatically.",
        )

    def _get_ai_settings_owner(self):
        owner = self.settings_owner
        visited = set()
        while owner is not None and id(owner) not in visited:
            if all(
                hasattr(owner, name)
                for name in (
                    "ai_provider_var", "ai_model_var", "ai_base_url_var",
                    "ai_timeout_var", "resolve_ai_api_key",
                )
            ):
                return owner
            visited.add(id(owner))
            owner = getattr(owner, "master", None)
        raise ValueError(
            "AI provider settings are unavailable. Open NN Studio Settings "
            "and configure a provider first."
        )

    def select_ai_code_project(self):
        path = filedialog.askdirectory(
            parent=self,
            title="Select Existing Website, Software, or Hardware-SDK Project",
        )
        if not path:
            return
        try:
            scan = scan_ai_integration_project(path)
        except Exception as exc:
            messagebox.showerror("Project Scan Failed", str(exc), parent=self)
            return
        self._load_ai_code_scan(scan)

    def select_ai_code_files(self):
        paths = filedialog.askopenfilenames(
            parent=self,
            title="Select the Source-Code Files to Analyse",
            filetypes=[
                ("Source code", "*.py *.js *.jsx *.ts *.tsx *.html *.css *.cs *.java *.c *.cpp *.h *.hpp *.ino *.go *.rs *.php *.rb *.swift *.kt *.vue *.svelte *.json *.yaml *.yml *.toml *.ini *.xml *.md *.txt"),
                ("All files", "*.*"),
            ],
        )
        if not paths:
            return
        try:
            scan = scan_ai_integration_files(paths)
        except Exception as exc:
            messagebox.showerror("Code Selection Failed", str(exc), parent=self)
            return
        self._load_ai_code_scan(scan)

    def _load_ai_code_scan(self, scan):
        self.ai_code_project_root = scan["root"]
        self.ai_code_scan = scan
        self.ai_code_sources = []
        self.ai_code_result = None
        self.ai_code_validation = None
        self.ai_code_approval_var.set(False)
        for item in self.ai_code_tree.get_children():
            self.ai_code_tree.delete(item)
        recommended_used = 0
        for candidate in scan["candidates"]:
            share = "No"
            if candidate["recommended"] and recommended_used < 6:
                share = "Yes"
                recommended_used += 1
            self.ai_code_tree.insert(
                "",
                tk.END,
                values=(
                    share,
                    candidate["path"],
                    candidate["language"],
                    candidate["size"],
                ),
                tags=("recommended",) if candidate["recommended"] else (),
            )
        self.ai_code_tree.tag_configure("recommended", foreground="#245a85")
        blocked_count = len(scan["blocked"])
        selected_count = len(self.current_ai_code_selected_paths())
        self.ai_code_project_var.set(
            f"{Path(scan['root']).name}: {len(scan['candidates'])} source files; "
            f"{selected_count} selected; {blocked_count} blocked"
        )
        summary = summarise_ai_integration_project(scan)
        self.ai_code_project_summary_var.set(format_ai_project_summary(summary))
        self.ai_code_status_var.set(
            "Local scan complete. Double-click a row to change Yes/No, then "
            "review the approval statement."
        )
        self._set_text(
            self.ai_code_result_text,
            "Project structure was analysed locally. No source has been sent.\n\n"
            + (
                "Blocked examples:\n"
                + "\n".join(
                    f"- {item['path']}: {item['reason']}"
                    for item in scan["blocked"][:20]
                )
                if scan["blocked"]
                else "No credential-like source files were detected."
            ),
        )

    def build_beginner_ai_code_request(self):
        current = self.ai_code_goal_text.get("1.0", tk.END).strip()
        if current.startswith("Example:"):
            current = ""
        summary = (
            summarise_ai_integration_project(self.ai_code_scan)
            if self.ai_code_scan
            else None
        )
        request_text = build_beginner_integration_request(
            self.ai_code_application_type_var.get(),
            self.ai_code_result_type_var.get(),
            self.ai_code_data_source_var.get(),
            current,
            summary,
            self.integration_mode_var.get(),
            self._get_ai_settings_owner().ai_provider_var.get(),
            self.ai_api_purpose_var.get(),
        )
        self.ai_code_goal_text.delete("1.0", tk.END)
        self.ai_code_goal_text.insert("1.0", request_text)
        self.ai_code_status_var.set(
            "Implementation request prepared. Continue to Step 2 and choose a "
            "local model, an AI provider API, or both."
        )
        self.notebook.select(self.integration_tab)

    def toggle_ai_code_file(self, event=None):
        item = self.ai_code_tree.identify_row(event.y) if event else ""
        if not item:
            selection = self.ai_code_tree.selection()
            item = selection[0] if selection else ""
        if not item:
            return
        values = list(self.ai_code_tree.item(item, "values"))
        values[0] = "No" if str(values[0]) == "Yes" else "Yes"
        if values[0] == "Yes" and len(self.current_ai_code_selected_paths()) >= AI_CODE_MAX_SELECTED_FILES:
            messagebox.showwarning(
                "Sharing Limit",
                f"Select no more than {AI_CODE_MAX_SELECTED_FILES} files per request.",
                parent=self,
            )
            return
        self.ai_code_tree.item(item, values=values)
        self.ai_code_approval_var.set(False)
        self.ai_code_status_var.set(
            f"{len(self.current_ai_code_selected_paths())} file(s) marked Share. "
            "Review and approve again before requesting AI changes."
        )

    def current_ai_code_selected_paths(self):
        paths = []
        for item in self.ai_code_tree.get_children():
            values = self.ai_code_tree.item(item, "values")
            if values and str(values[0]) == "Yes":
                paths.append(str(values[1]))
        return paths

    def select_recommended_ai_code_files(self):
        if not self.ai_code_scan:
            return
        chosen = 0
        for item in self.ai_code_tree.get_children():
            tags = self.ai_code_tree.item(item, "tags")
            values = list(self.ai_code_tree.item(item, "values"))
            should_share = "recommended" in tags and chosen < 8
            values[0] = "Yes" if should_share else "No"
            if should_share:
                chosen += 1
            self.ai_code_tree.item(item, values=values)
        if not chosen:
            for item in self.ai_code_tree.get_children()[:3]:
                values = list(self.ai_code_tree.item(item, "values"))
                values[0] = "Yes"
                self.ai_code_tree.item(item, values=values)
        self.ai_code_approval_var.set(False)
        self.ai_code_status_var.set(
            f"{len(self.current_ai_code_selected_paths())} recommended file(s) selected."
        )

    def clear_ai_code_file_selection(self):
        for item in self.ai_code_tree.get_children():
            values = list(self.ai_code_tree.item(item, "values"))
            values[0] = "No"
            self.ai_code_tree.item(item, values=values)
        self.ai_code_approval_var.set(False)
        self.ai_code_status_var.set("No source files are marked Share.")

    def build_ai_model_contract(self):
        combined_report = self.run_compatibility_check(select_tab=False)
        if combined_report["failures"]:
            raise ValueError(
                "Resolve deployment compatibility first: "
                + "; ".join(combined_report["failures"])
            )
        reference_subset = {}
        descriptor = None
        if self.uses_local_model():
            if not self.descriptor or not self.selected_model_path:
                raise ValueError("Select the trained model or model package first.")
            source = Path(self.selected_model_path)
            artifact_name = "NN_STUDIO_INTEGRATION/model/" + (
                "source_model" if source.is_dir() else source.name
            )
            reference = generate_deployment_text_artifacts(
                self.descriptor,
                self.target_var.get(),
                self.protocol_var.get(),
                self.current_input_mappings(),
                self.current_output_actions(),
                self.export_format_var.get(),
                artifact_name,
            )
            total = 0
            for name in (
                "app.py", "deployment_config.json", "requirements.txt",
                "client.py", "README.md",
            ):
                content = reference.get(name)
                if isinstance(content, str) and total + len(content) <= 80_000:
                    reference_subset[name] = content
                    total += len(content)
            descriptor = {
                key: self.descriptor.get(key)
                for key in (
                    "name", "model_type", "display_type", "data_mode",
                    "task_type", "trainable", "inference_ready", "details",
                )
            }

        provider_contract = None
        if self.uses_provider_api():
            owner = self._get_ai_settings_owner()
            provider = owner.ai_provider_var.get()
            provider_reference = generate_ai_provider_api_artifacts(
                provider,
                owner.ai_model_var.get().strip(),
                owner.ai_base_url_var.get(),
                self.ai_api_purpose_var.get(),
                self.target_var.get(),
                self.protocol_var.get(),
            )
            provider_contract = {
                "provider": provider,
                "model": owner.ai_model_var.get().strip(),
                "base_url": owner.ai_base_url_var.get().strip(),
                "api_key_environment": OFFICIAL_PROVIDER_SETTINGS[provider][
                    "api_key_environment"
                ],
                "purpose": self.ai_api_purpose_var.get(),
                "credential_embedded": False,
            }
            for name, content in provider_reference.items():
                if isinstance(content, str) and len(content) <= 80_000:
                    reference_subset["provider_api/" + name] = content

        return {
            "generated_by": f"{APPLICATION_NAME} {APP_VERSION}",
            "integration_mode": self.integration_mode_var.get(),
            "model": descriptor,
            "ai_provider_api": provider_contract,
            "deployment_target": self.target_var.get(),
            "communication_protocol": self.protocol_var.get(),
            "requested_export_format": self.export_format_var.get(),
            "implementation_preferences": {
                "application_type": self.ai_code_application_type_var.get(),
                "desired_result": self.ai_code_result_type_var.get(),
                "data_source": self.ai_code_data_source_var.get(),
            },
            "input_mappings": self.current_input_mappings(),
            "output_actions": self.current_output_actions(),
            "compatibility": combined_report,
            "reference_runtime_files": reference_subset,
            "privacy_rules": {
                "no_embedded_provider_credentials": True,
                "no_unrequested_data_upload": True,
                "hardware_read_only_by_default": True,
            },
        }

    def start_ai_code_integration(self):
        try:
            if self.ai_code_running:
                return
            if not self.ai_code_project_root:
                raise ValueError("Select the existing application project first.")
            if not self.ai_code_approval_var.get():
                raise ValueError(
                    "Review the files marked Share and tick the approval checkbox."
                )
            goal = self.ai_code_goal_text.get("1.0", tk.END).strip()
            if goal.startswith("Example:"):
                goal = ""
            if "Integration path:" not in goal:
                summary = summarise_ai_integration_project(self.ai_code_scan)
                goal = build_beginner_integration_request(
                    self.ai_code_application_type_var.get(),
                    self.ai_code_result_type_var.get(),
                    self.ai_code_data_source_var.get(),
                    goal,
                    summary,
                    self.integration_mode_var.get(),
                    self._get_ai_settings_owner().ai_provider_var.get(),
                    self.ai_api_purpose_var.get(),
                )
                self.ai_code_goal_text.delete("1.0", tk.END)
                self.ai_code_goal_text.insert("1.0", goal)
            selected_paths = self.current_ai_code_selected_paths()
            sources = collect_ai_integration_sources(
                self.ai_code_project_root,
                selected_paths,
            )
            model_contract = self.build_ai_model_contract()
            owner = self._get_ai_settings_owner()
            provider_name = owner.ai_provider_var.get()
            if provider_name not in OFFICIAL_PROVIDER_SETTINGS:
                raise ValueError(
                    "Connect OpenAI, DeepSeek, Claude, or an OpenAI-compatible API in Settings."
                )
            api_key = owner.resolve_ai_api_key(provider_name)
            if not api_key:
                raise ValueError("No API key is available for the selected provider.")
            model_name = owner.ai_model_var.get().strip()
            base_url = _validate_provider_base_url(owner.ai_base_url_var.get())
            timeout_seconds = float(owner.ai_timeout_var.get())
            selected_host = urlparse(base_url).hostname
            approved = messagebox.askyesno(
                "Confirm Source-Code Transmission",
                (
                    f"Provider: {provider_name}\n"
                    f"Host: {selected_host}\n"
                    f"Files: {len(sources)}\n"
                    f"Source text: {sum(len(item['content'].encode('utf-8')) for item in sources) / 1000:.1f} KB\n\n"
                    "The selected source files and integration contract "
                    "will be sent to this host. API keys, blocked files, model "
                    "weights, datasets, and unselected files are not included.\n\n"
                    "Continue?"
                ),
                parent=self,
            )
            if not approved:
                return
            self.ai_code_sources = sources
            self.ai_code_result = None
            self.ai_code_validation = None
            self.ai_code_running = True
            self.ai_code_generate_button.config(state=tk.DISABLED)
            self.ai_code_status_var.set(
                f"{provider_name} is analysing the approved source files..."
            )
            self._set_text(
                self.ai_code_result_text,
                "AI request in progress. The result will be validated locally "
                "before it is displayed.",
            )
            self.ai_code_queue = queue.Queue()
            threading.Thread(
                target=self._ai_code_integration_worker,
                args=(
                    provider_name, api_key, model_name, base_url,
                    timeout_seconds, sources, model_contract, goal,
                ),
                daemon=True,
            ).start()
            self.after(120, self._poll_ai_code_integration)
        except Exception as exc:
            messagebox.showerror("AI Code Integration", str(exc), parent=self)

    def _ai_code_integration_worker(
        self,
        provider_name,
        api_key,
        model_name,
        base_url,
        timeout_seconds,
        sources,
        model_contract,
        goal,
    ):
        try:
            result = request_ai_code_integration(
                provider_name=provider_name,
                api_key=api_key,
                model=model_name,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                source_files=sources,
                model_contract=model_contract,
                user_goal=goal,
                project_root=self.ai_code_project_root,
            )
            validation = validate_ai_integrated_source_files(result)
            self.ai_code_queue.put(("success", result, validation, provider_name, model_name))
        except Exception as exc:
            self.ai_code_queue.put(("error", str(exc)))

    def _poll_ai_code_integration(self):
        if not self.winfo_exists():
            return
        try:
            message = self.ai_code_queue.get_nowait()
        except queue.Empty:
            self.after(120, self._poll_ai_code_integration)
            return
        self.ai_code_running = False
        self.ai_code_generate_button.config(state=tk.NORMAL)
        if message[0] == "error":
            self.ai_code_status_var.set("AI integration failed local validation.")
            messagebox.showerror("AI Code Integration Failed", message[1], parent=self)
            return
        _, result, validation, provider_name, model_name = message
        self.ai_code_result = result
        self.ai_code_validation = validation
        self.ai_code_provider_summary = {
            "provider": provider_name,
            "model": model_name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        views = [
            "Integration plan",
            "Copy-and-paste guide",
            "Unified diff",
            "Validation",
        ]
        views.extend("Updated: " + item["path"] for item in result["changes"])
        views.extend("New: " + item["path"] for item in result["new_files"])
        self.ai_code_result_combo.config(values=views)
        preferred_view = (
            "Copy-and-paste guide"
            if "copy-and-paste" in self.ai_code_result_type_var.get().lower()
            else "Integration plan"
        )
        self.ai_code_result_file_var.set(preferred_view)
        self.show_ai_code_result_view()
        if validation["errors"]:
            self.ai_code_status_var.set(
                "Generated, but syntax validation found problems. Review Validation."
            )
            self.ai_code_result_file_var.set("Validation")
            self.show_ai_code_result_view()
        else:
            self.ai_code_status_var.set(
                f"Validated proposal: {len(result['changes'])} updated, "
                f"{len(result['new_files'])} new file(s). Review before export."
            )

    def _format_ai_code_plan(self):
        result = self.ai_code_result
        if not result:
            return "No AI integration result is available."
        lines = [
            "ANALYSIS",
            result["analysis_summary"],
            "",
            "INTEGRATION PLAN",
        ]
        lines.extend(f"{index}. {item}" for index, item in enumerate(result["integration_plan"], 1))
        lines.extend(["", "WARNINGS"])
        if result["warnings"]:
            lines.extend(f"- {item}" for item in result["warnings"])
        else:
            lines.append("- None")
        lines.extend(["", "CHANGED FILES"])
        lines.extend(
            f"- {item['path']}: {item['explanation']}"
            for item in result["changes"]
        )
        lines.extend(["", "NEW FILES"])
        lines.extend(
            f"- {item['path']}: {item['purpose']}"
            for item in result["new_files"]
        )
        lines.extend(["", "DEPENDENCIES"])
        if result["dependencies"]:
            lines.extend(f"- {item}" for item in result["dependencies"])
        else:
            lines.append("- None")
        lines.extend(["", "MANUAL STEPS"])
        if result["manual_steps"]:
            lines.extend(f"- {item}" for item in result["manual_steps"])
        else:
            lines.append("- None")
        return "\n".join(lines)

    def _format_ai_copy_paste_guide(self):
        result = self.ai_code_result
        if not result:
            return "No AI integration result is available."
        lines = [
            "BEGINNER COPY-AND-PASTE IMPLEMENTATION GUIDE",
            "=" * 72,
            "Work in a copy of the application. Do not paste provider API keys "
            "into source code. Keep any selected local model together with its "
            "saved preprocessing and follow the generated environment-variable "
            "instructions for provider API credentials.",
            "",
            "WHAT THIS IMPLEMENTATION DOES",
            result["analysis_summary"],
            "",
            "STEPS",
        ]
        plan = result.get("integration_plan") or []
        lines.extend(f"{index}. {item}" for index, item in enumerate(plan, 1))
        if result.get("dependencies"):
            lines.extend(["", "DEPENDENCIES TO ADD"])
            lines.extend(f"- {item}" for item in result["dependencies"])
        lines.extend(["", "COMPLETE FILES"])
        for item in result.get("changes", []):
            lines.extend([
                "",
                f"REPLACE THE COMPLETE CONTENT OF: {item['path']}",
                f"Reason: {item['explanation']}",
                "----- BEGIN FILE -----",
                item["updated_content"].rstrip(),
                "----- END FILE -----",
            ])
        for item in result.get("new_files", []):
            lines.extend([
                "",
                f"CREATE NEW FILE: {item['path']}",
                f"Purpose: {item['purpose']}",
                "----- BEGIN FILE -----",
                item["content"].rstrip(),
                "----- END FILE -----",
            ])
        lines.extend(["", "MANUAL SETUP"])
        manual = result.get("manual_steps") or ["No additional manual step reported."]
        lines.extend(f"{index}. {item}" for index, item in enumerate(manual, 1))
        lines.extend(["", "HOW TO TEST"])
        checks = result.get("validation_steps") or ["Run the application's normal test workflow."]
        lines.extend(f"{index}. {item}" for index, item in enumerate(checks, 1))
        if result.get("warnings"):
            lines.extend(["", "WARNINGS"])
            lines.extend(f"- {item}" for item in result["warnings"])
        return "\n".join(lines).rstrip() + "\n"

    def show_ai_code_result_view(self, _event=None):
        if not self.ai_code_result:
            return
        view = self.ai_code_result_file_var.get()
        if view == "Integration plan":
            value = self._format_ai_code_plan()
        elif view == "Copy-and-paste guide":
            value = self._format_ai_copy_paste_guide()
        elif view == "Unified diff":
            value = build_ai_code_integration_diff(
                self.ai_code_result,
                self.ai_code_sources,
            )
        elif view == "Validation":
            validation = self.ai_code_validation or {"passed": [], "errors": []}
            value = "PASSED\n" + (
                "\n".join("✓ " + item for item in validation["passed"]) or "None"
            ) + "\n\nERRORS\n" + (
                "\n".join("✗ " + item for item in validation["errors"]) or "None"
            )
        elif view.startswith("Updated: "):
            path = view[len("Updated: "):]
            value = next(
                item["updated_content"]
                for item in self.ai_code_result["changes"]
                if item["path"] == path
            )
        elif view.startswith("New: "):
            path = view[len("New: "):]
            value = next(
                item["content"]
                for item in self.ai_code_result["new_files"]
                if item["path"] == path
            )
        else:
            value = "Unknown result view."
        self._set_text(self.ai_code_result_text, value)
        if view.startswith("Updated: ") or view.startswith("New: "):
            self.ai_code_result_text.config(state=tk.NORMAL)

    def copy_ai_code_result_view(self):
        value = self.ai_code_result_text.get("1.0", tk.END).rstrip()
        if not value:
            messagebox.showinfo(
                "Nothing to Copy",
                "Generate an implementation or select a result view first.",
                parent=self,
            )
            return
        self.clipboard_clear()
        self.clipboard_append(value)
        self.update_idletasks()
        self.ai_code_status_var.set(
            f"Copied {self.ai_code_result_file_var.get()} to the clipboard."
        )

    def save_ai_code_result_edit(self):
        if not self.ai_code_result:
            return
        view = self.ai_code_result_file_var.get()
        path = self._capture_ai_code_result_edit_if_active()
        if not path:
            messagebox.showinfo(
                "Select a Source File",
                "Choose an Updated or New file view before saving edits.",
                parent=self,
            )
            return
        self.ai_code_validation = validate_ai_integrated_source_files(
            self.ai_code_result
        )
        self.ai_code_status_var.set(
            f"Saved local edits to {path}. Re-run validation before export."
        )

    def _capture_ai_code_result_edit_if_active(self):
        if not self.ai_code_result:
            return None
        view = self.ai_code_result_file_var.get()
        content = self.ai_code_result_text.get("1.0", tk.END).rstrip() + "\n"
        if view.startswith("Updated: "):
            path = view[len("Updated: "):]
            item = next(
                item for item in self.ai_code_result["changes"]
                if item["path"] == path
            )
            item["updated_content"] = content
            return path
        if view.startswith("New: "):
            path = view[len("New: "):]
            item = next(
                item for item in self.ai_code_result["new_files"]
                if item["path"] == path
            )
            item["content"] = content
            return path
        return None

    def validate_ai_code_result(self):
        if not self.ai_code_result:
            messagebox.showwarning(
                "No Integration Result",
                "Generate an AI integration proposal first.",
                parent=self,
            )
            return
        self._capture_ai_code_result_edit_if_active()
        self.ai_code_validation = validate_ai_integrated_source_files(
            self.ai_code_result
        )
        self.ai_code_result_file_var.set("Validation")
        self.show_ai_code_result_view()
        if self.ai_code_validation["errors"]:
            messagebox.showerror(
                "Validation Problems",
                "Review the Validation view before export.",
                parent=self,
            )
        else:
            messagebox.showinfo(
                "Validation Passed",
                "All proposed files passed non-executing local syntax/path checks. "
                "Runtime and hardware testing are still required.",
                parent=self,
            )

    def export_ai_integrated_project(self):
        if not self.ai_code_result or not self.ai_code_project_root:
            messagebox.showwarning(
                "No Integration Result",
                "Generate and review an AI integration proposal first.",
                parent=self,
            )
            return
        self._capture_ai_code_result_edit_if_active()
        validation = validate_ai_integrated_source_files(self.ai_code_result)
        if validation["errors"]:
            messagebox.showerror(
                "Integration Not Ready",
                "Resolve generated syntax errors before creating a project copy.",
                parent=self,
            )
            return
        current_sources = collect_ai_integration_sources(
            self.ai_code_project_root,
            [item["path"] for item in self.ai_code_sources],
        )
        current_hashes = {item["path"]: item["sha256"] for item in current_sources}
        changed_since_request = [
            item["path"]
            for item in self.ai_code_sources
            if current_hashes.get(item["path"]) != item["sha256"]
        ]
        if changed_since_request:
            messagebox.showerror(
                "Project Changed",
                "These source files changed after the AI request:\n"
                + "\n".join(changed_since_request)
                + "\n\nGenerate a new proposal from the current files.",
                parent=self,
            )
            return
        confirmed = messagebox.askyesno(
            "Create Complete Updated Code Package",
            (
                "NN Studio will copy the complete existing application, apply "
                "every reviewed AI update/new file, add the selected model and "
                "integration evidence, then create and verify a ZIP.\n\n"
                "Credentials, secret-like files, dependency/cache folders, "
                "repository metadata, symlinks, and old NN Studio evidence are "
                "excluded. Other files already stored in the project—including "
                "data or media assets—may be copied. Review the project folder "
                "before continuing.\n\nContinue?"
            ),
            parent=self,
        )
        if not confirmed:
            return
        parent_directory = filedialog.askdirectory(
            parent=self,
            title="Choose Parent Folder for the Complete Updated Code Package",
        )
        if not parent_directory:
            return
        source_root = Path(self.ai_code_project_root).resolve()
        destination = Path(parent_directory) / (
            source_root.name + "_NN_Integrated_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        archive_path = destination.with_suffix(".zip")
        try:
            if archive_path.exists():
                raise FileExistsError(
                    f"Package ZIP already exists; wait one second and try again: {archive_path}"
                )
            copy_audit = copy_complete_ai_project(source_root, destination)
            evidence_root = destination / "NN_STUDIO_INTEGRATION"
            originals_root = evidence_root / "original_files"
            originals_root.mkdir(parents=True, exist_ok=True)
            updated_paths = []
            new_paths = []
            for item in self.ai_code_result["changes"]:
                original = next(
                    source for source in self.ai_code_sources
                    if source["path"] == item["path"]
                )
                backup_path = originals_root / item["path"]
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                backup_path.write_text(original["content"], encoding="utf-8")
                output_path = destination / item["path"]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(item["updated_content"], encoding="utf-8")
                updated_paths.append(item["path"])
            for item in self.ai_code_result["new_files"]:
                output_path = destination / item["path"]
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(item["content"], encoding="utf-8")
                new_paths.append(item["path"])

            model_sha256 = None
            model_package_path = None
            if self.uses_local_model():
                model_root = evidence_root / "model"
                model_root.mkdir(parents=True, exist_ok=True)
                model_source = Path(self.selected_model_path)
                if model_source.is_dir():
                    shutil.copytree(model_source, model_root / "source_model")
                    model_package_path = "NN_STUDIO_INTEGRATION/model/source_model"
                else:
                    shutil.copy2(model_source, model_root / model_source.name)
                    model_package_path = (
                        "NN_STUDIO_INTEGRATION/model/" + model_source.name
                    )
                model_sha256 = self._artifact_checksum(model_source)

            result_without_content = {
                key: value
                for key, value in self.ai_code_result.items()
                if key not in {"changes", "new_files"}
            }
            result_without_content["changes"] = [
                {key: value for key, value in item.items() if key != "updated_content"}
                for item in self.ai_code_result["changes"]
            ]
            result_without_content["new_files"] = [
                {key: value for key, value in item.items() if key != "content"}
                for item in self.ai_code_result["new_files"]
            ]
            manifest = {
                "schema_version": 1,
                "application_version": APP_VERSION,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "provider": getattr(self, "ai_code_provider_summary", {}),
                "source_project_name": source_root.name,
                "integration_mode": self.integration_mode_var.get(),
                "model_sha256": model_sha256,
                "model_package_path": model_package_path,
                "updated_files": updated_paths,
                "new_files": new_paths,
                "copied_original_file_count": len(copy_audit["included"]),
                "omitted_item_count": len(copy_audit["omitted"]),
                "proposal": result_without_content,
                "validation": validation,
                "privacy": {
                    "provider_api_key_included": False,
                    "blocked_secret_files_copied": False,
                    "training_dataset_added_by_nn_studio": False,
                    "source_project_data_files_may_be_present": True,
                    "original_project_overwritten": False,
                },
            }
            (evidence_root / "integration_manifest.json").write_text(
                json.dumps(manifest, indent=2, default=_result_json_default),
                encoding="utf-8",
            )
            (evidence_root / "changes.diff").write_text(
                build_ai_code_integration_diff(
                    self.ai_code_result,
                    self.ai_code_sources,
                ),
                encoding="utf-8",
            )
            (evidence_root / "omitted_items.json").write_text(
                json.dumps(copy_audit["omitted"], indent=2),
                encoding="utf-8",
            )
            (evidence_root / "AI_DEPENDENCIES.txt").write_text(
                (
                    "\n".join(self.ai_code_result.get("dependencies") or [])
                    or "No additional dependency was reported."
                )
                + "\n",
                encoding="utf-8",
            )
            (evidence_root / "README.md").write_text(
                "# NN Studio AI Integration\n\n"
                "Review every changed file and run the validation steps before "
                "production use. AI-generated code was not executed by NN Studio.\n\n"
                "## Rollback\n\nCopy the corresponding file from "
                "`original_files/` back to the project path, or return to the "
                "untouched original project. New files may be removed manually.\n",
                encoding="utf-8",
            )
            start_lines = [
                "# Start Here — Complete Updated Application",
                "",
                "This package contains the existing application plus every reviewed "
                "AI-updated and AI-created source file. The original project was not changed.",
                "",
                "## Updated files",
            ]
            start_lines.extend(
                f"- `{item}`" for item in updated_paths
            )
            if not updated_paths:
                start_lines.append("- None")
            start_lines.extend(["", "## New files"])
            start_lines.extend(f"- `{item}`" for item in new_paths)
            if not new_paths:
                start_lines.append("- None")
            start_lines.extend(["", "## Manual setup"])
            manual_steps = self.ai_code_result.get("manual_steps") or [
                "No additional manual setup was reported."
            ]
            start_lines.extend(
                f"{index}. {item}"
                for index, item in enumerate(manual_steps, 1)
            )
            start_lines.extend(["", "## Validation"])
            validation_steps = self.ai_code_result.get("validation_steps") or [
                "Run the application's normal test workflow."
            ]
            start_lines.extend(
                f"{index}. {item}"
                for index, item in enumerate(validation_steps, 1)
            )
            start_lines.extend(
                [
                    "",
                    "## Package evidence",
                    "See `NN_STUDIO_INTEGRATION/package_inventory.json` for the "
                    "complete file list and SHA-256 checksums. See "
                    "`omitted_items.json` for files intentionally excluded.",
                    "",
                    "Do not place provider API keys in source code. Configure them "
                    "using the environment-variable instructions generated for the application.",
                ]
            )
            (destination / "START_HERE.md").write_text(
                "\n".join(start_lines).rstrip() + "\n",
                encoding="utf-8",
            )

            inventory = build_complete_ai_package_inventory(
                destination,
                updated_paths=updated_paths,
                new_paths=new_paths,
            )
            inventory_by_path = {item["path"]: item for item in inventory}
            missing_ai_files = sorted(
                (set(updated_paths) | set(new_paths)) - set(inventory_by_path)
            )
            wrongly_classified = sorted(
                path
                for path in updated_paths
                if inventory_by_path.get(path, {}).get("category")
                != "ai_updated_source"
            ) + sorted(
                path
                for path in new_paths
                if inventory_by_path.get(path, {}).get("category")
                != "ai_new_source"
            )
            if missing_ai_files or wrongly_classified:
                raise ValueError(
                    "AI-updated package verification failed. Missing: "
                    + (", ".join(missing_ai_files) or "none")
                    + "; incorrectly classified: "
                    + (", ".join(wrongly_classified) or "none")
                )
            (evidence_root / "package_inventory.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                        "file_count_excluding_this_inventory": len(inventory),
                        "files": inventory,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            zip_result_directory(destination, archive_path)
            verification = verify_complete_ai_package_zip(
                destination,
                archive_path,
            )
            self.ai_code_status_var.set(
                f"Complete package verified: {verification['file_count']} files; "
                f"{len(updated_paths)} updated; {len(new_paths)} new."
            )
            messagebox.showinfo(
                "Complete Updated Code Package Created",
                "The original project was not changed. Every reviewed AI update "
                "and new file was applied, inventoried, and verified in the ZIP.\n\n"
                f"Project folder:\n{destination}\n\nZIP package:\n{archive_path}\n\n"
                f"Verified files: {verification['file_count']}\n"
                f"AI-updated: {len(updated_paths)}\n"
                f"AI-created: {len(new_paths)}\n"
                f"Excluded items: {len(copy_audit['omitted'])}",
                parent=self,
            )
        except Exception as exc:
            if destination.exists():
                shutil.rmtree(destination, ignore_errors=True)
            if archive_path.exists():
                archive_path.unlink(missing_ok=True)
            messagebox.showerror("Complete Package Export Failed", str(exc), parent=self)

    def build_package_tab(self):
        checklist = ttk.LabelFrame(
            self.package_tab,
            text="Deployment Package Contents",
        )
        checklist.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 5))
        ttk.Label(
            checklist,
            text=(
                "✓ Selected local model and preprocessing, when used\n"
                "✓ Key-free AI provider client and configuration, when used\n"
                "✓ Input mapping and output actions\n"
                "✓ Editable inference and integration code\n"
                "✓ Optional AI-assisted changes for an existing application\n"
                "✓ Dependencies and Docker/Windows files when required\n"
                "✓ Compatibility report, checksum, examples, and README\n\n"
                "Excluded: raw training data, API keys, cloud tokens, "
                "passwords, unapproved source files, and result records."
            ),
            justify=tk.LEFT,
            wraplength=900,
        ).pack(anchor="w", padx=12, pady=12)
        action = ttk.Frame(self.package_tab)
        action.pack(fill=tk.X, padx=10, pady=(5, 10))
        ttk.Button(
            action,
            text="Create Complete Deployment ZIP",
            command=self.create_deployment_package,
        ).pack(side=tk.LEFT)
        ttk.Button(
            action,
            text="Open Model Export / Application Workspace",
            command=self.open_actual_test,
        ).pack(side=tk.LEFT, padx=8)
        ttk.Label(
            action,
            textvariable=self.package_status_var,
            foreground="#245a85",
            wraplength=650,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

    @staticmethod
    def _set_text(widget, value):
        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", str(value))
        widget.config(state=tk.DISABLED)

    def select_model(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Select Saved Model or Detector",
            filetypes=[
                ("Supported models", "*.zip *.pt *.yaml *.yml *.onnx *.engine *.torchscript *.tflite *.pb *.xml"),
                ("NN Studio package", "*.zip"),
                ("YOLO weights", "*.pt"),
                ("Inference exports", "*.onnx *.engine *.torchscript *.tflite *.pb *.xml"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.inspect_model_path(path)

    def select_model_directory(self):
        path = filedialog.askdirectory(
            parent=self,
            title="Select Exported Detector Directory",
        )
        if path:
            self.inspect_model_path(path)

    def inspect_model_path(self, path):
        try:
            descriptor = inspect_saved_model_descriptor(path)
        except Exception as exc:
            messagebox.showerror(
                "Unsupported Model",
                str(exc),
                parent=self,
            )
            return
        self.descriptor = descriptor
        self.selected_model_path = descriptor["path"]
        self.model_path_var.set(descriptor["name"])
        recommended = deployment_recommended_targets(descriptor)
        self.target_var.set(recommended[0])
        self.protocol_var.set(
            deployment_default_protocol(self.target_var.get())
        )
        self.populate_model_summary()
        self.populate_default_mappings()
        self.invalidate_generated_files()
        self.notebook.select(self.integration_tab)
        self.integration_notebook.select(self.model_tab)

    def populate_model_summary(self):
        descriptor = self.descriptor or {}
        model_path = Path(descriptor.get("path", ""))
        size_text = "Directory"
        if model_path.is_file():
            size_text = f"{model_path.stat().st_size / (1024 * 1024):.2f} MB"
        recommended = deployment_recommended_targets(descriptor)
        lines = [
            f"Model: {descriptor.get('name', '')}",
            f"Type: {descriptor.get('display_type', '')}",
            f"Task: {descriptor.get('task_type', '')}",
            f"Data mode: {descriptor.get('data_mode', '')}",
            f"Artifact size: {size_text}",
            (
                "Training: Continue / fine-tune supported"
                if descriptor.get("trainable")
                else "Training: Inference/evaluation artifact"
            ),
            (
                "Inference: Ready"
                if descriptor.get("inference_ready")
                else "Inference: Not ready (architecture only)"
            ),
            "",
            "Required inputs:",
        ]
        required = deployment_required_inputs(descriptor)
        lines.extend(
            ["• " + value for value in required]
            or ["• Confirm from the model input shape"]
        )
        lines.extend(["", "Recommended deployment targets:"])
        lines.extend("• " + value for value in recommended)
        details = descriptor.get("details") or {}
        if details:
            lines.extend(["", "Saved metadata:"])
            for key, value in details.items():
                if value not in (None, "", [], {}):
                    lines.append(
                        f"• {key.replace('_', ' ').title()}: {value}"
                    )
        self._set_text(self.model_summary_text, "\n".join(lines))

    def populate_default_mappings(self):
        for tree in (self.input_mapping_tree, self.output_mapping_tree):
            for item in tree.get_children():
                tree.delete(item)
        required_inputs = deployment_required_inputs(self.descriptor)
        for model_input in required_inputs:
            processing = (
                "Image decoding / resize saved in model package"
                if model_input in {"image_path", "image_or_video_source"}
                else "Saved package preprocessing"
            )
            self.input_mapping_tree.insert(
                "",
                tk.END,
                values=(model_input, model_input, processing),
            )

        details = (self.descriptor or {}).get("details") or {}
        outputs = (
            details.get("class_names")
            or details.get("target_columns")
            or ["prediction"]
        )
        if isinstance(outputs, dict):
            outputs = list(outputs.values())
        elif isinstance(outputs, str):
            outputs = [outputs]
        for output in outputs:
            output_text = str(output)
            lower = output_text.strip().lower()
            if lower in {"normal", "0"}:
                action = "Continue operation"
            elif "fault" in lower or lower in {"sef", "def", "mef"}:
                action = "Display warning and record event"
            elif self.descriptor.get("data_mode") == DATA_MODE_DETECTION:
                action = "Display detection and confidence"
            else:
                action = "Publish predicted value / status"
            self.output_mapping_tree.insert(
                "",
                tk.END,
                values=(output_text, action),
            )

    def load_selected_mapping(self, _event=None):
        selection = self.input_mapping_tree.selection()
        if not selection:
            return
        values = self.input_mapping_tree.item(
            selection[0], "values"
        )
        self.mapping_source_var.set(values[0])
        self.mapping_processing_var.set(values[2])

    def update_selected_mapping(self):
        selection = self.input_mapping_tree.selection()
        if not selection:
            messagebox.showwarning(
                "Select Mapping",
                "Select one input-mapping row first.",
                parent=self,
            )
            return
        values = self.input_mapping_tree.item(selection[0], "values")
        source = self.mapping_source_var.get().strip()
        if not source:
            messagebox.showwarning(
                "Missing Input Name",
                "Enter the application or device input name.",
                parent=self,
            )
            return
        self.input_mapping_tree.item(
            selection[0],
            values=(
                source,
                values[1],
                self.mapping_processing_var.get().strip(),
            ),
        )
        self.invalidate_generated_files()

    def load_selected_output(self, _event=None):
        selection = self.output_mapping_tree.selection()
        if not selection:
            return
        values = self.output_mapping_tree.item(selection[0], "values")
        self.output_action_var.set(values[1])

    def update_selected_output(self):
        selection = self.output_mapping_tree.selection()
        if not selection:
            messagebox.showwarning(
                "Select Output",
                "Select one output-action row first.",
                parent=self,
            )
            return
        values = self.output_mapping_tree.item(selection[0], "values")
        self.output_mapping_tree.item(
            selection[0],
            values=(values[0], self.output_action_var.get().strip()),
        )
        self.invalidate_generated_files()

    def current_input_mappings(self):
        records = []
        for item in self.input_mapping_tree.get_children():
            values = self.input_mapping_tree.item(item, "values")
            records.append(
                {
                    "application_input": str(values[0]),
                    "model_input": str(values[1]),
                    "processing": str(values[2]),
                }
            )
        return records

    def current_output_actions(self):
        records = []
        for item in self.output_mapping_tree.get_children():
            values = self.output_mapping_tree.item(item, "values")
            records.append(
                {
                    "model_output": str(values[0]),
                    "application_action": str(values[1]),
                }
            )
        return records

    def uses_local_model(self):
        return self.integration_mode_var.get() in {
            INTEGRATION_MODE_LOCAL_MODEL,
            INTEGRATION_MODE_HYBRID,
        }

    def uses_provider_api(self):
        return self.integration_mode_var.get() in {
            INTEGRATION_MODE_PROVIDER_API,
            INTEGRATION_MODE_HYBRID,
        }

    def refresh_provider_integration_status(self):
        owner = self._get_ai_settings_owner()
        provider = owner.ai_provider_var.get()
        model = owner.ai_model_var.get().strip() or "No model selected"
        state = getattr(owner, "ai_connection_state", "offline")
        state_label = {
            "connected": "CONNECTED",
            "testing": "CHECKING",
            "configured": "NOT VERIFIED",
            "disconnected": "FAILED",
            "offline": "OFFLINE",
        }.get(state, state.upper())
        self.provider_integration_status_var.set(
            f"Provider API: {state_label} — {provider} — {model}. "
            "Credentials stay in Settings and are not copied into generated code."
        )

    def open_provider_settings(self):
        owner = self._get_ai_settings_owner()
        owner.show_settings()
        owner.deiconify()
        owner.lift()
        self.lower(owner)
        self.after(500, self.refresh_provider_integration_status)

    def on_integration_mode_changed(self, _event=None):
        use_local = self.uses_local_model()
        use_provider = self.uses_provider_api()
        local_state = tk.NORMAL if use_local else tk.DISABLED
        provider_state = "readonly" if use_provider else tk.DISABLED
        if hasattr(self, "select_model_button"):
            self.select_model_button.config(state=local_state)
            self.select_model_directory_button.config(state=local_state)
            self.export_combo.config(
                state="readonly" if use_local else tk.DISABLED
            )
            self.ai_api_purpose_combo.config(state=provider_state)
        if hasattr(self, "integration_notebook"):
            self.integration_notebook.tab(
                self.mapping_tab,
                state="normal" if use_local else "disabled",
            )
        if not use_local and hasattr(self, "model_summary_text"):
            self._set_text(
                self.model_summary_text,
                "This path adds the configured AI provider API to the existing "
                "application. A locally trained model is not required. The "
                "generated client reads its credential from an environment variable.",
            )
        elif use_local and self.descriptor:
            self.populate_model_summary()
        elif use_local and hasattr(self, "model_summary_text"):
            self._set_text(
                self.model_summary_text,
                "Select a saved model package, YOLO weight, or exported detector.",
            )
        self.refresh_provider_integration_status()
        self.invalidate_generated_files()

    def on_target_changed(self, _event=None):
        self.protocol_var.set(
            deployment_default_protocol(self.target_var.get())
        )
        self.invalidate_generated_files()

    def invalidate_generated_files(self):
        self.generated_files = {}
        self.current_generated_preview_name = None
        self.ai_code_result = None
        self.ai_code_validation = None
        if hasattr(self, "ai_code_result_combo"):
            self.ai_code_result_combo.config(values=["Integration plan"])
            self.ai_code_result_file_var.set("Integration plan")
        if hasattr(self, "ai_code_result_text"):
            self._set_text(
                self.ai_code_result_text,
                "The model, target, protocol, or mapping changed. Generate a "
                "new AI integration proposal from the current contract.",
            )
        self.compatibility_status_var.set(
            "Configuration changed — run the compatibility check."
        )
        self.package_status_var.set(
            "Generate code and recheck compatibility before packaging."
        )
        if hasattr(self, "ai_code_status_var"):
            self.ai_code_status_var.set(
                "Model integration contract changed — generate a new proposal."
            )

    def run_compatibility_check(self, select_tab=True):
        reports = []
        if self.uses_local_model():
            reports.append(
                deployment_compatibility_report(
                    self.descriptor,
                    self.target_var.get(),
                    self.protocol_var.get(),
                    self.current_input_mappings(),
                    self.export_format_var.get(),
                )
            )
        if self.uses_provider_api():
            reports.append(
                provider_api_compatibility_report(
                    self._get_ai_settings_owner(),
                    self.target_var.get(),
                    self.ai_api_purpose_var.get(),
                )
            )
        report = {
            "failures": [item for value in reports for item in value["failures"]],
            "warnings": [item for value in reports for item in value["warnings"]],
            "passed": [item for value in reports for item in value["passed"]],
        }
        report["status"] = "Ready" if not report["failures"] else "Action required"
        if not report["failures"] and report["warnings"]:
            report["status"] = "Ready with warnings"
        self.compatibility_report = report
        self.compatibility_status_var.set(report["status"])
        lines = [
            f"STATUS: {report['status']}",
            f"Target: {self.target_var.get()}",
            f"Communication: {self.protocol_var.get()}",
            f"Integration path: {self.integration_mode_var.get()}",
            (
                f"Requested model format: {self.export_format_var.get()}"
                if self.uses_local_model()
                else f"AI API purpose: {self.ai_api_purpose_var.get()}"
            ),
        ]
        for heading, key, marker in (
            ("Passed", "passed", "✓"),
            ("Warnings", "warnings", "⚠"),
            ("Action Required", "failures", "✗"),
        ):
            lines.extend(["", heading + ":"])
            values = report[key]
            lines.extend(
                [f"{marker} {value}" for value in values]
                or ["• None"]
            )
        self._set_text(self.compatibility_text, "\n".join(lines))
        if select_tab:
            self.notebook.select(self.test_tab)
        return report

    def test_configured_provider(self):
        owner = self._get_ai_settings_owner()
        if owner.ai_provider_var.get() not in OFFICIAL_PROVIDER_SETTINGS:
            messagebox.showwarning(
                "AI Provider Not Configured",
                "Open Provider Settings and select OpenAI, DeepSeek, Claude, "
                "or another compatible provider first.",
                parent=self,
            )
            return
        try:
            owner.start_ai_connection_test()
        except Exception as exc:
            messagebox.showerror("Provider Test", str(exc), parent=self)
            return
        self.sample_status_var.set("Testing the configured AI provider…")
        self.after(400, lambda: self._poll_provider_test(0))

    def _poll_provider_test(self, attempt):
        if not self.winfo_exists():
            return
        owner = self._get_ai_settings_owner()
        self.refresh_provider_integration_status()
        if getattr(owner, "ai_connection_testing", False) and attempt < 300:
            self.after(400, lambda: self._poll_provider_test(attempt + 1))
            return
        state = getattr(owner, "ai_connection_state", "offline")
        self.sample_status_var.set(
            "AI provider connected and ready."
            if state == "connected"
            else "AI provider test did not connect. Review Settings."
        )

    def select_sample_file(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Select Sample Input",
            filetypes=[
                ("Data / media", "*.csv *.xlsx *.xls *.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.sample_path = path
            self.sample_status_var.set(Path(path).name)

    def select_sample_folder(self):
        path = filedialog.askdirectory(
            parent=self,
            title="Select Sample Image / Data Folder",
        )
        if path:
            self.sample_path = path
            self.sample_status_var.set(Path(path).name)

    def validate_sample_input(self):
        if not self.uses_local_model():
            messagebox.showinfo(
                "No Local Model Test Required",
                "This integration uses an AI provider API without a local model. "
                "Use Test Configured AI Provider, then generate the request-test "
                "client in Step 4.",
                parent=self,
            )
            return
        if not self.descriptor:
            messagebox.showwarning(
                "No Model",
                "Select a model first.",
                parent=self,
            )
            return
        if not self.sample_path:
            messagebox.showwarning(
                "No Sample",
                "Select a sample file or folder first.",
                parent=self,
            )
            return
        path = Path(self.sample_path)
        try:
            data_mode = self.descriptor.get("data_mode")
            if data_mode == DATA_MODE_TABULAR:
                if path.suffix.lower() == ".csv":
                    frame = pd.read_csv(path, nrows=10)
                elif path.suffix.lower() in {".xlsx", ".xls"}:
                    frame = pd.read_excel(path, nrows=10)
                else:
                    raise ValueError(
                        "Signal/tabular models require CSV or Excel sample data."
                    )
                required = [
                    item["application_input"]
                    for item in self.current_input_mappings()
                ]
                missing = [
                    name for name in required if name not in frame.columns
                ]
                if missing:
                    raise ValueError(
                        "Sample data is missing mapped application inputs: "
                        + ", ".join(missing)
                    )
                numeric = [
                    name
                    for name in required
                    if not pd.api.types.is_numeric_dtype(frame[name])
                ]
                if numeric:
                    raise ValueError(
                        "Mapped model inputs must be numeric: "
                        + ", ".join(numeric)
                    )
                detail = (
                    f"Passed — {len(frame.columns)} columns detected; all "
                    f"{len(required)} mapped inputs are available."
                )
            else:
                if not path.exists():
                    raise ValueError("The selected media path does not exist.")
                supported = SUPPORTED_IMAGE_EXTENSIONS | {
                    ".mp4",
                    ".avi",
                    ".mov",
                    ".mkv",
                }
                if path.is_file() and path.suffix.lower() not in supported:
                    raise ValueError(
                        "Select an image, video, or folder compatible with "
                        "this model."
                    )
                detail = (
                    "Passed — the selected image/video source is available. "
                    "Use Actual Prediction / Live Test to verify shape, "
                    "decoding, latency, and model output."
                )
            self.sample_status_var.set(detail)
            messagebox.showinfo(
                "Sample Validation Passed",
                detail,
                parent=self,
            )
        except Exception as exc:
            self.sample_status_var.set("Failed — " + str(exc))
            messagebox.showerror(
                "Sample Validation Failed",
                str(exc),
                parent=self,
            )

    def open_actual_test(self):
        if not self.uses_local_model():
            messagebox.showinfo(
                "Provider API Test",
                "Use Test Configured AI Provider on this page. Step 4 generates "
                "a runnable private test client and, for API targets, a backend "
                "test endpoint.",
                parent=self,
            )
            return
        if not self.selected_model_path or not self.descriptor:
            messagebox.showwarning(
                "No Model",
                "Select a model before opening the application test.",
                parent=self,
            )
            return
        if self.descriptor["model_type"] == "keras_package":
            ModelEvaluationWindow(
                self.master,
                initial_package_path=self.selected_model_path,
            )
        else:
            ObjectDetectionWindow(
                self.master,
                initial_model_path=self.selected_model_path,
                application_mode=True,
            )

    def generate_code(self):
        report = self.run_compatibility_check()
        if report["failures"]:
            messagebox.showwarning(
                "Compatibility Action Required",
                "Resolve the listed compatibility problems before generating "
                "deployment code.",
                parent=self,
            )
            return
        generated = {}
        if self.uses_local_model():
            artifact_name = (
                "model/source_model"
                if Path(self.selected_model_path).is_dir()
                else "model/" + Path(self.selected_model_path).name
            )
            generated.update(
                generate_deployment_text_artifacts(
                    self.descriptor,
                    self.target_var.get(),
                    self.protocol_var.get(),
                    self.current_input_mappings(),
                    self.current_output_actions(),
                    self.export_format_var.get(),
                    artifact_name,
                )
            )
        if self.uses_provider_api():
            owner = self._get_ai_settings_owner()
            provider_files = generate_ai_provider_api_artifacts(
                owner.ai_provider_var.get(),
                owner.ai_model_var.get().strip(),
                owner.ai_base_url_var.get(),
                self.ai_api_purpose_var.get(),
                self.target_var.get(),
                self.protocol_var.get(),
            )
            if generated:
                provider_files["AI_API_README.md"] = provider_files.pop("README.md")
                provider_requirements = provider_files.pop("requirements.txt", "")
                local_requirements = generated.get("requirements.txt", "")
                generated["requirements.txt"] = "\n".join(
                    dict.fromkeys(
                        line
                        for line in (local_requirements + "\n" + provider_requirements).splitlines()
                        if line.strip()
                    )
                ) + "\n"
            generated.update(provider_files)
        self.generated_files = generated
        names = sorted(self.generated_files)
        self.generated_file_combo.config(values=names)
        self.generated_file_var.set(
            "app.py" if "app.py" in names else names[0]
        )
        self.show_selected_generated_file()
        self.notebook.select(self.output_tab)
        self.output_notebook.select(self.code_tab)
        self.package_status_var.set(
            "Integration code generated. Review and validate it before export."
        )

    def show_selected_generated_file(self, _event=None):
        if (
            self.current_generated_preview_name
            and self.current_generated_preview_name in self.generated_files
        ):
            self.generated_files[
                self.current_generated_preview_name
            ] = self.integration_code_text.get(
                "1.0", tk.END
            ).rstrip() + "\n"
        name = self.generated_file_var.get()
        value = self.generated_files.get(name, "")
        self.integration_code_text.config(state=tk.NORMAL)
        self.integration_code_text.delete("1.0", tk.END)
        self.integration_code_text.insert("1.0", value)
        self.current_generated_preview_name = name

    def _capture_current_code(self):
        name = (
            self.current_generated_preview_name
            or self.generated_file_var.get()
        )
        if name and self.generated_files:
            self.generated_files[name] = self.integration_code_text.get(
                "1.0", tk.END
            ).rstrip() + "\n"

    def validate_generated_code(self):
        if not self.generated_files:
            self.generate_code()
            if not self.generated_files:
                return
        self._capture_current_code()
        errors = []
        checked = 0
        for name, value in self.generated_files.items():
            if not name.endswith(".py"):
                continue
            checked += 1
            try:
                ast.parse(value, filename=name)
            except SyntaxError as exc:
                errors.append(f"{name}: line {exc.lineno}: {exc.msg}")
        if errors:
            messagebox.showerror(
                "Python Validation Failed",
                "\n".join(errors),
                parent=self,
            )
        else:
            messagebox.showinfo(
                "Python Validation Passed",
                f"{checked} generated Python file(s) passed syntax checking.",
                parent=self,
            )

    def save_current_code(self):
        if not self.generated_files:
            self.generate_code()
            if not self.generated_files:
                return
        self._capture_current_code()
        name = self.generated_file_var.get() or "integration.py"
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Generated Integration File",
            initialfile=name,
            defaultextension=Path(name).suffix,
            filetypes=[("All files", "*.*")],
        )
        if not path:
            return
        Path(path).write_text(
            self.generated_files[name],
            encoding="utf-8",
        )
        messagebox.showinfo(
            "Code Saved",
            f"Saved:\n{path}",
            parent=self,
        )

    @staticmethod
    def _artifact_checksum(path):
        path = Path(path)
        digest = hashlib.sha256()
        if path.is_file():
            with path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        for child in sorted(path.rglob("*")):
            if child.is_file():
                digest.update(
                    str(child.relative_to(path)).encode("utf-8")
                )
                with child.open("rb") as source:
                    for block in iter(
                        lambda: source.read(1024 * 1024),
                        b"",
                    ):
                        digest.update(block)
        return digest.hexdigest()

    def create_deployment_package(self):
        report = self.run_compatibility_check()
        if report["failures"]:
            messagebox.showerror(
                "Deployment Not Ready",
                "Resolve the Action Required items before creating a package.",
                parent=self,
            )
            return
        if not self.generated_files:
            self.generate_code()
            if not self.generated_files:
                return
        self._capture_current_code()
        package_stem = (
            Path(self.selected_model_path).stem
            if self.uses_local_model() and self.selected_model_path
            else self._get_ai_settings_owner().ai_provider_var.get().lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )
        default_name = (
            package_stem
            + "_"
            + self.target_var.get().lower().replace(" ", "_").replace("/", "_")
            + "_deployment.zip"
        )
        save_path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Complete Deployment Package",
            initialfile=default_name,
            defaultextension=".zip",
            filetypes=[("ZIP Deployment Package", "*.zip")],
        )
        if not save_path:
            return

        temp_dir = Path(tempfile.mkdtemp(prefix="nn_studio_deploy_"))
        try:
            for relative_name, value in self.generated_files.items():
                destination = temp_dir / relative_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(value, encoding="utf-8")

            source_model_manifest = None
            if self.uses_local_model():
                model_dir = temp_dir / "model"
                model_dir.mkdir(parents=True, exist_ok=True)
                source = Path(self.selected_model_path)
                if source.is_dir():
                    destination = model_dir / "source_model"
                    shutil.copytree(source, destination)
                    model_artifact = "model/source_model"
                else:
                    destination = model_dir / source.name
                    shutil.copy2(source, destination)
                    model_artifact = f"model/{source.name}"
                source_model_manifest = {
                    "artifact": model_artifact,
                    "sha256": self._artifact_checksum(source),
                    "descriptor": self.descriptor,
                }

            provider_manifest = None
            if self.uses_provider_api():
                owner = self._get_ai_settings_owner()
                provider = owner.ai_provider_var.get()
                provider_manifest = {
                    "provider": provider,
                    "model": owner.ai_model_var.get().strip(),
                    "base_url": owner.ai_base_url_var.get().strip(),
                    "api_key_environment": OFFICIAL_PROVIDER_SETTINGS[provider][
                        "api_key_environment"
                    ],
                    "api_key_included": False,
                    "purpose": self.ai_api_purpose_var.get(),
                }

            manifest = {
                "schema_version": 1,
                "application_version": APP_VERSION,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "integration_mode": self.integration_mode_var.get(),
                "source_model": source_model_manifest,
                "ai_provider_api": provider_manifest,
                "deployment": {
                    "target": self.target_var.get(),
                    "protocol": self.protocol_var.get(),
                    "requested_export_format": self.export_format_var.get(),
                    "input_mappings": self.current_input_mappings(),
                    "output_actions": self.current_output_actions(),
                },
                "compatibility": report,
                "privacy": {
                    "api_keys_included": False,
                    "cloud_tokens_included": False,
                    "training_dataset_included": False,
                },
            }
            (temp_dir / "deployment_manifest.json").write_text(
                json.dumps(
                    manifest,
                    indent=2,
                    default=_result_json_default,
                ),
                encoding="utf-8",
            )
            (temp_dir / "compatibility_report.txt").write_text(
                self.compatibility_text.get("1.0", tk.END),
                encoding="utf-8",
            )
            zip_result_directory(temp_dir, save_path)
            self.package_status_var.set(
                f"Deployment package saved: {save_path}"
            )
            messagebox.showinfo(
                "Deployment Package Created",
                "The selected integration artifacts, editable source code, "
                "dependencies, compatibility evidence, configuration, and "
                f"instructions were saved:\n{save_path}",
                parent=self,
            )
        except Exception as exc:
            messagebox.showerror(
                "Deployment Package Error",
                str(exc),
                parent=self,
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
