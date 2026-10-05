"""Ui / wizard home for NN Training Studio."""

from pathlib import Path
from tkinter import messagebox
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from nn_training_studio.constants import (
    DATA_MODE_IMAGE,
    DATA_MODE_TABULAR,
    IMAGE_MODEL_CNN,
    TASK_CLASSIFICATION,
    WORKSPACE_IMAGE,
    WORKSPACE_SIGNAL,
)
from nn_training_studio.image_data import (
    build_image_dataset_profile,
    scan_image_dataset,
)


class HomeMixin:
    """Home behavior for the main application."""

    def show_home(self):
        if self.training_running or self.ai_request_running:
            messagebox.showinfo(
                "Operation Running",
                "Wait for the active training or AI request to finish before "
                "returning home.",
            )
            return
        self.current_view = "home"
        self.clear_content()
        self.title_label.config(text="Neural Network Training Studio")
        self.progress_label.config(
            text="Create AI, prepare data, or connect AI to your application."
        )
        self.set_workflow_navigation_visible(False)
        self.home_button.config(state=tk.DISABLED)
        self.settings_button.config(state=tk.NORMAL)
        self.build_home_page()
        self.update_provider_status_display()

    def build_advanced_home_page(self):
        hero = ttk.Frame(self.content_frame)
        self.home_hero = hero
        hero.pack(fill=tk.X, padx=12, pady=(10, 6))
        hero.columnconfigure(1, weight=1)

        home_logo = getattr(self, "_nn_studio_icon_photo", None)
        if home_logo is not None:
            self.home_logo_label = ttk.Label(
                hero,
                image=home_logo,
            )
            self.home_logo_label.grid(
                row=0,
                column=0,
                rowspan=2,
                sticky="nw",
                padx=(0, 12),
            )

        self.home_hero_title = ttk.Label(
            hero,
            text="What do you want to achieve?",
            font=("Arial", 18, "bold"),
        )
        self.home_hero_title.grid(row=0, column=1, sticky="w")
        self.home_hero_description = ttk.Label(
            hero,
            text=(
                "Start with the required result. NN Studio will keep training, "
                "model application, data preparation, and project management "
                "in separate focused workspaces."
            ),
            wraplength=850,
        )
        self.home_hero_description.grid(
            row=1,
            column=1,
            sticky="w",
            pady=(3, 0),
        )
        self.home_help_button = ttk.Button(
            hero,
            text="Help me choose",
            command=lambda: self.open_studio_guide(
                "Help me choose the correct NN Studio workspace."
            ),
        )
        self.home_help_button.grid(
            row=0,
            column=2,
            rowspan=2,
            sticky="e",
            padx=(18, 0),
        )

        status_frame = ttk.Frame(self.content_frame)
        self.home_status_frame = status_frame
        status_frame.pack(fill=tk.X, padx=12, pady=(0, 6))
        status_frame.columnconfigure(1, weight=1)
        if self.df is not None:
            if self.data_mode_var.get() == DATA_MODE_IMAGE:
                project_status = "Current project: image classification"
            else:
                project_status = "Current project: signal / tabular"
        else:
            project_status = "Current project: none"
        self.home_project_status_label = ttk.Label(
            status_frame,
            text=project_status,
            foreground="#245a85",
        )
        self.home_project_status_label.grid(
            row=0,
            column=0,
            sticky="w",
        )
        self.home_provider_status_label = ttk.Label(
            status_frame,
            text="  •  " + self.ai_connection_message,
            foreground="#666666",
            wraplength=760,
        )
        self.home_provider_status_label.grid(
            row=0,
            column=1,
            sticky="w",
            padx=(4, 0),
        )

        create_frame = ttk.LabelFrame(
            self.content_frame,
            text="1. Create & Train a New Model",
        )
        self.home_create_frame = create_frame
        create_frame.pack(fill=tk.X, padx=12, pady=6)
        for column in range(3):
            create_frame.columnconfigure(
                column,
                weight=1,
                uniform="create_action",
            )

        signal_action = (
            "Continue Project"
            if self.df is not None
            and self.data_mode_var.get() == DATA_MODE_TABULAR
            else "Start Signal Project"
        )
        signal_card = self.add_home_action(
            create_frame,
            row=0,
            column=0,
            title="Signal / Tabular",
            description=(
                "CSV or Excel → class, continuous values, future signals, "
                "or reconstructed signals."
            ),
            button_text=signal_action,
            command=self.start_signal_project,
        )

        image_action = (
            "Continue Project"
            if self.df is not None
            and self.data_mode_var.get() == DATA_MODE_IMAGE
            else "Start Image Project"
        )
        image_card = self.add_home_action(
            create_frame,
            row=0,
            column=1,
            title="Image Classification",
            description=(
                "Class-named image folders → one predicted class for each "
                "complete image."
            ),
            button_text=image_action,
            command=self.start_image_project,
        )

        detection_card = self.add_home_action(
            create_frame,
            row=0,
            column=2,
            title="Object Detection",
            description=(
                "YOLO images and bounding boxes → classes, confidence, and "
                "object locations."
            ),
            button_text="Open Detection",
            command=self.open_object_detection,
        )
        self.home_create_cards = [signal_card, image_card, detection_card]

        middle = ttk.Frame(self.content_frame)
        self.home_middle = middle
        middle.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)
        middle.columnconfigure(0, weight=2, uniform="middle")
        middle.columnconfigure(1, weight=1, uniform="middle")
        middle.rowconfigure(0, weight=1)

        apply_frame = ttk.LabelFrame(
            middle,
            text="2. Apply, Validate, Deploy & Integrate",
        )
        self.home_apply_frame = apply_frame
        apply_frame.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(0, 6),
        )
        ttk.Label(
            apply_frame,
            text="Load & Use Existing Model",
            font=("Arial", 13, "bold"),
        ).pack(anchor="w", padx=14, pady=(12, 4))
        ttk.Label(
            apply_frame,
            text=(
                "Open signal, image-classification, or object-detection "
                "models. Check compatibility, run prediction/evaluation, "
                "preview outputs inside NN Studio, customise validation "
                "graphs manually or with AI, and export complete results."
            ),
            wraplength=650,
            justify=tk.LEFT,
        ).pack(anchor="w", padx=14, pady=(0, 10))
        ttk.Button(
            apply_frame,
            text="Load & Use Model",
            command=self.open_model_evaluation,
        ).pack(anchor="w", padx=14, pady=(0, 8))
        ttk.Separator(
            apply_frame,
            orient=tk.HORIZONTAL,
        ).pack(fill=tk.X, padx=14, pady=(2, 8))
        ttk.Label(
            apply_frame,
            text="Deploy & Integrate",
            font=("Arial", 12, "bold"),
        ).pack(anchor="w", padx=14, pady=(0, 3))
        ttk.Label(
            apply_frame,
            text=(
                "Map a finished model to software, APIs, Docker, edge devices, "
                "cameras, sensors, MQTT, Modbus TCP, OPC UA, or cloud services. "
                "The AI Code Assistant can also analyse approved files from an "
                "existing website, program, or hardware SDK and prepare an "
                "integrated project copy."
            ),
            wraplength=650,
            justify=tk.LEFT,
        ).pack(anchor="w", padx=14, pady=(0, 7))
        ttk.Button(
            apply_frame,
            text="Open Deploy & Integrate",
            command=self.open_deploy_integrate,
        ).pack(anchor="w", padx=14, pady=(0, 12))

        prepare_frame = ttk.LabelFrame(
            middle,
            text="3. Prepare Data",
        )
        self.home_prepare_frame = prepare_frame
        prepare_frame.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=(6, 0),
        )
        ttk.Label(
            prepare_frame,
            text="Annotate & Prepare Data",
            font=("Arial", 12, "bold"),
        ).pack(anchor="w", padx=14, pady=(12, 4))
        ttk.Label(
            prepare_frame,
            text=(
                "Label 1D signal intervals/events or 2D image classes/boxes. "
                "Use local or AI-assisted suggestions, review every label, "
                "then export or continue to training."
            ),
            wraplength=330,
            justify=tk.LEFT,
        ).pack(anchor="w", padx=14, pady=(0, 7))
        ttk.Button(
            prepare_frame,
            text="Open Annotate & Prepare",
            command=self.open_annotation_workspace,
        ).pack(anchor="w", padx=14, pady=(0, 8))
        ttk.Separator(prepare_frame, orient=tk.HORIZONTAL).pack(
            fill=tk.X, padx=14, pady=(0, 8)
        )
        ttk.Label(
            prepare_frame,
            text="Filter & Export",
            font=("Arial", 12, "bold"),
        ).pack(anchor="w", padx=14, pady=(0, 4))
        ttk.Label(
            prepare_frame,
            text=(
                "Clean selected CSV/Excel signals, compare before and after, "
                "save data or presets, then optionally continue to training."
            ),
            wraplength=330,
            justify=tk.LEFT,
        ).pack(anchor="w", padx=14, pady=(0, 10))
        ttk.Button(
            prepare_frame,
            text="Open Filter & Export",
            command=self.show_filter_workspace,
        ).pack(anchor="w", padx=14, pady=(0, 12))

        manage_frame = ttk.LabelFrame(
            self.content_frame,
            text="4. Manage Studio",
        )
        self.home_manage_frame = manage_frame
        manage_frame.pack(fill=tk.X, padx=12, pady=(6, 12))
        self.home_manage_buttons = ttk.Frame(manage_frame)
        self.home_manage_buttons.grid(
            row=0,
            column=0,
            sticky="w",
            padx=(8, 4),
            pady=8,
        )
        manage_frame.columnconfigure(1, weight=1)
        self.home_projects_button = ttk.Button(
            self.home_manage_buttons,
            text="Projects & Checkpoints",
            command=self.show_projects,
        )
        self.home_projects_button.grid(row=0, column=0, padx=4, pady=2)
        self.home_settings_button = ttk.Button(
            self.home_manage_buttons,
            text="Settings & AI Provider",
            command=self.show_settings,
        )
        self.home_settings_button.grid(row=0, column=1, padx=4, pady=2)
        self.home_guide_button = ttk.Button(
            self.home_manage_buttons,
            text="NN Studio Guide",
            command=self.open_studio_guide,
        )
        self.home_guide_button.grid(row=0, column=2, padx=4, pady=2)
        self.home_manage_description = ttk.Label(
            manage_frame,
            text=(
                "Recover runs, inspect checkpoints, configure providers, or "
                "ask for context-aware training and deployment help."
            ),
            foreground="#666666",
            wraplength=700,
        )
        self.home_manage_description.grid(
            row=0,
            column=1,
            sticky="w",
            padx=8,
            pady=10,
        )
        self.after_idle(self._apply_responsive_layout)

    def _reflow_advanced_home_page(self, width):
        """Rearrange homepage cards instead of allowing edge clipping."""
        required = (
            "home_hero",
            "home_create_frame",
            "home_middle",
            "home_apply_frame",
            "home_prepare_frame",
            "home_manage_frame",
        )
        if not all(hasattr(self, name) for name in required):
            return
        compact = width < 920
        medium = 920 <= width < 1400
        try:
            # Hero: move the guide action below the introduction when narrow.
            self.home_help_button.grid_forget()
            self.home_hero.columnconfigure(1, weight=1)
            if compact or medium:
                self.home_hero_title.configure(
                    font=("Arial", 15 if compact else 18, "bold")
                )
                self.home_help_button.grid(
                    row=2,
                    column=1,
                    sticky="w",
                    pady=(7, 0),
                )
            else:
                self.home_hero_title.configure(font=("Arial", 18, "bold"))
                self.home_help_button.grid(
                    row=0,
                    column=2,
                    rowspan=2,
                    sticky="e",
                    padx=(18, 0),
                )

            # Status: stack connection details below the project on compact UI.
            self.home_project_status_label.grid_forget()
            self.home_provider_status_label.grid_forget()
            if compact:
                self.home_project_status_label.grid(
                    row=0,
                    column=0,
                    columnspan=2,
                    sticky="w",
                )
                self.home_provider_status_label.grid(
                    row=1,
                    column=0,
                    columnspan=2,
                    sticky="w",
                    padx=0,
                    pady=(2, 0),
                )
                self.home_provider_status_label.configure(
                    text=self.ai_connection_message
                )
            else:
                self.home_project_status_label.grid(
                    row=0,
                    column=0,
                    sticky="w",
                )
                self.home_provider_status_label.grid(
                    row=0,
                    column=1,
                    sticky="w",
                    padx=(4, 0),
                )
                self.home_provider_status_label.configure(
                    text="  •  " + self.ai_connection_message
                )

            # Training cards: 3 columns wide, 2+1 medium, 1 compact.
            for card in self.home_create_cards:
                card.grid_forget()
            for column in range(3):
                self.home_create_frame.columnconfigure(
                    column,
                    weight=0,
                    uniform="",
                )
            column_count = 1 if compact else 2 if medium else 3
            for column in range(column_count):
                self.home_create_frame.columnconfigure(
                    column,
                    weight=1,
                    uniform="home_create",
                )
            for index, card in enumerate(self.home_create_cards):
                card.grid(
                    row=index // column_count,
                    column=index % column_count,
                    sticky="nsew",
                    padx=10,
                    pady=8,
                )

            # Application/data panels stack below the wide desktop breakpoint.
            self.home_apply_frame.grid_forget()
            self.home_prepare_frame.grid_forget()
            self.home_middle.columnconfigure(0, weight=1, uniform="")
            self.home_middle.columnconfigure(1, weight=0, uniform="")
            if compact or medium:
                self.home_apply_frame.grid(
                    row=0,
                    column=0,
                    sticky="nsew",
                    pady=(0, 6),
                )
                self.home_prepare_frame.grid(
                    row=1,
                    column=0,
                    sticky="nsew",
                    pady=(6, 0),
                )
            else:
                self.home_middle.columnconfigure(0, weight=2, uniform="middle")
                self.home_middle.columnconfigure(1, weight=1, uniform="middle")
                self.home_apply_frame.grid(
                    row=0,
                    column=0,
                    sticky="nsew",
                    padx=(0, 6),
                )
                self.home_prepare_frame.grid(
                    row=0,
                    column=1,
                    sticky="nsew",
                    padx=(6, 0),
                )

            # Management actions wrap to two rows when space is limited.
            for button in (
                self.home_projects_button,
                self.home_settings_button,
                self.home_guide_button,
            ):
                button.grid_forget()
            self.home_manage_description.grid_forget()
            if compact:
                self.home_manage_frame.columnconfigure(0, weight=1)
                self.home_manage_frame.columnconfigure(1, weight=0)
                self.home_manage_buttons.grid(
                    row=0,
                    column=0,
                    sticky="ew",
                    padx=8,
                    pady=(8, 2),
                )
                self.home_manage_buttons.columnconfigure(0, weight=1)
                self.home_manage_buttons.columnconfigure(1, weight=1)
                self.home_projects_button.grid(
                    row=0, column=0, sticky="ew", padx=3, pady=2
                )
                self.home_settings_button.grid(
                    row=0, column=1, sticky="ew", padx=3, pady=2
                )
                self.home_guide_button.grid(
                    row=1,
                    column=0,
                    columnspan=2,
                    sticky="ew",
                    padx=3,
                    pady=2,
                )
                self.home_manage_description.grid(
                    row=1,
                    column=0,
                    sticky="ew",
                    padx=12,
                    pady=(2, 10),
                )
            elif medium:
                self.home_manage_frame.columnconfigure(0, weight=1)
                self.home_manage_frame.columnconfigure(1, weight=0)
                self.home_manage_buttons.grid(
                    row=0,
                    column=0,
                    sticky="w",
                    padx=8,
                    pady=(8, 2),
                )
                self.home_projects_button.grid(row=0, column=0, padx=4, pady=2)
                self.home_settings_button.grid(row=0, column=1, padx=4, pady=2)
                self.home_guide_button.grid(row=0, column=2, padx=4, pady=2)
                self.home_manage_description.grid(
                    row=1,
                    column=0,
                    sticky="ew",
                    padx=12,
                    pady=(2, 10),
                )
            else:
                self.home_manage_frame.columnconfigure(0, weight=0)
                self.home_manage_frame.columnconfigure(1, weight=1)
                self.home_manage_buttons.grid(
                    row=0,
                    column=0,
                    sticky="w",
                    padx=(8, 4),
                    pady=8,
                )
                self.home_projects_button.grid(row=0, column=0, padx=4, pady=2)
                self.home_settings_button.grid(row=0, column=1, padx=4, pady=2)
                self.home_guide_button.grid(row=0, column=2, padx=4, pady=2)
                self.home_manage_description.grid(
                    row=0,
                    column=1,
                    sticky="w",
                    padx=8,
                    pady=10,
                )
        except tk.TclError:
            pass

    def add_home_action(
        self,
        parent,
        row,
        column,
        title,
        description,
        button_text,
        command,
    ):
        """Create a compact goal-based training action."""
        card = ttk.Frame(parent)
        card.grid(
            row=row,
            column=column,
            sticky="nsew",
            padx=10,
            pady=10,
        )
        ttk.Label(
            card,
            text=title,
            font=("Arial", 12, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            card,
            text=description,
            wraplength=320,
            justify=tk.LEFT,
        ).pack(anchor="w", fill=tk.X, pady=(4, 8))
        ttk.Button(
            card,
            text=button_text,
            command=command,
        ).pack(anchor="w")
        return card

    def add_home_card(
        self,
        parent,
        row,
        column,
        title,
        description=None,
        button_text="Open",
        command=None,
        text=None,
    ):
        """Create one consistent homepage workspace card."""
        card = ttk.LabelFrame(parent, text=title)
        card.grid(
            row=row,
            column=column,
            sticky="nsew",
            padx=7,
            pady=7,
        )
        ttk.Label(
            card,
            text=description if description is not None else text,
            wraplength=315,
            justify=tk.LEFT,
        ).pack(anchor="w", padx=14, pady=(16, 12))
        ttk.Button(
            card,
            text=button_text,
            command=command,
        ).pack(anchor="w", padx=14, pady=(0, 14))
        return card

    def set_training_workspace(self, data_mode):
        """Set the wizard's mode and its mode-specific page names."""
        self.data_mode_var.set(data_mode)
        if data_mode == DATA_MODE_IMAGE:
            self.active_workspace = WORKSPACE_IMAGE
            self.step_titles = list(self.image_step_titles)
        else:
            self.active_workspace = WORKSPACE_SIGNAL
            self.step_titles = list(self.signal_step_titles)

    def reset_training_project(self, data_mode):
        """Clear project data while preserving global provider settings."""
        self.clear_content()
        self.early_stopping_enabled_var.set(True)
        self.early_stopping_patience_var.set("10")
        self.early_stopping_min_delta_var.set("0")
        self.early_stopping_warmup_var.set("0")
        self.guided_stage = 0
        self.guided_detail_step = None
        self.project_goal_var.set("")
        self.project_draft_path = None
        self._editor_drafts = {}
        self._pending_preparation_steps = set()
        self._last_training_log = ""
        self.last_saved_model_package_path = None
        self.filter_method_var.set("No filter")
        self.task_type_var.set(TASK_CLASSIFICATION)
        self.model_type_var.set(IMAGE_MODEL_CNN if data_mode == DATA_MODE_IMAGE else "DNN")
        self.output_mode_var.set("Auto")
        self.output_activation_var.set("softmax")
        self.loss_var.set("sparse_categorical_crossentropy")
        self.guided_data_order_var.set("Independent rows")
        self.set_training_workspace(data_mode)
        self.current_step = 0
        self.raw_df = None
        self.filtered_df = None
        self.custom_filtered_df = None
        self.df = None
        self.file_path = None
        self.image_directory = None
        self.image_records = None
        self.image_class_names = []
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
        self.safe_filter_spec = None
        self.safe_model_spec = None
        self.custom_filter_enabled_var.set(False)
        self.selected_filter_cols = []
        self.selected_custom_filter_cols = []
        self.feature_cols = []
        self.label_col = ""
        self.target_cols = []
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
        self.clear_training_custom_results()

    def start_training_workspace(self, data_mode):
        if self.training_running or self.ai_request_running:
            return
        same_mode = (
            self.df is not None
            and self.data_mode_var.get() == data_mode
        )
        if same_mode:
            continue_existing = messagebox.askyesno(
                "Existing Project",
                "Continue the current project?\n\n"
                "Choose No to start a new project in this workspace.",
            )
            if continue_existing:
                self.set_training_workspace(data_mode)
                self.show_step()
                return
        elif self.df is not None:
            switch_workspace = messagebox.askyesno(
                "Switch Training Workspace",
                "Starting this workflow clears the current in-memory training "
                "project. Saved model packages and checkpoints are unchanged.\n\n"
                "Continue?",
            )
            if not switch_workspace:
                return
        self.reset_training_project(data_mode)
        self.show_step()

    def start_or_continue_project(self):
        self.start_training_workspace(self.data_mode_var.get())

    def start_signal_project(self):
        self.start_training_workspace(DATA_MODE_TABULAR)

    def start_image_project(self):
        self.start_training_workspace(DATA_MODE_IMAGE)

    def import_annotated_signal_dataframe(self, dataframe, source_path=None):
        """Transfer reviewed 1D labels into the normal training wizard."""
        if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
            raise ValueError("The annotated signal dataset is empty.")
        if "annotation_label" not in dataframe.columns:
            raise ValueError("The reviewed annotation label column is missing.")
        labelled = dataframe[dataframe["annotation_label"].notna()].copy()
        if labelled.empty:
            raise ValueError("No approved interval labels cover signal rows.")
        # Training uses only approved labelled rows. Evidence columns remain
        # available for inspection but are not selected as inputs by default.
        self.reset_training_project(DATA_MODE_TABULAR)
        self.raw_df = labelled.reset_index(drop=True)
        self.filtered_df = self.raw_df.copy()
        self.custom_filtered_df = self.raw_df.copy()
        self.df = self.raw_df.copy()
        self.file_path = str(source_path or "annotated_signal_in_memory.csv")
        self.initialize_signal_column_defaults()
        self.task_type_var.set(TASK_CLASSIFICATION)
        self.label_col = "annotation_label"
        self.label_var.set("annotation_label")
        self.target_cols = ["annotation_label"]
        excluded = {
            "annotation_label",
            "annotation_label_source",
            "annotation_label_confidence",
            "annotation_event",
            "annotation_event_source",
            "annotation_original_row",
            "annotation_segment_id",
        }
        self.feature_cols = [
            column for column in self.df.select_dtypes(include=[np.number]).columns
            if column not in excluded
        ]
        self.current_step = 0
        self.show_step()

    def import_annotated_image_folder(self, directory):
        """Transfer reviewed whole-image classes into image training."""
        image_records, class_names = scan_image_dataset(directory)
        if len(class_names) < 2:
            raise ValueError("Image classification requires two approved classes.")
        self.reset_training_project(DATA_MODE_IMAGE)
        self.image_directory = str(Path(directory).resolve())
        self.image_records = image_records.copy()
        self.image_class_names = list(class_names)
        self.raw_df = image_records.copy()
        self.filtered_df = image_records.copy()
        self.custom_filtered_df = image_records.copy()
        self.df = image_records.copy()
        self.file_path = self.image_directory
        self.dataset_profile = build_image_dataset_profile(
            image_records, class_names, self.image_directory
        )
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
        self.current_step = 0
        self.show_step()
