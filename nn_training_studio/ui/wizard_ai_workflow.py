"""Ui / wizard ai workflow for NN Training Studio."""

from tkinter.scrolledtext import ScrolledText
from datetime import datetime
import json
try:
    import keyring
except Exception:
    keyring = None
from tkinter import messagebox
import threading
import tkinter as tk
from tkinter import ttk
from urllib.parse import urlparse
from nn_training_studio.ai_providers import (
    create_ai_provider,
    format_ai_generation_report,
    format_ai_recommendation_report,
    make_ai_request_profile,
    recommendation_request_summary,
)
from nn_training_studio.ai_transport import (
    _validate_provider_base_url,
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
    AI_GOAL_MAX_CHARS,
    CUSTOM_FILTER_MODE_SAFE_AI,
    DATA_MODE_IMAGE,
    MODEL_TYPE_SAFE_CUSTOM,
    OFFICIAL_PROVIDER_SETTINGS,
    TASK_AUTOENCODER,
    TASK_CLASSIFICATION,
    TASK_FORECASTING,
    TASK_MULTI_OUTPUT,
    TASK_REGRESSION,
    TASK_TYPES,
)
from nn_training_studio.filters import (
    apply_safe_filter_spec,
)
from nn_training_studio.models import (
    build_safe_model_from_spec,
)


class AIWorkflowMixin:
    """AIWorkflow behavior for the main application."""

    def build_step_ai_analysis(self):
        if self.raw_df is None:
            ttk.Label(
                self.content_frame,
                text="Please load a dataset first."
            ).pack()
            return

        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            frame = ttk.LabelFrame(
                self.content_frame,
                text="Image Dataset Analysis",
            )
            frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            text_widget = ScrolledText(frame, wrap=tk.WORD)
            text_widget.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            profile = self.dataset_profile or {}
            text_widget.insert(
                tk.END,
                "IMAGE CLASSIFICATION PROFILE\n"
                + "=" * 72
                + "\n\n"
                f"Images: {profile.get('image_count', 0)}\n"
                f"Classes: {profile.get('class_count', 0)}\n"
                f"Class distribution: {profile.get('class_counts', {})}\n"
                f"File types: {profile.get('extension_counts', {})}\n\n"
                "Recommended starting configuration:\n"
                "- Task: multi-class image classification\n"
                "- Model: Image CNN for a reliable offline baseline\n"
                "- Image size: 224 × 224 pixels, RGB\n"
                "- Augmentation: horizontal flip, small rotation, zoom, and "
                "contrast changes\n"
                "- Loss: sparse categorical cross-entropy\n"
                "- Evaluation: accuracy, weighted F1, classification report, "
                "and confusion matrix\n\n"
                "Structured AI analysis for tabular columns is not applied to "
                "image pixels in Version 18. Image-specific model choices and "
                "preprocessing remain fully editable in the following pages."
            )
            text_widget.config(state=tk.DISABLED)
            return

        controls = ttk.LabelFrame(
            self.content_frame,
            text="Local Profile Settings"
        )
        controls.pack(fill=tk.X, padx=5, pady=(5, 8))
        controls.columnconfigure(4, weight=1)

        ttk.Label(
            controls,
            text="Sampling frequency (Hz, optional):"
        ).grid(row=0, column=0, sticky="w", padx=(8, 5), pady=8)
        ttk.Entry(
            controls,
            textvariable=self.sampling_frequency_var,
            width=14
        ).grid(row=0, column=1, sticky="w", padx=5, pady=8)

        ttk.Button(
            controls,
            text="Refresh Local Profile",
            command=self.refresh_dataset_analysis_from_ui
        ).grid(row=0, column=2, sticky="w", padx=8, pady=8)

        ttk.Button(
            controls,
            text="Apply Local Recommendation",
            command=self.apply_dataset_recommendation
        ).grid(row=0, column=3, sticky="w", padx=8, pady=8)

        ttk.Label(
            controls,
            text=(
                "Local profiling never transmits the CSV or profile."
            ),
            foreground="#245a85",
            wraplength=850,
        ).grid(
            row=1,
            column=0,
            columnspan=5,
            sticky="w",
            padx=8,
            pady=(0, 7),
        )

        provider_frame = ttk.LabelFrame(
            self.content_frame,
            text="Optional AI Assistant"
        )
        provider_frame.pack(fill=tk.X, padx=5, pady=(0, 8))

        status_row = ttk.Frame(provider_frame)
        status_row.pack(fill=tk.X, padx=8, pady=(7, 3))
        self.ai_status_label = ttk.Label(
            status_row,
            text="",
            foreground="#245a85",
        )
        self.ai_status_label.pack(side=tk.LEFT, padx=3)
        ttk.Button(
            status_row,
            text="Open Provider Settings",
            command=self.show_settings,
        ).pack(side=tk.RIGHT, padx=3)

        goal_row = ttk.Frame(provider_frame)
        goal_row.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(goal_row, text="Goal / notes:").pack(
            side=tk.LEFT, padx=(3, 6)
        )
        self.ai_goal_entry = ttk.Entry(
            goal_row,
            textvariable=self.ai_goal_var,
        )
        self.ai_goal_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        action_row = ttk.Frame(provider_frame)
        action_row.pack(fill=tk.X, padx=8, pady=(4, 5))
        for column_index in range(3):
            action_row.columnconfigure(column_index, weight=1)
        self.ai_request_button = ttk.Button(
            action_row,
            text="Request Structured AI Recommendation",
            command=self.start_ai_recommendation,
        )
        self.ai_request_button.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=3,
            pady=2,
        )
        self.ai_apply_button = ttk.Button(
            action_row,
            text="Apply AI Task, Filter, and Model Settings",
            command=self.apply_ai_recommendation,
            state=(
                tk.NORMAL
                if self.ai_recommendation is not None
                else tk.DISABLED
            ),
        )
        self.ai_apply_button.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=3,
            pady=2,
        )
        self.ai_generate_button = ttk.Button(
            action_row,
            text="Generate Safe Filter and Model",
            command=self.start_ai_generation,
            state=(
                tk.NORMAL
                if self.ai_recommendation is not None
                else tk.DISABLED
            ),
        )
        self.ai_generate_button.grid(
            row=0,
            column=2,
            sticky="ew",
            padx=3,
            pady=2,
        )
        self.ai_apply_filter_button = ttk.Button(
            action_row,
            text="Approve Safe Filter",
            command=self.apply_ai_generated_filter,
            state=(
                tk.NORMAL
                if self.ai_generation is not None
                else tk.DISABLED
            ),
        )
        self.ai_apply_filter_button.grid(
            row=1,
            column=1,
            sticky="ew",
            padx=3,
            pady=2,
        )
        self.ai_apply_model_button = ttk.Button(
            action_row,
            text="Approve Safe Model",
            command=self.apply_ai_generated_model,
            state=(
                tk.NORMAL
                if self.ai_generation is not None
                else tk.DISABLED
            ),
        )
        self.ai_apply_model_button.grid(
            row=1,
            column=2,
            sticky="ew",
            padx=3,
            pady=2,
        )

        privacy_text = (
            "Only the compact local profile (column names and statistics) is "
            "sent after you click Request. Raw CSV rows are not sent. Provider "
            "credentials are managed globally in Settings and are never saved "
            "in reports, projects, logs, or model packages."
        )
        ttk.Label(
            provider_frame,
            text=privacy_text,
            foreground="#7a4b00",
            wraplength=1100,
        ).pack(anchor="w", padx=11, pady=(0, 7))

        notebook = ttk.Notebook(self.content_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        local_tab = ttk.Frame(notebook)
        ai_tab = ttk.Frame(notebook)
        generation_tab = ttk.Frame(notebook)
        notebook.add(local_tab, text="Local Profile")
        notebook.add(ai_tab, text="Structured AI Recommendation")
        notebook.add(generation_tab, text="Safe AI Generation")

        self.analysis_text = ScrolledText(
            local_tab,
            wrap=tk.WORD,
            font=("Consolas", 10)
        )
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.ai_analysis_text = ScrolledText(
            ai_tab,
            wrap=tk.WORD,
            font=("Consolas", 10),
        )
        self.ai_analysis_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=5,
            pady=5,
        )
        if self.ai_recommendation_report:
            self.ai_analysis_text.insert(
                tk.END,
                self.ai_recommendation_report,
            )
        else:
            self.ai_analysis_text.insert(
                tk.END,
                "Connect OpenAI, DeepSeek, or Claude in global Settings, then request "
                "a structured recommendation here.\n\n"
                "This analysis is optional. Filter and model customization "
                "work independently from recommendations through manual safe "
                "JSON, optional AI assistance, or Expert Python."
            )

        self.ai_generation_text = ScrolledText(
            generation_tab,
            wrap=tk.WORD,
            font=("Consolas", 10),
        )
        self.ai_generation_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=5,
            pady=5,
        )
        if self.ai_generation_report:
            self.ai_generation_text.insert(
                tk.END,
                self.ai_generation_report,
            )
        else:
            self.ai_generation_text.insert(
                tk.END,
                "Request and validate an AI recommendation first. Then select "
                "'Generate Safe Filter and Model'. Generated specifications "
                "remain inactive until you approve them."
            )

        if self.dataset_profile is None:
            self.run_dataset_analysis(show_message=False)
        else:
            self.analysis_text.insert(
                tk.END,
                self.dataset_analysis_report
            )

        self.on_ai_provider_changed(update_existing=False)

    def get_analysis_sampling_frequency(self):
        text_value = self.sampling_frequency_var.get().strip()
        if not text_value:
            return None

        sampling_frequency = float(text_value)
        if sampling_frequency <= 0:
            raise ValueError(
                "Sampling frequency must be greater than zero."
            )
        return sampling_frequency

    def run_dataset_analysis(self, show_message=False):
        if self.raw_df is None:
            raise ValueError("Please load a dataset first.")

        sampling_frequency = self.get_analysis_sampling_frequency()
        profile = analyze_dataset_profile(
            self.raw_df,
            sampling_frequency=sampling_frequency
        )
        report = format_dataset_analysis_report(profile)

        self.dataset_profile = profile
        self.dataset_analysis_report = report

        if hasattr(self, "analysis_text"):
            self.analysis_text.delete("1.0", tk.END)
            self.analysis_text.insert(tk.END, report)
            self.analysis_text.see("1.0")

        if show_message:
            messagebox.showinfo(
                "Analysis Complete",
                "The local dataset profile and task recommendation were updated."
            )

    def refresh_dataset_analysis_from_ui(self):
        try:
            self.run_dataset_analysis(show_message=True)
        except Exception as exc:
            messagebox.showerror("Analysis Error", str(exc))

    def on_ai_provider_changed(self, event=None, update_existing=True):
        provider_name = self.ai_provider_var.get()
        is_external = provider_name in OFFICIAL_PROVIDER_SETTINGS

        if is_external and update_existing:
            defaults = OFFICIAL_PROVIDER_SETTINGS[provider_name]
            self.ai_model_var.set(defaults["model"])
            self.ai_base_url_var.set(defaults["base_url"])
            self.ai_api_key_var.set("")
            self.ai_last_connection_check = None
            self.ai_last_connection_details = (
                "The selected provider has not been tested in this session."
            )
            credential = self.resolve_ai_api_key(provider_name)
            if credential:
                self.ai_connection_state = "configured"
                self.ai_connection_message = (
                    f"{provider_name} is configured. Test the connection to "
                    "verify it."
                )
            else:
                self.ai_connection_state = "disconnected"
                self.ai_connection_message = (
                    f"{provider_name} is selected but no API key is available."
                )
        elif not is_external:
            self.ai_model_var.set("")
            self.ai_base_url_var.set("")
            self.ai_api_key_var.set("")
            self.ai_connection_state = "offline"
            self.ai_connection_message = (
                "Offline mode — manual training remains available."
            )
            self.ai_last_connection_check = None
            self.ai_last_connection_details = (
                "Offline guide and manual workflows remain available."
            )

        entry_state = tk.NORMAL if is_external else tk.DISABLED
        for widget_name in (
            "ai_model_entry",
            "ai_key_entry",
            "ai_base_url_entry",
            "ai_timeout_entry",
            "ai_goal_entry",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.config(state=entry_state)

        if hasattr(self, "ai_request_button"):
            self.ai_request_button.config(
                state=(
                    tk.DISABLED
                    if not is_external or self.ai_request_running
                    else tk.NORMAL
                )
            )
        if hasattr(self, "ai_generate_button"):
            self.ai_generate_button.config(
                state=(
                    tk.NORMAL
                    if (
                        is_external
                        and not self.ai_request_running
                        and self.ai_recommendation is not None
                    )
                    else tk.DISABLED
                )
            )
        for component_button_name in (
            "custom_filter_ai_button",
            "custom_model_ai_button",
            "filter_tool_ai_button",
        ):
            component_button = getattr(
                self, component_button_name, None
            )
            if component_button is not None:
                component_button.config(
                    state=(
                        tk.NORMAL
                        if is_external and not self.ai_request_running
                        else tk.DISABLED
                    )
                )
        if hasattr(self, "settings_test_button"):
            self.settings_test_button.config(
                state=(
                    tk.NORMAL
                    if is_external and not self.ai_connection_testing
                    else tk.DISABLED
                )
            )
        if hasattr(self, "settings_disconnect_button"):
            self.settings_disconnect_button.config(
                state=tk.NORMAL if is_external else tk.DISABLED
            )
        if hasattr(self, "show_key_checkbutton"):
            self.show_key_checkbutton.config(
                state=tk.NORMAL if is_external else tk.DISABLED
            )
        if hasattr(self, "remember_key_checkbutton"):
            self.remember_key_checkbutton.config(
                state=(
                    tk.NORMAL
                    if is_external and keyring is not None
                    else tk.DISABLED
                )
            )

        if hasattr(self, "ai_status_label"):
            if is_external:
                self.ai_status_label.config(
                    text=self.ai_connection_message
                )
            else:
                self.ai_status_label.config(
                    text="Local recommendation remains available offline."
                )
        self.refresh_settings_status()

    def start_ai_recommendation(self):
        if self.ai_request_running or self.training_running:
            return
        try:
            if self.raw_df is None:
                raise ValueError("Please load a dataset first.")
            if self.dataset_profile is None:
                self.run_dataset_analysis(show_message=False)

            provider_name = self.ai_provider_var.get()
            if provider_name not in OFFICIAL_PROVIDER_SETTINGS:
                raise ValueError(
                    "Choose OpenAI, DeepSeek, Claude, or an OpenAI-compatible API before "
                    "requesting AI analysis."
                )

            defaults = OFFICIAL_PROVIDER_SETTINGS[provider_name]
            api_key = self.resolve_ai_api_key(provider_name)
            if not api_key:
                raise ValueError(
                    "Connect the provider in Settings or set "
                    f"{defaults['api_key_environment']} before continuing."
                )

            model_name = self.ai_model_var.get().strip()
            base_url = _validate_provider_base_url(
                self.ai_base_url_var.get()
            )
            timeout_seconds = float(self.ai_timeout_var.get())
            user_goal = self.ai_goal_var.get().strip()

            official_host = urlparse(defaults["base_url"]).hostname
            selected_host = urlparse(base_url).hostname
            if selected_host != official_host:
                approved = messagebox.askyesno(
                    "Confirm Custom Provider Host",
                    (
                        f"The selected Base URL is not the official "
                        f"{provider_name} host.\n\n"
                        f"Selected host: {selected_host}\n"
                        f"Official host: {official_host}\n\n"
                        "Your API key and compact dataset profile will be sent "
                        "to the selected host. Continue only if you trust it."
                    ),
                )
                if not approved:
                    return

            provider = create_ai_provider(
                provider_name=provider_name,
                api_key=api_key,
                model=model_name,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
            profile_snapshot = json.loads(json.dumps(self.dataset_profile))
            profile_snapshot["training_context"] = {
                "input_columns": list(self.feature_cols), "target_columns": list(self.target_cols),
                "forecast_horizon": int(self.forecast_horizon_var.get() or 1) if self.task_type_var.get() == TASK_FORECASTING else 1,
                "test_size_percent": float(self.test_size_var.get()),
                "data_order": self.guided_data_order_var.get(),
            }
            profile_snapshot = make_ai_request_profile(profile_snapshot)
            if len(user_goal) > AI_GOAL_MAX_CHARS:
                raise ValueError(f"Please shorten the goal/notes to {AI_GOAL_MAX_CHARS} characters.")
            self.ai_request_size_summary = recommendation_request_summary(profile_snapshot)

            self.ai_request_running = True
            self.ai_request_button.config(state=tk.DISABLED)
            self.ai_apply_button.config(state=tk.DISABLED)
            self.ai_generate_button.config(state=tk.DISABLED)
            self.ai_apply_filter_button.config(state=tk.DISABLED)
            self.ai_apply_model_button.config(state=tk.DISABLED)
            self.back_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
            self.ai_status_label.config(
                text="Requesting and validating recommendation..."
            )
            self.ai_analysis_text.delete("1.0", tk.END)
            self.ai_analysis_text.insert(
                tk.END,
                self.ai_request_size_summary + "\nRequesting structured advice; one correction attempt is allowed."
            )

            thread = threading.Thread(
                target=self.ai_recommendation_worker,
                args=(
                    provider,
                    profile_snapshot,
                    user_goal,
                    provider_name,
                    model_name,
                    base_url,
                ),
                daemon=True,
            )
            thread.start()

        except Exception as exc:
            messagebox.showerror("AI Provider Settings", str(exc))

    def ai_recommendation_worker(
        self,
        provider,
        profile_snapshot,
        user_goal,
        provider_name,
        model_name,
        base_url,
    ):
        try:
            recommendation = provider.recommend(
                profile_snapshot,
                user_goal=user_goal,
            )
            report = format_ai_recommendation_report(
                recommendation,
                provider_name,
                model_name,
            )
            report = getattr(provider, "last_request_summary", "") + "\n\n" + report
            diagnostics = getattr(provider, "recommendation_diagnostics", [])
            if diagnostics:
                report += "\n\nCORRECTION ATTEMPT\n" + "\n".join(diagnostics)
            self.ui_queue.put((
                "ai_finished",
                {
                    "recommendation": recommendation,
                    "report": report,
                    "provider_summary": {
                        "provider": provider_name,
                        "model": model_name,
                        "base_url": base_url,
                        "profile_only": True,
                        "generated_code": False,
                    },
                },
            ))
        except Exception as exc:
            self.ui_queue.put(("ai_failed", str(exc)))

    def ai_recommendation_finished(self, payload):
        self.ai_request_running = False
        provider_name = payload["provider_summary"]["provider"]
        self.ai_connection_state = "connected"
        self.ai_connection_message = (
            f"{provider_name} connected successfully."
        )
        self.ai_last_connection_check = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.ai_last_connection_details = (
            "An AI recommendation request completed successfully."
        )
        self.update_provider_status_display()
        self.ai_recommendation = payload["recommendation"]
        self.ai_recommendation_report = payload["report"]
        self.ai_provider_summary = payload["provider_summary"]
        self.ai_generation = None
        self.ai_generation_report = ""
        self.ai_generation_provider_summary = None
        self.safe_filter_spec = None
        self.safe_model_spec = None
        self.safe_filter_generation = None
        self.safe_filter_recommendation = None
        self.safe_model_generation = None
        self.safe_model_recommendation = None
        self.custom_filter_ai_report = ""
        self.custom_model_ai_report = ""

        if hasattr(self, "ai_analysis_text"):
            self.ai_analysis_text.delete("1.0", tk.END)
            self.ai_analysis_text.insert(
                tk.END,
                self.ai_recommendation_report,
            )
            self.ai_analysis_text.see("1.0")
        if hasattr(self, "ai_apply_button"):
            self.ai_apply_button.config(state=tk.NORMAL)
        if hasattr(self, "ai_generate_button"):
            self.ai_generate_button.config(state=tk.NORMAL)
        if hasattr(self, "ai_status_label"):
            self.ai_status_label.config(
                text="Structured recommendation validated locally."
            )
        self.back_button.config(state=tk.NORMAL)
        self.next_button.config(state=tk.NORMAL)
        self.on_ai_provider_changed(update_existing=False)

    def ai_recommendation_failed(self, error_message):
        self.ai_request_running = False
        if hasattr(self, "ai_analysis_text"):
            self.ai_analysis_text.delete("1.0", tk.END)
            self.ai_analysis_text.insert(
                tk.END,
                "AI recommendation failed.\n\n" + error_message,
            )
        if hasattr(self, "ai_status_label"):
            self.ai_status_label.config(text="AI advice unavailable. Your data is unchanged; local advice is still available.")
        self.back_button.config(state=tk.NORMAL)
        self.next_button.config(state=tk.NORMAL)
        if (
            hasattr(self, "ai_apply_button")
            and self.ai_recommendation is not None
        ):
            self.ai_apply_button.config(state=tk.NORMAL)
        self.on_ai_provider_changed(update_existing=False)
        messagebox.showerror(
            "AI Recommendation Error",
            error_message,
        )

    def start_ai_generation(self):
        try:
            if self.ai_recommendation is None:
                raise ValueError(
                    "Request and validate an AI recommendation first."
                )
            if self.dataset_profile is None:
                self.run_dataset_analysis(show_message=False)

            provider_name = self.ai_provider_var.get()
            if provider_name not in OFFICIAL_PROVIDER_SETTINGS:
                raise ValueError(
                    "Choose OpenAI, DeepSeek, Claude, or an OpenAI-compatible API before "
                    "requesting generation."
                )
            defaults = OFFICIAL_PROVIDER_SETTINGS[provider_name]
            api_key = self.resolve_ai_api_key(provider_name)
            if not api_key:
                raise ValueError(
                    "Connect the provider in Settings or set "
                    f"{defaults['api_key_environment']} before continuing."
                )
            model_name = self.ai_model_var.get().strip()
            base_url = _validate_provider_base_url(
                self.ai_base_url_var.get()
            )
            timeout_seconds = float(self.ai_timeout_var.get())
            user_goal = self.ai_goal_var.get().strip()

            official_host = urlparse(defaults["base_url"]).hostname
            selected_host = urlparse(base_url).hostname
            if selected_host != official_host:
                approved = messagebox.askyesno(
                    "Confirm Custom Provider Host",
                    (
                        f"The selected Base URL is not the official "
                        f"{provider_name} host.\n\n"
                        f"Selected host: {selected_host}\n"
                        f"Official host: {official_host}\n\n"
                        "Your API key, compact dataset profile, and validated "
                        "recommendation will be sent to that host. Continue?"
                    ),
                )
                if not approved:
                    return

            provider = create_ai_provider(
                provider_name=provider_name,
                api_key=api_key,
                model=model_name,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
            profile_snapshot = json.loads(json.dumps(self.dataset_profile))
            recommendation_snapshot = json.loads(json.dumps(
                self.ai_recommendation
            ))

            self.ai_request_running = True
            self.ai_request_button.config(state=tk.DISABLED)
            self.ai_apply_button.config(state=tk.DISABLED)
            self.ai_generate_button.config(state=tk.DISABLED)
            self.ai_apply_filter_button.config(state=tk.DISABLED)
            self.ai_apply_model_button.config(state=tk.DISABLED)
            self.back_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
            self.ai_status_label.config(
                text="Generating and validating safe specifications..."
            )
            self.ai_generation_text.delete("1.0", tk.END)
            self.ai_generation_text.insert(
                tk.END,
                "Generating declarative specifications. No Python code or raw "
                "CSV rows are requested..."
            )

            threading.Thread(
                target=self.ai_generation_worker,
                args=(
                    provider,
                    profile_snapshot,
                    recommendation_snapshot,
                    user_goal,
                    provider_name,
                    model_name,
                    base_url,
                ),
                daemon=True,
            ).start()
        except Exception as exc:
            messagebox.showerror("AI Generation Settings", str(exc))

    def ai_generation_worker(
        self,
        provider,
        profile_snapshot,
        recommendation_snapshot,
        user_goal,
        provider_name,
        model_name,
        base_url,
    ):
        try:
            generation = provider.generate(
                profile_snapshot,
                recommendation_snapshot,
                user_goal=user_goal,
            )
            report = format_ai_generation_report(
                generation,
                provider_name,
                model_name,
            )
            self.ui_queue.put((
                "ai_generation_finished",
                {
                    "generation": generation,
                    "report": report,
                    "provider_summary": {
                        "provider": provider_name,
                        "model": model_name,
                        "base_url": base_url,
                        "profile_only": True,
                        "generated_python": False,
                        "declarative_safe_builders": True,
                    },
                },
            ))
        except Exception as exc:
            self.ui_queue.put(("ai_generation_failed", str(exc)))

    def ai_generation_finished(self, payload):
        self.ai_request_running = False
        provider_name = payload["provider_summary"]["provider"]
        self.ai_connection_state = "connected"
        self.ai_connection_message = (
            f"{provider_name} connected successfully."
        )
        self.ai_last_connection_check = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.ai_last_connection_details = (
            "A safe-component generation request completed successfully."
        )
        self.update_provider_status_display()
        self.ai_generation = payload["generation"]
        self.ai_generation_report = payload["report"]
        self.ai_generation_provider_summary = payload["provider_summary"]
        if hasattr(self, "ai_generation_text"):
            self.ai_generation_text.delete("1.0", tk.END)
            self.ai_generation_text.insert(
                tk.END,
                self.ai_generation_report,
            )
            self.ai_generation_text.see("1.0")
        self.ai_apply_filter_button.config(state=tk.NORMAL)
        self.ai_apply_model_button.config(state=tk.NORMAL)
        self.ai_apply_button.config(state=tk.NORMAL)
        self.ai_status_label.config(
            text="Safe specifications validated; approval is still required."
        )
        self.back_button.config(state=tk.NORMAL)
        self.next_button.config(state=tk.NORMAL)
        self.on_ai_provider_changed(update_existing=False)

    def ai_generation_failed(self, error_message):
        self.ai_request_running = False
        if hasattr(self, "ai_generation_text"):
            self.ai_generation_text.delete("1.0", tk.END)
            self.ai_generation_text.insert(
                tk.END,
                "AI safe generation failed.\n\n" + error_message,
            )
        self.ai_status_label.config(text="Safe generation failed.")
        if self.ai_recommendation is not None:
            self.ai_apply_button.config(state=tk.NORMAL)
        self.back_button.config(state=tk.NORMAL)
        self.next_button.config(state=tk.NORMAL)
        self.on_ai_provider_changed(update_existing=False)
        messagebox.showerror("AI Generation Error", error_message)

    def build_current_generation_recommendation(
        self,
        input_columns=None,
        component="model",
    ):
        """Build generation context from current user choices, not recommendations."""
        if self.dataset_profile is None:
            self.run_dataset_analysis(show_message=False)

        selected_inputs = list(input_columns or self.feature_cols)
        if not selected_inputs:
            selected_inputs = [
                column for column in self.dataset_profile["numeric_columns"]
                if column not in set(self.target_cols)
            ]
        selected_inputs = list(dict.fromkeys(selected_inputs))

        # A filter can be designed before the task/target page. Use an
        # autoencoder-shaped validation context because it needs only numeric
        # inputs and does not impose an artificial target recommendation.
        if component == "filter":
            task_type = TASK_AUTOENCODER
            selected_targets = []
        else:
            task_type = self.task_type_var.get()
            selected_targets = list(self.target_cols)

        if task_type == TASK_AUTOENCODER:
            selected_targets = []
        elif task_type in (TASK_CLASSIFICATION, TASK_REGRESSION):
            selected_targets = selected_targets[:1]
        elif task_type in (TASK_MULTI_OUTPUT, TASK_FORECASTING):
            if len(selected_targets) < (
                2 if task_type == TASK_MULTI_OUTPUT else 1
            ):
                raise ValueError(
                    f"{task_type} requires more confirmed target columns "
                    "before AI model generation."
                )

        if task_type in (TASK_CLASSIFICATION, TASK_REGRESSION) and not selected_targets:
            raise ValueError(
                f"{task_type} requires one confirmed target column before "
                "AI component generation."
            )

        target_set = set(selected_targets)
        selected_inputs = [
            column for column in selected_inputs if column not in target_set
        ]
        if not selected_inputs:
            raise ValueError("Select at least one numeric input column.")

        task = {
            "task_type": task_type,
            "confidence": 1.0,
            "input_columns": selected_inputs,
            "target_columns": selected_targets,
            "reasons": [
                "Built from the user's current task and column selections."
            ],
            "warnings": [],
            "alternatives": [
                candidate for candidate in TASK_TYPES
                if candidate != task_type
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
                "The user controls the custom filter design."
            ],
            "risks": [
                "Filtering must be checked for signal distortion."
            ],
            "validation_checks": [
                "Compare correlation, RMSE, variance, and spectral preservation."
            ],
        }

        input_mode = self.custom_model_input_mode_var.get()
        if task_type == TASK_CLASSIFICATION:
            model_type = (
                "DNN" if input_mode == "Row-based (2D)" else "CNN-LSTM"
            )
            output_activation = "softmax"
            loss_function = "sparse_categorical_crossentropy"
        elif task_type == TASK_AUTOENCODER:
            model_type = (
                "Dense Autoencoder"
                if input_mode == "Row-based (2D)"
                else "LSTM Autoencoder"
            )
            output_activation = "linear"
            loss_function = "mean_squared_error"
        elif task_type == TASK_FORECASTING:
            model_type = "LSTM"
            output_activation = "linear"
            loss_function = "mean_squared_error"
        else:
            model_type = (
                "DNN" if input_mode == "Row-based (2D)" else "CNN-LSTM"
            )
            output_activation = "linear"
            loss_function = "mean_squared_error"

        model_plan = {
            "model_type": model_type,
            "window_size": max(2, int(self.window_size_var.get())),
            "stride": max(1, int(self.stride_var.get())),
            "epochs": max(1, int(self.epochs_var.get())),
            "batch_size": max(1, int(self.batch_size_var.get())),
            "validation_split_percent": min(
                79.0,
                max(1.0, float(self.validation_split_var.get())),
            ),
            "hidden_activation": self.hidden_activation_var.get(),
            "dropout_rate": min(
                0.95,
                max(0.0, float(self.dropout_var.get())),
            ),
            "output_activation": output_activation,
            "loss_function": loss_function,
            "optimizer": self.optimizer_var.get(),
            "learning_rate": float(self.lr_var.get()),
            "feature_scaling": self.scaler_var.get(),
            "target_scaling": self.target_scaler_var.get(),
            "confidence": 1.0,
            "reasons": [
                "The component uses the current user-defined settings."
            ],
            "warnings": [],
        }
        recommendation = {
            "summary": (
                "Direct customization context built from current GUI choices; "
                "no analysis recommendation was used."
            ),
            "task": task,
            "filter": filter_plan,
            "model": model_plan,
            "future_generation": {
            "custom_filter_would_help": component == "filter",
            "custom_filter_reason": (
                "The filter is controlled directly from the customization page."
            ),
            "custom_model_would_help": component == "model",
            "custom_model_reason": (
                "The model is controlled directly from the customization page."
            ),
            },
        }
        return validate_ai_recommendation(
            recommendation,
            self.dataset_profile,
        )

    def get_component_ai_provider(self):
        provider_name = self.ai_provider_var.get()
        if provider_name not in OFFICIAL_PROVIDER_SETTINGS:
            raise ValueError(
                "Choose OpenAI, DeepSeek, Claude, or an OpenAI-compatible API in the "
                "provider settings before "
                "using an AI customization assistant."
            )
        defaults = OFFICIAL_PROVIDER_SETTINGS[provider_name]
        api_key = self.resolve_ai_api_key(provider_name)
        if not api_key:
            raise ValueError(
                "Connect the provider in Settings or set "
                f"{defaults['api_key_environment']}."
            )
        model_name = self.ai_model_var.get().strip()
        base_url = _validate_provider_base_url(self.ai_base_url_var.get())
        timeout_seconds = float(self.ai_timeout_var.get())

        official_host = urlparse(defaults["base_url"]).hostname
        selected_host = urlparse(base_url).hostname
        if selected_host != official_host:
            approved = messagebox.askyesno(
                "Confirm Custom Provider Host",
                (
                    f"The selected Base URL is not the official "
                    f"{provider_name} host.\n\n"
                    f"Selected host: {selected_host}\n"
                    f"Official host: {official_host}\n\n"
                    "Your API key, compact dataset profile, current settings, "
                    "and customization request will be sent to that host. "
                    "Continue only if you trust it."
                ),
            )
            if not approved:
                return None

        provider = create_ai_provider(
            provider_name=provider_name,
            api_key=api_key,
            model=model_name,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
        )
        return provider, provider_name, model_name, base_url

    def start_component_ai_generation(
        self,
        component,
        customization_prompt,
        input_columns=None,
    ):
        try:
            if self.ai_request_running:
                raise ValueError(
                    "Wait for the current AI request to finish."
                )
            provider_result = self.get_component_ai_provider()
            if provider_result is None:
                return
            provider, provider_name, model_name, base_url = provider_result
            recommendation = self.build_current_generation_recommendation(
                input_columns=input_columns,
                component=component,
            )
            prompt = customization_prompt.strip()
            if not prompt:
                raise ValueError(
                    "Describe what you want the AI to generate before continuing."
                )

            if component == "filter":
                focus = (
                    "Generate a customized FILTER for only these confirmed "
                    f"columns: {recommendation['task']['input_columns']}. "
                    "The filter_spec is the requested deliverable. Return a "
                    "minimal compatible model_spec only because the shared "
                    "schema requires it. "
                )
                status_widget = getattr(
                    self, "custom_filter_ai_status_label", None
                )
                result_widget = getattr(
                    self, "custom_filter_preview_text", None
                )
            elif component == "model":
                focus = (
                    "Generate a customized NEURAL-NETWORK MODEL for the "
                    "confirmed task, inputs, targets, and current input mode. "
                    "The model_spec is the requested deliverable. Return an "
                    "empty filter pipeline only because the shared schema "
                    "requires a filter_spec. "
                )
                status_widget = getattr(
                    self, "custom_model_ai_status_label", None
                )
                result_widget = getattr(
                    self, "custom_model_preview_text", None
                )
            else:
                raise ValueError(f"Unsupported AI component: {component}")

            complete_prompt = (
                focus
                + "\nUSER CUSTOMIZATION REQUEST:\n"
                + prompt
            )
            profile_snapshot = json.loads(json.dumps(self.dataset_profile))
            recommendation_snapshot = json.loads(json.dumps(recommendation))
            self.ai_request_running = True
            self.back_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
            active_button = getattr(
                self,
                (
                    "custom_filter_ai_button"
                    if component == "filter"
                    else "custom_model_ai_button"
                ),
                None,
            )
            if active_button is not None:
                active_button.config(state=tk.DISABLED)
            if status_widget is not None:
                status_widget.config(
                    text="Generating and validating safe specification..."
                )
            if result_widget is not None:
                result_widget.delete("1.0", tk.END)
                result_widget.insert(
                    tk.END,
                    "The AI is generating a declarative component. "
                    "No Python code or raw CSV rows are being sent.",
                )

            threading.Thread(
                target=self.component_ai_generation_worker,
                args=(
                    component,
                    provider,
                    profile_snapshot,
                    recommendation_snapshot,
                    complete_prompt,
                    provider_name,
                    model_name,
                    base_url,
                ),
                daemon=True,
            ).start()
        except Exception as exc:
            messagebox.showerror("AI Customization Settings", str(exc))

    def component_ai_generation_worker(
        self,
        component,
        provider,
        profile_snapshot,
        recommendation_snapshot,
        customization_prompt,
        provider_name,
        model_name,
        base_url,
    ):
        try:
            generation = provider.generate(
                profile_snapshot,
                recommendation_snapshot,
                user_goal=customization_prompt,
            )
            report = format_ai_generation_report(
                generation,
                provider_name,
                model_name,
            )
            self.ui_queue.put((
                "component_ai_generation_finished",
                {
                    "component": component,
                    "generation": generation,
                    "recommendation": recommendation_snapshot,
                    "report": report,
                    "provider_summary": {
                        "provider": provider_name,
                        "model": model_name,
                        "base_url": base_url,
                        "profile_only": True,
                        "generated_python": False,
                        "declarative_safe_builders": True,
                        "component": component,
                    },
                },
            ))
        except Exception as exc:
            self.ui_queue.put((
                "component_ai_generation_failed",
                {"component": component, "error": str(exc)},
            ))

    def component_ai_generation_finished(self, payload):
        self.ai_request_running = False
        provider_name = payload["provider_summary"]["provider"]
        self.ai_connection_state = "connected"
        self.ai_connection_message = (
            f"{provider_name} connected successfully."
        )
        self.ai_last_connection_check = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.ai_last_connection_details = (
            "A component-generation request completed successfully."
        )
        self.update_provider_status_display()
        component = payload["component"]
        generation = payload["generation"]
        recommendation = payload["recommendation"]
        generation = validate_ai_generation(
            json.loads(json.dumps(generation)),
            self.dataset_profile,
            recommendation,
        )
        self.ai_generation = generation
        self.ai_generation_report = payload["report"]
        self.ai_generation_provider_summary = payload["provider_summary"]

        if component == "filter":
            self.safe_filter_spec = generation["filter_spec"]
            self.safe_filter_generation = generation
            self.safe_filter_recommendation = recommendation
            self.selected_custom_filter_cols = [
                pipeline["column"]
                for pipeline in self.safe_filter_spec["pipelines"]
            ]
            self.custom_filter_mode_var.set(CUSTOM_FILTER_MODE_SAFE_AI)
            self.custom_filter_enabled_var.set(bool(
                self.safe_filter_spec["pipelines"]
            ))
            self.custom_filter_ai_report = payload["report"]
            if hasattr(self, "custom_filter_code_text"):
                self.on_custom_filter_mode_changed()
            if hasattr(self, "custom_filter_ai_status_label"):
                self.custom_filter_ai_status_label.config(
                    text=(
                        "Safe filter generated and selected. Review the "
                        "preservation metrics before continuing."
                    )
                )
            if self.safe_filter_spec["pipelines"]:
                self.preview_custom_filter_result(show_message=False)
            else:
                self.skip_custom_filter()
            messagebox.showinfo(
                "AI Filter Generated",
                "The customized safe filter is ready. Review its specification "
                "and preview before continuing.",
            )
        else:
            self.safe_model_spec = generation["model_spec"]
            self.safe_model_generation = generation
            self.safe_model_recommendation = recommendation
            self.custom_model_input_mode_var.set(
                self.safe_model_spec["input_mode"]
            )
            self.model_type_var.set(MODEL_TYPE_SAFE_CUSTOM)
            self.custom_model_ai_report = payload["report"]
            if hasattr(self, "safe_model_summary_text"):
                self.safe_model_summary_text.delete("1.0", tk.END)
                self.safe_model_summary_text.insert(
                    tk.END,
                    json.dumps(self.safe_model_spec, indent=2),
                )
            if hasattr(self, "custom_model_ai_status_label"):
                self.custom_model_ai_status_label.config(
                    text=(
                        "Safe model generated and selected. Validate the model "
                        "summary before training."
                    )
                )
            self.validate_safe_model_preview()

        self.back_button.config(state=tk.NORMAL)
        self.next_button.config(state=tk.NORMAL)
        active_button = getattr(
            self,
            (
                "custom_filter_ai_button"
                if component == "filter"
                else "custom_model_ai_button"
            ),
            None,
        )
        if active_button is not None:
            active_button.config(state=tk.NORMAL)

    def component_ai_generation_failed(self, payload):
        self.ai_request_running = False
        component = payload["component"]
        error_message = payload["error"]
        status_widget = getattr(
            self,
            (
                "custom_filter_ai_status_label"
                if component == "filter"
                else "custom_model_ai_status_label"
            ),
            None,
        )
        if status_widget is not None:
            status_widget.config(text="AI customization failed.")
        self.back_button.config(state=tk.NORMAL)
        self.next_button.config(state=tk.NORMAL)
        active_button = getattr(
            self,
            (
                "custom_filter_ai_button"
                if component == "filter"
                else "custom_model_ai_button"
            ),
            None,
        )
        if active_button is not None:
            active_button.config(state=tk.NORMAL)
        messagebox.showerror("AI Customization Error", error_message)

    def apply_ai_generated_filter(self):
        if self.ai_generation is None:
            messagebox.showwarning(
                "No Safe Filter",
                "Generate safe AI specifications first.",
            )
            return
        try:
            generation = validate_ai_generation(
                json.loads(json.dumps(self.ai_generation)),
                self.dataset_profile,
                self.ai_recommendation,
            )
            filter_spec = generation["filter_spec"]
            if filter_spec["pipelines"]:
                smoke_rows = min(len(self.raw_df), 5000)
                apply_safe_filter_spec(
                    df=self.raw_df.iloc[:smoke_rows].copy(),
                    filter_spec=filter_spec,
                    sampling_frequency=self.dataset_profile.get(
                        "sampling_frequency_hz"
                    ),
                    allowed_columns=self.ai_recommendation[
                        "task"
                    ]["input_columns"],
                )
            self.safe_filter_spec = filter_spec
            self.safe_filter_generation = generation
            self.safe_filter_recommendation = json.loads(json.dumps(
                self.ai_recommendation
            ))
            self.selected_custom_filter_cols = [
                pipeline["column"]
                for pipeline in filter_spec["pipelines"]
            ]
            self.custom_filter_mode_var.set(CUSTOM_FILTER_MODE_SAFE_AI)
            self.custom_filter_enabled_var.set(bool(
                filter_spec["pipelines"]
            ))
            messagebox.showinfo(
                "Safe Filter Approved",
                (
                    "The declarative filter specification is ready for Step 4. "
                    "It will be previewed and compared before it is applied."
                    if filter_spec["pipelines"]
                    else "The generated specification recommends no custom "
                    "filter. The custom filter step remains disabled."
                ),
            )
        except Exception as exc:
            messagebox.showerror("Safe Filter Error", str(exc))

    def apply_ai_generated_model(self):
        if self.ai_generation is None:
            messagebox.showwarning(
                "No Safe Model",
                "Generate safe AI specifications first.",
            )
            return
        try:
            generation = validate_ai_generation(
                json.loads(json.dumps(self.ai_generation)),
                self.dataset_profile,
                self.ai_recommendation,
            )
            task = self.ai_recommendation["task"]
            self.task_type_var.set(task["task_type"])
            self.feature_cols = list(task["input_columns"])
            self.target_cols = list(task["target_columns"])
            self.label_col = (
                self.target_cols[0] if self.target_cols else ""
            )
            self.label_var.set(self.label_col)
            self.apply_task_defaults()

            self.safe_model_spec = generation["model_spec"]
            self.safe_model_generation = generation
            self.safe_model_recommendation = json.loads(json.dumps(
                self.ai_recommendation
            ))
            self.custom_model_input_mode_var.set(
                self.safe_model_spec["input_mode"]
            )
            model_plan = self.ai_recommendation["model"]
            window_size = int(model_plan["window_size"])
            feature_count = max(1, len(self.feature_cols))
            input_shape = (
                (window_size, feature_count)
                if self.safe_model_spec["input_mode"] == "Window-based (3D)"
                else (feature_count,)
            )
            if self.task_type_var.get() == TASK_CLASSIFICATION:
                output_units = int(
                    self.raw_df[self.label_col].dropna().nunique()
                )
            elif self.task_type_var.get() == TASK_FORECASTING:
                output_units = len(self.target_cols) * int(
                    self.forecast_horizon_var.get()
                )
            elif self.task_type_var.get() == TASK_AUTOENCODER:
                output_units = feature_count
            else:
                output_units = len(self.target_cols)
            build_safe_model_from_spec(
                model_spec=self.safe_model_spec,
                input_shape=input_shape,
                output_units=output_units,
                output_activation=model_plan["output_activation"],
                task_type=self.task_type_var.get(),
            )
            self.model_type_var.set(MODEL_TYPE_SAFE_CUSTOM)
            messagebox.showinfo(
                "Safe Model Approved",
                "The allowlisted model architecture is selected for Step 7. "
                "The program will build it locally and validate its shapes "
                "before training.",
            )
        except Exception as exc:
            messagebox.showerror("Safe Model Error", str(exc))

    def apply_ai_recommendation(self):
        if self.ai_recommendation is None:
            messagebox.showwarning(
                "No AI Recommendation",
                "Request and validate an AI recommendation first.",
            )
            return

        try:
            recommendation = validate_ai_recommendation(
                json.loads(json.dumps(self.ai_recommendation)),
                self.dataset_profile,
            )
            task = recommendation["task"]
            filter_plan = recommendation["filter"]
            model = recommendation["model"]
            available_columns = set(self.raw_df.columns)

            self.task_type_var.set(task["task_type"])
            self.feature_cols = [
                column for column in task["input_columns"]
                if column in available_columns
            ]
            self.target_cols = [
                column for column in task["target_columns"]
                if column in available_columns
            ]
            self.label_col = (
                self.target_cols[0] if self.target_cols else ""
            )
            self.label_var.set(self.label_col)

            # Apply deterministic task defaults first, then the validated
            # provider settings. This prevents stale settings from a prior task.
            self.apply_task_defaults()
            self.model_type_var.set(model["model_type"])
            self.window_size_var.set(str(model["window_size"]))
            self.stride_var.set(str(model["stride"]))
            self.epochs_var.set(str(model["epochs"]))
            self.batch_size_var.set(str(model["batch_size"]))
            self.validation_split_var.set(
                str(model["validation_split_percent"])
            )
            self.hidden_activation_var.set(model["hidden_activation"])
            self.dropout_var.set(str(model["dropout_rate"]))
            self.output_activation_var.set(model["output_activation"])
            self.loss_var.set(model["loss_function"])
            self.optimizer_var.set(model["optimizer"])
            self.lr_var.set(str(model["learning_rate"]))
            self.scaler_var.set(model["feature_scaling"])
            self.target_scaler_var.set(model["target_scaling"])

            if (
                task["task_type"] == TASK_CLASSIFICATION
                and model["loss_function"] == "binary_crossentropy"
            ):
                self.output_mode_var.set("Custom")
                self.output_units_var.set("1")
            else:
                self.output_mode_var.set("Auto")
                if task["task_type"] == TASK_CLASSIFICATION:
                    output_units = self.raw_df[
                        self.target_cols[0]
                    ].dropna().nunique()
                elif task["task_type"] == TASK_FORECASTING:
                    horizon = max(
                        1,
                        int(self.forecast_horizon_var.get()),
                    )
                    output_units = len(self.target_cols) * horizon
                elif task["task_type"] == TASK_AUTOENCODER:
                    output_units = len(self.feature_cols)
                else:
                    output_units = len(self.target_cols)
                self.output_units_var.set(str(max(1, output_units)))

            self.filter_method_var.set(filter_plan["method"])
            self.selected_filter_cols = list(filter_plan["columns"])
            self.moving_window_var.set(
                str(filter_plan["moving_average_window"])
            )
            self.ema_span_var.set(str(filter_plan["ema_span"]))
            self.median_window_var.set(
                str(filter_plan["median_window"])
            )
            self.kalman_q_var.set(str(filter_plan["kalman_q"]))
            self.kalman_r_var.set(str(filter_plan["kalman_r"]))

            # AI advice updates controls; explicit Apply in the filter page updates data.
            self._pending_preparation_steps.add(2)
            if self.custom_filter_enabled_var.get():
                self._pending_preparation_steps.add(3)
            if hasattr(self, "ai_analysis_text"):
                self.ai_analysis_text.insert(
                    tk.END,
                    "\n\nAPPLIED TO PROGRAM CONTROLS\n"
                    f"Task: {task['task_type']}\n"
                    f"Inputs: {self.feature_cols}\n"
                    f"Targets: {self.target_cols or 'None'}\n"
                    f"Built-in filter: {filter_plan['method']} on "
                    f"{self.selected_filter_cols or 'no columns'}\n"
                    f"Model: {model['model_type']}\n\n"
                    "Review the filter preview, task columns, and model "
                    "settings on the following pages before training."
                )
                self.ai_analysis_text.see(tk.END)

            messagebox.showinfo(
                "AI Recommendation Applied",
                "The validated task, columns, built-in filter, and model "
                "settings were applied. Review each page before training.",
            )
        except Exception as exc:
            messagebox.showerror(
                "Apply AI Recommendation Error",
                str(exc),
            )

    def apply_dataset_recommendation(self):
        if self.dataset_profile is None:
            self.run_dataset_analysis(show_message=False)

        recommendation = self.dataset_profile["recommendation"]
        task_type = recommendation["task_type"]
        available_columns = set(self.raw_df.columns)

        self.task_type_var.set(task_type)
        self.feature_cols = [
            column
            for column in recommendation["input_columns"]
            if column in available_columns
        ]
        self.target_cols = [
            column
            for column in recommendation["target_columns"]
            if column in available_columns
        ]
        self.label_col = self.target_cols[0] if self.target_cols else ""
        self.label_var.set(self.label_col)
        self.apply_task_defaults()

        if task_type == TASK_CLASSIFICATION and self.target_cols:
            output_units = self.raw_df[
                self.target_cols[0]
            ].dropna().nunique()
        elif task_type == TASK_FORECASTING:
            horizon = max(1, int(self.forecast_horizon_var.get()))
            output_units = max(1, len(self.target_cols) * horizon)
        elif task_type == TASK_AUTOENCODER:
            output_units = max(1, len(self.feature_cols))
        else:
            output_units = max(1, len(self.target_cols))
        self.output_units_var.set(str(output_units))

        if hasattr(self, "analysis_text"):
            self.analysis_text.insert(
                tk.END,
                "\n\nAPPLIED TO PROGRAM SETTINGS\n"
                f"Task: {task_type}\n"
                f"Inputs: {self.feature_cols}\n"
                f"Targets: {self.target_cols or 'None'}\n"
                "You can revise every selection on the task/column page."
            )
            self.analysis_text.see(tk.END)

        messagebox.showinfo(
            "Recommendation Applied",
            "The suggested task and columns were applied. "
            "You can review and change them before training."
        )

    def validate_step_ai_analysis(self):
        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            return
        if self.ai_request_running:
            raise ValueError(
                "Wait for the AI recommendation request to finish."
            )
        if self.dataset_profile is None:
            self.run_dataset_analysis(show_message=False)
