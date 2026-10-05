"""Ui / wizard projects settings for NN Training Studio."""

from datetime import datetime
import json
try:
    import keyring
except Exception:
    keyring = None
from tkinter import messagebox
import os
import platform
import subprocess
import threading
import tkinter as tk
from tkinter import ttk
from urllib.parse import urlparse
from nn_training_studio.ai_transport import (
    _validate_provider_base_url,
    test_ai_provider_connection,
)
from nn_training_studio.checkpoints import (
    _training_checkpoint_root,
)
from nn_training_studio.constants import (
    AI_PROVIDER_COMPATIBLE,
    AI_PROVIDER_TYPES,
    BEST_MODEL_FILENAME,
    BEST_STATE_FILENAME,
    OFFICIAL_PROVIDER_SETTINGS,
    RECOVERY_DIRECTORY_NAME,
    WORKSPACE_FILTER,
)
from nn_training_studio.detection import (
    _object_detection_run_root,
)


class ProjectsSettingsMixin:
    """ProjectsSettings behavior for the main application."""

    def show_projects(self):
        if self.training_running or self.ai_request_running:
            messagebox.showinfo(
                "Operation Running",
                "Wait for the active operation to finish before opening "
                "projects and checkpoints.",
            )
            return
        self.current_view = "projects"
        self.clear_content()
        self.title_label.config(text="Projects & Checkpoints")
        self.progress_label.config(
            text="Best-model checkpoints and interrupted-run recovery states"
        )
        self.set_workflow_navigation_visible(False)
        self.home_button.config(state=tk.NORMAL)
        self.settings_button.config(state=tk.NORMAL)
        self.build_projects_page()

    def build_projects_page(self):
        info = ttk.LabelFrame(
            self.content_frame,
            text="Training Checkpoint Store",
        )
        info.pack(fill=tk.X, padx=12, pady=(12, 6))
        ttk.Label(
            info,
            text=(
                "Each training configuration receives a stable run ID. "
                "Starting the same project and settings again automatically "
                "offers recovery through Keras BackupAndRestore. YOLO object-"
                "detection runs preserve best.pt, last.pt, metrics, and plots "
                "in a separate detection run store."
            ),
            wraplength=1050,
        ).pack(anchor="w", padx=12, pady=10)

        table_frame = ttk.Frame(self.content_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)
        columns = (
            "run_id",
            "best_epoch",
            "best_value",
            "best_model",
            "recovery",
            "updated",
        )
        self.projects_tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
        )
        headings = {
            "run_id": "Run ID",
            "best_epoch": "Best Epoch",
            "best_value": "Best Validation Loss",
            "best_model": "Best Model",
            "recovery": "Recovery State",
            "updated": "Last Updated",
        }
        widths = {
            "run_id": 205,
            "best_epoch": 95,
            "best_value": 145,
            "best_model": 100,
            "recovery": 110,
            "updated": 165,
        }
        for column in columns:
            self.projects_tree.heading(column, text=headings[column])
            self.projects_tree.column(
                column,
                width=widths[column],
                anchor=tk.CENTER,
            )
        scroll = ttk.Scrollbar(
            table_frame,
            orient=tk.VERTICAL,
            command=self.projects_tree.yview,
        )
        self.projects_tree.configure(yscrollcommand=scroll.set)
        self.projects_tree.pack(
            side=tk.LEFT,
            fill=tk.BOTH,
            expand=True,
        )
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.checkpoint_directory_by_item = {}
        root = _training_checkpoint_root()
        if root.exists():
            run_directories = sorted(
                (path for path in root.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        else:
            run_directories = []
        for run_directory in run_directories:
            state_path = run_directory / BEST_STATE_FILENAME
            state = {}
            try:
                if state_path.exists():
                    with open(
                        state_path,
                        "r",
                        encoding="utf-8",
                    ) as state_file:
                        state = json.load(state_file)
            except Exception:
                state = {}
            best_model_exists = (
                run_directory / BEST_MODEL_FILENAME
            ).exists()
            recovery_path = run_directory / RECOVERY_DIRECTORY_NAME
            recovery_exists = (
                recovery_path.exists()
                and any(recovery_path.iterdir())
            )
            updated_timestamp = max(
                [run_directory.stat().st_mtime]
                + [
                    path.stat().st_mtime
                    for path in (
                        state_path,
                        run_directory / BEST_MODEL_FILENAME,
                    )
                    if path.exists()
                ]
            )
            item_id = self.projects_tree.insert(
                "",
                tk.END,
                values=(
                    run_directory.name,
                    state.get("best_epoch", "—"),
                    (
                        f"{float(state['best_value']):.6g}"
                        if state.get("best_value") is not None
                        else "—"
                    ),
                    "Available" if best_model_exists else "—",
                    "Available" if recovery_exists else "—",
                    datetime.fromtimestamp(updated_timestamp).strftime(
                        "%Y-%m-%d %H:%M"
                    ),
                ),
            )
            self.checkpoint_directory_by_item[item_id] = run_directory

        if not run_directories:
            ttk.Label(
                table_frame,
                text=(
                    "No checkpoints are available yet. They appear here after "
                    "the first training epoch is saved."
                ),
                foreground="#245a85",
            ).place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        actions = ttk.Frame(self.content_frame)
        actions.pack(fill=tk.X, padx=12, pady=(6, 12))
        ttk.Button(
            actions,
            text="Refresh",
            command=self.show_projects,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            actions,
            text="Copy Selected Folder Path",
            command=self.copy_selected_checkpoint_path,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            actions,
            text="Open Selected Folder",
            command=self.open_selected_checkpoint_folder,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            actions,
            text="Load Packaged Model",
            command=self.open_model_evaluation,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            actions,
            text="Open Object Detection Runs",
            command=self.open_object_detection_runs,
        ).pack(side=tk.LEFT, padx=4)

    def selected_checkpoint_directory(self):
        if not hasattr(self, "projects_tree"):
            return None
        selection = self.projects_tree.selection()
        if not selection:
            return None
        return self.checkpoint_directory_by_item.get(selection[0])

    def copy_selected_checkpoint_path(self):
        directory = self.selected_checkpoint_directory()
        if directory is None:
            messagebox.showinfo(
                "Select Checkpoint",
                "Select a checkpoint row first.",
            )
            return
        self.clipboard_clear()
        self.clipboard_append(str(directory))
        messagebox.showinfo(
            "Path Copied",
            "The checkpoint folder path was copied to the clipboard.",
        )

    def open_selected_checkpoint_folder(self):
        directory = self.selected_checkpoint_directory()
        if directory is None:
            messagebox.showinfo(
                "Select Checkpoint",
                "Select a checkpoint row first.",
            )
            return
        try:
            system_name = platform.system()
            if system_name == "Windows":
                os.startfile(str(directory))
            elif system_name == "Darwin":
                subprocess.Popen(["open", str(directory)])
            else:
                subprocess.Popen(["xdg-open", str(directory)])
        except Exception as exc:
            messagebox.showerror(
                "Open Folder Error",
                f"Could not open the folder:\n{exc}",
            )

    def open_object_detection_runs(self):
        directory = _object_detection_run_root()
        try:
            system_name = platform.system()
            if system_name == "Windows":
                os.startfile(str(directory))
            elif system_name == "Darwin":
                subprocess.Popen(["open", str(directory)])
            else:
                subprocess.Popen(["xdg-open", str(directory)])
        except Exception as exc:
            messagebox.showerror(
                "Open Folder Error",
                f"Could not open the object-detection runs folder:\n{exc}",
            )

    def show_settings(self):
        if self.training_running or self.ai_request_running:
            messagebox.showinfo(
                "Operation Running",
                "Provider settings cannot be changed while training or an AI "
                "request is running.",
            )
            return
        if self.current_view != "settings":
            self.settings_return_view = self.current_view
        self.current_view = "settings"
        self.clear_content()
        self.title_label.config(text="Settings")
        self.progress_label.config(
            text="Connect AI Provider — credentials and privacy controls"
        )
        self.set_workflow_navigation_visible(False)
        self.home_button.config(state=tk.NORMAL)
        self.settings_button.config(state=tk.DISABLED)
        self.build_settings_page()
        self.on_ai_provider_changed(update_existing=False)
        self.refresh_settings_status()

    def close_settings(self):
        if self.settings_return_view == "wizard":
            self.show_step()
        elif self.settings_return_view == WORKSPACE_FILTER:
            self.show_filter_workspace()
        elif self.settings_return_view == "projects":
            self.show_projects()
        else:
            self.show_home()

    def build_settings_page(self):
        shell = ttk.Frame(self.content_frame)
        shell.pack(fill=tk.BOTH, expand=True, padx=12, pady=(8, 6))
        shell.rowconfigure(1, weight=1)
        shell.columnconfigure(0, weight=1)

        # Connection health stays visible even when the detailed settings below
        # need to scroll on a smaller laptop display.
        connection_frame = ttk.LabelFrame(
            shell,
            text="AI Connection Status",
        )
        connection_frame.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 8),
        )
        connection_frame.columnconfigure(1, weight=1)

        self.settings_connection_badge = tk.Label(
            connection_frame,
            text="OFFLINE",
            background="#6b7280",
            foreground="white",
            font=("Arial", 10, "bold"),
            padx=14,
            pady=7,
        )
        self.settings_connection_badge.grid(
            row=0,
            column=0,
            rowspan=2,
            sticky="nw",
            padx=(12, 14),
            pady=12,
        )

        connection_summary = ttk.Frame(connection_frame)
        connection_summary.grid(
            row=0,
            column=1,
            sticky="ew",
            pady=(10, 2),
        )
        self.settings_connection_provider_label = ttk.Label(
            connection_summary,
            text="Provider:",
            font=("Arial", 10, "bold"),
        )
        self.settings_connection_provider_label.pack(
            side=tk.LEFT,
            padx=(0, 16),
        )
        self.settings_connection_model_label = ttk.Label(
            connection_summary,
            text="Model:",
        )
        self.settings_connection_model_label.pack(
            side=tk.LEFT,
            padx=(0, 16),
        )
        self.settings_connection_key_label = ttk.Label(
            connection_summary,
            text="API key:",
        )
        self.settings_connection_key_label.pack(side=tk.LEFT)

        self.settings_connection_status_label = ttk.Label(
            connection_frame,
            text=self.ai_connection_message,
            wraplength=760,
            foreground="#245a85",
        )
        self.settings_connection_status_label.grid(
            row=1,
            column=1,
            sticky="nw",
            pady=(0, 10),
        )
        self.settings_last_test_label = ttk.Label(
            connection_frame,
            text="",
            justify=tk.LEFT,
            foreground="#666666",
        )
        self.settings_last_test_label.grid(
            row=0,
            column=2,
            rowspan=2,
            sticky="e",
            padx=12,
            pady=10,
        )
        self.settings_test_button = ttk.Button(
            connection_frame,
            text="Test Connection",
            command=self.start_ai_connection_test,
        )
        self.settings_test_button.grid(
            row=0,
            column=3,
            rowspan=2,
            sticky="e",
            padx=(4, 12),
            pady=12,
        )

        scroll_host = ttk.Frame(shell)
        scroll_host.grid(row=1, column=0, sticky="nsew")
        scroll_host.rowconfigure(0, weight=1)
        scroll_host.columnconfigure(0, weight=1)
        self.settings_canvas = tk.Canvas(
            scroll_host,
            highlightthickness=0,
            borderwidth=0,
        )
        settings_scrollbar = ttk.Scrollbar(
            scroll_host,
            orient=tk.VERTICAL,
            command=self.settings_canvas.yview,
        )
        self.settings_canvas.configure(
            yscrollcommand=settings_scrollbar.set
        )
        self.settings_canvas.grid(row=0, column=0, sticky="nsew")
        settings_scrollbar.grid(row=0, column=1, sticky="ns")

        outer = ttk.Frame(self.settings_canvas)
        self.settings_canvas_window = self.settings_canvas.create_window(
            (0, 0),
            window=outer,
            anchor="nw",
        )
        outer.bind("<Configure>", self._resize_settings_scroll_region)
        self.settings_canvas.bind(
            "<Configure>",
            self._resize_settings_inner_width,
        )
        self.bind_all("<MouseWheel>", self._on_settings_mousewheel)
        self.bind_all("<Button-4>", self._on_settings_mousewheel)
        self.bind_all("<Button-5>", self._on_settings_mousewheel)

        provider_frame = ttk.LabelFrame(
            outer,
            text="Provider Configuration",
        )
        provider_frame.pack(fill=tk.X, padx=(0, 5), pady=(0, 8))
        provider_frame.columnconfigure(1, weight=1)
        provider_frame.columnconfigure(3, weight=1)

        ttk.Label(provider_frame, text="Provider:").grid(
            row=0, column=0, sticky="w", padx=10, pady=7
        )
        self.settings_provider_combo = ttk.Combobox(
            provider_frame,
            textvariable=self.ai_provider_var,
            values=AI_PROVIDER_TYPES,
            state="readonly",
            width=32,
        )
        self.settings_provider_combo.grid(
            row=0, column=1, sticky="ew", padx=6, pady=7
        )
        self.settings_provider_combo.bind(
            "<<ComboboxSelected>>",
            self.on_ai_provider_changed,
        )

        ttk.Label(provider_frame, text="Model:").grid(
            row=0, column=2, sticky="w", padx=(18, 6), pady=7
        )
        self.ai_model_entry = ttk.Entry(
            provider_frame,
            textvariable=self.ai_model_var,
        )
        self.ai_model_entry.grid(
            row=0, column=3, sticky="ew", padx=(6, 10), pady=7
        )
        self.ai_model_entry.bind(
            "<KeyRelease>",
            self.mark_ai_settings_changed,
        )
        self.ai_model_entry.bind(
            "<FocusOut>",
            self.mark_ai_settings_changed,
        )

        ttk.Label(provider_frame, text="API key:").grid(
            row=1, column=0, sticky="w", padx=10, pady=7
        )
        key_row = ttk.Frame(provider_frame)
        key_row.grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=6, pady=7
        )
        key_row.columnconfigure(0, weight=1)
        self.settings_api_key_entry = ttk.Entry(
            key_row,
            textvariable=self.ai_api_key_var,
            show="•",
        )
        self.settings_api_key_entry.grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        self.settings_api_key_entry.bind(
            "<KeyRelease>",
            self.mark_ai_settings_changed,
        )
        self.settings_api_key_entry.bind(
            "<FocusOut>",
            self.mark_ai_settings_changed,
        )
        self.ai_key_entry = self.settings_api_key_entry
        self.show_key_checkbutton = ttk.Checkbutton(
            key_row,
            text="Show",
            variable=self.show_ai_key_var,
            command=self.toggle_ai_key_visibility,
        )
        self.show_key_checkbutton.grid(row=0, column=1, padx=4)
        remember_state = tk.NORMAL if keyring is not None else tk.DISABLED
        self.remember_key_checkbutton = ttk.Checkbutton(
            key_row,
            text="Remember securely on this computer",
            variable=self.remember_ai_key_var,
            state=remember_state,
        )
        self.remember_key_checkbutton.grid(row=0, column=2, padx=8)

        ttk.Label(provider_frame, text="Base URL:").grid(
            row=2, column=0, sticky="w", padx=10, pady=7
        )
        self.ai_base_url_entry = ttk.Entry(
            provider_frame,
            textvariable=self.ai_base_url_var,
        )
        self.ai_base_url_entry.grid(
            row=2, column=1, sticky="ew", padx=6, pady=7
        )
        self.ai_base_url_entry.bind(
            "<KeyRelease>",
            self.mark_ai_settings_changed,
        )
        self.ai_base_url_entry.bind(
            "<FocusOut>",
            self.mark_ai_settings_changed,
        )

        ttk.Label(provider_frame, text="Timeout (s):").grid(
            row=2, column=2, sticky="w", padx=(18, 6), pady=7
        )
        self.ai_timeout_entry = ttk.Entry(
            provider_frame,
            textvariable=self.ai_timeout_var,
            width=12,
        )
        self.ai_timeout_entry.grid(
            row=2, column=3, sticky="w", padx=6, pady=7
        )

        self.settings_key_status_label = ttk.Label(
            provider_frame,
            text="",
            foreground="#245a85",
        )
        self.settings_key_status_label.grid(
            row=3,
            column=0,
            columnspan=4,
            sticky="w",
            padx=10,
            pady=(0, 7),
        )

        if keyring is None:
            ttk.Label(
                provider_frame,
                text=(
                    "Optional secure key storage is unavailable. Install the "
                    "'keyring' package to enable Remember securely. Session "
                    "keys and environment variables still work."
                ),
                foreground="#7a4b00",
                wraplength=1050,
            ).grid(
                row=4,
                column=0,
                columnspan=4,
                sticky="w",
                padx=10,
                pady=(0, 8),
            )

        privacy_frame = ttk.LabelFrame(
            outer,
            text="Privacy and Data Transmission",
        )
        privacy_frame.pack(fill=tk.X, padx=(0, 5), pady=8)
        ttk.Checkbutton(
            privacy_frame,
            text="Send compact statistical profile only",
            variable=self.profile_only_var,
            state=tk.DISABLED,
        ).pack(anchor="w", padx=12, pady=(10, 5))
        ttk.Checkbutton(
            privacy_frame,
            text="Allow representative raw data samples (future option)",
            variable=self.allow_sample_rows_var,
            state=tk.DISABLED,
        ).pack(anchor="w", padx=12, pady=5)
        ttk.Label(
            privacy_frame,
            text=(
                "Connection testing sends only provider authentication and a "
                "model-list request. AI analysis sends column names and compact "
                "statistics only. The AI Code Assistant sends source text only "
                "for files the user marks Share and approves in a second "
                "confirmation. API keys are never written to projects, reports, "
                "logs, model packages, or generated integration code."
            ),
            wraplength=1050,
            foreground="#245a85",
        ).pack(anchor="w", padx=12, pady=(5, 10))

        guide_frame = ttk.LabelFrame(
            outer,
            text="NN Studio Guide",
        )
        guide_frame.pack(fill=tk.X, padx=(0, 5), pady=8)
        ttk.Checkbutton(
            guide_frame,
            text="Open the guide automatically when NN Studio starts",
            variable=self.guide_auto_open_var,
        ).pack(anchor="w", padx=12, pady=(10, 5))
        ttk.Checkbutton(
            guide_frame,
            text=(
                "Use the connected AI provider by default for guide questions"
            ),
            variable=self.guide_use_ai_var,
        ).pack(anchor="w", padx=12, pady=5)
        ttk.Label(
            guide_frame,
            text=(
                "Offline navigation always remains available. AI guide "
                "questions send only the text you type, the current page name, "
                "and a fixed list of NN Studio functions—never datasets, model "
                "files, predictions, or API keys."
            ),
            wraplength=1050,
            foreground="#245a85",
        ).pack(anchor="w", padx=12, pady=(4, 10))

        ttk.Label(
            outer,
            text=(
                "Tip: after changing the provider, model, API key, or Base "
                "URL, select Test Connection again. A green CONNECTED badge "
                "means the current settings were verified successfully."
            ),
            foreground="#245a85",
            wraplength=1050,
        ).pack(anchor="w", padx=10, pady=(2, 10))

        # Primary actions remain fixed below the scrollable settings content.
        ttk.Separator(shell, orient=tk.HORIZONTAL).grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(8, 6),
        )
        action_row = ttk.Frame(shell)
        action_row.grid(row=3, column=0, sticky="ew")
        self.settings_save_button = ttk.Button(
            action_row,
            text="Save Settings",
            command=self.save_ai_settings,
        )
        self.settings_save_button.pack(side=tk.LEFT, padx=4)
        self.settings_disconnect_button = ttk.Button(
            action_row,
            text="Disconnect Provider",
            command=self.disconnect_ai_provider,
        )
        self.settings_disconnect_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(
            action_row,
            text="Back",
            command=self.close_settings,
        ).pack(side=tk.RIGHT, padx=4)

    def start_ai_connection_test(self):
        try:
            if self.ai_connection_testing:
                return
            provider_name = self.ai_provider_var.get()
            if provider_name not in OFFICIAL_PROVIDER_SETTINGS:
                raise ValueError(
                    "Choose OpenAI, DeepSeek, Claude, or an OpenAI-compatible API "
                    "before testing the connection."
                )
            api_key = self.resolve_ai_api_key(provider_name)
            if not api_key:
                raise ValueError(
                    "Enter an API key or configure the provider environment "
                    "variable before testing."
                )
            model_name = self.ai_model_var.get().strip()
            base_url = _validate_provider_base_url(
                self.ai_base_url_var.get()
            )
            timeout_seconds = float(self.ai_timeout_var.get())

            defaults = OFFICIAL_PROVIDER_SETTINGS[provider_name]
            official_host = urlparse(defaults["base_url"]).hostname
            selected_host = urlparse(base_url).hostname
            if (
                provider_name == AI_PROVIDER_COMPATIBLE
                or selected_host != official_host
            ):
                expected_text = (
                    "Provider type: user-configured OpenAI-compatible endpoint"
                    if provider_name == AI_PROVIDER_COMPATIBLE
                    else f"Official host: {official_host}"
                )
                approved = messagebox.askyesno(
                    "Confirm Custom Provider Host",
                    (
                        "The API key will be sent to the selected provider "
                        "endpoint for this connection test.\n\n"
                        f"Selected host: {selected_host}\n"
                        f"{expected_text}\n\n"
                        "Continue only if you trust this host."
                    ),
                )
                if not approved:
                    return

            self.ai_connection_testing = True
            self.ai_connection_state = "testing"
            self.ai_connection_message = (
                f"Testing {provider_name} authentication..."
            )
            self.ai_last_connection_details = (
                "Waiting for the provider response."
            )
            self.update_provider_status_display()
            self.refresh_settings_status()
            if hasattr(self, "settings_test_button"):
                self.settings_test_button.config(state=tk.DISABLED)
            for widget_name in (
                "settings_provider_combo",
                "settings_save_button",
                "settings_disconnect_button",
            ):
                widget = getattr(self, widget_name, None)
                if widget is not None:
                    widget.config(state=tk.DISABLED)

            threading.Thread(
                target=self.ai_connection_test_worker,
                args=(
                    provider_name,
                    api_key,
                    model_name,
                    base_url,
                    timeout_seconds,
                ),
                daemon=True,
            ).start()
        except Exception as exc:
            messagebox.showerror("Connection Test", str(exc))

    def ai_connection_test_worker(
        self,
        provider_name,
        api_key,
        model_name,
        base_url,
        timeout_seconds,
    ):
        try:
            result = test_ai_provider_connection(
                provider_name=provider_name,
                api_key=api_key,
                model=model_name,
                base_url=base_url,
                timeout_seconds=timeout_seconds,
            )
            self.ui_queue.put(("ai_connection_test_finished", result))
        except Exception as exc:
            self.ui_queue.put(("ai_connection_test_failed", str(exc)))

    def ai_connection_test_finished(self, result):
        self.ai_connection_testing = False
        self.ai_last_connection_check = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        settings_unchanged = (
            self.ai_provider_var.get() == result["provider"]
            and self.ai_model_var.get().strip() == result["model"]
            and self.ai_base_url_var.get().strip().rstrip("/")
            == result["base_url"]
        )
        model_status = result.get("model_available")
        if not settings_unchanged:
            self.ai_connection_state = "configured"
            self.ai_connection_message = (
                "The connection test completed, but provider settings changed "
                "during the test. Test the current settings again."
            )
            self.ai_last_connection_details = (
                "The completed test does not match the current settings."
            )
        elif model_status is False:
            self.ai_connection_state = "connected"
            self.ai_connection_message = (
                f"{result['provider']} authentication succeeded, but "
                f"'{result['model']}' was not found in the returned model "
                "list. Review the model name before making AI requests."
            )
            self.ai_last_connection_details = (
                "Authentication passed; selected model was not listed."
            )
        else:
            self.ai_connection_state = "connected"
            self.ai_connection_message = (
                f"{result['provider']} connected successfully with model "
                f"'{result['model']}'."
            )
            model_count = int(result.get("available_model_count") or 0)
            self.ai_last_connection_details = (
                "Authentication passed"
                + (
                    f"; provider returned {model_count} model(s)."
                    if model_count
                    else "; provider did not expose a model list."
                )
            )
        self.save_ai_settings(show_confirmation=False)
        for widget_name, widget_state in (
            ("settings_test_button", tk.NORMAL),
            ("settings_provider_combo", "readonly"),
            ("settings_save_button", tk.NORMAL),
            ("settings_disconnect_button", tk.NORMAL),
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.config(state=widget_state)
        self.refresh_settings_status()

    def ai_connection_test_failed(self, error_message):
        self.ai_connection_testing = False
        self.ai_connection_state = "disconnected"
        self.ai_last_connection_check = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        self.ai_connection_message = (
            "Connection test failed: " + str(error_message)
        )
        self.ai_last_connection_details = "Authentication or network check failed."
        for widget_name, widget_state in (
            ("settings_test_button", tk.NORMAL),
            ("settings_provider_combo", "readonly"),
            ("settings_save_button", tk.NORMAL),
            ("settings_disconnect_button", tk.NORMAL),
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.config(state=widget_state)
        self.refresh_settings_status()
        messagebox.showerror("Connection Test Failed", error_message)
