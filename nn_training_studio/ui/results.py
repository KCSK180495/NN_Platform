"""Ui / results for NN Training Studio."""

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
from tkinter.scrolledtext import ScrolledText
from datetime import datetime
from tkinter import filedialog
import json
import queue
from tkinter import messagebox
import numpy as np
import os
import platform
import matplotlib.pyplot as plt
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import ttk
from urllib.parse import urlparse
from nn_training_studio.ai_transport import (
    _validate_provider_base_url,
)
from nn_training_studio.constants import (
    APP_VERSION,
    BUILT_IN_RESULT_PLOTS,
    OFFICIAL_PROVIDER_SETTINGS,
    PLOT_MODE_AI,
    PLOT_MODE_BUILT_IN,
    PLOT_MODE_MANUAL,
)
from nn_training_studio.plotting import (
    _normalise_result_context,
    build_result_context_schema,
    execute_custom_plot_code,
    request_ai_plot_recipe,
    validate_custom_plot_code,
    snapshot_plot_provider_settings,
)
from nn_training_studio.results import (
    _result_json_default,
    zip_result_directory,
)


from nn_training_studio.result_customization import extra_plot_code
from nn_training_studio.ui.scrollable import ScrollableForm


class CustomResultsStudioWindow(tk.Toplevel):
    """Built-in, manual, and reviewed AI result visualisation workspace."""

    def __init__(
        self,
        parent,
        context_provider,
        settings_owner=None,
        attach_callback=None,
    ):
        super().__init__(parent)
        self.title(f"NN Training Studio {APP_VERSION} — Custom Results")
        self.geometry("1320x860")
        self.minsize(760, 520)
        self.context_provider = context_provider
        self.settings_owner = settings_owner
        self.attach_callback = attach_callback
        self.context = None
        self.context_schema = None
        self.current_output_directory = None
        self.preview_canvas = None
        self.ai_recipe = None
        self.ai_running = False
        self.plot_running = False
        self.completed_plot_snapshot = None
        self._worker_results = queue.Queue()
        self._worker_lock = threading.Lock()
        self._closed = False

        self.mode_var = tk.StringVar(value=PLOT_MODE_BUILT_IN)
        self.table_var = tk.StringVar(value="")
        self.x_column_var = tk.StringVar(value="<Index>")
        self.y_column_var = tk.StringVar(value="")
        self.plot_type_var = tk.StringVar(value=BUILT_IN_RESULT_PLOTS[0])
        self.status_var = tk.StringVar(value="Loading result context...")

        self.protocol("WM_DELETE_WINDOW", self.close_window)
        self.bind("<Destroy>", self._on_studio_destroy, add="+")
        self.build_interface()
        self.refresh_context()
        self._poll_after_id = self.after(100, self._poll_workers)

    def build_interface(self):
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=12, pady=(10, 5))
        ttk.Label(
            header,
            text="Training & Model Results Studio",
            font=("Arial", 18, "bold"),
        ).pack(side=tk.LEFT)
        ttk.Button(
            header,
            text="Refresh Result Data",
            command=self.refresh_context,
        ).pack(side=tk.RIGHT)

        ttk.Label(
            self,
            text=(
                "Create built-in graphs, write a manual plotting function, or "
                "ask AI to draft code. AI code is displayed for review and is "
                "never executed automatically."
            ),
            wraplength=1240,
        ).pack(anchor="w", padx=12, pady=(0, 6))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=5)
        self.data_tab = ttk.Frame(self.notebook)
        self.design_tab = ttk.Frame(self.notebook)
        self.preview_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.data_tab, text="1. Available Data")
        self.notebook.add(self.design_tab, text="2. Design Plot")
        self.notebook.add(self.preview_tab, text="3. Preview & Export")

        self.schema_text = ScrolledText(
            self.data_tab,
            wrap=tk.NONE,
            font=("Consolas", 10),
        )
        self.schema_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.design_form = ScrollableForm(self.design_tab)
        self.design_form.pack(fill=tk.BOTH, expand=True)
        design_content = self.design_form.content
        controls = ttk.LabelFrame(design_content, text="Plot Mode and Data")
        controls.pack(fill=tk.X, padx=8, pady=8)
        mode_row = ttk.Frame(controls)
        mode_row.pack(fill=tk.X)
        for text_value in (
            PLOT_MODE_BUILT_IN,
            PLOT_MODE_MANUAL,
            PLOT_MODE_AI,
        ):
            ttk.Radiobutton(
                mode_row,
                text=text_value,
                value=text_value,
                variable=self.mode_var,
            ).pack(side=tk.LEFT, padx=8, pady=7)

        data_row = ttk.Frame(controls)
        data_row.pack(fill=tk.X, pady=4)
        ttk.Label(data_row, text="Table:").pack(side=tk.LEFT, padx=(18, 4))
        self.table_combo = ttk.Combobox(
            data_row,
            textvariable=self.table_var,
            state="readonly",
            width=22,
        )
        self.table_combo.pack(side=tk.LEFT, padx=4)
        self.table_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.refresh_column_choices(),
        )
        ttk.Label(data_row, text="Built-in:").pack(
            side=tk.LEFT, padx=(18, 4)
        )
        self.plot_type_combo = ttk.Combobox(
            data_row,
            textvariable=self.plot_type_var,
            values=BUILT_IN_RESULT_PLOTS,
            state="readonly",
            width=25,
        )
        self.plot_type_combo.pack(side=tk.LEFT, padx=4)
        self.plot_type_combo.bind("<<ComboboxSelected>>", self.on_plot_type_selected)

        columns = ttk.Frame(design_content)
        columns.pack(fill=tk.X, padx=8, pady=(0, 5))
        ttk.Label(columns, text="X / Actual column:").pack(side=tk.LEFT)
        self.x_combo = ttk.Combobox(
            columns,
            textvariable=self.x_column_var,
            state="readonly",
            width=34,
        )
        self.x_combo.pack(side=tk.LEFT, padx=(5, 18))
        ttk.Label(columns, text="Y / Predicted column:").pack(side=tk.LEFT)
        self.y_combo = ttk.Combobox(
            columns,
            textvariable=self.y_column_var,
            state="readonly",
            width=34,
        )
        self.y_combo.pack(side=tk.LEFT, padx=5)
        ttk.Button(
            columns,
            text="Insert Built-in Template",
            command=self.insert_builtin_template,
        ).pack(side=tk.RIGHT, padx=4)

        ai_frame = ttk.LabelFrame(
            design_content,
            text="AI Plot Request (schema and statistics only; no raw rows)",
        )
        ai_frame.pack(fill=tk.X, padx=8, pady=5)
        self.ai_prompt_text = ScrolledText(ai_frame, height=4, wrap=tk.WORD)
        self.ai_prompt_text.pack(
            side=tk.LEFT,
            fill=tk.X,
            expand=True,
            padx=6,
            pady=6,
        )
        self.ai_prompt_text.insert(
            tk.END,
            "Plot actual and predicted values over the sample sequence, "
            "highlight large errors, and use clear publication-style labels.",
        )
        self.ai_generate_button = ttk.Button(
            ai_frame,
            text="Generate Plot Code with AI",
            command=self.start_ai_generation,
        )
        self.ai_generate_button.pack(side=tk.RIGHT, padx=8, pady=8)

        editor_frame = ttk.LabelFrame(
            design_content,
            text=(
                "Editable Python — define create_plot(context) and return a "
                "Matplotlib Figure"
            ),
        )
        editor_frame.pack(
            fill=tk.BOTH,
            expand=True,
            padx=8,
            pady=5,
        )
        self.code_text = ScrolledText(
            editor_frame,
            wrap=tk.NONE,
            font=("Consolas", 10),
            undo=True,
            height=12,
        )
        self.code_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        actions = ttk.Frame(design_content)
        actions.pack(fill=tk.X, padx=8, pady=(3, 8))
        ttk.Button(
            actions,
            text="Validate Code",
            command=self.validate_editor_code,
        ).pack(side=tk.LEFT, padx=4)
        self.run_button = ttk.Button(
            actions,
            text="Validate and Run Preview",
            command=self.start_plot_execution,
        )
        self.run_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(
            actions,
            text="Save Plot Recipe",
            command=self.save_plot_recipe,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(
            actions,
            textvariable=self.status_var,
            foreground="#245a85",
        ).pack(side=tk.LEFT, padx=14)

        self.preview_host = ttk.Frame(self.preview_tab)
        self.preview_host.pack(
            fill=tk.BOTH,
            expand=True,
            padx=8,
            pady=8,
        )
        self.preview_message = ttk.Label(
            self.preview_host,
            text="Run a plot to display its preview here.",
        )
        self.preview_message.pack(expand=True)

        export_actions = ttk.Frame(self.preview_tab)
        export_actions.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.save_result_button = ttk.Button(
            export_actions,
            text="Save Custom Result ZIP",
            command=self.save_custom_result_bundle,
            state=tk.DISABLED,
        )
        self.save_result_button.pack(side=tk.LEFT, padx=4)
        self.attach_button = ttk.Button(
            export_actions,
            text="Add to Complete Results",
            command=self.attach_to_complete_results,
            state=tk.DISABLED,
        )
        self.attach_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(
            export_actions,
            text="Open PNG",
            command=self.open_current_png,
        ).pack(side=tk.LEFT, padx=4)
        self.ai_explanation_text = ScrolledText(
            self.preview_tab,
            height=6,
            wrap=tk.WORD,
        )
        self.ai_explanation_text.pack(
            fill=tk.X,
            padx=8,
            pady=(0, 8),
        )
        self.ai_explanation_text.insert(
            tk.END,
            "AI explanation and assumptions will appear here.",
        )

    def refresh_context(self):
        try:
            self.context = _normalise_result_context(
                self.context_provider()
            )
            self.context_schema = build_result_context_schema(self.context)
            table_names = list(self.context["tables"])
            self.table_combo.config(values=table_names)
            if self.table_var.get() not in table_names:
                preferred = (
                    "results" if "results" in table_names else table_names[0]
                )
                self.table_var.set(preferred)
            self.refresh_column_choices()
            self.schema_text.delete("1.0", tk.END)
            self.schema_text.insert(
                tk.END,
                json.dumps(
                    self.context_schema,
                    indent=2,
                    default=_result_json_default,
                ),
            )
            if not self.code_text.get("1.0", tk.END).strip():
                self.insert_builtin_template()
            self.status_var.set(
                f"Ready — {len(table_names)} table(s) available."
            )
        except Exception as exc:
            self.status_var.set(str(exc))
            messagebox.showerror("Result Context Error", str(exc), parent=self)

    def refresh_column_choices(self):
        if self.context is None:
            return
        table_name = self.table_var.get()
        frame = self.context["tables"].get(table_name)
        if frame is None:
            return
        columns = [str(column) for column in frame.columns]
        self.x_combo.config(values=["<Index>"] + columns)
        self.y_combo.config(values=columns)
        if self.x_column_var.get() not in ["<Index>"] + columns:
            self.x_column_var.set("<Index>")
        if self.y_column_var.get() not in columns:
            numeric = [
                str(column)
                for column in frame.select_dtypes(include=[np.number]).columns
            ]
            self.y_column_var.set(
                numeric[0] if numeric else (columns[0] if columns else "")
            )

    def _selected_expression(self, column, table_variable="df"):
        if column == "<Index>":
            return f"{table_variable}.index"
        return f"{table_variable}[{column!r}]"

    def create_builtin_code(self):
        table_name = self.table_var.get()
        x_column = self.x_column_var.get()
        y_column = self.y_column_var.get()
        plot_type = self.plot_type_var.get()
        if plot_type in ("Loss & Accuracy", "Correlation Heatmap", "t-SNE") or (
                plot_type == "Training Curves" and table_name == "training_history"):
            return extra_plot_code(plot_type, table_name, self.context["tables"][table_name])
        if not table_name or self.context is None:
            raise ValueError("Choose an available result table.")
        frame = self.context["tables"][table_name]
        columns = [str(column) for column in frame.columns]
        if y_column not in columns and plot_type not in (
            "Training Curves",
            "Confusion Matrix",
        ):
            raise ValueError("Choose a Y/result column.")
        x_expression = self._selected_expression(x_column)
        y_expression = self._selected_expression(y_column)

        body = []
        title = plot_type
        if plot_type == "Line Plot":
            body = [
                f"ax.plot({x_expression}, {y_expression}, linewidth=1.5)",
                f"ax.set_xlabel({('Sample' if x_column == '<Index>' else x_column)!r})",
                f"ax.set_ylabel({y_column!r})",
            ]
        elif plot_type == "Scatter Plot":
            if x_column == "<Index>":
                raise ValueError("Scatter Plot requires an X column.")
            body = [
                f"ax.scatter({x_expression}, {y_expression}, s=18, alpha=0.7)",
                f"ax.set_xlabel({x_column!r})",
                f"ax.set_ylabel({y_column!r})",
            ]
        elif plot_type == "Histogram":
            body = [
                f"values = pd.to_numeric({y_expression}, errors='coerce').dropna()",
                "ax.hist(values, bins=35, alpha=0.85, edgecolor='black')",
                f"ax.set_xlabel({y_column!r})",
                "ax.set_ylabel('Count')",
            ]
        elif plot_type in (
            "Category Counts",
            "Detection Count by Class",
        ):
            body = [
                f"counts = {y_expression}.astype(str).value_counts().head(30)",
                "ax.bar(counts.index, counts.values)",
                "ax.tick_params(axis='x', rotation=45)",
                f"ax.set_xlabel({y_column!r})",
                "ax.set_ylabel('Count')",
            ]
        elif plot_type == "Confidence Distribution":
            body = [
                f"values = pd.to_numeric({y_expression}, errors='coerce').dropna()",
                "ax.hist(values, bins=30, range=(0, 1), alpha=0.85)",
                f"ax.set_xlabel({y_column!r})",
                "ax.set_ylabel('Count')",
            ]
        elif plot_type == "Confusion Matrix":
            actual_column = (
                x_column
                if x_column != "<Index>"
                else (
                    "actual_label"
                    if "actual_label" in columns
                    else "actual"
                    if "actual" in columns
                    else ""
                )
            )
            predicted_column = (
                y_column
                if y_column in columns
                else (
                    "predicted_label"
                    if "predicted_label" in columns
                    else "predicted"
                    if "predicted" in columns
                    else ""
                )
            )
            if (
                actual_column not in columns
                or predicted_column not in columns
            ):
                raise ValueError(
                    "Choose actual labels as X and predicted labels as Y."
                )
            body = [
                f"matrix = pd.crosstab(df[{actual_column!r}], df[{predicted_column!r}])",
                "labels = context['class_names'] or sorted(set(matrix.index) | set(matrix.columns))",
                "matrix = matrix.reindex(index=labels, columns=labels, fill_value=0)",
                "image = ax.imshow(matrix.to_numpy())",
                "ax.set_xticks(range(len(matrix.columns)), [str(v) for v in matrix.columns], rotation=45, ha='right')",
                "ax.set_yticks(range(len(matrix.index)), [str(v) for v in matrix.index])",
                "ax.set_xlabel('Predicted label')",
                "ax.set_ylabel('Actual label')",
                "fig.colorbar(image, ax=ax)",
                "for row in range(len(matrix.index)):\n"
                "        for column in range(len(matrix.columns)):\n"
                "            ax.text(column, row, int(matrix.iloc[row, column]), ha='center', va='center')",
            ]
        elif plot_type == "Actual vs Predicted":
            if x_column == "<Index>":
                raise ValueError(
                    "Choose the actual column as X and predicted column as Y."
                )
            body = [
                f"actual = pd.to_numeric({x_expression}, errors='coerce')",
                f"predicted = pd.to_numeric({y_expression}, errors='coerce')",
                "valid = actual.notna() & predicted.notna()",
                "ax.scatter(actual[valid], predicted[valid], s=18, alpha=0.7)",
                "low = min(actual[valid].min(), predicted[valid].min())",
                "high = max(actual[valid].max(), predicted[valid].max())",
                "ax.plot([low, high], [low, high], linestyle='--', color='red', label='Ideal')",
                f"ax.set_xlabel({'Actual ' + x_column!r})",
                f"ax.set_ylabel({'Predicted ' + y_column!r})",
                "ax.legend()",
            ]
        elif plot_type == "Residual Distribution":
            if x_column == "<Index>":
                raise ValueError(
                    "Choose the actual column as X and predicted column as Y."
                )
            body = [
                f"actual = pd.to_numeric({x_expression}, errors='coerce')",
                f"predicted = pd.to_numeric({y_expression}, errors='coerce')",
                "residual = (actual - predicted).dropna()",
                "ax.hist(residual, bins=35, alpha=0.85, edgecolor='black')",
                "ax.axvline(0, linestyle='--', color='red')",
                "ax.set_xlabel('Residual (actual - predicted)')",
                "ax.set_ylabel('Count')",
            ]
        elif plot_type == "Training Curves":
            numeric_columns = [
                str(column)
                for column in frame.select_dtypes(include=[np.number]).columns
                if str(column).lower() not in ("epoch", "index")
            ]
            if not numeric_columns:
                raise ValueError(
                    "The selected table contains no numeric training curves."
                )
            body = [
                f"for column in {numeric_columns!r}:",
                "    ax.plot(df[column], label=column)",
                "ax.set_xlabel('Epoch')",
                "ax.set_ylabel('Metric value')",
                "ax.legend()",
            ]
        else:
            raise ValueError(f"Unsupported built-in plot: {plot_type}")

        indented_body = "\n    ".join(body)
        return (
            "def create_plot(context):\n"
            "    pd = context['pd']\n"
            "    plt = context['plt']\n"
            f"    df = context['tables'][{table_name!r}]\n"
            "    fig, ax = plt.subplots(figsize=(10, 6))\n"
            f"    {indented_body}\n"
            f"    ax.set_title({title!r})\n"
            "    ax.grid(True, alpha=0.3)\n"
            "    fig.tight_layout()\n"
            "    return fig\n"
        )

    def insert_builtin_template(self):
        try:
            code = self.create_builtin_code()
            self.code_text.delete("1.0", tk.END)
            self.code_text.insert(tk.END, code)
            self.mode_var.set(PLOT_MODE_BUILT_IN)
            self.ai_recipe = None
            self.status_var.set("Built-in template inserted. Review or run it.")
        except Exception as exc:
            self.status_var.set(str(exc))

    def validate_editor_code(self, show_message=True):
        try:
            code = self.code_text.get("1.0", tk.END).strip()
            result = validate_custom_plot_code(
                code,
                self.context["tables"].keys() if self.context else [],
            )
            self.status_var.set(
                f"Code valid — {result['line_count']} line(s)."
            )
            if show_message:
                messagebox.showinfo(
                    "Plot Code Valid",
                    "The plotting contract and restricted-code checks passed.",
                    parent=self,
                )
            return True
        except Exception as exc:
            self.status_var.set(str(exc))
            if show_message:
                messagebox.showerror(
                    "Plot Code Invalid",
                    str(exc),
                    parent=self,
                )
            return False

    def start_ai_generation(self):
        if self.ai_running:
            return
        if self.context_schema is None:
            messagebox.showwarning(
                "No Result Context",
                "Refresh the result data before requesting a plot.",
                parent=self,
            )
            return
        request_text = self.ai_prompt_text.get("1.0", tk.END).strip()
        if not request_text:
            messagebox.showwarning(
                "Plot Request Required",
                "Describe the graph you want the AI to generate.",
                parent=self,
            )
            return
        try:
            provider_name = self.settings_owner.ai_provider_var.get()
            if provider_name not in OFFICIAL_PROVIDER_SETTINGS:
                raise ValueError(
                    "Choose OpenAI, DeepSeek, Claude, or an OpenAI-compatible API in "
                    "Settings before requesting "
                    "AI plot code."
                )
            selected_url = _validate_provider_base_url(
                self.settings_owner.ai_base_url_var.get()
            )
            official_url = OFFICIAL_PROVIDER_SETTINGS[
                provider_name
            ]["base_url"]
            selected_host = urlparse(selected_url).hostname
            official_host = urlparse(official_url).hostname
            if selected_host != official_host:
                approved = messagebox.askyesno(
                    "Confirm Custom Provider Host",
                    (
                        f"The selected Base URL is not the official "
                        f"{provider_name} host.\n\n"
                        f"Selected host: {selected_host}\n"
                        f"Official host: {official_host}\n\n"
                        "Your API key and result schema will be sent to the "
                        "selected host. No raw result rows are included. "
                        "Continue only if you trust it."
                    ),
                    parent=self,
                )
                if not approved:
                    return
            provider_settings = snapshot_plot_provider_settings(self.settings_owner)
        except Exception as exc:
            messagebox.showerror(
                "AI Provider Settings",
                str(exc),
                parent=self,
            )
            return
        self.ai_running = True
        self.ai_generate_button.config(state=tk.DISABLED)
        self.status_var.set(
            "Generating and locally validating plot code..."
        )
        threading.Thread(
            target=self._ai_generation_worker,
            args=(provider_settings, json.loads(json.dumps(self.context_schema)), request_text),
            daemon=True,
        ).start()

    def _ai_generation_worker(self, provider_settings, context_schema, request_text):
        try:
            recipe = request_ai_plot_recipe(
                provider_settings,
                context_schema,
                request_text,
            )
            self._deliver_worker_result("ai_done", recipe)
        except Exception as exc:
            details = str(exc)
            self._deliver_worker_result("ai_failed", details)

    def finish_ai_generation(self, recipe):
        self.ai_running = False
        self.ai_generate_button.config(state=tk.NORMAL)
        self.ai_recipe = recipe
        self.mode_var.set(PLOT_MODE_AI)
        if recipe.get("table") in self.context["tables"]:
            self.table_var.set(recipe["table"])
            self.refresh_column_choices()
        self.code_text.delete("1.0", tk.END)
        self.code_text.insert(tk.END, recipe["code"])
        self.ai_explanation_text.delete("1.0", tk.END)
        explanation_lines = [
            recipe.get("title", "AI-generated plot"),
            "",
            recipe.get("explanation", ""),
            "",
            "Required columns: "
            + ", ".join(recipe.get("required_columns", [])),
            "Assumptions:",
        ]
        explanation_lines.extend(
            f"- {item}" for item in recipe.get("assumptions", [])
        )
        explanation_lines.extend([
            "",
            f"Provider: {recipe.get('provider')}",
            f"Model: {recipe.get('model')}",
            f"Data sent: {recipe.get('data_sent')}",
            "",
            "The generated code has not been executed. Review it, then select "
            "Validate and Run Preview.",
        ])
        self.ai_explanation_text.insert(
            tk.END,
            "\n".join(explanation_lines),
        )
        self.status_var.set(
            "AI code generated and locally validated; review before running."
        )
        self.notebook.select(self.design_tab)

    def fail_ai_generation(self, details):
        self.ai_running = False
        self.ai_generate_button.config(state=tk.NORMAL)
        self.status_var.set("AI plot generation failed.")
        messagebox.showerror(
            "AI Plot Generation Failed",
            details,
            parent=self,
        )

    def start_plot_execution(self):
        if self.plot_running or self.context is None:
            return
        if not self.validate_editor_code(show_message=False):
            messagebox.showerror(
                "Plot Code Invalid",
                self.status_var.get(),
                parent=self,
            )
            return
        code = self.code_text.get("1.0", tk.END).strip()
        context_snapshot = _normalise_result_context(self.context)
        self._pending_plot_snapshot = {"context": context_snapshot, "recipe": self.current_recipe(),
                                       "explanation": self.ai_explanation_text.get("1.0", tk.END).strip()}
        self.plot_running = True
        self.run_button.config(state=tk.DISABLED)
        self.status_var.set("Running plot in a separate process...")
        threading.Thread(
            target=self._plot_execution_worker,
            args=(code, context_snapshot),
            daemon=True,
        ).start()

    def _plot_execution_worker(self, code, context_snapshot):
        try:
            directory = execute_custom_plot_code(
                code,
                context_snapshot,
                timeout_seconds=30,
            )
            self._deliver_worker_result("plot_done", directory)
        except Exception as exc:
            details = str(exc)
            self._deliver_worker_result("plot_failed", details)

    def finish_plot_execution(self, directory):
        self.plot_running = False
        self.run_button.config(state=tk.NORMAL)
        old_directory = self.current_output_directory
        self.current_output_directory = directory
        self.completed_plot_snapshot = self._pending_plot_snapshot
        if old_directory and old_directory != directory:
            shutil.rmtree(old_directory, ignore_errors=True)
        self.display_plot_preview(Path(directory) / "plot.png")
        self.save_result_button.config(state=tk.NORMAL)
        self.attach_button.config(
            state=tk.NORMAL if self.attach_callback else tk.DISABLED
        )
        self.status_var.set("Plot completed. Preview and export are ready.")
        self.notebook.select(self.preview_tab)

    def fail_plot_execution(self, details):
        self.plot_running = False
        self.run_button.config(state=tk.NORMAL)
        self.status_var.set("Plot execution failed.")
        messagebox.showerror(
            "Custom Plot Failed",
            details,
            parent=self,
        )

    def display_plot_preview(self, png_path):
        if self.preview_canvas is not None:
            self.preview_canvas.get_tk_widget().destroy()
            self.preview_canvas = None
        self.preview_message.pack_forget()
        image = plt.imread(png_path)
        height, width = image.shape[:2]
        aspect = width / max(1, height)
        figure = plt.Figure(
            figsize=(min(11, max(6, 6 * aspect)), 6),
            dpi=100,
        )
        axis = figure.add_subplot(111)
        axis.imshow(image)
        axis.axis("off")
        figure.tight_layout(pad=0)
        self.preview_canvas = FigureCanvasTkAgg(
            figure,
            master=self.preview_host,
        )
        self.preview_canvas.draw()
        self.preview_canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True,
        )

    def current_recipe(self):
        return {
            "schema_version": 1,
            "application_version": APP_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": self.mode_var.get(),
            "table": self.table_var.get(),
            "plot_type": self.plot_type_var.get(),
            "x_column": self.x_column_var.get(),
            "y_column": self.y_column_var.get(),
            "code": self.code_text.get("1.0", tk.END).strip(),
            "ai_recipe": self.ai_recipe,
            "context_schema": self.context_schema,
        }

    def save_plot_recipe(self):
        if not self.validate_editor_code(show_message=False):
            messagebox.showerror(
                "Plot Code Invalid",
                self.status_var.get(),
                parent=self,
            )
            return
        save_path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Reusable Plot Recipe",
            defaultextension=".json",
            filetypes=[("JSON Plot Recipe", "*.json")],
        )
        if not save_path:
            return
        try:
            Path(save_path).write_text(
                json.dumps(
                    self.current_recipe(),
                    indent=2,
                    default=_result_json_default,
                ),
                encoding="utf-8",
            )
            messagebox.showinfo(
                "Plot Recipe Saved",
                f"The reusable plot recipe was saved:\n{save_path}",
                parent=self,
            )
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc), parent=self)

    def materialize_custom_result(self, destination):
        if not self.current_output_directory:
            raise ValueError("Run a plot before exporting it.")
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        source = Path(self.current_output_directory)
        for name in ("plot.png", "plot.svg", "plot_code.py"):
            shutil.copy2(source / name, destination / name)
        snapshot = self.completed_plot_snapshot
        recipe = snapshot["recipe"]
        context = snapshot["context"]
        table_name = recipe["table"]
        # Preserve every table used by multi-table AI recipes, not just the selector.
        for path in source.glob("table_*.csv"):
            shutil.copy2(path, destination / path.name)
        shutil.copy2(source / "context.json", destination / "context.json")
        selected_table = context["tables"][table_name]
        selected_table.to_csv(destination / "plot_data.csv", index=False)
        (destination / "plot_recipe.json").write_text(
            json.dumps(recipe, indent=2, default=_result_json_default),
            encoding="utf-8",
        )
        explanation = snapshot["explanation"]
        (destination / "ai_explanation.txt").write_text(
            explanation,
            encoding="utf-8",
        )
        (destination / "result_metadata.json").write_text(
            json.dumps(
                {
                    "application_version": APP_VERSION,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "title": context["title"],
                    "task_type": context["task_type"],
                    "model_type": context["model_type"],
                    "source": context["source"],
                    "selected_table": table_name,
                    "metrics": context["metrics"],
                },
                indent=2,
                default=_result_json_default,
            ),
            encoding="utf-8",
        )
        return str(destination)

    def save_custom_result_bundle(self):
        if not self.current_output_directory:
            return
        save_path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Custom Result Bundle",
            defaultextension=".zip",
            initialfile="custom_model_result.zip",
            filetypes=[("ZIP Result Bundle", "*.zip")],
        )
        if not save_path:
            return
        temp_directory = tempfile.mkdtemp(prefix="nn_custom_export_")
        try:
            self.materialize_custom_result(temp_directory)
            zip_result_directory(temp_directory, save_path)
            messagebox.showinfo(
                "Custom Result Saved",
                "The plot, SVG, source data, code, recipe, explanation, and "
                f"metadata were saved:\n{save_path}",
                parent=self,
            )
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc), parent=self)
        finally:
            shutil.rmtree(temp_directory, ignore_errors=True)

    def attach_to_complete_results(self):
        if not self.current_output_directory or self.attach_callback is None:
            return
        transfer_directory = tempfile.mkdtemp(
            prefix="nn_custom_result_attach_"
        )
        try:
            self.materialize_custom_result(transfer_directory)
            self.attach_callback(
                transfer_directory,
                self.completed_plot_snapshot["recipe"],
            )
            messagebox.showinfo(
                "Custom Result Added",
                "This visualization will be included the next time you save "
                "the complete training, evaluation, or detection results.",
                parent=self,
            )
        except Exception as exc:
            messagebox.showerror("Attach Error", str(exc), parent=self)
        finally:
            shutil.rmtree(transfer_directory, ignore_errors=True)

    def open_current_png(self):
        if not self.current_output_directory:
            messagebox.showinfo(
                "No Plot",
                "Run a plot before opening its PNG.",
                parent=self,
            )
            return
        path = str(Path(self.current_output_directory) / "plot.png")
        try:
            if platform.system().lower() == "windows":
                os.startfile(path)
            elif platform.system().lower() == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            messagebox.showerror("Open Plot Error", str(exc), parent=self)

    def _release_result_resources(self):
        callback = getattr(self, "_poll_after_id", None)
        if callback is not None:
            self.after_cancel(callback)
            self._poll_after_id = None
        with self._worker_lock:
            self._closed = True
            while not self._worker_results.empty():
                kind, payload = self._worker_results.get_nowait()
                if kind == "plot_done":
                    shutil.rmtree(payload, ignore_errors=True)
        if self.current_output_directory:
            shutil.rmtree(
                self.current_output_directory,
                ignore_errors=True,
            )
        self.current_output_directory = None

    def close_window(self):
        self._release_result_resources()
        self.destroy()

    def _on_studio_destroy(self, event):
        if event.widget is self:
            self._release_result_resources()

    def on_plot_type_selected(self, _event=None):
        if self.context is None:
            return
        plot = self.plot_type_var.get()
        table = {"Training Curves": "training_history", "Loss & Accuracy": "training_history",
                 "Confusion Matrix": "results", "Correlation Heatmap": "embedding_features",
                 "t-SNE": "embedding_features"}.get(plot)
        if table in self.context["tables"]:
            self.table_var.set(table)
        self.refresh_column_choices()
        if plot == "Confusion Matrix":
            columns = self.context["tables"][self.table_var.get()].columns
            if "actual_label" in columns and "predicted_label" in columns:
                self.x_column_var.set("actual_label")
                self.y_column_var.set("predicted_label")
        self.insert_builtin_template()

    def _deliver_worker_result(self, kind, payload):
        with self._worker_lock:
            if self._closed:
                if kind == "plot_done":
                    shutil.rmtree(payload, ignore_errors=True)
            else:
                self._worker_results.put((kind, payload))

    def _poll_workers(self):
        if self._closed:
            return
        handlers = {"ai_done": self.finish_ai_generation, "ai_failed": self.fail_ai_generation,
                    "plot_done": self.finish_plot_execution, "plot_failed": self.fail_plot_execution}
        while not self._worker_results.empty():
            kind, payload = self._worker_results.get_nowait()
            handlers[kind](payload)
        self._poll_after_id = self.after(100, self._poll_workers)
