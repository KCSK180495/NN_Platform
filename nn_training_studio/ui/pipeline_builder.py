"""Build, fit, preview, and export a record-based preprocessing pipeline."""

import json
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
import tkinter as tk
from tkinter import filedialog, font, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from zipfile import ZipFile

from ..pipeline_ai import draft_recipe
from ..pipeline_package import export_pipeline, predict, read_csv, save_output
from ..processing_pipeline import ARITHMETIC, OPERATIONS, RECORD, SCHEMA, Pipeline, validate_recipe


DRAFT_GOAL_LIMIT = 3000


class PipelineDraftDialog(tk.Toplevel):
    """Collect a multi-line request with examples tied to the recipe inputs."""

    def __init__(self, parent, inputs, sample_rate, initial_goal=""):
        super().__init__(parent)
        self.withdraw()
        self.transient(parent)
        self.title("Draft a pipeline with AI")
        self.result = None

        body = ttk.Frame(self, padding=18)
        body.grid(sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(4, weight=1)

        heading = font.nametofont("TkDefaultFont", root=self).copy()
        heading.configure(size=15, weight="bold")
        self.heading_font = heading
        ttk.Label(body, text="What should the pipeline do?", font=heading).grid(sticky="w")
        description = ttk.Label(
            body, text="Describe which signals to process, which filters to use, and what features to keep.",
            justify=tk.LEFT,
        )
        description.grid(row=1, sticky="ew", pady=(4, 12))

        columns = list(inputs.items())
        shown = ", ".join(f"{name[:48]} ({unit[:16]})" for name, unit in columns[:4])
        if len(columns) > 4:
            shown += f" and {len(columns) - 4} more"
        rate = "Sample rate unspecified" if sample_rate is None else f"Sample rate {sample_rate:g} Hz"
        context = ttk.Label(body, text=f"Inputs  {shown}\n{rate}", justify=tk.LEFT)
        context.grid(row=2, sticky="ew", pady=(0, 12))

        examples = ttk.Frame(body)
        examples.grid(row=3, sticky="w", pady=(0, 8))
        ttk.Label(examples, text="Insert an example").pack(side=tk.LEFT, padx=(0, 8))
        first = columns[0][0]
        prompts = (
            ("Smooth a signal", f"Smooth {first!r} with an exponential moving average over 15 samples. "
                                "Keep both the original and smoothed values."),
            ("RMS and scaling", f"Compute a trailing 100-sample RMS of {first!r}. "
                                "Apply standard scaling to the RMS and keep both outputs."),
            ("Sample changes", f"Compute the change in {first!r} between consecutive samples. "
                               "Keep the original signal and the difference."),
        )
        for label, prompt in prompts:
            ttk.Button(examples, text=label, command=lambda value=prompt: self._insert_example(value)).pack(
                side=tk.LEFT, padx=(0, 4),
            )

        self.editor_font = font.nametofont("TkTextFont", root=self).copy()
        self.editor_font.configure(size=max(12, self.editor_font.cget("size")))
        self.prompt = ScrolledText(body, width=1, height=6, wrap=tk.WORD, undo=True,
                                   font=self.editor_font, padx=8, pady=8, spacing1=2, spacing3=2)
        self.prompt.grid(row=4, sticky="nsew")
        self.prompt.insert("1.0", initial_goal)
        self.prompt.mark_set("insert", "end-1c")
        self.prompt.bind("<<Modified>>", self._refresh_count)
        self.prompt.bind("<Tab>", lambda event: self._move_focus(event.widget.tk_focusNext()))
        self.prompt.bind("<Shift-Tab>", lambda event: self._move_focus(event.widget.tk_focusPrev()))

        footer = ttk.Frame(body)
        footer.grid(row=5, sticky="ew", pady=(12, 0))
        footer.columnconfigure(0, weight=1)
        self.counter = ttk.Label(footer)
        self.counter.grid(row=0, column=0, sticky="w")
        ttk.Button(footer, text="Cancel", command=self.destroy).grid(row=0, column=1, padx=(8, 6))
        self.submit_button = ttk.Button(footer, text="Generate draft", command=self._submit)
        self.submit_button.grid(row=0, column=2, sticky="e")
        ttk.Label(body, text="Ctrl+Enter generates the draft.").grid(row=6, sticky="w", pady=(6, 0))

        def wrap_labels(event):
            for label in (description, context):
                label.configure(wraplength=max(200, event.width - 36))

        body.bind("<Configure>", wrap_labels)
        self.bind("<Control-Return>", self._submit)
        self.bind("<Escape>", self._cancel)
        self._refresh_count()
        self._place_over(parent)
        self.deiconify()
        self.wait_visibility()
        self.grab_set()
        self.prompt.focus_set()

    def _place_over(self, parent):
        parent.update_idletasks()
        screen_width, screen_height = self.winfo_screenwidth(), self.winfo_screenheight()
        width, height = min(760, screen_width - 32), min(530, screen_height - 80)
        x = parent.winfo_rootx() + (parent.winfo_width() - width) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - height) // 2
        x = max(0, min(x, screen_width - width))
        y = max(0, min(y, screen_height - height - 48))
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(min(560, width), min(405, height))

    def _text(self):
        return self.prompt.get("1.0", "end-1c")

    def _refresh_count(self, event=None):
        if event is not None and not self.prompt.edit_modified():
            return
        text = self._text()
        count = len(text.strip())
        over_limit = count > DRAFT_GOAL_LIMIT
        self.counter.configure(
            text=f"{count:,} / {DRAFT_GOAL_LIMIT:,} characters",
            foreground="#b42318" if over_limit else "",
        )
        self.submit_button.state(["disabled" if not count or over_limit else "!disabled"])
        self.prompt.edit_modified(False)

    def _insert_example(self, example):
        if self._text().strip():
            self.prompt.insert("end-1c", "\n\n")
        self.prompt.insert("end-1c", example)
        self.prompt.mark_set("insert", "end-1c")
        self.prompt.see("insert")
        self.prompt.focus_set()
        self._refresh_count()

    @staticmethod
    def _move_focus(widget):
        widget.focus_set()
        return "break"

    def _submit(self, event=None):
        goal = self._text().strip()
        if 0 < len(goal) <= DRAFT_GOAL_LIMIT:
            self.result = goal
            self.destroy()
        return "break"

    def _cancel(self, event):
        self.destroy()
        return "break"


class PipelineBuilder(tk.Toplevel):
    def __init__(self, owner, initial_path=None):
        super().__init__(owner)
        self.owner = owner
        self.title("NN Studio | Processing Pipeline")
        self.geometry("1060x770")
        self.minsize(760, 570)
        self.data = tk.StringVar(self)
        self.record = tk.StringVar(self)
        self.label = tk.StringVar(self)
        self.trust = tk.BooleanVar(self)
        self.healthy = tk.BooleanVar(self)
        self.operation = tk.StringVar(self, "butterworth")
        self.status = tk.StringVar(self, "Load a CSV or open an existing pipeline.")
        self.pipeline = None
        self.recipe_snapshot = None
        self.sources = {}
        self.fit_data_path = None
        self.model_path = None
        self.last_ai_goal = ""
        self._build()
        self._set_recipe(self._template("signal"))
        if initial_path:
            self._act(self._load_recipe, initial_path)

    @staticmethod
    def _template(column):
        return {"schema": SCHEMA, "name": "Processing pipeline", "sample_rate_hz": None,
                "inputs": {column: "1"}, "steps": [], "outputs": [column]}

    def _build(self):
        body = ttk.Frame(self, padding=12)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(1, weight=1)
        source = ttk.LabelFrame(body, text="Recording data", padding=8)
        source.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        source.columnconfigure(1, weight=1)
        ttk.Button(source, text="Load CSV…", command=lambda: self._act(self._load_csv)).grid(row=0, column=0)
        ttk.Label(source, textvariable=self.data, wraplength=610).grid(row=0, column=1, columnspan=3, sticky="ew", padx=8)
        for index, (title, variable) in enumerate((("Recording ID column", self.record), ("Label column", self.label))):
            ttk.Label(source, text=title).grid(row=1, column=2 * index, sticky="w", padx=(0, 8), pady=(6, 0))
            combo = ttk.Combobox(source, textvariable=variable, state="readonly", values=("",), width=18)
            combo.grid(row=1, column=2 * index + 1, sticky="ew", padx=(0, 12), pady=(6, 0))
            if index == 0:
                self.record_combo = combo
            else:
                self.label_combo = combo

        design = ttk.LabelFrame(body, text="Recipe", padding=8)
        design.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        row = ttk.Frame(design)
        row.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(row, text="Open JSON or ZIP…", command=lambda: self._act(self._open_recipe)).pack(side=tk.LEFT)
        self.ai_button = ttk.Button(row, text="Draft with AI…", command=lambda: self._act(self._draft))
        self.ai_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="AI Settings", command=self._settings).pack(side=tk.LEFT)
        row = ttk.Frame(design)
        row.pack(fill=tk.X, pady=(0, 6))
        ttk.Combobox(row, textvariable=self.operation, values=tuple(OPERATIONS),
                     state="readonly", width=16).pack(side=tk.LEFT)
        ttk.Button(row, text="Add step", command=lambda: self._act(self._add_step)).pack(side=tk.LEFT, padx=4)
        ttk.Button(row, text="Attach Python…", command=lambda: self._act(self._attach_code)).pack(side=tk.LEFT)
        self.editor = ScrolledText(design, width=42, height=12, wrap=tk.NONE, undo=True)
        self.editor.pack(fill=tk.BOTH, expand=True)

        preview = ttk.LabelFrame(body, text="Preview", padding=8)
        preview.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
        row = ttk.Frame(preview)
        row.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(row, text="Fit from training CSV…", command=lambda: self._act(self._fit)).pack(side=tk.LEFT)
        ttk.Button(row, text="Compute", command=lambda: self._act(self._preview)).pack(side=tk.LEFT, padx=4)
        self.preview = ScrolledText(preview, width=34, height=12, wrap=tk.NONE, state=tk.DISABLED)
        self.preview.pack(fill=tk.BOTH, expand=True)
        ttk.Checkbutton(preview, text="Allow trusted Python and model scaler", variable=self.trust).pack(anchor=tk.W)
        ttk.Checkbutton(preview, text="Known healthy startup (Scope v1)", variable=self.healthy).pack(anchor=tk.W)
        ttk.Button(preview, text="Predict with loaded model", command=lambda: self._act(self._predict)).pack(anchor=tk.W)

        exports = ttk.Frame(body)
        exports.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        actions = (
            ("Save features…", self._save_features),
            ("Export runnable ZIP…", self._export),
            ("Export ZIP with model…", lambda: self._export(with_model=True)),
            ("Continue to training", self._train),
        )
        for title, command in actions:
            ttk.Button(exports, text=title, command=lambda cb=command: self._act(cb)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Label(body, textvariable=self.status, foreground="#245a85", wraplength=710).grid(
            row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _act(self, callback, *args):
        try:
            callback(*args)
        except Exception as exc:
            self.status.set(str(exc))
            messagebox.showerror("Processing Pipeline", str(exc), parent=self)

    def _recipe(self):
        return json.loads(self.editor.get("1.0", "end-1c"))

    def _set_recipe(self, recipe):
        self.editor.delete("1.0", tk.END)
        self.editor.insert("1.0", json.dumps(recipe, indent=2, ensure_ascii=False))

    def _pipeline(self):
        recipe = self._recipe()
        if self.pipeline is not None and recipe == self.recipe_snapshot:
            return self.pipeline
        recipe = validate_recipe(recipe)
        files = {step["params"]["file"] for step in recipe["steps"] if step["op"] == "python"}
        pipeline = Pipeline.build(recipe, {name: self.sources[name] for name in files if name in self.sources})
        self.pipeline = pipeline
        self.recipe_snapshot = recipe
        self.fit_data_path = None
        self._set_recipe(recipe)
        return pipeline

    def _settings(self):
        self.owner.show_settings()
        self.lower(self.owner)

    def _load_csv(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("CSV data", "*.csv")])
        if not path:
            return
        import pandas as pd

        columns = list(pd.read_csv(path, nrows=0))
        if not columns:
            raise ValueError("CSV contains no columns.")
        self.data.set(path)
        for widget, variable in ((self.record_combo, self.record), (self.label_combo, self.label)):
            widget.configure(values=("", *columns))
            if variable.get() not in columns:
                variable.set("")
        if RECORD in columns:
            self.record.set(RECORD)
        if not self.label.get():
            self.label.set(next((name for name in columns if name.lower() in ("label", "class", "target")), ""))
        if self.pipeline is None and self._recipe() == self._template("signal") and "signal" not in columns:
            first = next((name for name in columns if name not in (self.label.get(), self.record.get())), columns[0])
            self._set_recipe(self._template(first))
        self.status.set("Describe the inputs and pipeline steps. Select a recording ID column for multiple recordings.")

    def _open_recipe(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("Pipeline recipe or package", "*.json *.zip")])
        if path:
            self._load_recipe(path)

    def _load_recipe(self, path):
        pipeline = Pipeline.read(path)
        model = None
        if Path(path).suffix.lower() == ".zip":
            with ZipFile(path) as archive:
                if "model.keras" in archive.namelist():
                    model = path
        self.pipeline = pipeline
        self.recipe_snapshot = pipeline.recipe
        self.sources = dict(pipeline.sources)
        self.fit_data_path = None
        self.model_path = model
        self.trust.set(False)
        self.healthy.set(False)
        self._set_recipe(pipeline.recipe)
        self.status.set("Scope v1 recipe loaded with its original parameters." if pipeline.legacy
                        else "Recipe and saved parameters loaded.")

    def _add_step(self):
        recipe = self._recipe()
        if recipe.get("schema") != SCHEMA:
            raise ValueError("The original Scope v1 recipe is preserved. Open a Studio v2 recipe to edit steps.")
        op = self.operation.get()
        output = f"{op}_{len(recipe['steps']) + 1}"
        source = [recipe["outputs"][-1]]
        if op in ARITHMETIC or op == "sscn":
            source.append("speed_rpm" if op == "sscn" else "second_signal")
        recipe["steps"].append({"op": op, "inputs": source, "outputs": [output], "params": OPERATIONS[op]})
        recipe["outputs"].append(output)
        self._set_recipe(recipe)
        self.status.set("Review the new step's input columns, output names, and settings before computing.")

    def _attach_code(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("Python source", "*.py")])
        if path:
            self.sources[Path(path).name] = Path(path).read_bytes()
            self.recipe_snapshot = None
            self.trust.set(False)
            self.status.set(f"Attached {Path(path).name}. Use this filename in a Python step.")

    def _draft(self):
        recipe = self._recipe()
        if recipe.get("schema") != SCHEMA:
            raise ValueError("Use a Studio v2 recipe for AI drafting.")
        inputs = recipe.get("inputs")
        seed = validate_recipe({**recipe, "steps": [], "outputs": list(inputs or {})})
        if len(inputs) > 50:
            raise ValueError("Limit the draft to 50 input columns.")
        dialog = PipelineDraftDialog(self, inputs, seed["sample_rate_hz"], self.last_ai_goal)
        self.wait_window(dialog)
        self.focus_set()
        goal = dialog.result
        if not goal:
            return
        self.last_ai_goal = goal
        selected = self.owner.get_component_ai_provider()
        if not selected:
            return
        provider, provider_name, _, _ = selected
        self.ai_button.state(["disabled"])
        self.status.set(f"Waiting for a recipe draft from {provider_name}.")
        result = Queue()

        def work():
            try:
                result.put((True, draft_recipe(provider, provider_name, goal, inputs, seed["sample_rate_hz"])))
            except Exception as exc:
                result.put((False, str(exc).replace(provider.api_key, "[redacted]")))

        def poll():
            try:
                success, value = result.get_nowait()
            except Empty:
                self.after(150, poll)
                return
            self.ai_button.state(["!disabled"])
            if not success:
                self.status.set(value)
            elif self.editor.get("1.0", "end-1c") != draft_start and not messagebox.askyesno(
                "Use AI draft", "Replace the recipe you edited while the draft was being generated?", parent=self
            ):
                self.status.set("Kept the current recipe.")
            else:
                self._set_recipe(value)
                self.status.set("Review the AI draft, then compute a local preview.")

        draft_start = self.editor.get("1.0", "end-1c")
        Thread(target=work, daemon=True).start()
        self.after(150, poll)

    def _fit(self):
        pipeline = self._pipeline()
        path = filedialog.askopenfilename(
            parent=self, title="Choose training or calibration records", filetypes=[("CSV data", "*.csv")])
        if not path:
            return
        if self.data.get() and Path(path).resolve() == Path(self.data.get()).resolve():
            raise ValueError("Use separate training/calibration and preview CSV files.")
        record = self.record.get() or None
        self.pipeline = pipeline.fit(read_csv(path, record), record, trust_code=self.trust.get())
        self.fit_data_path = Path(path).resolve()
        self.status.set("Fitted transforms saved in memory. Compute to preview them on your recording data.")

    def _compute(self, with_labels=True):
        pipeline = self._pipeline()
        path = self.data.get()
        if not path:
            raise ValueError("Load a recording CSV first.")
        record = self.record.get() or None
        label = (self.label.get() or None) if with_labels else None
        data = read_csv(path, record, label)
        return pipeline, pipeline.transform(data, record, label, trust_code=self.trust.get(), healthy=self.healthy.get())

    def _show_frame(self, frame, description):
        self.preview.configure(state=tk.NORMAL)
        self.preview.delete("1.0", tk.END)
        self.preview.insert("1.0", frame.head(14).to_string(index=False))
        self.preview.configure(state=tk.DISABLED)
        self.status.set(f"{len(frame):,} {description} across {frame[RECORD].nunique()} recording(s).")

    def _preview(self):
        _, frame = self._compute()
        self._show_frame(frame, "feature rows")

    def _predict(self):
        if not self.model_path:
            raise ValueError("Open a pipeline ZIP that contains its bound model first.")
        pipeline, frame = self._compute(with_labels=False)
        result = predict(pipeline, frame, self.model_path, trust_artifacts=self.trust.get())
        self._show_frame(result, "predictions")
        self._save_frame(result, "Save predictions")

    def _save_frame(self, frame, title):
        path = filedialog.asksaveasfilename(parent=self, title=title, defaultextension=".csv",
                                           filetypes=[("CSV table", "*.csv"), ("NumPy arrays", "*.npz")])
        if path:
            if Path(path).resolve() == Path(self.data.get()).resolve():
                raise ValueError("Choose a different filename from the input recording.")
            save_output(path, frame)
            self.status.set(f"Saved {len(frame):,} rows to {Path(path).name}.")

    def _save_features(self):
        _, frame = self._compute()
        self._save_frame(frame, "Save computed features")

    def _export(self, with_model=False):
        pipeline = self._pipeline()
        model = None
        if with_model:
            previous = self.model_path or self.owner.last_saved_model_package_path
            options = ({"initialdir": str(Path(previous).parent), "initialfile": Path(previous).name}
                       if previous else {})
            model = filedialog.askopenfilename(parent=self, title="Choose the model trained with this pipeline",
                                               filetypes=[("NN Studio or NN Scope model ZIP", "*.zip")], **options)
            if not model:
                return
        path = filedialog.asksaveasfilename(parent=self, title="Export runnable pipeline", defaultextension=".zip",
                                           filetypes=[("Pipeline ZIP", "*.zip")])
        if path:
            export_pipeline(path, pipeline, model_source=model)
            self.status.set(f"Saved {Path(path).name}. Extract it and run run_pipeline.py.")

    def _train(self):
        if self.owner._operation_busy():
            raise ValueError("Finish the current Studio operation before opening training.")
        pipeline, frame = self._compute()
        if pipeline.legacy:
            raise ValueError("Use a Studio v2 recipe when training a new classifier.")
        label = self.label.get()
        if not label or frame[label].isna().any() or frame[label].nunique() < 2:
            raise ValueError("Select a complete label column with at least two classes.")
        if frame[RECORD].nunique() < 3:
            raise ValueError("Select at least three recording IDs for separate training, validation and test sets.")
        if self.fit_data_path and self.fit_data_path == Path(self.data.get()).resolve():
            raise ValueError("Use separate training/calibration and evaluation recordings.")
        if self.owner.df is not None and not messagebox.askyesno(
            "Start pipeline training", "Replace the current in-memory project with these features?", parent=self
        ):
            return
        self.owner.start_pipeline_training(pipeline, frame, label, self.data.get())
        # Keep the fitted recipe available when returning from training.
        self.withdraw()
