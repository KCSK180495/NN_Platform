"""Platform window for NN Scope playback, acquisition and deployment export."""

from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..scope_runtime.capture_analysis import save_capture
from ..scope_runtime.csv_capture import capture_metadata
from ..scope_runtime.export import export_session, load_session
from ..scope_runtime.package_io import inspect_package, sha256
from ..scope_runtime.scope_types import AcquisitionCancelled, AcquisitionTimeout, Channel, validate_channels
from ..scope_runtime.session import ScopeRecording, ScopeWorker
from .scrollable import ScrollableForm


class ScopeDeploymentWindow(tk.Toplevel):
    def __init__(self, parent, initial_model_path=None, session_path=None):
        super().__init__(parent)
        self.title("NN Platform · NN Scope")
        self.geometry("1020x740")
        self.minsize(660, 500)
        self.worker = ScopeWorker()
        self.future = self.poll_id = self.tick_id = None
        self.capture = self.result = self.capabilities = None
        self.running = self.paused = self.closed = False
        self.live = self.recording = None
        self.live_count = 0
        self.position = 0
        self.tasks = []
        self.editing_id = None
        self.csv_channels = ()
        self.csv_rate = None
        self.editor = None
        self.input_fields = {}
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        self.bind("<Destroy>", self._on_destroy, add="+")
        self._build()
        if session_path:
            self._restore(session_path)
        elif initial_model_path:
            self.model.set(str(initial_model_path))
            self._inspect()
            self.status.set("Choose a data source, then check the model inputs below.")

    def _build(self):
        header = ttk.Frame(self, padding=(14, 10, 14, 4))
        header.pack(fill="x")
        ttk.Label(header, text="NN Scope", font=("TkDefaultFont", 16, "bold")).pack(anchor="w")
        bar = ttk.Frame(header, padding=(0, 7, 0, 0))
        bar.pack(fill="x")
        self.setup_button = ttk.Button(bar, text="Back to setup", command=lambda: self.tabs.select(self.setup))
        self.mode = tk.StringVar(self, "CSV playback")
        mode = ttk.Combobox(bar, textvariable=self.mode, values=("CSV playback", "Oscilloscope"),
                            state="readonly", width=17)
        mode.pack(side="left", padx=(0, 8))
        mode.bind("<<ComboboxSelected>>", self._source_changed)
        self.mode_box = mode
        self.run_button = ttk.Button(bar, text="Run diagnosis", command=self.start)
        self.run_button.pack(side="left", padx=(0, 5))
        self.preview_button = ttk.Button(bar, text="Preview waveform", command=lambda: self.start(waveform_only=True))
        self.preview_button.pack(side="left")
        self.live_button = ttk.Button(bar, text="Live view",
                                      command=lambda: self.start(waveform_only=True, live=True))
        self.record_button = ttk.Button(bar, text="Record…", command=self._record)
        self.pause_button = ttk.Button(bar, text="Pause", command=self.pause, state="disabled")
        self.stop_button = ttk.Button(bar, text="Stop", command=self.stop)
        self.healthy = tk.BooleanVar(self)

        footer = ttk.Frame(self, padding=(8, 7))
        footer.pack(side="bottom", fill="x")
        self.status = tk.StringVar(self, "Choose a capture to get started.")
        status = ttk.Label(footer, textvariable=self.status, wraplength=660)
        status.pack(side="left", fill="x", expand=True)
        footer.bind("<Configure>", lambda event: status.configure(wraplength=max(250, event.width - 290)))
        self.save_button = ttk.Button(footer, text="Save results…", command=self._save_capture, state="disabled")
        self.save_button.pack(side="right", padx=(4, 0))
        self.export_button = ttk.Button(footer, text="Export deployment…", command=self._export)
        self.export_button.pack(side="right")
        self.tabs = ttk.Notebook(self)
        self.tabs.pack(fill="both", expand=True, padx=8)
        self.setup = ScrollableForm(self.tabs)
        self.monitor = ttk.Frame(self.tabs, padding=8)
        self.tabs.add(self.setup, text="Setup")
        self.tabs.add(self.monitor, text="Monitor")
        self.source_panel = ttk.Frame(self.setup.content)
        self.source_panel.pack(fill="x")
        self._build_source(self.source_panel)
        self._build_models(self.setup.content)
        self._build_monitor()
        self._source_changed()
        self.tabs.bind("<<NotebookTabChanged>>", self._update_actions)
        self._update_actions()

    def _field(self, frame, row, label, value="", values=None):
        variable = tk.StringVar(self, value)
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", padx=7, pady=4)
        widget = (ttk.Entry(frame, textvariable=variable) if values is None else
                  ttk.Combobox(frame, textvariable=variable, values=values, state="readonly"))
        widget.grid(row=row, column=1, sticky="ew", padx=5, pady=4)
        frame.columnconfigure(1, weight=1)
        return variable

    def _build_source(self, parent):
        self.csv_box = ttk.LabelFrame(parent, text="Capture file")
        self.csv = tk.StringVar(self)
        ttk.Button(self.csv_box, text="Choose capture…", command=self._choose_csv).grid(
            row=0, column=0, sticky="nw", padx=8, pady=7)
        self.csv_info = tk.StringVar(self, "Choose a saved NN Scope CSV and keep its metadata.json beside it.")
        self.capture_label = ttk.Label(self.csv_box, textvariable=self.csv_info, wraplength=485, justify="left")
        self.capture_label.grid(row=0, column=1, sticky="w", padx=4, pady=7)
        self.csv_box.columnconfigure(1, weight=1)

        self.scope_box = ttk.LabelFrame(parent, text="Oscilloscope settings")
        self.resource = tk.StringVar(self)
        ttk.Label(self.scope_box, text="Instrument address").grid(row=0, column=0, sticky="w", padx=7, pady=4)
        self.resources = ttk.Combobox(self.scope_box, textvariable=self.resource, width=28)
        self.resources.grid(row=0, column=1, sticky="ew", padx=5, pady=4)
        self.backend = self._field(self.scope_box, 1, "VISA backend", "@ivi", ("@ivi", "@py"))
        controls = ttk.Frame(self.scope_box)
        controls.grid(row=2, columnspan=2, sticky="w", padx=5, pady=5)
        for text, command in (("Scan USB", self._scan), ("Connect", self._connect), ("Disconnect", self._disconnect)):
            ttk.Button(controls, text=text, command=command).pack(side="left", padx=3)
        self.connection = tk.StringVar(self, "Disconnected")
        ttk.Label(self.scope_box, textvariable=self.connection, wraplength=640).grid(
            row=3, columnspan=2, sticky="w", padx=8, pady=5)
        self.capture_options = ttk.Frame(self.scope_box)
        self.capture_options_button = ttk.Button(
            self.scope_box, text="Show capture settings", command=self._toggle_capture_options)
        self.capture_options_button.grid(row=4, column=0, sticky="w", padx=8, pady=5)
        self.timeout = self._field(self.capture_options, 0, "Trigger timeout (s)", "10")
        self.time_scale = self._field(self.capture_options, 1, "Timebase (s/div)", "0.01")
        self.depth = self._field(self.capture_options, 2, "Record depth", "100000", ("10000", "100000", "1000000"))
        self.configure_scope = tk.BooleanVar(self)
        self.force = tk.BooleanVar(self)
        ttk.Checkbutton(self.capture_options, text="Set timebase and depth on DHO814",
                        variable=self.configure_scope).grid(row=3, columnspan=2, sticky="w", padx=8)
        self.force_button = ttk.Checkbutton(self.scope_box, text="Force trigger while waiting",
                                            variable=self.force)

        self.channel_box = ttk.LabelFrame(parent, text="Physical channel calibration")
        self.channel_vars = []
        for index, label in enumerate(("Enable", "Channel name", "Unit", "Gain", "Sensor zero", "Offset")):
            ttk.Label(self.channel_box, text=label).grid(row=0, column=index, padx=5, pady=4)
        for number in range(1, 5):
            enabled = tk.BooleanVar(self, number <= 3)
            ttk.Checkbutton(self.channel_box, text=f"CH{number}", variable=enabled).grid(
                row=number, column=0, sticky="w", padx=5)
            values = [tk.StringVar(self, value) for value in (f"CH{number}", "V", "1", "0", "0")]
            for col, var in enumerate(values, 1):
                ttk.Entry(self.channel_box, textvariable=var, width=12).grid(row=number, column=col, padx=4, pady=3)
            self.channel_vars.append((enabled, values))

    def _toggle_capture_options(self):
        visible = bool(self.capture_options.winfo_manager())
        self.capture_options_button.configure(text="Show capture settings" if visible else "Hide capture settings")
        if visible:
            self.capture_options.grid_remove()
        else:
            self.capture_options.grid(row=5, columnspan=2, sticky="ew", padx=4, pady=5)

    def _build_models(self, parent):
        editor = ttk.LabelFrame(parent, text="Diagnosis model")
        editor.pack(fill="x", pady=5, padx=6)
        self.model = tk.StringVar(self)
        self.model_info = tk.StringVar(self, "Choose a trusted model ZIP to see predictions.")
        ttk.Button(editor, text="Choose model…", command=lambda: self._browse_model(self.model)).grid(
            row=0, column=0, sticky="nw", padx=8, pady=7)
        ttk.Label(editor, textvariable=self.model_info, wraplength=490, justify="left").grid(
            row=0, column=1, sticky="w", padx=4, pady=7)
        self.inputs_box = ttk.LabelFrame(parent, text="Map calibrated inputs")
        self.extras = ttk.Frame(parent, padding=(6, 3))
        self.extras.pack(fill="x", padx=6)
        self.advanced_button = ttk.Button(self.extras, text="Advanced options", command=self._toggle_advanced)
        self.advanced_button.pack(anchor="w", pady=(0, 4))
        self.advanced = ttk.LabelFrame(self.extras, text="Preprocessing and multiple diagnoses")
        self.profile = self._field(self.advanced, 0, "Profile override")
        self.advanced.grid_slaves(row=0, column=1)[0].configure(state="readonly")
        ttk.Button(self.advanced, text="Choose profile…", command=lambda: self._browse_model(self.profile)).grid(
            row=0, column=2, padx=4)
        ttk.Button(self.advanced, text="Use model profile", command=self._clear_profile).grid(
            row=0, column=3, padx=(0, 6))
        actions = ttk.Frame(self.advanced)
        actions.grid(row=1, column=0, columnspan=4, sticky="w", padx=7, pady=5)
        for label, command in (("Save diagnosis", self._apply_task), ("Add another model…", self._add_another),
                               ("Edit selected", self._edit_task), ("Remove selected", self._remove_task)):
            ttk.Button(actions, text=label, command=command).pack(side="left", padx=(0, 4))
        self.task_list = ttk.Treeview(self.advanced, columns=("model", "mapping"), show="headings", height=4)
        self.task_list.heading("model", text="Diagnosis")
        self.task_list.heading("mapping", text="Input mapping")
        self.task_list.grid(row=2, column=0, columnspan=4, sticky="ew", padx=6, pady=5)
        self.task_list.bind("<Double-1>", self._edit_task)

    def _toggle_advanced(self):
        visible = bool(self.advanced.winfo_manager())
        if visible:
            self.advanced.pack_forget()
        else:
            self.advanced.pack(fill="x")
        self._update_advanced_label()

    def _update_advanced_label(self):
        count = f" ({len(self.tasks)} saved)" if self.tasks else ""
        label = "Hide advanced options" if self.advanced.winfo_manager() else "Advanced options"
        self.advanced_button.configure(text=label + count)

    def _build_monitor(self):
        self.monitor.columnconfigure(0, weight=1)
        self.monitor.rowconfigure(2, weight=1)
        summary = ttk.Frame(self.monitor)
        summary.grid(row=0, column=0, sticky="ew")
        self.run_info = tk.StringVar(self, "No capture yet")
        ttk.Label(summary, textvariable=self.run_info).pack(side="left")
        speed = ttk.Frame(summary)
        speed.pack(side="right")
        self.play_speed = self._field(speed, 0, "Playback speed", "0.25×", ("0.1×", "0.25×", "1×"))
        self.speed_controls = speed
        self.progress = ttk.Progressbar(self.monitor, mode="determinate", maximum=100)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        figure = Figure(figsize=(7, 3), layout="tight")
        self.axes = figure.add_subplot()
        self.axes.set(xlabel="Time (s)", ylabel="Calibrated value")
        self.chart = FigureCanvasTkAgg(figure, master=self.monitor)
        self.chart.get_tk_widget().grid(row=2, column=0, sticky="nsew")
        holder = ttk.Frame(self.monitor)
        holder.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        columns = ("diagnosis", "label", "confidence", "count")
        self.results = ttk.Treeview(holder, columns=columns, show="headings", height=4)
        for column, label, width in zip(columns, ("Diagnosis", "Latest prediction", "Confidence", "Windows"),
                                        (240, 133, 75, 65)):
            self.results.heading(column, text=label)
            self.results.column(column, width=width, stretch=column == "diagnosis")
        scroll = ttk.Scrollbar(holder, orient="vertical", command=self.results.yview)
        self.results.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.results.pack(side="left", fill="x", expand=True)
        self.results.bind("<<TreeviewSelect>>", self._show_result_detail)
        self.result_notes = {}
        self.result_detail = tk.StringVar(self, "Run a diagnosis to see its predictions here.")
        detail = ttk.Label(self.monitor, textvariable=self.result_detail, wraplength=610, justify="left")
        detail.grid(row=4, column=0, sticky="ew", pady=(5, 0))
        self.monitor.bind("<Configure>", lambda event: detail.configure(wraplength=max(300, event.width - 20)))

    def _source_changed(self, _event=None):
        self.csv_box.pack_forget()
        self.scope_box.pack_forget()
        self.channel_box.pack_forget()
        is_csv = self.mode.get() == "CSV playback"
        panels = (self.csv_box,) if is_csv else (self.scope_box, self.channel_box)
        for panel in panels:
            panel.pack(fill="x", padx=6, pady=5)
        self._refresh_fields()
        if is_csv:
            self.speed_controls.pack(side="right")
        else:
            self.speed_controls.pack_forget()
        self._update_actions()

    def _update_actions(self, _event=None):
        setup = self.tabs.select() == str(self.setup)
        self.mode_box.pack_forget()
        self.setup_button.pack_forget()
        self.run_button.pack_forget()
        self.preview_button.pack_forget()
        self.live_button.pack_forget()
        self.record_button.pack_forget()
        self.pause_button.pack_forget()
        self.stop_button.pack_forget()
        if setup:
            self.mode_box.pack(side="left", padx=(0, 8))
        else:
            self.setup_button.pack(side="left", padx=(0, 8))
        if self.running:
            if self.mode.get() == "CSV playback":
                self.pause_button.pack(side="left", padx=(0, 4))
            elif self.live and self.recording is None and not self.worker.cancel.is_set():
                self.record_button.pack(side="left", padx=(0, 5))
            self.stop_button.pack(side="left")
        else:
            is_csv = self.mode.get() == "CSV playback"
            has_model = bool(self.model.get().strip() or self.tasks)
            if has_model:
                self.run_button.configure(text="Run diagnosis" if is_csv else "Capture and diagnose")
                self.run_button.pack(side="left", padx=(0, 5))
            self.preview_button.configure(text="Preview waveform" if is_csv else "Capture waveform")
            self.preview_button.pack(side="left")
            if not is_csv:
                self.live_button.pack(side="left", padx=5)
                self.record_button.pack(side="left")

    def _choose_csv(self):
        if self.running or self.future:
            return
        path = filedialog.askopenfilename(parent=self, title="Choose a saved NN Scope capture",
                                           filetypes=[("Saved capture CSV", "*.csv")])
        if not path:
            return
        try:
            metadata = capture_metadata(path)
            if metadata.get("export_complete") is False:
                raise ValueError("Choose a capture whose export finished.")
            channels = tuple(Channel(**item) for item in metadata["channels"])
            validate_channels(channels)
            field = "native_sample_rate_hz" if Path(path).name.lower() == "native_samples.csv" else "analysis_sample_rate_hz"
            rate = float(metadata[field])
            if not np.isfinite(rate) or rate <= 0:
                raise ValueError("This capture does not have a valid sample rate in metadata.json.")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.status.set(str(exc))
            return
        if self.csv.get() and self.csv.get() != path:
            self.rpm.set("")
            self.healthy.set(False)
        self.csv.set(path)
        self.csv_channels = channels
        self.csv_rate = rate
        file = Path(path)
        self.csv_info.set(f"{file.parent.name} / {file.name}\n{rate:g} Hz · " +
                          ", ".join(f"{c.name} ({c.unit})" for c in channels))
        self._refresh_fields()
        self.status.set("Capture ready. Choose a model or preview the waveform." if not self.tasks and not self.editor
                        else "Capture ready. Check the model inputs and run diagnosis.")

    def _channels(self):
        channels = []
        for number, (enabled, fields) in enumerate(self.channel_vars, 1):
            if enabled.get():
                name, unit, *calibration = (var.get().strip() for var in fields)
                if not unit:
                    raise ValueError(f"Set the unit for CH{number}. Use '1' for dimensionless.")
                channels.append(Channel(number, name, unit, *(float(value) for value in calibration)))
        validate_channels(channels)
        return channels

    def _browse_model(self, variable):
        if self.running or self.future:
            return
        types = [("Model package", "*.zip")] if variable is self.model else [("Preprocessing profile", "*.zip *.json")]
        path = filedialog.askopenfilename(parent=self, filetypes=types, title=(
            "Choose a signal classification model" if variable is self.model else "Choose a preprocessing profile"))
        if path:
            if variable is self.model:
                self.profile.set("")
            variable.set(path)
            self._inspect()

    def _clear_profile(self):
        if not self.running and not self.future and self.model.get():
            self.profile.set("")
            self._inspect()

    def _inspect(self):
        try:
            self.editor = inspect_package(self.model.get(), self.profile.get() or None)
            meta = self.editor["metadata"]
            phase = f" · trained phase {meta['phase']}" if meta.get("phase") else ""
            limit = "\nPhysical fault validation is pending." if meta.get("physical_hardware_fault_validation") is False else ""
            if "PROBE_TEST" in Path(self.model.get()).stem.upper():
                limit += " U Open Fault detection remains unverified."
            startup = " · healthy start required" if self.editor["healthy_start_required"] else ""
            self.model_info.set(
                f"{Path(self.model.get()).name}\n{meta['sampling_frequency_hz']:g} Hz{phase}{startup}{limit}")
        except Exception as exc:
            self.editor = None
            self.model_info.set(f"Package blocked · {exc}")
        self._refresh_fields(reset=True)
        self._update_actions()

    def _channel_choices(self, unit):
        if self.mode.get() == "CSV playback":
            source = [(channel.name, channel.unit) for channel in self.csv_channels]
        else:
            source = [(fields[0].get().strip(), fields[1].get().strip())
                      for enabled, fields in self.channel_vars if enabled.get()]
        return [name for name, actual in source if name and (not unit or unit.casefold() == actual.casefold())]

    def _refresh_fields(self, reset=False):
        prior = {name: (channel.get(), unit.get()) for name, (channel, unit) in self.input_fields.items()}
        if reset:
            prior.clear()
        old_speed = ((self.speed_choice.get(), self.rpm.get()) if hasattr(self, "speed_choice") and not reset else
                     ("Channel" if self._channel_choices("rpm") else "User value", ""))
        for child in self.inputs_box.winfo_children():
            child.destroy()
        self.input_fields.clear()
        self.speed_choice = tk.StringVar(self, old_speed[0])
        self.rpm = tk.StringVar(self, old_speed[1])
        phase = self.physical_phase.get() if hasattr(self, "physical_phase") and not reset else "Unspecified"
        self.physical_phase = tk.StringVar(self, phase)
        if not self.editor:
            self.inputs_box.pack_forget()
            if any(task.get("healthy_start_required", True) for task in self.tasks):
                self.inputs_box.pack(fill="x", padx=6, pady=5, before=self.extras)
                ttk.Checkbutton(self.inputs_box, text="I confirmed this recording begins healthy",
                                variable=self.healthy).grid(row=0, column=0, sticky="w", padx=7, pady=4)
            return
        self.inputs_box.pack(fill="x", padx=6, pady=5, before=self.extras)
        has_speed = self._has_profile_speed()
        if has_speed and not self._channel_choices("rpm"):
            self.speed_choice.set("User value")
        speed_row = None
        for index, (name, declared) in enumerate(self.editor["inputs"].items()):
            channels = self._channel_choices(declared)
            selected, expected = prior.get(name, ("", declared))
            if selected not in channels:
                selected = name if name in channels else channels[0] if declared and len(channels) == 1 else ""
            channel = self._field(self.inputs_box, index, name, selected, channels)
            widget = self.inputs_box.grid_slaves(row=index, column=1)[0]
            widget.configure(postcommand=lambda combo=widget, unit=declared: combo.configure(values=self._channel_choices(unit)))
            unit = tk.StringVar(self, declared or expected)
            if declared:
                ttk.Label(self.inputs_box, text=declared).grid(row=index, column=2, sticky="w", padx=5)
            else:
                ttk.Entry(self.inputs_box, textvariable=unit, width=10).grid(row=index, column=2, padx=5)
                ttk.Label(self.inputs_box, text="Model unit").grid(row=index, column=3, sticky="w")
            self.input_fields[name] = channel, unit
            if has_speed and name == "speed_rpm":
                speed_row = self.inputs_box.grid_slaves(row=index)
        row = len(self.input_fields)
        if has_speed:
            speed_options = ttk.Frame(self.inputs_box)
            speed_options.grid(row=row, column=0, columnspan=4, sticky="ew", padx=7, pady=5)
            ttk.Label(speed_options, text="Motor speed").pack(side="left", padx=(0, 7))
            manual = ttk.Frame(speed_options)
            entry = ttk.Entry(manual, textvariable=self.rpm, width=8)
            entry.pack(side="left")
            ttk.Label(manual, text="rpm").pack(side="left", padx=3)
            def update_speed(*_):
                use_channel = self.speed_choice.get() == "Channel"
                for item in speed_row:
                    item.grid() if use_channel else item.grid_remove()
                if use_channel:
                    manual.pack_forget()
                else:
                    manual.pack(side="left", padx=5)
            for text, value in (("Enter RPM", "User value"), ("Use RPM channel", "Channel")):
                ttk.Radiobutton(speed_options, text=text, variable=self.speed_choice,
                                value=value, command=update_speed).pack(side="left", padx=(0, 4))
            self._update_speed_fields = update_speed
            update_speed()
            refs = self._reference_speeds(self.editor)
            if refs:
                ttk.Label(self.inputs_box, text="Saved reference speeds: " + ", ".join(f"{rpm:g}" for rpm in refs) + " rpm",
                          wraplength=525).grid(row=row + 1, column=0, columnspan=4, sticky="w", padx=7, pady=(0, 4))
        if self.editor["metadata"].get("phase"):
            row += 2
            ttk.Label(self.inputs_box, text="Physical phase (optional)").grid(row=row, column=0, padx=7, sticky="w")
            ttk.Combobox(self.inputs_box, textvariable=self.physical_phase,
                         values=("Unspecified", "U", "V", "W"), state="readonly").grid(row=row, column=1, sticky="ew", padx=5)
        if (self.editor["healthy_start_required"] or
                any(task.get("healthy_start_required", True) for task in self.tasks)):
            ttk.Checkbutton(self.inputs_box, text="I confirmed this recording begins healthy",
                            variable=self.healthy).grid(row=row + 1, column=0, columnspan=4, sticky="w", padx=7, pady=4)

    @staticmethod
    def _reference_speeds(inspected):
        profile = inspected["profile"]
        if profile:
            for stage in profile.spec["steps"]:
                if stage["op"] == "healthy_peak_by_rpm.v1":
                    return [ref["speed_rpm"] for ref in stage["references"]]
        return []

    def _has_profile_speed(self):
        return self.editor and "speed_rpm" in self.editor["inputs"] and (
            self.editor["profile"] is not None or self.editor["healthy_start_required"])

    def _apply_task(self, *, check_source=False, quiet=False):
        if self.running or self.future:
            return False
        if not self.model.get().strip():
            self.status.set("Choose a diagnosis model first.")
            return False
        try:
            inspected = inspect_package(self.model.get(), self.profile.get() or None)
            if list(inspected["inputs"]) != list(self.input_fields):
                raise ValueError("Inspect the selected package before mapping its inputs.")
            mapping = {name: field.get() for name, (field, _) in self.input_fields.items()}
            units = {name: unit.get().strip() for name, (_, unit) in self.input_fields.items()}
            rpm, speed = None, {}
            if self._has_profile_speed():
                speed = dict(speed_source=self.speed_choice.get(), speed_channel=mapping.pop("speed_rpm"))
                if speed["speed_source"] == "User value":
                    rpm = float(self.rpm.get() or "nan")
                    if not np.isfinite(rpm) or rpm <= 0:
                        raise ValueError("Enter the motor speed in rpm for this recording.")
                    refs = self._reference_speeds(inspected)
                    if refs and sum(np.isclose(rpm, refs, rtol=.001, atol=.01)) != 1:
                        raise ValueError(f"This model has no saved reference for {rpm:g} rpm. "
                                         "Check the supported speeds under Motor speed.")
                elif not speed["speed_channel"]:
                    raise ValueError("Choose a calibrated RPM channel or enter the motor speed.")
            if any(not channel for channel in mapping.values()) or any(not unit for unit in units.values()):
                raise ValueError("Map all model inputs and confirm their expected units.")
            model = str(Path(self.model.get()).resolve())
            profile = str(Path(self.profile.get()).resolve()) if self.profile.get() else None
            if check_source:
                self._check_mapping(inspected, mapping, units, speed)
            name = Path(model).stem + (" · " + ", ".join(mapping.values()) if mapping else "")
            phase = self.physical_phase.get()
            physical = None if phase == "Unspecified" else phase
            if physical and inspected["metadata"].get("phase") not in (None, physical):
                name += " · cross-phase comparison"
            task = dict(id=self.editing_id or uuid4().hex, name=name, model=model, profile=profile,
                        model_sha256=sha256(model), profile_sha256=sha256(profile) if profile else None,
                        profile_digest=inspected["profile"].digest if inspected["profile"] else None,
                        healthy_start_required=inspected["healthy_start_required"],
                        mapping=mapping, input_units=units, rpm=rpm, speed_settings=speed, physical_phase=physical)
            previous = next((item for item in self.tasks if item["id"] == self.editing_id), None)
            if previous:
                for key in ("model", "profile"):
                    checksum = key + "_sha256"
                    if (previous.get(key) == task[key] and previous.get(checksum)
                            and previous[checksum] != task[checksum]):
                        raise ValueError(f"The {key} file changed since it was saved. Add it as a new diagnosis.")
                if (previous.get("profile") == profile and previous.get("profile_digest") and
                        previous["profile_digest"] != task["profile_digest"] and previous["model"] == model):
                    raise ValueError("The preprocessing profile changed since it was saved. Add it as a new diagnosis.")
        except Exception as exc:
            self.status.set(str(exc))
            return False
        for index, previous in enumerate(self.tasks):
            if previous["id"] == task["id"]:
                self.tasks[index] = task
                break
        else:
            self.tasks.append(task)
        self.editing_id = task["id"]
        self._task_rows()
        if not quiet:
            self.status.set(f"Saved {name}.")
        return True

    def _check_mapping(self, inspected, mapping, units, speed):
        is_csv = self.mode.get() == "CSV playback"
        channels = self.csv_channels if is_csv else self._channels()
        available = {channel.name: channel.unit for channel in channels}
        rate = inspected["metadata"]["sampling_frequency_hz"]
        if is_csv and self.csv_rate is not None and rate > self.csv_rate * (1 + 1e-8):
            raise ValueError(f"This model needs {rate:g} Hz. Choose a capture recorded at that rate or higher.")
        inputs = dict(mapping)
        if speed.get("speed_source") == "Channel":
            if speed["speed_channel"] in inputs.values():
                raise ValueError("Choose a separate channel for motor speed.")
            inputs["speed_rpm"] = speed["speed_channel"]
        for name, channel in inputs.items():
            required = inspected["inputs"][name] or units[name]
            if channel not in available:
                raise ValueError(f"Choose an enabled channel for {name}.")
            if required.casefold() != available[channel].strip().casefold():
                raise ValueError(f"{name} requires {required}. {channel} is recorded in {available[channel]}.")

    def _task_rows(self):
        self.task_list.delete(*self.task_list.get_children())
        for task in self.tasks:
            members = [f"{key} = {value}" for key, value in task["mapping"].items()]
            speed = task.get("speed_settings", {})
            if speed:
                members.append(f"speed = {task['rpm']:g} rpm" if task["rpm"] else f"speed = {speed['speed_channel']}")
            self.task_list.insert("", "end", iid=task["id"], values=(task["name"], ", ".join(members)))
        if self.editing_id and self.task_list.exists(self.editing_id):
            self.task_list.selection_set(self.editing_id)
        self._update_advanced_label()
        self._update_actions()

    def _add_another(self):
        if self.running or self.future:
            return
        if self.model.get().strip() and not self._apply_task(quiet=True):
            return
        self._new_task()
        self._browse_model(self.model)

    def _new_task(self):
        self.editing_id = None
        self.model.set("")
        self.profile.set("")
        self.editor = None
        self.model_info.set("Choose a trusted model ZIP to see predictions.")
        self._refresh_fields(reset=True)
        self._update_actions()

    def _edit_task(self, _event=None):
        if self.running or self.future:
            return
        selection = self.task_list.selection()
        if not selection:
            return
        task = next(item for item in self.tasks if item["id"] == selection[0])
        self.editing_id = task["id"]
        self.model.set(task["model"])
        self.profile.set(task.get("profile") or "")
        self._inspect()
        for name, (channel, unit) in self.input_fields.items():
            channel.set(task["mapping"].get(name, task.get("speed_settings", {}).get("speed_channel", "")))
            unit.set(task["input_units"].get(name, unit.get()))
        self.speed_choice.set(task.get("speed_settings", {}).get("speed_source", "Channel"))
        if self._has_profile_speed():
            self._update_speed_fields()
        self.rpm.set(f"{task['rpm']:g}" if task["rpm"] else "")
        self.physical_phase.set(task.get("physical_phase") or "Unspecified")

    def _remove_task(self):
        if self.running or self.future:
            return
        selected = self.task_list.selection()
        self.tasks = [task for task in self.tasks if task["id"] not in selected]
        if self.editing_id in selected:
            self._new_task()
        self._task_rows()

    def _acquisition(self):
        timeout = float(self.timeout.get())
        scale = float(self.time_scale.get())
        if not np.isfinite(timeout) or not 0 < timeout <= 120 or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Set a trigger timeout from 0 to 120 seconds and a positive timebase.")
        return dict(trigger_timeout=timeout, time_scale=scale, memory_depth=int(self.depth.get()),
                    setup=self.configure_scope.get(), force_trigger=self.force.get())

    def _submit(self, operation, callback, *args):
        if self.future:
            return
        self.future = self.worker.submit(operation, *args)
        self.poll_id = self.after(25, self._poll, callback)

    def _poll(self, callback):
        self.poll_id = None
        if not self.future.done():
            self.poll_id = self.after(25, self._poll, callback)
            return
        future, self.future = self.future, None
        try:
            value = future.result()
            if self.running and self.live is None and self.worker.cancel.is_set():
                raise AcquisitionCancelled()
            callback(value)
        except (AcquisitionTimeout, AcquisitionCancelled) as exc:
            if self.live and isinstance(exc, AcquisitionTimeout) and not self.worker.cancel.is_set():
                self.status.set(f"{self._live_summary()} · Waiting for a trigger. Retrying…")
                self.tick_id = self.after(250, self._next_live)
                return
            self._finish(str(exc) if isinstance(exc, AcquisitionTimeout) and not self.worker.cancel.is_set()
                         else "Stopped")
            self._submit(self.worker.clear_playback, lambda _value: None)
        except Exception as exc:
            if isinstance(exc, ConnectionError):
                self._disconnected(None)
            self._finish(str(exc))

    def _scan(self):
        if not self.running:
            self._submit(self.worker.scan, self._scanned, self.backend.get())

    def _scanned(self, resources):
        self.resources.configure(values=resources)
        if self.capabilities:
            self.status.set("USB scan complete. The current instrument session is still open.")
            return
        if resources and self.resource.get() not in resources:
            self.resource.set(resources[0])
        self.status.set("Choose an instrument address and click Connect." if resources else
                        "No USB oscilloscope found. Check the USB connection and VISA backend.")

    def _connect(self):
        if self.running or self.future:
            return
        if not self.resource.get().strip():
            self.status.set("Scan USB or enter your instrument address first.")
            return
        self._disconnected(None)
        self.status.set("Connecting to the oscilloscope…")
        self._submit(self.worker.connect, self._connected, self.resource.get().strip(), self.backend.get())

    def _connected(self, value):
        identity, self.capabilities = value
        self.connection.set(f"{identity}\n{self.capabilities.validation}")
        if self.capabilities.force_trigger:
            self.force_button.grid(row=4, column=1, sticky="w", padx=5)
        else:
            self.force.set(False)
        self.status.set("Oscilloscope connected.")

    def _disconnect(self):
        if not self.running:
            self._submit(self.worker.disconnect, self._disconnected)

    def _disconnected(self, _value):
        self.capabilities = None
        self.force_button.grid_remove()
        self.connection.set("Disconnected")

    def _choose_recording(self):
        session = self.live
        folder = filedialog.askdirectory(parent=self, title="Choose where to save this recording")
        if not folder or self.closed or self.live is not session or (session and self.worker.cancel.is_set()):
            return None
        try:
            return ScopeRecording(folder)
        except OSError as exc:
            self.status.set(f"Could not create a recording folder. {exc}")
            return None

    def _record(self):
        if not self.running:
            self.start(waveform_only=True, live=True, record=True)
        elif self.live and self.recording is None and not self.worker.cancel.is_set():
            recording = self._choose_recording()
            if recording:
                self.recording = recording
                self._update_actions()
                self.status.set(f"Recording begins with the next acquisition in {recording.folder}")

    def start(self, *, waveform_only=False, live=False, record=False):
        if self.running or self.future:
            return
        is_csv = self.mode.get() == "CSV playback"
        try:
            if is_csv and not self.csv.get():
                raise ValueError("Choose a saved capture on the Setup page.")
            if not is_csv and not self.capabilities:
                raise ValueError("Connect an oscilloscope on the Setup page.")
            channels, setup = (None, None) if is_csv else (self._channels(), self._acquisition())
            if not waveform_only:
                if self.model.get().strip() and not self._apply_task(check_source=True, quiet=True):
                    self.tabs.select(self.setup)
                    return
                if not self.tasks:
                    raise ValueError("Choose a model to run diagnosis, or preview the waveform.")
                if all(task.get("healthy_start_required") for task in self.tasks) and not self.healthy.get():
                    raise ValueError("Confirm that this recording begins healthy before running this diagnosis.")
        except ValueError as exc:
            self.tabs.select(self.setup)
            self.status.set(str(exc))
            return
        if record:
            self.recording = self._choose_recording()
            if self.recording is None:
                return
        self.live = (channels, setup) if live and not is_csv else None
        self.live_count = 0
        tasks = [] if waveform_only else self.tasks.copy()
        healthy = self.healthy.get()
        self.healthy.set(False)
        self.worker.cancel.clear()
        self.running, self.paused = True, False
        self.position = 0
        self.result = None
        self.progress.configure(value=0)
        self.results.delete(*self.results.get_children())
        self.result_notes.clear()
        self.result_detail.set("Waiting for the recording.")
        self.save_button.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self.mode_box.configure(state="disabled")
        self.run_info.set("Previous record · historical" if self.capture else "Preparing capture")
        self.tabs.select(self.monitor)
        self._update_actions()
        self.status.set("Opening saved capture…" if is_csv else "Waiting for instrument capture…")
        if is_csv:
            self._submit(self.worker.open_playback, self._opened, self.csv.get(), tasks, healthy)
        elif self.live:
            self.status.set(f"{self._live_summary()} · Waiting for the first capture…")
            self._next_live()
        else:
            self._submit(self.worker.capture, self._captured, channels, setup, tasks, healthy)

    def _live_summary(self):
        if self.recording:
            return f"{self.recording.saved} saved to {self.recording.folder}"
        return f"Live view · {self.live_count} captures"

    def _next_live(self):
        self.tick_id = None
        if self.live and not self.closed and not self.worker.cancel.is_set():
            self._submit(self.worker.capture_live, self._live_captured, *self.live, self.recording)

    def _live_captured(self, value):
        capture, self.result, save_error = value
        self.live_count += 1
        self._plot(capture)
        self.position = len(capture.time_s)
        self._show(self.result)
        if save_error or self.worker.cancel.is_set():
            self._finish(save_error or "Stopped")
        else:
            self.status.set(f"{self._live_summary()} · Acquiring next waveform…")
            self._next_live()

    def _plot(self, capture):
        self.capture = capture
        self.axes.clear()
        every = max(1, len(capture.time_s) // 6000)
        for index, channel in enumerate(capture.channels):
            self.axes.plot(capture.time_s[::every], capture.values[::every, index],
                           label=f"{channel.name} ({channel.unit})", linewidth=0.9)
        self.axes.set(xlabel="Time (s)", ylabel="Calibrated value")
        self.axes.legend(loc="upper right")
        self.cursor = self.axes.axvline(capture.time_s[0], color="#555555", alpha=0.6)
        self.chart.draw_idle()

    def _opened(self, value):
        capture, result = value
        self._plot(capture)
        self._show(result)
        self.pause_button.configure(state="normal", text="Pause")
        self._schedule_tick()

    def _captured(self, value):
        capture, result = value
        self._plot(capture)
        self.position = len(capture.time_s)
        self._completed(result)

    def _show(self, result):
        self.status.set(result.status)
        size = len(self.capture.time_s)
        self.progress.configure(value=self.position / size * 100)
        self.run_info.set(f"{self.capture.sample_rate_hz:g} Hz · {self.position:,} / {size:,} samples")
        for row in result.diagnosis_results:
            identity = row["diagnosis_id"]
            predictions = row["predictions"]
            blocked = row["status"].startswith(("INFERENCE BLOCKED", "ANALYSIS BLOCKED"))
            label, confidence = (("Blocked" if blocked else "Waiting…"), "—") if not predictions else (
                predictions[-1]["prediction"], f"{predictions[-1]['confidence']:.1%}")
            values = (row["diagnosis_name"], label, confidence, len(predictions))
            self.result_notes[identity] = row["status"]
            if self.results.exists(identity):
                self.results.item(identity, values=values)
            else:
                self.results.insert("", "end", iid=identity, values=values)
        self._show_result_detail()
        if self.position:
            time = self.capture.time_s[self.position - 1]
            self.cursor.set_xdata([time, time])
            self.chart.draw_idle()

    def _show_result_detail(self, _event=None):
        selection = self.results.selection()
        if selection:
            self.result_detail.set(self.result_notes.get(selection[0], ""))
        elif self.result_notes:
            self.result_detail.set(next((note for note in self.result_notes.values() if "BLOCKED" in note),
                                        next(iter(self.result_notes.values()))))
        elif self.live:
            self.result_detail.set("Each waveform has its own trigger. Transfer and rearming leave gaps between captures.")
        else:
            self.result_detail.set("Waveform preview")

    def _schedule_tick(self):
        self.tick_id = self.after(45, self._tick)

    def _tick(self):
        self.tick_id = None
        speed = float(self.play_speed.get().rstrip("×"))
        size = len(self.capture.time_s)
        self.position = min(size, self.position + max(1, round(self.capture.sample_rate_hz * speed * .045)))
        self._submit(self.worker.advance, self._advanced, self.position)

    def _advanced(self, result):
        self._show(result)
        if self.position == len(self.capture.time_s):
            self._completed(result)
            self._submit(self.worker.clear_playback, lambda _value: None)
        elif not self.paused:
            self._schedule_tick()
        else:
            self.status.set("Playback paused.")

    def _completed(self, result):
        self.result = result
        self._show(result)
        self._finish(result.status)
        self.save_button.configure(state="normal")

    def pause(self):
        if not self.running:
            return
        self.paused = not self.paused
        self.pause_button.configure(text="Resume" if self.paused else "Pause")
        if self.paused:
            self._cancel_tick()
            self.status.set("Playback paused.")
        elif not self.future:
            self._schedule_tick()

    def stop(self):
        if self.running:
            self.worker.cancel.set()
            self._cancel_tick()
            self.status.set("Stopping…")
            if not self.future:
                self._finish("Stopped")
                self._submit(self.worker.clear_playback, lambda _value: None)

    def _cancel_tick(self):
        if self.tick_id:
            self.after_cancel(self.tick_id)
            self.tick_id = None

    def _finish(self, status):
        if self.live:
            status = f"{status} · {self._live_summary()}"
            self.live = self.recording = None
            self.save_button.configure(state="normal" if self.result is not None else "disabled")
        if self.running and self.result is None:
            self.run_info.set("Previous or incomplete record · historical")
        self.running = False
        self._cancel_tick()
        self.pause_button.configure(text="Pause", state="disabled")
        self.run_button.configure(state="normal")
        self.mode_box.configure(state="readonly")
        self.status.set(status)
        self._update_actions()

    def _save_capture(self):
        if self.running or self.future or self.result is None:
            return
        folder = filedialog.askdirectory(parent=self, title="Save capture in folder")
        if folder:
            self._submit(save_capture, lambda path: self.status.set(f"Saved {path}"),
                         folder, self.capture, self.result)

    def _export(self):
        if self.running or self.future:
            return
        if self.model.get().strip() and not self._apply_task(quiet=True):
            self.tabs.select(self.setup)
            return
        if not self.tasks and self.mode.get() != "Oscilloscope":
            self.status.set("Choose and map a model before exporting a deployment.")
            return
        try:
            active = self.mode.get() == "Oscilloscope"
            config = dict(mode=self.mode.get(), tasks=self.tasks.copy(),
                          instrument=dict(resource=self.resource.get(), backend=self.backend.get(),
                                          acquisition=self._acquisition() if active else {},
                                          channels=[asdict(c) for c in self._channels()] if active else []))
        except ValueError as exc:
            self.status.set(str(exc))
            return
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".zip", initialfile="nn_scope_deployment.zip",
                                           filetypes=[("Deployment bundle", "*.zip")])
        if path:
            self._submit(export_session, lambda _value: self.status.set(f"Exported {path}"), path, config)

    def _restore(self, path):
        try:
            session = load_session(path)
            self.tasks = session["tasks"]
            instrument = session["instrument"]
            self.mode.set(session["mode"])
            self.backend.set(instrument["backend"])
            self.resource.set(instrument["resource"])
            acq = instrument["acquisition"]
            if acq:
                self.timeout.set(acq["trigger_timeout"])
                self.time_scale.set(acq["time_scale"])
                self.depth.set(acq["memory_depth"])
                self.force.set(acq["force_trigger"])
                self.configure_scope.set(acq["setup"])
            channels = {item["number"]: item for item in instrument["channels"]}
            if channels:
                for number, (enabled, variables) in enumerate(self.channel_vars, 1):
                    enabled.set(number in channels)
                    if number in channels:
                        for field, var in zip(("name", "unit", "gain", "sensor_zero", "offset"), variables):
                            var.set(channels[number][field])
            self._source_changed()
            self._task_rows()
            if self.tasks:
                self.task_list.selection_set(self.tasks[0]["id"])
                self._edit_task()
                self.status.set("Deployment loaded. Choose a capture or connect the oscilloscope.")
            elif self.mode.get() == "Oscilloscope":
                self.status.set("Connect the oscilloscope, then choose Live view or Record.")
        except (ValueError, OSError, KeyError, TypeError) as exc:
            self.status.set(f"Could not restore deployment · {exc}")

    def _on_destroy(self, event):
        if event.widget is not self or self.closed:
            return
        self.closed = True
        for identifier in (self.tick_id, self.poll_id, self.chart._idle_draw_id):
            if identifier:
                self.after_cancel(identifier)
        self.worker.close()
