"""Ui / annotations for NN Training Studio."""

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from pathlib import Path
from tkinter.scrolledtext import ScrolledText
from datetime import datetime
from tkinter import filedialog
import hashlib
import json
from tkinter import messagebox
import numpy as np
import pandas as pd
import queue
import re
import shutil
import threading
import tkinter as tk
from tkinter import ttk
from nn_training_studio.annotation_ai import (
    request_ai_annotation_plan,
)
from nn_training_studio.annotations import (
    _new_annotation_id,
    build_signal_label_dataframe,
    suggest_signal_anomaly_intervals,
    validate_box_annotation,
    validate_signal_annotation,
    write_yolo_annotation_dataset,
)
from nn_training_studio.branding import (
    apply_window_branding,
)
from nn_training_studio.constants import (
    ANNOTATION_IMAGE_EXTENSIONS,
    ANNOTATION_STATUS_APPROVED,
    ANNOTATION_STATUS_PENDING,
    ANNOTATION_STATUS_REJECTED,
    APPLICATION_NAME,
    APP_VERSION,
)
from nn_training_studio.results import (
    _result_json_default,
)


class AnnotationWorkspaceWindow(tk.Toplevel):
    """Human-in-the-loop annotation for signals and images."""

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        apply_window_branding(self)
        self.title(f"{APPLICATION_NAME} {APP_VERSION} — Annotate & Prepare Data")
        self.geometry("1320x850")
        self.minsize(1000, 680)
        self.transient(owner)

        self.mode_var = tk.StringVar(value="1D Signal / Table")
        self.label_var = tk.StringVar(value="Normal")
        self.start_var = tk.StringVar(value="0")
        self.end_var = tk.StringVar(value="0")
        self.event_var = tk.StringVar(value="0")
        self.signal_column_var = tk.StringVar(value="")
        self.sensitivity_var = tk.StringVar(value="3.5")
        self.ai_goal_var = tk.StringVar(
            value="Recommend a conservative label scheme and review workflow."
        )
        self.status_var = tk.StringVar(value="Load signal data or images to begin.")
        self.signal_df = None
        self.signal_path = None
        self.image_paths = []
        self.image_index = 0
        self.annotations = []
        self.label_names = ["Normal", "Fault", "Uncertain"]
        self.ai_plan = None
        self.ai_running = False
        self._ai_plan_queue = queue.Queue()
        self._ai_plan_poll_id = None
        self.signal_figure = None
        self.signal_canvas = None
        self.signal_toolbar = None
        self.image_photo = None
        self.image_display = None
        self.image_draw_start = None
        self.pending_box = None
        self.annotation_history = []
        self._build_ui()
        self.bind("<Destroy>", self._cancel_ai_plan_poll, add="+")

    def _build_ui(self):
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=14, pady=(12, 6))
        ttk.Label(
            header,
            text="Annotate & Prepare Data",
            font=("Arial", 18, "bold"),
        ).pack(side=tk.LEFT)
        ttk.Label(
            header,
            text=(
                "Load → annotate or suggest → review → export / continue to training"
            ),
            foreground="#245a85",
        ).pack(side=tk.LEFT, padx=18)
        ttk.Label(header, text=APP_VERSION, font=("Arial", 10, "bold")).pack(
            side=tk.RIGHT
        )

        controls = ttk.Frame(self)
        controls.pack(fill=tk.X, padx=14, pady=(0, 6))
        ttk.Label(controls, text="Data type:").pack(side=tk.LEFT)
        mode_box = ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=["1D Signal / Table", "2D Images"],
            state="readonly",
            width=20,
        )
        mode_box.pack(side=tk.LEFT, padx=6)
        mode_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_mode())
        ttk.Button(controls, text="Load Data", command=self.load_data).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(controls, text="Save Annotation Project", command=self.save_project).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(controls, text="Load Annotation Project", command=self.load_project).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(controls, text="AI Annotation Planner", command=self.start_ai_plan).pack(
            side=tk.LEFT, padx=6
        )
        ttk.Button(controls, text="AI Settings", command=self.owner.show_settings).pack(
            side=tk.RIGHT, padx=6
        )

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)
        self.annotate_tab = ttk.Frame(self.notebook)
        self.review_tab = ttk.Frame(self.notebook)
        self.ai_tab = ttk.Frame(self.notebook)
        self.export_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.annotate_tab, text="1. Annotate")
        self.notebook.add(self.review_tab, text="2. Review Labels")
        self.notebook.add(self.ai_tab, text="3. AI Plan")
        self.notebook.add(self.export_tab, text="4. Export & Train")

        self._build_annotate_tab()
        self._build_review_tab()
        self._build_ai_tab()
        self._build_export_tab()
        ttk.Label(self, textvariable=self.status_var, foreground="#245a85").pack(
            fill=tk.X, padx=16, pady=(2, 10)
        )
        self._refresh_mode()

    def _build_annotate_tab(self):
        self.annotate_tab.columnconfigure(1, weight=1)
        self.annotate_tab.rowconfigure(0, weight=1)
        self.annotation_controls = ttk.LabelFrame(
            self.annotate_tab, text="Annotation Controls"
        )
        self.annotation_controls.grid(
            row=0, column=0, sticky="nsw", padx=(8, 4), pady=8
        )
        ttk.Label(self.annotation_controls, text="Label / class:").pack(
            anchor="w", padx=10, pady=(10, 2)
        )
        self.label_box = ttk.Combobox(
            self.annotation_controls,
            textvariable=self.label_var,
            values=self.label_names,
            width=24,
        )
        self.label_box.pack(fill=tk.X, padx=10)
        ttk.Button(
            self.annotation_controls, text="Add Label Name", command=self.add_label_name
        ).pack(fill=tk.X, padx=10, pady=5)

        self.signal_controls = ttk.LabelFrame(
            self.annotation_controls, text="1D Interval / Event"
        )
        self.signal_controls.pack(fill=tk.X, padx=8, pady=8)
        for label, variable in (
            ("Start row", self.start_var),
            ("End row", self.end_var),
            ("Event row", self.event_var),
        ):
            row = ttk.Frame(self.signal_controls)
            row.pack(fill=tk.X, padx=6, pady=2)
            ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
            ttk.Entry(row, textvariable=variable, width=12).pack(side=tk.LEFT)
        ttk.Button(
            self.signal_controls, text="Add Labelled Interval", command=self.add_signal_interval
        ).pack(fill=tk.X, padx=6, pady=(5, 2))
        ttk.Button(
            self.signal_controls, text="Add Event Point", command=self.add_signal_event
        ).pack(fill=tk.X, padx=6, pady=2)
        ttk.Separator(self.signal_controls).pack(fill=tk.X, padx=6, pady=7)
        ttk.Label(self.signal_controls, text="Suggestion signal:").pack(
            anchor="w", padx=6
        )
        self.signal_column_box = ttk.Combobox(
            self.signal_controls,
            textvariable=self.signal_column_var,
            state="readonly",
            width=22,
        )
        self.signal_column_box.pack(fill=tk.X, padx=6, pady=2)
        sensitivity_row = ttk.Frame(self.signal_controls)
        sensitivity_row.pack(fill=tk.X, padx=6, pady=2)
        ttk.Label(sensitivity_row, text="Sensitivity").pack(side=tk.LEFT)
        ttk.Entry(sensitivity_row, textvariable=self.sensitivity_var, width=8).pack(
            side=tk.RIGHT
        )
        ttk.Button(
            self.signal_controls,
            text="Suggest Local Anomaly Regions",
            command=self.suggest_local_signal_regions,
        ).pack(fill=tk.X, padx=6, pady=(4, 7))

        self.image_controls = ttk.LabelFrame(
            self.annotation_controls, text="2D Image"
        )
        self.image_controls.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(
            self.image_controls,
            text="Assign Class to Current Image",
            command=self.assign_image_class,
        ).pack(fill=tk.X, padx=6, pady=(6, 2))
        ttk.Label(
            self.image_controls,
            text="Drag on the image to draw a box, then add it.",
            wraplength=230,
        ).pack(anchor="w", padx=6, pady=3)
        ttk.Button(
            self.image_controls,
            text="Add Drawn Bounding Box",
            command=self.add_pending_box,
        ).pack(fill=tk.X, padx=6, pady=2)
        navigation = ttk.Frame(self.image_controls)
        navigation.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(navigation, text="← Previous", command=self.previous_image).pack(
            side=tk.LEFT
        )
        ttk.Button(navigation, text="Next →", command=self.next_image).pack(
            side=tk.RIGHT
        )

        self.annotation_preview = ttk.Frame(self.annotate_tab)
        self.annotation_preview.grid(
            row=0, column=1, sticky="nsew", padx=(4, 8), pady=8
        )

    def _build_review_tab(self):
        actions = ttk.Frame(self.review_tab)
        actions.pack(fill=tk.X, padx=8, pady=8)
        ttk.Button(actions, text="Approve Selected", command=lambda: self.set_selected_status(ANNOTATION_STATUS_APPROVED)).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Reject Selected", command=lambda: self.set_selected_status(ANNOTATION_STATUS_REJECTED)).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Return to Pending", command=lambda: self.set_selected_status(ANNOTATION_STATUS_PENDING)).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Delete Selected", command=self.delete_selected).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Quality Check", command=self.show_quality_check).pack(side=tk.LEFT, padx=4)
        columns = ("kind", "label", "location", "source", "confidence", "status")
        self.annotation_tree = ttk.Treeview(
            self.review_tab, columns=columns, show="headings", selectmode="extended"
        )
        headings = {
            "kind": "Type", "label": "Label", "location": "Location",
            "source": "Source", "confidence": "Confidence", "status": "Review status",
        }
        for column in columns:
            self.annotation_tree.heading(column, text=headings[column])
            self.annotation_tree.column(column, width=150, anchor="w")
        self.annotation_tree.column("location", width=300)
        scrollbar = ttk.Scrollbar(
            self.review_tab, orient=tk.VERTICAL, command=self.annotation_tree.yview
        )
        self.annotation_tree.configure(yscrollcommand=scrollbar.set)
        self.annotation_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=(0, 8))
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=(0, 8))

    def _build_ai_tab(self):
        top = ttk.Frame(self.ai_tab)
        top.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(top, text="Describe the annotation goal:").pack(anchor="w")
        ttk.Entry(top, textvariable=self.ai_goal_var).pack(fill=tk.X, pady=4)
        ttk.Button(top, text="Create AI Annotation Plan", command=self.start_ai_plan).pack(anchor="w")
        ttk.Label(
            top,
            text=(
                "Privacy: only compact statistics, columns, dimensions, and file counts are sent. "
                "Raw signals and image pixels remain local."
            ),
            foreground="#666666",
            wraplength=1000,
        ).pack(anchor="w", pady=5)
        self.ai_plan_text = ScrolledText(self.ai_tab, wrap=tk.WORD)
        self.ai_plan_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.ai_plan_text.insert(tk.END, "No AI annotation plan generated yet.")

    def _build_export_tab(self):
        frame = ttk.LabelFrame(self.export_tab, text="Reviewed Dataset Outputs")
        frame.pack(fill=tk.X, padx=12, pady=12)
        ttk.Label(
            frame,
            text=(
                "Exports include annotation source, confidence, review status, label scheme, "
                "and history. Pending/rejected suggestions are preserved in JSON but are not "
                "used as training labels."
            ),
            wraplength=1000,
        ).pack(anchor="w", padx=12, pady=(12, 8))
        buttons = ttk.Frame(frame)
        buttons.pack(fill=tk.X, padx=8, pady=(0, 12))
        ttk.Button(buttons, text="Export Annotation JSON", command=self.export_json).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Export Labelled Signal CSV", command=self.export_signal_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Export Image Classification Folders", command=self.export_image_classes).pack(side=tk.LEFT, padx=4)
        ttk.Button(buttons, text="Export YOLO Dataset", command=self.export_yolo).pack(side=tk.LEFT, padx=4)
        transfer = ttk.LabelFrame(self.export_tab, text="Continue in NN Studio")
        transfer.pack(fill=tk.X, padx=12, pady=12)
        ttk.Button(
            transfer,
            text="Use Labelled Signal Data in Training",
            command=self.continue_signal_training,
        ).pack(side=tk.LEFT, padx=12, pady=12)
        ttk.Button(
            transfer,
            text="Use Image Classes in Training",
            command=self.continue_image_training,
        ).pack(side=tk.LEFT, padx=12, pady=12)

    def _refresh_mode(self):
        for child in self.annotation_preview.winfo_children():
            child.destroy()
        signal_mode = self.mode_var.get().startswith("1D")
        if signal_mode:
            self.image_canvas_widget = None
            self.image_controls.pack_forget()
            if not self.signal_controls.winfo_manager():
                self.signal_controls.pack(fill=tk.X, padx=8, pady=8)
            self._build_signal_preview()
        else:
            self.signal_axis = None
            self.signal_canvas = None
            self.signal_controls.pack_forget()
            if not self.image_controls.winfo_manager():
                self.image_controls.pack(fill=tk.X, padx=8, pady=8)
            self._build_image_preview()

    def _build_signal_preview(self):
        from matplotlib.figure import Figure
        self.signal_figure = Figure(figsize=(8, 5), dpi=100)
        self.signal_axis = self.signal_figure.add_subplot(111)
        self.signal_axis.set_title("Load a CSV/Excel signal dataset")
        self.signal_axis.set_xlabel("Sample row")
        self.signal_axis.grid(True, alpha=0.3)
        self.signal_canvas = FigureCanvasTkAgg(self.signal_figure, master=self.annotation_preview)
        self.signal_canvas.draw()
        self.signal_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.signal_toolbar = NavigationToolbar2Tk(self.signal_canvas, self.annotation_preview)
        self.signal_toolbar.update()
        self.signal_canvas.mpl_connect("button_press_event", self._signal_click)
        self.plot_signal_annotations()

    def _build_image_preview(self):
        self.image_title = ttk.Label(
            self.annotation_preview, text="Load an image file or folder", font=("Arial", 11, "bold")
        )
        self.image_title.pack(anchor="w", pady=(0, 4))
        self.image_canvas_widget = tk.Canvas(
            self.annotation_preview, background="#222222", highlightthickness=0
        )
        self.image_canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.image_canvas_widget.bind("<ButtonPress-1>", self._box_press)
        self.image_canvas_widget.bind("<B1-Motion>", self._box_drag)
        self.image_canvas_widget.bind("<ButtonRelease-1>", self._box_release)
        self.image_canvas_widget.bind("<Configure>", lambda _event: self.display_current_image())
        self.display_current_image()

    def load_data(self):
        if self.mode_var.get().startswith("1D"):
            path = filedialog.askopenfilename(
                title="Load Signal / Table Data",
                filetypes=[("Data", "*.csv *.xlsx *.xls"), ("CSV", "*.csv"), ("Excel", "*.xlsx *.xls")],
            )
            if not path:
                return
            try:
                self.signal_df = pd.read_csv(path) if Path(path).suffix.lower() == ".csv" else pd.read_excel(path)
                if self.signal_df.empty:
                    raise ValueError("The selected dataset is empty.")
                self.signal_path = str(Path(path).resolve())
                numeric = list(self.signal_df.select_dtypes(include=[np.number]).columns)
                self.signal_column_box.config(values=numeric)
                if numeric:
                    self.signal_column_var.set(numeric[0])
                self.end_var.set(str(len(self.signal_df) - 1))
                self.status_var.set(f"Loaded {len(self.signal_df):,} rows and {len(self.signal_df.columns)} columns.")
                self.plot_signal_annotations()
            except Exception as exc:
                messagebox.showerror("Load Error", str(exc), parent=self)
        else:
            directory = filedialog.askdirectory(title="Load Image Folder")
            if not directory:
                single = filedialog.askopenfilename(
                    title="Or select one image",
                    filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")],
                )
                paths = [Path(single)] if single else []
            else:
                paths = [
                    path for path in Path(directory).rglob("*")
                    if path.is_file() and path.suffix.lower() in ANNOTATION_IMAGE_EXTENSIONS
                ]
            self.image_paths = sorted(str(path.resolve()) for path in paths)
            self.image_index = 0
            self.status_var.set(f"Loaded {len(self.image_paths):,} image files.")
            self.display_current_image()

    def add_label_name(self):
        label = " ".join(self.label_var.get().split()).strip()
        if not label:
            messagebox.showwarning("Label Required", "Enter a label name.", parent=self)
            return
        if label not in self.label_names:
            self.label_names.append(label)
            self.label_box.config(values=self.label_names)

    def _remember_change(self, action):
        self.annotation_history.append({"time": datetime.now().isoformat(), "action": action})

    def add_signal_interval(self):
        try:
            item = validate_signal_annotation({
                "kind": "interval", "label": self.label_var.get(),
                "start": int(self.start_var.get()), "end": int(self.end_var.get()),
                "source": "Manual", "confidence": 1.0,
                "status": ANNOTATION_STATUS_APPROVED,
            }, len(self.signal_df) if self.signal_df is not None else 0)
            self.annotations.append(item)
            self._remember_change(f"Added interval {item['id']}")
            self.refresh_annotations()
        except Exception as exc:
            messagebox.showerror("Annotation Error", str(exc), parent=self)

    def add_signal_event(self):
        try:
            item = validate_signal_annotation({
                "kind": "event", "label": self.label_var.get(),
                "start": int(self.event_var.get()), "source": "Manual",
                "confidence": 1.0, "status": ANNOTATION_STATUS_APPROVED,
            }, len(self.signal_df) if self.signal_df is not None else 0)
            self.annotations.append(item)
            self._remember_change(f"Added event {item['id']}")
            self.refresh_annotations()
        except Exception as exc:
            messagebox.showerror("Annotation Error", str(exc), parent=self)

    def _signal_click(self, event):
        if event.xdata is None or self.signal_df is None:
            return
        row = max(0, min(len(self.signal_df) - 1, int(round(event.xdata))))
        if event.button == 1:
            self.start_var.set(str(row))
            self.event_var.set(str(row))
        elif event.button == 3:
            self.end_var.set(str(row))

    def suggest_local_signal_regions(self):
        try:
            column = self.signal_column_var.get()
            intervals = suggest_signal_anomaly_intervals(
                self.signal_df, [column], float(self.sensitivity_var.get())
            )
            if "Possible Anomaly" not in self.label_names:
                self.label_names.append("Possible Anomaly")
                self.label_box.config(values=self.label_names)
            for interval in intervals:
                self.annotations.append(validate_signal_annotation({
                    "kind": "interval", "label": "Possible Anomaly",
                    "start": interval["start"], "end": interval["end"],
                    "source": "Local suggestion", "confidence": interval["confidence"],
                    "status": ANNOTATION_STATUS_PENDING,
                    "notes": f"Robust local score on {column}; human review required.",
                }, len(self.signal_df)))
            self._remember_change(f"Created {len(intervals)} local signal suggestions")
            self.refresh_annotations()
            self.notebook.select(self.review_tab)
            self.status_var.set(f"Created {len(intervals)} pending local suggestions; none were auto-approved.")
        except Exception as exc:
            messagebox.showerror("Suggestion Error", str(exc), parent=self)

    def plot_signal_annotations(self):
        if not getattr(self, "signal_axis", None):
            return
        self.signal_axis.clear()
        if self.signal_df is None:
            self.signal_axis.set_title("Load a CSV/Excel signal dataset")
        else:
            numeric = list(self.signal_df.select_dtypes(include=[np.number]).columns)
            selected = self.signal_column_var.get()
            columns = [selected] if selected in numeric else numeric[:3]
            maximum = 8000
            step = max(1, int(np.ceil(len(self.signal_df) / maximum)))
            x_values = np.arange(0, len(self.signal_df), step)
            for column in columns[:3]:
                self.signal_axis.plot(x_values, self.signal_df[column].iloc[::step], label=str(column), linewidth=0.9)
            colors = {ANNOTATION_STATUS_APPROVED: "#2ca02c", ANNOTATION_STATUS_PENDING: "#ffbf00", ANNOTATION_STATUS_REJECTED: "#999999"}
            for item in self.annotations:
                if item.get("data_type") != "signal":
                    continue
                color = colors.get(item.get("status"), "#4c78a8")
                if item.get("kind") == "event":
                    self.signal_axis.axvline(item["start"], color=color, alpha=0.8, linestyle="--")
                else:
                    self.signal_axis.axvspan(item["start"], item["end"], color=color, alpha=0.15)
            self.signal_axis.set_title("Left-click: start/event row • Right-click: end row")
            if columns:
                self.signal_axis.legend(loc="upper right")
        self.signal_axis.set_xlabel("Sample row")
        self.signal_axis.grid(True, alpha=0.25)
        if self.signal_canvas:
            try:
                self.signal_canvas.draw_idle()
            except tk.TclError:
                pass

    def current_image_path(self):
        if not self.image_paths:
            return None
        self.image_index = max(0, min(self.image_index, len(self.image_paths) - 1))
        return self.image_paths[self.image_index]

    def display_current_image(self):
        canvas = getattr(self, "image_canvas_widget", None)
        if canvas is None:
            return
        try:
            if not canvas.winfo_exists():
                return
            canvas.delete("all")
        except tk.TclError:
            return
        path = self.current_image_path()
        if not path:
            canvas.create_text(20, 20, anchor="nw", fill="white", text="Load an image file or folder.")
            return
        try:
            from PIL import Image, ImageTk
            with Image.open(path) as source:
                image = source.convert("RGB")
                width = max(canvas.winfo_width(), 600)
                height = max(canvas.winfo_height(), 450)
                scale = min(width / image.width, height / image.height, 1.0)
                display = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
            self.image_photo = ImageTk.PhotoImage(display)
            offset_x = max(0, (width - display.width) // 2)
            offset_y = max(0, (height - display.height) // 2)
            self.image_display = {"x": offset_x, "y": offset_y, "w": display.width, "h": display.height}
            canvas.create_image(offset_x, offset_y, image=self.image_photo, anchor="nw")
            for item in self.annotations:
                if item.get("kind") != "box" or item.get("image_path") != path or item.get("status") == ANNOTATION_STATUS_REJECTED:
                    continue
                x1 = offset_x + item["x"] * display.width
                y1 = offset_y + item["y"] * display.height
                x2 = x1 + item["w"] * display.width
                y2 = y1 + item["h"] * display.height
                color = "#2ecc71" if item.get("status") == ANNOTATION_STATUS_APPROVED else "#f1c40f"
                canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=2)
                canvas.create_text(x1 + 3, y1 + 3, anchor="nw", text=item["label"], fill=color)
            self.image_title.config(text=f"{self.image_index + 1}/{len(self.image_paths)} — {Path(path).name}")
        except Exception as exc:
            canvas.create_text(20, 20, anchor="nw", fill="white", text=f"Preview failed: {exc}")

    def _box_press(self, event):
        self.image_draw_start = (event.x, event.y)

    def _box_drag(self, event):
        if self.image_draw_start is None:
            return
        self.image_canvas_widget.delete("pending_box")
        self.image_canvas_widget.create_rectangle(
            self.image_draw_start[0], self.image_draw_start[1], event.x, event.y,
            outline="#00bfff", width=2, tags="pending_box"
        )

    def _box_release(self, event):
        if self.image_draw_start is None or not self.image_display:
            return
        x1, y1 = self.image_draw_start
        x2, y2 = event.x, event.y
        display = self.image_display
        left = max(display["x"], min(x1, x2))
        top = max(display["y"], min(y1, y2))
        right = min(display["x"] + display["w"], max(x1, x2))
        bottom = min(display["y"] + display["h"], max(y1, y2))
        self.pending_box = {
            "x": (left - display["x"]) / display["w"],
            "y": (top - display["y"]) / display["h"],
            "w": max(0, right - left) / display["w"],
            "h": max(0, bottom - top) / display["h"],
        }
        self.image_draw_start = None

    def add_pending_box(self):
        try:
            if not self.pending_box:
                raise ValueError("Draw a bounding box on the current image first.")
            item = validate_box_annotation({
                **self.pending_box, "label": self.label_var.get(),
                "image_path": self.current_image_path(), "source": "Manual",
                "status": ANNOTATION_STATUS_APPROVED, "confidence": 1.0,
            })
            self.annotations.append(item)
            self.pending_box = None
            self._remember_change(f"Added bounding box {item['id']}")
            self.refresh_annotations()
        except Exception as exc:
            messagebox.showerror("Bounding Box Error", str(exc), parent=self)

    def assign_image_class(self):
        path = self.current_image_path()
        label = " ".join(self.label_var.get().split()).strip()
        if not path or not label:
            messagebox.showwarning("Image and Label Required", "Load an image and enter a class.", parent=self)
            return
        # One active whole-image class per image; prior class becomes rejected evidence.
        for item in self.annotations:
            if item.get("kind") == "image_class" and item.get("image_path") == path and item.get("status") == ANNOTATION_STATUS_APPROVED:
                item["status"] = ANNOTATION_STATUS_REJECTED
        item = {
            "id": _new_annotation_id("image"), "data_type": "image", "kind": "image_class",
            "label": label, "image_path": path, "source": "Manual", "confidence": 1.0,
            "status": ANNOTATION_STATUS_APPROVED, "created_at": datetime.now().isoformat(), "notes": "",
        }
        self.annotations.append(item)
        self._remember_change(f"Assigned image class {item['id']}")
        self.refresh_annotations()

    def previous_image(self):
        if self.image_paths:
            self.image_index = (self.image_index - 1) % len(self.image_paths)
            self.pending_box = None
            self.display_current_image()

    def next_image(self):
        if self.image_paths:
            self.image_index = (self.image_index + 1) % len(self.image_paths)
            self.pending_box = None
            self.display_current_image()

    def refresh_annotations(self):
        for row in self.annotation_tree.get_children():
            self.annotation_tree.delete(row)
        for item in self.annotations:
            if item.get("data_type") == "signal":
                location = f"Rows {item['start']}–{item['end']}"
            elif item.get("kind") == "box":
                location = f"{Path(item['image_path']).name}: x={item['x']:.3f}, y={item['y']:.3f}, w={item['w']:.3f}, h={item['h']:.3f}"
            else:
                location = Path(item.get("image_path", "")).name
            self.annotation_tree.insert("", tk.END, iid=item["id"], values=(
                item.get("kind", ""), item.get("label", ""), location,
                item.get("source", ""), f"{float(item.get('confidence', 0)):.2f}", item.get("status", ""),
            ))
        if self.mode_var.get().startswith("1D"):
            self.plot_signal_annotations()
        else:
            self.display_current_image()

    def set_selected_status(self, status):
        selected = set(self.annotation_tree.selection())
        if not selected:
            return
        for item in self.annotations:
            if item["id"] in selected:
                item["status"] = status
        self._remember_change(f"Set {len(selected)} annotations to {status}")
        self.refresh_annotations()

    def delete_selected(self):
        selected = set(self.annotation_tree.selection())
        if not selected:
            return
        self.annotations = [item for item in self.annotations if item["id"] not in selected]
        self._remember_change(f"Deleted {len(selected)} annotations")
        self.refresh_annotations()

    def quality_report(self):
        counts = {status: 0 for status in (ANNOTATION_STATUS_APPROVED, ANNOTATION_STATUS_PENDING, ANNOTATION_STATUS_REJECTED)}
        labels = {}
        for item in self.annotations:
            counts[item.get("status", ANNOTATION_STATUS_PENDING)] = counts.get(item.get("status"), 0) + 1
            if item.get("status") == ANNOTATION_STATUS_APPROVED:
                labels[item.get("label")] = labels.get(item.get("label"), 0) + 1
        warnings = []
        if not counts[ANNOTATION_STATUS_APPROVED]:
            warnings.append("No reviewed/approved annotations are available for training.")
        if counts[ANNOTATION_STATUS_PENDING]:
            warnings.append(f"{counts[ANNOTATION_STATUS_PENDING]} AI/local suggestions still require review.")
        if len(labels) < 2:
            warnings.append("Classification training normally needs at least two approved classes.")
        approved_intervals = sorted(
            [
                item for item in self.annotations
                if item.get("data_type") == "signal"
                and item.get("kind") == "interval"
                and item.get("status") == ANNOTATION_STATUS_APPROVED
            ],
            key=lambda item: (item.get("start", 0), item.get("end", 0)),
        )
        conflicts = 0
        for previous, current in zip(approved_intervals, approved_intervals[1:]):
            if (
                current.get("start", 0) <= previous.get("end", -1)
                and current.get("label") != previous.get("label")
            ):
                conflicts += 1
        if conflicts:
            warnings.append(
                f"{conflicts} overlapping approved signal intervals have different labels; later labels take priority in CSV export."
            )
        if self.signal_df is not None and approved_intervals:
            coverage = np.zeros(len(self.signal_df), dtype=bool)
            for item in approved_intervals:
                coverage[item["start"]:item["end"] + 1] = True
            warnings.append(
                f"Approved signal intervals cover {100.0 * coverage.mean():.1f}% of rows. Unlabelled rows are excluded from the training handoff."
            )
        return {"counts": counts, "approved_labels": labels, "warnings": warnings}

    def show_quality_check(self):
        report = self.quality_report()
        lines = ["ANNOTATION QUALITY CHECK", "=" * 50, ""]
        lines.extend(f"{key}: {value}" for key, value in report["counts"].items())
        lines.append("\nApproved label distribution:")
        lines.extend(f"- {key}: {value}" for key, value in report["approved_labels"].items())
        lines.append("\nWarnings:")
        lines.extend(f"- {item}" for item in report["warnings"] or ["No basic consistency warning."])
        messagebox.showinfo("Annotation Quality Check", "\n".join(lines), parent=self)

    def compact_ai_profile(self):
        if self.mode_var.get().startswith("1D"):
            if self.signal_df is None:
                raise ValueError("Load signal data first.")
            numeric = self.signal_df.select_dtypes(include=[np.number])
            summaries = {}
            def finite_or_none(value):
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    return None
                return number if np.isfinite(number) else None
            for column in numeric.columns[:20]:
                series = numeric[column]
                summaries[str(column)] = {
                    "missing": int(series.isna().sum()),
                    "mean": finite_or_none(series.mean()) if series.notna().any() else None,
                    "std": finite_or_none(series.std()) if series.notna().sum() > 1 else None,
                    "min": finite_or_none(series.min()) if series.notna().any() else None,
                    "max": finite_or_none(series.max()) if series.notna().any() else None,
                }
            return {"data_type": "1D signal/table", "rows": len(self.signal_df), "columns": [str(c) for c in self.signal_df.columns], "numeric_summaries": summaries, "existing_label_names": self.label_names}
        if not self.image_paths:
            raise ValueError("Load images first.")
        extensions = {}
        for path in self.image_paths:
            extension = Path(path).suffix.lower()
            extensions[extension] = extensions.get(extension, 0) + 1
        return {"data_type": "2D images", "image_count": len(self.image_paths), "extension_counts": extensions, "existing_label_names": self.label_names, "existing_annotation_kinds": sorted({item.get('kind') for item in self.annotations})}

    def start_ai_plan(self):
        if self.ai_running:
            return
        try:
            profile = self.compact_ai_profile()
        except Exception as exc:
            messagebox.showwarning("Data Required", str(exc), parent=self)
            return
        self.ai_running = True
        self.status_var.set("Creating annotation plan using compact metadata...")
        self.notebook.select(self.ai_tab)
        goal = self.ai_goal_var.get()
        provider = self.owner.ai_provider_var.get()
        settings_snapshot = {
            "provider": provider,
            "api_key": self.owner.resolve_ai_api_key(provider),
            "model": self.owner.ai_model_var.get().strip(),
            "base_url": self.owner.ai_base_url_var.get().strip(),
            "timeout": self.owner.ai_timeout_var.get() or 90,
        }
        threading.Thread(
            target=self._ai_plan_worker,
            args=(profile, goal, settings_snapshot),
            daemon=True,
        ).start()
        self._ai_plan_poll_id = self.after(100, self._poll_ai_plan)

    def _ai_plan_worker(self, profile, goal, settings_snapshot):
        try:
            plan = request_ai_annotation_plan(settings_snapshot, profile, goal)
            self._ai_plan_queue.put(("success", plan))
        except Exception as exc:
            self._ai_plan_queue.put(("error", str(exc)))

    def _poll_ai_plan(self):
        """Deliver worker results on Tk's owning thread."""
        self._ai_plan_poll_id = None
        try:
            status, payload = self._ai_plan_queue.get_nowait()
        except queue.Empty:
            if self.ai_running:
                self._ai_plan_poll_id = self.after(100, self._poll_ai_plan)
            return
        if status == "success":
            self._ai_plan_finished(payload)
        else:
            self._ai_plan_failed(payload)

    def _cancel_ai_plan_poll(self, event):
        if event.widget is self and self._ai_plan_poll_id is not None:
            self.after_cancel(self._ai_plan_poll_id)
            self._ai_plan_poll_id = None

    def _ai_plan_finished(self, plan):
        self.ai_running = False
        self.ai_plan = plan
        for label in plan.get("suggested_labels", []):
            if label not in self.label_names:
                self.label_names.append(label)
        self.label_box.config(values=self.label_names)
        lines = ["AI ANNOTATION PLAN — HUMAN REVIEW REQUIRED", "=" * 72, f"Provider: {plan['provider']}", f"Model: {plan['model']}", f"Type: {plan['annotation_type']}", "", plan["summary"], "", "Suggested labels:"]
        lines.extend(f"- {item}" for item in plan["suggested_labels"])
        lines.append("\nLocal processing methods:")
        lines.extend(f"- {item}" for item in plan["local_methods"])
        lines.append("\nReview checks:")
        lines.extend(f"- {item}" for item in plan["review_checks"])
        lines.append("\nWarnings:")
        lines.extend(f"- {item}" for item in plan["warnings"] or ["No provider warning returned."])
        lines.append("\n" + plan["data_sent"])
        self.ai_plan_text.delete("1.0", tk.END)
        self.ai_plan_text.insert(tk.END, "\n".join(lines))
        self.status_var.set("AI plan received. Suggested labels were added to the selector but no data labels were auto-approved.")

    def _ai_plan_failed(self, message):
        self.ai_running = False
        self.status_var.set("AI annotation planning failed. Manual and local annotation remain available.")
        messagebox.showerror("AI Annotation Planner", message, parent=self)

    def project_payload(self):
        return {
            "schema_version": 1, "application": APPLICATION_NAME, "application_version": APP_VERSION,
            "saved_at": datetime.now().isoformat(), "mode": self.mode_var.get(),
            "signal_path": self.signal_path, "image_paths": self.image_paths,
            "labels": self.label_names, "annotations": self.annotations,
            "history": self.annotation_history, "ai_plan": self.ai_plan,
            "quality": self.quality_report(),
        }

    def save_project(self):
        path = filedialog.asksaveasfilename(title="Save Annotation Project", defaultextension=".json", filetypes=[("Annotation Project", "*.json")])
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(self.project_payload(), indent=2, default=_result_json_default), encoding="utf-8")
            self.status_var.set(f"Saved annotation project: {path}")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc), parent=self)

    def load_project(self):
        path = filedialog.askopenfilename(title="Load Annotation Project", filetypes=[("Annotation Project", "*.json")])
        if not path:
            return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            self.mode_var.set(payload.get("mode", "1D Signal / Table"))
            self.label_names = list(dict.fromkeys(payload.get("labels") or self.label_names))
            self.label_box.config(values=self.label_names)
            self.annotations = list(payload.get("annotations") or [])
            self.annotation_history = list(payload.get("history") or [])
            self.ai_plan = payload.get("ai_plan")
            signal_path = payload.get("signal_path")
            if signal_path and Path(signal_path).is_file():
                self.signal_df = pd.read_csv(signal_path) if Path(signal_path).suffix.lower() == ".csv" else pd.read_excel(signal_path)
                self.signal_path = signal_path
                numeric = list(self.signal_df.select_dtypes(include=[np.number]).columns)
                self.signal_column_box.config(values=numeric)
                if numeric:
                    self.signal_column_var.set(numeric[0])
            self.image_paths = [path for path in payload.get("image_paths", []) if Path(path).is_file()]
            self.image_index = 0
            self._refresh_mode()
            self.refresh_annotations()
            self.status_var.set("Annotation project loaded. Missing source files were skipped.")
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc), parent=self)

    def export_json(self):
        path = filedialog.asksaveasfilename(title="Export Annotation JSON", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            try:
                Path(path).write_text(json.dumps(self.project_payload(), indent=2, default=_result_json_default), encoding="utf-8")
                self.status_var.set(f"Exported annotation evidence: {path}")
            except Exception as exc:
                messagebox.showerror("Export Error", str(exc), parent=self)

    def export_signal_csv(self):
        try:
            output = build_signal_label_dataframe(self.signal_df, self.annotations)
            path = filedialog.asksaveasfilename(title="Export Labelled Signal CSV", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if path:
                output.to_csv(path, index=False)
                evidence = Path(path).with_suffix(".annotations.json")
                evidence.write_text(json.dumps(self.project_payload(), indent=2, default=_result_json_default), encoding="utf-8")
                self.status_var.set(f"Exported labelled signal data: {path}")
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc), parent=self)

    def approved_image_classes(self):
        result = {}
        for item in self.annotations:
            if item.get("kind") == "image_class" and item.get("status") == ANNOTATION_STATUS_APPROVED:
                result[item["image_path"]] = item["label"]
        return result

    def export_image_classes(self):
        classes = self.approved_image_classes()
        if not classes:
            messagebox.showwarning("No Approved Classes", "Approve whole-image class labels first.", parent=self)
            return
        directory = filedialog.askdirectory(title="Choose Image Classification Export Folder")
        if not directory:
            return
        try:
            root = Path(directory) / "annotated_image_classes"
            root.mkdir(parents=True, exist_ok=True)
            for source, label in classes.items():
                target_dir = root / re.sub(r"[^A-Za-z0-9_. -]+", "_", label)
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / Path(source).name
                if target.exists():
                    target = target_dir / f"{hashlib.sha256(source.encode()).hexdigest()[:8]}_{Path(source).name}"
                shutil.copy2(source, target)
            (root / "annotations.json").write_text(json.dumps(self.project_payload(), indent=2, default=_result_json_default), encoding="utf-8")
            self.status_var.set(f"Exported {len(classes)} classified images to {root}")
            return root
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc), parent=self)

    def export_yolo(self):
        boxes = [item for item in self.annotations if item.get("kind") == "box" and item.get("status") == ANNOTATION_STATUS_APPROVED]
        if not boxes:
            messagebox.showwarning("No Approved Boxes", "Approve bounding-box labels first.", parent=self)
            return
        directory = filedialog.askdirectory(title="Choose YOLO Export Folder")
        if not directory:
            return
        try:
            root = Path(directory) / "annotated_yolo_dataset"
            report = write_yolo_annotation_dataset(root, boxes)
            (root / "annotations.json").write_text(json.dumps(self.project_payload(), indent=2, default=_result_json_default), encoding="utf-8")
            self.status_var.set(
                f"Exported {report['box_count']} boxes for "
                f"{report['image_count']} images to {root}"
            )
        except Exception as exc:
            messagebox.showerror("YOLO Export Error", str(exc), parent=self)

    def continue_signal_training(self):
        try:
            output = build_signal_label_dataframe(self.signal_df, self.annotations)
            if output["annotation_label"].notna().sum() == 0:
                raise ValueError("No approved interval labels cover any signal rows.")
            self.owner.import_annotated_signal_dataframe(output, self.signal_path)
            self.destroy()
        except Exception as exc:
            messagebox.showerror("Training Handoff", str(exc), parent=self)

    def continue_image_training(self):
        classes = self.approved_image_classes()
        if len(set(classes.values())) < 2:
            messagebox.showwarning("Two Classes Required", "Approve at least two whole-image classes before image-classification training.", parent=self)
            return
        directory = filedialog.askdirectory(title="Choose Prepared Training Dataset Folder")
        if not directory:
            return
        try:
            root = Path(directory) / "nn_studio_annotated_training_images"
            root.mkdir(parents=True, exist_ok=True)
            for source, label in classes.items():
                target_dir = root / re.sub(r"[^A-Za-z0-9_. -]+", "_", label)
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / f"{hashlib.sha256(source.encode()).hexdigest()[:8]}_{Path(source).name}"
                shutil.copy2(source, target)
            (root / "annotations.json").write_text(json.dumps(self.project_payload(), indent=2, default=_result_json_default), encoding="utf-8")
            self.owner.import_annotated_image_folder(root)
            self.destroy()
        except Exception as exc:
            messagebox.showerror("Training Handoff", str(exc), parent=self)
