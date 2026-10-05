"""Ui / detection for NN Training Studio."""

from pathlib import Path
from tkinter.scrolledtext import ScrolledText
from datetime import datetime
from tkinter import filedialog
import json
from tkinter import messagebox
import numpy as np
import os
import pandas as pd
import platform
import queue
import shutil
import subprocess
import tempfile
import threading
import tkinter as tk
import traceback
from tkinter import ttk
from nn_training_studio.constants import (
    APP_VERSION,
    DATA_MODE_DETECTION,
    DEVICE_AUTO,
    SUPPORTED_IMAGE_EXTENSIONS,
    YOLO_DEFAULT_MODELS,
    YOLO_EXPORT_FORMATS,
)
from nn_training_studio.deployment import (
    detect_saved_model_type,
)
from nn_training_studio.detection import (
    _find_detection_weights,
    _object_detection_run_root,
    create_yolo_preview_image,
    format_yolo_validation_report,
    load_detection_result_bundle,
    save_detection_result_bundle,
    validate_yolo_dataset,
)
from nn_training_studio.devices import (
    DeviceManager,
)
from nn_training_studio.results import (
    _result_json_default,
    zip_result_directory,
)
from nn_training_studio.ui.results import (
    CustomResultsStudioWindow,
)


class ObjectDetectionWindow(tk.Toplevel):
    """Dedicated Ultralytics YOLO training, evaluation, and inference UI."""

    def __init__(
        self,
        parent,
        initial_model_path=None,
        application_mode=False,
    ):
        super().__init__(parent)
        self.application_mode = bool(application_mode or initial_model_path)
        self.title(
            f"NN Training Studio {APP_VERSION} — "
            + (
                "Object Detection Application"
                if self.application_mode
                else "Object Detection Training"
            )
        )
        self.geometry("1280x840")
        self.minsize(1080, 720)

        self.data_yaml_path = None
        self.dataset_config = None
        self.validation_report = None
        self.dataset_records = []
        self.current_weights_path = None
        self.current_run_directory = None
        self.loaded_package = None
        self.detector_trainable = True
        self.detector_inference_ready = False
        self.inference_directories = []
        self.latest_evaluation_directory = None
        self.latest_evaluation_metrics = {}
        self.latest_inference_summary = {}
        self.custom_result_directories = []
        self.detection_preview_files = []
        self.detection_preview_index = 0
        self.detection_preview_photo = None
        self.detection_preview_directory = None
        self.worker_queue = queue.Queue()
        self.operation_running = False
        self.stop_event = threading.Event()
        self.active_yolo_model = None

        self.model_var = tk.StringVar(value="yolo26n.pt")
        self.custom_weights_var = tk.StringVar(value="")
        self.epochs_var = tk.StringVar(value="100")
        self.batch_var = tk.StringVar(value="16")
        self.image_size_var = tk.StringVar(value="640")
        self.patience_var = tk.StringVar(value="30")
        self.learning_rate_var = tk.StringVar(value="0.01")
        self.optimizer_var = tk.StringVar(value="auto")
        self.training_device_var = tk.StringVar(value=DEVICE_AUTO)
        self.inference_device_var = tk.StringVar(value=DEVICE_AUTO)
        self.training_device_status_var = tk.StringVar(value="")
        self.inference_device_status_var = tk.StringVar(value="")
        self.workers_var = tk.StringVar(value="4")
        self.project_name_var = tk.StringVar(value="object_detector")
        self.cache_var = tk.BooleanVar(value=False)
        self.amp_var = tk.BooleanVar(value=True)
        self.pretrained_var = tk.BooleanVar(value=True)
        self.resume_var = tk.BooleanVar(value=False)

        self.inference_source_var = tk.StringVar(value="")
        self.confidence_var = tk.StringVar(value="0.25")
        self.iou_var = tk.StringVar(value="0.70")
        self.save_txt_var = tk.BooleanVar(value=True)
        self.save_conf_var = tk.BooleanVar(value=True)
        self.inference_status_var = tk.StringVar(
            value="Load or train detector weights, then select an image/folder."
        )
        self.header_title_var = tk.StringVar(
            value=(
                "Object Detection Application"
                if self.application_mode
                else "Object Detection Training"
            )
        )
        self.workflow_description_var = tk.StringVar(
            value=(
                "Loaded-model workflow: review detector → select an image, "
                "folder, video, or webcam → run detection → inspect annotated "
                "preview and detection table → export results"
                if self.application_mode
                else (
                    "YOLO workflow: dataset and bounding-box validation → "
                    "training → mAP evaluation → inference → result/model export"
                )
            )
        )
        self.loaded_model_summary_var = tk.StringVar(
            value="No detector loaded yet."
        )
        self.preview_counter_var = tk.StringVar(
            value="No annotated prediction preview available."
        )
        self.export_format_var = tk.StringVar(value=YOLO_EXPORT_FORMATS[0])
        self.status_var = tk.StringVar(
            value="Select a YOLO data.yaml file to begin."
        )

        self.protocol("WM_DELETE_WINDOW", self.close_window)
        self.build_interface()
        self.refresh_device_options()
        self.after(100, self.process_worker_queue)
        if initial_model_path:
            self.after_idle(
                lambda: self.load_detector_path(initial_model_path)
            )

    def build_interface(self):
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=12, pady=(10, 5))
        ttk.Label(
            header,
            textvariable=self.header_title_var,
            font=("Arial", 19, "bold"),
        ).pack(side=tk.LEFT)
        ttk.Label(
            header,
            textvariable=self.status_var,
            foreground="#245a85",
        ).pack(side=tk.RIGHT, padx=8)

        ttk.Label(
            self,
            textvariable=self.workflow_description_var,
        ).pack(anchor="w", padx=14, pady=(0, 7))

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=6)
        self.dataset_tab = ttk.Frame(self.notebook)
        self.training_tab = ttk.Frame(self.notebook)
        self.results_tab = ttk.Frame(self.notebook)
        self.inference_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.dataset_tab, text="1. Dataset & Annotations")
        self.notebook.add(self.training_tab, text="2. Model & Training")
        self.notebook.add(self.results_tab, text="3. Evaluation & Export")
        self.notebook.add(self.inference_tab, text="4. Detection / Prediction")

        self.build_dataset_tab()
        self.build_training_tab()
        self.build_results_tab()
        self.build_inference_tab()

    def build_dataset_tab(self):
        actions = ttk.Frame(self.dataset_tab)
        actions.pack(fill=tk.X, padx=10, pady=9)
        ttk.Button(
            actions,
            text="Select data.yaml",
            command=self.select_data_yaml,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            actions,
            text="Load Detection Package",
            command=self.load_detection_package,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(
            actions,
            text="Validate Dataset",
            command=self.start_dataset_validation,
        ).pack(side=tk.LEFT, padx=4)
        self.preview_button = ttk.Button(
            actions,
            text="Preview Bounding Boxes",
            command=self.preview_annotations,
            state=tk.DISABLED,
        )
        self.preview_button.pack(side=tk.LEFT, padx=4)

        format_frame = ttk.LabelFrame(
            self.dataset_tab,
            text="Expected Ultralytics YOLO Dataset",
        )
        format_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(
            format_frame,
            text=(
                "data.yaml defines path, train, val, optional test, and names. "
                "Each image uses a matching labels/.../*.txt file with one "
                "normalised box per line: class_id x_center y_center width height. "
                "Missing/empty labels are reported as background images."
            ),
            wraplength=1160,
        ).pack(anchor="w", padx=9, pady=8)

        pane = ttk.PanedWindow(self.dataset_tab, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=7)
        report_frame = ttk.LabelFrame(pane, text="Validation Report")
        class_frame = ttk.LabelFrame(pane, text="Class Distribution")
        pane.add(report_frame, weight=3)
        pane.add(class_frame, weight=2)

        self.dataset_report_text = ScrolledText(
            report_frame,
            wrap=tk.WORD,
            height=22,
        )
        self.dataset_report_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=6,
            pady=6,
        )
        self.dataset_report_text.insert(
            tk.END,
            "No object-detection dataset loaded.\n",
        )

        self.class_tree = ttk.Treeview(
            class_frame,
            columns=("id", "class", "instances"),
            show="headings",
            height=18,
        )
        self.class_tree.heading("id", text="ID")
        self.class_tree.heading("class", text="Class")
        self.class_tree.heading("instances", text="Instances")
        self.class_tree.column("id", width=55, anchor="center")
        self.class_tree.column("class", width=180)
        self.class_tree.column("instances", width=100, anchor="e")
        self.class_tree.pack(
            fill=tk.BOTH,
            expand=True,
            padx=6,
            pady=6,
        )

    def build_training_tab(self):
        pane = ttk.PanedWindow(self.training_tab, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=9)
        settings = ttk.LabelFrame(pane, text="YOLO Training Configuration")
        log_frame = ttk.LabelFrame(pane, text="Live Training Log")
        pane.add(settings, weight=2)
        pane.add(log_frame, weight=3)
        settings.columnconfigure(1, weight=1)

        rows = [
            ("Base model:", self.model_var, YOLO_DEFAULT_MODELS),
            ("Epochs:", self.epochs_var, None),
            ("Batch size (-1 = auto):", self.batch_var, None),
            ("Image size:", self.image_size_var, None),
            ("Early-stopping patience:", self.patience_var, None),
            ("Initial learning rate:", self.learning_rate_var, None),
            (
                "Optimizer:",
                self.optimizer_var,
                [
                    "auto",
                    "MuSGD",
                    "SGD",
                    "Adam",
                    "AdamW",
                    "NAdam",
                    "RAdam",
                    "RMSProp",
                ],
            ),
            (
                "Training device:",
                self.training_device_var,
                DeviceManager.pytorch_options(),
            ),
            ("Data-loader workers:", self.workers_var, None),
            ("Project / run name:", self.project_name_var, None),
        ]
        for row_index, (label, variable, values) in enumerate(rows):
            ttk.Label(settings, text=label).grid(
                row=row_index,
                column=0,
                sticky="w",
                padx=8,
                pady=5,
            )
            if values is None:
                widget = ttk.Entry(settings, textvariable=variable)
            else:
                widget = ttk.Combobox(
                    settings,
                    textvariable=variable,
                    values=values,
                    state="readonly",
                )
            widget.grid(
                row=row_index,
                column=1,
                sticky="ew",
                padx=8,
                pady=5,
            )
            if variable is self.training_device_var:
                self.training_device_combo = widget
                widget.bind(
                    "<<ComboboxSelected>>",
                    lambda _event: self.update_device_status_labels(),
                )

        device_status_row = ttk.Frame(settings)
        device_status_row.grid(
            row=len(rows),
            column=0,
            columnspan=2,
            sticky="ew",
            padx=8,
            pady=(0, 5),
        )
        ttk.Label(
            device_status_row,
            textvariable=self.training_device_status_var,
            foreground="#245a85",
            wraplength=430,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(
            device_status_row,
            text="Refresh Devices",
            command=self.refresh_device_options,
        ).pack(side=tk.RIGHT)

        custom_row = ttk.Frame(settings)
        custom_row.grid(
            row=len(rows) + 1,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=8,
            pady=5,
        )
        custom_row.columnconfigure(0, weight=1)
        ttk.Entry(
            custom_row,
            textvariable=self.custom_weights_var,
        ).grid(row=0, column=0, sticky="ew")
        ttk.Button(
            custom_row,
            text="Browse Custom .pt / .yaml",
            command=self.select_custom_weights,
        ).grid(row=0, column=1, padx=(6, 0))

        option_frame = ttk.LabelFrame(settings, text="Training Options")
        option_frame.grid(
            row=len(rows) + 2,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=8,
            pady=7,
        )
        for text_value, variable in [
            ("Use cache", self.cache_var),
            ("Mixed precision (AMP)", self.amp_var),
            ("Use pretrained weights", self.pretrained_var),
            ("Resume from last checkpoint", self.resume_var),
        ]:
            ttk.Checkbutton(
                option_frame,
                text=text_value,
                variable=variable,
            ).pack(anchor="w", padx=7, pady=3)

        action_frame = ttk.Frame(settings)
        action_frame.grid(
            row=len(rows) + 3,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=8,
            pady=10,
        )
        self.train_button = ttk.Button(
            action_frame,
            text="Start Training",
            command=self.start_training,
        )
        self.train_button.pack(side=tk.LEFT, padx=3)
        self.stop_button = ttk.Button(
            action_frame,
            text="Stop Safely",
            command=self.stop_training,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT, padx=3)
        ttk.Button(
            action_frame,
            text="Load Detector File",
            command=self.load_existing_weights,
        ).pack(side=tk.LEFT, padx=3)
        ttk.Button(
            action_frame,
            text="Load Exported Folder",
            command=self.load_exported_detector_folder,
        ).pack(side=tk.LEFT, padx=3)

        self.training_log_text = ScrolledText(
            log_frame,
            wrap=tk.WORD,
            height=30,
        )
        self.training_log_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=6,
            pady=6,
        )
        self.training_log_text.insert(
            tk.END,
            "Training has not started.\n",
        )

    def build_results_tab(self):
        actions = ttk.Frame(self.results_tab)
        actions.pack(fill=tk.X, padx=10, pady=9)
        self.evaluate_button = ttk.Button(
            actions,
            text="Evaluate best.pt",
            command=self.start_evaluation,
            state=tk.DISABLED,
        )
        self.evaluate_button.pack(side=tk.LEFT, padx=4)
        self.save_bundle_button = ttk.Button(
            actions,
            text="Save Complete Detection Package",
            command=self.save_complete_bundle,
            state=tk.DISABLED,
        )
        self.save_bundle_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(
            actions,
            text="Open Results Folder",
            command=self.open_current_results_folder,
        ).pack(side=tk.LEFT, padx=4)
        self.custom_results_button = ttk.Button(
            actions,
            text="Open Custom Results Studio",
            command=self.open_custom_results_studio,
            state=tk.DISABLED,
        )
        self.custom_results_button.pack(side=tk.LEFT, padx=4)

        export_frame = ttk.LabelFrame(
            self.results_tab,
            text="Deploy / Export Detector",
        )
        export_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(export_frame, text="Format:").pack(
            side=tk.LEFT,
            padx=(8, 4),
            pady=8,
        )
        ttk.Combobox(
            export_frame,
            textvariable=self.export_format_var,
            values=YOLO_EXPORT_FORMATS,
            state="readonly",
            width=28,
        ).pack(side=tk.LEFT, padx=4, pady=8)
        self.export_button = ttk.Button(
            export_frame,
            text="Export Best Detector",
            command=self.start_model_export,
            state=tk.DISABLED,
        )
        self.export_button.pack(side=tk.LEFT, padx=6, pady=8)

        self.result_log_text = ScrolledText(
            self.results_tab,
            wrap=tk.WORD,
            height=28,
        )
        self.result_log_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=7,
        )
        self.result_log_text.insert(
            tk.END,
            "Evaluation reports precision, recall, mAP50, mAP50–95, "
            "per-class AP, losses, plots, and confusion matrices generated "
            "by Ultralytics.\n",
        )

    def build_inference_tab(self):
        model_frame = ttk.LabelFrame(
            self.inference_tab,
            text="1. Loaded Detector",
        )
        model_frame.pack(fill=tk.X, padx=10, pady=(9, 4))
        ttk.Label(
            model_frame,
            textvariable=self.loaded_model_summary_var,
            foreground="#245a85",
            wraplength=870,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8, pady=7)
        ttk.Button(
            model_frame,
            text="Load / Change Detector...",
            command=self.load_existing_weights,
        ).pack(side=tk.RIGHT, padx=(4, 8), pady=7)
        ttk.Button(
            model_frame,
            text="Load Exported Folder...",
            command=self.load_exported_detector_folder,
        ).pack(side=tk.RIGHT, padx=4, pady=7)

        source_frame = ttk.LabelFrame(
            self.inference_tab,
            text="2. Select Application Input",
        )
        source_frame.pack(fill=tk.X, padx=10, pady=4)
        source_frame.columnconfigure(0, weight=1)
        ttk.Entry(
            source_frame,
            textvariable=self.inference_source_var,
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ttk.Button(
            source_frame,
            text="Select Image / Video",
            command=self.select_inference_file,
        ).grid(row=0, column=1, padx=4, pady=8)
        ttk.Button(
            source_frame,
            text="Select Folder",
            command=self.select_inference_folder,
        ).grid(row=0, column=2, padx=4, pady=8)
        ttk.Button(
            source_frame,
            text="Use Webcam 0",
            command=self.select_webcam,
        ).grid(row=0, column=3, padx=4, pady=8)

        options = ttk.LabelFrame(
            self.inference_tab,
            text="3. Detection Controls",
        )
        options.pack(fill=tk.X, padx=10, pady=4)
        ttk.Label(options, text="Inference device:").grid(
            row=0, column=0, padx=7, pady=7
        )
        self.inference_device_combo = ttk.Combobox(
            options,
            textvariable=self.inference_device_var,
            values=DeviceManager.pytorch_options(),
            state="readonly",
            width=32,
        )
        self.inference_device_combo.grid(
            row=0, column=1, columnspan=2, padx=7, pady=7, sticky="w"
        )
        self.inference_device_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self.update_device_status_labels(),
        )
        ttk.Label(
            options,
            textvariable=self.inference_device_status_var,
            foreground="#245a85",
        ).grid(
            row=0, column=3, columnspan=3, padx=7, pady=7, sticky="w"
        )
        ttk.Label(options, text="Confidence threshold:").grid(
            row=1, column=0, padx=7, pady=7
        )
        ttk.Entry(
            options,
            textvariable=self.confidence_var,
            width=10,
        ).grid(row=1, column=1, padx=7, pady=7)
        ttk.Label(options, text="NMS IoU threshold:").grid(
            row=1, column=2, padx=7, pady=7
        )
        ttk.Entry(
            options,
            textvariable=self.iou_var,
            width=10,
        ).grid(row=1, column=3, padx=7, pady=7)
        ttk.Checkbutton(
            options,
            text="Save YOLO text labels",
            variable=self.save_txt_var,
        ).grid(row=1, column=4, padx=7, pady=7)
        ttk.Checkbutton(
            options,
            text="Include confidence in labels",
            variable=self.save_conf_var,
        ).grid(row=1, column=5, padx=7, pady=7)

        action_row = ttk.Frame(self.inference_tab)
        action_row.pack(fill=tk.X, padx=10, pady=(5, 4))
        self.predict_button = ttk.Button(
            action_row,
            text="4. Run Object Detection",
            command=self.start_inference,
            state=tk.DISABLED,
        )
        self.predict_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(
            action_row,
            text="Open Latest Prediction Folder",
            command=self.open_latest_prediction_folder,
        ).pack(side=tk.LEFT, padx=4)
        self.save_prediction_results_button = ttk.Button(
            action_row,
            text="Save Prediction Results...",
            command=self.save_prediction_results,
            state=tk.DISABLED,
        )
        self.save_prediction_results_button.pack(side=tk.LEFT, padx=4)
        ttk.Button(
            action_row,
            text="Custom Detection Results",
            command=self.open_custom_results_studio,
        ).pack(side=tk.LEFT, padx=4)
        ttk.Label(
            action_row,
            textvariable=self.inference_status_var,
            foreground="#245a85",
        ).pack(side=tk.LEFT, padx=12)

        result_pane = ttk.PanedWindow(
            self.inference_tab,
            orient=tk.HORIZONTAL,
        )
        result_pane.pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=(3, 8),
        )
        preview_frame = ttk.LabelFrame(
            result_pane,
            text="5. Annotated Result Preview",
        )
        detail_frame = ttk.LabelFrame(
            result_pane,
            text="6. Inspect Detections and Log",
        )
        result_pane.add(preview_frame, weight=3)
        result_pane.add(detail_frame, weight=2)

        self.detection_preview_host = tk.Frame(
            preview_frame,
            background="#202020",
        )
        self.detection_preview_host.pack(
            fill=tk.BOTH,
            expand=True,
            padx=7,
            pady=(7, 3),
        )
        self.detection_preview_label = tk.Label(
            self.detection_preview_host,
            text=(
                "Run object detection to preview annotated results here.\n\n"
                "Image, folder, video, and webcam results are supported."
            ),
            foreground="white",
            background="#202020",
            justify=tk.CENTER,
        )
        self.detection_preview_label.pack(fill=tk.BOTH, expand=True)

        preview_actions = ttk.Frame(preview_frame)
        preview_actions.pack(fill=tk.X, padx=7, pady=(3, 7))
        self.previous_preview_button = ttk.Button(
            preview_actions,
            text="← Previous",
            command=lambda: self.change_detection_preview(-1),
            state=tk.DISABLED,
        )
        self.previous_preview_button.pack(side=tk.LEFT, padx=2)
        self.next_preview_button = ttk.Button(
            preview_actions,
            text="Next →",
            command=lambda: self.change_detection_preview(1),
            state=tk.DISABLED,
        )
        self.next_preview_button.pack(side=tk.LEFT, padx=2)
        self.full_preview_button = ttk.Button(
            preview_actions,
            text="Open Full-Size Preview",
            command=self.open_full_detection_preview,
            state=tk.DISABLED,
        )
        self.full_preview_button.pack(side=tk.LEFT, padx=5)
        ttk.Label(
            preview_actions,
            textvariable=self.preview_counter_var,
            anchor=tk.E,
        ).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=6)

        detail_notebook = ttk.Notebook(detail_frame)
        detail_notebook.pack(
            fill=tk.BOTH,
            expand=True,
            padx=6,
            pady=6,
        )
        detection_table_tab = ttk.Frame(detail_notebook)
        inference_log_tab = ttk.Frame(detail_notebook)
        detail_notebook.add(
            detection_table_tab,
            text="Detected Objects",
        )
        detail_notebook.add(
            inference_log_tab,
            text="Application Log",
        )

        detection_columns = (
            "source",
            "class",
            "confidence",
            "box",
        )
        self.detection_result_tree = ttk.Treeview(
            detection_table_tab,
            columns=detection_columns,
            show="headings",
            height=12,
        )
        headings = {
            "source": "Source",
            "class": "Class",
            "confidence": "Confidence",
            "box": "Bounding Box (x1, y1, x2, y2)",
        }
        widths = {
            "source": 145,
            "class": 120,
            "confidence": 85,
            "box": 210,
        }
        for column in detection_columns:
            self.detection_result_tree.heading(
                column,
                text=headings[column],
            )
            self.detection_result_tree.column(
                column,
                width=widths[column],
                anchor=tk.CENTER,
            )
        vertical_scroll = ttk.Scrollbar(
            detection_table_tab,
            orient=tk.VERTICAL,
            command=self.detection_result_tree.yview,
        )
        horizontal_scroll = ttk.Scrollbar(
            detection_table_tab,
            orient=tk.HORIZONTAL,
            command=self.detection_result_tree.xview,
        )
        self.detection_result_tree.configure(
            yscrollcommand=vertical_scroll.set,
            xscrollcommand=horizontal_scroll.set,
        )
        self.detection_result_tree.grid(
            row=0,
            column=0,
            sticky="nsew",
        )
        vertical_scroll.grid(row=0, column=1, sticky="ns")
        horizontal_scroll.grid(row=1, column=0, sticky="ew")
        detection_table_tab.rowconfigure(0, weight=1)
        detection_table_tab.columnconfigure(0, weight=1)

        self.inference_log_text = ScrolledText(
            inference_log_tab,
            wrap=tk.WORD,
            height=16,
        )
        self.inference_log_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=5,
            pady=5,
        )
        self.inference_log_text.insert(
            tk.END,
            "Annotated results will be previewed inside this page and saved "
            "to the prediction results folder. A machine-readable detection "
            "CSV is also created.\n",
        )

    def reset_detection_preview(self, message=None):
        self.detection_preview_files = []
        self.detection_preview_index = 0
        self.detection_preview_photo = None
        self.detection_preview_directory = None
        self.detection_preview_label.config(
            image="",
            text=(
                message
                or "Run object detection to preview annotated results here."
            ),
        )
        self.previous_preview_button.config(state=tk.DISABLED)
        self.next_preview_button.config(state=tk.DISABLED)
        self.full_preview_button.config(state=tk.DISABLED)
        self.preview_counter_var.set(
            "No annotated prediction preview available."
        )
        for item in self.detection_result_tree.get_children():
            self.detection_result_tree.delete(item)

    def load_detection_preview(self, prediction_directory):
        """Load annotated prediction outputs and detection rows into the UI."""
        directory = Path(prediction_directory)
        self.reset_detection_preview()
        self.detection_preview_directory = str(directory)

        preview_path = directory / "nn_studio_preview.jpg"
        output_images = sorted(
            (
                path
                for path in directory.rglob("*")
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
                and path.name != preview_path.name
            ),
            key=lambda value: str(value).lower(),
        )
        if output_images:
            self.detection_preview_files = output_images
        elif preview_path.is_file():
            self.detection_preview_files = [preview_path]

        detection_csv = directory / "nn_studio_detections.csv"
        if detection_csv.is_file():
            try:
                detection_frame = pd.read_csv(detection_csv)
                for _, row in detection_frame.head(5000).iterrows():
                    box_text = (
                        f"{float(row.get('x1', 0)):.1f}, "
                        f"{float(row.get('y1', 0)):.1f}, "
                        f"{float(row.get('x2', 0)):.1f}, "
                        f"{float(row.get('y2', 0)):.1f}"
                    )
                    self.detection_result_tree.insert(
                        "",
                        tk.END,
                        values=(
                            Path(str(row.get("source", ""))).name,
                            row.get("class_name", row.get("class_id", "")),
                            (
                                f"{float(row.get('confidence', 0)):.3f}"
                                if pd.notna(row.get("confidence"))
                                else ""
                            ),
                            box_text,
                        ),
                    )
            except Exception as exc:
                self.append_log(
                    self.inference_log_text,
                    f"Detection table preview could not be read: {exc}",
                )

        if self.detection_preview_files:
            self.detection_preview_index = 0
            self.display_current_detection_preview()
            has_multiple = len(self.detection_preview_files) > 1
            self.previous_preview_button.config(
                state=tk.NORMAL if has_multiple else tk.DISABLED
            )
            self.next_preview_button.config(
                state=tk.NORMAL if has_multiple else tk.DISABLED
            )
            self.full_preview_button.config(state=tk.NORMAL)
        else:
            self.detection_preview_label.config(
                text=(
                    "Detection completed, but no still-image preview was "
                    "found.\nUse Open Latest Prediction Folder for the saved "
                    "video or other output."
                )
            )
            self.preview_counter_var.set("No still-image preview found.")

    def display_current_detection_preview(self):
        if not self.detection_preview_files:
            return
        preview_path = self.detection_preview_files[
            self.detection_preview_index
        ]
        try:
            from PIL import Image, ImageTk

            self.update_idletasks()
            available_width = max(
                480,
                self.detection_preview_host.winfo_width() - 20,
            )
            available_height = max(
                260,
                self.detection_preview_host.winfo_height() - 20,
            )
            with Image.open(preview_path) as source:
                display_image = source.convert("RGB")
                resampling = getattr(
                    getattr(Image, "Resampling", Image),
                    "LANCZOS",
                )
                display_image.thumbnail(
                    (available_width, available_height),
                    resampling,
                )
            self.detection_preview_photo = ImageTk.PhotoImage(display_image)
            self.detection_preview_label.config(
                image=self.detection_preview_photo,
                text="",
            )
            self.preview_counter_var.set(
                f"{self.detection_preview_index + 1} / "
                f"{len(self.detection_preview_files)} — {preview_path.name}"
            )
        except Exception as exc:
            self.detection_preview_label.config(
                image="",
                text=f"Preview unavailable:\n{exc}",
            )
            self.preview_counter_var.set(preview_path.name)

    def change_detection_preview(self, offset):
        if len(self.detection_preview_files) < 2:
            return
        self.detection_preview_index = (
            self.detection_preview_index + int(offset)
        ) % len(self.detection_preview_files)
        self.display_current_detection_preview()

    def open_full_detection_preview(self):
        if not self.detection_preview_files:
            return
        preview_path = self.detection_preview_files[
            self.detection_preview_index
        ]
        try:
            from PIL import Image, ImageTk

            window = tk.Toplevel(self)
            window.title(f"Detection Preview — {preview_path.name}")
            window.geometry("1120x820")
            image_host = tk.Frame(window, background="#202020")
            image_host.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
            with Image.open(preview_path) as source:
                display_image = source.convert("RGB")
                resampling = getattr(
                    getattr(Image, "Resampling", Image),
                    "LANCZOS",
                )
                display_image.thumbnail((1060, 740), resampling)
            photo = ImageTk.PhotoImage(display_image)
            label = tk.Label(
                image_host,
                image=photo,
                background="#202020",
            )
            label.image = photo
            label.pack(fill=tk.BOTH, expand=True)
            ttk.Label(
                window,
                text=str(preview_path),
                wraplength=1080,
            ).pack(padx=8, pady=(0, 8))
        except Exception as exc:
            messagebox.showerror(
                "Preview Error",
                str(exc),
                parent=self,
            )

    def set_operation_state(self, running, message=None):
        self.operation_running = bool(running)
        training_allowed = (
            self.detector_trainable
            or self.model_var.get() != "Custom weights..."
        )
        self.train_button.config(
            state=(
                tk.NORMAL
                if not running and training_allowed
                else tk.DISABLED
            )
        )
        self.stop_button.config(
            state=tk.NORMAL if running else tk.DISABLED
        )
        if message is not None:
            self.status_var.set(message)

    def refresh_device_options(self):
        options = DeviceManager.pytorch_options()
        if hasattr(self, "training_device_combo"):
            self.training_device_combo.config(values=options)
        if hasattr(self, "inference_device_combo"):
            self.inference_device_combo.config(values=options)
        if self.training_device_var.get() not in options:
            self.training_device_var.set(DEVICE_AUTO)
        if self.inference_device_var.get() not in options:
            self.inference_device_var.set(DEVICE_AUTO)
        self.update_device_status_labels()

    def update_device_status_labels(self):
        try:
            training_info = DeviceManager.resolve_yolo(
                self.training_device_var.get()
            )
            self.training_device_status_var.set(
                DeviceManager.format_status(training_info)
            )
        except Exception as exc:
            self.training_device_status_var.set(str(exc))
        try:
            inference_info = DeviceManager.resolve_yolo(
                self.inference_device_var.get()
            )
            self.inference_device_status_var.set(
                DeviceManager.format_status(inference_info)
            )
        except Exception as exc:
            self.inference_device_status_var.set(str(exc))

    def append_log(self, widget, message):
        widget.insert(tk.END, str(message).rstrip() + "\n")
        widget.see(tk.END)

    def select_data_yaml(self):
        path = filedialog.askopenfilename(
            title="Select Ultralytics YOLO data.yaml",
            filetypes=[
                ("YAML dataset", "*.yaml *.yml"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.data_yaml_path = str(Path(path).resolve())
        self.validation_report = None
        self.dataset_records = []
        self.dataset_report_text.delete("1.0", tk.END)
        self.dataset_report_text.insert(
            tk.END,
            f"Selected: {self.data_yaml_path}\nClick Validate Dataset.\n",
        )
        self.status_var.set("Dataset selected; validation required.")
        self.start_dataset_validation()

    def start_dataset_validation(self):
        if self.operation_running:
            return
        if not self.data_yaml_path:
            messagebox.showwarning(
                "Dataset Required",
                "Select a YOLO data.yaml file first.",
            )
            return
        self.set_operation_state(True, "Validating dataset and annotations...")
        self.stop_button.config(state=tk.DISABLED)
        self.preview_button.config(state=tk.DISABLED)
        threading.Thread(
            target=self.dataset_validation_worker,
            args=(self.data_yaml_path,),
            daemon=True,
        ).start()

    def dataset_validation_worker(self, data_yaml_path):
        try:
            config, report, records = validate_yolo_dataset(data_yaml_path)
            self.worker_queue.put(
                ("dataset_validated", config, report, records)
            )
        except Exception as exc:
            self.worker_queue.put(
                ("operation_error", "Dataset Validation Error", str(exc))
            )

    def display_validation_report(self):
        self.dataset_report_text.delete("1.0", tk.END)
        self.dataset_report_text.insert(
            tk.END,
            format_yolo_validation_report(self.validation_report),
        )
        for item in self.class_tree.get_children():
            self.class_tree.delete(item)
        for index, class_name in enumerate(
            self.validation_report.get("class_names", [])
        ):
            count = self.validation_report.get(
                "class_instance_counts",
                {},
            ).get(str(index), 0)
            self.class_tree.insert(
                "",
                tk.END,
                values=(index, class_name, count),
            )

    def preview_annotations(self):
        if not self.dataset_records or not self.validation_report:
            return
        record = next(
            (
                item
                for item in self.dataset_records
                if item.get("boxes") and not item.get("errors")
            ),
            self.dataset_records[0],
        )
        preview_path = Path(tempfile.gettempdir()) / (
            "nn_studio_yolo_annotation_preview.jpg"
        )
        try:
            create_yolo_preview_image(
                record,
                self.validation_report.get("class_names", []),
                preview_path,
            )
            from PIL import Image, ImageTk

            window = tk.Toplevel(self)
            window.title("YOLO Annotation Preview")
            window.geometry("900x700")
            with Image.open(preview_path) as source:
                source.thumbnail((850, 610))
                display_image = source.copy()
            photo = ImageTk.PhotoImage(display_image)
            label = ttk.Label(window, image=photo)
            label.image = photo
            label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            ttk.Label(
                window,
                text=(
                    f"{record['split']}: {record['image_path']} — "
                    f"{record['instance_count']} object(s)"
                ),
                wraplength=850,
            ).pack(padx=10, pady=(0, 10))
        except Exception as exc:
            messagebox.showerror("Preview Error", str(exc))

    def select_custom_weights(self):
        path = filedialog.askopenfilename(
            title="Select YOLO weights or architecture",
            filetypes=[
                ("YOLO model", "*.pt *.yaml *.yml"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.custom_weights_var.set(str(Path(path).resolve()))
            self.model_var.set("Custom weights...")

    def load_existing_weights(self):
        path = filedialog.askopenfilename(
            title="Load Existing YOLO Detector",
            filetypes=[
                (
                    "Supported detectors",
                    "*.pt *.yaml *.yml *.onnx *.engine *.torchscript "
                    "*.tflite *.pb *.xml *.zip",
                ),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.load_detector_path(path)

    def load_exported_detector_folder(self):
        path = filedialog.askdirectory(
            title="Load Exported Detector Folder"
        )
        if path:
            self.load_detector_path(path)

    def load_detector_path(self, path):
        try:
            model_type = detect_saved_model_type(path)
            if model_type == "detection_package":
                self.load_detection_package_path(path)
                return
            if model_type == "keras_package":
                raise ValueError(
                    "This is a signal/image-classification package. Open it "
                    "from the common Load Existing Model workspace."
                )

            resolved_path = str(Path(path).resolve())
            self.current_weights_path = resolved_path
            self.current_run_directory = None
            self.detector_trainable = (
                model_type == "detection_trainable"
            )
            self.detector_inference_ready = (
                Path(resolved_path).is_dir()
                or Path(resolved_path).suffix.lower() not in {".yaml", ".yml"}
            )
            self.custom_weights_var.set(resolved_path)
            self.model_var.set("Custom weights...")
            self.resume_var.set(False)
            self.application_mode = True
            self.header_title_var.set("Object Detection Application")
            self.workflow_description_var.set(
                "Loaded-model workflow: review detector → select an image, "
                "folder, video, or webcam → run detection → inspect annotated "
                "preview and detection table → export results"
            )
            self.title(
                f"NN Training Studio {APP_VERSION} — "
                "Object Detection Application"
            )
            self.reset_detection_preview(
                "Detector loaded. Select an input source and run object "
                "detection to create an annotated preview."
            )
            self.update_model_ready_state()
            mode_text = (
                "trainable / fine-tunable"
                if self.detector_trainable
                else "inference-only export"
            )
            readiness_text = (
                "Ready for prediction"
                if self.detector_inference_ready
                else "Architecture only; train or load weights before prediction"
            )
            self.loaded_model_summary_var.set(
                f"{Path(resolved_path).name} — {mode_text} — {readiness_text}"
            )
            self.status_var.set(
                f"Loaded detector: {Path(resolved_path).name}"
            )
            self.append_log(
                self.result_log_text,
                f"Existing detector loaded: {resolved_path}\n"
                f"Capability: {mode_text}",
            )
            self.notebook.select(self.inference_tab)
        except Exception as exc:
            messagebox.showerror("Detector Load Error", str(exc))

    def load_detection_package(self):
        path = filedialog.askopenfilename(
            title="Load NN Studio Detection Package",
            filetypes=[
                ("Detection package", "*.zip"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.load_detection_package_path(path)

    def load_detection_package_path(self, path):
        try:
            package = load_detection_result_bundle(path)
            previous = self.loaded_package
            self.loaded_package = package
            if previous:
                shutil.rmtree(
                    previous.get("temp_directory", ""),
                    ignore_errors=True,
                )
            self.current_weights_path = package["weights_path"]
            self.current_run_directory = package["temp_directory"]
            self.detector_trainable = True
            self.detector_inference_ready = True
            self.data_yaml_path = package.get("data_yaml_path")
            self.training_config_from_package(
                package.get("training_config", {})
            )
            report = package.get("validation_report", {})
            if report:
                self.validation_report = report
                self.dataset_records = []
                self.display_validation_report()
            self.custom_weights_var.set(self.current_weights_path)
            self.model_var.set("Custom weights...")
            self.application_mode = True
            self.header_title_var.set("Object Detection Application")
            self.workflow_description_var.set(
                "Loaded-model workflow: review detector → select an image, "
                "folder, video, or webcam → run detection → inspect annotated "
                "preview and detection table → export results"
            )
            self.title(
                f"NN Training Studio {APP_VERSION} — "
                "Object Detection Application"
            )
            self.loaded_model_summary_var.set(
                f"{Path(path).name} — complete trainable detector package — "
                f"weights: {Path(self.current_weights_path).name}"
            )
            self.update_model_ready_state()
            self.status_var.set(
                f"Detection package loaded: {Path(path).name}"
            )
            self.append_log(
                self.result_log_text,
                f"Loaded detection package: {path}\n"
                f"Weights: {self.current_weights_path}",
            )
            self.notebook.select(self.inference_tab)
            previous_predictions = sorted(
                Path(package["temp_directory"]).rglob(
                    "nn_studio_prediction.json"
                ),
                key=lambda candidate: candidate.stat().st_mtime,
            )
            if previous_predictions:
                latest_directory = str(previous_predictions[-1].parent)
                self.inference_directories.append(latest_directory)
                try:
                    self.latest_inference_summary = json.loads(
                        previous_predictions[-1].read_text(
                            encoding="utf-8"
                        )
                    )
                except Exception:
                    self.latest_inference_summary = {}
                if self.latest_inference_summary:
                    self.inference_status_var.set(
                        "Loaded saved prediction: "
                        f"{self.latest_inference_summary.get('detections', 0)} "
                        "detections in "
                        f"{self.latest_inference_summary.get('processed_items', 0)} "
                        "item(s)."
                    )
                self.after_idle(
                    lambda directory=latest_directory:
                    self.load_detection_preview(directory)
                )
                self.update_model_ready_state()
            else:
                self.reset_detection_preview(
                    "Detector package loaded. Select an input source and run "
                    "object detection to create an annotated preview."
                )
        except Exception as exc:
            messagebox.showerror("Package Error", str(exc))

    def training_config_from_package(self, config):
        mappings = [
            ("epochs", self.epochs_var),
            ("batch", self.batch_var),
            ("imgsz", self.image_size_var),
            ("patience", self.patience_var),
            ("lr0", self.learning_rate_var),
            ("optimizer", self.optimizer_var),
            ("workers", self.workers_var),
            ("name", self.project_name_var),
        ]
        for key, variable in mappings:
            if key in config and config[key] is not None:
                variable.set(str(config[key]))
        saved_training = str(config.get("device_selection") or "")
        if saved_training in DeviceManager.pytorch_options():
            self.training_device_var.set(saved_training)
        else:
            self.training_device_var.set(
                DeviceManager.yolo_label_from_backend(config.get("device"))
            )
        if config.get("inference_device_selection"):
            saved_inference = str(
                config["inference_device_selection"]
            )
            if saved_inference in DeviceManager.pytorch_options():
                self.inference_device_var.set(saved_inference)
        self.update_device_status_labels()

    def resolved_training_weights(self):
        if (
            self.model_var.get() == "Custom weights..."
            and not self.detector_trainable
        ):
            raise ValueError(
                "The loaded detector is an inference-only export. Load .pt "
                "weights or a complete detection package to resume or "
                "fine-tune training."
            )
        if self.resume_var.get():
            last_path = (
                _find_detection_weights(
                    self.current_run_directory,
                    "last.pt",
                )
                if self.current_run_directory
                else None
            )
            if last_path:
                return last_path
            if self.current_weights_path and Path(
                self.current_weights_path
            ).name.lower() == "last.pt":
                return self.current_weights_path
            raise ValueError(
                "Resume is enabled, but no last.pt checkpoint is available."
            )
        if self.model_var.get() == "Custom weights...":
            custom_path = self.custom_weights_var.get().strip()
            if not custom_path:
                raise ValueError("Select custom YOLO weights or architecture.")
            if not Path(custom_path).is_file():
                raise ValueError(f"Custom model does not exist: {custom_path}")
            return custom_path
        return self.model_var.get()

    def collect_training_config(self):
        def integer(variable, name, minimum, allow_minus_one=False):
            try:
                value = int(variable.get().strip())
            except Exception as exc:
                raise ValueError(f"{name} must be an integer.") from exc
            if allow_minus_one and value == -1:
                return value
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}.")
            return value

        try:
            learning_rate = float(self.learning_rate_var.get().strip())
        except Exception as exc:
            raise ValueError("Learning rate must be numeric.") from exc
        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than zero.")
        project_name = self.project_name_var.get().strip()
        safe_name = "".join(
            character
            if character.isalnum() or character in {"-", "_"}
            else "_"
            for character in project_name
        ).strip("_")
        if not safe_name:
            raise ValueError("Enter a project/run name.")
        device_info = DeviceManager.resolve_yolo(
            self.training_device_var.get()
        )
        return {
            "model": self.resolved_training_weights(),
            "data": self.data_yaml_path,
            "epochs": integer(self.epochs_var, "Epochs", 1),
            "batch": integer(
                self.batch_var,
                "Batch size",
                1,
                allow_minus_one=True,
            ),
            "imgsz": integer(self.image_size_var, "Image size", 32),
            "patience": integer(self.patience_var, "Patience", 0),
            "lr0": learning_rate,
            "optimizer": self.optimizer_var.get(),
            "device": device_info["backend_value"],
            "device_selection": device_info["selected"],
            "resolved_device": device_info["actual_device"],
            "inference_device_selection": self.inference_device_var.get(),
            "workers": integer(self.workers_var, "Workers", 0),
            "name": safe_name,
            "project": str(_object_detection_run_root()),
            "cache": self.cache_var.get(),
            "amp": self.amp_var.get(),
            "pretrained": self.pretrained_var.get(),
            "resume": self.resume_var.get(),
            "exist_ok": False,
            "plots": True,
            "verbose": True,
        }

    def start_training(self):
        if self.operation_running:
            return
        if not self.validation_report:
            messagebox.showwarning(
                "Validation Required",
                "Load and validate a YOLO dataset before training.",
            )
            return
        if not self.validation_report.get("is_valid"):
            messagebox.showerror(
                "Invalid Dataset",
                "Correct the invalid YOLO annotations before training.",
            )
            return
        try:
            config = self.collect_training_config()
        except Exception as exc:
            messagebox.showerror("Training Configuration", str(exc))
            return
        if self.validation_report.get("warnings"):
            proceed = messagebox.askyesno(
                "Dataset Warnings",
                "The dataset has warnings, commonly missing/empty labels that "
                "will be treated as background images.\n\nContinue training?",
            )
            if not proceed:
                return
        self.stop_event.clear()
        self.training_log_text.delete("1.0", tk.END)
        self.append_log(
            self.training_log_text,
            "Starting YOLO object-detection training...",
        )
        self.append_log(
            self.training_log_text,
            json.dumps(config, indent=2, default=_result_json_default),
        )
        self.set_operation_state(True, "YOLO training is running...")
        threading.Thread(
            target=self.training_worker,
            args=(config,),
            daemon=True,
        ).start()

    def training_worker(self, config):
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self.worker_queue.put(
                (
                    "operation_error",
                    "Ultralytics Required",
                    "Object detection requires Ultralytics. Install it with:\n"
                    "pip install ultralytics pyyaml pillow\n\n"
                    f"Details: {exc}",
                )
            )
            return
        try:
            model = YOLO(config["model"])
            self.active_yolo_model = model

            def epoch_callback(trainer):
                epoch = int(getattr(trainer, "epoch", -1)) + 1
                metrics = dict(getattr(trainer, "metrics", {}) or {})
                summary = ", ".join(
                    f"{key}={float(value):.5g}"
                    for key, value in metrics.items()
                    if isinstance(value, (int, float, np.number))
                )
                self.worker_queue.put(
                    (
                        "training_log",
                        f"Epoch {epoch}/{config['epochs']}"
                        + (f" — {summary}" if summary else ""),
                    )
                )
                if self.stop_event.is_set():
                    trainer.stop = True

            model.add_callback("on_fit_epoch_end", epoch_callback)
            train_arguments = dict(config)
            model_value = train_arguments.pop("model")
            train_arguments.pop("device_selection", None)
            train_arguments.pop("resolved_device", None)
            train_arguments.pop("inference_device_selection", None)
            if train_arguments.get("device") is None:
                train_arguments.pop("device")
            if train_arguments.get("resume"):
                train_arguments = {"resume": True}
                if config.get("device") is not None:
                    train_arguments["device"] = config["device"]
            results = model.train(**train_arguments)
            trainer = getattr(model, "trainer", None)
            save_directory = getattr(trainer, "save_dir", None)
            if save_directory is None:
                save_directory = getattr(results, "save_dir", None)
            if save_directory is None:
                raise RuntimeError(
                    "Training finished, but the Ultralytics run directory "
                    "could not be located."
                )
            save_directory = str(Path(save_directory).resolve())
            best_path = _find_detection_weights(save_directory, "best.pt")
            last_path = _find_detection_weights(save_directory, "last.pt")
            selected_weights = best_path or last_path or model_value
            self.worker_queue.put(
                (
                    "training_complete",
                    save_directory,
                    selected_weights,
                    config,
                    self.stop_event.is_set(),
                )
            )
        except Exception as exc:
            self.worker_queue.put(
                (
                    "operation_error",
                    "YOLO Training Error",
                    f"{exc}\n\n{traceback.format_exc()}",
                )
            )
        finally:
            self.active_yolo_model = None

    def stop_training(self):
        if not self.operation_running:
            return
        self.stop_event.set()
        self.status_var.set(
            "Stop requested; finishing the current item/batch safely..."
        )
        if self.notebook.select() == str(self.inference_tab):
            self.append_log(
                self.inference_log_text,
                "Stop requested. The current prediction item will finish.",
            )
        else:
            self.append_log(
                self.training_log_text,
                "Stop requested. Ultralytics will preserve last.pt.",
            )

    def update_model_ready_state(self):
        training_allowed = (
            self.detector_trainable
            or self.model_var.get() != "Custom weights..."
        )
        self.train_button.config(
            state=(
                tk.NORMAL
                if training_allowed and not self.operation_running
                else tk.DISABLED
            )
        )
        ready = bool(
            self.current_weights_path
            and Path(self.current_weights_path).exists()
            and self.detector_inference_ready
        )
        state = tk.NORMAL if ready and not self.operation_running else tk.DISABLED
        self.evaluate_button.config(
            state=(
                state
                if self.data_yaml_path
                else tk.DISABLED
            )
        )
        self.export_button.config(
            state=state if self.detector_trainable else tk.DISABLED
        )
        self.predict_button.config(state=state)
        self.save_prediction_results_button.config(
            state=(
                tk.NORMAL
                if self.inference_directories and not self.operation_running
                else tk.DISABLED
            )
        )
        bundle_ready = bool(
            self.current_run_directory
            and Path(self.current_run_directory).is_dir()
            and self.data_yaml_path
            and self.validation_report
        )
        self.save_bundle_button.config(
            state=tk.NORMAL if bundle_ready else tk.DISABLED
        )
        result_ready = bool(
            self.latest_evaluation_metrics
            or self.latest_inference_summary
            or self.validation_report
            or (
                self.current_run_directory
                and Path(self.current_run_directory).is_dir()
            )
        )
        self.custom_results_button.config(
            state=(
                tk.NORMAL
                if result_ready and not self.operation_running
                else tk.DISABLED
            )
        )

    def start_evaluation(self):
        if self.operation_running:
            return
        if not self.current_weights_path or not self.data_yaml_path:
            messagebox.showwarning(
                "Model and Dataset Required",
                "Load detector weights and a compatible data.yaml first.",
            )
            return
        try:
            image_size = int(self.image_size_var.get())
            batch = int(self.batch_var.get())
            device_info = DeviceManager.resolve_yolo(
                self.inference_device_var.get()
            )
        except Exception:
            messagebox.showerror(
                "Evaluation Settings",
                "Check the image size, batch size, and inference device.",
            )
            return
        self.set_operation_state(True, "Evaluating detector...")
        self.stop_button.config(state=tk.DISABLED)
        self.result_log_text.delete("1.0", tk.END)
        self.append_log(
            self.result_log_text,
            f"Evaluating: {self.current_weights_path}",
        )
        threading.Thread(
            target=self.evaluation_worker,
            args=(
                self.current_weights_path,
                self.data_yaml_path,
                image_size,
                batch,
                device_info["backend_value"],
            ),
            daemon=True,
        ).start()

    def evaluation_worker(
        self,
        weights_path,
        data_yaml_path,
        image_size,
        batch,
        device,
    ):
        try:
            from ultralytics import YOLO

            model = YOLO(weights_path)
            arguments = {
                "data": data_yaml_path,
                "imgsz": image_size,
                "batch": batch,
                "plots": True,
                "project": str(_object_detection_run_root()),
                "name": (
                    "evaluation_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S")
                ),
            }
            if device:
                arguments["device"] = device
            metrics = model.val(**arguments)
            results_dict = {
                str(key): float(value)
                if isinstance(value, (int, float, np.number))
                else str(value)
                for key, value in dict(
                    getattr(metrics, "results_dict", {}) or {}
                ).items()
            }
            results_dict["execution_device"] = device or "auto"
            box_metrics = getattr(metrics, "box", None)
            per_class_maps = getattr(box_metrics, "maps", None)
            if per_class_maps is not None:
                model_names = getattr(model, "names", {}) or {}
                results_dict["per_class_mAP50_95"] = {
                    str(
                        model_names.get(index, index)
                        if isinstance(model_names, dict)
                        else model_names[index]
                    ): float(value)
                    for index, value in enumerate(per_class_maps)
                }
            save_directory = str(
                Path(getattr(metrics, "save_dir")).resolve()
            )
            metrics_path = Path(save_directory) / "nn_studio_metrics.json"
            metrics_path.write_text(
                json.dumps(results_dict, indent=2),
                encoding="utf-8",
            )
            self.worker_queue.put(
                (
                    "evaluation_complete",
                    save_directory,
                    results_dict,
                )
            )
        except Exception as exc:
            self.worker_queue.put(
                (
                    "operation_error",
                    "YOLO Evaluation Error",
                    f"{exc}\n\n{traceback.format_exc()}",
                )
            )

    def save_complete_bundle(self):
        if (
            not self.current_run_directory
            or not self.data_yaml_path
            or not self.validation_report
        ):
            return
        default_name = (
            self.project_name_var.get().strip() or "object_detector"
        ) + "_detection_package.zip"
        save_path = filedialog.asksaveasfilename(
            title="Save Complete Object Detection Package",
            defaultextension=".zip",
            initialfile=default_name,
            filetypes=[("ZIP package", "*.zip")],
        )
        if not save_path:
            return
        try:
            config = self.collect_training_config()
            config["selected_weights"] = self.current_weights_path
            save_detection_result_bundle(
                save_path=save_path,
                run_directory=self.current_run_directory,
                data_yaml_path=self.data_yaml_path,
                training_config=config,
                validation_report=self.validation_report,
                inference_directories=self.inference_directories,
                custom_result_directories=self.custom_result_directories,
            )
            messagebox.showinfo(
                "Detection Package Saved",
                "The complete detector, checkpoints, validation report, "
                f"metrics, plots, and predictions were saved:\n{save_path}",
            )
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    def start_model_export(self):
        if self.operation_running or not self.current_weights_path:
            return
        format_map = {
            "ONNX": "onnx",
            "TorchScript": "torchscript",
            "OpenVINO": "openvino",
            "TensorFlow SavedModel": "saved_model",
            "LiteRT / TensorFlow Lite": "litert",
        }
        export_format = format_map[self.export_format_var.get()]
        self.set_operation_state(
            True,
            f"Exporting detector to {self.export_format_var.get()}...",
        )
        self.stop_button.config(state=tk.DISABLED)
        threading.Thread(
            target=self.export_worker,
            args=(
                self.current_weights_path,
                export_format,
                int(self.image_size_var.get()),
            ),
            daemon=True,
        ).start()

    def export_worker(self, weights_path, export_format, image_size):
        try:
            from ultralytics import YOLO

            exported_path = YOLO(weights_path).export(
                format=export_format,
                imgsz=image_size,
            )
            self.worker_queue.put(
                ("export_complete", str(exported_path))
            )
        except Exception as exc:
            self.worker_queue.put(
                (
                    "operation_error",
                    "Detector Export Error",
                    f"{exc}\n\n{traceback.format_exc()}",
                )
            )

    def select_inference_file(self):
        path = filedialog.askopenfilename(
            title="Select Image or Video",
            filetypes=[
                (
                    "Images and video",
                    "*.jpg *.jpeg *.png *.bmp *.gif *.mp4 *.avi *.mov *.mkv",
                ),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.inference_source_var.set(str(Path(path).resolve()))

    def select_inference_folder(self):
        path = filedialog.askdirectory(
            title="Select Folder for Batch Object Detection"
        )
        if path:
            self.inference_source_var.set(str(Path(path).resolve()))

    def select_webcam(self):
        self.inference_source_var.set("0")
        self.inference_status_var.set(
            "Webcam 0 selected. Use Stop Safely to end the stream."
        )

    def start_inference(self):
        if self.operation_running or not self.current_weights_path:
            return
        source = self.inference_source_var.get().strip()
        webcam_source = source.isdigit()
        if not source or (not webcam_source and not Path(source).exists()):
            messagebox.showwarning(
                "Source Required",
                "Select an existing image, video, folder, or webcam index.",
            )
            return
        try:
            confidence = float(self.confidence_var.get())
            iou = float(self.iou_var.get())
            image_size = int(self.image_size_var.get())
            device_info = DeviceManager.resolve_yolo(
                self.inference_device_var.get()
            )
        except Exception:
            messagebox.showerror(
                "Detection Settings",
                "Confidence, IoU, and image size must be numeric.",
            )
            return
        if not (0 <= confidence <= 1 and 0 <= iou <= 1):
            messagebox.showerror(
                "Detection Settings",
                "Confidence and IoU thresholds must be between 0 and 1.",
            )
            return
        self.stop_event.clear()
        self.set_operation_state(True, "Running object detection...")
        self.inference_log_text.delete("1.0", tk.END)
        self.append_log(
            self.inference_log_text,
            f"Source: {source}\nConfidence: {confidence}\nIoU: {iou}",
        )
        threading.Thread(
            target=self.inference_worker,
            args=(
                self.current_weights_path,
                int(source) if webcam_source else source,
                confidence,
                iou,
                image_size,
                device_info["backend_value"],
                self.save_txt_var.get(),
                self.save_conf_var.get(),
            ),
            daemon=True,
        ).start()

    def inference_worker(
        self,
        weights_path,
        source,
        confidence,
        iou,
        image_size,
        device,
        save_txt,
        save_conf,
    ):
        try:
            from ultralytics import YOLO

            run_name = (
                "prediction_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            arguments = {
                "source": source,
                "conf": confidence,
                "iou": iou,
                "imgsz": image_size,
                "save": True,
                "save_txt": save_txt,
                "save_conf": save_conf,
                "project": str(_object_detection_run_root()),
                "name": run_name,
                "stream": True,
            }
            if device:
                arguments["device"] = device
            results = YOLO(weights_path).predict(**arguments)
            prediction_directory = None
            detection_count = 0
            image_count = 0
            detection_rows = []
            class_counts = {}
            preview_written = False
            for result in results:
                if prediction_directory is None:
                    prediction_directory = str(
                        Path(result.save_dir).resolve()
                    )
                image_count += 1
                boxes = getattr(result, "boxes", None)
                if boxes is not None:
                    detection_count += len(boxes)
                    try:
                        xyxy_values = boxes.xyxy.detach().cpu().numpy()
                        class_values = boxes.cls.detach().cpu().numpy()
                        confidence_values = (
                            boxes.conf.detach().cpu().numpy()
                        )
                        result_names = getattr(result, "names", {}) or {}
                        source_name = str(
                            getattr(result, "path", source)
                        )
                        for box_index, coordinates in enumerate(
                            xyxy_values
                        ):
                            class_id = int(class_values[box_index])
                            if isinstance(result_names, dict):
                                class_name = str(
                                    result_names.get(class_id, class_id)
                                )
                            else:
                                class_name = (
                                    str(result_names[class_id])
                                    if 0 <= class_id < len(result_names)
                                    else str(class_id)
                                )
                            confidence_value = float(
                                confidence_values[box_index]
                            )
                            class_counts[class_name] = (
                                int(class_counts.get(class_name, 0)) + 1
                            )
                            detection_rows.append(
                                {
                                    "source": source_name,
                                    "frame_or_item": image_count,
                                    "detection_index": box_index + 1,
                                    "class_id": class_id,
                                    "class_name": class_name,
                                    "confidence": confidence_value,
                                    "x1": float(coordinates[0]),
                                    "y1": float(coordinates[1]),
                                    "x2": float(coordinates[2]),
                                    "y2": float(coordinates[3]),
                                }
                            )
                    except Exception:
                        # Ultralytics still saves the annotated output even if
                        # an unusual backend does not expose tensor-like boxes.
                        pass
                if not preview_written and prediction_directory:
                    try:
                        from PIL import Image

                        plotted = np.asarray(result.plot())
                        if plotted.ndim == 3 and plotted.shape[2] >= 3:
                            rgb_preview = plotted[..., :3][..., ::-1]
                            Image.fromarray(
                                rgb_preview.astype(np.uint8)
                            ).save(
                                Path(prediction_directory)
                                / "nn_studio_preview.jpg",
                                quality=92,
                            )
                            preview_written = True
                    except Exception:
                        pass
                if self.stop_event.is_set():
                    break
            if prediction_directory is None:
                prediction_directory = str(
                    (_object_detection_run_root() / run_name).resolve()
                )
            prediction_path = Path(prediction_directory)
            prediction_path.mkdir(parents=True, exist_ok=True)
            detection_columns = [
                "source",
                "frame_or_item",
                "detection_index",
                "class_id",
                "class_name",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
            pd.DataFrame(
                detection_rows,
                columns=detection_columns,
            ).to_csv(
                prediction_path / "nn_studio_detections.csv",
                index=False,
            )
            summary = {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "weights": weights_path,
                "source": source,
                "confidence_threshold": confidence,
                "iou_threshold": iou,
                "execution_device": device or "auto",
                "processed_items": image_count,
                "detections": detection_count,
                "class_counts": class_counts,
                "detection_table": "nn_studio_detections.csv",
                "embedded_preview": (
                    "nn_studio_preview.jpg"
                    if (
                        prediction_path / "nn_studio_preview.jpg"
                    ).is_file()
                    else None
                ),
            }
            Path(
                prediction_directory,
                "nn_studio_prediction.json",
            ).write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            self.worker_queue.put(
                (
                    "inference_complete",
                    prediction_directory,
                    summary,
                )
            )
        except Exception as exc:
            self.worker_queue.put(
                (
                    "operation_error",
                    "Object Detection Error",
                    f"{exc}\n\n{traceback.format_exc()}",
                )
            )

    def open_path(self, path):
        if not path or not Path(path).exists():
            messagebox.showinfo(
                "Folder Unavailable",
                "No results folder is available yet.",
            )
            return
        try:
            if platform.system().lower() == "windows":
                os.startfile(path)
            elif platform.system().lower() == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as exc:
            messagebox.showerror("Open Folder Error", str(exc))

    def open_current_results_folder(self):
        self.open_path(self.current_run_directory)

    def open_latest_prediction_folder(self):
        self.open_path(
            self.inference_directories[-1]
            if self.inference_directories
            else None
        )

    def save_prediction_results(self):
        if not self.inference_directories:
            messagebox.showinfo(
                "No Prediction Results",
                "Run object detection before saving prediction results.",
                parent=self,
            )
            return
        source_directory = Path(self.inference_directories[-1])
        if not source_directory.is_dir():
            messagebox.showerror(
                "Results Unavailable",
                "The latest prediction results folder is unavailable.",
                parent=self,
            )
            return
        save_path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Object Detection Prediction Results",
            defaultextension=".zip",
            initialfile=source_directory.name + "_results.zip",
            filetypes=[("ZIP result bundle", "*.zip")],
        )
        if not save_path:
            return
        try:
            zip_result_directory(source_directory, save_path)
            messagebox.showinfo(
                "Prediction Results Saved",
                "Annotated outputs, the detection CSV, YOLO labels, preview, "
                f"and result summary were saved:\n{save_path}",
                parent=self,
            )
        except Exception as exc:
            messagebox.showerror(
                "Save Error",
                str(exc),
                parent=self,
            )

    def process_worker_queue(self):
        try:
            while True:
                message = self.worker_queue.get_nowait()
                event = message[0]
                if event == "dataset_validated":
                    _, config, report, records = message
                    self.dataset_config = config
                    self.validation_report = report
                    self.dataset_records = records
                    self.display_validation_report()
                    self.preview_button.config(
                        state=tk.NORMAL if records else tk.DISABLED
                    )
                    self.set_operation_state(
                        False,
                        (
                            "Dataset valid and ready for training."
                            if report.get("is_valid")
                            else "Dataset has invalid annotations."
                        ),
                    )
                    self.update_model_ready_state()
                elif event == "training_log":
                    self.append_log(self.training_log_text, message[1])
                elif event == "training_complete":
                    (
                        _,
                        run_directory,
                        weights_path,
                        config,
                        stopped,
                    ) = message
                    self.current_run_directory = run_directory
                    self.current_weights_path = weights_path
                    self.detector_trainable = True
                    self.detector_inference_ready = True
                    self.set_operation_state(
                        False,
                        "Training stopped; checkpoint preserved."
                        if stopped
                        else "Training completed successfully.",
                    )
                    self.append_log(
                        self.training_log_text,
                        (
                            "Training stopped safely."
                            if stopped
                            else "Training completed."
                        )
                        + f"\nResults: {run_directory}\n"
                        f"Selected weights: {weights_path}",
                    )
                    self.append_log(
                        self.result_log_text,
                        f"Training results available: {run_directory}",
                    )
                    self.update_model_ready_state()
                    self.notebook.select(self.results_tab)
                elif event == "evaluation_complete":
                    _, directory, metrics = message
                    self.inference_directories.append(directory)
                    self.latest_evaluation_directory = directory
                    self.latest_evaluation_metrics = dict(metrics)
                    self.set_operation_state(
                        False,
                        "Detector evaluation completed.",
                    )
                    self.append_log(
                        self.result_log_text,
                        "Evaluation metrics:\n"
                        + json.dumps(metrics, indent=2)
                        + f"\nEvaluation artifacts: {directory}",
                    )
                    self.update_model_ready_state()
                elif event == "export_complete":
                    self.set_operation_state(
                        False,
                        "Detector export completed.",
                    )
                    self.append_log(
                        self.result_log_text,
                        f"Exported detector: {message[1]}",
                    )
                    self.update_model_ready_state()
                    messagebox.showinfo(
                        "Detector Exported",
                        f"Export completed:\n{message[1]}",
                    )
                elif event == "inference_complete":
                    _, directory, summary = message
                    self.inference_directories.append(directory)
                    self.latest_inference_summary = dict(summary)
                    self.set_operation_state(
                        False,
                        "Object detection completed.",
                    )
                    self.inference_status_var.set(
                        f"{summary['detections']} detections in "
                        f"{summary['processed_items']} item(s)."
                    )
                    self.append_log(
                        self.inference_log_text,
                        json.dumps(summary, indent=2)
                        + f"\nSaved output: {directory}",
                    )
                    self.load_detection_preview(directory)
                    self.update_model_ready_state()
                elif event == "operation_error":
                    _, title, details = message
                    self.set_operation_state(False, "Operation failed.")
                    self.update_model_ready_state()
                    messagebox.showerror(title, details)
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(100, self.process_worker_queue)

    def get_custom_results_context(self):
        """Build tabular plotting sources from YOLO validation and inference."""
        tables = {}
        class_names = list(
            (self.validation_report or {}).get("class_names", [])
        )

        if self.validation_report:
            class_counts = []
            for class_id, class_name in enumerate(class_names):
                class_counts.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "instance_count": int(
                        self.validation_report.get(
                            "class_instance_counts",
                            {},
                        ).get(str(class_id), 0)
                    ),
                })
            if class_counts:
                tables["class_distribution"] = pd.DataFrame(class_counts)

            split_rows = []
            for split_name, values in self.validation_report.get(
                "splits",
                {},
            ).items():
                row = {"split": split_name}
                row.update({
                    str(key): value
                    for key, value in values.items()
                    if isinstance(value, (str, int, float, bool))
                })
                split_rows.append(row)
            if split_rows:
                tables["dataset_splits"] = pd.DataFrame(split_rows)

        if self.dataset_records:
            record_rows = []
            for record in self.dataset_records:
                record_rows.append({
                    "split": record.get("split"),
                    "image_path": record.get("image_path"),
                    "label_state": record.get("label_state"),
                    "instance_count": record.get("instance_count", 0),
                    "error_count": len(record.get("errors") or []),
                })
            tables["dataset_images"] = pd.DataFrame(record_rows)

        candidate_directories = []
        for directory in (
            self.current_run_directory,
            self.latest_evaluation_directory,
        ):
            if directory and Path(directory).is_dir():
                candidate_directories.append(Path(directory))
        for directory in self.inference_directories:
            if directory and Path(directory).is_dir():
                candidate_directories.append(Path(directory))

        seen_csv = set()
        for directory in candidate_directories:
            for csv_path in directory.rglob("results.csv"):
                resolved = str(csv_path.resolve())
                if resolved in seen_csv:
                    continue
                seen_csv.add(resolved)
                try:
                    frame = pd.read_csv(csv_path)
                except Exception:
                    continue
                table_name = (
                    "training_history"
                    if "training_history" not in tables
                    else f"training_history_{len(seen_csv)}"
                )
                tables[table_name] = frame

        metric_rows = []
        for key, value in self.latest_evaluation_metrics.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    metric_rows.append({
                        "metric": f"{key}.{nested_key}",
                        "value": nested_value,
                    })
            else:
                metric_rows.append({"metric": key, "value": value})
        if metric_rows:
            tables["evaluation_metrics"] = pd.DataFrame(metric_rows)

        summary_rows = []
        for directory in self.inference_directories:
            summary_path = Path(directory) / "nn_studio_prediction.json"
            if not summary_path.is_file():
                continue
            try:
                summary = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
                summary_rows.append(summary)
            except Exception:
                continue
        if self.latest_inference_summary and not summary_rows:
            summary_rows.append(self.latest_inference_summary)
        if summary_rows:
            tables["inference_summary"] = pd.DataFrame(summary_rows)

        detection_rows = []
        for directory in self.inference_directories:
            detection_csv = (
                Path(directory) / "nn_studio_detections.csv"
            )
            if detection_csv.is_file():
                try:
                    detection_rows.extend(
                        pd.read_csv(detection_csv).to_dict("records")
                    )
                    continue
                except Exception:
                    pass
            labels_directory = Path(directory) / "labels"
            if not labels_directory.is_dir():
                continue
            for label_path in sorted(labels_directory.glob("*.txt")):
                for line_number, line in enumerate(
                    label_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    ).splitlines(),
                    start=1,
                ):
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        class_id = int(float(parts[0]))
                        row = {
                            "source": label_path.stem,
                            "line_number": line_number,
                            "class_id": class_id,
                            "class_name": (
                                class_names[class_id]
                                if 0 <= class_id < len(class_names)
                                else str(class_id)
                            ),
                            "x_center": float(parts[1]),
                            "y_center": float(parts[2]),
                            "width": float(parts[3]),
                            "height": float(parts[4]),
                            "confidence": (
                                float(parts[5])
                                if len(parts) >= 6
                                else np.nan
                            ),
                        }
                    except Exception:
                        continue
                    detection_rows.append(row)
        if detection_rows:
            tables["detection_results"] = pd.DataFrame(detection_rows)

        if not tables:
            raise ValueError(
                "No detection results are available yet. Validate a dataset, "
                "evaluate the detector, or run object detection first."
            )
        preferred = (
            "detection_results"
            if "detection_results" in tables
            else "evaluation_metrics"
            if "evaluation_metrics" in tables
            else next(iter(tables))
        )
        tables["results"] = tables[preferred].copy()
        return {
            "title": "Object Detection Application Results",
            "task_type": DATA_MODE_DETECTION,
            "model_type": (
                Path(self.current_weights_path).name
                if self.current_weights_path
                else "YOLO detector"
            ),
            "tables": tables,
            "metrics": self.latest_evaluation_metrics,
            "metadata": {
                "weights": self.current_weights_path,
                "training_config": self.collect_training_config(),
                "latest_inference": self.latest_inference_summary,
            },
            "class_names": class_names,
            "source": (
                self.latest_evaluation_directory
                or (
                    self.inference_directories[-1]
                    if self.inference_directories
                    else self.data_yaml_path
                )
                or "Object detection workspace"
            ),
        }

    def attach_custom_result(self, source_directory, _recipe=None):
        destination = tempfile.mkdtemp(
            prefix="nn_attached_detection_result_"
        )
        shutil.rmtree(destination, ignore_errors=True)
        shutil.copytree(source_directory, destination)
        self.custom_result_directories.append(destination)
        self.append_log(
            self.result_log_text,
            "Custom visualization attached for the next detection package.",
        )

    def open_custom_results_studio(self):
        try:
            self.get_custom_results_context()
        except Exception as exc:
            messagebox.showwarning(
                "No Detection Results",
                str(exc),
                parent=self,
            )
            return
        CustomResultsStudioWindow(
            self,
            context_provider=self.get_custom_results_context,
            settings_owner=self.master,
            attach_callback=self.attach_custom_result,
        )

    def close_window(self):
        if self.operation_running:
            close_anyway = messagebox.askyesno(
                "Operation Running",
                "An object-detection operation is still running. Request a "
                "safe stop and close this window?",
            )
            if not close_anyway:
                return
            self.stop_event.set()
        if self.loaded_package is not None:
            shutil.rmtree(
                self.loaded_package.get("temp_directory", ""),
                ignore_errors=True,
            )
        for directory in self.custom_result_directories:
            shutil.rmtree(directory, ignore_errors=True)
        self.destroy()
