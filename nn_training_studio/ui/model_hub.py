"""Ui / model hub for NN Training Studio."""

from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
from nn_training_studio.constants import (
    APP_VERSION,
)
from nn_training_studio.deployment import (
    inspect_saved_model_descriptor,
)
from nn_training_studio.ui.deploy import (
    DeployIntegrateWindow,
)
from nn_training_studio.ui.detection import (
    ObjectDetectionWindow,
)
from nn_training_studio.ui.evaluation import (
    ModelEvaluationWindow,
)


class ModelLoadingHubWindow(tk.Toplevel):
    """Guided, read-only model inspection before opening an application."""

    def __init__(self, parent, initial_path=None):
        super().__init__(parent)
        self.title(f"NN Training Studio {APP_VERSION} — Load Existing Model")
        self.geometry("1040x720")
        self.minsize(900, 620)
        self.selected_path = None
        self.descriptor = None
        self.path_var = tk.StringVar(value="")
        self.model_type_var = tk.StringVar(value="No model selected")
        self.status_var = tk.StringVar(
            value="Step 1: select a saved model package, detector, or export."
        )
        self.build_interface()
        if initial_path:
            self.after_idle(lambda: self.inspect_path(initial_path))

    def build_interface(self):
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=14, pady=(12, 6))
        ttk.Label(
            header,
            text="Model Loading & Application",
            font=("Arial", 19, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "Select → inspect compatibility → open the correct application "
                "→ load input → predict or evaluate → preview and export"
            ),
            foreground="#245a85",
        ).pack(anchor="w", pady=(3, 0))

        steps = ttk.Frame(self)
        steps.pack(fill=tk.X, padx=14, pady=(2, 8))
        for index, text_value in enumerate(
            [
                "1  Select Model",
                "2  Review Model",
                "3  Open Application",
                "4  Preview & Export",
            ]
        ):
            ttk.Label(
                steps,
                text=text_value,
                anchor=tk.CENTER,
                relief=tk.GROOVE,
                padding=(10, 7),
            ).grid(
                row=0,
                column=index,
                sticky="ew",
                padx=3,
            )
            steps.columnconfigure(index, weight=1, uniform="load_step")

        selection = ttk.LabelFrame(self, text="1. Select Saved Model")
        selection.pack(fill=tk.X, padx=14, pady=6)
        selection.columnconfigure(0, weight=1)
        ttk.Entry(
            selection,
            textvariable=self.path_var,
            state="readonly",
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ttk.Button(
            selection,
            text="Browse Model File...",
            command=self.select_model_file,
        ).grid(row=0, column=1, padx=4, pady=8)
        ttk.Button(
            selection,
            text="Browse Exported Folder...",
            command=self.select_model_folder,
        ).grid(row=0, column=2, padx=(4, 8), pady=8)

        main_pane = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=14, pady=6)
        overview_frame = ttk.LabelFrame(
            main_pane,
            text="2. Model Overview & Compatibility",
        )
        capability_frame = ttk.LabelFrame(
            main_pane,
            text="Available Application Functions",
        )
        main_pane.add(overview_frame, weight=3)
        main_pane.add(capability_frame, weight=2)

        ttk.Label(
            overview_frame,
            textvariable=self.model_type_var,
            font=("Arial", 13, "bold"),
            foreground="#245a85",
            wraplength=570,
        ).pack(anchor="w", padx=9, pady=(9, 5))
        self.overview_text = ScrolledText(
            overview_frame,
            wrap=tk.WORD,
            height=18,
            state=tk.DISABLED,
        )
        self.overview_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=8,
            pady=(0, 8),
        )

        self.capability_list = tk.Listbox(
            capability_frame,
            activestyle="none",
            height=12,
        )
        self.capability_list.pack(
            fill=tk.BOTH,
            expand=True,
            padx=8,
            pady=8,
        )
        ttk.Label(
            capability_frame,
            text=(
                "Object detectors open with an embedded annotated-image "
                "preview, detection table, thresholds, device selection, "
                "custom result plots, and export controls."
            ),
            wraplength=340,
            foreground="#245a85",
        ).pack(anchor="w", padx=9, pady=(0, 9))

        action = ttk.LabelFrame(
            self,
            text="3. Open Model Application or Deploy",
        )
        action.pack(fill=tk.X, padx=14, pady=(5, 12))
        ttk.Label(
            action,
            textvariable=self.status_var,
            wraplength=650,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=9, pady=9)
        self.open_button = ttk.Button(
            action,
            text="Open Model Application →",
            command=self.open_selected_model,
            state=tk.DISABLED,
        )
        self.open_button.pack(side=tk.RIGHT, padx=9, pady=9)
        self.deploy_button = ttk.Button(
            action,
            text="Deploy & Integrate →",
            command=self.open_deployment_workspace,
            state=tk.DISABLED,
        )
        self.deploy_button.pack(side=tk.RIGHT, padx=(4, 0), pady=9)

        self.show_overview(
            "No model has been inspected yet.\n\n"
            "Supported inputs:\n"
            "• NN Studio signal/image model package (.zip)\n"
            "• NN Studio object-detection package (.zip)\n"
            "• YOLO trainable weights or architecture (.pt/.yaml)\n"
            "• ONNX, TensorRT, TorchScript, TFLite, SavedModel, or OpenVINO "
            "detector exports"
        )

    def show_overview(self, text_value):
        self.overview_text.config(state=tk.NORMAL)
        self.overview_text.delete("1.0", tk.END)
        self.overview_text.insert(tk.END, text_value)
        self.overview_text.config(state=tk.DISABLED)

    def select_model_file(self):
        path = filedialog.askopenfilename(
            parent=self,
            title="Select Existing Model",
            filetypes=[
                (
                    "Supported models",
                    "*.zip *.pt *.yaml *.yml *.onnx *.engine "
                    "*.torchscript *.tflite *.pb *.xml",
                ),
                ("NN Studio package", "*.zip"),
                ("YOLO trainable model", "*.pt *.yaml *.yml"),
                (
                    "Exported detector",
                    "*.onnx *.engine *.torchscript *.tflite *.pb *.xml",
                ),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.inspect_path(path)

    def select_model_folder(self):
        path = filedialog.askdirectory(
            parent=self,
            title="Select Exported Detector Folder",
        )
        if path:
            self.inspect_path(path)

    def inspect_path(self, path):
        try:
            descriptor = inspect_saved_model_descriptor(path)
        except Exception as exc:
            self.selected_path = None
            self.descriptor = None
            self.path_var.set(str(path))
            self.model_type_var.set("Unsupported or invalid model")
            self.show_overview(str(exc))
            self.capability_list.delete(0, tk.END)
            self.open_button.config(state=tk.DISABLED)
            self.deploy_button.config(state=tk.DISABLED)
            self.status_var.set(
                "Choose another supported model file or exported folder."
            )
            return

        self.selected_path = descriptor["path"]
        self.descriptor = descriptor
        self.path_var.set(descriptor["path"])
        self.model_type_var.set(descriptor["display_type"])

        detail_lines = [
            f"File / folder: {descriptor['name']}",
            f"Model type: {descriptor['display_type']}",
            f"Data mode: {descriptor['data_mode'] or 'Not specified'}",
            f"Task: {descriptor['task_type'] or 'Not specified'}",
            (
                "Training capability: Resume / fine-tune available"
                if descriptor["trainable"]
                else "Training capability: Inference / evaluation only"
            ),
            (
                "Prediction readiness: Ready"
                if descriptor["inference_ready"]
                else (
                    "Prediction readiness: Architecture only — train or load "
                    "weights first"
                )
            ),
        ]
        useful_details = {
            key: value
            for key, value in descriptor.get("details", {}).items()
            if value not in (None, "", [], {})
        }
        if useful_details:
            detail_lines.extend(["", "Saved configuration:"])
            for key, value in useful_details.items():
                if isinstance(value, (list, tuple)) and len(value) > 12:
                    value = [*value[:12], f"... ({len(value)} total)"]
                detail_lines.append(
                    f"• {key.replace('_', ' ').title()}: {value}"
                )
        self.show_overview("\n".join(detail_lines))

        self.capability_list.delete(0, tk.END)
        for capability in descriptor.get("capabilities", []):
            self.capability_list.insert(tk.END, "✓ " + capability)
        self.open_button.config(state=tk.NORMAL)
        self.deploy_button.config(state=tk.NORMAL)
        self.status_var.set(
            "Model inspection passed. Review the information, then open the "
            "application workspace."
        )

    def open_selected_model(self):
        if not self.selected_path or not self.descriptor:
            return
        model_type = self.descriptor["model_type"]
        parent = self.master
        selected_path = self.selected_path
        if model_type == "keras_package":
            self.destroy()
            ModelEvaluationWindow(
                parent,
                initial_package_path=selected_path,
            )
            return
        if not messagebox.askyesno(
            "Trusted Detector",
            "Load this detector only if you created it or trust its source. "
            "Trainable model files may contain serialized objects. Continue?",
            parent=self,
        ):
            return
        self.destroy()
        ObjectDetectionWindow(
            parent,
            initial_model_path=selected_path,
            application_mode=True,
        )

    def open_deployment_workspace(self):
        if not self.selected_path:
            return
        DeployIntegrateWindow(
            self.master,
            initial_model_path=self.selected_path,
        )
