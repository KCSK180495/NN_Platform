"""Ui / wizard layout for NN Training Studio."""

from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
from nn_training_studio.constants import (
    APP_VERSION,
    DATA_MODE_IMAGE,
    GUIDE_ACTION_DEPLOY,
    GUIDE_ACTION_MODEL_APPLICATION,
    GUIDE_ACTION_PROJECTS,
    GUIDE_ACTION_SETTINGS,
    WORKSPACE_ANNOTATE,
    WORKSPACE_DEPLOY,
    WORKSPACE_DETECTION,
    WORKSPACE_FILTER,
    WORKSPACE_IMAGE,
    WORKSPACE_SIGNAL,
)
from nn_training_studio.ui.annotations import (
    AnnotationWorkspaceWindow,
)
from nn_training_studio.ui.deploy import (
    DeployIntegrateWindow,
)
from nn_training_studio.ui.detection import (
    ObjectDetectionWindow,
)
from nn_training_studio.ui.guide import (
    StudioGuideWindow,
)
from nn_training_studio.ui.model_hub import (
    ModelLoadingHubWindow,
)


class LayoutMixin:
    """Layout behavior for the main application."""

    def create_layout(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)
        # Reserve a real row for the wizard controls. Without an explicit
        # minimum, a content page with a large requested size can make the
        # footer appear outside a short display.
        self.rowconfigure(2, weight=0, minsize=0)

        # ========================================================
        # Top header frame
        # ========================================================

        self.header_frame = ttk.Frame(self)
        self.header_frame.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=15,
            pady=(10, 5),
        )
        self.header_frame.columnconfigure(0, weight=1)

        # First row: title and page number. Keeping this separate from the
        # application buttons prevents either group from pushing the other
        # outside the window.
        title_frame = ttk.Frame(self.header_frame)
        title_frame.grid(row=0, column=0, sticky="ew")
        title_frame.columnconfigure(0, weight=1)

        self.title_label = ttk.Label(
            title_frame,
            text="",
            font=("Arial", 18, "bold")
        )
        self.title_label.grid(row=0, column=0, sticky="w")

        self.progress_label = ttk.Label(
            title_frame,
            text="",
            font=("Arial", 10)
        )
        self.progress_label.grid(row=1, column=0, sticky="w", pady=3)

        # This makes it obvious which corrected source/executable is active.
        # It is especially useful when several NN Studio versions are stored
        # beside each other.
        self.version_badge_label = ttk.Label(
            title_frame,
            text=APP_VERSION,
            font=("Arial", 10, "bold"),
            foreground="#245a85",
        )
        self.version_badge_label.grid(
            row=0,
            column=1,
            rowspan=2,
            sticky="ne",
            padx=(12, 2),
        )

        mode_frame = ttk.Frame(self.header_frame)
        mode_frame.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        ttk.Label(mode_frame, text="Interface mode:").pack(side=tk.LEFT, padx=(0, 6))
        self.ui_mode_combo = ttk.Combobox(mode_frame, textvariable=self.ui_mode_var,
                                         values=("Guided", "Advanced"), state="readonly", width=12)
        self.ui_mode_combo.pack(side=tk.LEFT)
        self.ui_mode_combo.bind("<<ComboboxSelected>>", self.change_ui_mode)
        ttk.Button(mode_frame, text="Open draft", command=self.open_project_draft).pack(side=tk.LEFT, padx=8)

        # Second row: global application navigation. Wizard Back / Next
        # controls deliberately live in the fixed footer below the page.
        self.nav_top_frame = ttk.Frame(self.header_frame)
        self.nav_top_frame.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        self.nav_top_frame.columnconfigure(0, weight=1)

        self.nav_button_frame = ttk.Frame(self.nav_top_frame)
        self.nav_button_frame.grid(row=0, column=0, sticky="w")

        self.home_button = ttk.Button(
            self.nav_button_frame,
            text="Home",
            command=self.show_home,
        )
        self.home_button.grid(row=0, column=0, padx=(0, 5))

        self.evaluate_existing_button = ttk.Button(
            self.nav_button_frame,
            text="Use Model",
            command=self.open_model_evaluation
        )
        self.evaluate_existing_button.grid(row=0, column=1, padx=5)

        self.deploy_button = ttk.Button(
            self.nav_button_frame,
            text="Deploy",
            command=self.open_deploy_integrate,
        )
        self.deploy_button.grid(row=0, column=2, padx=5)

        self.annotate_button = ttk.Button(
            self.nav_button_frame,
            text="Annotate",
            command=self.open_annotation_workspace,
        )
        self.annotate_button.grid(row=0, column=3, padx=5)

        self.projects_button = ttk.Button(
            self.nav_button_frame,
            text="Projects",
            command=self.show_projects,
        )
        self.projects_button.grid(row=0, column=4, padx=5)

        self.guide_button = ttk.Button(
            self.nav_button_frame,
            text="Guide",
            command=self.open_studio_guide,
        )
        self.guide_button.grid(row=0, column=5, padx=5)

        self.settings_button = ttk.Button(
            self.nav_button_frame,
            text="⚙ Settings",
            command=self.show_settings,
        )
        self.settings_button.grid(row=0, column=6, padx=5)

        # Connection health gets a separate row. A long provider/model label
        # can no longer push navigation—or wizard controls—past the right edge.
        self.provider_status_button = ttk.Button(
            self.nav_top_frame,
            text="○ Offline Mode",
            command=self.show_settings,
        )
        self.provider_status_button.grid(
            row=1,
            column=0,
            sticky="w",
            pady=(5, 0),
        )

        # ========================================================
        # Persistent workflow footer
        # ========================================================

        self.workflow_footer = ttk.Frame(self)
        self.workflow_footer.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=15,
            pady=(4, 10),
        )
        ttk.Separator(
            self.workflow_footer,
            orient=tk.HORIZONTAL,
        ).pack(fill=tk.X, pady=(0, 7))

        self.workflow_row = ttk.Frame(self.workflow_footer)
        self.workflow_row.pack(fill=tk.X)
        self.workflow_row.columnconfigure(0, weight=1, minsize=120)
        self.workflow_row.columnconfigure(1, weight=0)
        self.workflow_row.columnconfigure(2, weight=0)

        self.workflow_hint_label = ttk.Label(
            self.workflow_row,
            text="",
            foreground="#245a85",
            wraplength=620,
        )
        self.workflow_hint_label.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(2, 12),
        )

        self.back_button = ttk.Button(
            self.workflow_row,
            text="← Back",
            width=14,
            command=self.go_back,
        )
        self.back_button.grid(
            row=0,
            column=1,
            sticky="e",
            padx=(4, 5),
        )

        self.next_button = ttk.Button(
            self.workflow_row,
            text="Next →",
            width=18,
            command=self.go_next,
        )
        self.next_button.grid(
            row=0,
            column=2,
            sticky="e",
            padx=(5, 0),
        )

        # ========================================================
        # Main content frame
        # ========================================================

        self.content_shell = ttk.Frame(self)
        self.content_shell.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=15,
            pady=5,
        )
        self.content_shell.rowconfigure(0, weight=1)
        self.content_shell.columnconfigure(0, weight=1)

        self.content_canvas = tk.Canvas(
            self.content_shell,
            highlightthickness=0,
            borderwidth=0,
        )
        self.content_scrollbar = ttk.Scrollbar(
            self.content_shell,
            orient=tk.VERTICAL,
            command=self.content_canvas.yview,
        )
        self.content_canvas.configure(
            yscrollcommand=self.content_scrollbar.set,
        )
        self.content_canvas.grid(row=0, column=0, sticky="nsew")
        self.content_scrollbar.grid(row=0, column=1, sticky="ns")

        self.content_frame = ttk.Frame(self.content_canvas)
        self.content_canvas_window = self.content_canvas.create_window(
            (0, 0),
            window=self.content_frame,
            anchor="nw",
        )
        self.content_frame.bind(
            "<Configure>",
            self._on_content_frame_configure,
            add="+",
        )
        self.content_canvas.bind(
            "<Configure>",
            self._on_content_canvas_configure,
            add="+",
        )
        self.workflow_footer.grid_remove()
        self.bind("<Alt-Left>", self._navigate_back_shortcut)
        self.bind("<Alt-Right>", self._navigate_next_shortcut)
        self.bind("<MouseWheel>", self._on_content_mousewheel, add="+")
        self.bind("<Button-4>", self._on_content_mousewheel, add="+")
        self.bind("<Button-5>", self._on_content_mousewheel, add="+")
        self.bind("<Configure>", self._schedule_responsive_layout, add="+")
        self.update_provider_status_display()
        self.after_idle(self._apply_responsive_layout)

    def _on_content_frame_configure(self, _event=None):
        """Keep the page scroll range synchronized with its requested size."""
        try:
            canvas_width = max(1, int(self.content_canvas.winfo_width()))
            self.content_canvas.itemconfigure(
                self.content_canvas_window,
                width=canvas_width,
            )
            self.content_canvas.configure(
                scrollregion=self.content_canvas.bbox("all")
            )
        except tk.TclError:
            pass

    def _on_content_canvas_configure(self, event):
        """Match the embedded page width without constraining its height."""
        try:
            visible_width = max(1, int(event.width))
            self.content_canvas.itemconfigure(
                self.content_canvas_window,
                width=visible_width,
            )
            self.content_frame.configure(width=visible_width)
        except tk.TclError:
            return
        self._schedule_responsive_layout()

    @staticmethod
    def _widget_is_inside(widget, ancestor):
        current = widget
        while current is not None:
            if current == ancestor:
                return True
            current = getattr(current, "master", None)
        return False

    def _on_content_mousewheel(self, event):
        """Scroll the active page without stealing wheel input from editors."""
        try:
            widget = event.widget
            if not self._widget_is_inside(widget, self.content_shell):
                return None
            if widget != self.content_canvas and widget.winfo_class() in {
                "Text", "Listbox", "Treeview", "Canvas"
            }:
                return None
            if getattr(event, "num", None) == 4:
                units = -3
            elif getattr(event, "num", None) == 5:
                units = 3
            else:
                delta = int(getattr(event, "delta", 0) or 0)
                units = -int(delta / 120) if delta else 0
            if units:
                self.content_canvas.yview_scroll(units, "units")
                return "break"
        except tk.TclError:
            pass
        return None

    def _schedule_responsive_layout(self, event=None):
        """Debounce the many Configure events generated during live resizing."""
        if event is not None and getattr(event, "widget", self) not in {
            self,
            getattr(self, "content_canvas", None),
        }:
            return
        if self.winfo_width() < 100 or self.winfo_height() < 100:
            # Windows temporarily reports tiny dimensions while minimizing.
            return
        if self._responsive_after_id is not None:
            try:
                self.after_cancel(self._responsive_after_id)
            except (tk.TclError, ValueError):
                pass
        self._responsive_after_id = self.after(70, self._apply_responsive_layout)

    def _responsive_window_width(self):
        try:
            return max(1, int(self.winfo_width()))
        except tk.TclError:
            return 1280

    def _apply_responsive_layout(self):
        """Apply compact/medium/wide rules without rebuilding page state."""
        self._responsive_after_id = None
        width = self._responsive_window_width()
        if width < 100:
            return
        breakpoint = (
            "compact" if width < 920 else "medium" if width < 1400 else "wide"
        )
        self._responsive_breakpoint = breakpoint
        self._responsive_width = width
        compact = breakpoint == "compact"

        side_padding = 8 if compact else 12 if breakpoint == "medium" else 15
        try:
            self.header_frame.grid_configure(
                padx=side_padding,
                pady=(7 if compact else 10, 4),
            )
            self.content_shell.grid_configure(
                padx=side_padding,
                pady=3 if compact else 5,
            )
            self.title_label.configure(
                font=("Arial", 15 if compact else 18, "bold"),
                wraplength=max(430, width - (95 if compact else 135)),
            )
            self.progress_label.configure(font=("Arial", 9 if compact else 10))
        except tk.TclError:
            return

        self._reflow_global_navigation(compact)
        if self.current_view == "wizard":
            self.workflow_footer.grid_configure(
                padx=side_padding,
                pady=(3, 7 if compact else 10),
            )
            self._reflow_workflow_footer(compact)
        else:
            self.workflow_footer.grid_remove()
            self.rowconfigure(2, minsize=0)
        if self.current_view == "home":
            self._reflow_home_page(width)
        elif self.current_view == "wizard" and self.current_step == 0:
            self._reflow_step_one(width)
        self._update_responsive_wraps(self.content_frame)
        self._on_content_frame_configure()
        self.after_idle(self._finish_responsive_layout)

    def _finish_responsive_layout(self):
        """Refresh wrapping after Tk has applied the new grid geometry."""
        try:
            if not self.content_frame.winfo_exists():
                return
        except tk.TclError:
            return
        self._update_responsive_wraps(self.content_frame)
        self._on_content_frame_configure()

    def _reflow_global_navigation(self, compact):
        buttons = [
            self.home_button,
            self.evaluate_existing_button,
            self.deploy_button,
            self.annotate_button,
            self.projects_button,
            self.guide_button,
            self.settings_button,
        ]
        try:
            for button in buttons:
                button.grid_forget()
            for column in range(7):
                self.nav_button_frame.columnconfigure(
                    column,
                    weight=0,
                    uniform="",
                )
            if compact:
                self.nav_button_frame.grid_configure(sticky="ew")
                for column in range(4):
                    self.nav_button_frame.columnconfigure(
                        column,
                        weight=1,
                        uniform="compact_nav",
                    )
                for index, button in enumerate(buttons):
                    button.grid(
                        row=index // 4,
                        column=index % 4,
                        sticky="ew",
                        padx=2,
                        pady=2,
                    )
                self.provider_status_button.grid_configure(
                    row=2,
                    pady=(4, 0),
                )
            else:
                self.nav_button_frame.grid_configure(sticky="w")
                for index, button in enumerate(buttons):
                    button.grid(
                        row=0,
                        column=index,
                        padx=(0, 5) if index == 0 else 5,
                        pady=0,
                    )
                self.provider_status_button.grid_configure(
                    row=1,
                    pady=(5, 0),
                )
        except tk.TclError:
            pass

    def _reflow_workflow_footer(self, compact):
        if self.current_view != "wizard":
            try:
                self.workflow_footer.grid_remove()
                self.rowconfigure(2, minsize=0)
            except tk.TclError:
                pass
            return
        try:
            self.workflow_hint_label.grid_forget()
            self.back_button.grid_forget()
            self.next_button.grid_forget()
            for column in range(3):
                self.workflow_row.columnconfigure(
                    column,
                    weight=0,
                    minsize=0,
                    uniform="",
                )
            if compact:
                self.rowconfigure(2, minsize=88)
                self.workflow_row.columnconfigure(0, weight=1, uniform="footer")
                self.workflow_row.columnconfigure(1, weight=1, uniform="footer")
                self.workflow_hint_label.grid(
                    row=0,
                    column=0,
                    columnspan=2,
                    sticky="ew",
                    padx=2,
                    pady=(0, 5),
                )
                self.back_button.grid(
                    row=1,
                    column=0,
                    sticky="e",
                    padx=(2, 4),
                )
                self.next_button.grid(
                    row=1,
                    column=1,
                    sticky="w",
                    padx=(4, 2),
                )
                self.workflow_hint_label.configure(
                    wraplength=max(360, self._responsive_window_width() - 50)
                )
            else:
                self.rowconfigure(2, minsize=58)
                self.workflow_row.columnconfigure(0, weight=1, minsize=120)
                self.workflow_hint_label.grid(
                    row=0,
                    column=0,
                    sticky="ew",
                    padx=(2, 12),
                )
                self.back_button.grid(
                    row=0,
                    column=1,
                    sticky="e",
                    padx=(4, 5),
                )
                self.next_button.grid(
                    row=0,
                    column=2,
                    sticky="e",
                    padx=(5, 0),
                )
                available_width = max(self._responsive_window_width() - 390, 220)
                self.workflow_hint_label.configure(
                    wraplength=min(760, available_width)
                )
        except tk.TclError:
            pass

    def _update_responsive_wraps(self, parent):
        """Shrink existing wrap lengths to the actual parent width."""
        try:
            children = parent.winfo_children()
        except tk.TclError:
            return
        for widget in children:
            if isinstance(widget, ttk.Label):
                try:
                    current = int(float(widget.cget("wraplength") or 0))
                    if current > 0:
                        if not hasattr(widget, "_nn_base_wraplength"):
                            widget._nn_base_wraplength = current
                        parent_width = int(widget.master.winfo_width())
                        if parent_width > 100:
                            target = max(
                                160,
                                min(
                                    int(widget._nn_base_wraplength),
                                    parent_width - 28,
                                ),
                            )
                            widget.configure(wraplength=target)
                except (tk.TclError, TypeError, ValueError):
                    pass
            self._update_responsive_wraps(widget)

    def _reflow_step_one(self, width):
        frame = getattr(self, "step1_top_frame", None)
        button = getattr(self, "step1_load_button", None)
        label = getattr(self, "file_label", None)
        if frame is None or button is None or label is None:
            return
        try:
            button.grid_forget()
            label.grid_forget()
            frame.columnconfigure(0, weight=0)
            frame.columnconfigure(1, weight=1)
            if width < 920:
                button.grid(row=0, column=0, sticky="w", padx=5, pady=(0, 4))
                label.grid(
                    row=1,
                    column=0,
                    columnspan=2,
                    sticky="ew",
                    padx=5,
                )
                label.configure(wraplength=max(300, width - 70))
            else:
                button.grid(row=0, column=0, sticky="w", padx=5)
                label.grid(row=0, column=1, sticky="ew", padx=10)
                label.configure(wraplength=max(400, width - 330))
        except tk.TclError:
            pass

    def _navigate_back_shortcut(self, _event=None):
        if (
            self.current_view == "wizard"
            and str(self.back_button.cget("state")) != str(tk.DISABLED)
        ):
            self.go_back()
        return "break"

    def _navigate_next_shortcut(self, _event=None):
        if (
            self.current_view == "wizard"
            and str(self.next_button.cget("state")) != str(tk.DISABLED)
        ):
            self.go_next()
        return "break"

    def set_workflow_navigation_visible(self, visible):
        """Show or hide the fixed wizard navigation without changing content."""
        if visible:
            footer_height = 88 if self._responsive_window_width() < 920 else 58
            self.rowconfigure(2, minsize=footer_height)
            self.workflow_footer.grid()
            self.workflow_footer.tkraise()
        else:
            self.workflow_footer.grid_remove()
            self.rowconfigure(2, minsize=0)

    def keep_workflow_navigation_visible(self):
        """Reassert footer placement after a wizard page finishes building."""
        if self.current_view != "wizard":
            return
        self.workflow_footer.grid()
        self.workflow_footer.tkraise()
        self._reflow_workflow_footer(self._responsive_window_width() < 920)

    def get_workflow_hint(self):
        signal_hints = [
            "Load a CSV or Excel dataset, then select Next.",
            "Optional step — review local or AI advice, or continue without AI.",
            "Choose any built-in filtering required before training.",
            "Optional step — approve a safe filter or add trusted Python code.",
            "Choose the prediction task, model inputs, and target outputs.",
            "Confirm missing-value handling, scaling, and the test split.",
            "Review model settings. You can still train without AI assistance.",
            "Train the model, inspect the results, and save the model package.",
        ]
        image_hints = [
            "Load a class-organized image folder, then select Next.",
            "Review the image count and class balance before continuing.",
            "Review the image-processing pipeline.",
            "Confirm how image transformations preserve class meaning.",
            "Confirm the image-classification task and class folders.",
            "Choose image size, augmentation, and dataset splits.",
            "Choose the image model and training settings.",
            "Train the model, inspect the results, and save the model package.",
        ]
        hints = (
            image_hints
            if self.data_mode_var.get() == DATA_MODE_IMAGE
            else signal_hints
        )
        if 0 <= self.current_step < len(hints):
            return hints[self.current_step]
        return "Use Back and Next to move through the training workflow."

    def open_model_evaluation(self):
        ModelLoadingHubWindow(self)

    def open_deploy_integrate(self, initial_model_path=None, initial_goal=None):
        DeployIntegrateWindow(
            self,
            initial_model_path=initial_model_path,
            initial_goal=initial_goal,
        )

    def open_object_detection(self, initial_model_path=None):
        ObjectDetectionWindow(
            self,
            initial_model_path=initial_model_path,
        )

    def open_annotation_workspace(self):
        existing = self.annotation_workspace_window
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.deiconify()
                    existing.lift()
                    existing.focus_force()
                    return
            except tk.TclError:
                pass
        self.annotation_workspace_window = AnnotationWorkspaceWindow(self)

    def get_studio_guide_context(self):
        """Describe only the visible application page; never include user data."""
        if self.current_view == "wizard" and self.is_guided() and self.guided_detail_step is None:
            return "Guided · " + self.GUIDED_STAGES[self.guided_stage]
        if self.current_view == "wizard":
            return self.step_titles[self.current_step]
        context_labels = {
            "home": "Homepage",
            WORKSPACE_FILTER: "Filter & Export Data",
            WORKSPACE_ANNOTATE: "Annotate & Prepare Data",
            WORKSPACE_DEPLOY: "Deploy & Integrate Model",
            "projects": "Projects & Checkpoints",
            "settings": "Settings",
        }
        return context_labels.get(
            self.current_view,
            str(self.current_view or "Homepage"),
        )

    def open_studio_guide(self, initial_question=None):
        existing = self.studio_guide_window
        if existing is not None:
            try:
                if existing.winfo_exists():
                    existing.deiconify()
                    existing.lift()
                    existing.focus_force()
                    if initial_question:
                        existing.ask_question(initial_question)
                    return
            except tk.TclError:
                pass
        self.studio_guide_window = StudioGuideWindow(
            self,
            initial_question=initial_question,
        )

    def run_studio_guide_action(self, action):
        """Route a guide recommendation through the normal public workspace."""
        action_map = {
            WORKSPACE_SIGNAL: self.start_signal_project,
            WORKSPACE_IMAGE: self.start_image_project,
            WORKSPACE_DETECTION: self.open_object_detection,
            WORKSPACE_FILTER: self.show_filter_workspace,
            WORKSPACE_ANNOTATE: self.open_annotation_workspace,
            GUIDE_ACTION_MODEL_APPLICATION: self.open_model_evaluation,
            GUIDE_ACTION_DEPLOY: self.open_deploy_integrate,
            GUIDE_ACTION_PROJECTS: self.show_projects,
            GUIDE_ACTION_SETTINGS: self.show_settings,
        }
        command = action_map.get(action)
        if command is None:
            messagebox.showwarning(
                "NN Studio Guide",
                "The recommended workspace is not available.",
            )
            return
        command()
