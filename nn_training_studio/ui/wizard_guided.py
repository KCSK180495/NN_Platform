"""Ui / wizard guided for NN Training Studio."""

from pathlib import Path
from tkinter.scrolledtext import ScrolledText
from tkinter import filedialog
import io
import json
from tkinter import messagebox
import os
import pandas as pd
import tempfile
import tkinter as tk
from tkinter import ttk
from nn_training_studio.constants import (
    CUSTOM_FILTER_MODE_EXPERT,
    CUSTOM_FILTER_MODE_SAFE_AI,
    DATA_MODE_IMAGE,
    DATA_MODE_TABULAR,
    IMAGE_MODEL_CNN,
    TASK_AUTOENCODER,
    TASK_CLASSIFICATION,
    TASK_FORECASTING,
    TASK_MULTI_OUTPUT,
    TASK_REGRESSION,
    TASK_TYPES,
)
from nn_training_studio.results import (
    make_training_prediction_dataframe,
)
from nn_training_studio.training import (
    validate_early_stopping_settings,
)


class GuidedWorkflowMixin:
    """GuidedWorkflow behavior for the main application."""

    GUIDED_STAGES = ("Goal & Data", "Prepare Data", "Train AI", "Check Results", "Use & Export")

    GUIDED_STEP_MAP = (0, 5, 7, 7, 7)

    GUIDED_GOALS = {
        "Identify a category (for example, good or defective)": TASK_CLASSIFICATION,
        "Predict one numerical value": TASK_REGRESSION,
        "Predict several numerical values": TASK_MULTI_OUTPUT,
        "Predict future measurements": TASK_FORECASTING,
        "Reconstruct data and score unusual samples": TASK_AUTOENCODER,
    }

    DRAFT_VARIABLES = (
        "task_type_var", "label_var", "forecast_horizon_var", "anomaly_percentile_var",
        "missing_var", "scaler_var", "target_scaler_var", "test_size_var",
        "model_type_var", "window_size_var", "stride_var", "epochs_var", "batch_size_var",
        "training_device_var", "validation_split_var", "image_height_var", "image_width_var",
        "early_stopping_enabled_var", "early_stopping_patience_var",
        "early_stopping_min_delta_var", "early_stopping_warmup_var",
        "image_color_mode_var", "image_augmentation_var", "image_pretrained_var",
        "hidden_activation_var", "dropout_var", "output_mode_var", "output_units_var",
        "output_activation_var", "loss_var", "optimizer_var", "lr_var",
        "filter_method_var", "moving_window_var", "ema_span_var", "median_window_var",
        "kalman_q_var", "kalman_r_var", "sampling_frequency_var",
        "custom_filter_enabled_var", "custom_filter_mode_var", "custom_model_input_mode_var",
        "project_goal_var", "guided_goal_var", "guided_data_order_var",
    )

    def is_guided(self):
        return self.ui_mode_var.get() == "Guided"

    def _operation_busy(self):
        return bool(self.training_running or self.ai_request_running)

    def _capture_page_state(self):
        """Capture UI edits without validating, running code, or changing data."""
        for widget_name, field in (("feature_listbox", "feature_cols"),
                                   ("target_listbox", "target_cols"),
                                   ("filter_listbox", "selected_filter_cols"),
                                   ("custom_filter_listbox", "selected_custom_filter_cols")):
            widget = getattr(self, widget_name, None)
            if widget is not None and widget.winfo_exists():
                setattr(self, field, [widget.get(i) for i in widget.curselection()])
        for name in ("custom_filter_code_text", "custom_model_code_text", "safe_filter_json_text",
                     "safe_filter_summary_text", "safe_model_summary_text"):
            widget = getattr(self, name, None)
            if widget is not None and widget.winfo_exists():
                self._editor_drafts[name] = widget.get("1.0", "end-1c")
        for name, field in (("custom_filter_code_text", "custom_filter_code"),
                            ("custom_model_code_text", "custom_model_code")):
            if name in self._editor_drafts and not (
                field == "custom_filter_code" and self.custom_filter_mode_var.get() == CUSTOM_FILTER_MODE_SAFE_AI
            ):
                setattr(self, field, self._editor_drafts[name])
        if hasattr(self, "result_text") and self.result_text.winfo_exists():
            self._last_training_log = self.result_text.get("1.0", "end-1c")

    def _restore_editor_drafts(self):
        for name, value in self._editor_drafts.items():
            widget = getattr(self, name, None)
            if widget is not None and widget.winfo_exists():
                widget.delete("1.0", tk.END)
                widget.insert("1.0", value)

    def change_ui_mode(self, _event=None):
        if self._operation_busy():
            self.ui_mode_var.set(self._active_ui_mode)
            messagebox.showinfo("Operation running", "Finish the current operation before changing mode.")
            return
        self._capture_page_state()
        if self.current_view == "wizard" and self.current_step in (2, 3):
            self._pending_preparation_steps.add(self.current_step)
        if self.is_guided():
            self.guided_stage = {0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5: 1, 6: 2, 7: 2}.get(self.current_step, 0)
            if self.current_step == 7 and self.trained_model is not None:
                self.guided_stage = 3
        elif self.guided_detail_step is None:
            self.current_step = self.GUIDED_STEP_MAP[self.guided_stage]
        self.guided_detail_step = None
        self._active_ui_mode = self.ui_mode_var.get()
        if self.current_view == "wizard":
            self.show_step()
        elif self.current_view == "home":
            self.show_home()

    def _guided_note(self, text, parent=None):
        label = ttk.Label(parent or self.content_frame, text=text, wraplength=1000, justify=tk.LEFT)
        label.pack(fill=tk.X, padx=10, pady=7)
        return label

    def _guided_actions(self, actions, parent=None):
        frame = ttk.Frame(parent or self.content_frame)
        frame.pack(fill=tk.X, padx=8, pady=6)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        for index, (label, command) in enumerate(actions):
            ttk.Button(frame, text=label, command=command).grid(
                row=index // 2, column=index % 2, sticky="ew", padx=3, pady=3)
        return frame

    def build_home_page(self):
        if not self.is_guided():
            self.build_advanced_home_page()
            return
        self._guided_note("What do you want to achieve?").configure(font=("Arial", 18, "bold"))
        self._guided_note("Choose a starting point. Each workflow explains what you need and what to do next.")
        if self.df is not None:
            self._guided_note(f"Current project: {Path(self.file_path or 'Prepared data').name} · "
                              f"{self.GUIDED_STAGES[self.guided_stage]}")
            self._guided_actions([("Continue my project", self.show_step),
                                  ("Save project draft", self.save_project_draft)])
        else:
            self._guided_actions([("Open project draft", self.open_project_draft),
                                  ("Help me choose", self.open_start_chooser)])
        self.guided_home_grid = ttk.Frame(self.content_frame)
        self.guided_home_grid.pack(fill=tk.X, padx=8, pady=8)
        choices = (
            ("Create AI using my data", "Start with a spreadsheet, measurements, or pictures.", self.open_start_chooser),
            ("Use an existing AI model", "Load a saved model and try it on your data.", self.open_model_evaluation),
            ("Add AI to my application", "Connect a trained model or AI service to your code.", self.open_deploy_integrate),
            ("Prepare and label my data", "Clean measurements or label pictures and signal intervals.", self.open_prepare_chooser),
        )
        self.guided_home_cards = []
        for index, (title, description, command) in enumerate(choices):
            card = self.add_home_action(self.guided_home_grid, index // 2, index % 2,
                                        title, description, "Open workflow", command)
            self.guided_home_cards.append(card)
        self._guided_note("Guided mode works without an AI connection. Optional AI assistance is configured in Settings.")
        self.after_idle(self._apply_responsive_layout)

    def _reflow_home_page(self, width):
        if not self.is_guided():
            self._reflow_advanced_home_page(width)
            return
        if not hasattr(self, "guided_home_grid"):
            return
        columns = 1 if width < 920 else 2
        for col in range(2):
            self.guided_home_grid.columnconfigure(col, weight=int(col < columns), uniform="guided_cards")
        for index, card in enumerate(self.guided_home_cards):
            card.grid(row=index // columns, column=index % columns, sticky="nsew", padx=6, pady=6)

    def open_start_chooser(self):
        if self._operation_busy():
            return
        dialog = tk.Toplevel(self)
        dialog.title("Create AI · choose your data")
        dialog.geometry("580x380")
        dialog.minsize(440, 330)
        self._guided_note("What will your AI learn from?", dialog)
        self._guided_note("Choose the example closest to your application. You can describe your exact goal next.", dialog)
        def choose(command):
            dialog.destroy()
            command()
        for label, command in (
            ("Spreadsheet or measurements", self.start_signal_project),
            ("Pictures — identify a whole-image category", self.start_image_project),
            ("Pictures — find and locate objects", self.open_object_detection),
        ):
            ttk.Button(dialog, text=label, command=lambda c=command: choose(c)).pack(fill=tk.X, padx=15, pady=8)
        ttk.Button(dialog, text="I am not sure — ask the Guide",
                   command=lambda: choose(self.open_studio_guide)).pack(fill=tk.X, padx=15, pady=8)

    def open_prepare_chooser(self):
        dialog = tk.Toplevel(self)
        dialog.title("Prepare and label data")
        self._guided_note("What would you like to prepare?", dialog)
        for label, command in (("Label images or signal intervals", self.open_annotation_workspace),
                               ("Clean measurements and compare the signal", self.show_filter_workspace)):
            def choose(c=command):
                dialog.destroy()
                c()
            ttk.Button(dialog, text=label, command=choose).pack(fill=tk.X, padx=15, pady=10)

    def show_step(self):
        if self._operation_busy() and self.current_view == "wizard":
            return
        if not self.is_guided() or self.guided_detail_step is not None:
            self._show_advanced_step()
            self._restore_editor_drafts()
            if self.is_guided():
                self.title_label.configure(text=f"{self.GUIDED_STAGES[self.guided_stage]} · More options")
                self.progress_label.configure(text="Apply your changes to return to the guided workflow.")
                self.back_button.configure(text="Return", state=tk.NORMAL)
                self.next_button.configure(text="Apply & return", state=tk.NORMAL)
            return
        self.current_view = "wizard"
        self.clear_content()
        self.current_step = self.GUIDED_STEP_MAP[self.guided_stage]
        self.set_workflow_navigation_visible(True)
        self.home_button.configure(state=tk.NORMAL)
        self.settings_button.configure(state=tk.NORMAL)
        self.title_label.configure(text=self.GUIDED_STAGES[self.guided_stage])
        self.progress_label.configure(text=f"Stage {self.guided_stage + 1} of 5 · Guided mode")
        self._guided_actions([("Help with this step", self.show_guided_help),
                              ("Save project draft", self.save_project_draft)])
        self.guided_error_var = tk.StringVar(value="")
        ttk.Label(self.content_frame, textvariable=self.guided_error_var, foreground="#b42318",
                  wraplength=1000).pack(fill=tk.X, padx=10, pady=2)
        builders = (self.build_guided_goal, self.build_guided_prepare, self.build_guided_train,
                    self.build_guided_results, self.build_guided_export)
        builders[self.guided_stage]()
        self._update_guided_footer()
        self.after_idle(self._apply_responsive_layout)

    def _update_guided_footer(self):
        if not self.is_guided() or self.guided_detail_step is not None:
            return
        hints = ("Confirm the result you want and the data used to learn it.",
                 "Review your data. Cleaning and labelling tools are optional.",
                 "Check the setup, then start training. Results become available when training finishes.",
                 "Review performance on held-out data and inspect example mistakes.",
                 "Save the trained model and results before using them elsewhere.")
        self.workflow_hint_label.configure(text=hints[self.guided_stage])
        self.back_button.configure(text="Back", state=tk.NORMAL if self.guided_stage > 0 else tk.DISABLED)
        labels = ("Review data →", "Review training →", "Check results →", "Use & export →", "Return home")
        ready = not self._operation_busy() and (self.guided_stage not in (2, 3) or self.trained_model is not None)
        self.next_button.configure(text=labels[self.guided_stage], state=tk.NORMAL if ready else tk.DISABLED)
        if self._operation_busy():
            self.back_button.configure(state=tk.DISABLED)
        self.keep_workflow_navigation_visible()

    def _guided_error(self, exc):
        message = "Action needed: " + str(exc)
        if hasattr(self, "guided_error_var"):
            self.guided_error_var.set(message)
            self.content_canvas.yview_moveto(0.0)
        else:
            messagebox.showerror("Check your setup", message)

    def go_next(self):
        if self._operation_busy():
            return
        if not self.is_guided():
            if self.current_step < 7:
                self._advanced_go_next()
            return
        try:
            if self.guided_detail_step is not None:
                validators = {1: self.validate_step_ai_analysis, 2: self.validate_step_2_filter,
                              3: self.validate_step_3_custom_filter, 4: self.validate_step_2,
                              5: self.validate_step_3, 6: self.validate_step_4}
                validators[self.guided_detail_step]()
                if self.guided_detail_step == 2 and self.custom_filter_enabled_var.get():
                    # The custom stage must be reapplied to the new built-in output.
                    self.current_step = self.guided_detail_step = 3
                    self.show_step()
                    return
                self.guided_detail_step = None
            elif self.guided_stage == 0:
                self.validate_step_1()
                self.validate_step_2()
                self.guided_stage = 1
            elif self.guided_stage == 1:
                self._check_preparation_applied()
                self.validate_step_3()
                self.guided_stage = 2
            elif self.guided_stage in (2, 3):
                if self.trained_model is None:
                    raise ValueError("Train the model successfully before checking or exporting results.")
                self.guided_stage += 1
            else:
                self.show_home()
                return
            self.show_step()
            self._autosave_project_draft()
        except Exception as exc:
            self._guided_error(exc)

    def go_back(self):
        if self._operation_busy():
            return
        if not self.is_guided():
            self._advanced_go_back()
            return
        if self.guided_detail_step is not None:
            self.guided_detail_step = None
        elif self.guided_stage > 0:
            self.guided_stage -= 1
        self.show_step()

    def open_guided_detail(self, step):
        if self._operation_busy():
            return
        if self.df is None:
            self._guided_error("Load your data first.")
            return
        if step in (1, 2, 3) and self.dataset_profile is None:
            self.run_dataset_analysis(show_message=False)
        if step in (2, 3):
            self._pending_preparation_steps.add(step)
        self.guided_detail_step = self.current_step = step
        self.show_step()

    def _check_preparation_applied(self):
        if self._pending_preparation_steps:
            names = {2: "Clean signals", 3: "Custom filter"}
            tools = ", ".join(names[step] for step in sorted(self._pending_preparation_steps))
            raise ValueError(f"Open {tools} and select Apply & return to confirm your preparation settings.")

    def _guided_goal_changed(self, _event=None):
        self._capture_page_state()
        self.task_type_var.set(self.GUIDED_GOALS[self.guided_goal_var.get()])
        self.apply_task_defaults()
        if self.task_type_var.get() == TASK_CLASSIFICATION and self.guided_data_order_var.get() == "Independent rows":
            self.model_type_var.set("DNN")
        if self.task_type_var.get() == TASK_AUTOENCODER:
            self.target_cols = []
        self.show_step()

    def build_guided_goal(self):
        self._guided_note("Describe your application (optional)")
        ttk.Entry(self.content_frame, textvariable=self.project_goal_var).pack(fill=tk.X, padx=10, pady=4)
        self._guided_note("Example: predict energy consumption from temperature and operating hours.")
        self.build_step_1()
        self.step1_text.configure(height=5)
        if self.df is None:
            return
        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            self._guided_note("Your AI will identify one category for each whole image. Folder names are the answers it learns.")
            self._guided_note("Check the folder names and examples above. For object locations, use the object-detection workflow from Home.")
            return
        goal = next((label for label, task in self.GUIDED_GOALS.items() if task == self.task_type_var.get()),
                    next(iter(self.GUIDED_GOALS)))
        self.guided_goal_var.set(goal)
        self._guided_note("What result would you like to predict?")
        combo = ttk.Combobox(self.content_frame, textvariable=self.guided_goal_var,
                             values=list(self.GUIDED_GOALS), state="readonly")
        combo.pack(fill=tk.X, padx=10, pady=4)
        combo.bind("<<ComboboxSelected>>", self._guided_goal_changed)
        self._guided_note("Choose the answer column(s), then the input columns. Click a name to select or deselect it.")
        self._guided_note("Suggested columns are a starting point. Confirm that the answer really represents your intended result.")
        for title, name, selected in (("Answers the AI should learn", "target_listbox", self.target_cols),
                                       ("Inputs available when making a prediction", "feature_listbox", self.feature_cols)):
            frame = ttk.LabelFrame(self.content_frame, text=title)
            frame.pack(fill=tk.X, padx=10, pady=5)
            box = tk.Listbox(frame, selectmode=tk.MULTIPLE, height=5, exportselection=False)
            scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=box.yview)
            scroll.pack(side=tk.RIGHT, fill=tk.Y)
            box.configure(yscrollcommand=scroll.set)
            box.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
            setattr(self, name, box)
            for index, column in enumerate(self.df.columns):
                box.insert(tk.END, column)
                if column in selected:
                    box.selection_set(index)
            if name == "target_listbox" and self.task_type_var.get() == TASK_AUTOENCODER:
                box.selection_clear(0, tk.END)
                box.configure(state=tk.DISABLED)
        self._guided_actions([("Select numeric inputs", self.select_numeric_features),
                              ("Help choose inputs and answers", self.show_guided_help)])

    def build_guided_prepare(self):
        if self.df is None:
            self._guided_note("No data is loaded. Go Back to load your data.")
            return
        self._guided_note(f"Loaded: {len(self.df):,} records. Inputs: {', '.join(map(str, self.feature_cols))}. "
                          f"Answers: {', '.join(map(str, self.target_cols)) or 'reconstruct inputs'}.")
        if self._pending_preparation_steps:
            self._guided_note("Some filter settings are awaiting Apply. Reopen the filter tool and select Apply & return before training.")
        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            self._guided_note(f"Images will be resized to {self.image_height_var.get()} × {self.image_width_var.get()}. "
                              "Only use augmentation that preserves the meaning of your labels.")
        else:
            cols = [c for c in self.feature_cols + self.target_cols if c in self.df.columns]
            missing = int(self.df[cols].isna().sum().sum())
            self._guided_note(f"Missing values in selected columns: {missing:,}. "
                              f"Built-in filter: {self.filter_method_var.get()}. "
                              f"Custom filter: {'enabled' if self.custom_filter_enabled_var.get() else 'off'}.")
            self._guided_note("Are your rows independent examples, or consecutive measurements? This guides the starting model choice.")
            ttk.Combobox(self.content_frame, textvariable=self.guided_data_order_var, state="readonly",
                         values=("Independent rows", "Ordered measurements")).pack(fill=tk.X, padx=10, pady=5)
            self._guided_note("For ordered measurements, preserve time order and keep related recordings together when evaluating results.")
        self._guided_note(f"Test data: {self.test_size_var.get()}%. Validation: {self.validation_split_var.get()}% of the remaining data. "
                          "Test data checks performance after training; validation helps select the training checkpoint.")
        actions = [("Data checks and suggestions", lambda: self.open_guided_detail(1)),
                   ("Data preparation settings", lambda: self.open_guided_detail(5))]
        if self.data_mode_var.get() != DATA_MODE_IMAGE:
            actions += [("Clean signals and preview plots", lambda: self.open_guided_detail(2)),
                        ("Custom filter (optional)", lambda: self.open_guided_detail(3))]
        actions += [("Label data", self.open_annotation_workspace)]
        self._guided_actions(actions)

    def apply_guided_starting_settings(self):
        if self._operation_busy():
            return
        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            self.model_type_var.set(IMAGE_MODEL_CNN)
            self.output_activation_var.set("softmax")
            self.loss_var.set("sparse_categorical_crossentropy")
        else:
            self.apply_task_defaults()
            if self.task_type_var.get() == TASK_CLASSIFICATION:
                self.model_type_var.set("CNN-LSTM" if self.guided_data_order_var.get() == "Ordered measurements" else "DNN")
        self.output_mode_var.set("Auto")
        self.optimizer_var.set("Adam")
        self.lr_var.set("0.001")
        self.show_step()

    def build_guided_train(self):
        self._guided_note(f"Current model: {self.model_type_var.get()}. Current training length: {self.epochs_var.get()} rounds (epochs).")
        self._guided_note("Starting settings provide a baseline. Check results before deciding whether the model is suitable for your application.")
        form = ttk.Frame(self.content_frame)
        self.guided_training_settings_frame = form
        form.pack(fill=tk.X, padx=10, pady=5)
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Training rounds (epochs)").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(form, textvariable=self.epochs_var).grid(row=0, column=1, sticky="ew")
        self.build_early_stopping_controls(form).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )
        self._guided_actions([("Use starting settings", self.apply_guided_starting_settings),
                              ("Advanced model settings", lambda: self.open_guided_detail(6))])
        self.build_step_5()

    def _readonly_text(self, text, height=9):
        widget = ScrolledText(self.content_frame, height=height, wrap=tk.WORD)
        widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=7)
        widget.insert("1.0", text)
        widget.configure(state=tk.DISABLED)
        return widget

    def build_guided_results(self):
        if self.trained_model is None:
            self._guided_note("No completed model is available. Return to Train AI to train your model.")
            return
        metadata = self.metadata or {}
        task = metadata.get("task_type", self.task_type_var.get())
        self._guided_note("Last completed training run · held-out test results")
        metrics = metadata.get("metrics", {})
        lines = [f"{name.replace('_', ' ').capitalize()}: {value}" for name, value in metrics.items()]
        if task == TASK_CLASSIFICATION:
            meaning = "Accuracy is the fraction of test examples classified correctly. Inspect errors in every class, especially rare classes."
        elif task == TASK_AUTOENCODER:
            meaning = "Reconstruction error measures how closely the model reproduces its inputs. An unusual score alone does not confirm a fault."
        else:
            meaning = "MAE is the average absolute prediction error; RMSE gives more weight to larger errors. Read each output in its own units."
        self._guided_note(meaning)
        self._readonly_text("\n".join(lines) or "Detailed metrics are available in the saved results.", height=5)
        try:
            predictions = make_training_prediction_dataframe(
                task_type=task, actual_values=self.y_test_result, predicted_values=self.y_pred_result,
                class_names=self.class_names, target_names=self.result_target_names, anomaly_scores=self.anomaly_scores)
            self._guided_note("Example predictions (first 12 test examples)")
            self._readonly_text(predictions.head(12).to_string(index=False), height=8)
        except Exception as exc:
            self._guided_note(f"Example preview unavailable: {exc}")
        self._guided_actions([("Show result graphs", self.show_plots),
                              ("Customize plots / AI", self.open_training_results_studio),
                              ("Save complete results", self.save_training_results)])
        self._guided_note("Use new data from your intended application to check how well these results generalise.")

    def build_guided_export(self):
        self._guided_note("Save the last completed model with its preprocessing and input order, then connect it to your application.")
        self._guided_actions([("Save trained model package", self.save_model_package),
                              ("Save complete test results", self.save_training_results),
                              ("Connect saved model to my application", self.open_last_saved_deployment),
                              ("Try an existing model on new data", self.open_model_evaluation)])
        self._guided_note("The integration workspace can also add an external AI service to your application without local training.")
        self._guided_note("Project drafts save your setup and prepared tables. Trained models, result files, and image files are saved separately.")

    def show_guided_help(self):
        help_text = (
            "Choose the outcome first. Answers are the values or labels the model should learn; inputs are information available when making a prediction. The answer column must not also be selected as an input.",
            "Inspect missing values, labels, and class counts. Use cleaning only if it fits your data. Open a filter tool to compare the original and processed signal. You can continue with no filter.",
            "Check training setup to see how many examples will be used for training, validation, and testing. Start training when the checks pass. Advanced model settings contains optional architecture and optimisation controls.",
            "These results describe the last completed run. Review mistakes and performance for each class or output. Strong performance on this test set does not guarantee the same result on new applications.",
            "Save the model package and results. Connect the saved model using the integration wizard. A project draft lets you return to data preparation and settings; it does not contain a trained model.",
        )
        stage = self.guided_stage if self.is_guided() else {0: 0, 1: 1, 2: 1, 3: 1, 4: 0, 5: 1, 6: 2, 7: 2}.get(self.current_step, 0)
        dialog = tk.Toplevel(self)
        dialog.title("Help · " + self.GUIDED_STAGES[stage])
        dialog.geometry("560x310")
        dialog.minsize(420, 300)
        label = self._guided_note(help_text[stage], dialog)
        label.configure(wraplength=510)
        dialog.bind("<Configure>", lambda e: label.configure(wraplength=max(280, e.width - 35)) if e.widget == dialog else None)
        ttk.Button(dialog, text="Ask the optional AI Guide", command=lambda: self.open_studio_guide(
            "Explain this step in plain language: " + self.GUIDED_STAGES[stage])).pack(fill=tk.X, padx=15, pady=10)
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=5)

    def _draft_payload(self):
        self._capture_page_state()
        return {
            "format": "nnstudio-project-draft", "version": 1,
            "data_mode": self.data_mode_var.get(), "source_file": self.file_path,
            "image_directory": self.image_directory, "image_class_names": self.image_class_names,
            "guided_stage": min(self.guided_stage, 2), "current_step": min(self.current_step, 6),
            "variables": {name: getattr(self, name).get() for name in self.DRAFT_VARIABLES},
            "columns": {name: list(getattr(self, name)) for name in (
                "feature_cols", "target_cols", "selected_filter_cols", "selected_custom_filter_cols")},
            "frames": {name: getattr(self, name).to_json(orient="table", index=False)
                       for name in ("raw_df", "filtered_df", "custom_filtered_df", "df")
                       if isinstance(getattr(self, name, None), pd.DataFrame)},
            "specs": {"safe_filter_spec": self.safe_filter_spec, "safe_model_spec": self.safe_model_spec},
            "code": {"custom_filter_code": self.custom_filter_code, "custom_model_code": self.custom_model_code},
            "editor_drafts": dict(self._editor_drafts),
            "pending_preparation_steps": sorted(self._pending_preparation_steps),
        }

    def _write_project_draft(self, path):
        payload = self._draft_payload()
        destination = Path(path)
        # Same-directory staging provides an atomic update of an existing draft.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent,
                                             prefix=".nnstudio-draft-", suffix=".tmp", delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False)
            os.replace(temporary, destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def save_project_draft(self):
        if self._operation_busy():
            messagebox.showinfo("Operation running", "Save your draft after the current operation finishes.")
            return
        if self.df is None:
            messagebox.showinfo("Load data first", "Load your data before saving a project draft.")
            return
        path = filedialog.asksaveasfilename(title="Save project draft (setup and prepared data)",
                    defaultextension=".nnstudio.json", filetypes=[("NN Studio draft", "*.nnstudio.json")])
        if not path:
            return
        try:
            self._write_project_draft(path)
            self.project_draft_path = path
            messagebox.showinfo("Draft saved", "Setup and prepared tables saved. This draft updates automatically when you advance a guided stage.\n\nSave trained models separately; keep image files at their existing locations.")
        except Exception as exc:
            messagebox.showerror("Draft save failed", str(exc))

    def _autosave_project_draft(self):
        if not self.project_draft_path or self._operation_busy():
            return
        try:
            self._write_project_draft(self.project_draft_path)
        except Exception as exc:
            self._guided_error("Draft could not be updated: " + str(exc))

    def open_project_draft(self):
        if self._operation_busy():
            return
        path = filedialog.askopenfilename(title="Open NN Studio project draft", filetypes=[("NN Studio draft", "*.nnstudio.json")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("format") != "nnstudio-project-draft" or payload.get("version") != 1:
                raise ValueError("This file is not a supported NN Studio project draft.")
            mode = payload.get("data_mode")
            if mode not in (DATA_MODE_IMAGE, DATA_MODE_TABULAR):
                raise ValueError("The draft has an unsupported data type.")
            frames = {name: pd.read_json(io.StringIO(payload["frames"][name]), orient="table")
                      for name in ("raw_df", "filtered_df", "custom_filtered_df", "df") if name in payload.get("frames", {})}
            if "df" not in frames or frames["df"].empty or "raw_df" not in frames:
                raise ValueError("The draft does not contain a usable dataset.")
            variables = payload.get("variables", {})
            if not isinstance(variables, dict):
                raise ValueError("Invalid draft settings.")
            for name in self.DRAFT_VARIABLES:
                if name in variables:
                    if isinstance(getattr(self, name), tk.BooleanVar):
                        if not isinstance(variables[name], bool):
                            raise ValueError(f"Invalid draft option: {name}.")
                    elif not isinstance(variables[name], str):
                        raise ValueError(f"Invalid draft field: {name}.")
            if variables.get("task_type_var", TASK_CLASSIFICATION) not in TASK_TYPES:
                raise ValueError("Unsupported prediction task in draft.")
            for name in ("epochs_var", "batch_size_var", "window_size_var", "stride_var", "image_height_var", "image_width_var"):
                if name in variables and int(variables[name]) <= 0:
                    raise ValueError(f"The draft setting {name} must be a positive whole number.")
            validate_early_stopping_settings({
                "enabled": variables.get("early_stopping_enabled_var", True),
                "patience": variables.get("early_stopping_patience_var", "10"),
                "min_delta": variables.get("early_stopping_min_delta_var", "0"),
                "start_from_epoch": variables.get("early_stopping_warmup_var", "0"),
            }, int(variables.get("epochs_var", self.epochs_var.get())))
            for name in ("columns", "specs", "code", "editor_drafts"):
                if not isinstance(payload.get(name, {}), dict):
                    raise ValueError(f"Invalid draft section: {name}.")
            for values in payload.get("columns", {}).values():
                if not isinstance(values, list) or not all(isinstance(c, str) for c in values):
                    raise ValueError("Invalid column names in draft.")
            for name in ("source_file", "image_directory"):
                if payload.get(name) is not None and not isinstance(payload[name], str):
                    raise ValueError(f"Invalid draft path: {name}.")
            if not isinstance(payload.get("image_class_names", []), list):
                raise ValueError("Invalid image classes in draft.")
            if not isinstance(payload.get("pending_preparation_steps", []), list):
                raise ValueError("Invalid preparation status in draft.")
            stage = max(0, min(2, int(payload.get("guided_stage", 0))))
            step = max(0, min(6, int(payload.get("current_step", 0))))
            if self.df is not None and not messagebox.askyesno("Open another draft", "Replace the current in-memory project with this draft?"):
                return
            self.clear_content()
            self.reset_training_project(mode)
            for name in self.DRAFT_VARIABLES:
                if name in variables:
                    getattr(self, name).set(variables[name])
            for name, frame in frames.items():
                setattr(self, name, frame)
            self.filtered_df = frames.get("filtered_df", self.raw_df.copy())
            self.custom_filtered_df = frames.get("custom_filtered_df", self.df.copy())
            for name in ("feature_cols", "target_cols", "selected_filter_cols", "selected_custom_filter_cols"):
                setattr(self, name, [c for c in payload.get("columns", {}).get(name, []) if c in self.df.columns])
            self.file_path = payload.get("source_file")
            self.label_col = self.target_cols[0] if self.target_cols else ""
            self.label_var.set(self.label_col)
            self.image_directory = payload.get("image_directory")
            self.image_class_names = list(payload.get("image_class_names", []))
            self.image_records = self.df.copy() if mode == DATA_MODE_IMAGE else None
            self._pending_preparation_steps = {int(step) for step in payload.get("pending_preparation_steps", []) if step in (2, 3)}
            for name in ("safe_filter_spec", "safe_model_spec"):
                setattr(self, name, payload.get("specs", {}).get(name))
            for name in ("custom_filter_code", "custom_model_code"):
                if isinstance(payload.get("code", {}).get(name), str):
                    setattr(self, name, payload["code"][name])
            self._editor_drafts = {k: v for k, v in payload.get("editor_drafts", {}).items()
                                  if k in ("custom_filter_code_text", "custom_model_code_text", "safe_filter_json_text",
                                           "safe_filter_summary_text", "safe_model_summary_text") and isinstance(v, str)}
            # Restoring text never executes it. Expert filters must be enabled again explicitly.
            if self.custom_filter_mode_var.get() == CUSTOM_FILTER_MODE_EXPERT and self.custom_filter_enabled_var.get():
                self.custom_filter_enabled_var.set(False)
                self._pending_preparation_steps.add(3)
            if self.model_type_var.get() == "Custom Python Model":
                self.guided_stage = 2
                self.guided_detail_step = self.current_step = 6
            else:
                self.guided_stage, self.current_step = stage, step
            self.project_draft_path = path
            self.show_step()
        except Exception as exc:
            messagebox.showerror("Cannot open draft", str(exc))
