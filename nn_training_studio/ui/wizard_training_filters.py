"""Ui / wizard training filters for NN Training Studio."""

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from nn_training_studio.constants import (
    DATA_MODE_IMAGE,
    TASK_AUTOENCODER,
    TASK_CLASSIFICATION,
    TASK_FORECASTING,
)
from nn_training_studio.filters import (
    apply_selected_filter,
)


class TrainingFiltersMixin:
    """TrainingFilters behavior for the main application."""

    def build_step_2_filter(self):
        if self.raw_df is None:
            ttk.Label(
                self.content_frame,
                text="Please load dataset first."
            ).pack()
            return

        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            frame = ttk.LabelFrame(
                self.content_frame,
                text="Image Filtering",
            )
            frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            ttk.Label(
                frame,
                text=(
                    "Numeric signal filters are skipped for photo datasets.\n\n"
                    "Images will be decoded, resized, and normalized by the "
                    "selected image model. Optional image augmentation is "
                    "configured on the preprocessing page."
                ),
                justify=tk.LEFT,
                wraplength=900,
                font=("Arial", 12),
            ).pack(anchor="w", padx=20, pady=20)
            return

        left_frame = ttk.LabelFrame(
            self.content_frame,
            text="Filter Settings"
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        right_frame = ttk.LabelFrame(
            self.content_frame,
            text="Filtered Data Preview"
        )
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        numeric_columns = self.raw_df.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns.tolist()

        # Default is no filter column selected.
        # This prevents accidentally filtering columns such as speed, ID, label, or other features.
        default_filter_cols = []

        ttk.Label(
            left_frame,
            text="Select columns to filter only. Unselected columns stay unchanged:",
            font=("Arial", 11, "bold")
        ).pack(anchor="w", padx=5, pady=5)

        self.filter_listbox = tk.Listbox(
            left_frame,
            selectmode=tk.MULTIPLE,
            height=7,
            exportselection=False
        )
        self.filter_listbox.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for col in numeric_columns:
            self.filter_listbox.insert(tk.END, col)

        # If user previously applied a filter, keep the selected columns.
        previous_filter_cols = getattr(self, "selected_filter_cols", default_filter_cols)

        for i, col in enumerate(numeric_columns):
            if col in previous_filter_cols:
                self.filter_listbox.selection_set(i)

        method_row = ttk.Frame(left_frame)
        method_row.pack(fill=tk.X, padx=5, pady=(6, 3))
        ttk.Label(
            method_row,
            text="Filter Method:",
            font=("Arial", 10, "bold")
        ).pack(side=tk.LEFT, padx=(0, 6))

        ttk.Combobox(
            method_row,
            textvariable=self.filter_method_var,
            state="readonly",
            values=[
                "No filter",
                "Moving average",
                "Exponential moving average",
                "Median filter",
                "Simple Kalman filter"
            ]
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        parameter_frame = ttk.Frame(left_frame)
        parameter_frame.pack(fill=tk.X, padx=5, pady=3)
        parameter_frame.columnconfigure(1, weight=1)
        parameter_fields = [
            ("Moving Average Window:", self.moving_window_var),
            ("EMA Span:", self.ema_span_var),
            ("Median Window:", self.median_window_var),
            ("Kalman Q — Process Noise:", self.kalman_q_var),
            ("Kalman R — Measurement Noise:", self.kalman_r_var),
        ]
        for row_index, (field_label, field_variable) in enumerate(
            parameter_fields
        ):
            ttk.Label(
                parameter_frame,
                text=field_label,
            ).grid(
                row=row_index,
                column=0,
                sticky="w",
                padx=(0, 6),
                pady=2,
            )
            ttk.Entry(
                parameter_frame,
                textvariable=field_variable,
            ).grid(
                row=row_index,
                column=1,
                sticky="ew",
                pady=2,
            )

        quick_select_row = ttk.Frame(left_frame)
        quick_select_row.pack(fill=tk.X, padx=5, pady=(8, 2))

        ttk.Button(
            quick_select_row,
            text="Select Ia/Ib/Ic",
            command=self.select_common_current_columns
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        ttk.Button(
            quick_select_row,
            text="Select Signal Columns",
            command=self.select_all_signal_filter_columns
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        ttk.Button(
            quick_select_row,
            text="Clear Selection",
            command=self.clear_filter_columns
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        button_row = ttk.Frame(left_frame)
        button_row.pack(fill=tk.X, padx=5, pady=10)

        ttk.Button(
            button_row,
            text="Preview Filter",
            command=self.preview_filter_result
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        ttk.Button(
            button_row,
            text="Use No Filter",
            command=self.set_no_filter
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        ttk.Button(
            button_row,
            text="Plot Signals",
            command=self.plot_training_filter_result,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        training_preview_notebook = ttk.Notebook(right_frame)
        training_preview_notebook.pack(
            fill=tk.BOTH,
            expand=True,
            padx=5,
            pady=5,
        )
        training_data_tab = ttk.Frame(training_preview_notebook)
        training_plot_tab = ttk.Frame(training_preview_notebook)
        training_preview_notebook.add(
            training_data_tab,
            text="Data Preview",
        )
        training_preview_notebook.add(
            training_plot_tab,
            text="Signal Plot",
        )
        self.training_filter_preview_notebook = training_preview_notebook
        self.filter_preview_text = ScrolledText(
            training_data_tab,
            wrap=tk.WORD,
        )
        self.filter_preview_text.pack(
            fill=tk.BOTH,
            expand=True,
            padx=3,
            pady=3,
        )
        self.training_filter_plot_host = ttk.Frame(training_plot_tab)
        self.training_filter_plot_host.pack(
            fill=tk.BOTH,
            expand=True,
            padx=3,
            pady=3,
        )
        self.training_filter_plot_placeholder = ttk.Label(
            self.training_filter_plot_host,
            text=(
                "Select signal columns and choose Preview Filter.\n"
                "The before/after plot will be generated automatically."
            ),
            foreground="#666666",
            justify=tk.CENTER,
        )
        self.training_filter_plot_placeholder.pack(
            expand=True, padx=20, pady=20
        )

        self.preview_filter_result()

    def set_no_filter(self):
        self.filter_method_var.set("No filter")

        if hasattr(self, "filter_listbox"):
            self.filter_listbox.selection_clear(0, tk.END)

        self.preview_filter_result()

    def clear_filter_columns(self):
        if hasattr(self, "filter_listbox"):
            self.filter_listbox.selection_clear(0, tk.END)

        self.preview_filter_result()

    def select_common_current_columns(self):
        """
        Quick selection for common motor-current columns.
        Only these selected columns will be filtered. Other columns remain unchanged.
        """
        if not hasattr(self, "filter_listbox") or self.raw_df is None:
            return

        common_names = {"ia", "ib", "ic", "i_a", "i_b", "i_c", "u", "v", "w", "u_current", "v_current", "w_current"}

        self.filter_listbox.selection_clear(0, tk.END)

        numeric_columns = self.raw_df.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns.tolist()

        for i, col in enumerate(numeric_columns):
            if col.lower() in common_names:
                self.filter_listbox.selection_set(i)

        self.preview_filter_result()

    def select_all_signal_filter_columns(self):
        """
        Select numeric signal columns but skip ID/label/target/class columns.
        Useful when the user wants to filter most sensor signals.
        """
        if not hasattr(self, "filter_listbox") or self.raw_df is None:
            return

        self.filter_listbox.selection_clear(0, tk.END)

        numeric_columns = self.raw_df.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns.tolist()

        for i, col in enumerate(numeric_columns):
            col_lower = col.lower()

            if (
                "id" in col_lower
                or "label" in col_lower
                or "target" in col_lower
                or "class" in col_lower
            ):
                continue

            self.filter_listbox.selection_set(i)

        self.preview_filter_result()

    def get_selected_filter_columns(self):
        if not hasattr(self, "filter_listbox"):
            return list(self.selected_filter_cols)
        numeric_columns = self.raw_df.select_dtypes(
            include=["int64", "float64", "int32", "float32"]
        ).columns.tolist()

        selected_indices = self.filter_listbox.curselection()

        selected_cols = [
            numeric_columns[i]
            for i in selected_indices
        ]

        return selected_cols

    def preview_filter_result(self):
        try:
            selected_filter_cols = self.get_selected_filter_columns()
            filter_method = self.filter_method_var.get()

            moving_window = int(self.moving_window_var.get())
            ema_span = int(self.ema_span_var.get())
            median_window = int(self.median_window_var.get())
            kalman_q = float(self.kalman_q_var.get())
            kalman_r = float(self.kalman_r_var.get())

            temp_filtered_df = apply_selected_filter(
                df=self.raw_df,
                selected_filter_cols=selected_filter_cols,
                filter_method=filter_method,
                moving_window=moving_window,
                ema_span=ema_span,
                median_window=median_window,
                kalman_q=kalman_q,
                kalman_r=kalman_r
            )

            self.filter_preview_text.delete("1.0", tk.END)

            self.filter_preview_text.insert(tk.END, "Filter preview generated.\n\n")
            self.filter_preview_text.insert(tk.END, f"Filter method: {filter_method}\n")
            self.filter_preview_text.insert(tk.END, f"Filtered columns only: {selected_filter_cols}\n")
            unfiltered_cols = [col for col in self.raw_df.columns if col not in selected_filter_cols]
            self.filter_preview_text.insert(tk.END, f"Unfiltered columns remain unchanged: {unfiltered_cols}\n\n")

            self.filter_preview_text.insert(tk.END, "Original data preview:\n")
            self.filter_preview_text.insert(tk.END, self.raw_df.head(8).to_string())

            self.filter_preview_text.insert(tk.END, "\n\nFiltered data preview:\n")
            self.filter_preview_text.insert(tk.END, temp_filtered_df.head(8).to_string())

            if selected_filter_cols and filter_method != "No filter":
                compare_cols = selected_filter_cols[:4]
                self.filter_preview_text.insert(tk.END, "\n\nBefore/after difference preview:\n")

                for col in compare_cols:
                    before = pd.to_numeric(self.raw_df[col], errors="coerce").head(8).values
                    after = pd.to_numeric(temp_filtered_df[col], errors="coerce").head(8).values
                    self.filter_preview_text.insert(tk.END, f"\nColumn: {col}\n")
                    self.filter_preview_text.insert(tk.END, f"Original: {np.round(before, 6)}\n")
                    self.filter_preview_text.insert(tk.END, f"Filtered: {np.round(after, 6)}\n")

            if selected_filter_cols:
                self.after_idle(self.plot_training_filter_result)

        except Exception as e:
            if hasattr(self, "filter_preview_text"):
                self.filter_preview_text.delete("1.0", tk.END)
                self.filter_preview_text.insert(
                    tk.END,
                    "Filter preview failed.\n\n" + str(e),
                )
            messagebox.showerror("Filter Preview Error", str(e))

    def plot_training_filter_result(self):
        """Embed a before/after plot in the signal-training filter page."""
        try:
            selected_columns = self.get_selected_filter_columns()
            if not selected_columns:
                raise ValueError(
                    "Select at least one numeric signal column to plot."
                )
            result = apply_selected_filter(
                df=self.raw_df,
                selected_filter_cols=selected_columns,
                filter_method=self.filter_method_var.get(),
                moving_window=int(self.moving_window_var.get()),
                ema_span=int(self.ema_span_var.get()),
                median_window=int(self.median_window_var.get()),
                kalman_q=float(self.kalman_q_var.get()),
                kalman_r=float(self.kalman_r_var.get()),
            )
            if not hasattr(self, "training_filter_plot_host"):
                raise ValueError("Reopen the filter page and try again.")
            if vars(self).get("training_filter_plot_figure") is not None:
                plt.close(self.training_filter_plot_figure)
            for child in self.training_filter_plot_host.winfo_children():
                child.destroy()

            row_count = len(result)
            max_points = 5000
            step = max(1, int(np.ceil(row_count / max_points)))
            positions = np.arange(0, row_count, step)
            columns_to_plot = selected_columns[:4]
            figure, axes = plt.subplots(
                len(columns_to_plot),
                1,
                figsize=(7.3, max(3.4, 2.45 * len(columns_to_plot))),
                squeeze=False,
                sharex=True,
            )
            for row_index, column in enumerate(columns_to_plot):
                axis = axes[row_index][0]
                before = pd.to_numeric(
                    self.raw_df.iloc[positions][column],
                    errors="coerce",
                ).to_numpy(dtype=float)
                after = pd.to_numeric(
                    result.iloc[positions][column],
                    errors="coerce",
                ).to_numpy(dtype=float)
                axis.plot(
                    positions,
                    before,
                    label="Before",
                    color="#777777",
                    alpha=0.72,
                    linewidth=0.9,
                )
                axis.plot(
                    positions,
                    after,
                    label="After",
                    color="#1769aa",
                    linewidth=1.15,
                )
                axis.set_title(str(column), loc="left", fontsize=10)
                axis.set_ylabel("Amplitude")
                axis.legend(loc="upper right")
                axis.grid(alpha=0.25)
            axes[-1][0].set_xlabel("Sample index")
            figure.suptitle(
                (
                    "Built-in filter comparison"
                    + (
                        f" (displaying every {step}th point)"
                        if step > 1
                        else ""
                    )
                ),
                fontsize=11,
            )
            figure.tight_layout()
            canvas = FigureCanvasTkAgg(
                figure,
                master=self.training_filter_plot_host,
            )
            canvas.draw()
            toolbar = NavigationToolbar2Tk(
                canvas,
                self.training_filter_plot_host,
                pack_toolbar=False,
            )
            toolbar.update()
            toolbar.pack(side=tk.TOP, fill=tk.X)
            canvas.get_tk_widget().pack(
                side=tk.TOP,
                fill=tk.BOTH,
                expand=True,
            )
            self.training_filter_plot_canvas = canvas
            self.training_filter_plot_figure = figure
            self.training_filter_plot_toolbar = toolbar
            self.training_filter_preview_notebook.select(1)
            self.training_filter_plot_host.update_idletasks()
            canvas.draw_idle()
            self.after_idle(
                lambda current_canvas=canvas: self.redraw_embedded_canvas(
                    current_canvas
                )
            )
        except Exception as exc:
            self.show_embedded_plot_message(
                "training_filter_plot_host",
                "Signal plot could not be rendered:\n" + str(exc),
            )
            messagebox.showerror("Filter Plot Error", str(exc))

    def refresh_recommended_columns_after_data_change(self):
        """
        Keep an applied analysis recommendation when filtering changes values.
        Fall back to numeric columns only when the prior selection is no longer
        usable.
        """
        if self.df is None:
            return

        available_columns = set(self.df.columns)
        task_type = self.task_type_var.get()

        if task_type == TASK_AUTOENCODER:
            self.target_cols = []
            self.label_col = ""
            self.label_var.set("")
        else:
            self.target_cols = [
                column
                for column in self.target_cols
                if column in available_columns
            ]
            if not self.target_cols:
                selected_label = self.label_var.get()
                if selected_label in available_columns:
                    self.target_cols = [selected_label]
            self.label_col = (
                self.target_cols[0] if self.target_cols else ""
            )
            self.label_var.set(self.label_col)

        target_set = set(self.target_cols)
        preserved_features = [
            column
            for column in self.feature_cols
            if column in available_columns
            and column not in target_set
            and pd.api.types.is_numeric_dtype(self.df[column])
        ]
        if preserved_features:
            self.feature_cols = preserved_features
        else:
            self.feature_cols = [
                column
                for column in self.df.select_dtypes(
                    include=["number"]
                ).columns
                if column not in target_set
                and "id" not in str(column).lower()
            ]

        if task_type == TASK_CLASSIFICATION and self.target_cols:
            output_units = self.df[
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

    def validate_step_2_filter(self):
        if self.raw_df is None:
            raise ValueError("Please load a dataset first.")
        if self.data_mode_var.get() == DATA_MODE_IMAGE:
            self.filter_method_var.set("No filter")
            self.selected_filter_cols = []
            self.filtered_df = self.raw_df.copy()
            self.df = self.filtered_df.copy()
            return

        selected_filter_cols = self.get_selected_filter_columns()
        filter_method = self.filter_method_var.get()

        moving_window = int(self.moving_window_var.get())
        ema_span = int(self.ema_span_var.get())
        median_window = int(self.median_window_var.get())
        kalman_q = float(self.kalman_q_var.get())
        kalman_r = float(self.kalman_r_var.get())

        if moving_window <= 0:
            raise ValueError("Moving average window must be larger than 0.")

        if ema_span <= 0:
            raise ValueError("EMA span must be larger than 0.")

        if median_window <= 0:
            raise ValueError("Median window must be larger than 0.")

        if kalman_q <= 0:
            raise ValueError("Kalman Q must be larger than 0.")

        if kalman_r <= 0:
            raise ValueError("Kalman R must be larger than 0.")

        self.selected_filter_cols = selected_filter_cols

        self.filtered_df = apply_selected_filter(
            df=self.raw_df,
            selected_filter_cols=selected_filter_cols,
            filter_method=filter_method,
            moving_window=moving_window,
            ema_span=ema_span,
            median_window=median_window,
            kalman_q=kalman_q,
            kalman_r=kalman_r
        )

        # From this point onward, all following steps use filtered data.
        self.df = self.filtered_df.copy()
        self.refresh_recommended_columns_after_data_change()

        messagebox.showinfo(
            "Filter Applied",
            f"Filter applied successfully.\n\nMethod: {filter_method}\nFiltered columns only: {selected_filter_cols}\nUnselected columns were kept unchanged."
        )

        self._pending_preparation_steps.discard(2)
        if self.custom_filter_enabled_var.get():
            self._pending_preparation_steps.add(3)
