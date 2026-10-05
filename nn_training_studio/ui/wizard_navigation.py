"""Ui / wizard navigation for NN Training Studio."""

from tkinter import messagebox
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk


class NavigationMixin:
    """Navigation behavior for the main application."""

    def clear_content(self):
        self._capture_page_state()
        for figure_name in (
            "filter_tool_plot_figure",
            "training_filter_plot_figure",
            "custom_filter_plot_figure",
        ):
            figure = vars(self).get(figure_name)
            if figure is not None:
                plt.close(figure)
                delattr(self, figure_name)
        for widget in self.content_frame.winfo_children():
            widget.destroy()
        try:
            self.content_canvas.yview_moveto(0.0)
            self.content_canvas.xview_moveto(0.0)
            self.content_canvas.configure(scrollregion=(0, 0, 1, 1))
        except tk.TclError:
            pass

        # Tk keeps the Python attribute after its underlying widget command is
        # destroyed. Remove stale content-page references so later validation
        # uses the saved specification/code instead of calling .get() on a dead
        # ScrolledText widget.
        for attribute_name, value in list(vars(self).items()):
            if not isinstance(value, tk.Misc):
                continue
            try:
                widget_exists = bool(value.winfo_exists())
            except tk.TclError:
                widget_exists = False
            if not widget_exists:
                delattr(self, attribute_name)

    @staticmethod
    def redraw_embedded_canvas(canvas):
        """Complete a deferred Tk canvas draw if its widget still exists."""
        try:
            if canvas.get_tk_widget().winfo_exists():
                canvas.draw()
        except (tk.TclError, RuntimeError):
            pass

    def show_embedded_plot_message(
        self,
        host_attribute,
        message,
        foreground="#b42318",
    ):
        """Keep a plot tab informative when rendering cannot complete."""
        host = getattr(self, host_attribute, None)
        if host is None:
            return
        try:
            if not host.winfo_exists():
                return
            for child in host.winfo_children():
                child.destroy()
            ttk.Label(
                host,
                text=str(message),
                foreground=foreground,
                justify=tk.CENTER,
                wraplength=620,
            ).pack(expand=True, padx=20, pady=20)
        except tk.TclError:
            pass

    def _show_advanced_step(self):
        self.current_view = "wizard"
        self.clear_content()
        self.set_workflow_navigation_visible(True)
        self.home_button.config(state=tk.NORMAL)
        self.settings_button.config(state=tk.NORMAL)

        self.title_label.config(text=self.step_titles[self.current_step])
        self.progress_label.config(
            text=f"Page {self.current_step + 1} of {len(self.step_titles)}"
        )
        self.workflow_hint_label.config(text=self.get_workflow_hint())

        if self.current_step == 0:
            self.build_step_1()
        elif self.current_step == 1:
            self.build_step_ai_analysis()
        elif self.current_step == 2:
            self.build_step_2_filter()
        elif self.current_step == 3:
            self.build_step_3_custom_filter()
        elif self.current_step == 4:
            self.build_step_2()
        elif self.current_step == 5:
            self.build_step_3()
        elif self.current_step == 6:
            self.build_step_4()
        elif self.current_step == 7:
            self.build_step_5()

        if self.training_running or self.ai_request_running:
            self.back_button.config(state=tk.DISABLED)
            self.next_button.config(state=tk.DISABLED)
            return

        if self.current_step > 0:
            self.back_button.config(state=tk.NORMAL)
        else:
            self.back_button.config(state=tk.DISABLED)

        if self.current_step == len(self.step_titles) - 1:
            self.next_button.config(state=tk.DISABLED, text="Finish")
        elif self.current_step == 6:
            self.next_button.config(state=tk.NORMAL, text="Go to Training →")
        else:
            self.next_button.config(state=tk.NORMAL, text="Next →")
        self.update_provider_status_display()
        # Some page builders create large notebooks or plots. Reasserting the
        # footer after idle layout prevents those requested sizes from
        # displacing the page navigation on compact displays.
        self.after_idle(self.keep_workflow_navigation_visible)

    def _advanced_go_back(self):
        if self.training_running or self.ai_request_running:
            return

        if self.current_step > 0:
            self.current_step -= 1
            self.show_step()

    def _advanced_go_next(self):
        if self.ai_request_running:
            return
        try:
            if self.current_step == 0:
                self.validate_step_1()
            elif self.current_step == 1:
                self.validate_step_ai_analysis()
            elif self.current_step == 2:
                self.validate_step_2_filter()
            elif self.current_step == 3:
                self.validate_step_3_custom_filter()
            elif self.current_step == 4:
                self.validate_step_2()
            elif self.current_step == 5:
                self.validate_step_3()
            elif self.current_step == 6:
                self.validate_step_4()

            self.current_step += 1
            self.show_step()

        except Exception as e:
            messagebox.showerror("Check Required", str(e))
