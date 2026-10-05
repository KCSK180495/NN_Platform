"""Scrollable form container that keeps controls reachable on small displays."""

import tkinter as tk
from tkinter import ttk


class ScrollableForm(ttk.Frame):
    """Scroll natural-size form content without clipping wide controls."""

    def __init__(self, parent):
        super().__init__(parent)
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, highlightthickness=0, width=1, height=1)
        vertical = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        horizontal = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        self.content = ttk.Frame(self.canvas)
        self.window = self.canvas.create_window(0, 0, window=self.content, anchor="nw")
        self.content.bind("<Configure>", self._resize)
        self.canvas.bind("<Configure>", self._resize)
        self._top = self.winfo_toplevel()
        self._bindings = [(event, self._top.bind(event, self._wheel, add="+"))
                          for event in ("<MouseWheel>", "<Button-4>", "<Button-5>")]
        self.bind("<Destroy>", self._cleanup, add="+")

    def _resize(self, _event=None):
        width = max(self.canvas.winfo_width(), self.content.winfo_reqwidth())
        self.canvas.itemconfigure(self.window, width=width)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _wheel(self, event):
        widget = event.widget
        if isinstance(widget, (tk.Text, tk.Listbox, ttk.Treeview, ttk.Combobox)):
            return
        while widget is not None and widget is not self:
            widget = getattr(widget, "master", None)
        if widget is None:
            return
        if getattr(event, "num", None) in (4, 5):
            units = -1 if event.num == 4 else 1
        else:
            delta = event.delta
            units = (-1 if delta > 0 else 1) * max(1, abs(int(delta / 120)))
        self.canvas.yview_scroll(units, "units")
        return "break"

    def _cleanup(self, event):
        if event.widget is self:
            for sequence, binding in self._bindings:
                self._top.unbind(sequence, binding)
