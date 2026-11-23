"""Background file import with main-thread delivery and close-safe polling."""
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from nn_training_studio.simulink_data import load_simulink_files


class SimulinkImportDialog(tk.Toplevel):
    def __init__(self, parent, paths, on_success, *, loader=load_simulink_files,
                 title="Import MATLAB / Simulink", description="Reading signals and checking time, columns and sample rates..."):
        super().__init__(parent)
        self.title(title)
        self._title = title
        self.transient(parent)
        self.resizable(False, False)
        self._closed = False
        self._poll_id = None
        self._results = queue.Queue(maxsize=1)
        self._on_success = on_success
        ttk.Label(self, text=description,
                  padding=18, wraplength=420).pack(fill="x")
        progress = ttk.Progressbar(self, mode="indeterminate", length=400)
        progress.pack(padx=18, pady=(0, 12))
        progress.start(15)
        ttk.Button(self, text="Cancel import", command=self.close).pack(pady=(0, 12))
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Destroy>", self._destroyed, add="+")
        self.grab_set()
        # The worker only owns immutable paths and its queue, never Tk objects.
        results = self._results
        def worker():
            try:
                results.put((True, loader(paths)))
            except Exception as exc:
                results.put((False, str(exc)))
        threading.Thread(target=worker, daemon=True, name="Signal file import").start()
        self._poll_id = self.after(75, self._poll)

    def _poll(self):
        self._poll_id = None
        if self._closed:
            return
        try:
            success, payload = self._results.get_nowait()
        except queue.Empty:
            self._poll_id = self.after(75, self._poll)
            return
        parent, callback = self.master, self._on_success
        self.close()
        if success:
            try:
                callback(payload)
            except Exception as exc:
                messagebox.showerror(self._title, str(exc), parent=parent)
        else:
            messagebox.showerror(self._title, payload, parent=parent)

    def _destroyed(self, event):
        if event.widget is self:
            self._closed = True
            if self._poll_id is not None:
                self.after_cancel(self._poll_id)
                self._poll_id = None

    def close(self):
        if not self._closed:
            self.grab_release()
            self.destroy()
