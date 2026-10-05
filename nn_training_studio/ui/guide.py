"""Ui / guide for NN Training Studio."""

from tkinter.scrolledtext import ScrolledText
from tkinter import messagebox
import threading
import tkinter as tk
from tkinter import ttk
from nn_training_studio.constants import (
    APP_VERSION,
    GUIDE_ACTION_DEPLOY,
    GUIDE_ACTION_LABELS,
    GUIDE_ACTION_NONE,
    OFFICIAL_PROVIDER_SETTINGS,
)
from nn_training_studio.studio_guide import (
    _offline_studio_guide_answer,
    request_ai_studio_guide,
)


class StudioGuideWindow(tk.Toplevel):
    """Small task-selection assistant with offline and optional AI modes."""

    QUICK_QUESTIONS = [
        ("Signal / values", "I have CSV signal data. Which task should I choose?"),
        ("Image classes", "I have images arranged by class folders."),
        ("Object detection", "I have YOLO bounding-box annotations."),
        ("Use saved model", "I want to load a model and check new results."),
        ("Filter data", "I only want to filter and export my signals."),
        ("Implement model / API", "Analyse my existing code and implement my trained model or AI provider API for me."),
        ("Show workflow", "Show me the main NN Studio workflow."),
    ]

    def __init__(self, parent, initial_question=None):
        super().__init__(parent)
        self.parent_app = parent
        self.title(f"NN Studio Guide — {APP_VERSION}")
        self.geometry("720x760")
        self.minsize(560, 560)
        self.protocol("WM_DELETE_WINDOW", self.close_window)
        self.request_running = False
        self.last_failed_question = None
        self.deployment_handoff_question = None
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        heading = ttk.Frame(self)
        heading.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 6))
        ttk.Label(
            heading,
            text="NN Studio Guide",
            font=("Arial", 16, "bold"),
        ).pack(anchor="w")
        self.mode_label = ttk.Label(
            heading,
            text="",
            foreground="#245a85",
        )
        self.mode_label.pack(anchor="w", pady=(3, 0))

        quick_frame = ttk.LabelFrame(self, text="Quick choices")
        quick_frame.grid(row=1, column=0, sticky="ew", padx=14, pady=6)
        for index, (label, prompt) in enumerate(self.QUICK_QUESTIONS):
            row, column = divmod(index, 3)
            quick_frame.columnconfigure(column, weight=1)
            ttk.Button(
                quick_frame,
                text=label,
                command=lambda value=prompt: self.ask_question(value),
            ).grid(
                row=row,
                column=column,
                sticky="ew",
                padx=5,
                pady=5,
            )

        self.chat_text = ScrolledText(
            self,
            wrap=tk.WORD,
            height=22,
            state=tk.DISABLED,
            font=("Arial", 10),
        )
        self.chat_text.grid(
            row=2,
            column=0,
            sticky="nsew",
            padx=14,
            pady=6,
        )
        self.chat_text.tag_configure(
            "guide",
            foreground="#184f78",
            spacing1=4,
            spacing3=8,
        )
        self.chat_text.tag_configure(
            "user",
            foreground="#333333",
            spacing1=4,
            spacing3=8,
        )

        self.action_frame = ttk.Frame(self)
        self.action_frame.grid(row=3, column=0, sticky="ew", padx=14)
        self.action_frame.columnconfigure(0, weight=1)
        self.action_button = ttk.Button(
            self.action_frame,
            text="",
            command=self.run_recommended_action,
        )
        self.recommended_action = GUIDE_ACTION_NONE

        input_frame = ttk.LabelFrame(self, text="Ask a question")
        input_frame.grid(
            row=4,
            column=0,
            sticky="ew",
            padx=14,
            pady=(6, 5),
        )
        input_frame.columnconfigure(0, weight=1)
        self.question_text = tk.Text(
            input_frame,
            height=3,
            wrap=tk.WORD,
        )
        self.question_text.grid(
            row=0,
            column=0,
            columnspan=3,
            sticky="ew",
            padx=8,
            pady=(8, 5),
        )
        self.question_text.bind(
            "<Return>",
            self._send_from_key,
        )
        self.question_text.bind(
            "<Shift-Return>",
            self._insert_newline,
        )
        self.question_text.bind(
            "<Control-Return>",
            self._send_from_key,
        )
        self.use_ai_check = ttk.Checkbutton(
            input_frame,
            text="Use connected AI for this question",
            variable=self.parent_app.guide_use_ai_var,
            command=self.refresh_mode_label,
        )
        self.use_ai_check.grid(
            row=1,
            column=0,
            sticky="w",
            padx=8,
            pady=(0, 8),
        )
        self.send_button = ttk.Button(
            input_frame,
            text="Send  (Enter)",
            command=self.send_current_question,
        )
        self.send_button.grid(
            row=1,
            column=1,
            padx=5,
            pady=(0, 8),
        )
        self.clear_button = ttk.Button(
            input_frame,
            text="Clear",
            command=self.clear_conversation,
        )
        self.clear_button.grid(
            row=1,
            column=2,
            padx=(5, 8),
            pady=(0, 8),
        )

        self.composer_status_var = tk.StringVar(
            value="Ready — Enter sends; Shift+Enter adds a new line."
        )
        self.composer_status_label = ttk.Label(
            input_frame,
            textvariable=self.composer_status_var,
            foreground="#486070",
        )
        self.composer_status_label.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky="w",
            padx=8,
            pady=(0, 7),
        )
        self.retry_button = ttk.Button(
            input_frame,
            text="Retry",
            command=self.retry_last_question,
        )
        self.retry_button.grid(
            row=2,
            column=2,
            sticky="e",
            padx=(5, 8),
            pady=(0, 7),
        )
        self.retry_button.grid_remove()

        self.privacy_label = ttk.Label(
            self,
            text=(
                "Privacy: AI mode sends only your typed question, the current "
                "page name, and NN Studio's fixed capability list. Import source "
                "files through Deploy & Integrate; do not paste secrets here."
            ),
            wraplength=570,
            foreground="#666666",
        )
        self.privacy_label.grid(
            row=5,
            column=0,
            sticky="ew",
            padx=16,
            pady=(0, 12),
        )

        self.refresh_mode_label()
        if self.parent_app.guide_history:
            for role, message in self.parent_app.guide_history:
                self.append_message(role, message, record=False)
        else:
            self.append_message(
                "guide",
                "Hello. Ask directly in plain language. You can describe your "
                "data, paste an error, or say that you want a trained model or "
                "AI provider API added to an existing application. I will plan "
                "the next steps and open the correct workspace. Press Enter to send.",
            )
        if initial_question:
            self.after(120, lambda: self.ask_question(initial_question))
        else:
            self.after_idle(self.question_text.focus_set)
        self.bind("<Configure>", self._resize_guide_text)

    def _send_from_key(self, _event=None):
        self.send_current_question()
        return "break"

    @staticmethod
    def _insert_newline(_event=None):
        return None

    def _resize_guide_text(self, _event=None):
        usable = max(self.winfo_width() - 50, 260)
        self.privacy_label.config(wraplength=usable)

    def retry_last_question(self):
        if self.request_running or not self.last_failed_question:
            return
        question = self.last_failed_question
        self.last_failed_question = None
        self.retry_button.grid_remove()
        self.question_text.delete("1.0", tk.END)
        self.question_text.insert("1.0", question)
        self.send_current_question()

    def refresh_mode_label(self):
        provider = self.parent_app.ai_provider_var.get()
        if self.parent_app.guide_use_ai_var.get():
            if provider in OFFICIAL_PROVIDER_SETTINGS:
                text = f"Mode: AI Guide ({provider})"
            else:
                text = "Mode: AI requested — connect a provider in Settings"
        else:
            text = "Mode: Offline Guide — no API key required"
        self.mode_label.config(text=text)

    def append_message(self, role, message, record=True):
        label = "You" if role == "user" else "Guide"
        tag_name = "user" if role == "user" else "guide"
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.insert(tk.END, f"{label}:\n{message.strip()}\n\n", tag_name)
        self.chat_text.see(tk.END)
        self.chat_text.config(state=tk.DISABLED)
        if record:
            self.parent_app.guide_history.append((role, message.strip()))
            self.parent_app.guide_history[:] = (
                self.parent_app.guide_history[-30:]
            )

    def clear_conversation(self):
        if self.request_running:
            return
        self.parent_app.guide_history.clear()
        self.chat_text.config(state=tk.NORMAL)
        self.chat_text.delete("1.0", tk.END)
        self.chat_text.config(state=tk.DISABLED)
        self.action_button.grid_remove()
        self.retry_button.grid_remove()
        self.last_failed_question = None
        self.deployment_handoff_question = None
        self.composer_status_var.set(
            "Ready — Enter sends; Shift+Enter adds a new line."
        )
        self.recommended_action = GUIDE_ACTION_NONE
        self.append_message(
            "guide",
            "Conversation cleared. What data do you have, and what should the "
            "model produce?",
        )

    def ask_question(self, question):
        if self.request_running:
            return
        self.question_text.delete("1.0", tk.END)
        self.question_text.insert("1.0", question)
        self.send_current_question()

    def send_current_question(self):
        if self.request_running:
            return
        question = self.question_text.get("1.0", tk.END).strip()
        if not question:
            messagebox.showinfo(
                "NN Studio Guide",
                "Enter a question or choose one of the quick choices.",
                parent=self,
            )
            return
        self.question_text.delete("1.0", tk.END)
        self.append_message("user", question)
        self.deployment_handoff_question = question[:2000]
        self.action_button.grid_remove()
        self.retry_button.grid_remove()
        self.recommended_action = GUIDE_ACTION_NONE
        context = self.parent_app.get_studio_guide_context()

        code_markers = (
            "\ndef ", "\nclass ", "\nimport ", "#include", "<html",
            "function ", "public static", "using namespace", "const ",
        )
        looks_like_code = (
            len(question) > 3500
            or (
                question.count("\n") >= 6
                and any(marker in "\n" + question.lower() for marker in code_markers)
            )
        )
        if looks_like_code:
            self.deployment_handoff_question = (
                "Analyse my existing application source and help me implement "
                "the required trained model or AI provider API. I will select "
                "the approved code files in Step 1."
            )
            self.show_answer({
                "answer": (
                    "This looks like application source code. For privacy and a "
                    "more accurate result, do not send the full code through the "
                    "Guide. Open Deploy & Integrate Step 1, select the project or "
                    "specific files, and describe the implementation goal there. "
                    "NN Studio will scan locally, block likely secrets, and let "
                    "you approve exactly which files may be analysed."
                ),
                "checklist": [
                    "Open Deploy & Integrate.",
                    "Select the project folder or individual source files.",
                    "Review the files marked Share and describe the desired result.",
                    "Choose a local model, provider API, or both in Step 2.",
                ],
                "recommended_action": GUIDE_ACTION_DEPLOY,
                "mode": "Local privacy check",
            })
            self.composer_status_var.set(
                "Code detected — use the protected project-import workflow."
            )
            return

        if not self.parent_app.guide_use_ai_var.get():
            self.show_answer(
                _offline_studio_guide_answer(question, context)
            )
            self.composer_status_var.set(
                "Answered by Offline Guide — ready for another question."
            )
            self.question_text.focus_set()
            return

        self.request_running = True
        self.send_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)
        self.composer_status_var.set(
            "Sending to the connected provider… please wait."
        )
        self.append_message(
            "guide",
            "Preparing a private navigation answer from the connected provider...",
        )
        threading.Thread(
            target=self.ai_request_worker,
            args=(question, context),
            daemon=True,
        ).start()

    def ai_request_worker(self, question, context):
        try:
            answer = request_ai_studio_guide(
                self.parent_app,
                question,
                context,
            )
            self.parent_app.ui_queue.put(
                (
                    "studio_guide_ai_finished",
                    {"window": self, "answer": answer},
                )
            )
        except Exception as exc:
            self.parent_app.ui_queue.put(
                (
                    "studio_guide_ai_failed",
                    {
                        "window": self,
                        "error": str(exc),
                        "fallback": _offline_studio_guide_answer(
                            question,
                            context,
                        ),
                    },
                )
            )

    def finish_ai_request(
        self,
        answer,
        error_message,
        fallback_answer=None,
    ):
        if not self.winfo_exists():
            return
        self.request_running = False
        self.send_button.config(state=tk.NORMAL)
        self.clear_button.config(state=tk.NORMAL)
        if error_message:
            self.last_failed_question = next(
                (
                    message
                    for role, message in reversed(self.parent_app.guide_history)
                    if role == "user"
                ),
                None,
            )
            self.retry_button.grid()
            self.composer_status_var.set(
                "AI request failed — offline guidance is shown; Retry is available."
            )
            self.append_message(
                "guide",
                "AI guidance was unavailable: "
                + error_message
                + "\n\nI will use the offline guide for navigation.",
            )
            if fallback_answer:
                self.show_answer(fallback_answer)
            return
        self.last_failed_question = None
        self.retry_button.grid_remove()
        self.composer_status_var.set(
            "AI response received — ready for another question."
        )
        self.show_answer(answer)
        self.question_text.focus_set()

    def show_answer(self, answer):
        message = answer["answer"].strip()
        checklist = answer.get("checklist") or []
        if checklist:
            message += "\n\nNext steps:\n" + "\n".join(
                f"{index}. {item}"
                for index, item in enumerate(checklist, start=1)
            )
        message += f"\n\nResponse mode: {answer.get('mode', 'Guide')}"
        self.append_message("guide", message)
        self.recommended_action = answer.get(
            "recommended_action",
            GUIDE_ACTION_NONE,
        )
        label = GUIDE_ACTION_LABELS.get(self.recommended_action, "")
        if label:
            self.action_button.config(text=label)
            self.action_button.grid(row=0, column=0, sticky="ew", pady=(2, 2))
        else:
            self.action_button.grid_remove()

    def run_recommended_action(self):
        action = self.recommended_action
        if action == GUIDE_ACTION_NONE:
            return
        if action == GUIDE_ACTION_DEPLOY:
            latest_question = self.deployment_handoff_question or next(
                (
                    message
                    for role, message in reversed(self.parent_app.guide_history)
                    if role == "user"
                ),
                None,
            )
            self.parent_app.open_deploy_integrate(initial_goal=latest_question)
        else:
            self.parent_app.run_studio_guide_action(action)
        if self.winfo_exists():
            self.withdraw()

    def close_window(self):
        self.parent_app.studio_guide_window = None
        self.destroy()
