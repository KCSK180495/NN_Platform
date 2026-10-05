"""Ui / wizard ai settings for NN Training Studio."""

import json
try:
    import keyring
except Exception:
    keyring = None
from tkinter import messagebox
import os
import tkinter as tk
from nn_training_studio.ai_transport import (
    _masked_api_key,
    _validate_provider_base_url,
)
from nn_training_studio.constants import (
    AI_PROVIDER_LOCAL,
    AI_PROVIDER_TYPES,
    KEYRING_SERVICE_NAME,
    OFFICIAL_PROVIDER_SETTINGS,
)


class AISettingsMixin:
    """AISettings behavior for the main application."""

    def load_ai_settings(self):
        """Load non-secret preferences and an optional OS-stored credential."""
        settings = {}
        try:
            if self.ai_settings_path.exists():
                with open(
                    self.ai_settings_path,
                    "r",
                    encoding="utf-8",
                ) as settings_file:
                    loaded = json.load(settings_file)
                if isinstance(loaded, dict):
                    settings = loaded
        except Exception:
            settings = {}

        provider_name = settings.get("provider", AI_PROVIDER_LOCAL)
        if provider_name not in AI_PROVIDER_TYPES:
            provider_name = AI_PROVIDER_LOCAL
        self.ai_provider_var.set(provider_name)

        if provider_name in OFFICIAL_PROVIDER_SETTINGS:
            defaults = OFFICIAL_PROVIDER_SETTINGS[provider_name]
            self.ai_model_var.set(
                str(settings.get("model") or defaults["model"])
            )
            self.ai_base_url_var.set(
                str(settings.get("base_url") or defaults["base_url"])
            )
            self.ai_timeout_var.set(
                str(settings.get("timeout_seconds", "90"))
            )
        else:
            self.ai_model_var.set("")
            self.ai_base_url_var.set("")
            self.ai_timeout_var.set("90")

        self.guide_auto_open_var.set(
            bool(settings.get("guide_auto_open", False))
        )
        self.guide_use_ai_var.set(
            bool(settings.get("guide_use_ai", False))
        )
        remember_requested = bool(settings.get("remember_securely", False))
        self.remember_ai_key_var.set(
            remember_requested and keyring is not None
        )
        self.profile_only_var.set(True)
        self.allow_sample_rows_var.set(False)

        credential = ""
        if provider_name in OFFICIAL_PROVIDER_SETTINGS:
            env_name = OFFICIAL_PROVIDER_SETTINGS[
                provider_name
            ]["api_key_environment"]
            credential = os.getenv(env_name, "").strip()
            if (
                not credential
                and remember_requested
                and keyring is not None
            ):
                try:
                    credential = (
                        keyring.get_password(
                            KEYRING_SERVICE_NAME,
                            provider_name,
                        )
                        or ""
                    ).strip()
                except Exception:
                    credential = ""
        self.ai_api_key_var.set(credential)

        if provider_name == AI_PROVIDER_LOCAL:
            self.ai_connection_state = "offline"
            self.ai_connection_message = (
                "Offline mode — manual training remains available."
            )
        elif credential:
            self.ai_connection_state = "configured"
            self.ai_connection_message = (
                f"{provider_name} is configured. Test the connection to verify it."
            )
        else:
            self.ai_connection_state = "disconnected"
            self.ai_connection_message = (
                f"{provider_name} is selected but no API key is available."
            )

    def save_non_secret_ai_settings(self):
        """Persist provider preferences without writing any API key."""
        settings = {
            "provider": self.ai_provider_var.get(),
            "model": self.ai_model_var.get().strip(),
            "base_url": self.ai_base_url_var.get().strip(),
            "timeout_seconds": self.ai_timeout_var.get().strip(),
            "remember_securely": bool(
                self.remember_ai_key_var.get() and keyring is not None
            ),
            "profile_only": True,
            "allow_sample_rows": False,
            "guide_auto_open": bool(self.guide_auto_open_var.get()),
            "guide_use_ai": bool(self.guide_use_ai_var.get()),
        }
        self.ai_settings_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.ai_settings_path.with_suffix(".tmp")
        with open(temporary_path, "w", encoding="utf-8") as settings_file:
            json.dump(settings, settings_file, indent=2)
        os.replace(temporary_path, self.ai_settings_path)

    def resolve_ai_api_key(self, provider_name=None):
        provider = provider_name or self.ai_provider_var.get()
        if provider not in OFFICIAL_PROVIDER_SETTINGS:
            return ""

        key_value = self.ai_api_key_var.get().strip()
        if key_value:
            return key_value

        env_name = OFFICIAL_PROVIDER_SETTINGS[
            provider
        ]["api_key_environment"]
        key_value = os.getenv(env_name, "").strip()
        if key_value:
            self.ai_api_key_var.set(key_value)
            return key_value

        if keyring is not None:
            try:
                key_value = (
                    keyring.get_password(
                        KEYRING_SERVICE_NAME,
                        provider,
                    )
                    or ""
                ).strip()
            except Exception:
                key_value = ""
            if key_value:
                self.ai_api_key_var.set(key_value)
        return key_value

    def save_ai_settings(self, show_confirmation=True):
        try:
            provider_name = self.ai_provider_var.get()
            if provider_name in OFFICIAL_PROVIDER_SETTINGS:
                model_name = self.ai_model_var.get().strip()
                if not model_name:
                    raise ValueError("Enter a provider model name.")
                _validate_provider_base_url(self.ai_base_url_var.get())
                timeout_seconds = float(self.ai_timeout_var.get())
                if timeout_seconds <= 0 or timeout_seconds > 600:
                    raise ValueError(
                        "Timeout must be between 1 and 600 seconds."
                    )

                key_value = self.resolve_ai_api_key(provider_name)
                if self.remember_ai_key_var.get():
                    if keyring is None:
                        raise ValueError(
                            "Secure credential storage requires the optional "
                            "'keyring' package. Install keyring or leave "
                            "'Remember securely' unchecked."
                        )
                    if not key_value:
                        raise ValueError(
                            "Enter an API key before saving it securely."
                        )
                    keyring.set_password(
                        KEYRING_SERVICE_NAME,
                        provider_name,
                        key_value,
                    )
                elif keyring is not None:
                    try:
                        keyring.delete_password(
                            KEYRING_SERVICE_NAME,
                            provider_name,
                        )
                    except Exception:
                        pass

                if key_value and self.ai_connection_state not in (
                    "connected",
                    "testing",
                ):
                    self.ai_connection_state = "configured"
                    self.ai_connection_message = (
                        f"{provider_name} settings are saved. "
                        "Test the connection to verify them."
                    )
            else:
                self.ai_connection_state = "offline"
                self.ai_connection_message = (
                    "Offline mode — manual training remains available."
                )

            self.save_non_secret_ai_settings()
            self.update_provider_status_display()
            self.refresh_settings_status()
            if show_confirmation:
                messagebox.showinfo(
                    "Settings Saved",
                    "Provider preferences were saved. No API key was written "
                    "to the settings file.",
                )
            return True
        except Exception as exc:
            messagebox.showerror("Settings Error", str(exc))
            return False

    def disconnect_ai_provider(self):
        if self.ai_connection_testing:
            messagebox.showinfo(
                "Connection Test Running",
                "Wait for the connection test to finish before disconnecting.",
            )
            return
        provider_name = self.ai_provider_var.get()
        if provider_name in OFFICIAL_PROVIDER_SETTINGS and keyring is not None:
            try:
                keyring.delete_password(
                    KEYRING_SERVICE_NAME,
                    provider_name,
                )
            except Exception:
                pass

        self.ai_api_key_var.set("")
        self.remember_ai_key_var.set(False)
        self.ai_provider_var.set(AI_PROVIDER_LOCAL)
        self.ai_model_var.set("")
        self.ai_base_url_var.set("")
        self.ai_connection_state = "offline"
        self.ai_connection_message = (
            "Offline mode — manual training remains available."
        )
        self.ai_last_connection_check = None
        self.ai_last_connection_details = "Provider disconnected."
        try:
            self.save_non_secret_ai_settings()
        except Exception as exc:
            messagebox.showerror("Disconnect Error", str(exc))
            return
        self.on_ai_provider_changed(update_existing=False)
        self.update_provider_status_display()
        self.refresh_settings_status()

    def update_provider_status_display(self):
        if not hasattr(self, "provider_status_button"):
            return
        provider_name = self.ai_provider_var.get()
        if self.ai_connection_state == "connected":
            text = f"● {provider_name} Connected"
        elif self.ai_connection_state == "testing":
            text = f"◐ Testing {provider_name}"
        elif (
            self.ai_connection_state == "disconnected"
            and provider_name in OFFICIAL_PROVIDER_SETTINGS
        ):
            text = f"● {provider_name} Not Connected"
        elif provider_name in OFFICIAL_PROVIDER_SETTINGS:
            text = f"○ {provider_name} Not Verified"
        else:
            text = "○ Offline Mode"
        self.provider_status_button.config(text=text)

    def refresh_settings_status(self):
        if hasattr(self, "settings_connection_status_label"):
            self.settings_connection_status_label.config(
                text=self.ai_connection_message
            )
        if hasattr(self, "settings_key_status_label"):
            provider_name = self.ai_provider_var.get()
            key_value = self.resolve_ai_api_key(provider_name)
            self.settings_key_status_label.config(
                text=_masked_api_key(key_value)
            )
        state_visuals = {
            "offline": ("OFFLINE", "#6b7280", "white"),
            "configured": ("NOT VERIFIED", "#b26a00", "white"),
            "testing": ("CHECKING…", "#1769aa", "white"),
            "connected": ("CONNECTED", "#17803d", "white"),
            "disconnected": ("FAILED", "#b42318", "white"),
        }
        badge_text, badge_background, badge_foreground = state_visuals.get(
            self.ai_connection_state,
            ("UNKNOWN", "#6b7280", "white"),
        )
        if hasattr(self, "settings_connection_badge"):
            self.settings_connection_badge.config(
                text=badge_text,
                background=badge_background,
                foreground=badge_foreground,
            )
        provider_name = self.ai_provider_var.get()
        if hasattr(self, "settings_connection_provider_label"):
            self.settings_connection_provider_label.config(
                text=f"Provider: {provider_name}"
            )
        if hasattr(self, "settings_connection_model_label"):
            self.settings_connection_model_label.config(
                text=(
                    "Model: "
                    + (
                        self.ai_model_var.get().strip()
                        if provider_name in OFFICIAL_PROVIDER_SETTINGS
                        else "Not applicable"
                    )
                )
            )
        if hasattr(self, "settings_connection_key_label"):
            key_value = self.resolve_ai_api_key(provider_name)
            self.settings_connection_key_label.config(
                text=f"API key: {_masked_api_key(key_value)}"
            )
        if hasattr(self, "settings_last_test_label"):
            last_checked = (
                self.ai_last_connection_check
                or "Not tested in this session"
            )
            self.settings_last_test_label.config(
                text=(
                    f"Last check: {last_checked}\n"
                    f"{self.ai_last_connection_details}"
                )
            )
        self.update_provider_status_display()

    def mark_ai_settings_changed(self, _event=None):
        """Require another test after editing connection-defining settings."""
        if self.ai_connection_testing:
            return
        provider_name = self.ai_provider_var.get()
        if provider_name not in OFFICIAL_PROVIDER_SETTINGS:
            return
        key_value = self.resolve_ai_api_key(provider_name)
        self.ai_connection_state = (
            "configured" if key_value else "disconnected"
        )
        self.ai_connection_message = (
            f"{provider_name} settings changed. Test the connection again."
            if key_value
            else f"{provider_name} is selected but no API key is available."
        )
        self.ai_last_connection_check = None
        self.ai_last_connection_details = "Current settings are not verified."
        self.refresh_settings_status()

    def _resize_settings_scroll_region(self, _event=None):
        if not hasattr(self, "settings_canvas"):
            return
        try:
            self.settings_canvas.configure(
                scrollregion=self.settings_canvas.bbox("all")
            )
        except tk.TclError:
            pass

    def _resize_settings_inner_width(self, event):
        if not hasattr(self, "settings_canvas_window"):
            return
        try:
            self.settings_canvas.itemconfigure(
                self.settings_canvas_window,
                width=event.width,
            )
        except tk.TclError:
            pass

    def _on_settings_mousewheel(self, event):
        if (
            self.current_view != "settings"
            or not hasattr(self, "settings_canvas")
        ):
            return
        if getattr(event, "num", None) == 4:
            units = -3
        elif getattr(event, "num", None) == 5:
            units = 3
        else:
            delta = getattr(event, "delta", 0)
            units = -int(delta / 120) if delta else 0
        if units:
            self.settings_canvas.yview_scroll(units, "units")
        return "break"

    def toggle_ai_key_visibility(self):
        if hasattr(self, "settings_api_key_entry"):
            self.settings_api_key_entry.config(
                show="" if self.show_ai_key_var.get() else "•"
            )
