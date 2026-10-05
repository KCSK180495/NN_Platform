"""Training for NN Training Studio."""

from datetime import datetime
import json
import keras
import numpy as np
import os
import re
import tensorflow as tf


def validate_early_stopping_settings(options=None, epochs=None):
    """Normalize old projects to the previous defaults and validate UI input."""
    options = {} if options is None else options
    if not isinstance(options, dict):
        raise ValueError("Early-stopping settings must be an object.")
    enabled = options.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("Enable early stopping must be true or false.")
    result = {"enabled": enabled, "patience": 10, "min_delta": 0.0,
              "start_from_epoch": 0}
    # Disabled controls may contain unfinished edits. They do not affect a run.
    if not enabled:
        return result
    for key, label in (("patience", "Patience"),
                       ("start_from_epoch", "Warm-up epochs")):
        value = options.get(key, result[key])
        if isinstance(value, bool) or not re.fullmatch(r"\+?\d+", str(value).strip()):
            raise ValueError(f"Early stopping: {label} must be a non-negative whole number.")
        result[key] = int(value)
    try:
        value = options.get("min_delta", 0.0)
        if isinstance(value, bool):
            raise ValueError()
        result["min_delta"] = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("Early stopping: minimum improvement must be a finite non-negative number.") from exc
    if not np.isfinite(result["min_delta"]) or result["min_delta"] < 0:
        raise ValueError("Early stopping: minimum improvement must be a finite non-negative number.")
    if epochs is not None and result["start_from_epoch"] >= epochs:
        raise ValueError("Early stopping: warm-up epochs must be smaller than the total epochs.")
    return result


def build_early_stopping_callbacks(settings, log_function):
    options = validate_early_stopping_settings(
        settings.get("early_stopping"), settings.get("epochs")
    )
    if not options["enabled"]:
        log_function("Early stopping: disabled; training uses the maximum epoch count.")
        return []
    log_function(
        "Early stopping: validation loss; "
        f"patience={options['patience']}, minimum improvement={options['min_delta']:g}, "
        f"warm-up={options['start_from_epoch']} epoch(s)."
    )
    return [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=options["patience"],
        min_delta=options["min_delta"], start_from_epoch=options["start_from_epoch"],
        restore_best_weights=True,
    )]


class TrainingProgressCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        log_function,
        epoch_function=None,
        metric_name="accuracy",
        metric_label="accuracy"
    ):
        super().__init__()
        self.log_function = log_function
        self.epoch_function = epoch_function
        self.metric_name = metric_name
        self.metric_label = metric_label

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        loss = float(logs.get("loss", 0.0))
        metric = logs.get(self.metric_name)
        val_loss = logs.get("val_loss")
        val_metric = logs.get("val_" + self.metric_name)

        val_loss_value = None if val_loss is None else float(val_loss)
        metric_value = None if metric is None else float(metric)
        val_metric_value = None if val_metric is None else float(val_metric)

        if val_loss_value is not None:
            msg = (
                f"Epoch {epoch + 1}: "
                f"loss={loss:.6f}, {self.metric_label}="
                f"{metric_value if metric_value is not None else float('nan'):.6f}, "
                f"val_loss={val_loss_value:.6f}, val_{self.metric_label}="
                f"{val_metric_value if val_metric_value is not None else float('nan'):.6f}"
            )
        else:
            msg = (
                f"Epoch {epoch + 1}: "
                f"loss={loss:.6f}, {self.metric_label}="
                f"{metric_value if metric_value is not None else float('nan'):.6f}"
            )

        self.log_function(msg)

        if self.epoch_function is not None:
            self.epoch_function(
                epoch + 1,
                loss,
                metric_value,
                val_loss_value,
                val_metric_value
            )


class PersistentBestModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """
    ModelCheckpoint with a small persistent record of the best monitored value.

    Keras restores the model/optimizer/epoch through BackupAndRestore, but a
    newly constructed ModelCheckpoint callback would otherwise forget the best
    validation loss seen before an interruption. This class restores that
    value so a worse post-recovery epoch cannot overwrite the true best model.
    """

    def __init__(self, filepath, state_path, **kwargs):
        self.state_path = str(state_path)
        self.checkpoint_mode = str(kwargs.get("mode", "auto"))
        self.best_epoch = None
        self.restored_previous_best = False
        super().__init__(filepath=str(filepath), **kwargs)
        self._restore_persistent_state()

    def _restore_persistent_state(self):
        if not (
            os.path.exists(self.filepath)
            and os.path.exists(self.state_path)
        ):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as state_file:
                state = json.load(state_file)
            if state.get("monitor") != self.monitor:
                return
            best_value = float(state["best_value"])
            if not np.isfinite(best_value):
                return
            self.best = best_value
            self.best_epoch = int(state["best_epoch"])
            self.restored_previous_best = True
        except Exception:
            # A missing or damaged state record must never block training.
            self.best_epoch = None
            self.restored_previous_best = False

    def _save_persistent_state(self):
        best_value = self._finite_best_value(self.best)
        if best_value is None or self.best_epoch is None:
            return

        state_directory = os.path.dirname(self.state_path)
        if state_directory:
            os.makedirs(state_directory, exist_ok=True)
        temporary_path = self.state_path + ".tmp"
        state = {
            "monitor": self.monitor,
            "mode": self.checkpoint_mode,
            "best_value": best_value,
            "best_epoch": int(self.best_epoch),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(temporary_path, "w", encoding="utf-8") as state_file:
            json.dump(state, state_file, indent=2)
        os.replace(temporary_path, self.state_path)

    @staticmethod
    def _finite_best_value(value):
        """Return a finite float, or None while Keras has no valid best value."""
        try:
            numeric_value = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return numeric_value if np.isfinite(numeric_value) else None

    def on_epoch_end(self, epoch, logs=None):
        # Keras 3 may leave ModelCheckpoint.best as None until it processes the
        # first monitored epoch. Older releases may use +/- infinity instead.
        # Treat both states as "no previous best" rather than converting them
        # directly with float(), which raises TypeError for None.
        previous_best = self._finite_best_value(self.best)
        super().on_epoch_end(epoch, logs)
        current_best = self._finite_best_value(self.best)
        if current_best is None:
            return
        if previous_best is None or current_best != previous_best:
            self.best_epoch = int(epoch) + 1
            self._save_persistent_state()


def load_best_checkpoint_model(model_path):
    """Load a complete best-model checkpoint with safe compatibility fallbacks."""
    load_errors = []
    try:
        return keras.models.load_model(str(model_path))
    except Exception as exc:
        load_errors.append(str(exc))

    try:
        return keras.models.load_model(
            str(model_path),
            compile=False,
            safe_mode=False,
        )
    except TypeError:
        try:
            return keras.models.load_model(
                str(model_path),
                compile=False,
            )
        except Exception as exc:
            load_errors.append(str(exc))
    except Exception as exc:
        load_errors.append(str(exc))

    raise ValueError(
        "The automatic best-model checkpoint could not be reloaded.\n"
        + "\n".join(load_errors[-2:])
    )
