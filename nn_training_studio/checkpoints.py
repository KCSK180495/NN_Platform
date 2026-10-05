"""Checkpoints for NN Training Studio."""

import hashlib
import json
import pandas as pd
from nn_training_studio.constants import (
    BEST_MODEL_FILENAME,
    BEST_STATE_FILENAME,
    DATA_MODE_IMAGE,
    RECOVERY_DIRECTORY_NAME,
    TRAINING_CHECKPOINT_DIRECTORY,
)
from nn_training_studio.settings import (
    _application_settings_path,
)
from nn_training_studio.training import (
    validate_early_stopping_settings,
)


def _training_checkpoint_root():
    """Return the per-user directory used for training checkpoints."""
    return (
        _application_settings_path().parent
        / TRAINING_CHECKPOINT_DIRECTORY
    )


def _training_run_fingerprint(df, settings):
    """
    Create a stable identifier for one dataset and training configuration.

    The identifier lets BackupAndRestore locate an interrupted run after the
    application restarts. Provider credentials and AI response text are
    intentionally excluded.
    """
    fingerprint_keys = (
        "data_mode",
        "task_type",
        "feature_cols",
        "target_cols",
        "missing_method",
        "scaler_option",
        "target_scaler_option",
        "model_type",
        "custom_model_input_mode",
        "custom_model_code",
        "safe_model_spec",
        "window_size",
        "stride",
        "forecast_horizon",
        "anomaly_percentile",
        "batch_size",
        "validation_split",
        "hidden_activation",
        "dropout_rate",
        "output_mode",
        "output_units",
        "output_activation",
        "loss_name",
        "optimizer_name",
        "learning_rate",
        "test_size",
        "built_in_filter_method",
        "built_in_filter_columns",
        "moving_window",
        "ema_span",
        "median_window",
        "kalman_q",
        "kalman_r",
        "custom_filter_enabled",
        "custom_filter_mode",
        "custom_filter_columns",
        "custom_filter_code",
        "safe_filter_spec",
        "image_directory",
        "image_class_names",
        "image_height",
        "image_width",
        "image_color_mode",
        "image_augmentation",
        "image_pretrained_weights",
    )
    configuration = {
        key: settings.get(key)
        for key in fingerprint_keys
    }
    # Changed preprocessing must not reuse a model fitted with old scalers.
    configuration["training_pipeline_revision"] = "training-only-scaling-v1"
    configuration["early_stopping"] = validate_early_stopping_settings(
        settings.get("early_stopping")
    )

    hasher = hashlib.sha256()
    hasher.update(
        json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    )
    hasher.update(
        json.dumps(
            {
                "rows": int(len(df)),
                "columns": [str(column) for column in df.columns],
            },
            sort_keys=True,
        ).encode("utf-8")
    )

    selected_columns = list(dict.fromkeys(
        list(settings.get("feature_cols") or [])
        + list(settings.get("target_cols") or [])
        + (["annotation_segment_id"] if "annotation_segment_id" in df.columns else [])
    ))
    selected_columns = [
        column for column in selected_columns if column in df.columns
    ]
    if settings.get("data_mode") == DATA_MODE_IMAGE:
        selected_columns = [
            column
            for column in ("file_path", "class_name", "class_index")
            if column in df.columns
        ]
    if selected_columns:
        try:
            data_hashes = pd.util.hash_pandas_object(
                df[selected_columns],
                index=True,
                categorize=True,
            )
            hasher.update(data_hashes.to_numpy().tobytes())
        except Exception:
            fallback = pd.concat(
                [df[selected_columns].head(50), df[selected_columns].tail(50)]
            )
            hasher.update(
                fallback.to_csv(index=True).encode(
                    "utf-8",
                    errors="replace",
                )
            )

    return hasher.hexdigest()[:20]


def prepare_training_checkpoint_paths(df, settings):
    """Create and return stable checkpoint locations for one training run."""
    run_id = _training_run_fingerprint(df, settings)
    run_directory = _training_checkpoint_root() / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    return {
        "run_id": run_id,
        "run_directory": run_directory,
        "best_model_path": run_directory / BEST_MODEL_FILENAME,
        "best_state_path": run_directory / BEST_STATE_FILENAME,
        "recovery_directory": run_directory / RECOVERY_DIRECTORY_NAME,
    }
