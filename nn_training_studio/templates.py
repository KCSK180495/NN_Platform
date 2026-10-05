"""Templates for NN Training Studio."""

from nn_training_studio.constants import (
    TASK_AUTOENCODER,
)


def _safe_layer_template(layer_type, **overrides):
    """Return one complete editable allowlisted layer specification."""
    layer = {
        "layer_type": layer_type,
        "units": 64,
        "activation": "relu",
        "filters": 64,
        "kernel_size": 3,
        "strides": 1,
        "padding": "same",
        "pool_size": 2,
        "dropout_rate": 0.2,
        "return_sequences": False,
        "bidirectional": False,
    }
    layer.update(overrides)
    return layer


def make_safe_filter_template(columns):
    """Create an editable safe filter without requiring an AI recommendation."""
    unique_columns = list(dict.fromkeys(str(column) for column in columns))
    pipelines = []
    for column in unique_columns:
        pipelines.append({
            "column": column,
            "steps": [{
                "operation": "moving_average",
                "window_size": 5,
                "span": 10,
                "polyorder": 2,
                "cutoff_hz": 1.0,
                "lowcut_hz": 1.0,
                "highcut_hz": 2.0,
                "order": 4,
                "notch_hz": 1.0,
                "quality_factor": 30.0,
                "process_noise": 0.00001,
                "measurement_noise": 0.01,
                "causal": True,
            }],
        })
    return {
        "name": "User Custom Filter",
        "description": (
            "Editable allowlisted filter created from the user's selected "
            "columns. Change each operation and parameter as required."
        ),
        "pipelines": pipelines,
        "reasons": ["Created directly by the user; no recommendation required."],
        "risks": ["Check that useful transients and frequency content remain."],
        "validation_checks": [
            "Review correlation, RMSE, standard-deviation ratio, and plots."
        ],
    }


def make_safe_model_template(input_mode, task_type):
    """Create an editable safe model without requiring an AI recommendation."""
    if input_mode == "Row-based (2D)":
        layer_specs = [
            _safe_layer_template("Dense", units=128),
            _safe_layer_template("Dropout", dropout_rate=0.2),
            _safe_layer_template("Dense", units=64),
        ]
    elif task_type == TASK_AUTOENCODER:
        # A sequence autoencoder must retain the time dimension. The builder
        # adds the TimeDistributed reconstruction output automatically.
        layer_specs = [
            _safe_layer_template("Conv1D", filters=64, kernel_size=5),
            _safe_layer_template("Dropout", dropout_rate=0.2),
            _safe_layer_template("Conv1D", filters=32, kernel_size=3),
        ]
    else:
        layer_specs = [
            _safe_layer_template("Conv1D", filters=64, kernel_size=5),
            _safe_layer_template("MaxPooling1D", pool_size=2),
            _safe_layer_template("GlobalAveragePooling1D"),
            _safe_layer_template("Dropout", dropout_rate=0.2),
            _safe_layer_template("Dense", units=64),
        ]
    return {
        "name": "User Custom Neural Network",
        "description": (
            "Editable allowlisted architecture created directly from the "
            "current task and input format."
        ),
        "input_mode": input_mode,
        "layers": layer_specs,
        "reasons": ["Created directly by the user; no recommendation required."],
        "warnings": ["Validate the model shape before starting training."],
        "training_notes": [
            "Output units, output activation, loss, and optimizer come from "
            "the current model settings."
        ],
    }
