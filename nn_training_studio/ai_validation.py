"""Ai validation for NN Training Studio."""

import json
import numpy as np
from nn_training_studio.constants import (
    SAFE_FILTER_OPERATION_TYPES,
    SAFE_MODEL_INPUT_MODES,
    SAFE_MODEL_LAYER_TYPES,
    SUPPORTED_FILTER_METHODS,
    SUPPORTED_MODEL_TYPES,
    TASK_AUTOENCODER,
    TASK_CLASSIFICATION,
    TASK_FORECASTING,
    TASK_MULTI_OUTPUT,
    TASK_REGRESSION,
    TASK_TYPES,
)


class AIProviderError(RuntimeError):
    """Raised when an AI provider request or response cannot be used safely."""


def _require_exact_keys(mapping, required_keys, location):
    if not isinstance(mapping, dict):
        raise AIProviderError(f"{location} must be a JSON object.")

    required = set(required_keys)
    actual = set(mapping)
    missing = sorted(required - actual)
    extra = sorted(actual - required)
    if missing:
        raise AIProviderError(
            f"{location} is missing required fields: {', '.join(missing)}"
        )
    if extra:
        raise AIProviderError(
            f"{location} contains unsupported fields: {', '.join(extra)}"
        )


def _validate_string_list(value, location, allowed_values=None):
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise AIProviderError(f"{location} must be a list of non-empty strings.")
    cleaned = list(dict.fromkeys(item.strip() for item in value))
    if allowed_values is not None:
        invalid = [item for item in cleaned if item not in allowed_values]
        if invalid:
            raise AIProviderError(
                f"{location} contains unsupported values: {invalid}"
            )
    return cleaned


def _validated_confidence(value, location):
    if isinstance(value, bool):
        raise AIProviderError(f"{location} must be numeric, not a boolean.")
    try:
        confidence = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AIProviderError(f"{location} must be numeric.") from exc
    if not 0.0 <= confidence <= 1.0:
        raise AIProviderError(f"{location} must be between 0 and 1.")
    return confidence


def _validated_positive_int(value, location, minimum=1, maximum=1000000):
    if isinstance(value, bool):
        raise AIProviderError(f"{location} must be a whole number; received a boolean.")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AIProviderError(f"{location} must be a whole number; received {str(value)[:60]!r}.") from exc
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise AIProviderError(f"{location} must be a finite whole number; received {str(value)[:60]!r}.")
    number = int(numeric)
    if not minimum <= number <= maximum:
        raise AIProviderError(f"{location} must be between {minimum} and {maximum}; received {number}.")
    return number


def _validated_positive_float(value, location, maximum=1000000.0):
    if isinstance(value, bool):
        raise AIProviderError(f"{location} must be numeric, not a boolean.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AIProviderError(f"{location} must be numeric.") from exc
    if not np.isfinite(number) or number <= 0 or number > maximum:
        raise AIProviderError(
            f"{location} must be greater than 0 and no more than {maximum}."
        )
    return number


def validate_ai_recommendation(recommendation, dataset_profile):
    """Strictly validate and normalize a provider recommendation locally."""
    if isinstance(recommendation, dict):
        recommendation = json.loads(json.dumps(recommendation))
        _normalize_unused_recommendation_controls(recommendation, dataset_profile)
    limits = recommendation_dataset_limits(dataset_profile)
    root_keys = [
        "summary",
        "task",
        "filter",
        "model",
        "future_generation",
    ]
    _require_exact_keys(recommendation, root_keys, "AI recommendation")

    if not isinstance(recommendation["summary"], str):
        raise AIProviderError("AI recommendation.summary must be text.")
    recommendation["summary"] = recommendation["summary"].strip()

    task = recommendation["task"]
    task_keys = [
        "task_type",
        "confidence",
        "input_columns",
        "target_columns",
        "reasons",
        "warnings",
        "alternatives",
    ]
    _require_exact_keys(task, task_keys, "AI recommendation.task")

    available_columns = {
        item["name"] for item in dataset_profile["column_profiles"]
    }
    numeric_columns = set(dataset_profile["numeric_columns"])
    task_type = task["task_type"]
    if task_type not in TASK_TYPES:
        raise AIProviderError(f"Unsupported AI task recommendation: {task_type}")

    task["confidence"] = _validated_confidence(
        task["confidence"],
        "AI recommendation.task.confidence",
    )
    task["input_columns"] = _validate_string_list(
        task["input_columns"],
        "AI recommendation.task.input_columns",
        available_columns,
    )
    task["target_columns"] = _validate_string_list(
        task["target_columns"],
        "AI recommendation.task.target_columns",
        available_columns,
    )
    task["reasons"] = _validate_string_list(
        task["reasons"],
        "AI recommendation.task.reasons",
    )
    task["warnings"] = _validate_string_list(
        task["warnings"],
        "AI recommendation.task.warnings",
    )
    task["alternatives"] = _validate_string_list(
        task["alternatives"],
        "AI recommendation.task.alternatives",
        set(TASK_TYPES),
    )

    if not task["input_columns"]:
        raise AIProviderError("The AI recommendation must contain input columns.")
    non_numeric_inputs = [
        column for column in task["input_columns"]
        if column not in numeric_columns
    ]
    if non_numeric_inputs:
        raise AIProviderError(
            "AI input columns must be numeric: "
            + ", ".join(non_numeric_inputs)
        )
    overlap = set(task["input_columns"]) & set(task["target_columns"])
    if overlap:
        raise AIProviderError(
            "AI input and target columns overlap: " + ", ".join(sorted(overlap))
        )

    target_count = len(task["target_columns"])
    if task_type in (TASK_CLASSIFICATION, TASK_REGRESSION) and target_count != 1:
        raise AIProviderError(f"{task_type} requires exactly one target column.")
    if task_type == TASK_MULTI_OUTPUT and target_count < 2:
        raise AIProviderError(
            "Multi-output Regression requires at least two target columns."
        )
    if task_type == TASK_FORECASTING and target_count < 1:
        raise AIProviderError(
            "Time-series Forecasting requires at least one target column."
        )
    if task_type == TASK_AUTOENCODER and target_count != 0:
        raise AIProviderError("Autoencoder recommendations must not use targets.")
    if task_type in (
        TASK_REGRESSION,
        TASK_MULTI_OUTPUT,
        TASK_FORECASTING,
    ):
        non_numeric_targets = [
            column for column in task["target_columns"]
            if column not in numeric_columns
        ]
        if non_numeric_targets:
            raise AIProviderError(
                "Continuous-output targets must be numeric: "
                + ", ".join(non_numeric_targets)
            )

    filter_plan = recommendation["filter"]
    filter_keys = [
        "method",
        "columns",
        "moving_average_window",
        "ema_span",
        "median_window",
        "kalman_q",
        "kalman_r",
        "confidence",
        "reasons",
        "risks",
        "validation_checks",
    ]
    _require_exact_keys(filter_plan, filter_keys, "AI recommendation.filter")
    if filter_plan["method"] not in SUPPORTED_FILTER_METHODS:
        raise AIProviderError(
            f"Unsupported AI filter recommendation: {filter_plan['method']}"
        )
    filter_plan["columns"] = _validate_string_list(
        filter_plan["columns"],
        "AI recommendation.filter.columns",
        numeric_columns,
    )
    if filter_plan["method"] == "No filter":
        filter_plan["columns"] = []
    elif not filter_plan["columns"]:
        raise AIProviderError(
            "A non-empty filter column list is required for the selected filter."
        )
    filter_plan["moving_average_window"] = _validated_positive_int(
        filter_plan["moving_average_window"],
        "AI recommendation.filter.moving_average_window",
        maximum=limits["filter_window_max"],
    )
    filter_plan["ema_span"] = _validated_positive_int(
        filter_plan["ema_span"],
        "AI recommendation.filter.ema_span",
        maximum=limits["filter_window_max"],
    )
    filter_plan["median_window"] = _validated_positive_int(
        filter_plan["median_window"],
        "AI recommendation.filter.median_window",
        maximum=limits["filter_window_max"],
    )
    filter_plan["kalman_q"] = _validated_positive_float(
        filter_plan["kalman_q"],
        "AI recommendation.filter.kalman_q",
    )
    filter_plan["kalman_r"] = _validated_positive_float(
        filter_plan["kalman_r"],
        "AI recommendation.filter.kalman_r",
    )
    filter_plan["confidence"] = _validated_confidence(
        filter_plan["confidence"],
        "AI recommendation.filter.confidence",
    )
    for key in ("reasons", "risks", "validation_checks"):
        filter_plan[key] = _validate_string_list(
            filter_plan[key],
            f"AI recommendation.filter.{key}",
        )

    model = recommendation["model"]
    model_keys = [
        "model_type",
        "window_size",
        "stride",
        "epochs",
        "batch_size",
        "validation_split_percent",
        "hidden_activation",
        "dropout_rate",
        "output_activation",
        "loss_function",
        "optimizer",
        "learning_rate",
        "feature_scaling",
        "target_scaling",
        "confidence",
        "reasons",
        "warnings",
    ]
    _require_exact_keys(model, model_keys, "AI recommendation.model")
    if model["model_type"] not in SUPPORTED_MODEL_TYPES:
        raise AIProviderError(
            f"Unsupported AI model recommendation: {model['model_type']}"
        )
    model["window_size"] = _validated_positive_int(
        model["window_size"],
        "AI recommendation.model.window_size",
        maximum=limits["model_window_max"],
    )
    model["stride"] = _validated_positive_int(
        model["stride"],
        "AI recommendation.model.stride",
        maximum=limits["model_window_max"],
    )
    model["epochs"] = _validated_positive_int(
        model["epochs"],
        "AI recommendation.model.epochs",
        maximum=100000,
    )
    model["batch_size"] = _validated_positive_int(
        model["batch_size"],
        "AI recommendation.model.batch_size",
        maximum=100000,
    )
    model["validation_split_percent"] = float(
        model["validation_split_percent"]
    )
    if not 1.0 <= model["validation_split_percent"] <= 79.0:
        raise AIProviderError(
            "AI model validation_split_percent must be between 1 and 79."
        )
    model["dropout_rate"] = float(model["dropout_rate"])
    if not 0.0 <= model["dropout_rate"] < 1.0:
        raise AIProviderError(
            "AI model dropout_rate must be between 0 and 0.99."
        )
    model["learning_rate"] = _validated_positive_float(
        model["learning_rate"],
        "AI recommendation.model.learning_rate",
        maximum=10.0,
    )
    model["confidence"] = _validated_confidence(
        model["confidence"],
        "AI recommendation.model.confidence",
    )
    for key in ("reasons", "warnings"):
        model[key] = _validate_string_list(
            model[key],
            f"AI recommendation.model.{key}",
        )

    allowed_model_values = {
        "hidden_activation": {"relu", "tanh", "sigmoid", "elu", "selu"},
        "output_activation": {"softmax", "sigmoid", "linear", "tanh"},
        "loss_function": {
            "sparse_categorical_crossentropy",
            "categorical_crossentropy",
            "binary_crossentropy",
            "mean_squared_error",
            "mean_absolute_error",
        },
        "optimizer": {"Adam", "SGD", "RMSprop", "Nadam"},
        "feature_scaling": {
            "MinMaxScaler",
            "StandardScaler",
            "No normalization",
        },
        "target_scaling": {
            "StandardScaler",
            "MinMaxScaler",
            "No normalization",
        },
    }
    for key, allowed in allowed_model_values.items():
        if model[key] not in allowed:
            raise AIProviderError(
                f"Unsupported AI model {key}: {model[key]}"
            )

    autoencoder_models = {"Dense Autoencoder", "LSTM Autoencoder"}
    if task_type == TASK_AUTOENCODER and model["model_type"] not in autoencoder_models:
        raise AIProviderError(
            "Autoencoder tasks require Dense Autoencoder or LSTM Autoencoder."
        )
    if task_type != TASK_AUTOENCODER and model["model_type"] in autoencoder_models:
        raise AIProviderError(
            "Autoencoder model types cannot be used for a supervised task."
        )
    if task_type == TASK_CLASSIFICATION:
        if model["loss_function"] not in {
            "sparse_categorical_crossentropy",
            "categorical_crossentropy",
            "binary_crossentropy",
        }:
            raise AIProviderError(
                "Classification requires a classification loss function."
            )
        if model["loss_function"] == "binary_crossentropy":
            if model["output_activation"] != "sigmoid":
                raise AIProviderError(
                    "Binary classification requires sigmoid output."
                )
        elif model["output_activation"] != "softmax":
            raise AIProviderError(
                "Categorical classification requires softmax output."
            )
    else:
        if model["loss_function"] not in {
            "mean_squared_error",
            "mean_absolute_error",
        }:
            raise AIProviderError(
                f"{task_type} requires mean_squared_error or mean_absolute_error."
            )
        if model["output_activation"] != "linear":
            raise AIProviderError(f"{task_type} requires linear output.")

    future = recommendation["future_generation"]
    future_keys = [
        "custom_filter_would_help",
        "custom_filter_reason",
        "custom_model_would_help",
        "custom_model_reason",
    ]
    _require_exact_keys(
        future,
        future_keys,
        "AI recommendation.future_generation",
    )
    for key in ("custom_filter_would_help", "custom_model_would_help"):
        if not isinstance(future[key], bool):
            raise AIProviderError(
                f"AI recommendation.future_generation.{key} must be boolean."
            )
    for key in ("custom_filter_reason", "custom_model_reason"):
        if not isinstance(future[key], str):
            raise AIProviderError(
                f"AI recommendation.future_generation.{key} must be text."
            )
        future[key] = future[key].strip()

    _check_recommendation_sample_fit(recommendation, dataset_profile)
    return recommendation


def _validated_text(value, location):
    if not isinstance(value, str) or not value.strip():
        raise AIProviderError(f"{location} must be non-empty text.")
    return value.strip()


def _validated_nonnegative_float(value, location, maximum=1000000.0):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AIProviderError(f"{location} must be numeric.") from exc
    if not np.isfinite(number) or number < 0 or number > maximum:
        raise AIProviderError(
            f"{location} must be between 0 and {maximum}."
        )
    return number


def validate_ai_generation(
    generation,
    dataset_profile,
    recommendation,
):
    """Validate and normalize generated filter/model specifications locally."""
    _require_exact_keys(
        generation,
        ["summary", "filter_spec", "model_spec"],
        "AI generation",
    )
    generation["summary"] = _validated_text(
        generation["summary"],
        "AI generation.summary",
    )

    validated_recommendation = validate_ai_recommendation(
        json.loads(json.dumps(recommendation)),
        dataset_profile,
    )
    task = validated_recommendation["task"]
    input_columns = set(task["input_columns"])
    target_columns = set(task["target_columns"])
    sampling_frequency = dataset_profile.get("sampling_frequency_hz")

    filter_spec = generation["filter_spec"]
    filter_keys = [
        "name",
        "description",
        "pipelines",
        "reasons",
        "risks",
        "validation_checks",
    ]
    _require_exact_keys(
        filter_spec,
        filter_keys,
        "AI generation.filter_spec",
    )
    filter_spec["name"] = _validated_text(
        filter_spec["name"],
        "AI generation.filter_spec.name",
    )
    filter_spec["description"] = _validated_text(
        filter_spec["description"],
        "AI generation.filter_spec.description",
    )
    for key in ("reasons", "risks", "validation_checks"):
        filter_spec[key] = _validate_string_list(
            filter_spec[key],
            f"AI generation.filter_spec.{key}",
        )
    pipelines = filter_spec["pipelines"]
    if not isinstance(pipelines, list):
        raise AIProviderError(
            "AI generation.filter_spec.pipelines must be a list."
        )
    if len(pipelines) > len(input_columns):
        raise AIProviderError(
            "The generated filter has more pipelines than input columns."
        )

    seen_columns = set()
    frequency_operations = {
        "butterworth_lowpass",
        "butterworth_highpass",
        "butterworth_bandpass",
        "notch",
    }
    step_keys = [
        "operation",
        "window_size",
        "span",
        "polyorder",
        "cutoff_hz",
        "lowcut_hz",
        "highcut_hz",
        "order",
        "notch_hz",
        "quality_factor",
        "process_noise",
        "measurement_noise",
        "causal",
    ]
    for pipeline_index, pipeline in enumerate(pipelines):
        location = (
            f"AI generation.filter_spec.pipelines[{pipeline_index}]"
        )
        _require_exact_keys(pipeline, ["column", "steps"], location)
        column = _validated_text(pipeline["column"], location + ".column")
        if column not in input_columns:
            raise AIProviderError(
                f"Generated filters may only use recommended inputs: {column}"
            )
        if column in target_columns:
            raise AIProviderError(
                f"Generated filters may not modify a target column: {column}"
            )
        if column in seen_columns:
            raise AIProviderError(
                f"Duplicate generated filter pipeline for column: {column}"
            )
        seen_columns.add(column)

        steps = pipeline["steps"]
        if not isinstance(steps, list) or not 1 <= len(steps) <= 3:
            raise AIProviderError(
                f"{location}.steps must contain between 1 and 3 operations."
            )
        for step_index, step in enumerate(steps):
            step_location = f"{location}.steps[{step_index}]"
            _require_exact_keys(step, step_keys, step_location)
            operation = step["operation"]
            if operation not in SAFE_FILTER_OPERATION_TYPES:
                raise AIProviderError(
                    f"Unsupported safe filter operation: {operation}"
                )
            # Structured Outputs requires every field, even when a parameter
            # does not belong to the selected operation. Normalize irrelevant
            # placeholders instead of rejecting an otherwise valid filter.
            # Parameters actually used by the operation remain strict.
            if operation in ("moving_average", "median", "savitzky_golay"):
                step["window_size"] = _validated_positive_int(
                    step["window_size"],
                    step_location + ".window_size",
                    maximum=10001,
                )
            else:
                step["window_size"] = 5

            if operation == "exponential_moving_average":
                step["span"] = _validated_positive_int(
                    step["span"],
                    step_location + ".span",
                    maximum=10001,
                )
            else:
                step["span"] = 10

            if operation == "savitzky_golay":
                step["polyorder"] = _validated_positive_int(
                    step["polyorder"],
                    step_location + ".polyorder",
                    minimum=0,
                    maximum=15,
                )
            else:
                step["polyorder"] = 2

            if operation.startswith("butterworth_"):
                step["order"] = _validated_positive_int(
                    step["order"],
                    step_location + ".order",
                    maximum=12,
                )
            else:
                step["order"] = 4

            active_float_keys = set({
                "butterworth_lowpass": ("cutoff_hz",),
                "butterworth_highpass": ("cutoff_hz",),
                "butterworth_bandpass": ("lowcut_hz", "highcut_hz"),
                "notch": ("notch_hz", "quality_factor"),
                "simple_kalman": ("process_noise", "measurement_noise"),
            }.get(operation, ()))
            float_defaults = {
                "cutoff_hz": 1.0,
                "lowcut_hz": 1.0,
                "highcut_hz": 2.0,
                "notch_hz": 1.0,
                "quality_factor": 30.0,
                "process_noise": 0.00001,
                "measurement_noise": 0.01,
            }
            for key, default_value in float_defaults.items():
                if key in active_float_keys:
                    step[key] = _validated_positive_float(
                        step[key],
                        step_location + "." + key,
                    )
                else:
                    step[key] = default_value
            if not isinstance(step["causal"], bool):
                raise AIProviderError(
                    f"{step_location}.causal must be boolean."
                )

            if operation in ("moving_average", "median"):
                if step["window_size"] % 2 == 0:
                    step["window_size"] += 1
            if operation == "savitzky_golay":
                if step["window_size"] % 2 == 0:
                    step["window_size"] += 1
                if step["window_size"] <= step["polyorder"]:
                    raise AIProviderError(
                        f"{step_location}: Savitzky-Golay window must be "
                        "larger than polyorder."
                    )
            if operation in frequency_operations:
                if sampling_frequency is None:
                    raise AIProviderError(
                        "A sampling frequency is required for generated "
                        f"{operation} filters."
                    )
                nyquist = float(sampling_frequency) / 2.0
                checked_frequencies = []
                if operation in (
                    "butterworth_lowpass",
                    "butterworth_highpass",
                ):
                    checked_frequencies = [step["cutoff_hz"]]
                elif operation == "butterworth_bandpass":
                    checked_frequencies = [
                        step["lowcut_hz"],
                        step["highcut_hz"],
                    ]
                    if step["lowcut_hz"] >= step["highcut_hz"]:
                        raise AIProviderError(
                            f"{step_location}: band-pass lowcut_hz must be "
                            "below highcut_hz."
                        )
                elif operation == "notch":
                    checked_frequencies = [step["notch_hz"]]
                if any(
                    frequency <= 0 or frequency >= nyquist
                    for frequency in checked_frequencies
                ):
                    raise AIProviderError(
                        f"{step_location}: filter frequencies must be between "
                        f"0 and Nyquist ({nyquist:g} Hz)."
                    )

    model_spec = generation["model_spec"]
    model_keys = [
        "name",
        "description",
        "input_mode",
        "layers",
        "reasons",
        "warnings",
        "training_notes",
    ]
    _require_exact_keys(
        model_spec,
        model_keys,
        "AI generation.model_spec",
    )
    model_spec["name"] = _validated_text(
        model_spec["name"],
        "AI generation.model_spec.name",
    )
    model_spec["description"] = _validated_text(
        model_spec["description"],
        "AI generation.model_spec.description",
    )
    if model_spec["input_mode"] not in SAFE_MODEL_INPUT_MODES:
        raise AIProviderError(
            f"Unsupported safe model input mode: {model_spec['input_mode']}"
        )
    for key in ("reasons", "warnings", "training_notes"):
        model_spec[key] = _validate_string_list(
            model_spec[key],
            f"AI generation.model_spec.{key}",
        )

    layer_specs = model_spec["layers"]
    if not isinstance(layer_specs, list) or not 1 <= len(layer_specs) <= 12:
        raise AIProviderError(
            "AI generation.model_spec.layers must contain 1 to 12 layers."
        )
    layer_keys = [
        "layer_type",
        "units",
        "activation",
        "filters",
        "kernel_size",
        "strides",
        "padding",
        "pool_size",
        "dropout_rate",
        "return_sequences",
        "bidirectional",
    ]
    row_allowed = {"Dense", "Dropout", "BatchNormalization"}
    activation_allowed = {
        "relu",
        "tanh",
        "sigmoid",
        "elu",
        "selu",
        "linear",
    }
    has_rank_reduction = model_spec["input_mode"] == "Row-based (2D)"
    for layer_index, layer_spec in enumerate(layer_specs):
        location = f"AI generation.model_spec.layers[{layer_index}]"
        _require_exact_keys(layer_spec, layer_keys, location)
        layer_type = layer_spec["layer_type"]
        if layer_type not in SAFE_MODEL_LAYER_TYPES:
            raise AIProviderError(
                f"Unsupported safe model layer type: {layer_type}"
            )
        if (
            model_spec["input_mode"] == "Row-based (2D)"
            and layer_type not in row_allowed
        ):
            raise AIProviderError(
                f"{layer_type} cannot be used with row-based input."
            )
        layer_spec["units"] = _validated_positive_int(
            layer_spec["units"],
            location + ".units",
            maximum=4096,
        )
        layer_spec["filters"] = _validated_positive_int(
            layer_spec["filters"],
            location + ".filters",
            maximum=2048,
        )
        layer_spec["kernel_size"] = _validated_positive_int(
            layer_spec["kernel_size"],
            location + ".kernel_size",
            maximum=101,
        )
        layer_spec["strides"] = _validated_positive_int(
            layer_spec["strides"],
            location + ".strides",
            maximum=32,
        )
        layer_spec["pool_size"] = _validated_positive_int(
            layer_spec["pool_size"],
            location + ".pool_size",
            maximum=64,
        )
        layer_spec["dropout_rate"] = _validated_nonnegative_float(
            layer_spec["dropout_rate"],
            location + ".dropout_rate",
            maximum=0.95,
        )
        if layer_spec["activation"] not in activation_allowed:
            raise AIProviderError(
                f"Unsupported activation: {layer_spec['activation']}"
            )
        if layer_spec["padding"] not in {"same", "valid", "causal"}:
            raise AIProviderError(
                f"Unsupported padding: {layer_spec['padding']}"
            )
        if not isinstance(layer_spec["return_sequences"], bool):
            raise AIProviderError(
                f"{location}.return_sequences must be boolean."
            )
        if not isinstance(layer_spec["bidirectional"], bool):
            raise AIProviderError(
                f"{location}.bidirectional must be boolean."
            )

        if layer_type in {
            "Flatten",
            "GlobalAveragePooling1D",
            "GlobalMaxPooling1D",
        }:
            if has_rank_reduction:
                raise AIProviderError(
                    f"{location}: sequence rank has already been reduced."
                )
            has_rank_reduction = True
        elif layer_type in {"LSTM", "GRU"}:
            if has_rank_reduction:
                raise AIProviderError(
                    f"{location}: recurrent layers require sequence input."
                )
            if not layer_spec["return_sequences"]:
                has_rank_reduction = True
        elif layer_type in {
            "Conv1D",
            "MaxPooling1D",
            "AveragePooling1D",
        } and has_rank_reduction:
            raise AIProviderError(
                f"{location}: {layer_type} requires sequence input."
            )

    task_type = task["task_type"]
    if task_type == TASK_FORECASTING and (
        model_spec["input_mode"] != "Window-based (3D)"
    ):
        raise AIProviderError("Forecasting generation must be window-based.")
    if task_type == TASK_AUTOENCODER:
        if model_spec["input_mode"] == "Window-based (3D)" and has_rank_reduction:
            raise AIProviderError(
                "A window-based autoencoder must preserve the sequence "
                "dimension through all generated hidden layers."
            )
    elif model_spec["input_mode"] == "Window-based (3D)" and not has_rank_reduction:
        raise AIProviderError(
            "A supervised window model must reduce the sequence dimension "
            "before the task output layer."
        )

    return generation


def recommendation_dataset_limits(profile):
    rows = _validated_positive_int(profile.get("row_count", 0), "Dataset row count", maximum=10**15)
    return {"rows": rows, "filter_window_max": min(rows, 100001),
            "model_window_max": min(rows, 100000)}


def _normalize_unused_recommendation_controls(recommendation, profile):
    """Only unused controls receive defaults; active settings are never clamped."""
    plan = recommendation.get("filter")
    if isinstance(plan, dict) and plan.get("method") in SUPPORTED_FILTER_METHODS:
        active = {"Moving average": {"moving_average_window"},
                  "Exponential moving average": {"ema_span"}, "Median filter": {"median_window"},
                  "Simple Kalman filter": {"kalman_q", "kalman_r"}}.get(plan["method"], set())
        defaults = {"moving_average_window": 1, "ema_span": 1, "median_window": 1,
                    "kalman_q": 0.00001, "kalman_r": 0.01}
        changed = []
        if plan["method"] == "No filter" and "columns" in plan:
            if plan["columns"]: changed.append("columns")
            plan["columns"] = []
        for key, default in defaults.items():
            if key in plan and key not in active:
                if plan[key] != default: changed.append(key)
                plan[key] = default
        if changed and isinstance(plan.get("risks"), list):
            note = "Unused filter controls reset to neutral defaults: " + ", ".join(changed) + ". The selected filter is unchanged."
            if note not in plan["risks"]: plan["risks"].append(note)
    model = recommendation.get("model")
    task = recommendation.get("task", {}).get("task_type") if isinstance(recommendation.get("task"), dict) else None
    if isinstance(model, dict) and model.get("model_type") in ("DNN", "Dense Autoencoder") and task != TASK_FORECASTING:
        changed = any(key in model and model[key] != 1 for key in ("window_size", "stride"))
        for key in ("window_size", "stride"):
            if key in model: model[key] = 1
        if changed and isinstance(model.get("warnings"), list):
            note = "Window and stride are unused by this row-based model and were set to 1."
            if note not in model["warnings"]: model["warnings"].append(note)


def _check_recommendation_sample_fit(recommendation, profile):
    limits = recommendation_dataset_limits(profile)
    task, model, plan = recommendation["task"], recommendation["model"], recommendation["filter"]
    if set(plan["columns"]) - set(task["input_columns"]):
        raise AIProviderError("Filter columns must be chosen only from model inputs; do not filter answer columns.")
    uses_windows = model["model_type"] in ("CNN", "LSTM", "CNN-LSTM", "LSTM Autoencoder") or task["task_type"] == TASK_FORECASTING
    if uses_windows:
        window = model["window_size"]
        if window < 2:
            raise AIProviderError("AI recommendation.model.window_size must be at least 2 for sequence models.")
        horizon = int(profile.get("training_context", {}).get("forecast_horizon", 1)) if task["task_type"] == TASK_FORECASTING else 0
        upper_count = max(0, (limits["rows"] - window - horizon) // model["stride"] + 1)
        if upper_count < 3:
            raise AIProviderError(f"With {limits['rows']} rows, window_size={window}, stride={model['stride']} and "
                                  f"forecast_horizon={horizon}, at most {upper_count} windows remain. "
                                  "Choose a smaller window/stride or an appropriate row-based model; at least three windows are needed for data splitting.")
        note = "Window count is an upper bound. Missing values, label transitions and annotation boundaries can reduce it. Run Check my setup before training."
        if note not in model["warnings"]: model["warnings"].append(note)
    if limits["rows"] < 100:
        note = "This is a small dataset. A valid recommendation does not guarantee enough examples in every training, validation and test group."
        if note not in model["warnings"]: model["warnings"].append(note)
