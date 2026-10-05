"""Preprocessing for NN Training Studio."""

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from nn_training_studio.constants import (
    DATA_MODE_IMAGE,
    MODEL_TYPE_SAFE_AI_LEGACY,
    MODEL_TYPE_SAFE_CUSTOM,
    TASK_AUTOENCODER,
    TASK_CLASSIFICATION,
    TASK_FORECASTING,
    TASK_MULTI_OUTPUT,
    TASK_REGRESSION,
)
from nn_training_studio.image_data import (
    split_image_records,
)


def convert_features_to_numeric(df, feature_cols):
    df = df.copy()

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def handle_missing_values(df, feature_cols, label_col, method):
    """Backward-compatible wrapper for a single target column."""
    return handle_missing_values_multi(
        df,
        feature_cols,
        [] if label_col is None else [label_col],
        method
    )


def handle_missing_values_multi(df, feature_cols, target_cols, method):
    df = df.copy()
    target_cols = list(target_cols)
    selected_cols = list(dict.fromkeys(list(feature_cols) + target_cols))

    if method == "Drop rows":
        df = df.dropna(subset=selected_cols)

    elif method == "Fill mean":
        for col in feature_cols:
            df[col] = df[col].fillna(df[col].mean())
        df = df.dropna(subset=target_cols)

    elif method == "Fill median":
        for col in feature_cols:
            df[col] = df[col].fillna(df[col].median())
        df = df.dropna(subset=target_cols)

    elif method == "Forward/backward fill":
        df[selected_cols] = df[selected_cols].ffill().bfill()
        df = df.dropna(subset=selected_cols)

    return df


def create_windows(X, y, window_size, stride):
    """Create classification windows using the modal label."""
    X_windows = []
    y_windows = []

    for start in range(0, len(X) - window_size + 1, stride):
        end = start + window_size

        X_window = X[start:end]
        y_window = y[start:end]

        label = pd.Series(y_window).mode()[0]

        X_windows.append(X_window)
        y_windows.append(label)

    return np.array(X_windows), np.array(y_windows)


def create_supervised_windows(
    X,
    y,
    window_size,
    stride,
    task_type,
    forecast_horizon=1,
    group_ids=None,
):
    """
    Create task-aware windows.

    Classification uses the modal label in each input window. Regression and
    multi-output regression use the target at the final input row. Forecasting
    uses the rows immediately after the input window. Autoencoders reconstruct
    the complete input window.
    """
    X_windows = []
    y_windows = []

    if task_type == TASK_FORECASTING:
        final_start = len(X) - window_size - forecast_horizon + 1
    else:
        final_start = len(X) - window_size + 1

    for start in range(0, max(0, final_start), stride):
        end = start + window_size
        if group_ids is not None:
            group_end = (
                end + forecast_horizon
                if task_type == TASK_FORECASTING
                else end
            )
            group_window = np.asarray(group_ids[start:group_end])
            if (
                len(group_window) != group_end - start
                or len(set(map(str, group_window))) != 1
            ):
                continue
        X_window = X[start:end]

        if task_type == TASK_CLASSIFICATION:
            target = pd.Series(np.asarray(y[start:end]).reshape(-1)).mode().iloc[0]
        elif task_type in (TASK_REGRESSION, TASK_MULTI_OUTPUT):
            target = np.asarray(y[end - 1])
        elif task_type == TASK_FORECASTING:
            target = np.asarray(y[end:end + forecast_horizon]).reshape(-1)
        elif task_type == TASK_AUTOENCODER:
            target = X_window.copy()
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        X_windows.append(X_window)
        y_windows.append(target)

    return np.asarray(X_windows), np.asarray(y_windows)


def _test_partition_count(sample_count, test_size, task_type):
    if not np.isfinite(test_size) or not 0 < test_size < 1:
        raise ValueError("Test fraction must be strictly between 0 and 1.")
    if task_type == TASK_FORECASTING:
        split_index = int(round(sample_count * (1.0 - test_size)))
        split_index = min(max(1, split_index), sample_count - 1)
        return sample_count - split_index
    return int(np.ceil(sample_count * test_size))


def _validation_training_count(train_pool_count, validation_split):
    if not np.isfinite(validation_split) or not 0 < validation_split < 1:
        raise ValueError("Validation fraction must be strictly between 0 and 1.")
    # Match Keras' array validation_split boundary without clamping an invalid
    # split into a seemingly valid preflight result.
    training_count = int(np.floor(train_pool_count * (1.0 - validation_split)))
    if training_count < 1 or training_count >= train_pool_count:
        raise ValueError(
            "The current split percentages leave an empty training or "
            "validation partition. Load more data or reduce the percentages."
        )
    return training_count


def safe_train_test_split(X, y, test_size, task_type=TASK_CLASSIFICATION):
    if len(X) < 2:
        raise ValueError(
            "At least two prepared samples/windows are required for "
            "training and testing."
        )

    test_count = _test_partition_count(len(X), test_size, task_type)
    if task_type == TASK_FORECASTING:
        split_index = len(X) - test_count
        return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

    if task_type != TASK_CLASSIFICATION:
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            shuffle=True
        )

    unique, counts = np.unique(y, return_counts=True)
    train_count = len(X) - test_count

    # Stratification also requires both partitions to have room for every
    # class. Checking only the smallest class count lets sklearn fail when the
    # requested test set is smaller than the number of classes.
    if (
        len(unique) > 1
        and counts.min() >= 2
        and test_count >= len(unique)
        and train_count >= len(unique)
    ):
        return train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        shuffle=True
    )


def summarize_training_preflight(df, settings):
    """Validate sample counts without building a model or starting a thread."""
    if settings.get("data_mode") == DATA_MODE_IMAGE:
        train_records, validation_records, test_records = split_image_records(
            df,
            test_fraction=float(settings["test_size"]),
            validation_fraction=float(settings["validation_split"]),
            random_seed=42,
        )
        return {
            "source_rows": int(len(df)),
            "prepared_samples": int(len(df)),
            "training_samples": int(len(train_records)),
            "validation_samples": int(len(validation_records)),
            "test_samples": int(len(test_records)),
            "uses_windows": False,
        }

    feature_cols = list(settings.get("feature_cols") or [])
    target_cols = list(settings.get("target_cols") or [])
    if not feature_cols:
        raise ValueError("Select at least one model input column.")
    missing_columns = [
        column
        for column in feature_cols + target_cols
        if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "Training columns are missing from the prepared dataset: "
            + str(sorted(set(missing_columns)))
        )

    prepared = convert_features_to_numeric(df.copy(), feature_cols)
    if settings["task_type"] in (
        TASK_REGRESSION,
        TASK_MULTI_OUTPUT,
        TASK_FORECASTING,
    ):
        for column in target_cols:
            prepared[column] = pd.to_numeric(
                prepared[column], errors="coerce"
            )
    prepared = handle_missing_values_multi(
        df=prepared,
        feature_cols=feature_cols,
        target_cols=target_cols,
        method=settings["missing_method"],
    )
    if prepared.empty:
        raise ValueError(
            "No rows remain after converting inputs and handling missing values."
        )

    model_type = settings["model_type"]
    custom_uses_windows = (
        model_type in (
            "Custom Python Model",
            MODEL_TYPE_SAFE_CUSTOM,
            MODEL_TYPE_SAFE_AI_LEGACY,
        )
        and settings["custom_model_input_mode"] == "Window-based (3D)"
    )
    uses_windows = (
        settings["task_type"] == TASK_FORECASTING
        or model_type in ("CNN", "LSTM", "CNN-LSTM", "LSTM Autoencoder")
        or custom_uses_windows
    )
    if uses_windows:
        window_size = int(settings["window_size"])
        forecast_horizon = (
            int(settings["forecast_horizon"])
            if settings["task_type"] == TASK_FORECASTING
            else 0
        )
        final_start = len(prepared) - window_size - forecast_horizon + 1
        group_ids = (
            prepared["annotation_segment_id"].to_numpy()
            if "annotation_segment_id" in prepared.columns
            else None
        )
        if final_start <= 0:
            prepared_count = 0
        elif group_ids is None:
            prepared_count = ((final_start - 1) // int(settings["stride"])) + 1
        else:
            prepared_count = 0
            for start in range(0, final_start, int(settings["stride"])):
                end = start + window_size + forecast_horizon
                values = group_ids[start:end]
                if (
                    len(values) == end - start
                    and len(set(map(str, values))) == 1
                ):
                    prepared_count += 1
    else:
        prepared_count = len(prepared)

    if prepared_count < 3:
        requirement = (
            "Reduce the window size/forecast horizon or load more rows."
            if uses_windows
            else "Load more usable rows."
        )
        raise ValueError(
            "Training needs at least three prepared samples/windows so train, "
            f"validation, and test data can all be created. {requirement}"
        )

    test_count = _test_partition_count(
        prepared_count, float(settings["test_size"]), settings["task_type"]
    )
    train_pool = prepared_count - test_count
    training_count = _validation_training_count(
        train_pool, float(settings["validation_split"])
    )
    validation_count = train_pool - training_count
    if training_count < 1 or validation_count < 1 or test_count < 1:
        raise ValueError(
            "The current test and validation percentages leave an empty data "
            "partition. Load more data or reduce the split percentages."
        )
    return {
        "source_rows": int(len(df)),
        "usable_rows": int(len(prepared)),
        "prepared_samples": int(prepared_count),
        "training_samples": int(training_count),
        "validation_samples": int(validation_count),
        "test_samples": int(test_count),
        "uses_windows": bool(uses_windows),
    }


def fit_optional_scaler(values, scaler_option):
    if scaler_option == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif scaler_option == "StandardScaler":
        scaler = StandardScaler()
    else:
        return None, np.asarray(values, dtype=float)

    values = np.asarray(values, dtype=float)
    original_shape = values.shape
    values_2d = values.reshape(-1, original_shape[-1] if values.ndim > 1 else 1)
    transformed = scaler.fit_transform(values_2d).reshape(original_shape)
    return scaler, transformed


def inverse_transform_targets(values, target_scaler):
    values = np.asarray(values)
    if target_scaler is None:
        return values

    original_shape = values.shape
    feature_count = int(getattr(target_scaler, "n_features_in_", 1))
    restored = target_scaler.inverse_transform(
        values.reshape(-1, feature_count)
    )
    return restored.reshape(original_shape)


def prepare_scaled_training_data(
    X_pool, X_test, y_pool, y_test, scaler_option, target_scaler_option,
    task_type, validation_split, target_feature_count,
):
    """Hold out validation before fitting feature and target normalization.

    Keeps the existing last-fraction validation membership and per-feature
    scaler format used by saved-model inference, including forecast horizons.
    Window overlap and earlier imputation are separate preprocessing concerns.
    """
    boundary = _validation_training_count(len(X_pool), validation_split)
    X_train, X_validation = X_pool[:boundary], X_pool[boundary:]
    y_train, y_validation = y_pool[:boundary], y_pool[boundary:]
    scaler, X_train = fit_optional_scaler(X_train, scaler_option)

    def transform(values, fitted_scaler):
        values = np.asarray(values, dtype=float)
        if fitted_scaler is None:
            return values.copy()
        return fitted_scaler.transform(
            values.reshape(-1, fitted_scaler.n_features_in_)
        ).reshape(values.shape)

    X_validation = transform(X_validation, scaler)
    X_test = transform(X_test, scaler)
    target_scaler = None
    if task_type == TASK_AUTOENCODER:
        y_train, y_validation, y_test = (
            X_train.copy(), X_validation.copy(), X_test.copy()
        )
    elif task_type != TASK_CLASSIFICATION:
        if target_feature_count < 1:
            raise ValueError("At least one numeric target column is required.")
        original_shape = np.asarray(y_train).shape
        target_scaler, normalized = fit_optional_scaler(
            np.asarray(y_train).reshape(-1, target_feature_count),
            target_scaler_option,
        )
        y_train = normalized.reshape(original_shape)
        y_validation = transform(y_validation, target_scaler)
        y_test = transform(y_test, target_scaler)
    return {
        "X_train": X_train, "X_validation": X_validation, "X_test": X_test,
        "y_train": y_train, "y_validation": y_validation, "y_test": y_test,
        "scaler": scaler, "target_scaler": target_scaler,
    }


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_true_2d = y_true.reshape(len(y_true), -1)
    y_pred_2d = y_pred.reshape(len(y_pred), -1)
    mse = mean_squared_error(y_true_2d, y_pred_2d)
    r2 = (
        float(r2_score(
            y_true_2d,
            y_pred_2d,
            multioutput="uniform_average"
        ))
        if len(y_true_2d) >= 2
        else float("nan")
    )
    return {
        "mae": float(mean_absolute_error(
            y_true_2d,
            y_pred_2d
        )),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": r2,
    }
