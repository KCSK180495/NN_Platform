"""Inference for NN Training Studio."""

import numpy as np
import pandas as pd


def handle_missing_features_only(df, feature_cols, method):
    """Apply the saved missing-value rule when no evaluation label is used."""
    df = df.copy()

    if method == "Drop rows":
        df = df.dropna(subset=feature_cols)

    elif method == "Fill mean":
        for col in feature_cols:
            df[col] = df[col].fillna(df[col].mean())
        df = df.dropna(subset=feature_cols)

    elif method == "Fill median":
        for col in feature_cols:
            df[col] = df[col].fillna(df[col].median())
        df = df.dropna(subset=feature_cols)

    elif method == "Forward/backward fill":
        df[feature_cols] = df[feature_cols].ffill().bfill()
        df = df.dropna(subset=feature_cols)

    return df


def create_inference_windows(X, window_size, stride, y=None, row_ids=None):
    """Create model windows and retain their source-row positions."""
    X_windows = []
    y_windows = []
    start_rows = []
    end_rows = []

    if row_ids is None:
        row_ids = np.arange(len(X))

    for start in range(0, len(X) - window_size + 1, stride):
        end = start + window_size
        X_windows.append(X[start:end])
        start_rows.append(row_ids[start])
        end_rows.append(row_ids[end - 1])

        if y is not None:
            y_windows.append(pd.Series(y[start:end]).mode().iloc[0])

    X_windows = np.asarray(X_windows)
    y_result = None if y is None else np.asarray(y_windows)

    return (
        X_windows,
        y_result,
        np.asarray(start_rows),
        np.asarray(end_rows)
    )


def encode_external_labels(values, label_encoder):
    """Encode labels robustly, including equivalent string representations."""
    classes = list(label_encoder.classes_)
    direct_map = {value: index for index, value in enumerate(classes)}
    string_map = {str(value): index for index, value in enumerate(classes)}

    encoded = []
    unknown = []

    for value in values:
        if value in direct_map:
            encoded.append(direct_map[value])
        elif str(value) in string_map:
            encoded.append(string_map[str(value)])
        else:
            unknown.append(value)

    if unknown:
        unique_unknown = list(pd.unique(pd.Series(unknown)))
        raise ValueError(
            "The evaluation dataset contains labels not present during "
            f"training: {unique_unknown}"
        )

    return np.asarray(encoded, dtype=int)


def make_probability_column_name(index, class_name):
    safe_name = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in str(class_name)
    ).strip("_")
    if not safe_name:
        safe_name = f"class_{index}"
    return f"prob_{index}_{safe_name}"


def validate_model_input_shape(model, X_model):
    """Check the prepared external data against the loaded model input."""
    model_input_shape = model.input_shape

    if isinstance(model_input_shape, list):
        raise ValueError(
            "This evaluation page currently supports single-input Keras models."
        )

    expected = tuple(model_input_shape[1:])
    actual = tuple(X_model.shape[1:])

    if len(expected) != len(actual):
        raise ValueError(
            f"Model expects input shape {expected}, but the external data "
            f"produced shape {actual}."
        )

    for expected_size, actual_size in zip(expected, actual):
        if expected_size is not None and int(expected_size) != int(actual_size):
            raise ValueError(
                f"Model expects input shape {expected}, but the external data "
                f"produced shape {actual}."
            )
