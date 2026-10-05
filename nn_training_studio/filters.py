"""Filters for NN Training Studio."""

import numpy as np
import pandas as pd


def _clean_numeric_series(series):
    """
    Convert a signal column to numeric and fill temporary gaps before filtering.
    This prevents filters from breaking when a signal has small missing sections.
    """
    s = pd.to_numeric(series, errors="coerce")

    if s.isna().all():
        return s

    return s.ffill().bfill()


def moving_average_filter(series, window_size):
    s = _clean_numeric_series(series)

    return s.rolling(
        window=window_size,
        min_periods=1,
        center=True
    ).mean()


def exponential_moving_average_filter(series, span):
    s = _clean_numeric_series(series)

    return s.ewm(
        span=span,
        adjust=False
    ).mean()


def median_filter(series, window_size):
    s = _clean_numeric_series(series)

    return s.rolling(
        window=window_size,
        min_periods=1,
        center=True
    ).median()


def simple_kalman_filter(series, process_noise=1e-5, measurement_noise=1e-2):
    """
    Simple 1D Kalman filter for signal smoothing.
    Suitable for current/speed signals such as Ia, Ib, Ic, and speed.
    """
    s = _clean_numeric_series(series)

    if s.isna().all():
        return s

    values = s.values.astype(float)
    filtered = np.zeros_like(values)

    x_est = values[0]
    p_est = 1.0

    q = process_noise
    r = measurement_noise

    for i in range(len(values)):
        # Prediction
        x_pred = x_est
        p_pred = p_est + q

        # Update
        k_gain = p_pred / (p_pred + r)
        x_est = x_pred + k_gain * (values[i] - x_pred)
        p_est = (1 - k_gain) * p_pred

        filtered[i] = x_est

    return pd.Series(filtered, index=series.index)


def _apply_safe_filter_step(series, step, sampling_frequency=None):
    operation = step["operation"]
    numeric = pd.to_numeric(series, errors="coerce").ffill().bfill()
    if numeric.isna().any():
        raise ValueError(
            "The selected signal contains no usable numeric values."
        )

    if operation == "moving_average":
        return moving_average_filter(numeric, step["window_size"])
    if operation == "exponential_moving_average":
        return exponential_moving_average_filter(numeric, step["span"])
    if operation == "median":
        return median_filter(numeric, step["window_size"])
    if operation == "simple_kalman":
        return simple_kalman_filter(
            numeric,
            process_noise=step["process_noise"],
            measurement_noise=step["measurement_noise"],
        )

    try:
        from scipy.signal import (
            butter,
            filtfilt,
            iirnotch,
            lfilter,
            savgol_filter,
        )
    except ImportError as exc:
        raise ValueError(
            f"The safe {operation} filter requires scipy. "
            "Install it with: pip install scipy"
        ) from exc

    values = numeric.to_numpy(dtype=float)
    if operation == "savitzky_golay":
        window_size = int(step["window_size"])
        if len(values) < window_size:
            raise ValueError(
                f"Savitzky-Golay window {window_size} exceeds the signal "
                f"length {len(values)}."
            )
        filtered = savgol_filter(
            values,
            window_length=window_size,
            polyorder=int(step["polyorder"]),
            mode="interp",
        )
        return pd.Series(filtered, index=series.index)

    if sampling_frequency is None:
        raise ValueError(f"Sampling frequency is required for {operation}.")
    fs = float(sampling_frequency)
    if operation == "notch":
        b_coeff, a_coeff = iirnotch(
            float(step["notch_hz"]),
            float(step["quality_factor"]),
            fs=fs,
        )
    else:
        filter_kind = {
            "butterworth_lowpass": "lowpass",
            "butterworth_highpass": "highpass",
            "butterworth_bandpass": "bandpass",
        }[operation]
        critical_frequency = (
            [float(step["lowcut_hz"]), float(step["highcut_hz"])]
            if filter_kind == "bandpass"
            else float(step["cutoff_hz"])
        )
        b_coeff, a_coeff = butter(
            int(step["order"]),
            critical_frequency,
            btype=filter_kind,
            fs=fs,
        )

    if step["causal"]:
        filtered = lfilter(b_coeff, a_coeff, values)
    else:
        padding_length = 3 * (max(len(a_coeff), len(b_coeff)) - 1)
        if len(values) <= padding_length:
            raise ValueError(
                f"Signal length {len(values)} is too short for zero-phase "
                f"{operation}; at least {padding_length + 1} samples are needed."
            )
        filtered = filtfilt(b_coeff, a_coeff, values)
    return pd.Series(filtered, index=series.index)


def apply_safe_filter_spec(
    df,
    filter_spec,
    sampling_frequency=None,
    allowed_columns=None,
):
    """Apply an already validated declarative filter specification."""
    result = df.copy()
    original_columns = list(result.columns)
    allowed = set(allowed_columns or original_columns)

    for pipeline in filter_spec.get("pipelines", []):
        column = pipeline["column"]
        if column not in result.columns:
            raise ValueError(
                f"Generated filter column is missing from the dataset: {column}"
            )
        if column not in allowed:
            raise ValueError(
                f"Generated filter is not permitted to modify: {column}"
            )
        filtered_series = result[column]
        for step in pipeline["steps"]:
            filtered_series = _apply_safe_filter_step(
                filtered_series,
                step,
                sampling_frequency=sampling_frequency,
            )
        result[column] = np.asarray(filtered_series, dtype=float)

    if len(result) != len(df):
        raise ValueError("Safe filter changed the number of rows.")
    if list(result.columns) != original_columns:
        raise ValueError("Safe filter changed the dataset columns or order.")
    return result


def compare_filter_result(before_df, after_df, columns):
    """Return compact preservation metrics for a generated filter preview."""
    rows = []
    for column in columns:
        before = pd.to_numeric(before_df[column], errors="coerce").to_numpy(
            dtype=float
        )
        after = pd.to_numeric(after_df[column], errors="coerce").to_numpy(
            dtype=float
        )
        valid = np.isfinite(before) & np.isfinite(after)
        if not np.any(valid):
            continue
        before = before[valid]
        after = after[valid]
        correlation = (
            float(np.corrcoef(before, after)[0, 1])
            if len(before) > 1
            and np.std(before) > 0
            and np.std(after) > 0
            else 1.0
        )
        rmse = float(np.sqrt(np.mean((after - before) ** 2)))
        original_std = float(np.std(before))
        rows.append({
            "column": column,
            "correlation": correlation,
            "rmse": rmse,
            "std_ratio": (
                float(np.std(after) / original_std)
                if original_std > 0
                else 1.0
            ),
            "mean_absolute_change": float(np.mean(np.abs(after - before))),
        })
    return rows


def apply_selected_filter(
    df,
    selected_filter_cols,
    filter_method,
    moving_window,
    ema_span,
    median_window,
    kalman_q,
    kalman_r
):
    """
    Apply the selected filter to selected numeric columns only.
    Label, source, and ID columns should not be filtered.
    """
    filtered_df = df.copy()

    if filter_method == "No filter":
        return filtered_df

    for col in selected_filter_cols:
        filtered_df[col] = pd.to_numeric(filtered_df[col], errors="coerce")

        if filter_method == "Moving average":
            filtered_df[col] = moving_average_filter(
                filtered_df[col],
                moving_window
            )

        elif filter_method == "Exponential moving average":
            filtered_df[col] = exponential_moving_average_filter(
                filtered_df[col],
                ema_span
            )

        elif filter_method == "Median filter":
            filtered_df[col] = median_filter(
                filtered_df[col],
                median_window
            )

        elif filter_method == "Simple Kalman filter":
            filtered_df[col] = simple_kalman_filter(
                filtered_df[col],
                process_noise=kalman_q,
                measurement_noise=kalman_r
            )

    return filtered_df


def execute_custom_filter_code(df, selected_columns, custom_code):
    """Execute the user-defined custom_filter function and validate its result."""
    if not custom_code or not custom_code.strip():
        raise ValueError("Custom filter code is empty.")

    namespace = {
        "pd": pd,
        "np": np,
    }

    try:
        compiled_code = compile(custom_code, "<custom_filter>", "exec")
        exec(compiled_code, namespace)
    except Exception as exc:
        raise ValueError(f"Custom filter code could not be loaded: {exc}") from exc

    filter_function = namespace.get("custom_filter")

    if not callable(filter_function):
        raise ValueError(
            "Custom filter code must define: custom_filter(df, selected_columns)"
        )

    try:
        result = filter_function(df.copy(), list(selected_columns))
    except Exception as exc:
        raise ValueError(f"Custom filter execution failed: {exc}") from exc

    if not isinstance(result, pd.DataFrame):
        raise ValueError("custom_filter must return a pandas DataFrame.")

    if len(result) != len(df):
        raise ValueError(
            "Custom filter must keep the same number of rows as the input dataset."
        )

    missing_columns = [col for col in df.columns if col not in result.columns]
    if missing_columns:
        raise ValueError(
            "Custom filter removed required columns: " + str(missing_columns)
        )

    # Preserve the original column order. Extra generated columns are appended.
    ordered_columns = list(df.columns) + [
        col for col in result.columns if col not in df.columns
    ]
    result = result.loc[:, ordered_columns].copy()
    result.index = df.index

    return result
