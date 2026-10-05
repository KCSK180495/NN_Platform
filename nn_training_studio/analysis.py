"""Analysis for NN Training Studio."""

from datetime import datetime
import numpy as np
import pandas as pd
from nn_training_studio.ai_validation import (
    _validated_positive_int,
)
from nn_training_studio.constants import (
    ANALYSIS_MAX_ROWS,
    ID_NAME_TOKENS,
    TARGET_NAME_TOKENS,
    TASK_AUTOENCODER,
    TASK_CLASSIFICATION,
    TASK_FORECASTING,
    TASK_MULTI_OUTPUT,
    TASK_REGRESSION,
    TIME_NAME_TOKENS,
)


def _contiguous_signal_statistics(series, sampling_frequency=None):
    """Time/frequency statistics require consecutive finite samples."""
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(values)
    edges = np.diff(np.r_[False, finite, False].astype(int))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if len(starts):
        index = int(np.argmax(stops - starts))
        values = values[starts[index]:stops[index]]
    else:
        values = np.array([], dtype=float)
    return _numeric_signal_statistics(pd.Series(values), sampling_frequency)


def _safe_number(value, default=None):
    """Convert NumPy/Pandas numeric values into JSON-safe Python numbers."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    if not np.isfinite(number):
        return default
    return number


def _column_name_tokens(column_name):
    normalized = "".join(
        character.lower() if character.isalnum() else " "
        for character in str(column_name)
    )
    return [token for token in normalized.split() if token]


def _name_has_token(column_name, candidate_tokens):
    tokens = _column_name_tokens(column_name)
    return any(
        token == candidate
        or (len(candidate) >= 4 and token.startswith(candidate))
        for token in tokens
        for candidate in candidate_tokens
    )


def _numeric_signal_statistics(series, sampling_frequency=None):
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()

    result = {
        "mean": None,
        "std": None,
        "minimum": None,
        "maximum": None,
        "outlier_percent": 0.0,
        "extreme_repeat_percent": 0.0,
        "lag1_autocorrelation": None,
        "trend_strength": None,
        "roughness_ratio": None,
        "dominant_frequency_normalized": None,
        "dominant_frequency_hz": None,
        "spectral_peak_ratio": None,
    }

    if numeric.empty:
        return result

    values = numeric.to_numpy(dtype=float)
    result["mean"] = _safe_number(np.mean(values))
    result["std"] = _safe_number(np.std(values))
    result["minimum"] = _safe_number(np.min(values))
    result["maximum"] = _safe_number(np.max(values))

    if len(values) >= 4:
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            result["outlier_percent"] = float(
                100.0 * np.mean((values < lower) | (values > upper))
            )

        minimum_count = int(np.sum(values == np.min(values)))
        maximum_count = int(np.sum(values == np.max(values)))
        result["extreme_repeat_percent"] = float(
            100.0 * max(minimum_count, maximum_count) / len(values)
        )

    signal_std = float(np.std(values))
    if len(values) >= 3 and signal_std > 0:
        lag_correlation = np.corrcoef(values[:-1], values[1:])[0, 1]
        result["lag1_autocorrelation"] = _safe_number(lag_correlation)

        index_values = np.arange(len(values), dtype=float)
        trend_correlation = np.corrcoef(index_values, values)[0, 1]
        result["trend_strength"] = _safe_number(abs(trend_correlation))

        difference_std = float(np.std(np.diff(values)))
        result["roughness_ratio"] = _safe_number(
            difference_std / (signal_std + 1e-12)
        )

    if len(values) >= 16 and signal_std > 0:
        # Limit spectral work so analysis remains responsive on large CSV files.
        spectral_values = values[: min(len(values), 16384)]
        spectral_values = spectral_values - np.mean(spectral_values)
        spectrum = np.abs(np.fft.rfft(spectral_values)) ** 2
        if len(spectrum) > 1:
            spectrum[0] = 0.0
            total_power = float(np.sum(spectrum))
            peak_index = int(np.argmax(spectrum))
            if total_power > 0:
                result["spectral_peak_ratio"] = _safe_number(
                    float(spectrum[peak_index]) / total_power
                )
            normalized_frequency = peak_index / len(spectral_values)
            result["dominant_frequency_normalized"] = _safe_number(
                normalized_frequency
            )
            if sampling_frequency is not None:
                result["dominant_frequency_hz"] = _safe_number(
                    normalized_frequency * sampling_frequency
                )

    return result


def _recommend_filter_for_column(column_name, column_profile):
    if not column_profile.get("is_numeric"):
        return None

    if column_profile.get("unique_count", 0) <= 2:
        return {
            "column": column_name,
            "filter": "No filter",
            "reason": "Discrete or near-binary columns should not be smoothed.",
        }

    outlier_percent = column_profile.get("outlier_percent") or 0.0
    clipping_percent = column_profile.get("extreme_repeat_percent") or 0.0
    roughness_ratio = column_profile.get("roughness_ratio")
    lower_name = str(column_name).lower()

    if clipping_percent >= 10.0:
        return {
            "column": column_name,
            "filter": "Inspect sensor range before filtering",
            "reason": (
                f"{clipping_percent:.2f}% of analysed values repeat at one "
                "extreme, which may indicate clipping or saturation."
            ),
        }

    if outlier_percent >= 2.0:
        return {
            "column": column_name,
            "filter": "Median filter",
            "reason": (
                f"{outlier_percent:.2f}% IQR outliers suggest isolated spikes. "
                "Compare the filtered spectrum before applying it to all data."
            ),
        }

    if (
        roughness_ratio is not None
        and roughness_ratio >= 0.75
        and any(token in lower_name for token in ("speed", "temp", "pressure"))
    ):
        return {
            "column": column_name,
            "filter": "Simple Kalman filter",
            "reason": (
                "This slowly varying measurement has relatively strong "
                "sample-to-sample variation."
            ),
        }

    if roughness_ratio is not None and roughness_ratio >= 1.20:
        return {
            "column": column_name,
            "filter": "Moving average (candidate only)",
            "reason": (
                "High sample-to-sample roughness was detected. Validate that "
                "fault harmonics and transient edges are preserved."
            ),
        }

    return {
        "column": column_name,
        "filter": "No filter initially",
        "reason": (
            "No strong spike, clipping, or high-roughness indicator was "
            "detected by the local profile."
        ),
    }


def _recommend_task(column_profiles, row_count, time_columns):
    candidate_rows = []

    for position, profile in enumerate(column_profiles):
        name = profile["name"]
        unique_count = profile["unique_count"]
        unique_ratio = profile["unique_ratio"]
        score = 0.0
        reasons = []

        if _name_has_token(name, TARGET_NAME_TOKENS):
            score += 4.0
            reasons.append("column name suggests a target or label")

        if position == len(column_profiles) - 1:
            score += 0.5
            reasons.append("it is the last dataset column")

        if profile["is_id_like"]:
            score -= 5.0

        low_cardinality_limit = min(50, max(10, int(max(row_count, 1) * 0.05)))
        if unique_count <= low_cardinality_limit:
            score += 1.0
            reasons.append("it contains a limited set of repeated values")
        elif unique_ratio >= 0.90 and not _name_has_token(
            name,
            TARGET_NAME_TOKENS
        ):
            score -= 1.0

        if score > 0:
            candidate_rows.append(
                {
                    "name": name,
                    "score": score,
                    "reasons": reasons,
                    "is_numeric": profile["is_numeric"],
                    "unique_count": unique_count,
                    "unique_ratio": unique_ratio,
                    "lag1_autocorrelation": profile.get(
                        "lag1_autocorrelation"
                    ),
                }
            )

    candidate_rows.sort(key=lambda item: item["score"], reverse=True)
    strong_candidates = [
        candidate for candidate in candidate_rows
        if candidate["score"] >= 3.0
    ]

    profile_by_name = {
        profile["name"]: profile for profile in column_profiles
    }
    target_columns = []
    reasons = []
    warnings = [
        "Task inference is advisory. Confirm the intended prediction target "
        "before preprocessing or training."
    ]
    alternatives = []

    explicitly_named = [
        candidate for candidate in strong_candidates
        if _name_has_token(candidate["name"], TARGET_NAME_TOKENS)
    ]

    forecast_named = [
        candidate for candidate in explicitly_named
        if _name_has_token(
            candidate["name"],
            ("future", "next", "forecast", "horizon")
        )
    ]

    if forecast_named and time_columns and all(
        candidate["is_numeric"] for candidate in forecast_named
    ):
        task_type = TASK_FORECASTING
        target_columns = [
            candidate["name"] for candidate in forecast_named
        ]
        confidence = 0.86
        reasons.append(
            "Time/order data and explicitly named future-value targets were detected."
        )
        alternatives = [TASK_REGRESSION, TASK_MULTI_OUTPUT]
    elif len(explicitly_named) >= 2 and all(
        candidate["is_numeric"] for candidate in explicitly_named
    ):
        task_type = TASK_MULTI_OUTPUT
        target_columns = [
            candidate["name"] for candidate in explicitly_named
        ]
        confidence = min(0.92, 0.70 + 0.05 * len(target_columns))
        reasons.append(
            "Several numeric columns have names associated with targets or outputs."
        )
        alternatives = [TASK_REGRESSION, TASK_FORECASTING]
    elif candidate_rows and candidate_rows[0]["score"] >= 1.5:
        top_candidate = candidate_rows[0]
        target_columns = [top_candidate["name"]]
        target_profile = profile_by_name[top_candidate["name"]]
        low_cardinality_limit = min(
            50,
            max(10, int(max(row_count, 1) * 0.05))
        )
        name_is_forecast = _name_has_token(
            top_candidate["name"],
            ("future", "next", "forecast", "horizon")
        )

        if (
            name_is_forecast
            and target_profile["is_numeric"]
            and time_columns
        ):
            task_type = TASK_FORECASTING
            confidence = 0.82
            reasons.append(
                "A time/order column and a future-value target name were detected."
            )
            alternatives = [TASK_REGRESSION]
        elif (
            not target_profile["is_numeric"]
            or target_profile["unique_count"] <= low_cardinality_limit
        ):
            task_type = TASK_CLASSIFICATION
            confidence = 0.88 if top_candidate["score"] >= 4.0 else 0.68
            reasons.append(
                f"'{top_candidate['name']}' has "
                f"{target_profile['unique_count']} repeated discrete values."
            )
            alternatives = [TASK_AUTOENCODER]
        else:
            task_type = TASK_REGRESSION
            confidence = 0.82 if top_candidate["score"] >= 4.0 else 0.62
            reasons.append(
                f"'{top_candidate['name']}' is a continuous numeric target candidate."
            )
            alternatives = (
                [TASK_FORECASTING, TASK_MULTI_OUTPUT]
                if time_columns
                else [TASK_MULTI_OUTPUT]
            )
    else:
        task_type = TASK_AUTOENCODER
        confidence = 0.58
        reasons.append(
            "No sufficiently clear target column was detected, so an "
            "unsupervised reconstruction task is the safest initial suggestion."
        )
        alternatives = (
            [TASK_FORECASTING, TASK_CLASSIFICATION, TASK_REGRESSION]
            if time_columns
            else [TASK_CLASSIFICATION, TASK_REGRESSION]
        )

    excluded = set(target_columns)
    input_columns = [
        profile["name"]
        for profile in column_profiles
        if profile["is_numeric"]
        and not profile["is_id_like"]
        and not profile["is_time_like"]
        and profile["name"] not in excluded
        and profile["unique_count"] > 1
    ]

    if not input_columns:
        warnings.append(
            "No usable numeric input columns were detected automatically."
        )

    return {
        "task_type": task_type,
        "confidence": float(confidence),
        "input_columns": input_columns,
        "target_columns": target_columns,
        "reasons": reasons,
        "warnings": warnings,
        "alternatives": alternatives,
        "target_candidates": candidate_rows[:10],
    }


def analyze_dataset_profile(
    df,
    sampling_frequency=None,
    max_rows=ANALYSIS_MAX_ROWS
):
    """
    Build a compact, JSON-safe profile suitable for local review or a later
    structured AI API request. Raw dataset rows are not included.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Dataset analysis requires a pandas DataFrame.")
    if df.empty:
        raise ValueError("The dataset is empty.")
    if sampling_frequency is not None and (not np.isfinite(float(sampling_frequency)) or float(sampling_frequency) <= 0):
        raise ValueError("Sampling frequency must be greater than zero.")

    max_rows = _validated_positive_int(max_rows, "Analysis sample limit", maximum=1000000)
    row_count, column_count = df.shape
    # Use a real consecutive block for frequency/lag statistics. Randomly sampled
    # rows cannot be treated as adjacent samples at the original sample rate.
    signal_start = max(0, (row_count - min(row_count, max_rows)) // 2)
    signal_df = df.iloc[signal_start:signal_start + max_rows]
    if row_count > max_rows:
        analysed_df = df.sample(
            n=max_rows,
            random_state=42
        ).sort_index()
    else:
        analysed_df = df

    column_profiles = []
    time_columns = []

    for column_name in df.columns:
        full_series = df[column_name]
        sample_series = analysed_df[column_name]
        non_missing = sample_series.dropna()
        numeric_values = pd.to_numeric(sample_series, errors="coerce")
        numeric_success = (
            float(numeric_values.notna().sum()) / max(1, len(non_missing))
        )
        is_numeric = (
            pd.api.types.is_numeric_dtype(full_series)
            or numeric_success >= 0.95
        )
        unique_count = int(non_missing.nunique(dropna=True))
        is_time_like = _name_has_token(column_name, TIME_NAME_TOKENS)
        is_id_like = (
            _name_has_token(column_name, ID_NAME_TOKENS)
            and not _name_has_token(column_name, TARGET_NAME_TOKENS)
        )

        if is_time_like:
            time_columns.append(str(column_name))

        profile = {
            "name": str(column_name),
            "dtype": str(full_series.dtype),
            "is_numeric": bool(is_numeric),
            "is_time_like": bool(is_time_like),
            "is_id_like": bool(is_id_like),
            "missing_count": int(full_series.isna().sum()),
            "missing_percent": float(
                100.0 * full_series.isna().sum() / max(1, row_count)
            ),
            "unique_count": unique_count,
            "unique_ratio": float(unique_count / max(1, len(non_missing))),
        }

        if is_numeric:
            profile.update(
                _numeric_signal_statistics(
                    numeric_values,
                    sampling_frequency=sampling_frequency
                )
            )

        if is_numeric:
            sequence_stats = _contiguous_signal_statistics(signal_df[column_name], sampling_frequency)
            for key in ("lag1_autocorrelation", "trend_strength", "roughness_ratio",
                        "dominant_frequency_normalized", "dominant_frequency_hz", "spectral_peak_ratio"):
                profile[key] = sequence_stats[key]
        column_profiles.append(profile)

    numeric_profiles = [
        profile for profile in column_profiles if profile["is_numeric"]
    ]
    categorical_columns = [
        profile["name"]
        for profile in column_profiles
        if not profile["is_numeric"]
    ]
    constant_columns = [
        profile["name"]
        for profile in column_profiles
        if profile["unique_count"] <= 1
    ]

    correlation_warnings = []
    numeric_names = [
        profile["name"]
        for profile in numeric_profiles
        if profile["unique_count"] > 1
    ][:40]
    if len(numeric_names) >= 2:
        numeric_frame = analysed_df[numeric_names].apply(
            pd.to_numeric,
            errors="coerce"
        )
        correlation = numeric_frame.corr().abs()
        for left_index, left_name in enumerate(numeric_names):
            for right_index in range(left_index + 1, len(numeric_names)):
                right_name = numeric_names[right_index]
                value = _safe_number(
                    correlation.loc[left_name, right_name]
                )
                if value is not None and value >= 0.98:
                    correlation_warnings.append(
                        {
                            "columns": [left_name, right_name],
                            "absolute_correlation": value,
                        }
                    )
        correlation_warnings.sort(
            key=lambda item: item["absolute_correlation"],
            reverse=True
        )

    recommendation = _recommend_task(
        column_profiles,
        row_count=row_count,
        time_columns=time_columns
    )
    target_summaries = []
    for target_name in recommendation["target_columns"]:
        if target_name not in analysed_df.columns:
            continue
        target_profile = next(
            (
                item for item in column_profiles
                if item["name"] == target_name
            ),
            None
        )
        if target_profile is None:
            continue

        target_series = analysed_df[target_name].dropna()
        if recommendation["task_type"] == TASK_CLASSIFICATION:
            class_counts = target_series.value_counts(dropna=True)
            class_distribution = [
                {
                    "value": str(class_value),
                    "count": int(class_count),
                    "percent": float(
                        100.0 * class_count / max(1, class_counts.sum())
                    ),
                }
                for class_value, class_count in class_counts.head(25).items()
            ]
            imbalance_ratio = (
                float(class_counts.min() / class_counts.max())
                if len(class_counts) >= 2 and class_counts.max() > 0
                else None
            )
            target_summaries.append(
                {
                    "column": target_name,
                    "kind": "categorical",
                    "class_distribution": class_distribution,
                    "minority_to_majority_ratio": imbalance_ratio,
                }
            )
            if imbalance_ratio is not None and imbalance_ratio < 0.50:
                recommendation["warnings"].append(
                    f"Target '{target_name}' appears imbalanced "
                    f"(minority/majority ratio={imbalance_ratio:.3f})."
                )
        elif target_profile["is_numeric"]:
            numeric_target = pd.to_numeric(
                target_series,
                errors="coerce"
            ).dropna()
            if not numeric_target.empty:
                quantiles = numeric_target.quantile(
                    [0.0, 0.25, 0.5, 0.75, 1.0]
                )
                target_summaries.append(
                    {
                        "column": target_name,
                        "kind": "numeric",
                        "minimum": _safe_number(quantiles.loc[0.0]),
                        "q1": _safe_number(quantiles.loc[0.25]),
                        "median": _safe_number(quantiles.loc[0.5]),
                        "q3": _safe_number(quantiles.loc[0.75]),
                        "maximum": _safe_number(quantiles.loc[1.0]),
                    }
                )

    input_set = set(recommendation["input_columns"])
    filter_recommendations = [
        recommendation_row
        for profile in numeric_profiles
        if profile["name"] in input_set
        for recommendation_row in [
            _recommend_filter_for_column(profile["name"], profile)
        ]
        if recommendation_row is not None
    ]

    return {
        "analysis_version": 2,
        "analysis_sampling": {
            "distribution": "seeded random sample" if row_count > max_rows else "all rows",
            "sequence": "longest finite run within a contiguous positional block",
            "sequence_block_start_row": signal_start,
            "sequence_block_rows": len(signal_df),
            "note": "A limited block may miss operating regimes elsewhere in the dataset.",
        },
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "row_count": int(row_count),
        "column_count": int(column_count),
        "analysed_row_count": int(len(analysed_df)),
        "sampling_frequency_hz": (
            float(sampling_frequency)
            if sampling_frequency is not None
            else None
        ),
        "missing_value_count": int(df.isna().sum().sum()),
        "duplicate_row_count": int(df.duplicated().sum()),
        "numeric_columns": [
            profile["name"] for profile in numeric_profiles
        ],
        "categorical_columns": categorical_columns,
        "time_columns": time_columns,
        "constant_columns": constant_columns,
        "column_profiles": column_profiles,
        "high_correlation_pairs": correlation_warnings[:15],
        "target_summaries": target_summaries,
        "recommendation": recommendation,
        "filter_recommendations": filter_recommendations,
    }


def format_dataset_analysis_report(profile):
    recommendation = profile["recommendation"]
    lines = [
        "LOCAL DATASET ANALYSIS",
        "=" * 72,
        "No raw rows were sent to an external service.",
        "",
        "DATASET HEALTH",
        f"Rows: {profile['row_count']}",
        f"Columns: {profile['column_count']}",
        f"Rows analysed for signal statistics: {profile['analysed_row_count']}",
        f"Missing values: {profile['missing_value_count']}",
        f"Duplicate rows: {profile['duplicate_row_count']}",
        f"Numeric columns: {len(profile['numeric_columns'])}",
        f"Non-numeric columns: {len(profile['categorical_columns'])}",
        f"Constant columns: {profile['constant_columns'] or 'None'}",
        f"Time/order columns: {profile['time_columns'] or 'None detected'}",
        "",
        "TASK RECOMMENDATION",
        f"Suggested task: {recommendation['task_type']}",
        f"Confidence: {100.0 * recommendation['confidence']:.1f}%",
        f"Suggested inputs: {recommendation['input_columns'] or 'None'}",
        f"Suggested targets: {recommendation['target_columns'] or 'None'}",
    ]

    for reason in recommendation["reasons"]:
        lines.append(f"- Reason: {reason}")
    for warning in recommendation["warnings"]:
        lines.append(f"- Warning: {warning}")
    if recommendation["alternatives"]:
        lines.append(
            "- Alternatives to consider: "
            + ", ".join(recommendation["alternatives"])
        )

    if profile["target_summaries"]:
        lines.extend(["", "TARGET SUMMARY"])
        for target in profile["target_summaries"]:
            if target["kind"] == "categorical":
                lines.append(f"- {target['column']} class distribution:")
                for class_row in target["class_distribution"]:
                    lines.append(
                        f"    {class_row['value']}: {class_row['count']} "
                        f"({class_row['percent']:.2f}%)"
                    )
            else:
                lines.append(
                    f"- {target['column']}: min={target['minimum']}, "
                    f"Q1={target['q1']}, median={target['median']}, "
                    f"Q3={target['q3']}, max={target['maximum']}"
                )

    lines.extend(["", "COLUMN SUMMARY"])
    for column in profile["column_profiles"]:
        details = (
            f"- {column['name']}: dtype={column['dtype']}, "
            f"numeric={column['is_numeric']}, "
            f"missing={column['missing_percent']:.2f}%, "
            f"unique={column['unique_count']}"
        )
        if column["is_numeric"]:
            details += (
                f", outliers={column.get('outlier_percent', 0.0):.2f}%, "
                f"lag-1={column.get('lag1_autocorrelation')}"
            )
            dominant_hz = column.get("dominant_frequency_hz")
            if dominant_hz is not None:
                details += f", dominant≈{dominant_hz:.3f} Hz"
        lines.append(details)

    lines.extend(["", "PRELIMINARY FILTER NOTES"])
    for item in profile["filter_recommendations"]:
        lines.append(
            f"- {item['column']}: {item['filter']} — {item['reason']}"
        )

    if profile["high_correlation_pairs"]:
        lines.extend(["", "POSSIBLE REDUNDANCY / LEAKAGE CHECK"])
        for item in profile["high_correlation_pairs"]:
            left, right = item["columns"]
            lines.append(
                f"- {left} ↔ {right}: |correlation|="
                f"{item['absolute_correlation']:.4f}"
            )

    lines.extend(
        [
            "",
            "NEXT ACTION",
            "Review the recommendation, optionally apply it, then continue to "
            "filter selection. Filtering should be validated against signal "
            "and model performance before final use.",
        ]
    )
    return "\n".join(lines)
