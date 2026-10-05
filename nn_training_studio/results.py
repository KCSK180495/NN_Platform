"""Results for NN Training Studio."""

from pathlib import Path
from datetime import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
from nn_training_studio.constants import (
    TASK_CLASSIFICATION,
)


def _result_json_default(value):
    """Convert common NumPy and path values for result JSON files."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Object of type {type(value).__name__} is not JSON serializable"
    )


def _safe_result_column_name(value, fallback):
    safe_name = "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in str(value)
    ).strip("_")
    return safe_name or fallback


def make_training_prediction_dataframe(
    task_type,
    actual_values,
    predicted_values,
    class_names=None,
    target_names=None,
    anomaly_scores=None,
):
    """Create a complete, task-aware table of held-out test predictions."""
    if predicted_values is None:
        return None

    predicted = np.asarray(predicted_values)
    if predicted.ndim == 0:
        predicted = predicted.reshape(1)
    record_count = len(predicted)
    result = {
        "record_number": np.arange(1, record_count + 1),
    }

    if task_type == TASK_CLASSIFICATION:
        predicted_indices = predicted.reshape(-1).astype(int)
        result["predicted_class_index"] = predicted_indices
        names = list(class_names or [])
        if names:
            result["predicted_label"] = [
                names[index] if 0 <= index < len(names) else str(index)
                for index in predicted_indices
            ]

        if actual_values is not None:
            actual_indices = np.asarray(actual_values).reshape(-1).astype(int)
            result["actual_class_index"] = actual_indices
            if names:
                result["actual_label"] = [
                    names[index] if 0 <= index < len(names) else str(index)
                    for index in actual_indices
                ]
            result["correct"] = actual_indices == predicted_indices
        return pd.DataFrame(result)

    predicted_2d = predicted.reshape(record_count, -1)
    actual_2d = None
    if actual_values is not None:
        actual = np.asarray(actual_values)
        actual_2d = actual.reshape(len(actual), -1)
        if len(actual_2d) != record_count:
            raise ValueError(
                "Actual and predicted result counts do not match."
            )

    output_names = list(target_names or [])
    for index in range(predicted_2d.shape[1]):
        display_name = (
            output_names[index]
            if index < len(output_names)
            else f"output_{index + 1}"
        )
        safe_name = _safe_result_column_name(
            display_name,
            f"output_{index + 1}",
        )
        result[f"predicted_{safe_name}"] = predicted_2d[:, index]
        if actual_2d is not None:
            result[f"actual_{safe_name}"] = actual_2d[:, index]
            result[f"absolute_error_{safe_name}"] = np.abs(
                actual_2d[:, index] - predicted_2d[:, index]
            )

    if anomaly_scores is not None:
        scores = np.asarray(anomaly_scores).reshape(-1)
        if len(scores) == record_count:
            result["anomaly_score"] = scores

    return pd.DataFrame(result)


def write_complete_result_files(
    output_directory,
    *,
    title,
    task_type,
    source_description,
    metadata,
    predictions_df,
    metrics=None,
    report_text="",
    confusion_mat=None,
    class_names=None,
    history=None,
    log_text="",
    actual_values=None,
    predicted_values=None,
    target_names=None,
    anomaly_scores=None,
):
    """Write a reproducible result bundle and return its relative file list."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    created_files = []

    def remember(path):
        created_files.append(str(Path(path).relative_to(output_directory)))

    summary = {
        "title": title,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task_type": task_type,
        "source": source_description,
        "metrics": metrics or {},
        "model_type": (metadata or {}).get("model_type"),
        "data_mode": (metadata or {}).get("data_mode"),
        "checkpointing": (metadata or {}).get("checkpointing", {}),
    }
    summary_path = output_directory / "results_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            indent=4,
            default=_result_json_default,
        )
    remember(summary_path)

    metrics_path = output_directory / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(
            metrics or {},
            file,
            indent=4,
            default=_result_json_default,
        )
    remember(metrics_path)

    report_lines = [
        title,
        "=" * 72,
        f"Created: {summary['created_at']}",
        f"Source: {source_description}",
        f"Task type: {task_type}",
        "",
        "Metrics",
        "-" * 72,
    ]
    if metrics:
        report_lines.extend(
            f"{key}: {value}" for key, value in metrics.items()
        )
    else:
        report_lines.append(
            "No labelled evaluation metrics were available."
        )
    report_lines.extend([
        "",
        "Evaluation Report",
        "-" * 72,
        report_text or "Prediction completed without a labelled report.",
    ])
    report_path = output_directory / "evaluation_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    remember(report_path)

    if log_text:
        log_path = output_directory / "training_or_evaluation_log.txt"
        log_path.write_text(str(log_text), encoding="utf-8")
        remember(log_path)

    if predictions_df is not None:
        predictions_path = output_directory / "predictions.csv"
        predictions_df.to_csv(predictions_path, index=False)
        remember(predictions_path)

    history_dict = history or {}
    if history_dict:
        history_frame = pd.DataFrame(history_dict)
        history_frame.index = np.arange(1, len(history_frame) + 1)
        history_frame.index.name = "epoch"
        history_csv_path = output_directory / "training_history.csv"
        history_frame.to_csv(history_csv_path)
        remember(history_csv_path)

        history_json_path = output_directory / "training_history.json"
        with open(history_json_path, "w", encoding="utf-8") as file:
            json.dump(
                history_dict,
                file,
                indent=4,
                default=_result_json_default,
            )
        remember(history_json_path)

        metric_key = (
            "accuracy"
            if "accuracy" in history_dict
            else "mae"
            if "mae" in history_dict
            else None
        )
        figure, axes = plt.subplots(2, 1, figsize=(9, 8))
        if metric_key:
            axes[0].plot(
                history_dict.get(metric_key, []),
                label=f"Training {metric_key}",
            )
            if history_dict.get("val_" + metric_key):
                axes[0].plot(
                    history_dict["val_" + metric_key],
                    label=f"Validation {metric_key}",
                )
            axes[0].set_ylabel(metric_key.replace("_", " ").title())
            axes[0].legend()
        else:
            axes[0].text(
                0.5,
                0.5,
                "No accuracy or MAE history available",
                ha="center",
                va="center",
            )
        axes[0].set_title("Training and Validation Metric")
        axes[0].set_xlabel("Epoch")
        axes[0].grid(True)

        axes[1].plot(history_dict.get("loss", []), label="Training loss")
        if history_dict.get("val_loss"):
            axes[1].plot(
                history_dict["val_loss"],
                label="Validation loss",
            )
        axes[1].set_title("Training and Validation Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(True)
        figure.tight_layout()
        curves_path = output_directory / "training_curves.png"
        figure.savefig(curves_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        remember(curves_path)

    if confusion_mat is not None:
        cm = np.asarray(confusion_mat)
        labels = list(class_names or [])
        if len(labels) != len(cm):
            labels = [str(index) for index in range(len(cm))]
        cm_csv_path = output_directory / "confusion_matrix.csv"
        pd.DataFrame(
            cm,
            index=[f"actual_{label}" for label in labels],
            columns=[f"predicted_{label}" for label in labels],
        ).to_csv(cm_csv_path)
        remember(cm_csv_path)

        figure, axis = plt.subplots(figsize=(8, 7))
        image = axis.imshow(cm)
        axis.set_title("Confusion Matrix")
        axis.set_xlabel("Predicted Label")
        axis.set_ylabel("Actual Label")
        axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        axis.set_yticks(np.arange(len(labels)), labels)
        figure.colorbar(image, ax=axis)
        for row in range(len(labels)):
            for column in range(len(labels)):
                axis.text(
                    column,
                    row,
                    cm[row, column],
                    ha="center",
                    va="center",
                )
        figure.tight_layout()
        cm_png_path = output_directory / "confusion_matrix.png"
        figure.savefig(cm_png_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        remember(cm_png_path)

    actual_2d = None
    predicted_2d = None
    if actual_values is not None and predicted_values is not None:
        actual_array = np.asarray(actual_values)
        predicted_array = np.asarray(predicted_values)
        if len(actual_array) == len(predicted_array):
            actual_2d = actual_array.reshape(len(actual_array), -1)
            predicted_2d = predicted_array.reshape(len(predicted_array), -1)
    elif predictions_df is not None:
        actual_columns = [
            column
            for column in predictions_df.columns
            if column.startswith("actual_")
            and "class_index" not in column
            and column != "actual_label"
        ]
        pairs = []
        for actual_column in actual_columns:
            predicted_column = "predicted_" + actual_column[len("actual_"):]
            if predicted_column in predictions_df.columns:
                pairs.append((actual_column, predicted_column))
        if pairs:
            actual_2d = predictions_df[
                [pair[0] for pair in pairs]
            ].to_numpy()
            predicted_2d = predictions_df[
                [pair[1] for pair in pairs]
            ].to_numpy()
            target_names = [
                pair[0][len("actual_"):] for pair in pairs
            ]

    if (
        task_type != TASK_CLASSIFICATION
        and actual_2d is not None
        and predicted_2d is not None
    ):
        sample_count = min(300, len(actual_2d))
        series_count = min(4, actual_2d.shape[1], predicted_2d.shape[1])
        figure, axis = plt.subplots(figsize=(10, 6))
        names = list(target_names or [])
        for index in range(series_count):
            name = (
                names[index]
                if index < len(names)
                else f"Output {index + 1}"
            )
            axis.plot(
                actual_2d[:sample_count, index],
                label=f"Actual {name}",
            )
            axis.plot(
                predicted_2d[:sample_count, index],
                linestyle="--",
                label=f"Predicted {name}",
            )
        axis.set_title("Actual vs Predicted")
        axis.set_xlabel("Test Record")
        axis.set_ylabel("Value")
        axis.legend()
        axis.grid(True)
        figure.tight_layout()
        comparison_path = output_directory / "actual_vs_predicted.png"
        figure.savefig(comparison_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        remember(comparison_path)

    if anomaly_scores is not None:
        scores = np.asarray(anomaly_scores).reshape(-1)
        scores_path = output_directory / "anomaly_scores.csv"
        pd.DataFrame({
            "record_number": np.arange(1, len(scores) + 1),
            "anomaly_score": scores,
        }).to_csv(scores_path, index=False)
        remember(scores_path)

        figure, axis = plt.subplots(figsize=(9, 5))
        axis.hist(scores, bins=40, alpha=0.85)
        threshold = (metadata or {}).get("anomaly_threshold")
        if threshold is not None:
            axis.axvline(
                threshold,
                color="red",
                linestyle="--",
                label="Anomaly threshold",
            )
            axis.legend()
        axis.set_title("Anomaly Score Distribution")
        axis.set_xlabel("Reconstruction Error")
        axis.set_ylabel("Count")
        axis.grid(True)
        figure.tight_layout()
        anomaly_plot_path = output_directory / "anomaly_scores.png"
        figure.savefig(anomaly_plot_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        remember(anomaly_plot_path)

    manifest_path = output_directory / "results_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "schema_version": 1,
                "created_at": summary["created_at"],
                "files": sorted(created_files),
            },
            file,
            indent=4,
        )
    remember(manifest_path)
    return sorted(created_files)


def zip_result_directory(source_directory, destination_zip):
    """Create a ZIP containing every result file below source_directory."""
    source_directory = Path(source_directory)
    with zipfile.ZipFile(
        destination_zip,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(source_directory.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    arcname=str(path.relative_to(source_directory)),
                )
