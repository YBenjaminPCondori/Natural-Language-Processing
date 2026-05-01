"""Evaluation helpers for model comparison, reports, and plots."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data_setup import write_json


def safe_name(value: str) -> str:
    """Create a filesystem-safe name."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_") or "item"


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: Path | str, title: str) -> Path:
    """Save a confusion matrix plot."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    size = max(8, min(18, len(labels) * 0.55))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    positions = np.arange(len(labels))
    ax.set_xticks(positions)
    ax.set_yticks(positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def evaluate_predictions_common(
    *,
    model_family: str,
    model_name: str,
    training_type: str,
    y_true: list[int],
    y_pred: list[int],
    df: pd.DataFrame,
    id2label: dict[int, str],
    dataset_name: str,
    output_dir: Path | str,
    eval_split: str = "test",
    notes: str = "",
    valid_labels: list[int] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute common metrics and save report/confusion matrix files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = valid_labels or sorted(id2label)
    names = [id2label[label_id] for label_id in labels]
    report = classification_report(y_true, y_pred, labels=labels, target_names=names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report_path = write_json(output_dir / "classification_reports" / f"{safe_name(model_name)}_report.json", report)
    cm_path = plot_confusion_matrix(cm, names, output_dir / "confusion_matrices" / f"{safe_name(model_name)}_confusion_matrix.png", f"{model_name} Confusion Matrix")

    pred_df = df[["text", "label", "label_id"]].copy()
    pred_df["predicted_label_id"] = [int(pred) for pred in y_pred]
    pred_df["predicted_label"] = [id2label.get(int(pred), "INVALID_PREDICTION") for pred in y_pred]

    result = {
        "model_family": model_family,
        "model_name": model_name,
        "training_type": training_type,
        "dataset": dataset_name,
        "eval_split": eval_split,
        "sample_size": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "notes": notes,
        "classification_report_path": str(report_path),
        "confusion_matrix_path": str(cm_path),
    }
    return result, pred_df


def save_final_comparison(completed_results: list[dict[str, Any]], results_dir: Path | str) -> pd.DataFrame:
    """Save final model comparison CSV and plot."""
    results_dir = Path(results_dir)
    comparison_columns = [
        "model_family",
        "model_name",
        "training_type",
        "dataset",
        "eval_split",
        "sample_size",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "notes",
    ]
    comparison_df = pd.DataFrame(completed_results).reindex(columns=comparison_columns)
    comparison_df.to_csv(results_dir / "final_model_comparison.csv", index=False)

    plot_df = comparison_df.dropna(subset=["macro_f1"]).copy()
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        plot_df.sort_values("macro_f1", ascending=False).plot(kind="bar", x="model_name", y="macro_f1", ax=ax, legend=False)
        ax.set_title("Final Model Comparison by Macro-F1")
        ax.set_xlabel("Model")
        ax.set_ylabel("Macro-F1")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        fig.savefig(results_dir / "final_model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    return comparison_df
