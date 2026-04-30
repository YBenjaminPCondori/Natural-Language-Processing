"""Evaluation and JSONL helpers for the LEDGAR coursework pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


RANDOM_STATE = 42


def ensure_parent(path: Path | str) -> Path:
    """Create the parent directory for an output path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def make_json_safe(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-serialisable Python objects."""
    if isinstance(value, Mapping):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (list, tuple, dict, str)) else False:
        return None
    return value


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path | str, data: Any) -> Path:
    """Write JSON using stable formatting."""
    output_path = ensure_parent(path)
    output_path.write_text(
        json.dumps(make_json_safe(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def write_jsonl(path: Path | str, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Write rows to JSONL."""
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(make_json_safe(dict(row)), ensure_ascii=False) + "\n")
    return output_path


def append_jsonl(path: Path | str, row: Mapping[str, Any]) -> Path:
    """Append one result row to a JSONL file."""
    output_path = ensure_parent(path)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(make_json_safe(dict(row)), ensure_ascii=False) + "\n")
    return output_path


def reset_jsonl(path: Path | str) -> Path:
    """Create or truncate a JSONL file."""
    output_path = ensure_parent(path)
    output_path.write_text("", encoding="utf-8")
    return output_path


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int] | None = None,
) -> dict[str, float]:
    """Compute the primary and secondary classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        ),
        "macro_precision": precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        ),
    }


def build_classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int],
    target_names: Sequence[str],
) -> dict[str, Any]:
    """Return sklearn's classification report as a dictionary."""
    return classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        target_names=list(target_names),
        output_dict=True,
        zero_division=0,
    )


def build_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int],
) -> list[list[int]]:
    """Return a confusion matrix as nested Python lists."""
    return confusion_matrix(y_true, y_pred, labels=list(labels)).tolist()


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int],
    target_names: Sequence[str],
    model_name: str,
    split: str,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact result row for one model/split."""
    metrics = compute_metrics(y_true, y_pred, labels=labels)
    report = build_classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
    )
    per_class_f1 = {
        target_name: report.get(target_name, {}).get("f1-score", 0.0)
        for target_name in target_names
    }
    return {
        "model_name": model_name,
        "split": split,
        "primary_metric": "macro_f1",
        "metrics": metrics,
        "per_class_f1": per_class_f1,
        "params": dict(params or {}),
    }


def save_predictions_jsonl(
    path: Path | str,
    df: pd.DataFrame,
    predictions: Sequence[int],
    *,
    id_to_label: Mapping[int, str],
    model_name: str,
    split: str,
) -> Path:
    """Save prediction rows using the coursework JSONL schema."""
    rows = []
    for row, pred_id in zip(df.to_dict(orient="records"), predictions):
        pred_id = int(pred_id)
        rows.append(
            {
                "text": row["text"],
                "true_label": row["label"],
                "true_label_id": int(row["label_id"]),
                "predicted_label": id_to_label[pred_id],
                "predicted_label_id": pred_id,
                "model_name": model_name,
                "split": split,
            }
        )
    return write_jsonl(path, rows)


def append_result_row(path: Path | str, result_row: Mapping[str, Any]) -> Path:
    """Append a model result row to JSONL."""
    return append_jsonl(path, result_row)


def plot_confusion_matrix(
    cm: Sequence[Sequence[int]],
    labels: Sequence[str],
    path: Path | str,
    *,
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (10, 8),
) -> Path:
    """Save a simple matplotlib confusion matrix plot."""
    output_path = ensure_parent(path)
    matrix = np.array(cm)

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    tick_positions = np.arange(len(labels))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
