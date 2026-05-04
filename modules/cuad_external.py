"""CUAD external evaluation helpers.

CUAD is span-annotated, so this module converts non-empty answer spans into a
clause-level classification table for external LEDGAR label-space evaluation.
It deliberately does not expose any training or validation helpers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .data_setup import (
    ProjectPaths,
    cuad_label_from_qa,
    download_cuad_if_missing,
    load_cuad_raw_files,
    load_jsonl,
    normalise_whitespace,
    write_json,
)
from .evaluation import safe_name


CUAD_EXTERNAL_DATASET_NAME = "CUAD_external"
LEDGAR_TEST_DATASET_NAME = "LEDGAR_test"
CUAD_RAW_COLUMNS = ["text", "label_original", "label_mapped", "source_dataset", "contract_id", "span_id"]
CUAD_EVAL_COLUMNS = [
    "text",
    "label_original",
    "label_mapped",
    "label",
    "label_id",
    "source_dataset",
    "dataset_name",
    "split",
    "contract_id",
    "span_id",
]
CUAD_PREDICTION_COLUMNS = [
    "text",
    "true_label",
    "true_label_id",
    "predicted_label",
    "predicted_label_id",
    "confidence",
    "is_correct",
    "dataset_name",
    "model_name",
    "source_dataset",
    "contract_id",
    "span_id",
    "label_original",
    "label_mapped",
]


CUAD_TO_LEDGAR_LABEL_MAP: dict[str, str] = {
    "Governing Law": "Governing Laws",
    "Anti-Assignment": "Assignments",
    "Termination For Convenience": "Terminations",
    "Insurance": "Insurances",
    "Notice Period To Terminate Renewal": "Notices",
    "Expiration Date": "Terms",
    "Renewal Term": "Terms",
    "Effective Date": "Terms",
}


def ensure_cuad_output_dirs(paths: ProjectPaths) -> dict[str, Path]:
    """Create and return the CUAD output directories."""
    directories = {
        "data": paths.project_root / "outputs" / "data",
        "predictions": paths.project_root / "outputs" / "predictions",
        "metrics": paths.project_root / "outputs" / "metrics",
        "error_analysis": paths.project_root / "outputs" / "error_analysis",
        "logs": paths.project_root / "outputs" / "logs",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _span_is_too_short(text: str, *, min_chars: int, min_words: int) -> bool:
    return len(text) < min_chars or len(text.split()) < min_words


def convert_cuad_json_to_raw_dataframe(
    raw_cuad_json: dict[str, Any] | None,
    *,
    min_span_chars: int = 20,
    min_span_words: int = 4,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert CUAD answer spans to raw clause-classification rows."""
    stats = {
        "created_at_utc": _utc_now(),
        "min_span_chars": min_span_chars,
        "min_span_words": min_span_words,
        "documents_seen": 0,
        "answer_spans_seen": 0,
        "empty_spans_removed": 0,
        "short_spans_removed": 0,
        "duplicate_rows_removed": 0,
        "rows_after_short_filter": 0,
        "rows_after_deduplication": 0,
        "status": "completed",
        "reason": "",
    }
    if raw_cuad_json is None:
        stats["status"] = "failed"
        stats["reason"] = "CUAD JSON is unavailable."
        return pd.DataFrame(columns=CUAD_RAW_COLUMNS), stats

    records: list[dict[str, Any]] = []
    for doc_idx, document in enumerate(raw_cuad_json.get("data", [])):
        stats["documents_seen"] += 1
        contract_id = normalise_whitespace(document.get("title") or document.get("id") or f"cuad_doc_{doc_idx}")
        for paragraph_idx, paragraph in enumerate(document.get("paragraphs", [])):
            for qa_idx, qa in enumerate(paragraph.get("qas", [])):
                if qa.get("is_impossible") is True:
                    continue
                label_original = cuad_label_from_qa(qa)
                if not label_original:
                    continue
                for answer_idx, answer in enumerate(qa.get("answers", []) or []):
                    stats["answer_spans_seen"] += 1
                    raw_text = answer.get("text") if isinstance(answer, dict) else answer
                    text = normalise_whitespace(raw_text)
                    if not text:
                        stats["empty_spans_removed"] += 1
                        continue
                    if _span_is_too_short(text, min_chars=min_span_chars, min_words=min_span_words):
                        stats["short_spans_removed"] += 1
                        continue
                    records.append(
                        {
                            "text": text,
                            "label_original": label_original,
                            "label_mapped": pd.NA,
                            "source_dataset": "CUAD",
                            "contract_id": contract_id,
                            "span_id": f"cuad_{doc_idx}_{paragraph_idx}_{qa_idx}_{answer_idx}",
                        }
                    )

    raw_df = pd.DataFrame(records, columns=CUAD_RAW_COLUMNS)
    stats["rows_after_short_filter"] = int(len(raw_df))
    if not raw_df.empty:
        before = len(raw_df)
        raw_df = raw_df.drop_duplicates(subset=["text", "label_original"]).reset_index(drop=True)
        stats["duplicate_rows_removed"] = int(before - len(raw_df))
    stats["rows_after_deduplication"] = int(len(raw_df))
    if raw_df.empty and stats["status"] == "completed":
        stats["status"] = "failed"
        stats["reason"] = "No usable CUAD answer spans remained after filtering."
    return raw_df.reindex(columns=CUAD_RAW_COLUMNS), stats


def save_cuad_converted_raw(
    raw_cuad_json: dict[str, Any] | None,
    paths: ProjectPaths,
    *,
    min_span_chars: int = 20,
    min_span_words: int = 4,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    """Save outputs/data/cuad_converted_raw.csv and preprocessing stats."""
    directories = ensure_cuad_output_dirs(paths)
    raw_df, stats = convert_cuad_json_to_raw_dataframe(
        raw_cuad_json,
        min_span_chars=min_span_chars,
        min_span_words=min_span_words,
    )
    raw_path = directories["data"] / "cuad_converted_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    write_json(directories["data"] / "cuad_preprocessing_report.json", stats)
    return raw_df, stats, raw_path


def load_ledgar_label_space(paths: ProjectPaths) -> tuple[dict[str, int], dict[int, str]]:
    """Load the LEDGAR label encoder used by the current trained classifiers."""
    mapping_path = paths.project_root / "outputs" / "label_mapping.json"
    if mapping_path.exists():
        import json

        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        label2id = {str(label): int(label_id) for label, label_id in payload.get("label_to_id", {}).items()}
        id2label = {int(label_id): str(label) for label_id, label in payload.get("id_to_label", {}).items()}
        if label2id and id2label:
            return label2id, id2label

    label_path = paths.processed_data_dir / "label_names.txt"
    labels = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    return label2id, id2label


def apply_cuad_label_mapping(
    raw_df: pd.DataFrame,
    paths: ProjectPaths,
    *,
    preprocessing_stats: dict[str, Any] | None = None,
    label_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    """Apply the transparent manual CUAD-to-LEDGAR mapping and save coverage."""
    directories = ensure_cuad_output_dirs(paths)
    label_map = label_map or CUAD_TO_LEDGAR_LABEL_MAP
    if raw_df.empty:
        report = {
            "created_at_utc": _utc_now(),
            "status": "failed",
            "reason": "No CUAD raw rows available for mapping.",
            "number_of_original_cuad_labels": 0,
            "number_of_mapped_labels": 0,
            "number_of_unmapped_labels": 0,
            "number_of_extracted_spans_before_mapping": 0,
            "number_of_samples_retained_after_mapping": 0,
            "number_of_samples_dropped_due_to_unmapped_labels": 0,
            "mapped_labels": [],
            "unmapped_labels": [],
            "preprocessing": preprocessing_stats or {},
        }
        mapped_df = pd.DataFrame(columns=CUAD_RAW_COLUMNS)
    else:
        working = raw_df.copy()
        working["label_mapped"] = working["label_original"].map(label_map)
        mapped_df = working[working["label_mapped"].notna()].copy().reset_index(drop=True)
        original_labels = sorted(working["label_original"].dropna().astype(str).unique().tolist())
        mapped_original_labels = sorted([label for label in original_labels if label in label_map])
        unmapped_labels = sorted([label for label in original_labels if label not in label_map])
        report = {
            "created_at_utc": _utc_now(),
            "status": "completed" if not mapped_df.empty else "failed",
            "reason": "" if not mapped_df.empty else "No CUAD labels matched the manual LEDGAR label map.",
            "number_of_original_cuad_labels": len(original_labels),
            "number_of_mapped_labels": len(mapped_original_labels),
            "number_of_unmapped_labels": len(unmapped_labels),
            "number_of_extracted_spans_before_mapping": int(len(working)),
            "number_of_samples_retained_after_mapping": int(len(mapped_df)),
            "number_of_samples_dropped_due_to_unmapped_labels": int(len(working) - len(mapped_df)),
            "mapped_labels": [{"cuad_label": label, "ledgar_label": label_map[label]} for label in mapped_original_labels],
            "unmapped_labels": unmapped_labels,
            "preprocessing": preprocessing_stats or {},
        }

    mapped_path = directories["data"] / "cuad_converted_mapped.csv"
    mapped_df.reindex(columns=CUAD_RAW_COLUMNS).to_csv(mapped_path, index=False)
    write_json(directories["data"] / "cuad_label_mapping_report.json", report)
    return mapped_df.reindex(columns=CUAD_RAW_COLUMNS), report, mapped_path


def validate_cuad_against_ledgar_label_space(
    mapped_df: pd.DataFrame,
    paths: ProjectPaths,
    label2id: dict[str, int],
) -> tuple[pd.DataFrame, dict[str, Any], Path]:
    """Drop mappings that are not in the current LEDGAR label encoder."""
    directories = ensure_cuad_output_dirs(paths)
    if mapped_df.empty:
        report = {
            "created_at_utc": _utc_now(),
            "status": "failed",
            "reason": "No mapped CUAD rows available for LEDGAR label-space validation.",
            "invalid_mapped_labels": [],
            "rows_before_validation": 0,
            "rows_after_validation": 0,
            "rows_dropped_due_to_invalid_ledgar_label": 0,
        }
        eval_df = pd.DataFrame(columns=CUAD_EVAL_COLUMNS)
    else:
        working = mapped_df.copy()
        invalid_labels = sorted(set(working["label_mapped"].dropna().astype(str)) - set(label2id))
        eval_df = working[working["label_mapped"].isin(label2id)].copy().reset_index(drop=True)
        eval_df["label"] = eval_df["label_mapped"]
        eval_df["label_id"] = eval_df["label"].map(label2id).astype(int)
        eval_df["dataset_name"] = CUAD_EXTERNAL_DATASET_NAME
        eval_df["split"] = "external"
        dropped = int(len(working) - len(eval_df))
        report = {
            "created_at_utc": _utc_now(),
            "status": "completed" if not eval_df.empty else "failed",
            "reason": "" if not eval_df.empty else "No CUAD rows remained after LEDGAR label-space validation.",
            "invalid_mapped_labels": invalid_labels,
            "rows_before_validation": int(len(working)),
            "rows_after_validation": int(len(eval_df)),
            "rows_dropped_due_to_invalid_ledgar_label": dropped,
        }

    eval_path = directories["data"] / "cuad_external_eval.csv"
    eval_df.reindex(columns=CUAD_EVAL_COLUMNS).to_csv(eval_path, index=False)
    write_json(directories["data"] / "cuad_label_compatibility_report.json", report)
    return eval_df.reindex(columns=CUAD_EVAL_COLUMNS), report, eval_path


def prepare_cuad_external_eval_data(
    paths: ProjectPaths,
    *,
    raw_cuad_json: dict[str, Any] | None = None,
    min_span_chars: int = 20,
    min_span_words: int = 4,
) -> dict[str, Any]:
    """Run CUAD conversion, mapping, and LEDGAR label-space validation."""
    label2id, id2label = load_ledgar_label_space(paths)
    raw_df, preprocessing_report, raw_path = save_cuad_converted_raw(
        raw_cuad_json,
        paths,
        min_span_chars=min_span_chars,
        min_span_words=min_span_words,
    )
    mapped_df, mapping_report, mapped_path = apply_cuad_label_mapping(
        raw_df,
        paths,
        preprocessing_stats=preprocessing_report,
    )
    eval_df, compatibility_report, eval_path = validate_cuad_against_ledgar_label_space(mapped_df, paths, label2id)
    return {
        "raw_df": raw_df,
        "mapped_df": mapped_df,
        "eval_df": eval_df,
        "label2id": label2id,
        "id2label": id2label,
        "preprocessing_report": preprocessing_report,
        "mapping_report": mapping_report,
        "compatibility_report": compatibility_report,
        "paths": {
            "raw": raw_path,
            "mapped": mapped_path,
            "eval": eval_path,
            "mapping_report": paths.project_root / "outputs" / "data" / "cuad_label_mapping_report.json",
        },
    }


def _model_output_paths(paths: ProjectPaths, model_name: str) -> tuple[Path, Path, Path, Path]:
    safe = safe_name(model_name)
    predictions_path = paths.project_root / "outputs" / "predictions" / f"cuad_external_{safe}_predictions.csv"
    metrics_path = paths.project_root / "outputs" / "metrics" / f"cuad_external_{safe}_metrics.json"
    errors_path = paths.project_root / "outputs" / "error_analysis" / f"cuad_external_{safe}_errors.csv"
    summary_path = paths.project_root / "outputs" / "error_analysis" / f"cuad_external_{safe}_summary.json"
    return predictions_path, metrics_path, errors_path, summary_path


def _confidence_from_estimator(model: Any, texts: list[str]) -> list[float | None]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)
        return [float(np.max(row)) for row in probabilities]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(texts)
        if np.asarray(scores).ndim == 1:
            return [float(value) for value in np.asarray(scores)]
        return [float(np.max(row)) for row in scores]
    return [None for _ in texts]


def _prediction_frame(
    *,
    model_name: str,
    cuad_df: pd.DataFrame,
    y_pred: list[int],
    id2label: dict[int, str],
    confidence: list[float | None] | None = None,
) -> pd.DataFrame:
    confidence = confidence or [None for _ in y_pred]
    output = pd.DataFrame(
        {
            "text": cuad_df["text"].astype(str),
            "true_label": cuad_df["label"].astype(str),
            "true_label_id": cuad_df["label_id"].astype(int),
            "predicted_label_id": [int(value) for value in y_pred],
            "predicted_label": [id2label.get(int(value), "INVALID_PREDICTION") for value in y_pred],
            "confidence": confidence,
            "dataset_name": CUAD_EXTERNAL_DATASET_NAME,
            "model_name": model_name,
            "source_dataset": "CUAD",
            "contract_id": cuad_df["contract_id"].astype(str),
            "span_id": cuad_df["span_id"].astype(str),
            "label_original": cuad_df["label_original"].astype(str),
            "label_mapped": cuad_df["label_mapped"].astype(str),
        }
    )
    output["is_correct"] = output["true_label_id"].astype(int) == output["predicted_label_id"].astype(int)
    return output.reindex(columns=CUAD_PREDICTION_COLUMNS)


def _metrics_payload(
    *,
    model_name: str,
    pred_df: pd.DataFrame,
    id2label: dict[int, str],
    notes: str,
    stale_output_warning: str | None = None,
) -> dict[str, Any]:
    y_true = pred_df["true_label_id"].astype(int).tolist()
    y_pred = pred_df["predicted_label_id"].astype(int).tolist()
    true_label_ids = sorted(set(y_true))
    confusion_label_ids = sorted(id2label)
    true_label_names = [id2label[label_id] for label_id in true_label_ids]
    confusion_label_names = [id2label.get(label_id, f"unknown_{label_id}") for label_id in confusion_label_ids]
    report = classification_report(
        y_true,
        y_pred,
        labels=true_label_ids,
        target_names=true_label_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=confusion_label_ids)
    payload = {
        "created_at_utc": _utc_now(),
        "status": "completed",
        "model_name": model_name,
        "dataset_name": CUAD_EXTERNAL_DATASET_NAME,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=true_label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=true_label_ids, average="weighted", zero_division=0)),
        "macro_f1_all_ledgar_labels": float(f1_score(y_true, y_pred, labels=sorted(id2label), average="macro", zero_division=0)),
        "num_samples": int(len(pred_df)),
        "num_labels": int(len(true_label_ids)),
        "evaluated_labels": true_label_names,
        "per_class": {
            label: metrics
            for label, metrics in report.items()
            if isinstance(metrics, dict) and label not in {"micro avg", "macro avg", "weighted avg", "samples avg"}
        },
        "confusion_matrix": {
            "labels": confusion_label_names,
            "label_ids": confusion_label_ids,
            "matrix": cm.tolist(),
        },
        "notes": notes,
    }
    if stale_output_warning:
        payload["stale_output_warning"] = stale_output_warning
    return payload


def write_model_status(
    paths: ProjectPaths,
    *,
    model_name: str,
    status: str,
    reason: str,
    num_samples: int = 0,
) -> Path:
    """Write a non-silent skipped/failed CUAD metric record."""
    ensure_cuad_output_dirs(paths)
    _, metrics_path, _, _ = _model_output_paths(paths, model_name)
    payload = {
        "created_at_utc": _utc_now(),
        "status": status,
        "model_name": model_name,
        "dataset_name": CUAD_EXTERNAL_DATASET_NAME,
        "accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
        "num_samples": int(num_samples),
        "num_labels": 0,
        "notes": reason,
        "reason": reason,
    }
    write_json(metrics_path, payload)
    return metrics_path


def write_cuad_error_analysis(pred_df: pd.DataFrame, metrics: dict[str, Any], paths: ProjectPaths) -> dict[str, Path]:
    """Write CUAD external error CSV and summary JSON for one model."""
    ensure_cuad_output_dirs(paths)
    model_name = str(metrics["model_name"])
    _, _, errors_path, summary_path = _model_output_paths(paths, model_name)
    errors = pred_df[pred_df["is_correct"] != True].copy()
    if not errors.empty:
        errors["text_length_chars"] = errors["text"].astype(str).str.len()
        errors["text_length_words"] = errors["text"].astype(str).str.split().str.len()
        sort_columns = ["confidence", "text_length_chars"] if errors["confidence"].notna().any() else ["text_length_chars"]
        ascending = [False, True] if errors["confidence"].notna().any() else [True]
        errors = errors.sort_values(sort_columns, ascending=ascending)
    errors.to_csv(errors_path, index=False)

    top_confusions = []
    if not errors.empty:
        top_confusions = (
            errors.groupby(["true_label", "predicted_label"])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(20)
            .to_dict(orient="records")
        )

    per_label_accuracy = []
    if not pred_df.empty:
        for label, group in pred_df.groupby("true_label"):
            class_metrics = metrics.get("per_class", {}).get(label, {})
            per_label_accuracy.append(
                {
                    "label": label,
                    "accuracy": float(group["is_correct"].mean()),
                    "support": int(len(group)),
                    "precision": class_metrics.get("precision"),
                    "recall": class_metrics.get("recall"),
                    "f1": class_metrics.get("f1-score"),
                }
            )

    most_confident_wrong = pd.DataFrame()
    if not errors.empty and errors["confidence"].notna().any():
        most_confident_wrong = errors.sort_values("confidence", ascending=False).head(20)
    shortest_wrong = errors.sort_values("text_length_chars", ascending=True).head(20) if not errors.empty else pd.DataFrame()
    longest_wrong = errors.sort_values("text_length_chars", ascending=False).head(20) if not errors.empty else pd.DataFrame()

    sample_columns = ["text", "true_label", "predicted_label", "confidence", "contract_id", "span_id", "label_original"]
    summary = {
        "created_at_utc": _utc_now(),
        "status": "completed",
        "model_name": model_name,
        "dataset_name": CUAD_EXTERNAL_DATASET_NAME,
        "num_errors": int(len(errors)),
        "error_rate": float(1.0 - pred_df["is_correct"].mean()) if not pred_df.empty else None,
        "top_confused_label_pairs": top_confusions,
        "per_label_accuracy_f1": per_label_accuracy,
        "most_confident_wrong_predictions": most_confident_wrong.reindex(columns=sample_columns).to_dict(orient="records"),
        "shortest_wrong_predictions": shortest_wrong.reindex(columns=sample_columns).to_dict(orient="records"),
        "longest_wrong_predictions": longest_wrong.reindex(columns=sample_columns).to_dict(orient="records"),
        "sample_error_table": errors.reindex(columns=sample_columns).head(50).to_dict(orient="records"),
    }
    write_json(summary_path, summary)
    return {"errors": errors_path, "summary": summary_path}


def evaluate_sklearn_model_on_cuad(
    model: Any,
    *,
    model_name: str,
    cuad_df: pd.DataFrame,
    id2label: dict[int, str],
    paths: ProjectPaths,
    notes: str,
) -> dict[str, Any]:
    """Evaluate one fitted sklearn-compatible model on CUAD external data."""
    ensure_cuad_output_dirs(paths)
    predictions_path, metrics_path, _, _ = _model_output_paths(paths, model_name)
    stale_warning = None
    if predictions_path.exists() or metrics_path.exists():
        stale_warning = "Existing CUAD external output was overwritten by a fresh evaluation run."
    texts = cuad_df["text"].astype(str).tolist()
    y_pred = [int(value) for value in model.predict(texts)]
    confidence = _confidence_from_estimator(model, texts)
    pred_df = _prediction_frame(model_name=model_name, cuad_df=cuad_df, y_pred=y_pred, id2label=id2label, confidence=confidence)
    metrics = _metrics_payload(model_name=model_name, pred_df=pred_df, id2label=id2label, notes=notes, stale_output_warning=stale_warning)
    pred_df.to_csv(predictions_path, index=False)
    write_json(metrics_path, metrics)
    error_paths = write_cuad_error_analysis(pred_df, metrics, paths)
    metrics["prediction_path"] = str(predictions_path)
    metrics["metrics_path"] = str(metrics_path)
    metrics["error_analysis_paths"] = {key: str(value) for key, value in error_paths.items()}
    return metrics


def discover_classical_model_files(paths: ProjectPaths) -> dict[str, Path]:
    """Find fitted classical model artifacts without retraining."""
    discovered: dict[str, Path] = {}
    for model_name in ("logistic_regression", "linear_svm", "multinomial_nb"):
        candidates = [
            paths.project_root / "models" / "trained" / "classical" / f"{model_name}.joblib",
            paths.project_root / "models" / "classical" / f"{model_name}.joblib",
            paths.project_root / "checkpoints" / "classical" / f"{model_name}_best.joblib",
        ]
        for candidate in candidates:
            if candidate.exists():
                discovered[model_name] = candidate
                break
    return discovered


def evaluate_classical_models_on_cuad(
    cuad_df: pd.DataFrame,
    id2label: dict[int, str],
    paths: ProjectPaths,
) -> list[dict[str, Any]]:
    """Evaluate all available fitted classical LEDGAR models on CUAD."""
    results: list[dict[str, Any]] = []
    model_files = discover_classical_model_files(paths)
    for model_name in ("logistic_regression", "linear_svm", "multinomial_nb"):
        model_path = model_files.get(model_name)
        if model_path is None:
            write_model_status(paths, model_name=model_name, status="skipped", reason="No fitted classical model artifact found.", num_samples=len(cuad_df))
            continue
        try:
            model = joblib.load(model_path)
            results.append(
                evaluate_sklearn_model_on_cuad(
                    model,
                    model_name=model_name,
                    cuad_df=cuad_df,
                    id2label=id2label,
                    paths=paths,
                    notes=f"External CUAD evaluation using existing LEDGAR-trained artifact: {model_path}.",
                )
            )
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            write_model_status(paths, model_name=model_name, status="failed", reason=reason, num_samples=len(cuad_df))
    return results


def evaluate_transformer_model_on_cuad(
    cuad_df: pd.DataFrame,
    id2label: dict[int, str],
    paths: ProjectPaths,
    *,
    model_name: str = "distilbert-base-uncased",
    max_length: int = 256,
    batch_size: int = 16,
    run_transformer: bool = True,
) -> dict[str, Any] | None:
    """Evaluate the saved fine-tuned transformer model on CUAD if available."""
    if not run_transformer:
        write_model_status(paths, model_name=model_name, status="skipped", reason="Transformer CUAD evaluation disabled by caller.", num_samples=len(cuad_df))
        return None
    model_dir = paths.results_dir / "transformer" / "model"
    if not model_dir.exists():
        write_model_status(paths, model_name=model_name, status="skipped", reason=f"No saved transformer model directory found at {model_dir}.", num_samples=len(cuad_df))
        return None

    ensure_cuad_output_dirs(paths)
    predictions_path, metrics_path, _, _ = _model_output_paths(paths, model_name)
    stale_warning = None
    if predictions_path.exists() or metrics_path.exists():
        stale_warning = "Existing CUAD external output was overwritten by a fresh evaluation run."

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        texts = cuad_df["text"].astype(str).tolist()
        predictions: list[int] = []
        confidence: list[float] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1)
                batch_confidence, batch_predictions = probabilities.max(dim=-1)
            predictions.extend([int(value) for value in batch_predictions.detach().cpu().tolist()])
            confidence.extend([float(value) for value in batch_confidence.detach().cpu().tolist()])

        pred_df = _prediction_frame(model_name=model_name, cuad_df=cuad_df, y_pred=predictions, id2label=id2label, confidence=confidence)
        metrics = _metrics_payload(
            model_name=model_name,
            pred_df=pred_df,
            id2label=id2label,
            notes=f"External CUAD evaluation using saved LEDGAR fine-tuned transformer at {model_dir}.",
            stale_output_warning=stale_warning,
        )
        pred_df.to_csv(predictions_path, index=False)
        write_json(metrics_path, metrics)
        error_paths = write_cuad_error_analysis(pred_df, metrics, paths)
        metrics["prediction_path"] = str(predictions_path)
        metrics["metrics_path"] = str(metrics_path)
        metrics["error_analysis_paths"] = {key: str(value) for key, value in error_paths.items()}
        return metrics
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        write_model_status(paths, model_name=model_name, status="failed", reason=reason, num_samples=len(cuad_df))
        return None


def write_llm_external_statuses(paths: ProjectPaths, *, num_samples: int) -> list[Path]:
    """Write explicit CUAD skipped statuses for prompt-only LLM rows without reusable artifacts."""
    main_results_path = paths.project_root / "outputs" / "main_results.csv"
    if not main_results_path.exists():
        return []
    main_results = pd.read_csv(main_results_path)
    llm_rows = main_results[
        main_results.get("model_family", pd.Series(dtype=str)).astype(str).str.contains("llm", case=False, na=False)
        | main_results.get("model_name", pd.Series(dtype=str)).astype(str).str.contains("qwen", case=False, na=False)
    ]
    outputs = []
    for model_name in sorted(llm_rows["model_name"].dropna().astype(str).unique().tolist()):
        reason = (
            "CUAD external prompting was not run because the LEDGAR prompting stage does not save a reusable "
            "local model/tokenizer artifact. Run a dedicated CUAD prompting job with the model loaded to evaluate it."
        )
        outputs.append(write_model_status(paths, model_name=model_name, status="skipped", reason=reason, num_samples=num_samples))
    return outputs


def validate_cuad_protocol_safeguards(paths: ProjectPaths, cuad_eval_df: pd.DataFrame, label2id: dict[str, int]) -> dict[str, Any]:
    """Check that CUAD remains external and LEDGAR splits are not contaminated."""
    ensure_cuad_output_dirs(paths)
    split_checks = {}
    for split in ("train", "validation"):
        split_path = paths.processed_data_dir / f"ledgar_{split}.jsonl"
        if not split_path.exists():
            split_checks[split] = {"path": str(split_path), "status": "missing", "cuad_rows_found": None}
            continue
        try:
            split_df = load_jsonl(split_path)
            source_values = split_df.get("source_dataset", pd.Series(dtype=str)).astype(str)
            cuad_rows = int(source_values.eq("CUAD").sum())
            split_checks[split] = {"path": str(split_path), "status": "checked", "cuad_rows_found": cuad_rows}
        except Exception as exc:
            split_checks[split] = {"path": str(split_path), "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}

    invalid_eval_labels = sorted(set(cuad_eval_df["label"].dropna().astype(str)) - set(label2id)) if not cuad_eval_df.empty else []
    payload = {
        "created_at_utc": _utc_now(),
        "status": "passed" if not cuad_eval_df.empty and not invalid_eval_labels and all(check.get("cuad_rows_found", 0) in {0, None} for check in split_checks.values()) else "attention_required",
        "cuad_external_eval_not_empty": bool(not cuad_eval_df.empty),
        "mapped_labels_are_valid": not invalid_eval_labels,
        "invalid_eval_labels": invalid_eval_labels,
        "no_cuad_in_ledgar_train_or_validation": all(check.get("cuad_rows_found", 0) in {0, None} for check in split_checks.values()),
        "ledgar_split_checks": split_checks,
        "protocol": {
            "train": "LEDGAR train only",
            "validation": "LEDGAR validation only",
            "test": "LEDGAR test for primary in-distribution results",
            "external": "CUAD external post-test evaluation only",
        },
    }
    write_json(paths.project_root / "outputs" / "metrics" / "cuad_external_safeguards.json", payload)
    return payload


def _read_json_metric(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _clean_note(value: Any) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def build_final_model_comparison_summary(paths: ProjectPaths) -> Path:
    """Create one LEDGAR-test plus CUAD-external summary CSV with dataset labels."""
    ensure_cuad_output_dirs(paths)
    rows: list[dict[str, Any]] = []
    label_count = len(load_ledgar_label_space(paths)[1])
    ledgar_source = paths.results_dir / "final_model_comparison.csv"
    if ledgar_source.exists():
        ledgar_df = pd.read_csv(ledgar_source)
        for _, row in ledgar_df.iterrows():
            rows.append(
                {
                    "model_name": row.get("model_name"),
                    "dataset_name": LEDGAR_TEST_DATASET_NAME,
                    "accuracy": row.get("accuracy"),
                    "macro_f1": row.get("macro_f1"),
                    "weighted_f1": row.get("weighted_f1"),
                    "num_samples": row.get("sample_size"),
                    "num_labels": label_count,
                    "notes": _clean_note(row.get("notes", "")),
                }
            )
    else:
        main_source = paths.project_root / "outputs" / "main_results.csv"
        if main_source.exists():
            main_df = pd.read_csv(main_source)
            for _, row in main_df.iterrows():
                rows.append(
                    {
                        "model_name": row.get("model_name"),
                        "dataset_name": LEDGAR_TEST_DATASET_NAME,
                        "accuracy": row.get("test_accuracy"),
                        "macro_f1": row.get("test_macro_f1"),
                        "weighted_f1": row.get("test_weighted_f1"),
                        "num_samples": pd.NA,
                        "num_labels": label_count,
                        "notes": _clean_note(row.get("reason_if_skipped_or_failed")) or _clean_note(row.get("status", "")),
                    }
                )

    for metric_path in sorted((paths.project_root / "outputs" / "metrics").glob("cuad_external_*_metrics.json")):
        payload = _read_json_metric(metric_path)
        if not payload:
            continue
        rows.append(
            {
                "model_name": payload.get("model_name"),
                "dataset_name": CUAD_EXTERNAL_DATASET_NAME,
                "accuracy": payload.get("accuracy"),
                "macro_f1": payload.get("macro_f1"),
                "weighted_f1": payload.get("weighted_f1"),
                "num_samples": payload.get("num_samples"),
                "num_labels": payload.get("num_labels"),
                "notes": payload.get("notes") or payload.get("reason") or payload.get("status"),
            }
        )

    output_path = paths.project_root / "outputs" / "metrics" / "final_model_comparison_summary.csv"
    pd.DataFrame(rows).reindex(
        columns=["model_name", "dataset_name", "accuracy", "macro_f1", "weighted_f1", "num_samples", "num_labels", "notes"]
    ).to_csv(output_path, index=False)
    return output_path


def run_cuad_external_evaluation(
    paths: ProjectPaths,
    *,
    raw_cuad_json: dict[str, Any] | None = None,
    download_if_missing: bool = True,
    min_span_chars: int = 20,
    min_span_words: int = 4,
    run_transformer: bool = True,
    transformer_model_name: str = "distilbert-base-uncased",
    transformer_max_length: int = 256,
    transformer_batch_size: int = 16,
) -> dict[str, Any]:
    """Run the full CUAD external conversion, evaluation, and summary pipeline."""
    ensure_cuad_output_dirs(paths)
    if raw_cuad_json is None:
        cuad_json_path, master_clauses_path = download_cuad_if_missing(paths, download_if_missing=download_if_missing)
        raw_cuad_json, _ = load_cuad_raw_files(cuad_json_path, master_clauses_path)

    data_outputs = prepare_cuad_external_eval_data(
        paths,
        raw_cuad_json=raw_cuad_json,
        min_span_chars=min_span_chars,
        min_span_words=min_span_words,
    )
    cuad_eval_df: pd.DataFrame = data_outputs["eval_df"]
    label2id: dict[str, int] = data_outputs["label2id"]
    id2label: dict[int, str] = data_outputs["id2label"]
    safeguards = validate_cuad_protocol_safeguards(paths, cuad_eval_df, label2id)

    evaluated_results: list[dict[str, Any]] = []
    if cuad_eval_df.empty:
        reason = "CUAD external evaluation data is empty after conversion/mapping/label validation."
        for model_name in ("logistic_regression", "linear_svm", "multinomial_nb", transformer_model_name):
            write_model_status(paths, model_name=model_name, status="failed", reason=reason, num_samples=0)
        write_llm_external_statuses(paths, num_samples=0)
    else:
        evaluated_results.extend(evaluate_classical_models_on_cuad(cuad_eval_df, id2label, paths))
        transformer_result = evaluate_transformer_model_on_cuad(
            cuad_eval_df,
            id2label,
            paths,
            model_name=transformer_model_name,
            max_length=transformer_max_length,
            batch_size=transformer_batch_size,
            run_transformer=run_transformer,
        )
        if transformer_result is not None:
            evaluated_results.append(transformer_result)
        write_llm_external_statuses(paths, num_samples=len(cuad_eval_df))

    summary_path = build_final_model_comparison_summary(paths)
    run_summary = {
        "created_at_utc": _utc_now(),
        "dataset_name": CUAD_EXTERNAL_DATASET_NAME,
        "cuad_samples_extracted": int(data_outputs["preprocessing_report"].get("rows_after_deduplication", 0)),
        "cuad_samples_retained_after_mapping": int(data_outputs["mapping_report"].get("number_of_samples_retained_after_mapping", 0)),
        "cuad_samples_ready_for_external_eval": int(len(cuad_eval_df)),
        "cuad_labels_mapped": data_outputs["mapping_report"].get("mapped_labels", []),
        "cuad_labels_unmapped": data_outputs["mapping_report"].get("unmapped_labels", []),
        "models_evaluated_on_cuad": [result["model_name"] for result in evaluated_results if result.get("status") == "completed"],
        "safeguards": safeguards,
        "outputs": {
            "raw": str(data_outputs["paths"]["raw"]),
            "mapped": str(data_outputs["paths"]["mapped"]),
            "eval": str(data_outputs["paths"]["eval"]),
            "mapping_report": str(data_outputs["paths"]["mapping_report"]),
            "summary": str(summary_path),
        },
    }
    write_json(paths.project_root / "outputs" / "metrics" / "cuad_external_evaluation_summary.json", run_summary)
    return run_summary
