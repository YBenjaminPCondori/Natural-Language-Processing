"""Report artifact exports for the LEDGAR coursework notebook."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .data_setup import ProjectPaths, normalise_whitespace, write_json
from .evaluation import safe_name


STANDARD_PREDICTION_COLUMNS = [
    "text",
    "true_label",
    "true_label_id",
    "predicted_label",
    "predicted_label_id",
    "model_name",
    "split",
    "confidence",
]
QWEN_PREDICTION_COLUMNS = [
    "mode",
    "text",
    "label",
    "label_id",
    "raw_output",
    "predicted_label",
    "predicted_label_id",
    "is_invalid",
]
QWEN_RESULT_COLUMNS = [
    "model_family",
    "model_name",
    "training_type",
    "dataset",
    "eval_split",
    "sample_size",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "invalid_prediction_rate",
    "status",
    "notes",
]
TRANSFORMER_RESULT_COLUMNS = [
    "model_family",
    "model_name",
    "training_type",
    "dataset",
    "eval_split",
    "sample_size",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "status",
    "notes",
    "classification_report_path",
    "confusion_matrix_path",
]
STALE_OUTPUT_FILES = [
    "outputs/classical_results.jsonl",
    "outputs/classification_report_linear_svm.json",
    "outputs/classification_report_logistic_regression.json",
    "outputs/classification_report_transformer.json",
    "outputs/confusion_matrix_transformer.json",
    "outputs/predictions/random_baseline_test_predictions.jsonl",
    "outputs/figures/best_model_confusion_matrix.png",
    "outputs/figures/class_distribution.png",
    "outputs/figures/confusion_matrix_linear_svm.png",
    "outputs/figures/confusion_matrix_logistic_regression.png",
    "outputs/figures/final_macro_f1_comparison.png",
    "outputs/figures/final_weighted_f1_comparison.png",
    "outputs/figures/text_length_histogram.png",
]


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except Exception:
        return None


def _count_jsonl(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _artifact_for_manifest(value: Any, root: Path) -> Any:
    if isinstance(value, Path):
        return _safe_relative(value, root)
    if isinstance(value, dict):
        return {key: _artifact_for_manifest(inner, root) for key, inner in value.items()}
    if isinstance(value, list):
        return [_artifact_for_manifest(inner, root) for inner in value]
    return value


def cleanup_stale_report_outputs(paths: ProjectPaths) -> None:
    """Remove known legacy report artifacts that conflict with the regenerated run."""
    for relative_path in STALE_OUTPUT_FILES:
        path = paths.project_root / relative_path
        if path.exists() and path.is_file():
            path.unlink()


def _status_from_result(result: dict[str, Any]) -> str:
    metric = result.get("macro_f1")
    if pd.isna(metric):
        return "skipped_or_failed"
    return "completed"


def _normalise_prediction_table(model_name: str, pred_df: pd.DataFrame) -> pd.DataFrame:
    """Convert in-memory prediction tables to the stable report prediction schema."""
    if pred_df is None or pred_df.empty:
        return pd.DataFrame(columns=STANDARD_PREDICTION_COLUMNS)
    output = pd.DataFrame()
    output["text"] = pred_df.get("text", pd.Series(dtype=str)).astype(str)
    output["true_label"] = pred_df.get("true_label", pred_df.get("label", pd.Series(dtype=str)))
    output["true_label_id"] = pred_df.get("true_label_id", pred_df.get("label_id", pd.Series(dtype="Int64")))
    output["predicted_label"] = pred_df.get("predicted_label", pd.Series(dtype=str))
    output["predicted_label_id"] = pred_df.get("predicted_label_id", pd.Series(dtype="Int64"))
    output["model_name"] = model_name
    output["split"] = pred_df.get("split", "test")
    output["confidence"] = pred_df.get("confidence", pd.NA)
    return output.reindex(columns=STANDARD_PREDICTION_COLUMNS)


def _write_empty_csv(path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=columns).to_csv(path, index=False)


def collect_environment(seed: int, paths: ProjectPaths) -> dict[str, Any]:
    """Collect runtime details needed for reproducibility."""
    torch_info: dict[str, Any]
    try:
        import torch

        torch_info = {
            "torch_version": getattr(torch, "__version__", None),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_version": getattr(torch.version, "cuda", None),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        torch_info = {"torch_import_error": f"{type(exc).__name__}: {exc}"}

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(paths.project_root),
        "random_seed": seed,
        "python_version": sys.version,
        "platform": platform.platform(),
        **torch_info,
        "packages": {
            "pandas": _package_version("pandas"),
            "numpy": _package_version("numpy"),
            "scikit-learn": _package_version("scikit-learn"),
            "matplotlib": _package_version("matplotlib"),
            "datasets": _package_version("datasets"),
            "transformers": _package_version("transformers"),
            "accelerate": _package_version("accelerate"),
            "joblib": _package_version("joblib"),
        },
    }


def _word_and_char_stats(processed_splits: dict[str, pd.DataFrame]) -> dict[str, float | int | None]:
    if not processed_splits:
        return {
            "average_clause_length_words": None,
            "maximum_clause_length_words": None,
            "average_clause_length_characters": None,
            "maximum_clause_length_characters": None,
        }
    combined = pd.concat(processed_splits.values(), ignore_index=True)
    words = combined["text"].astype(str).str.split().str.len()
    chars = combined["text"].astype(str).str.len()
    return {
        "average_clause_length_words": float(words.mean()),
        "maximum_clause_length_words": int(words.max()),
        "average_clause_length_characters": float(chars.mean()),
        "maximum_clause_length_characters": int(chars.max()),
    }


def export_data_summary(
    paths: ProjectPaths,
    processed_splits: dict[str, pd.DataFrame],
    label_names: list[str],
    *,
    dataset_name: str,
) -> Path:
    """Save dataset summary values used by the report."""
    output_path = paths.project_root / "outputs" / "data_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_counts = {
        split: _count_jsonl(paths.ledgar_raw_dir / f"ledgar_{split}.jsonl")
        for split in ("train", "validation", "test")
    }
    processed_counts = {split: int(len(df)) for split, df in processed_splits.items()}
    length_stats = _word_and_char_stats(processed_splits)
    raw_total = sum(count for count in raw_counts.values() if count is not None)
    processed_total = sum(processed_counts.values())
    combined = pd.concat(processed_splits.values(), ignore_index=True) if processed_splits else pd.DataFrame()
    label_counts = combined["label"].value_counts().astype(int).to_dict() if not combined.empty else {}

    rows = [
        {"metric": "dataset", "value": dataset_name, "notes": "Main supervised dataset."},
        {
            "metric": "huggingface_loader",
            "value": 'load_dataset("coastalcph/lex_glue", "ledgar")',
            "notes": "Declared in modules/data_setup.py.",
        },
        {"metric": "raw_total_examples", "value": raw_total, "notes": "Counted from raw JSONL exports."},
        {"metric": "raw_train_examples", "value": raw_counts.get("train"), "notes": "Raw train split."},
        {"metric": "raw_validation_examples", "value": raw_counts.get("validation"), "notes": "Raw validation split."},
        {"metric": "raw_test_examples", "value": raw_counts.get("test"), "notes": "Raw test split."},
        {"metric": "filtered_total_examples", "value": processed_total, "notes": "After top-label filtering."},
        {"metric": "filtered_train_examples", "value": processed_counts.get("train"), "notes": "Filtered train split."},
        {
            "metric": "filtered_validation_examples",
            "value": processed_counts.get("validation"),
            "notes": "Filtered validation split.",
        },
        {"metric": "filtered_test_examples", "value": processed_counts.get("test"), "notes": "Filtered test split."},
        {"metric": "selected_label_count", "value": len(label_names), "notes": "Top labels selected from training split."},
        {
            "metric": "average_clause_length_words",
            "value": length_stats["average_clause_length_words"],
            "notes": "Processed examples, word count.",
        },
        {
            "metric": "maximum_clause_length_words",
            "value": length_stats["maximum_clause_length_words"],
            "notes": "Processed examples, word count.",
        },
        {
            "metric": "average_clause_length_characters",
            "value": length_stats["average_clause_length_characters"],
            "notes": "Processed examples, character count.",
        },
        {
            "metric": "maximum_clause_length_characters",
            "value": length_stats["maximum_clause_length_characters"],
            "notes": "Processed examples, character count.",
        },
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    write_json(
        paths.project_root / "outputs" / "dataset_summary.json",
        {
            "dataset": dataset_name,
            "rows_per_split": processed_counts,
            "unique_labels": len(label_names),
            "label_counts": label_counts,
            "raw_rows_per_split": raw_counts,
            "word_count_stats": {
                "mean": length_stats["average_clause_length_words"],
                "max": length_stats["maximum_clause_length_words"],
            },
            "character_count_stats": {
                "mean": length_stats["average_clause_length_characters"],
                "max": length_stats["maximum_clause_length_characters"],
            },
        },
    )
    return output_path


def export_label_distribution(
    paths: ProjectPaths,
    processed_splits: dict[str, pd.DataFrame],
    label2id: dict[str, int],
    label_names: list[str],
) -> Path:
    """Save selected-label counts by split."""
    output_path = paths.project_root / "outputs" / "label_distribution.csv"
    rows = []
    for label in label_names:
        row = {"label": label, "label_id": label2id[label]}
        total = 0
        for split in ("train", "validation", "test"):
            count = int((processed_splits.get(split, pd.DataFrame()).get("label", pd.Series(dtype=str)) == label).sum())
            row[f"{split}_count"] = count
            total += count
        row["total_count"] = total
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    outputs_dir = paths.project_root / "outputs"
    (outputs_dir / "label_names.txt").write_text("\n".join(label_names) + "\n", encoding="utf-8")
    write_json(outputs_dir / "selected_labels.json", label_names)
    write_json(
        outputs_dir / "label_mapping.json",
        {
            "label_to_id": {label: int(label_id) for label, label_id in label2id.items()},
            "id_to_label": {int(label_id): label for label, label_id in label2id.items()},
        },
    )
    return output_path


def export_main_results(paths: ProjectPaths, completed_results: list[dict[str, Any]]) -> Path:
    """Save report-ready model comparison table without dropping optional fields."""
    output_path = paths.project_root / "outputs" / "main_results.csv"
    columns = [
        "model_family",
        "model_name",
        "training_type",
        "dataset",
        "eval_split",
        "sample_size",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "invalid_prediction_rate",
        "notes",
        "classification_report_path",
        "confusion_matrix_path",
    ]
    results_df = pd.DataFrame(completed_results).reindex(columns=columns)
    results_df.to_csv(output_path, index=False)
    results_df.to_json(paths.project_root / "outputs" / "final_model_comparison.jsonl", orient="records", lines=True, force_ascii=False)
    return output_path


def export_hyperparameters(
    paths: ProjectPaths,
    *,
    seed: int,
    max_features_list: list[int] | None = None,
    ngram_ranges: list[tuple[int, int]] | None = None,
    transformer_model_name: str = "",
    max_transformer_length: int | None = None,
    qwen_model_name: str = "",
    qwen_eval_sample_size: int | None = None,
    qwen_few_shot_examples_per_class: int | None = None,
    run_naive_bayes: bool | None = None,
) -> Path:
    """Save model and runtime hyperparameters used by the notebook."""
    output_path = paths.project_root / "outputs" / "hyperparameters.csv"
    rows = [
        {"component": "general", "parameter": "random_seed", "value": seed, "notes": "Used for sampling and model seeds."},
        {
            "component": "tfidf",
            "parameter": "max_features_grid",
            "value": max_features_list if max_features_list is not None else "",
            "notes": "Classical feature grid.",
        },
        {
            "component": "tfidf",
            "parameter": "ngram_range_grid",
            "value": ngram_ranges if ngram_ranges is not None else "",
            "notes": "Classical feature grid.",
        },
        {"component": "tfidf", "parameter": "lowercase", "value": True, "notes": "Set in modules/classical_models.py."},
        {"component": "tfidf", "parameter": "stop_words", "value": None, "notes": "Legal stopwords are retained."},
        {"component": "logistic_regression", "parameter": "max_iter", "value": 2000, "notes": "Set in modules/classical_models.py."},
        {
            "component": "logistic_regression",
            "parameter": "class_weight_grid",
            "value": [None, "balanced"],
            "notes": "Validation-selected.",
        },
        {
            "component": "linear_svm",
            "parameter": "class_weight_grid",
            "value": [None, "balanced"],
            "notes": "Validation-selected.",
        },
        {"component": "multinomial_nb", "parameter": "enabled", "value": run_naive_bayes, "notes": "Optional classical comparison."},
        {
            "component": "transformer",
            "parameter": "model_name",
            "value": transformer_model_name,
            "notes": "Configured model; metrics exist only if the cell completes.",
        },
        {"component": "transformer", "parameter": "max_sequence_length", "value": max_transformer_length, "notes": "Tokenizer truncation length."},
        {"component": "transformer", "parameter": "learning_rate", "value": 2e-5, "notes": "Set in modules/transformer_model.py."},
        {"component": "transformer", "parameter": "epochs", "value": 3, "notes": "Set in modules/transformer_model.py."},
        {"component": "transformer", "parameter": "weight_decay", "value": 0.01, "notes": "Set in modules/transformer_model.py."},
        {
            "component": "transformer",
            "parameter": "batch_size_policy",
            "value": "16 if GPU memory > 12GB else 8",
            "notes": "Set in modules/transformer_model.py.",
        },
        {"component": "transformer", "parameter": "fp16", "value": "torch.cuda.is_available()", "notes": "Uses mixed precision only on CUDA."},
        {"component": "transformer", "parameter": "selection_metric", "value": "validation macro-F1", "notes": "Best model selection."},
        {"component": "qwen", "parameter": "model_name", "value": qwen_model_name, "notes": "Configured prompting model."},
        {"component": "qwen", "parameter": "eval_sample_size", "value": qwen_eval_sample_size, "notes": "Maximum sampled test examples."},
        {
            "component": "qwen",
            "parameter": "few_shot_examples_per_class",
            "value": qwen_few_shot_examples_per_class,
            "notes": "Training examples only.",
        },
        {"component": "qwen", "parameter": "max_new_tokens", "value": 24, "notes": "Set in modules/qwen_prompting.py."},
        {"component": "qwen", "parameter": "do_sample", "value": False, "notes": "Deterministic decoding."},
        {"component": "qwen", "parameter": "fuzzy_matching_cutoff", "value": 0.80, "notes": "Used only when unambiguous."},
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def export_per_class_results(paths: ProjectPaths, completed_results: list[dict[str, Any]]) -> Path:
    """Flatten saved classification reports into one CSV."""
    output_path = paths.project_root / "outputs" / "per_class_results.csv"
    rows = []
    for result in completed_results:
        report_path = result.get("classification_report_path")
        if not report_path or not Path(report_path).exists():
            continue
        report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        for label, metrics in report.items():
            if label in {"accuracy", "macro avg", "weighted avg"} or not isinstance(metrics, dict):
                continue
            if "f1-score" not in metrics:
                continue
            rows.append(
                {
                    "model_family": result.get("model_family"),
                    "model_name": result.get("model_name"),
                    "label": label,
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1_score": metrics.get("f1-score"),
                    "support": metrics.get("support"),
                    "classification_report_path": report_path,
                }
            )
    per_class = pd.DataFrame(rows)
    if not per_class.empty:
        per_class["f1_rank_within_model"] = per_class.groupby("model_name")["f1_score"].rank(ascending=False, method="dense")
    per_class.to_csv(output_path, index=False)
    return output_path


def export_error_tables(paths: ProjectPaths, error_outputs: dict[str, Any] | None) -> dict[str, Path]:
    """Save generic confusion and misclassification tables for the report."""
    outputs_dir = paths.project_root / "outputs"
    confusion_path = outputs_dir / "confusion_pairs.csv"
    misclassified_path = outputs_dir / "misclassified_examples.csv"
    error_outputs = error_outputs or {}

    confusions = error_outputs.get("classical_confusions", pd.DataFrame())
    if not isinstance(confusions, pd.DataFrame):
        confusions = pd.DataFrame(confusions)
    if not confusions.empty:
        confusions = confusions.rename(columns={"label": "true_label"}).copy()
        confusions.insert(0, "rank", range(1, len(confusions) + 1))
    confusions.to_csv(confusion_path, index=False)

    misclassified = error_outputs.get("classical_misclassified", pd.DataFrame())
    if not isinstance(misclassified, pd.DataFrame):
        misclassified = pd.DataFrame(misclassified)
    if not misclassified.empty:
        misclassified = misclassified.copy()
        misclassified["text_short"] = misclassified["text"].map(lambda value: normalise_whitespace(value)[:350])
        misclassified["analysis_note"] = "Manual analyst note required; do not auto-generate a reason as a result."
    misclassified.to_csv(misclassified_path, index=False)
    return {"confusion_pairs": confusion_path, "misclassified_examples": misclassified_path}


def export_prediction_tables(paths: ProjectPaths, prediction_tables: dict[str, pd.DataFrame]) -> dict[str, Path]:
    """Write current test prediction files for every completed model."""
    predictions_dir = paths.project_root / "outputs" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}
    for model_name, pred_df in sorted((prediction_tables or {}).items()):
        normalised = _normalise_prediction_table(model_name, pred_df)
        output_path = predictions_dir / f"{safe_name(model_name)}_test_predictions.jsonl"
        normalised.to_json(output_path, orient="records", lines=True, force_ascii=False)
        artifacts[model_name] = output_path
    return artifacts


def export_transformer_tables(
    paths: ProjectPaths,
    completed_results: list[dict[str, Any]],
    prediction_tables: dict[str, pd.DataFrame],
    transformer_model_name: str,
) -> dict[str, Path]:
    """Save stable transformer result/prediction tables, including skipped runs."""
    outputs_dir = paths.project_root / "outputs"
    transformer_rows = [
        {**result, "status": _status_from_result(result)}
        for result in completed_results
        if result.get("model_family") == "transformer"
    ]
    results_path = outputs_dir / "transformer_results.csv"
    pd.DataFrame(transformer_rows).reindex(columns=TRANSFORMER_RESULT_COLUMNS).to_csv(results_path, index=False)

    predictions_path = outputs_dir / "transformer_predictions.csv"
    pred_df = prediction_tables.get(transformer_model_name, pd.DataFrame()) if prediction_tables else pd.DataFrame()
    normalised = _normalise_prediction_table(transformer_model_name, pred_df)
    normalised.to_csv(predictions_path, index=False)
    return {"transformer_results": results_path, "transformer_predictions": predictions_path}


def export_qwen_results(paths: ProjectPaths, completed_results: list[dict[str, Any]]) -> Path:
    """Save stable Qwen result rows even when Qwen is skipped."""
    output_path = paths.project_root / "outputs" / "qwen_results.csv"
    rows = [
        {**result, "status": _status_from_result(result)}
        for result in completed_results
        if result.get("model_family") == "llm_prompting" or "qwen" in str(result.get("model_name", "")).lower()
    ]
    pd.DataFrame(rows).reindex(columns=QWEN_RESULT_COLUMNS).to_csv(output_path, index=False)
    return output_path


def export_qwen_tables(
    paths: ProjectPaths,
    qwen_predictions_df: pd.DataFrame | None,
    qwen_invalid_outputs_df: pd.DataFrame | None,
) -> dict[str, Path]:
    """Copy Qwen prediction tables to stable report filenames."""
    outputs_dir = paths.project_root / "outputs"
    predictions_path = outputs_dir / "qwen_predictions.csv"
    invalid_path = outputs_dir / "qwen_invalid_outputs.csv"
    predictions = qwen_predictions_df if qwen_predictions_df is not None else pd.DataFrame()
    invalid = qwen_invalid_outputs_df if qwen_invalid_outputs_df is not None else pd.DataFrame()
    if predictions.empty:
        predictions = pd.DataFrame(columns=QWEN_PREDICTION_COLUMNS)
    else:
        predictions = predictions.copy()
        if "predicted_label_id" not in predictions.columns and "predicted_label" in predictions.columns:
            predictions["predicted_label_id"] = pd.NA
        if "is_invalid" not in predictions.columns and "predicted_label" in predictions.columns:
            predictions["is_invalid"] = predictions["predicted_label"].eq("INVALID_PREDICTION")
        predictions = predictions.reindex(columns=QWEN_PREDICTION_COLUMNS)
    if invalid.empty:
        invalid = pd.DataFrame(columns=QWEN_PREDICTION_COLUMNS)
    else:
        invalid = invalid.reindex(columns=QWEN_PREDICTION_COLUMNS)
    predictions.to_csv(predictions_path, index=False)
    invalid.to_csv(invalid_path, index=False)
    return {"qwen_predictions": predictions_path, "qwen_invalid_outputs": invalid_path}


def export_qwen_prompt_examples(
    paths: ProjectPaths,
    processed_splits: dict[str, pd.DataFrame],
    label_names: list[str],
) -> Path:
    """Save prompt examples to a stable report-facing path."""
    output_path = paths.project_root / "outputs" / "qwen_prompt_examples.txt"
    source_path = paths.results_dir / "qwen" / "qwen_prompt_examples.txt"
    if source_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)
        return output_path

    test_df = processed_splits.get("test", pd.DataFrame())
    train_df = processed_splits.get("train", pd.DataFrame())
    if test_df.empty or not label_names:
        output_path.write_text("Qwen prompt examples unavailable because processed LEDGAR examples are unavailable.\n", encoding="utf-8")
        return output_path

    from .qwen_prompting import build_few_shot_examples, make_few_shot_prompt, make_zero_shot_prompt

    clause_text = str(test_df.iloc[0]["text"])
    few_shot_examples = build_few_shot_examples(train_df, label_names, examples_per_class=1)
    prompt_text = "\n\n".join(
        [
            "--- zero_shot prompt example ---",
            make_zero_shot_prompt(clause_text, label_names),
            "--- few_shot prompt example ---",
            make_few_shot_prompt(clause_text, label_names, few_shot_examples),
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt_text + "\n", encoding="utf-8")
    return output_path


def _copy_figure(source: Path | None, targets: list[Path]) -> Path | None:
    if source is None or not source.exists():
        return None
    copied = None
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied = target
    return copied


def _find_first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def plot_qwen_invalid_predictions(comparison_df: pd.DataFrame, output_path: Path) -> Path | None:
    """Plot Qwen invalid prediction rates when Qwen was run."""
    if comparison_df.empty or "invalid_prediction_rate" not in comparison_df:
        return None
    qwen_rows = comparison_df[
        comparison_df["model_name"].astype(str).str.contains("qwen", case=False, na=False)
        & comparison_df["invalid_prediction_rate"].notna()
    ]
    if qwen_rows.empty:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    qwen_rows.plot(kind="bar", x="model_name", y="invalid_prediction_rate", ax=ax, legend=False)
    ax.set_title("Qwen Invalid Prediction Rate")
    ax.set_xlabel("Prompting mode")
    ax.set_ylabel("Invalid prediction rate")
    ax.set_ylim(0, max(1.0, float(qwen_rows["invalid_prediction_rate"].max()) * 1.2))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_agentic_workflow(output_path: Path) -> Path:
    """Save a simple confidence-review workflow diagram."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.axis("off")
    labels = [
        "Clause text",
        "Best classifier",
        "Confidence check",
        "Human review if low",
        "Research triage output",
    ]
    xs = [0.08, 0.29, 0.50, 0.71, 0.90]
    for x, label in zip(xs, labels):
        ax.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#eef3f8", "edgecolor": "#385a7c"},
            fontsize=9,
        )
    for start, end in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(end - 0.075, 0.55), xytext=(start + 0.075, 0.55), arrowprops={"arrowstyle": "->"})
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def export_report_figures(paths: ProjectPaths, comparison_df: pd.DataFrame, error_outputs: dict[str, Any] | None) -> dict[str, Path | None]:
    """Create/copy figures to the exact names used by report.tex."""
    root_figures = paths.project_root / "figures"
    output_figures = paths.project_root / "outputs" / "figures"
    results_dir = paths.results_dir
    error_outputs = error_outputs or {}

    figure_map: dict[str, Path | None] = {}
    report_targets = lambda name: [root_figures / name, output_figures / name]

    class_source = _find_first_existing(
        [
            results_dir / "eda" / "class_distribution.png",
            paths.project_root / "outputs" / "figures" / "class_distribution.png",
        ]
    )
    figure_map["label_distribution"] = _copy_figure(class_source, report_targets("label_distribution.png"))

    length_source = _find_first_existing(
        [
            results_dir / "eda" / "clause_length_histogram.png",
            paths.project_root / "outputs" / "figures" / "text_length_histogram.png",
        ]
    )
    figure_map["clause_length_distribution"] = _copy_figure(length_source, report_targets("clause_length_distribution.png"))

    comparison_source = _find_first_existing(
        [
            results_dir / "final_model_comparison.png",
            paths.project_root / "outputs" / "figures" / "final_macro_f1_comparison.png",
        ]
    )
    figure_map["model_comparison_macro_f1"] = _copy_figure(comparison_source, report_targets("model_comparison_macro_f1.png"))

    best_cm = error_outputs.get("confusion_matrix_path")
    best_source = Path(best_cm) if best_cm else None
    if (best_source is None or not best_source.exists()) and not comparison_df.empty:
        completed = comparison_df.dropna(subset=["macro_f1"]).sort_values("macro_f1", ascending=False)
        if not completed.empty:
            candidate = completed.iloc[0].get("confusion_matrix_path")
            best_source = Path(candidate) if candidate else None
    if best_source is None or not best_source.exists():
        best_source = _find_first_existing([paths.project_root / "outputs" / "figures" / "best_model_confusion_matrix.png"])
    figure_map["confusion_matrix_best_model"] = _copy_figure(best_source, report_targets("confusion_matrix_best_model.png"))

    qwen_invalid = plot_qwen_invalid_predictions(comparison_df, root_figures / "qwen_invalid_predictions.png")
    if qwen_invalid is not None:
        _copy_figure(qwen_invalid, [output_figures / "qwen_invalid_predictions.png"])
    else:
        for stale_path in (root_figures / "qwen_invalid_predictions.png", output_figures / "qwen_invalid_predictions.png"):
            if stale_path.exists() and stale_path.is_file():
                stale_path.unlink()
    figure_map["qwen_invalid_predictions"] = qwen_invalid

    workflow = plot_agentic_workflow(root_figures / "agentic_review_workflow.png")
    _copy_figure(workflow, [output_figures / "agentic_review_workflow.png"])
    figure_map["agentic_review_workflow"] = workflow
    return figure_map


def export_report_artifacts(
    *,
    paths: ProjectPaths,
    processed_splits: dict[str, pd.DataFrame],
    label2id: dict[str, int],
    id2label: dict[int, str],
    completed_results: list[dict[str, Any]],
    prediction_tables: dict[str, pd.DataFrame],
    error_outputs: dict[str, Any] | None = None,
    qwen_predictions_df: pd.DataFrame | None = None,
    qwen_invalid_outputs_df: pd.DataFrame | None = None,
    seed: int = 42,
    dataset_name: str = "LEDGAR",
    max_features_list: list[int] | None = None,
    ngram_ranges: list[tuple[int, int]] | None = None,
    transformer_model_name: str = "",
    max_transformer_length: int | None = None,
    qwen_model_name: str = "",
    qwen_eval_sample_size: int | None = None,
    qwen_few_shot_examples_per_class: int | None = None,
    run_naive_bayes: bool | None = None,
) -> dict[str, Any]:
    """Export all report-facing tables, figures, and environment metadata."""
    outputs_dir = paths.project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    cleanup_stale_report_outputs(paths)
    label_names = [id2label[index] for index in sorted(id2label)] if id2label else []

    artifacts: dict[str, Any] = {}
    artifacts["data_summary"] = export_data_summary(paths, processed_splits, label_names, dataset_name=dataset_name)
    artifacts["label_distribution"] = export_label_distribution(paths, processed_splits, label2id, label_names)
    artifacts["main_results"] = export_main_results(paths, completed_results)
    artifacts["predictions"] = export_prediction_tables(paths, prediction_tables)
    artifacts.update(export_transformer_tables(paths, completed_results, prediction_tables, transformer_model_name))
    artifacts["hyperparameters"] = export_hyperparameters(
        paths,
        seed=seed,
        max_features_list=max_features_list,
        ngram_ranges=ngram_ranges,
        transformer_model_name=transformer_model_name,
        max_transformer_length=max_transformer_length,
        qwen_model_name=qwen_model_name,
        qwen_eval_sample_size=qwen_eval_sample_size,
        qwen_few_shot_examples_per_class=qwen_few_shot_examples_per_class,
        run_naive_bayes=run_naive_bayes,
    )
    artifacts["per_class_results"] = export_per_class_results(paths, completed_results)
    artifacts.update(export_error_tables(paths, error_outputs))
    artifacts["qwen_results"] = export_qwen_results(paths, completed_results)
    artifacts.update(export_qwen_tables(paths, qwen_predictions_df, qwen_invalid_outputs_df))
    artifacts["qwen_prompt_examples"] = export_qwen_prompt_examples(paths, processed_splits, label_names)

    comparison_df = pd.DataFrame(completed_results)
    artifacts["figures"] = export_report_figures(paths, comparison_df, error_outputs)

    environment = collect_environment(seed, paths)
    artifacts["environment"] = write_json(outputs_dir / "environment.json", environment)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {key: _artifact_for_manifest(value, paths.project_root) for key, value in artifacts.items()},
    }
    artifacts["manifest"] = write_json(outputs_dir / "report_artifact_manifest.json", manifest)
    return artifacts
