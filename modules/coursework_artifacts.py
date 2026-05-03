"""Repository audit and canonical coursework artifact generation."""

from __future__ import annotations

import importlib
import json
import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "model_name",
    "model_family",
    "training_type",
    "split_used",
    "validation_accuracy",
    "validation_macro_f1",
    "test_accuracy",
    "test_macro_f1",
    "test_weighted_f1",
    "test_macro_precision",
    "test_macro_recall",
    "invalid_prediction_rate",
    "selected_by_validation",
    "hyperparameter_source",
    "evidence_path",
    "prediction_path",
    "status",
    "reason_if_skipped_or_failed",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def backup_existing(path: Path, archive_dir: Path) -> None:
    if not path.exists() or path.is_dir():
        return
    destination = archive_dir / path.relative_to(path.parents[1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)


def write_text(path: Path, text: str, archive_dir: Path | None = None) -> Path:
    if archive_dir:
        backup_existing(path, archive_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_json(path: Path, payload: Any, archive_dir: Path | None = None) -> Path:
    return write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n", archive_dir)


def write_csv(path: Path, df: pd.DataFrame, archive_dir: Path | None = None) -> Path:
    if archive_dir:
        backup_existing(path, archive_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def markdown_table(df: pd.DataFrame, *, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    frame = df.head(max_rows).fillna("")
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in frame.iterrows():
        values = [str(row[column]).replace("\n", " ") for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines)


def resolve_evidence_path(project_root: Path, value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    raw = str(value).replace("\\", "/")
    for marker in ["results/", "outputs/", "figures/", "data/processed/", "checkpoints/", "models/"]:
        if marker in raw:
            rel = marker + raw.split(marker, 1)[1]
            return rel if (project_root / rel).exists() else rel
    return raw if raw else ""


def file_exists(project_root: Path, value: Any) -> bool:
    rel = resolve_evidence_path(project_root, value)
    return bool(rel and (project_root / rel).exists() and (project_root / rel).stat().st_size > 0)


def prediction_path_for_model(project_root: Path, model_name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", model_name.lower()).strip("_")
    candidates = [
        project_root / "outputs" / "predictions" / f"{safe}_test_predictions.jsonl",
        project_root / "outputs" / f"{safe}_predictions.csv",
        project_root / "results" / "qwen" / "qwen_predictions.csv",
        project_root / "results" / "transformer" / "transformer_predictions.csv",
    ]
    if model_name in {"qwen_zero_shot", "qwen_few_shot", "qwen_static_few_shot", "qwen_retrieval_few_shot"}:
        candidates.insert(0, project_root / "outputs" / "qwen_predictions.csv")
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate.relative_to(project_root).as_posix()
    return ""


def load_report_metrics(project_root: Path, report_path: Any) -> tuple[float | None, float | None]:
    rel = resolve_evidence_path(project_root, report_path)
    path = project_root / rel
    if not path.exists():
        return None, None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None
    macro = report.get("macro avg", {})
    return macro.get("precision"), macro.get("recall")


def transformer_validation_metrics(project_root: Path) -> dict[str, float]:
    path = project_root / "results" / "transformer" / "training_log_history.csv"
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    eval_rows = frame.dropna(subset=["eval_macro_f1"]) if "eval_macro_f1" in frame else pd.DataFrame()
    if eval_rows.empty:
        return {}
    best = eval_rows.sort_values("eval_macro_f1", ascending=False).iloc[0]
    return {
        "validation_accuracy": float(best.get("eval_accuracy")) if pd.notna(best.get("eval_accuracy")) else np.nan,
        "validation_macro_f1": float(best.get("eval_macro_f1")),
    }


def canonical_results(project_root: Path, archive_dir: Path) -> pd.DataFrame:
    source = project_root / "results" / "final_model_comparison.csv"
    if not source.exists():
        source = project_root / "outputs" / "archive"
        archived = sorted(source.glob("*/outputs/main_results.csv"), key=lambda path: path.stat().st_mtime, reverse=True) if source.exists() else []
        source = archived[0] if archived else project_root / "outputs" / "main_results.csv"
    existing = pd.read_csv(source) if source.exists() else pd.DataFrame()
    classical = pd.read_csv(project_root / "results" / "classical" / "classical_results.csv") if (project_root / "results" / "classical" / "classical_results.csv").exists() else pd.DataFrame()
    transformer_val = transformer_validation_metrics(project_root)

    rows: list[dict[str, Any]] = []
    for _, result in existing.iterrows():
        model_name = str(result.get("model_name"))
        raw_family = str(result.get("model_family"))
        family = {
            "transformer": "transformer_encoder",
            "llm_prompting": "llm_prompting",
            "classical": "classical",
            "baseline": "baseline",
        }.get(raw_family, raw_family)
        training_type = {
            "baseline": "dummy",
            "classical": "classical",
            "transformer_encoder": "fine_tuned_encoder",
            "llm_prompting": "prompted_llm",
        }.get(family, result.get("training_type"))
        report_path = resolve_evidence_path(project_root, result.get("classification_report_path"))
        cm_path = resolve_evidence_path(project_root, result.get("confusion_matrix_path"))
        precision, recall = load_report_metrics(project_root, result.get("classification_report_path"))
        val_accuracy = np.nan
        val_macro = np.nan
        selected = False
        hyper_source = "fixed_or_not_applicable"

        if family == "classical" and not classical.empty:
            match = classical[classical["model_name"] == model_name]
            if not match.empty:
                val_macro = float(match.iloc[0].get("validation_macro_f1"))
                selected = True
                hyper_source = "results/classical/classical_validation_grid.csv"
        elif family == "transformer_encoder":
            val_accuracy = transformer_val.get("validation_accuracy", np.nan)
            val_macro = transformer_val.get("validation_macro_f1", np.nan)
            selected = True
            hyper_source = "results/transformer/training_args.json"
        elif family == "llm_prompting":
            hyper_source = "results/qwen/qwen_run_config.json"

        invalid_rate = result.get("invalid_prediction_rate", np.nan)
        status = "completed"
        reason = ""
        if pd.isna(result.get("macro_f1")):
            status = "skipped"
            reason = str(result.get("notes", "metric missing"))
        if model_name == "qwen_few_shot" and pd.notna(invalid_rate) and float(invalid_rate) >= 1.0:
            status = "failed"
            reason = "All Qwen few-shot outputs were invalid; do not treat as a meaningful completed result."

        evidence_parts = [part for part in [report_path, cm_path] if part]
        rows.append(
            {
                "model_name": model_name,
                "model_family": family,
                "training_type": training_type,
                "split_used": result.get("eval_split", "test"),
                "validation_accuracy": val_accuracy,
                "validation_macro_f1": val_macro,
                "test_accuracy": result.get("accuracy"),
                "test_macro_f1": result.get("macro_f1"),
                "test_weighted_f1": result.get("weighted_f1"),
                "test_macro_precision": precision,
                "test_macro_recall": recall,
                "invalid_prediction_rate": invalid_rate,
                "selected_by_validation": selected,
                "hyperparameter_source": hyper_source,
                "evidence_path": "; ".join(evidence_parts),
                "prediction_path": prediction_path_for_model(project_root, model_name),
                "status": status,
                "reason_if_skipped_or_failed": reason,
            }
        )

    configured_pending = [
        ("bilstm", "neural_sequence", "neural_sequence", "outputs/bilstm_results.json", "Configured in modules/sequence_model.py; run required."),
        ("bert-base-uncased", "transformer_encoder", "fine_tuned_encoder", "outputs/hyperparameter_search_results.csv", "Runnable configuration pending; no completed evidence found."),
        ("nlpaueb/legal-bert-base-uncased", "transformer_encoder", "fine_tuned_encoder", "outputs/hyperparameter_search_results.csv", "Runnable configuration pending; no completed evidence found."),
        ("nlpaueb/bert-base-uncased-contracts", "transformer_encoder", "fine_tuned_encoder", "outputs/hyperparameter_search_results.csv", "Runnable configuration pending; no completed evidence found."),
        ("roberta-base", "transformer_encoder", "fine_tuned_encoder", "outputs/hyperparameter_search_results.csv", "Runnable configuration pending; no completed evidence found."),
        ("microsoft/deberta-v3-base", "transformer_encoder", "fine_tuned_encoder", "outputs/hyperparameter_search_results.csv", "Runnable configuration pending; no completed evidence found."),
        ("qwen_retrieval_few_shot", "llm_prompting", "retrieval_few_shot", "outputs/qwen_retrieval_few_shot_predictions.csv", "Configured in modules/qwen_prompting.py; rerun Qwen required."),
    ]
    completed_names = {row["model_name"] for row in rows}
    for model_name, family, training_type, hyper_source, reason in configured_pending:
        if model_name in completed_names:
            continue
        rows.append(
            {
                "model_name": model_name,
                "model_family": family,
                "training_type": training_type,
                "split_used": "test",
                "validation_accuracy": np.nan,
                "validation_macro_f1": np.nan,
                "test_accuracy": np.nan,
                "test_macro_f1": np.nan,
                "test_weighted_f1": np.nan,
                "test_macro_precision": np.nan,
                "test_macro_recall": np.nan,
                "invalid_prediction_rate": np.nan,
                "selected_by_validation": False,
                "hyperparameter_source": hyper_source,
                "evidence_path": "",
                "prediction_path": "",
                "status": "pending",
                "reason_if_skipped_or_failed": reason,
            }
        )

    frame = pd.DataFrame(rows).reindex(columns=CANONICAL_COLUMNS)
    write_csv(project_root / "outputs" / "main_results.csv", frame, archive_dir)
    write_json(project_root / "outputs" / "main_results.json", frame.replace({np.nan: None}).to_dict(orient="records"), archive_dir)
    write_text(project_root / "outputs" / "main_results_latex.tex", frame.to_latex(index=False, escape=True), archive_dir)
    write_text(project_root / "outputs" / "model_evidence_table.md", markdown_table(frame[["model_name", "status", "evidence_path", "prediction_path", "reason_if_skipped_or_failed"]], max_rows=50), archive_dir)
    return frame


def generate_dataset_verification(project_root: Path, archive_dir: Path) -> dict[str, Any]:
    raw = {split: read_jsonl(project_root / "data" / "raw" / "lexglue_ledgar" / f"ledgar_{split}.jsonl") for split in ("train", "validation", "test")}
    processed = {split: read_jsonl(project_root / "data" / "processed" / f"ledgar_{split}.jsonl") for split in ("train", "validation", "test")}
    label_names = [line.strip() for line in (project_root / "data" / "processed" / "label_names.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_label_path = project_root / "data" / "raw" / "lexglue_ledgar" / "label_names.txt"
    raw_label_names = [line.strip() for line in raw_label_path.read_text(encoding="utf-8").splitlines() if line.strip()] if raw_label_path.exists() else []
    train_labels = raw["train"]["label"].copy() if "label" in raw["train"] else pd.Series(dtype=str)
    if raw_label_names and pd.api.types.is_integer_dtype(train_labels):
        train_labels = train_labels.map(lambda label_id: raw_label_names[int(label_id)] if 0 <= int(label_id) < len(raw_label_names) else str(label_id))
    train_top20 = train_labels.value_counts().head(20).index.tolist()
    processed_labels = processed["train"]["label"].value_counts().index.tolist() if "label" in processed["train"] else []

    overlap = {}
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        left_pairs = set(zip(processed[left]["text"].astype(str), processed[left]["label"].astype(str), strict=False))
        right_pairs = set(zip(processed[right]["text"].astype(str), processed[right]["label"].astype(str), strict=False))
        overlap[f"{left}_vs_{right}"] = len(left_pairs & right_pairs)

    combined = pd.concat(processed.values(), ignore_index=True)
    combined["word_count"] = combined["text"].astype(str).str.split().str.len()
    combined["char_count"] = combined["text"].astype(str).str.len()

    label_rows = []
    for label in label_names:
        row = {"label": label}
        row["train_count"] = int((processed["train"]["label"] == label).sum())
        row["validation_count"] = int((processed["validation"]["label"] == label).sum())
        row["test_count"] = int((processed["test"]["label"] == label).sum())
        row["total_count"] = row["train_count"] + row["validation_count"] + row["test_count"]
        label_rows.append(row)
    label_df = pd.DataFrame(label_rows)
    length_df = combined.groupby("split").agg(
        examples=("text", "size"),
        mean_words=("word_count", "mean"),
        median_words=("word_count", "median"),
        max_words=("word_count", "max"),
        mean_chars=("char_count", "mean"),
        max_chars=("char_count", "max"),
    ).reset_index()

    verification = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_rows": {split: int(len(df)) for split, df in raw.items()},
        "processed_rows": {split: int(len(df)) for split, df in processed.items()},
        "number_of_labels": int(len(label_names)),
        "processed_labels": label_names,
        "top20_selected_from_training_only": train_top20 == label_names or set(label_names).issubset(set(train_top20)),
        "validation_test_labels_subset_of_train_selected": all(set(processed[split]["label"]).issubset(set(label_names)) for split in ("validation", "test")),
        "text_label_cross_split_overlaps": overlap,
        "label_id_consistent": all(
            processed[split].groupby("label")["label_id"].nunique().max() == 1 for split in ("train", "validation", "test") if not processed[split].empty
        ),
        "length_statistics": length_df.to_dict(orient="records"),
    }

    write_json(project_root / "outputs" / "dataset_verification.json", verification, archive_dir)
    write_csv(project_root / "outputs" / "label_distribution_top20.csv", label_df, archive_dir)
    write_csv(project_root / "outputs" / "text_length_statistics.csv", length_df, archive_dir)

    fig, ax = plt.subplots(figsize=(12, 5))
    label_df.sort_values("train_count", ascending=False).plot(kind="bar", x="label", y=["train_count", "validation_count", "test_count"], ax=ax)
    ax.set_title("LEDGAR Top-20 Label Distribution")
    ax.set_ylabel("Examples")
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    for rel in ["figures/label_distribution_top20.png", "figures/label_distribution.png"]:
        fig.savefig(project_root / rel, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    combined["word_count"].clip(upper=1000).hist(bins=50, ax=ax)
    ax.set_title("LEDGAR Clause Length Distribution")
    ax.set_xlabel("Words, clipped at 1000")
    ax.set_ylabel("Examples")
    fig.tight_layout()
    for rel in ["figures/text_length_distribution.png", "figures/clause_length_distribution.png"]:
        fig.savefig(project_root / rel, dpi=150, bbox_inches="tight")
    plt.close(fig)

    md = [
        "# Dataset Verification",
        "",
        f"Created: {verification['created_at_utc']}",
        "",
        "## Split Counts",
        markdown_table(pd.DataFrame([{"split": split, "raw_rows": len(raw[split]), "processed_rows": len(processed[split])} for split in ("train", "validation", "test")])),
        "",
        "## Integrity Checks",
        f"- Labels selected from training split only: `{verification['top20_selected_from_training_only']}`",
        f"- Validation/test labels subset of selected train labels: `{verification['validation_test_labels_subset_of_train_selected']}`",
        f"- Text+label cross-split overlaps after de-duplication: `{overlap}`",
        f"- Label IDs consistent: `{verification['label_id_consistent']}`",
    ]
    write_text(project_root / "outputs" / "dataset_verification.md", "\n".join(md) + "\n", archive_dir)
    return verification


def generate_tuning_artifacts(project_root: Path, archive_dir: Path) -> None:
    classical_grid = project_root / "results" / "classical" / "classical_validation_grid.csv"
    if classical_grid.exists():
        grid = pd.read_csv(classical_grid)
    else:
        grid = pd.DataFrame(columns=["model_name", "validation_macro_f1", "max_features", "ngram_range", "min_df", "C", "alpha", "class_weight"])
    write_csv(project_root / "outputs" / "classical_hyperparameter_results.csv", grid, archive_dir)
    best_configs = {}
    if not grid.empty and "validation_macro_f1" in grid:
        for model_name, group in grid.groupby("model_name"):
            best_configs[model_name] = group.sort_values("validation_macro_f1", ascending=False).iloc[0].replace({np.nan: None}).to_dict()
    write_json(project_root / "outputs" / "best_classical_configs.json", best_configs, archive_dir)

    transformer_rows = []
    completed = project_root / "results" / "transformer" / "transformer_results.csv"
    if completed.exists():
        frame = pd.read_csv(completed)
        for _, row in frame.iterrows():
            transformer_rows.append({"model_name": row["model_name"], "status": "completed", "validation_macro_f1": transformer_validation_metrics(project_root).get("validation_macro_f1"), "test_macro_f1": row.get("macro_f1"), "config_path": "results/transformer/training_args.json", "reason": ""})
    for model_name in ["bert-base-uncased", "nlpaueb/legal-bert-base-uncased", "nlpaueb/bert-base-uncased-contracts", "roberta-base", "microsoft/deberta-v3-base"]:
        transformer_rows.append({"model_name": model_name, "status": "pending", "validation_macro_f1": np.nan, "test_macro_f1": np.nan, "config_path": "outputs/best_transformer_configs.json", "reason": "Configured search space only; no completed checkpoint/result found."})
    transformer_df = pd.DataFrame(transformer_rows)
    write_csv(project_root / "outputs" / "hyperparameter_search_results.csv", transformer_df, archive_dir)
    search_space = {
        "learning_rate": [1e-5, 2e-5, 3e-5, 5e-5],
        "batch_size": [8, 16, 32],
        "epochs": [2, 3, 4, 5],
        "weight_decay": [0.0, 0.01, 0.05],
        "warmup_ratio": [0.0, 0.06, 0.1],
        "max_length": [128, 256, 512],
        "objective": "validation_macro_f1",
        "test_policy": "Test split is used only after validation selection.",
    }
    write_json(project_root / "outputs" / "best_transformer_configs.json", {"distilbert-base-uncased": {"source": "results/transformer/training_args.json"}, "pending_search_space": search_space}, archive_dir)
    write_text(project_root / "outputs" / "transformer_training_summary.md", "# Transformer Training Summary\n\n" + markdown_table(transformer_df, max_rows=20) + "\n", archive_dir)


def generate_llm_prompt_artifacts(project_root: Path, archive_dir: Path) -> None:
    qwen_predictions_path = project_root / "outputs" / "qwen_predictions.csv"
    qwen_results_path = project_root / "outputs" / "qwen_results.csv"
    predictions = pd.read_csv(qwen_predictions_path) if qwen_predictions_path.exists() else pd.DataFrame()
    results = pd.read_csv(qwen_results_path) if qwen_results_path.exists() else pd.DataFrame()
    if "mode" in predictions:
        zero_shot = predictions[predictions["mode"] == "zero_shot"]
        static_few_shot = predictions[predictions["mode"].isin(["few_shot", "static_few_shot"])]
        retrieval_few_shot = predictions[predictions["mode"] == "retrieval_few_shot"]
    else:
        zero_shot = static_few_shot = retrieval_few_shot = pd.DataFrame(columns=predictions.columns)
    write_csv(project_root / "outputs" / "qwen_zero_shot_predictions.csv", zero_shot, archive_dir)
    write_csv(project_root / "outputs" / "qwen_static_few_shot_predictions.csv", static_few_shot, archive_dir)
    write_csv(project_root / "outputs" / "qwen_retrieval_few_shot_predictions.csv", retrieval_few_shot, archive_dir)

    if not results.empty and {"model_name", "invalid_prediction_rate"}.issubset(results.columns):
        results = results.copy()
        invalid_rates = pd.to_numeric(results["invalid_prediction_rate"], errors="coerce")
        failed_mask = results["model_name"].astype(str).str.contains("few", case=False, na=False) & invalid_rates.ge(1.0)
        results.loc[failed_mask, "status"] = "failed"
        results.loc[failed_mask, "reason"] = "All outputs invalid under the recorded parser; do not report as meaningful completed result."
    write_csv(project_root / "outputs" / "llm_prompt_results.csv", results, archive_dir)

    few_status = "not present"
    if not results.empty and "model_name" in results:
        few_rows = results[results["model_name"].astype(str).str.contains("few", case=False, na=False)]
        if not few_rows.empty:
            invalid_rate = pd.to_numeric(few_rows.iloc[0].get("invalid_prediction_rate"), errors="coerce")
            few_status = "failed" if pd.notna(invalid_rate) and invalid_rate >= 1.0 else "completed"
    audit = [
        "# Qwen Prompt Audit",
        "",
        "- Qwen is prompt-based inference only; no Qwen fine-tuning is evidenced.",
        "- Current parser code now accepts only exact allowed-label matches after whitespace/case normalisation.",
        "- Retrieval few-shot prompts are configured to retrieve examples from the training split only.",
        f"- Current Qwen few-shot status from existing artifacts: `{few_status}`.",
        "- Rerun Qwen on CUDA to produce strict-parser zero/static/retrieval few-shot artifacts.",
    ]
    write_text(project_root / "outputs" / "qwen_prompt_audit.md", "\n".join(audit) + "\n", archive_dir)


def generate_comparison_figures(project_root: Path, main_results: pd.DataFrame, archive_dir: Path) -> None:
    completed = main_results[(main_results["status"] == "completed") & main_results["test_macro_f1"].notna()].copy()
    if completed.empty:
        return
    for metric, filename, title in [
        ("test_macro_f1", "model_comparison_macro_f1.png", "Model Comparison by Test Macro-F1"),
        ("test_accuracy", "model_comparison_accuracy.png", "Model Comparison by Test Accuracy"),
    ]:
        fig, ax = plt.subplots(figsize=(12, 5))
        completed.sort_values(metric, ascending=False).plot(kind="bar", x="model_name", y=metric, ax=ax, legend=False)
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        for base in ["figures", "outputs/figures"]:
            path = project_root / base / filename
            backup_existing(path, archive_dir)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    val_test = main_results[main_results["validation_macro_f1"].notna() & main_results["test_macro_f1"].notna()].copy()
    if not val_test.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(val_test["validation_macro_f1"], val_test["test_macro_f1"])
        for _, row in val_test.iterrows():
            ax.annotate(row["model_name"], (row["validation_macro_f1"], row["test_macro_f1"]), fontsize=8)
        ax.set_xlabel("Validation macro-F1")
        ax.set_ylabel("Test macro-F1")
        ax.set_title("Validation vs Test Macro-F1")
        fig.tight_layout()
        path = project_root / "figures" / "validation_vs_test_macro_f1.png"
        backup_existing(path, archive_dir)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def generate_error_artifacts(project_root: Path, main_results: pd.DataFrame, archive_dir: Path) -> None:
    per_class = pd.read_csv(project_root / "outputs" / "per_class_results.csv") if (project_root / "outputs" / "per_class_results.csv").exists() else pd.DataFrame()
    if not per_class.empty:
        write_csv(project_root / "outputs" / "per_label_f1_all_models.csv", per_class, archive_dir)
        completed = main_results[(main_results["status"] == "completed") & main_results["test_macro_f1"].notna()]
        best_model = completed.sort_values("test_macro_f1", ascending=False).iloc[0]["model_name"] if not completed.empty else None
        best = per_class[per_class["model_name"] == best_model].copy() if best_model else pd.DataFrame()
        write_csv(project_root / "outputs" / "per_label_f1_best_model.csv", best, archive_dir)
        if not best.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            best.sort_values("f1_score").plot(kind="bar", x="label", y="f1_score", ax=ax, legend=False)
            ax.set_title(f"Per-Label F1 for {best_model}")
            ax.set_ylabel("F1")
            ax.tick_params(axis="x", labelrotation=90)
            fig.tight_layout()
            path = project_root / "figures" / "per_label_f1_best_model.png"
            backup_existing(path, archive_dir)
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    examples = pd.read_csv(project_root / "outputs" / "misclassified_examples.csv") if (project_root / "outputs" / "misclassified_examples.csv").exists() else pd.DataFrame()
    if not examples.empty:
        examples = examples.copy()
        if "likely_reason" not in examples:
            examples["likely_reason"] = "manual_review_required"
        write_csv(project_root / "outputs" / "error_analysis_examples.csv", examples, archive_dir)
    summary = [
        "# Error Analysis Summary",
        "",
        "- Error tables are derived from saved predictions and classification reports.",
        "- Existing example-level reasons are kept as manual-review placeholders; no automatic legal explanation is invented.",
        "- Linear SVM and DistilBERT are close in macro-F1, which should be preserved as a finding if supported by the final table.",
        "- Likely discussion factors: boilerplate overlap, lexical ambiguity, long clauses/truncation, rare labels, semantically similar labels, and invalid LLM outputs.",
    ]
    write_text(project_root / "outputs" / "error_analysis_summary.md", "\n".join(summary) + "\n", archive_dir)


def generate_repo_audit(project_root: Path, main_results: pd.DataFrame, archive_dir: Path) -> None:
    files = [p for p in project_root.rglob("*") if p.is_file() and ".git" not in p.parts]
    module_rows = []
    for module_file in sorted((project_root / "modules").glob("*.py")):
        module_name = f"modules.{module_file.stem}"
        try:
            importlib.import_module(module_name)
            status = "importable"
            reason = ""
        except Exception as exc:
            status = "failed"
            reason = f"{type(exc).__name__}: {exc}"
        module_rows.append({"module": module_name, "status": status, "reason": reason})

    risky_patterns = ["debug=True", "sample_size", "head(", ".sample(", "max_steps", "TODO", "placeholder", "dummy", "hardcoded", "try:", "except"]
    pattern_rows = []
    for path in list((project_root / "modules").glob("*.py")) + list((project_root / "notebooks").glob("*.ipynb")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        hits = [pattern for pattern in risky_patterns if pattern in text]
        if hits:
            pattern_rows.append({"path": str(path.relative_to(project_root)), "patterns": ", ".join(hits)})

    evidence_rows = []
    for _, row in main_results.iterrows():
        evidence_parts = [part.strip() for part in str(row["evidence_path"]).split(";") if part.strip() and part.strip().lower() != "nan"]
        evidence_rows.append(
            {
                "model_name": row["model_name"],
                "status": row["status"],
                "evidence_path": row["evidence_path"],
                "evidence_exists": bool(evidence_parts) and all(file_exists(project_root, part) for part in evidence_parts),
                "prediction_path": row["prediction_path"],
                "prediction_exists": file_exists(project_root, row["prediction_path"]) if row["prediction_path"] else False,
            }
        )

    latest = pd.DataFrame(
        [
            {"path": str(path.relative_to(project_root)), "size_bytes": path.stat().st_size, "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()}
            for path in sorted(files, key=lambda item: item.stat().st_mtime, reverse=True)[:80]
        ]
    )
    md = [
        "# Repository Audit",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Inventory",
        f"- Total files inspected: {len(files)}",
        f"- Notebooks: {len(list((project_root / 'notebooks').glob('*.ipynb')))}",
        f"- Python modules: {len(list((project_root / 'modules').glob('*.py')))}",
        f"- Output files: {len(list((project_root / 'outputs').rglob('*')))}",
        f"- Result files: {len(list((project_root / 'results').rglob('*')))}",
        "",
        "## Latest Files",
        markdown_table(latest, max_rows=30),
        "",
        "## Module Import Check",
        markdown_table(pd.DataFrame(module_rows), max_rows=50),
        "",
        "## Shortcut / Cache Pattern Scan",
        markdown_table(pd.DataFrame(pattern_rows), max_rows=50),
        "",
        "## Model Evidence Check",
        markdown_table(pd.DataFrame(evidence_rows), max_rows=60),
        "",
        "## Remaining Reproducibility Risks",
        "- Generated artifacts include Colab absolute paths; canonical tables also include repo-relative evidence paths where possible.",
        "- Additional encoder models and BiLSTM are configured/pending unless their result artifacts are generated.",
        "- Qwen few-shot existing artifact has invalid_prediction_rate=1.0 and is marked failed in canonical outputs.",
    ]
    write_text(project_root / "outputs" / "repo_audit.md", "\n".join(md) + "\n", archive_dir)


def generate_reproducibility_and_final_audit(project_root: Path, main_results: pd.DataFrame, archive_dir: Path) -> None:
    completed = main_results[main_results["status"] == "completed"]
    failed = main_results[main_results["status"].isin(["failed", "skipped"])]
    pending = main_results[main_results["status"] == "pending"]
    required = [
        "outputs/main_results.csv",
        "outputs/dataset_verification.json",
        "outputs/repo_audit.md",
        "outputs/model_evidence_table.md",
        "outputs/error_analysis_summary.md",
    ]
    checks = []
    for rel in required:
        path = project_root / rel
        checks.append({"check": rel, "passed": path.exists() and path.stat().st_size > 0})
    checks_df = pd.DataFrame(checks)
    write_text(project_root / "outputs" / "reproducibility_check.md", "# Reproducibility Check\n\n" + markdown_table(checks_df) + "\n", archive_dir)

    seed_rows = []
    for model_name in ["linear_svm", "distilbert-base-uncased", "bilstm"]:
        for seed in [42, 43, 44]:
            seed_rows.append({"model_name": model_name, "seed": seed, "status": "pending" if seed != 42 or model_name == "bilstm" else "single_seed_available", "macro_f1": np.nan, "notes": "Run seed stability script/config on CUDA where needed."})
    write_csv(project_root / "outputs" / "seed_stability_results.csv", pd.DataFrame(seed_rows), archive_dir)

    final = [
        "# Final Coursework Audit",
        "",
        f"Created: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Completed Models",
        markdown_table(completed[["model_name", "model_family", "test_macro_f1", "evidence_path"]], max_rows=50),
        "",
        "## Failed or Skipped Models",
        markdown_table(failed[["model_name", "status", "reason_if_skipped_or_failed"]], max_rows=50),
        "",
        "## Configured but Not Run",
        markdown_table(pending[["model_name", "model_family", "reason_if_skipped_or_failed"]], max_rows=50),
        "",
        "## Protocol Check",
        "- Main dataset: LEDGAR.",
        "- Top-k labels are selected from training split only according to dataset verification.",
        "- Validation is used for classical model selection and DistilBERT checkpoint selection.",
        "- Test rows are used for final evaluation in canonical completed rows.",
        "- Qwen few-shot is not valid as a meaningful completed result because existing invalid-output rate is 1.0.",
        "",
        "## Validation Commands Used",
        "- `python scripts/build_coursework_artifacts.py`",
        "- `python -m py_compile modules/*.py scripts/*.py`",
    ]
    write_text(project_root / "outputs" / "final_coursework_audit.md", "\n".join(final) + "\n", archive_dir)


def build_all_coursework_artifacts(project_root: Path) -> dict[str, Any]:
    """Generate audit and canonical report-facing artifacts from current evidence."""
    project_root = project_root.resolve()
    archive_dir = project_root / "outputs" / "archive" / utc_stamp()
    archive_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset_verification(project_root, archive_dir)
    main = canonical_results(project_root, archive_dir)
    generate_tuning_artifacts(project_root, archive_dir)
    generate_llm_prompt_artifacts(project_root, archive_dir)
    generate_comparison_figures(project_root, main, archive_dir)
    generate_error_artifacts(project_root, main, archive_dir)
    generate_repo_audit(project_root, main, archive_dir)
    generate_reproducibility_and_final_audit(project_root, main, archive_dir)
    return {"project_root": str(project_root), "archive_dir": str(archive_dir), "main_results_rows": len(main), "dataset": dataset}
