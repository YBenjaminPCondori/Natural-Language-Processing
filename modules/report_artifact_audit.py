"""Strict report.tex artifact audit for the LEDGAR coursework project."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


STATUSES = {"PASS", "WARNING", "FAIL"}


@dataclass(frozen=True)
class RequiredArtifact:
    path: str
    report_need: str
    generator: str
    close_candidates: tuple[str, ...] = ()


REQUIRED_ARTIFACTS: tuple[RequiredArtifact, ...] = (
    RequiredArtifact(
        "figures/pipeline_overview.png",
        "Pipeline overview figure in report methodology section",
        "Create from report artifact/export stage or a dedicated diagram generation helper.",
    ),
    RequiredArtifact(
        "figures/label_distribution.png",
        "Dataset label-distribution figure",
        "modules.preprocessing.create_ledgar_eda or modules.coursework_artifacts.generate_dataset_verification",
        ("figures/label_distribution_top20.png",),
    ),
    RequiredArtifact(
        "figures/hpt_validation_macro_f1.png",
        "HPT validation macro-F1 figure",
        "modules.transformer_hpt.run_two_stage_transformer_hpt or scripts/run_transformer_hpt.py",
        ("outputs/figures/hpt_validation_macro_f1.png",),
    ),
    RequiredArtifact(
        "figures/confusion_matrix.png",
        "Main confusion matrix figure",
        "Final report export stage should copy the selected best-model confusion matrix to this canonical filename.",
        ("figures/confusion_matrix_best_model.png", "outputs/figures/confusion_matrix_best_model.png"),
    ),
    RequiredArtifact(
        "outputs/baseline_results.csv",
        "Baseline model results table",
        "modules.baselines.run_baseline_experiments should also export this canonical report filename.",
        ("results/baselines/baseline_results.csv",),
    ),
    RequiredArtifact(
        "outputs/sweep_results.csv",
        "Stage 5A/5B HPT sweep table",
        "modules.transformer_hpt.run_two_stage_transformer_hpt should export this canonical report filename.",
        ("outputs/hyperparameter_search_results.csv",),
    ),
    RequiredArtifact(
        "outputs/best_hyperparameters.json",
        "Best transformer HPT configuration",
        "modules.transformer_hpt.run_two_stage_transformer_hpt should export this canonical report filename.",
        ("outputs/best_transformer_configs.json", "results/transformer/training_args.json"),
    ),
    RequiredArtifact(
        "outputs/final_test_metrics.json",
        "Final held-out test metrics table/text",
        "Final evaluation/export stage should write selected final test metrics to JSON.",
        ("outputs/main_results.json", "results/final_model_comparison.csv", "results/transformer/transformer_results.csv"),
    ),
    RequiredArtifact(
        "outputs/final_test_predictions.csv",
        "Final selected model prediction table",
        "Final evaluation/export stage should write selected final model predictions to CSV.",
        ("results/transformer/transformer_predictions.csv", "outputs/transformer_predictions.csv"),
    ),
    RequiredArtifact(
        "outputs/per_class_metrics.csv",
        "Per-class precision/recall/F1 table",
        "Error analysis or report export stage should write per-class metrics to this canonical filename.",
        ("outputs/per_class_results.csv", "outputs/per_label_f1_all_models.csv"),
    ),
    RequiredArtifact(
        "outputs/error_analysis_examples.csv",
        "Error analysis example table",
        "modules.coursework_artifacts.generate_error_artifacts",
        ("outputs/misclassified_examples.csv",),
    ),
    RequiredArtifact(
        "outputs/failed_or_skipped_trials.csv",
        "Failed/skipped HPT/model trial log",
        "HPT/final audit stage should export failed and skipped trial rows.",
        ("outputs/main_results.csv", "outputs/hyperparameter_search_results.csv"),
    ),
)


MODEL_SPECS = (
    {
        "model_key": "logistic_regression",
        "display": "TF-IDF + Logistic Regression",
        "family": "classical",
        "result_file": "results/classical/classical_results.csv",
        "prediction_file": "outputs/predictions/logistic_regression_test_predictions.jsonl",
        "report_file": "results/classical/classification_reports/logistic_regression_report.json",
        "confusion_file": "results/classical/confusion_matrices/logistic_regression_confusion_matrix.png",
        "code": "modules/classical_models.py::run_classical_experiments",
    },
    {
        "model_key": "linear_svm",
        "display": "TF-IDF + Linear SVM",
        "family": "classical",
        "result_file": "results/classical/classical_results.csv",
        "prediction_file": "outputs/predictions/linear_svm_test_predictions.jsonl",
        "report_file": "results/classical/classification_reports/linear_svm_report.json",
        "confusion_file": "results/classical/confusion_matrices/linear_svm_confusion_matrix.png",
        "code": "modules/classical_models.py::run_classical_experiments",
    },
    {
        "model_key": "distilbert-base-uncased",
        "display": "Main transformer/BERT model",
        "family": "transformer",
        "result_file": "results/transformer/transformer_results.csv",
        "prediction_file": "results/transformer/transformer_predictions.csv",
        "report_file": "results/transformer/classification_reports/distilbert_base_uncased_report.json",
        "confusion_file": "results/transformer/confusion_matrices/distilbert_base_uncased_confusion_matrix.png",
        "code": "modules/transformer_model.py::train_transformer_classifier",
    },
    {
        "model_key": "qwen_zero_shot",
        "display": "Optional instruction-tuned LLM classifier",
        "family": "llm_prompting",
        "result_file": "outputs/qwen_results.csv",
        "prediction_file": "outputs/qwen_predictions.csv",
        "report_file": "results/qwen/classification_reports/qwen_zero_shot_report.json",
        "confusion_file": "results/qwen/confusion_matrices/qwen_zero_shot_confusion_matrix.png",
        "code": "modules/qwen_prompting.py::run_qwen_baseline",
    },
)


RISK_PATTERNS = (
    "debug",
    "sample_size",
    ".sample(",
    "head(",
    "max_steps",
    "load_from_cache_file",
    "resume_from_checkpoint",
    "if file exists",
    "skip",
    "placeholder",
    "dummy",
    "hardcoded",
    "try:",
    "except",
    "cuda",
    "oom",
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        import numpy as np

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
            return None if math.isnan(value) else value
        if isinstance(value, (np.bool_,)):
            return bool(value)
    except Exception:
        pass
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except Exception:
        pass
    return value


def markdown_table(rows: list[dict[str, Any]], columns: list[str] | None = None, *, max_rows: int = 80) -> str:
    if not rows:
        return "_No rows._"
    columns = columns or list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows[:max_rows]:
        values = []
        for column in columns:
            text = str(row.get(column, "") if row.get(column, "") is not None else "")
            text = text.replace("\n", " ").replace("|", "\\|")
            values.append(text)
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(rows)} rows._")
    return "\n".join(lines)


def file_info(project_root: Path, relative_path: str) -> dict[str, Any]:
    path = project_root / relative_path
    exists = path.exists()
    is_file = exists and path.is_file()
    return {
        "path": relative_path,
        "exists": exists,
        "is_file": is_file,
        "size_bytes": path.stat().st_size if is_file else None,
        "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() if is_file else None,
    }


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    rows.append({"__corrupt_line__": line})
    return pd.DataFrame(rows)


def read_csv_safe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_json_safe(path: Path) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_report_tex(project_root: Path) -> dict[str, Any]:
    report_path = project_root / "report.tex"
    text = report_path.read_text(encoding="utf-8", errors="ignore") if report_path.exists() else ""
    lines = text.splitlines()
    figures = []
    todos = []
    artifacts = []
    for line_no, line in enumerate(lines, start=1):
        for match in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", line):
            figures.append({"line": line_no, "path": match.group(1)})
        if "todo" in line.lower():
            todos.append({"line": line_no, "text": line.strip()})
        normalised = line.replace("\\_", "_")
        for match in re.finditer(r"(?:outputs|figures|results|data|models|checkpoints)/[A-Za-z0-9_./-]+", normalised):
            artifacts.append({"line": line_no, "path": match.group(0).rstrip(".,;)")})
    return {
        "report_path": str(report_path),
        "exists": report_path.exists(),
        "size_bytes": report_path.stat().st_size if report_path.exists() else None,
        "figures": figures,
        "todos": todos,
        "artifact_mentions": artifacts,
    }


def audit_dataset(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_dir = project_root / "data" / "raw" / "lexglue_ledgar"
    processed_dir = project_root / "data" / "processed"
    raw = {split: read_jsonl(raw_dir / f"ledgar_{split}.jsonl") for split in ("train", "validation", "test")}
    processed = {split: read_jsonl(processed_dir / f"ledgar_{split}.jsonl") for split in ("train", "validation", "test")}
    raw_label_file = raw_dir / "label_names.txt"
    processed_label_file = processed_dir / "label_names.txt"
    raw_label_names = [line.strip() for line in raw_label_file.read_text(encoding="utf-8").splitlines() if line.strip()] if raw_label_file.exists() else []
    processed_label_names = [line.strip() for line in processed_label_file.read_text(encoding="utf-8").splitlines() if line.strip()] if processed_label_file.exists() else []

    split_rows = {}
    for split, df in processed.items():
        split_rows[split] = {
            "path": f"data/processed/ledgar_{split}.jsonl",
            "exists": (processed_dir / f"ledgar_{split}.jsonl").exists(),
            "rows": int(len(df)),
            "non_empty": bool(len(df) > 0),
            "columns": list(df.columns),
            "unique_labels": int(df["label"].nunique()) if "label" in df else 0,
            "missing_text": int(df["text"].isna().sum()) if "text" in df else None,
            "missing_label": int(df["label"].isna().sum()) if "label" in df else None,
            "empty_text": int((df["text"].astype(str).str.strip() == "").sum()) if "text" in df else None,
            "duplicate_text_label_within_split": int(df.duplicated(["text", "label"]).sum()) if {"text", "label"}.issubset(df.columns) else None,
            "duplicate_text_within_split": int(df.duplicated(["text"]).sum()) if "text" in df else None,
        }

    raw_rows = {}
    for split, df in raw.items():
        raw_rows[split] = {
            "path": f"data/raw/lexglue_ledgar/ledgar_{split}.jsonl",
            "exists": (raw_dir / f"ledgar_{split}.jsonl").exists(),
            "rows": int(len(df)),
            "columns": list(df.columns),
            "unique_labels": int(df["label"].nunique()) if "label" in df else 0,
        }

    train_labels = raw["train"]["label"].copy() if "label" in raw["train"] else pd.Series(dtype=str)
    if raw_label_names and not train_labels.empty:
        def decode_label(label_value: Any) -> str:
            try:
                idx = int(label_value)
                return raw_label_names[idx] if 0 <= idx < len(raw_label_names) else str(label_value)
            except Exception:
                return str(label_value)

        train_labels = train_labels.map(decode_label)
    train_top20 = train_labels.value_counts().head(20).index.tolist() if not train_labels.empty else []
    selected_from_training_only = bool(processed_label_names) and set(processed_label_names) == set(train_top20)
    selected_order_exact = bool(processed_label_names) and processed_label_names == train_top20

    text_label_overlaps: dict[str, int] = {}
    exact_text_overlaps: dict[str, int] = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if {"text", "label"}.issubset(processed[left].columns) and {"text", "label"}.issubset(processed[right].columns):
            left_pairs = set(zip(processed[left]["text"].astype(str), processed[left]["label"].astype(str), strict=False))
            right_pairs = set(zip(processed[right]["text"].astype(str), processed[right]["label"].astype(str), strict=False))
            text_label_overlaps[f"{left}_vs_{right}"] = len(left_pairs & right_pairs)
        else:
            text_label_overlaps[f"{left}_vs_{right}"] = -1
        if "text" in processed[left] and "text" in processed[right]:
            exact_text_overlaps[f"{left}_vs_{right}"] = len(set(processed[left]["text"].astype(str)) & set(processed[right]["text"].astype(str)))
        else:
            exact_text_overlaps[f"{left}_vs_{right}"] = -1

    labels_ok = len(processed_label_names) > 0 and all(info["unique_labels"] == len(processed_label_names) for info in split_rows.values())
    splits_ok = all(info["non_empty"] for info in split_rows.values())
    leakage_ok = all(value == 0 for value in text_label_overlaps.values()) and all(value == 0 for value in exact_text_overlaps.values())
    missing_ok = all(
        (info["missing_text"] == 0 and info["missing_label"] == 0 and info["empty_text"] == 0)
        for info in split_rows.values()
    )

    status = "PASS" if splits_ok and labels_ok and selected_from_training_only and leakage_ok and missing_ok else "FAIL"
    audit = {
        "created_at_utc": now_utc(),
        "dataset_name": "LEDGAR",
        "raw_directory": "data/raw/lexglue_ledgar",
        "processed_directory": "data/processed",
        "raw_splits": raw_rows,
        "processed_splits": split_rows,
        "raw_label_count": len(raw_label_names),
        "processed_label_count": len(processed_label_names),
        "processed_label_names": processed_label_names,
        "top20_from_raw_training_split": train_top20,
        "top_labels_selected_only_from_training_split": selected_from_training_only,
        "top_label_order_matches_training_frequency_order": selected_order_exact,
        "cross_split_duplicate_text_label_pairs": text_label_overlaps,
        "cross_split_exact_text_leakage": exact_text_overlaps,
        "labels_present_and_consistent": labels_ok,
        "splits_exist_and_non_empty": splits_ok,
        "missing_or_empty_text_label_check_passed": missing_ok,
        "status": status,
    }
    stage = {
        "stage": "Dataset audit",
        "status": status,
        "evidence": "data/processed/ledgar_{train,validation,test}.jsonl; data/processed/label_names.txt",
        "summary": (
            f"Processed rows train/validation/test = "
            f"{split_rows['train']['rows']}/{split_rows['validation']['rows']}/{split_rows['test']['rows']}; "
            f"labels={len(processed_label_names)}; text+label overlaps={text_label_overlaps}; exact text leakage={exact_text_overlaps}."
        ),
    }
    return audit, stage


def artifact_status(project_root: Path, artifact: RequiredArtifact) -> dict[str, Any]:
    info = file_info(project_root, artifact.path)
    close = [file_info(project_root, candidate) for candidate in artifact.close_candidates]
    close_existing = [candidate for candidate in close if candidate["exists"] and candidate["is_file"] and (candidate["size_bytes"] or 0) > 0]
    schema_note = ""
    status = "FAIL"
    reason = "Required artifact is missing."
    if info["exists"] and info["is_file"] and (info["size_bytes"] or 0) > 0:
        status = "PASS"
        reason = "Required artifact exists and is non-empty."
        suffix = Path(artifact.path).suffix.lower()
        if suffix == ".csv":
            df = read_csv_safe(project_root / artifact.path)
            if df.empty:
                status = "WARNING"
                reason = "Required CSV exists but could not be parsed or has no rows."
            schema_note = f"columns={list(df.columns)} rows={len(df)}" if not df.empty else "empty_or_unreadable_csv"
        elif suffix == ".json":
            payload = read_json_safe(project_root / artifact.path)
            if payload is None:
                status = "WARNING"
                reason = "Required JSON exists but could not be parsed."
            schema_note = f"type={type(payload).__name__}" if payload is not None else "unreadable_json"
    elif close_existing:
        status = "WARNING"
        reason = "Required canonical filename is missing, but close noncanonical evidence exists."

    return {
        "path": artifact.path,
        "status": status,
        "exists": info["exists"],
        "size_bytes": info["size_bytes"],
        "report_need": artifact.report_need,
        "generator": artifact.generator,
        "reason": reason,
        "schema_note": schema_note,
        "close_existing_artifacts": [candidate["path"] for candidate in close_existing],
    }


def audit_required_artifacts(project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [artifact_status(project_root, artifact) for artifact in REQUIRED_ARTIFACTS]
    missing = [
        {
            "missing_filename": row["path"],
            "report_need": row["report_need"],
            "code_section_should_generate_it": row["generator"],
            "status": row["status"],
            "close_existing_artifacts": row["close_existing_artifacts"],
        }
        for row in rows
        if row["status"] != "PASS"
    ]
    suspicious = [
        {
            "path": row["path"],
            "issue": row["reason"],
            "close_existing_artifacts": row["close_existing_artifacts"],
        }
        for row in rows
        if row["status"] == "WARNING"
    ]
    return rows, missing, suspicious


def evidence_exists(project_root: Path, relative_path: str) -> bool:
    path = project_root / relative_path
    return path.exists() and path.is_file() and path.stat().st_size > 0


def count_prediction_rows(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    if path.suffix.lower() == ".jsonl":
        return sum(1 for line in path.open("r", encoding="utf-8") if line.strip())
    if path.suffix.lower() == ".csv":
        df = read_csv_safe(path)
        return int(len(df)) if not df.empty else None
    return None


def model_row_from_results(project_root: Path, spec: dict[str, str]) -> dict[str, Any]:
    result_file = project_root / spec["result_file"]
    result_df = read_csv_safe(result_file)
    key = spec["model_key"]
    result_match = pd.DataFrame()
    if not result_df.empty and "model_name" in result_df:
        result_match = result_df[result_df["model_name"].astype(str).str.lower() == key.lower()]
    if result_match.empty and key == "distilbert-base-uncased" and not result_df.empty:
        result_match = result_df[result_df.get("model_name", pd.Series(dtype=str)).astype(str).str.contains("distilbert", case=False, na=False)]

    metrics_saved = not result_match.empty and any(column in result_match for column in ("accuracy", "macro_f1", "weighted_f1"))
    predictions_exist = evidence_exists(project_root, spec["prediction_file"])
    report_exists = evidence_exists(project_root, spec["report_file"])
    confusion_exists = evidence_exists(project_root, spec["confusion_file"])
    prediction_rows = count_prediction_rows(project_root / spec["prediction_file"]) if predictions_exist else None
    validates = False
    trains = bool(metrics_saved and predictions_exist)
    real_not_placeholder = bool(metrics_saved and report_exists and confusion_exists)
    notes = []
    validation_macro_f1 = None
    test_macro_f1 = None

    if not result_match.empty:
        first = result_match.iloc[0]
        test_macro_f1 = first.get("macro_f1")
        validation_macro_f1 = first.get("validation_macro_f1")
        validates = pd.notna(validation_macro_f1) or spec["family"] in {"transformer", "classical"}
        if spec["family"] == "llm_prompting" and "invalid_prediction_rate" in first and pd.notna(first["invalid_prediction_rate"]):
            invalid_rate = float(first["invalid_prediction_rate"])
            if invalid_rate >= 1.0:
                notes.append("Invalid prediction rate is 1.0; do not treat as meaningful completed LLM evidence.")

    if spec["family"] == "classical":
        grid = read_csv_safe(project_root / "results/classical/classical_validation_grid.csv")
        validates = validates or (not grid.empty and key in grid.get("model_name", pd.Series(dtype=str)).astype(str).tolist())
    if spec["family"] == "transformer":
        log = read_csv_safe(project_root / "results/transformer/training_log_history.csv")
        validates = validates or (not log.empty and "eval_macro_f1" in log.columns and log["eval_macro_f1"].notna().any())
        trains = trains and evidence_exists(project_root, "results/transformer/model/model.safetensors")
    if spec["family"] == "llm_prompting":
        qwen_results = read_csv_safe(project_root / "outputs/qwen_results.csv")
        validates = False
        trains = metrics_saved and predictions_exist
        real_not_placeholder = trains and report_exists and confusion_exists
        if not qwen_results.empty and "model_name" in qwen_results:
            few = qwen_results[qwen_results["model_name"].astype(str).str.contains("few", case=False, na=False)]
            if not few.empty and "invalid_prediction_rate" in few:
                if pd.to_numeric(few["invalid_prediction_rate"], errors="coerce").max() >= 1.0:
                    notes.append("Qwen few-shot artifact exists but is invalid under recorded parser.")

    status = "PASS" if trains and metrics_saved and predictions_exist and report_exists and confusion_exists and (validates or spec["family"] == "llm_prompting") else "FAIL"
    if status == "FAIL" and (metrics_saved or predictions_exist or report_exists or confusion_exists):
        status = "WARNING"
    if notes and status == "PASS":
        status = "WARNING"

    return {
        "model": spec["display"],
        "model_key": key,
        "status": status,
        "trains_or_runs": trains,
        "validates": validates,
        "metrics_saved": metrics_saved,
        "predictions_exported": predictions_exist,
        "prediction_rows": prediction_rows,
        "report_exists": report_exists,
        "confusion_exists": confusion_exists,
        "real_not_placeholder": real_not_placeholder,
        "test_macro_f1": test_macro_f1,
        "validation_macro_f1": validation_macro_f1,
        "evidence": "; ".join([spec["result_file"], spec["prediction_file"], spec["report_file"], spec["confusion_file"]]),
        "code": spec["code"],
        "notes": " ".join(notes),
    }


def audit_models(project_root: Path) -> list[dict[str, Any]]:
    return [model_row_from_results(project_root, spec) for spec in MODEL_SPECS]


def audit_hpt(project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    hpt_module = project_root / "modules" / "transformer_hpt.py"
    hpt_script = project_root / "scripts" / "run_transformer_hpt.py"
    hpt_text = hpt_module.read_text(encoding="utf-8", errors="ignore") if hpt_module.exists() else ""
    script_text = hpt_script.read_text(encoding="utf-8", errors="ignore") if hpt_script.exists() else ""
    search_csv = project_root / "outputs" / "hyperparameter_search_results.csv"
    sweep_csv = project_root / "outputs" / "sweep_results.csv"
    hpt_runs = sorted((project_root / "results" / "transformer_hpt").glob("*/hyperparameter_search_results.csv"), key=lambda path: path.stat().st_mtime, reverse=True) if (project_root / "results" / "transformer_hpt").exists() else []
    latest_hpt_df = read_csv_safe(hpt_runs[0]) if hpt_runs else pd.DataFrame()
    output_search_df = read_csv_safe(search_csv)

    def row(check: str, status: str, evidence: str, fix: str) -> dict[str, Any]:
        assert status in STATUSES
        return {"check": check, "status": status, "evidence": evidence, "fix": fix}

    rows = [
        row(
            "HPT pipeline module/script exists",
            "PASS" if hpt_module.exists() and hpt_script.exists() else "FAIL",
            "modules/transformer_hpt.py; scripts/run_transformer_hpt.py",
            "Add or restore the reusable HPT runner.",
        ),
        row(
            "Stage 5A wide random W&B sweep exists",
            "WARNING" if "stage5a_random_trial" in hpt_text else "FAIL",
            "Found stage5a_random_trial naming but no W&B sweep controller artifact.",
            "Use W&B Sweeps or clearly rename the report to W&B tracked trials; export outputs/sweep_results.csv.",
        ),
        row(
            "Stage 5A uses 20 trials",
            "PASS" if "random_trials: int = 20" in hpt_text or "--random-trials 20" in script_text else "FAIL",
            "Current code defaults appear not to enforce 20 trials.",
            "Set TransformerHPTConfig.random_trials=20 and notebook/script defaults to 20.",
        ),
        row(
            "Stage 5A uses 2 epochs per trial",
            "PASS" if re.search(r'"epochs"\s*:\s*\[2\]', hpt_text) or "epochs=2" in hpt_text else "FAIL",
            "Current search space allows multiple epoch values rather than fixed 2 for Stage 5A.",
            "Force Stage 5A trial configs to num_train_epochs=2.",
        ),
        row(
            "Validation macro-F1 is the optimisation target",
            "PASS" if "validation_macro_f1" in hpt_text and "metric_for_best_model" in (project_root / "modules" / "transformer_model.py").read_text(encoding="utf-8", errors="ignore") else "FAIL",
            "HPT/training code references validation_macro_f1 and macro_f1 selection.",
            "Keep validation macro-F1 as the objective and export it in sweep rows.",
        ),
        row(
            "Stage 5B Bayesian W&B sweep exists",
            "WARNING" if "stage5b_bayes_trial" in hpt_text and "optuna" in hpt_text.lower() else "FAIL",
            "Optuna Bayesian trials are configured; no W&B sweep artifact found.",
            "Use W&B Sweeps or document Optuna + W&B runs accurately; export canonical sweep_results.csv.",
        ),
        row(
            "Stage 5B uses 15 trials",
            "PASS" if "bayes_trials: int = 15" in hpt_text or "--bayes-trials 15" in script_text else "FAIL",
            "Current code defaults appear not to enforce 15 Bayesian trials.",
            "Set TransformerHPTConfig.bayes_trials=15 and notebook/script defaults to 15.",
        ),
        row(
            "Stage 5B uses 3 epochs per trial",
            "PASS" if "epochs=3" in hpt_text or re.search(r'"epochs"\s*:\s*\[3\]', hpt_text) else "FAIL",
            "Current search space allows multiple epoch values rather than fixed 3 for Stage 5B.",
            "Force Stage 5B trial configs to num_train_epochs=3.",
        ),
        row(
            "Stage 5B uses narrowed search space",
            "FAIL" if "config.search_space" in hpt_text else "WARNING",
            "Bayesian stage appears to reuse the same search_space object.",
            "Create an explicit narrowed Stage 5B search space from Stage 5A results.",
        ),
        row(
            "Best hyperparameters exported to required filename",
            "PASS" if evidence_exists(project_root, "outputs/best_hyperparameters.json") else "WARNING",
            "outputs/best_hyperparameters.json missing; outputs/best_transformer_configs.json may exist.",
            "Write the selected HPT config to outputs/best_hyperparameters.json.",
        ),
        row(
            "Final retraining uses 4 epochs",
            "PASS" if "num_train_epochs=4" in hpt_text or '"epochs": 4' in hpt_text else "FAIL",
            "Final retrain currently appears to reuse the selected trial epoch count.",
            "Force final retrain to 4 epochs after validation-selected HPT.",
        ),
        row(
            "Final test evaluation occurs once after tuning",
            "PASS" if "evaluate_test=False" in hpt_text and "evaluate_test=True" in hpt_text else "WARNING",
            "HPT trial code disables test evaluation and final retrain enables it; runtime evidence still required.",
            "Keep test evaluation disabled during trials and verify final output timestamps after rerun.",
        ),
        row(
            "Completed HPT run evidence exists",
            "PASS" if hpt_runs and not latest_hpt_df.empty else "FAIL",
            f"Found HPT run files: {len(hpt_runs)}; outputs/hyperparameter_search_results rows={len(output_search_df)}.",
            "Run: python scripts/run_transformer_hpt.py --model-name distilbert-base-uncased --random-trials 20 --bayes-trials 15 --wandb",
        ),
        row(
            "Canonical sweep_results.csv exists",
            "PASS" if sweep_csv.exists() and sweep_csv.stat().st_size > 0 else "FAIL",
            "outputs/sweep_results.csv is required by report.tex but missing.",
            "Export HPT trial rows to outputs/sweep_results.csv.",
        ),
    ]
    suspicious = [item for item in rows if item["status"] != "PASS"]
    return rows, suspicious


def audit_evaluation(project_root: Path) -> list[dict[str, Any]]:
    main = read_csv_safe(project_root / "outputs/main_results.csv")
    per_class = read_csv_safe(project_root / "outputs/per_class_results.csv")
    errors = read_csv_safe(project_root / "outputs/error_analysis_examples.csv")
    transformer_preds = read_csv_safe(project_root / "results/transformer/transformer_predictions.csv")
    rows = []

    def add(check: str, status: str, evidence: str, fix: str = "") -> None:
        rows.append({"check": check, "status": status, "evidence": evidence, "fix": fix})

    metric_cols = {"test_accuracy", "test_macro_f1", "test_weighted_f1"}
    add(
        "Accuracy, macro-F1, weighted-F1 produced",
        "PASS" if not main.empty and metric_cols.issubset(main.columns) and main[list(metric_cols)].notna().any().all() else "FAIL",
        "outputs/main_results.csv",
        "Regenerate final model comparison and report exports.",
    )
    add(
        "Per-class precision/recall/F1 produced",
        "PASS" if not per_class.empty and {"precision", "recall", "f1_score"}.issubset(per_class.columns) else "FAIL",
        "outputs/per_class_results.csv",
        "Export canonical outputs/per_class_metrics.csv.",
    )
    add(
        "Confusion matrix produced",
        "WARNING" if evidence_exists(project_root, "figures/confusion_matrix_best_model.png") and not evidence_exists(project_root, "figures/confusion_matrix.png") else ("PASS" if evidence_exists(project_root, "figures/confusion_matrix.png") else "FAIL"),
        "figures/confusion_matrix_best_model.png; figures/confusion_matrix.png",
        "Copy or regenerate best-model confusion matrix as figures/confusion_matrix.png.",
    )
    confidence_cols = {"confidence", "score", "probability", "margin"}
    has_confidence = not transformer_preds.empty and bool(confidence_cols & set(transformer_preds.columns))
    add(
        "Prediction confidence exported when supported",
        "WARNING" if not has_confidence else "PASS",
        "Prediction files do not expose confidence/score columns for the final transformer.",
        "Add confidence/margin output for models that support it, or state confidence is unavailable.",
    )
    add(
        "Error analysis examples produced",
        "PASS" if not errors.empty else "FAIL",
        "outputs/error_analysis_examples.csv",
        "Run error analysis/report artifact stage.",
    )
    return rows


def audit_execution_risks(project_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    search_paths = list((project_root / "modules").glob("*.py")) + list((project_root / "scripts").glob("*.py")) + list((project_root / "notebooks").glob("*.ipynb")) + list((project_root / "notebooks" / "stages").glob("*.ipynb"))
    for path in search_paths:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        hits = sorted({pattern for pattern in RISK_PATTERNS if pattern in text})
        if hits:
            rows.append(
                {
                    "path": path.relative_to(project_root).as_posix(),
                    "status": "WARNING",
                    "risk_patterns": ", ".join(hits),
                    "note": "Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly.",
                }
            )
    qwen = read_csv_safe(project_root / "outputs/qwen_results.csv")
    if not qwen.empty and "sample_size" in qwen:
        sample_size = int(pd.to_numeric(qwen["sample_size"], errors="coerce").max())
        rows.append(
            {
                "path": "outputs/qwen_results.csv",
                "status": "WARNING" if sample_size < 4732 else "PASS",
                "risk_patterns": "reduced sample mode",
                "note": f"Qwen evaluated on max sample_size={sample_size}; full processed test split has 4732 rows.",
            }
        )
    transformer_runtime = read_json_safe(project_root / "results/transformer/runtime.json")
    if transformer_runtime:
        rows.append(
            {
                "path": "results/transformer/runtime.json",
                "status": "PASS" if transformer_runtime.get("cuda_available") else "WARNING",
                "risk_patterns": "cuda",
                "note": f"CUDA available in recorded runtime: {transformer_runtime.get('cuda_available')}; GPU={transformer_runtime.get('gpu_name')}",
            }
        )
    return rows


def exact_fix_rows(required_rows: list[dict[str, Any]], hpt_rows: list[dict[str, Any]], evaluation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fixes = []
    for row in required_rows:
        if row["status"] != "PASS":
            fixes.append(
                {
                    "priority": "critical" if row["status"] == "FAIL" else "important",
                    "what": row["path"],
                    "why": row["report_need"],
                    "where": row["generator"],
                    "action": "Generate canonical artifact at the required filename; do not rely only on close/noncanonical outputs.",
                }
            )
    for row in hpt_rows:
        if row["status"] != "PASS":
            fixes.append(
                {
                    "priority": "critical",
                    "what": row["check"],
                    "why": "Required by updated report.tex HPT methodology.",
                    "where": "modules/transformer_hpt.py; scripts/run_transformer_hpt.py; Stage 5 notebook",
                    "action": row["fix"],
                }
            )
    for row in evaluation_rows:
        if row["status"] != "PASS":
            fixes.append(
                {
                    "priority": "important",
                    "what": row["check"],
                    "why": "Required by evaluation/report artifact audit.",
                    "where": "modules/evaluation.py; modules/report_exports.py; modules/coursework_artifacts.py",
                    "action": row["fix"],
                }
            )
    return fixes


def readiness_score(stage_rows: list[dict[str, Any]], required_rows: list[dict[str, Any]], hpt_rows: list[dict[str, Any]], model_rows: list[dict[str, Any]], evaluation_rows: list[dict[str, Any]], execution_rows: list[dict[str, Any]]) -> float:
    score = 10.0
    for row in stage_rows:
        if row["stage"] != "Dataset audit":
            continue
        if row["status"] == "FAIL":
            score -= 2.0
        elif row["status"] == "WARNING":
            score -= 0.75
    for row in required_rows:
        if row["status"] == "FAIL":
            score -= 0.45
        elif row["status"] == "WARNING":
            score -= 0.25
    hpt_failures = sum(1 for row in hpt_rows if row["status"] == "FAIL")
    hpt_warnings = sum(1 for row in hpt_rows if row["status"] == "WARNING")
    score -= min(2.5, hpt_failures * 0.35 + hpt_warnings * 0.2)
    for row in model_rows:
        if row["status"] == "FAIL":
            score -= 0.75
        elif row["status"] == "WARNING":
            score -= 0.35
    eval_penalty = 0.0
    for row in evaluation_rows:
        if row["status"] == "FAIL":
            eval_penalty += 0.45
        elif row["status"] == "WARNING":
            eval_penalty += 0.2
    score -= min(1.0, eval_penalty)
    if any(row["status"] == "WARNING" for row in execution_rows):
        score -= 0.75
    return round(max(score, 0.0), 1)


def write_report(
    project_root: Path,
    *,
    report_tex: dict[str, Any],
    data_audit: dict[str, Any],
    stage_rows: list[dict[str, Any]],
    required_rows: list[dict[str, Any]],
    model_rows: list[dict[str, Any]],
    hpt_rows: list[dict[str, Any]],
    evaluation_rows: list[dict[str, Any]],
    execution_rows: list[dict[str, Any]],
    fixes: list[dict[str, Any]],
    readiness: float,
) -> str:
    missing = [row for row in required_rows if row["status"] != "PASS"]
    suspicious = [row for row in required_rows if row["status"] == "WARNING"]
    ready = readiness >= 8.0 and not missing and all(row["status"] == "PASS" for row in hpt_rows)
    lines = [
        "# Project Artifact Audit Against report.tex",
        "",
        f"Created: {now_utc()}",
        f"Project root: `{project_root}`",
        f"Report: `{report_tex['report_path']}`",
        f"Ready for report.tex: `{'YES' if ready else 'NO'}`",
        f"Overall readiness score: **{readiness}/10**",
        "",
        "## Pipeline Stage Status",
        markdown_table(stage_rows, ["stage", "status", "summary", "evidence"]),
        "",
        "## Required Report Artifacts",
        markdown_table(required_rows, ["path", "status", "reason", "report_need", "generator", "close_existing_artifacts", "schema_note"]),
        "",
        "## Missing Artifacts",
        markdown_table(
            [
                {
                    "missing_filename": row["path"],
                    "report_need": row["report_need"],
                    "code_section_should_generate_it": row["generator"],
                    "close_existing_artifacts": row["close_existing_artifacts"],
                }
                for row in missing
            ],
            ["missing_filename", "report_need", "code_section_should_generate_it", "close_existing_artifacts"],
        ),
        "",
        "## Suspicious / Noncanonical Artifacts",
        markdown_table(suspicious, ["path", "status", "reason", "close_existing_artifacts"]),
        "",
        "## Model Audit",
        markdown_table(model_rows, ["model", "status", "trains_or_runs", "validates", "metrics_saved", "predictions_exported", "prediction_rows", "real_not_placeholder", "evidence", "notes"]),
        "",
        "## HPT Audit",
        markdown_table(hpt_rows, ["check", "status", "evidence", "fix"]),
        "",
        "## Evaluation Audit",
        markdown_table(evaluation_rows, ["check", "status", "evidence", "fix"]),
        "",
        "## Execution / Silent-Skip Audit",
        markdown_table(execution_rows, ["path", "status", "risk_patterns", "note"], max_rows=120),
        "",
        "## report.tex References",
        f"- Figures referenced: `{len(report_tex['figures'])}`",
        f"- TODO lines found: `{len(report_tex['todos'])}`",
        f"- Artifact-like path mentions found: `{len(report_tex['artifact_mentions'])}`",
        "",
        "### Figure References",
        markdown_table(report_tex["figures"], ["line", "path"]),
        "",
        "### TODO Lines",
        markdown_table(report_tex["todos"], ["line", "text"], max_rows=80),
        "",
        "## Required Fixes",
        markdown_table(fixes, ["priority", "what", "why", "where", "action"], max_rows=120),
        "",
        "## Commands To Run When Evidence Is Missing",
        "```bash",
        "# Rebuild canonical report audit only",
        "python scripts/run_report_artifact_audit.py",
        "",
        "# Generate current report-facing derived artifacts from existing evidence",
        "python scripts/build_coursework_artifacts.py",
        "",
        "# Run the updated transformer HPT on CUDA/A100",
        "python scripts/run_transformer_hpt.py --model-name distilbert-base-uncased --random-trials 20 --bayes-trials 15 --wandb",
        "```",
    ]
    return "\n".join(lines) + "\n"


def build_report_artifact_audit(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    outputs = project_root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)

    report_tex = parse_report_tex(project_root)
    data_audit, dataset_stage = audit_dataset(project_root)
    required_rows, missing_artifacts, suspicious_artifacts = audit_required_artifacts(project_root)
    model_rows = audit_models(project_root)
    hpt_rows, hpt_suspicious = audit_hpt(project_root)
    evaluation_rows = audit_evaluation(project_root)
    execution_rows = audit_execution_risks(project_root)

    stage_rows = [
        dataset_stage,
        {
            "stage": "Required report artifacts",
            "status": "PASS" if all(row["status"] == "PASS" for row in required_rows) else ("FAIL" if any(row["status"] == "FAIL" for row in required_rows) else "WARNING"),
            "evidence": "figures/ and outputs/ required by report.tex",
            "summary": f"{sum(row['status']=='PASS' for row in required_rows)} PASS, {sum(row['status']=='WARNING' for row in required_rows)} WARNING, {sum(row['status']=='FAIL' for row in required_rows)} FAIL.",
        },
        {
            "stage": "Model evidence",
            "status": "PASS" if all(row["status"] == "PASS" for row in model_rows) else ("WARNING" if any(row["status"] == "PASS" for row in model_rows) else "FAIL"),
            "evidence": "results/, outputs/predictions/, classification reports, confusion matrices",
            "summary": f"{sum(row['status']=='PASS' for row in model_rows)} PASS, {sum(row['status']=='WARNING' for row in model_rows)} WARNING, {sum(row['status']=='FAIL' for row in model_rows)} FAIL.",
        },
        {
            "stage": "HPT methodology",
            "status": "PASS" if all(row["status"] == "PASS" for row in hpt_rows) else "FAIL",
            "evidence": "modules/transformer_hpt.py; scripts/run_transformer_hpt.py; outputs/sweep_results.csv",
            "summary": f"{sum(row['status']=='PASS' for row in hpt_rows)} PASS, {sum(row['status']=='WARNING' for row in hpt_rows)} WARNING, {sum(row['status']=='FAIL' for row in hpt_rows)} FAIL.",
        },
        {
            "stage": "Evaluation artifacts",
            "status": "PASS" if all(row["status"] == "PASS" for row in evaluation_rows) else "WARNING",
            "evidence": "outputs/main_results.csv; outputs/per_class_results.csv; error analysis outputs",
            "summary": f"{sum(row['status']=='PASS' for row in evaluation_rows)} PASS, {sum(row['status']=='WARNING' for row in evaluation_rows)} WARNING, {sum(row['status']=='FAIL' for row in evaluation_rows)} FAIL.",
        },
        {
            "stage": "Execution risks",
            "status": "WARNING" if any(row["status"] == "WARNING" for row in execution_rows) else "PASS",
            "evidence": "Static pattern scan of notebooks/modules/scripts plus runtime outputs",
            "summary": f"{len(execution_rows)} risk rows found; inspect warnings before filling report.tex.",
        },
    ]

    fixes = exact_fix_rows(required_rows, hpt_rows, evaluation_rows)
    readiness = readiness_score(stage_rows, required_rows, hpt_rows, model_rows, evaluation_rows, execution_rows)
    ready = readiness >= 8.0 and not missing_artifacts and all(row["status"] == "PASS" for row in hpt_rows)
    summary = {
        "created_at_utc": now_utc(),
        "project_root": str(project_root),
        "report_tex": report_tex,
        "ready_for_report_tex": ready,
        "overall_readiness_score": readiness,
        "stage_statuses": stage_rows,
        "missing_artifacts": missing_artifacts,
        "suspicious_artifacts": suspicious_artifacts + hpt_suspicious,
        "required_artifact_checks": required_rows,
        "model_audit": model_rows,
        "hpt_audit": hpt_rows,
        "evaluation_audit": evaluation_rows,
        "execution_risks": execution_rows,
        "required_fixes": fixes,
    }

    (outputs / "data_audit.json").write_text(json.dumps(json_safe(data_audit), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (outputs / "project_audit_summary.json").write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = write_report(
        project_root,
        report_tex=report_tex,
        data_audit=data_audit,
        stage_rows=stage_rows,
        required_rows=required_rows,
        model_rows=model_rows,
        hpt_rows=hpt_rows,
        evaluation_rows=evaluation_rows,
        execution_rows=execution_rows,
        fixes=fixes,
        readiness=readiness,
    )
    (outputs / "project_audit_report.md").write_text(report, encoding="utf-8")
    return summary
