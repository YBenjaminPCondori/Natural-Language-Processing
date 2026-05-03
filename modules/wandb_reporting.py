"""Optional Weights & Biases logging for notebook-generated artifacts."""

from __future__ import annotations

import importlib.util
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .data_setup import ProjectPaths, ensure_package


SAFE_CSV_ARTIFACTS = [
    "outputs/data_summary.csv",
    "outputs/label_distribution.csv",
    "outputs/main_results.csv",
    "outputs/per_class_results.csv",
    "outputs/confusion_pairs.csv",
    "outputs/hyperparameters.csv",
    "outputs/transformer_results.csv",
    "outputs/qwen_results.csv",
    "results/final_model_comparison.csv",
    "results/baselines/baseline_results.csv",
    "results/classical/classical_results.csv",
    "results/classical/classical_validation_grid.csv",
    "results/error_analysis/class_imbalance.csv",
    "results/error_analysis/classical_top_confusions.csv",
    "results/error_analysis/transformer_top_confusions.csv",
    "results/transformer/transformer_results.csv",
    "results/qwen/qwen_results.csv",
]
SAFE_JSON_ARTIFACTS = [
    "outputs/environment.json",
    "outputs/leakage_audit.json",
    "outputs/label_mapping.json",
    "outputs/report_artifact_manifest.json",
    "data/processed/dataset_summary.json",
    "data/processed/label_counts.json",
    "results/transformer/runtime.json",
    "results/transformer/training_args.json",
    "results/transformer/training_log_history.json",
    "results/qwen/runtime.json",
    "results/qwen/qwen_run_config.json",
]
TEXT_CONTAINING_ARTIFACTS = [
    "outputs/misclassified_examples.csv",
    "outputs/qwen_predictions.csv",
    "outputs/qwen_invalid_outputs.csv",
    "outputs/qwen_prompt_examples.txt",
    "outputs/transformer_predictions.csv",
    "results/error_analysis/classical_misclassified_examples.csv",
    "results/error_analysis/transformer_misclassified_examples.csv",
    "results/error_analysis/qwen_invalid_outputs.csv",
    "results/error_analysis/qwen_plausible_nonmatching_examples.csv",
    "results/qwen/qwen_predictions.csv",
    "results/qwen/qwen_invalid_outputs.csv",
    "results/qwen/qwen_prompt_examples.txt",
    "results/transformer/transformer_predictions.csv",
]


def _safe_key(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(value).lower()).strip("_") or "unknown"


def _existing_files(root: Path, relative_paths: list[str]) -> list[Path]:
    files = []
    for relative_path in relative_paths:
        path = root / relative_path
        if path.exists() and path.is_file() and path.stat().st_size > 0:
            files.append(path)
    return files


def _has_wandb_credentials(mode: str) -> bool:
    if mode.lower() in {"offline", "disabled"}:
        return True
    return bool(os.environ.get("WANDB_API_KEY")) or (Path.home() / ".netrc").exists()


def load_wandb_key_from_colab(secret_name: str = "WANDB_API_KEY") -> bool:
    """Load WANDB_API_KEY from Google Colab Secrets when available."""
    if os.environ.get("WANDB_API_KEY"):
        return True
    if importlib.util.find_spec("google.colab") is None:
        return False
    try:
        from google.colab import userdata

        api_key = userdata.get(secret_name)
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
            print(f"W&B API key loaded from Colab Secrets entry {secret_name!r}.")
            return True
    except Exception as exc:
        print(f"W&B Colab Secrets lookup skipped: {type(exc).__name__}: {exc}")
    return False


def start_wandb_run(
    *,
    enabled: bool,
    project: str,
    entity: str | None = None,
    run_name: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
    mode: str = "online",
) -> Any | None:
    """Start an optional W&B run without making notebook execution depend on W&B."""
    if not enabled:
        print("W&B logging disabled by RUN_WANDB=False.")
        return None

    mode = (mode or "online").strip().lower()
    if mode not in {"offline", "disabled"}:
        load_wandb_key_from_colab()
    if not _has_wandb_credentials(mode):
        print("W&B logging skipped: no WANDB_API_KEY or ~/.netrc credentials found.")
        print("Add a Colab Secret named WANDB_API_KEY, run wandb.login(), or set WANDB_API_KEY, then rerun the notebook.")
        return None

    try:
        ensure_package("wandb", "wandb>=0.16")
        import wandb

        os.environ.setdefault("WANDB_DISABLE_CODE", "true")
        run = wandb.init(
            project=project,
            entity=entity or None,
            name=run_name or f"ledgar-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
            group=group or "notebook",
            tags=tags or ["ledgar", "coursework", "notebook"],
            config=config or {},
            mode=mode,
            reinit=True,
        )
        print(f"W&B run active: {getattr(run, 'url', None) or run.name}")
        return run
    except Exception as exc:
        print(f"W&B logging skipped: {type(exc).__name__}: {exc}")
        return None


def _log_scalar_metrics(run: Any, comparison_df: pd.DataFrame) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if comparison_df is None or comparison_df.empty:
        return metrics

    for row in comparison_df.to_dict("records"):
        model_key = _safe_key(row.get("model_name", row.get("model_family", "model")))
        for metric_name in ("accuracy", "macro_f1", "weighted_f1", "invalid_prediction_rate"):
            value = row.get(metric_name)
            if pd.notna(value):
                metrics[f"test/{model_key}/{metric_name}"] = float(value)

    ranked = comparison_df.dropna(subset=["macro_f1"]).copy()
    if not ranked.empty:
        best = ranked.sort_values("macro_f1", ascending=False).iloc[0]
        metrics["test/best_macro_f1"] = float(best["macro_f1"])
        run.summary["best_model_name"] = str(best.get("model_name"))
        run.summary["best_model_family"] = str(best.get("model_family"))
    if metrics:
        run.log(metrics)
    return metrics


def _log_csv_table(run: Any, path: Path, *, max_rows: int) -> bool:
    try:
        import wandb

        df = pd.read_csv(path)
        if len(df) > max_rows:
            df = df.head(max_rows).copy()
        run.log({f"tables/{_safe_key(path.stem)}": wandb.Table(dataframe=df)})
        return True
    except Exception as exc:
        print(f"W&B table logging skipped for {path}: {type(exc).__name__}: {exc}")
        return False


def _artifact_files(paths: ProjectPaths, *, log_text_tables: bool, log_model_files: bool) -> list[Path]:
    root = paths.project_root
    files = _existing_files(root, SAFE_CSV_ARTIFACTS + SAFE_JSON_ARTIFACTS)
    for directory in (root / "outputs" / "figures", root / "figures", root / "results"):
        if directory.exists():
            files.extend(path for path in directory.glob("**/*.png") if path.is_file() and path.stat().st_size > 0)
    files.extend(path for path in (root / "results").glob("**/classification_reports/*.json") if path.is_file() and path.stat().st_size > 0)

    if log_text_tables:
        files.extend(_existing_files(root, TEXT_CONTAINING_ARTIFACTS))
        predictions_dir = root / "outputs" / "predictions"
        if predictions_dir.exists():
            files.extend(path for path in predictions_dir.glob("*.jsonl") if path.is_file() and path.stat().st_size > 0)

    if log_model_files:
        for directory in (root / "models", root / "checkpoints", root / "results" / "transformer" / "model"):
            if directory.exists():
                files.extend(path for path in directory.glob("**/*") if path.is_file() and path.stat().st_size > 0)

    seen: set[Path] = set()
    unique_files = []
    for path in files:
        resolved = path.resolve()
        if resolved not in seen:
            unique_files.append(path)
            seen.add(resolved)
    return unique_files


def log_wandb_outputs(
    run: Any | None,
    *,
    paths: ProjectPaths,
    comparison_df: pd.DataFrame | None = None,
    log_artifacts: bool = True,
    log_text_tables: bool = False,
    log_model_files: bool = False,
    table_max_rows: int = 5000,
) -> dict[str, Any]:
    """Log final metrics and safe report artifacts to an active W&B run."""
    if run is None:
        return {"status": "skipped", "reason": "no active W&B run"}

    try:
        import wandb

        root = paths.project_root
        metrics_logged = _log_scalar_metrics(run, comparison_df if comparison_df is not None else pd.DataFrame())
        table_files = _existing_files(
            root,
            [
                "outputs/main_results.csv",
                "outputs/per_class_results.csv",
                "outputs/confusion_pairs.csv",
                "outputs/data_summary.csv",
                "outputs/label_distribution.csv",
                "outputs/hyperparameters.csv",
                "results/final_model_comparison.csv",
            ],
        )
        tables_logged = sum(_log_csv_table(run, path, max_rows=table_max_rows) for path in table_files)

        artifact_count = 0
        if log_artifacts:
            artifact = wandb.Artifact(
                name=f"{_safe_key(getattr(run, 'name', 'ledgar'))}-report-artifacts",
                type="coursework-results",
                metadata={
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "text_containing_artifacts_logged": bool(log_text_tables),
                    "model_files_logged": bool(log_model_files),
                },
            )
            for path in _artifact_files(paths, log_text_tables=log_text_tables, log_model_files=log_model_files):
                artifact.add_file(str(path), name=str(path.relative_to(root)))
                artifact_count += 1
            if artifact_count:
                run.log_artifact(artifact)

        print(
            "W&B logging complete: "
            f"{len(metrics_logged)} scalar metrics, {tables_logged} tables, {artifact_count} artifact files."
        )
        return {
            "status": "completed",
            "scalar_metrics": len(metrics_logged),
            "tables": int(tables_logged),
            "artifact_files": artifact_count,
            "text_artifacts_logged": bool(log_text_tables),
            "model_files_logged": bool(log_model_files),
        }
    except Exception as exc:
        print(f"W&B output logging failed/skipped: {type(exc).__name__}: {exc}")
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}


def finish_wandb_run(run: Any | None) -> None:
    """Close an optional W&B run."""
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        print(f"W&B finish skipped: {type(exc).__name__}: {exc}")
