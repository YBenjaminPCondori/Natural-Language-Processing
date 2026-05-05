"""Two-stage hyperparameter tuning for the main transformer encoder."""

from __future__ import annotations

import random
import re
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data_setup import build_project_paths, ensure_package, write_json
from .evaluation import write_stage_status
from .transformer_model import train_transformer_classifier
from .wandb_reporting import finish_wandb_run, log_wandb_outputs, start_wandb_run


HPT_RESULT_COLUMNS = [
    "stage",
    "trial_number",
    "run_name",
    "model_name",
    "status",
    "validation_macro_f1",
    "validation_accuracy",
    "validation_weighted_f1",
    "learning_rate",
    "batch_size",
    "epochs",
    "weight_decay",
    "warmup_ratio",
    "max_length",
    "output_dir",
    "best_checkpoint",
    "reason",
    "error_type",
    "error_message",
    "started_at_utc",
    "finished_at_utc",
    "selected_for_bayes",
    "selected_for_final",
]


DEFAULT_STAGE5A_SEARCH_SPACE = {
    "learning_rate": [1e-6, 3e-6, 1e-5, 3e-5, 1e-4],
    "batch_size": [8, 16, 32],
    "epochs": [2],
    "weight_decay": [0.0, 0.01, 0.05, 0.1, 0.2],
    "warmup_ratio": [0.0, 0.06, 0.1, 0.2],
    "max_length": [128, 256, 512],
}

DEFAULT_STAGE5B_SEARCH_SPACE = {
    "learning_rate": [1e-5, 2e-5, 3e-5, 5e-5],
    "batch_size": [16, 32],
    "epochs": [3],
    "weight_decay": [0.01, 0.05, 0.1],
    "warmup_ratio": [0.05, 0.1, 0.15],
    "max_length": [256, 512],
}

# Backwards-compatible alias used by older notebooks.
DEFAULT_SEARCH_SPACE = DEFAULT_STAGE5A_SEARCH_SPACE


@dataclass
class TransformerHPTConfig:
    """Configuration for main-transformer HPT."""

    model_name: str = "distilbert-base-uncased"
    random_trials: int = 8
    bayes_trials: int = 8
    seed: int = 42
    early_stopping_patience: int = 1
    save_total_limit: int = 1
    final_retrain: bool = True
    final_retrain_epochs: int = 4
    search_space: dict[str, list[Any]] = field(default_factory=lambda: dict(DEFAULT_STAGE5A_SEARCH_SPACE))
    stage5a_search_space: dict[str, list[Any]] | None = None
    stage5b_search_space: dict[str, list[Any]] = field(default_factory=lambda: dict(DEFAULT_STAGE5B_SEARCH_SPACE))
    secondary_transformer_policy: str = "reuse_best_config_or_light_random_search_only"
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    max_eval_samples: int | None = None
    smoke_test: bool = False


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_") or "item"


def _label_distribution(train_df: pd.DataFrame) -> pd.DataFrame:
    counts = train_df["label"].value_counts().rename_axis("label").reset_index(name="train_count")
    counts["train_fraction"] = counts["train_count"] / max(int(counts["train_count"].sum()), 1)
    return counts


def _choice_config(rng: random.Random, search_space: dict[str, list[Any]]) -> dict[str, Any]:
    return {key: rng.choice(values) for key, values in search_space.items()}


def _dedupe_random_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[tuple[str, Any], ...]] = set()
    deduped = []
    for config in configs:
        key = tuple(sorted(config.items()))
        if key in seen:
            continue
        deduped.append(config)
        seen.add(key)
    return deduped


def _random_configs(config: TransformerHPTConfig) -> list[dict[str, Any]]:
    rng = random.Random(config.seed)
    search_space = config.stage5a_search_space or config.search_space
    candidates = [_choice_config(rng, search_space) for _ in range(max(config.random_trials * 4, config.random_trials))]
    return _dedupe_random_configs(candidates)[: config.random_trials]


def _metric_value(metrics: dict[str, Any], key: str) -> float:
    value = metrics.get(key)
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def _trial_row(
    *,
    stage: str,
    trial_number: int,
    run_name: str,
    model_name: str,
    status: str,
    trial_config: dict[str, Any],
    output_dir: Path,
    validation_metrics: dict[str, Any] | None = None,
    reason: str = "",
    started_at_utc: str,
    finished_at_utc: str | None = None,
) -> dict[str, Any]:
    validation_metrics = validation_metrics or {}
    return {
        "stage": stage,
        "trial_number": trial_number,
        "run_name": run_name,
        "model_name": model_name,
        "status": status,
        "validation_macro_f1": _metric_value(validation_metrics, "validation_macro_f1"),
        "validation_accuracy": _metric_value(validation_metrics, "validation_accuracy"),
        "validation_weighted_f1": _metric_value(validation_metrics, "validation_weighted_f1"),
        "learning_rate": trial_config.get("learning_rate"),
        "batch_size": trial_config.get("batch_size"),
        "epochs": trial_config.get("epochs"),
        "weight_decay": trial_config.get("weight_decay"),
        "warmup_ratio": trial_config.get("warmup_ratio"),
        "max_length": trial_config.get("max_length"),
        "output_dir": str(output_dir),
        "best_checkpoint": validation_metrics.get("best_checkpoint", ""),
        "reason": reason,
        "error_type": reason.split(":", 1)[0] if status in {"failed", "skipped"} and ":" in reason else (status.capitalize() if status in {"failed", "skipped"} else ""),
        "error_message": reason if status in {"failed", "skipped"} else "",
        "started_at_utc": started_at_utc,
        "finished_at_utc": finished_at_utc or datetime.now(timezone.utc).isoformat(),
        "selected_for_bayes": False,
        "selected_for_final": False,
    }


def _start_trial_run(
    *,
    enabled: bool,
    project: str,
    entity: str | None,
    mode: str,
    run_name: str,
    group: str,
    tags: list[str],
    config: dict[str, Any],
) -> Any | None:
    return start_wandb_run(
        enabled=enabled,
        project=project,
        entity=entity,
        run_name=run_name,
        group=group,
        tags=tags,
        config=config,
        mode=mode,
    )


def _log_trial_distribution(run: Any | None, label_distribution: pd.DataFrame) -> None:
    if run is None:
        return
    try:
        import wandb

        run.log({"train_label_distribution": wandb.Table(dataframe=label_distribution)})
    except Exception as exc:
        print(f"W&B label distribution logging skipped: {type(exc).__name__}: {exc}")


def _run_single_trial(
    *,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id2label: dict[int, str],
    results_dir: Path,
    run_root: Path,
    dataset_name: str,
    config: TransformerHPTConfig,
    trial_config: dict[str, Any],
    stage: str,
    trial_number: int,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_mode: str,
    label_distribution: pd.DataFrame,
) -> dict[str, Any]:
    run_name = f"{stage}_{trial_number:02d}"
    output_subdir = (run_root.relative_to(results_dir) / run_name).as_posix()
    output_dir = results_dir / output_subdir
    started_at = datetime.now(timezone.utc).isoformat()
    tags = ["hpt", "transformer", "legal-clause-classification"]
    tags.append("random-search" if "stage5a" in stage else "bayes-opt")
    trial_payload = {
        "stage": stage,
        "trial_number": trial_number,
        "model_name": config.model_name,
        "objective": "validation_macro_f1",
        "search_config": trial_config,
        "early_stopping_patience": config.early_stopping_patience,
        "save_total_limit": config.save_total_limit,
        "stage5a_policy": "random search, 2 epochs per trial, validation macro-F1 objective",
        "stage5b_policy": "narrowed Bayesian search, 3 epochs per trial, validation macro-F1 objective",
        "test_used": False,
    }
    run = _start_trial_run(
        enabled=wandb_enabled,
        project=wandb_project,
        entity=wandb_entity,
        mode=wandb_mode,
        run_name=run_name,
        group="stage5_transformer_hpt",
        tags=tags,
        config=trial_payload,
    )
    _log_trial_distribution(run, label_distribution)
    trial_epochs = 1 if config.smoke_test else int(trial_config["epochs"])
    try:
        output = train_transformer_classifier(
            train_df,
            validation_df,
            test_df,
            id2label,
            results_dir,
            model_name=config.model_name,
            max_length=int(trial_config["max_length"]),
            learning_rate=float(trial_config["learning_rate"]),
            num_train_epochs=trial_epochs,
            weight_decay=float(trial_config["weight_decay"]),
            warmup_ratio=float(trial_config["warmup_ratio"]),
            batch_size_override=int(trial_config["batch_size"]),
            dataset_name=dataset_name,
            seed=config.seed,
            run_transformer=True,
            wandb_enabled=run is not None,
            wandb_run_name=run_name,
            output_subdir=output_subdir,
            evaluate_test=False,
            early_stopping_patience=config.early_stopping_patience,
            save_total_limit=config.save_total_limit,
            trial_metadata=trial_payload,
            max_train_samples=config.max_train_samples,
            max_validation_samples=config.max_validation_samples,
            max_eval_samples=config.max_eval_samples,
            smoke_test=config.smoke_test,
        )
        validation_metrics = output.get("validation_metrics") or {}
        if output.get("skip_result") is not None:
            status = "skipped"
            reason = output["skip_result"].get("notes", "trial skipped")
        elif validation_metrics:
            status = "completed"
            reason = ""
        else:
            status = "failed"
            reason = "No validation metrics were produced."
        row = _trial_row(
            stage=stage,
            trial_number=trial_number,
            run_name=run_name,
            model_name=config.model_name,
            status=status,
            trial_config=trial_config,
            output_dir=output_dir,
            validation_metrics=validation_metrics,
            reason=reason,
            started_at_utc=started_at,
        )
        if run is not None and status == "completed":
            run.log(
                {
                    "validation/macro_f1": row["validation_macro_f1"],
                    "validation/accuracy": row["validation_accuracy"],
                    "validation/weighted_f1": row["validation_weighted_f1"],
                }
            )
        return row
    except Exception as exc:
        return _trial_row(
            stage=stage,
            trial_number=trial_number,
            run_name=run_name,
            model_name=config.model_name,
            status="failed",
            trial_config=trial_config,
            output_dir=output_dir,
            reason=f"{type(exc).__name__}: {exc}",
            started_at_utc=started_at,
        )
    finally:
        finish_wandb_run(run)


def _best_completed(results: pd.DataFrame) -> pd.Series | None:
    completed = results[(results["status"] == "completed") & results["validation_macro_f1"].notna()].copy()
    if completed.empty:
        return None
    return completed.sort_values("validation_macro_f1", ascending=False).iloc[0]


def _row_to_config(row: pd.Series) -> dict[str, Any]:
    return {
        "learning_rate": float(row["learning_rate"]),
        "batch_size": int(row["batch_size"]),
        "epochs": int(row["epochs"]),
        "weight_decay": float(row["weight_decay"]),
        "warmup_ratio": float(row["warmup_ratio"]),
        "max_length": int(row["max_length"]),
    }


def _run_bayes_search(
    *,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id2label: dict[int, str],
    results_dir: Path,
    run_root: Path,
    dataset_name: str,
    config: TransformerHPTConfig,
    existing_rows: list[dict[str, Any]],
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_mode: str,
    label_distribution: pd.DataFrame,
) -> list[dict[str, Any]]:
    if config.bayes_trials <= 0:
        return []

    try:
        ensure_package("optuna", "optuna>=3.6")
        import optuna
    except Exception as exc:
        return [
            _trial_row(
                stage="stage5b_bayes_trial",
                trial_number=1,
                run_name="stage5b_bayes_trial_skipped",
                model_name=config.model_name,
                status="skipped",
                trial_config={key: None for key in DEFAULT_STAGE5B_SEARCH_SPACE},
                output_dir=run_root / "stage5b_bayes_trial_skipped",
                reason=f"Optuna unavailable: {type(exc).__name__}: {exc}",
                started_at_utc=datetime.now(timezone.utc).isoformat(),
            )
        ]

    stage_rows: list[dict[str, Any]] = []
    study = optuna.create_study(direction="maximize", study_name=f"{safe_name(config.model_name)}_stage5b_bayes")
    random_frame = pd.DataFrame(existing_rows)
    best_random = _best_completed(random_frame) if not random_frame.empty else None
    if best_random is not None:
        study.enqueue_trial(_row_to_config(best_random))

    def objective(trial: Any) -> float:
        bayes_space = config.stage5b_search_space
        trial_config = {
            "learning_rate": trial.suggest_categorical("learning_rate", bayes_space["learning_rate"]),
            "batch_size": trial.suggest_categorical("batch_size", bayes_space["batch_size"]),
            "epochs": trial.suggest_categorical("epochs", bayes_space["epochs"]),
            "weight_decay": trial.suggest_categorical("weight_decay", bayes_space["weight_decay"]),
            "warmup_ratio": trial.suggest_categorical("warmup_ratio", bayes_space["warmup_ratio"]),
            "max_length": trial.suggest_categorical("max_length", bayes_space["max_length"]),
        }
        row = _run_single_trial(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            id2label=id2label,
            results_dir=results_dir,
            run_root=run_root,
            dataset_name=dataset_name,
            config=config,
            trial_config=trial_config,
            stage="stage5b_bayes_trial",
            trial_number=trial.number + 1,
            wandb_enabled=wandb_enabled,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_mode=wandb_mode,
            label_distribution=label_distribution,
        )
        stage_rows.append(row)
        score = row.get("validation_macro_f1")
        if row.get("status") != "completed" or pd.isna(score):
            return 0.0
        return float(score)

    study.optimize(objective, n_trials=config.bayes_trials)
    write_json(run_root / "bayes_study_best_params.json", study.best_params if len(study.trials) else {})
    return stage_rows


def _archive_existing_final(results_dir: Path, archive_root: Path) -> None:
    final_dir = results_dir / "transformer"
    if not final_dir.exists():
        return
    archive_target = archive_root / "previous_results_transformer"
    if archive_target.exists():
        shutil.rmtree(archive_target)
    shutil.copytree(final_dir, archive_target)


def _plot_hpt(results: pd.DataFrame, project_root: Path) -> None:
    figure_path = project_root / "figures" / "hpt_validation_macro_f1.png"
    output_figure_path = project_root / "outputs" / "figures" / "hpt_validation_macro_f1.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    output_figure_path.parent.mkdir(parents=True, exist_ok=True)

    completed = results[(results["status"] == "completed") & results["validation_macro_f1"].notna()].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    if completed.empty:
        ax.text(0.5, 0.5, "No completed transformer HPT trials yet", ha="center", va="center")
        ax.set_axis_off()
    else:
        completed["trial_label"] = completed["stage"].astype(str).str.replace("_trial", "", regex=False) + "-" + completed["trial_number"].astype(str)
        ax.plot(range(len(completed)), completed["validation_macro_f1"], marker="o", linewidth=1)
        ax.set_xticks(range(len(completed)))
        ax.set_xticklabels(completed["trial_label"], rotation=45, ha="right", fontsize=8)
        for stage, group in completed.groupby("stage"):
            ax.axhline(group["validation_macro_f1"].max(), linestyle="--", linewidth=1, label=f"{stage} best")
        ax.set_ylabel("Validation macro-F1")
        ax.set_title("Transformer HPT Validation Macro-F1 by Trial")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    frame = df.fillna("")
    headers = list(frame.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in frame.iterrows():
        values = [str(row[column]).replace("\n", " ") for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_hpt_artifacts(
    *,
    project_root: Path,
    run_root: Path,
    config: TransformerHPTConfig,
    rows: list[dict[str, Any]],
    best_config: dict[str, Any] | None,
    final_output: dict[str, Any] | None,
) -> pd.DataFrame:
    results = pd.DataFrame(rows).reindex(columns=HPT_RESULT_COLUMNS)
    best = _best_completed(results)
    if best is not None:
        results.loc[best.name, "selected_for_final"] = True
    random_best = _best_completed(results[results["stage"] == "stage5a_random_trial"]) if not results.empty else None
    if random_best is not None:
        results.loc[random_best.name, "selected_for_bayes"] = True

    results.to_csv(run_root / "hyperparameter_search_results.csv", index=False)
    project_outputs = project_root / "outputs"
    project_outputs.mkdir(parents=True, exist_ok=True)
    results.to_csv(project_outputs / "hyperparameter_search_results.csv", index=False)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "main_model": config.model_name,
        "objective": "validation_macro_f1",
        "early_stopping_patience": config.early_stopping_patience,
        "save_total_limit": config.save_total_limit,
        "test_policy": "Test split is not evaluated during HPT trials; final retrain evaluates test only after validation selection.",
        "stage5a_policy": "Random search with 2 epochs per trial.",
        "stage5b_policy": "Narrowed Bayesian search with 3 epochs per trial.",
        "final_retrain_epochs": config.final_retrain_epochs,
        "class_imbalance_policy": "Label distribution is logged and macro F1 is the main metric; class weights are not added by this HPT runner.",
        "secondary_transformer_policy": config.secondary_transformer_policy,
        "best_config": best_config,
        "final_retrain_output_dir": str(final_output.get("output_dir")) if final_output else None,
    }
    write_json(run_root / "best_transformer_configs.json", summary)
    write_json(project_outputs / "best_transformer_configs.json", summary)
    if not results.empty and results["status"].eq("completed").any():
        stage_status = "completed"
        error_type = ""
        error_message = ""
    elif not results.empty and results["status"].eq("skipped").all():
        stage_status = "skipped"
        error_type = "Skipped"
        error_message = "; ".join(results["reason"].dropna().astype(str).unique().tolist())
    else:
        stage_status = "failed"
        error_type = "NoCompletedTrial"
        error_message = "; ".join(results["reason"].dropna().astype(str).unique().tolist())
    write_stage_status(
        run_root / "transformer_hpt_stage_status.json",
        stage="transformer_hpt",
        status=stage_status,
        error_type=error_type,
        error_message=error_message,
        config=asdict(config),
        outputs={
            "hyperparameter_search_results": str(run_root / "hyperparameter_search_results.csv"),
            "best_transformer_configs": str(run_root / "best_transformer_configs.json"),
            "project_hpt_results": str(project_outputs / "hyperparameter_search_results.csv"),
        },
        notes="HPT trials evaluate validation only; final retrain/test runs only after validation selection.",
    )
    write_stage_status(
        project_outputs / "transformer_hpt_stage_status.json",
        stage="transformer_hpt",
        status=stage_status,
        error_type=error_type,
        error_message=error_message,
        config=asdict(config),
        outputs={
            "run_root": str(run_root),
            "hyperparameter_search_results": str(project_outputs / "hyperparameter_search_results.csv"),
            "best_transformer_configs": str(project_outputs / "best_transformer_configs.json"),
        },
        notes="HPT trials evaluate validation only; final retrain/test runs only after validation selection.",
    )

    lines = [
        "# Transformer HPT Summary",
        "",
        f"Main model: `{config.model_name}`",
        "",
        "- Objective: validation macro-F1.",
        "- Early stopping: patience=1 by default.",
        "- Checkpoint policy: save only the best checkpoint per trial.",
        "- Test policy: test split is reserved for final retraining/evaluation only.",
        "- Secondary transformer policy: reuse the selected config or run only a light search.",
        "",
        "## Trials",
        _markdown_table(results),
    ]
    (project_outputs / "transformer_training_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _plot_hpt(results, project_root)
    return results


def _log_hpt_summary_to_wandb(
    *,
    project_root: Path,
    config: TransformerHPTConfig,
    results: pd.DataFrame,
    run_root: Path,
    wandb_enabled: bool,
    wandb_project: str,
    wandb_entity: str | None,
    wandb_mode: str,
) -> None:
    """Log HPT summary artifacts after plots have been created."""
    if not wandb_enabled:
        return
    run = start_wandb_run(
        enabled=wandb_enabled,
        project=wandb_project,
        entity=wandb_entity,
        run_name="stage5_hpt_summary",
        group="stage5_transformer_hpt",
        tags=["hpt", "summary", "smoke-test" if config.smoke_test else "full-run"],
        config={
            "model_name": config.model_name,
            "run_root": str(run_root),
            "smoke_test": bool(config.smoke_test),
            "random_trials": int(config.random_trials),
            "bayes_trials": int(config.bayes_trials),
            "final_retrain": bool(config.final_retrain),
        },
        mode=wandb_mode,
    )
    try:
        if run is None:
            return
        comparison_df = results.copy()
        if "validation_macro_f1" in comparison_df:
            comparison_df["macro_f1"] = comparison_df["validation_macro_f1"]
        if "validation_accuracy" in comparison_df:
            comparison_df["accuracy"] = comparison_df["validation_accuracy"]
        if "validation_weighted_f1" in comparison_df:
            comparison_df["weighted_f1"] = comparison_df["validation_weighted_f1"]
        comparison_df["dataset_name"] = "LEDGAR_validation"
        log_wandb_outputs(
            run,
            paths=build_project_paths(project_root),
            comparison_df=comparison_df,
            log_artifacts=True,
            log_text_tables=False,
            log_model_files=False,
        )
    finally:
        finish_wandb_run(run)


def run_two_stage_transformer_hpt(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id2label: dict[int, str],
    results_dir: Path | str,
    *,
    dataset_name: str = "LEDGAR",
    config: TransformerHPTConfig | None = None,
    wandb_enabled: bool = False,
    wandb_project: str = "ledgar-clause-classification",
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
) -> dict[str, Any]:
    """Run random search, Bayesian search, then validation-selected final retraining."""
    config = config or TransformerHPTConfig()
    if config.smoke_test:
        config.random_trials = min(config.random_trials, 1)
        config.bayes_trials = 0
        config.final_retrain = False
    results_dir = Path(results_dir)
    project_root = results_dir.parent
    run_root = results_dir / "transformer_hpt" / f"{utc_stamp()}_{safe_name(config.model_name)}"
    run_root.mkdir(parents=True, exist_ok=True)

    label_distribution = _label_distribution(train_df)
    label_distribution.to_csv(run_root / "label_distribution.csv", index=False)
    (project_root / "outputs").mkdir(parents=True, exist_ok=True)
    label_distribution.to_csv(project_root / "outputs" / "transformer_hpt_label_distribution.csv", index=False)
    write_json(run_root / "hpt_config.json", asdict(config))

    rows: list[dict[str, Any]] = []

    try:
        ensure_package("torch", "torch")
        import torch

        if not torch.cuda.is_available():
            reason = "GPU/CUDA unavailable; transformer HPT requires CUDA."
            rows.append(
                _trial_row(
                    stage="stage5a_random_trial",
                    trial_number=1,
                    run_name="stage5a_random_trial_skipped",
                    model_name=config.model_name,
                    status="skipped",
                    trial_config={key: None for key in (config.stage5a_search_space or config.search_space)},
                    output_dir=run_root / "stage5a_random_trial_skipped",
                    reason=reason,
                    started_at_utc=datetime.now(timezone.utc).isoformat(),
                )
            )
            results = _write_hpt_artifacts(project_root=project_root, run_root=run_root, config=config, rows=rows, best_config=None, final_output=None)
            return {"status": "skipped", "reason": reason, "run_root": run_root, "results": results, "best_config": None, "final_output": None}
    except Exception as exc:
        reason = f"Torch/CUDA check failed: {type(exc).__name__}: {exc}"
        rows.append(
            _trial_row(
                stage="stage5a_random_trial",
                trial_number=1,
                run_name="stage5a_random_trial_failed",
                model_name=config.model_name,
                status="failed",
                trial_config={key: None for key in (config.stage5a_search_space or config.search_space)},
                output_dir=run_root / "stage5a_random_trial_failed",
                reason=reason,
                started_at_utc=datetime.now(timezone.utc).isoformat(),
            )
        )
        results = _write_hpt_artifacts(project_root=project_root, run_root=run_root, config=config, rows=rows, best_config=None, final_output=None)
        return {"status": "failed", "reason": reason, "run_root": run_root, "results": results, "best_config": None, "final_output": None}

    for index, trial_config in enumerate(_random_configs(config), start=1):
        rows.append(
            _run_single_trial(
                train_df=train_df,
                validation_df=validation_df,
                test_df=test_df,
                id2label=id2label,
                results_dir=results_dir,
                run_root=run_root,
                dataset_name=dataset_name,
                config=config,
                trial_config=trial_config,
                stage="stage5a_random_trial",
                trial_number=index,
                wandb_enabled=wandb_enabled,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                wandb_mode=wandb_mode,
                label_distribution=label_distribution,
            )
        )

    rows.extend(
        _run_bayes_search(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            id2label=id2label,
            results_dir=results_dir,
            run_root=run_root,
            dataset_name=dataset_name,
            config=config,
            existing_rows=rows,
            wandb_enabled=wandb_enabled,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            wandb_mode=wandb_mode,
            label_distribution=label_distribution,
        )
    )

    interim = pd.DataFrame(rows).reindex(columns=HPT_RESULT_COLUMNS)
    best = _best_completed(interim)
    best_config = _row_to_config(best) if best is not None else None
    final_output = None

    if best_config is not None and config.final_retrain:
        _archive_existing_final(results_dir, run_root)
        run_name = "final_retrain_best_transformer"
        run = _start_trial_run(
            enabled=wandb_enabled,
            project=wandb_project,
            entity=wandb_entity,
            mode=wandb_mode,
            run_name=run_name,
            group="stage5_transformer_hpt",
            tags=["hpt", "transformer", "legal-clause-classification", "final-retrain"],
            config={
                "model_name": config.model_name,
                "selected_by": "validation_macro_f1",
                "best_config": best_config,
                "test_used": True,
            },
        )
        try:
            final_output = train_transformer_classifier(
                train_df,
                validation_df,
                test_df,
                id2label,
                results_dir,
                model_name=config.model_name,
                max_length=int(best_config["max_length"]),
                learning_rate=float(best_config["learning_rate"]),
                num_train_epochs=1 if config.smoke_test else int(config.final_retrain_epochs),
                weight_decay=float(best_config["weight_decay"]),
                warmup_ratio=float(best_config["warmup_ratio"]),
                batch_size_override=int(best_config["batch_size"]),
                dataset_name=dataset_name,
                seed=config.seed,
                run_transformer=True,
                wandb_enabled=run is not None,
                wandb_run_name=run_name,
                output_subdir="transformer",
                evaluate_test=True,
                early_stopping_patience=config.early_stopping_patience,
                save_total_limit=config.save_total_limit,
                trial_metadata={
                    "stage": "final_retrain_best_transformer",
                    "selected_by": "validation_macro_f1",
                    "best_hpt_config": best_config,
                    "source_hpt_run": str(run_root),
                },
                max_train_samples=config.max_train_samples,
                max_validation_samples=config.max_validation_samples,
                max_eval_samples=config.max_eval_samples,
                smoke_test=config.smoke_test,
            )
            if run is not None and final_output and final_output.get("result"):
                result = final_output["result"]
                run.log(
                    {
                        "test/accuracy": float(result["accuracy"]),
                        "test/macro_f1": float(result["macro_f1"]),
                        "test/weighted_f1": float(result["weighted_f1"]),
                    }
                )
        finally:
            finish_wandb_run(run)

    results = _write_hpt_artifacts(project_root=project_root, run_root=run_root, config=config, rows=rows, best_config=best_config, final_output=final_output)
    _log_hpt_summary_to_wandb(
        project_root=project_root,
        config=config,
        results=results,
        run_root=run_root,
        wandb_enabled=wandb_enabled,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_mode=wandb_mode,
    )
    return {
        "status": "completed" if best_config is not None else "failed",
        "reason": "" if best_config is not None else "No completed HPT trial produced validation macro-F1.",
        "run_root": run_root,
        "results": results,
        "best_config": best_config,
        "final_output": final_output,
    }
