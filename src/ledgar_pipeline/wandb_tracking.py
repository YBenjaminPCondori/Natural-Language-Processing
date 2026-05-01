"""Mandatory Weights & Biases tracking helpers."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .config import WandbConfig


DISALLOWED_WANDB_MODES = {"offline", "disabled", "dryrun"}


def wandb_is_installed() -> bool:
    """Return whether the wandb package is installed."""
    return importlib.util.find_spec("wandb") is not None


def require_wandb(config: WandbConfig | None = None) -> Any:
    """Import W&B and enforce online tracking."""
    settings = config or WandbConfig()
    if not wandb_is_installed():
        raise ImportError(
            "wandb is required for this pipeline. Install it with `pip install wandb` "
            "and authenticate with `wandb login` before running experiments."
        )

    mode = (settings.mode or os.getenv("WANDB_MODE", "online")).lower()
    if settings.require_online and mode in DISALLOWED_WANDB_MODES:
        raise RuntimeError(
            "W&B is compulsory for this pipeline and must run online. "
            "Unset WANDB_MODE or set WANDB_MODE=online, then run `wandb login`."
        )

    os.environ["WANDB_MODE"] = "online"
    import wandb

    api_key = os.getenv("WANDB_API_KEY") or getattr(wandb.api, "api_key", None)
    if settings.require_online and not api_key:
        raise RuntimeError(
            "W&B online tracking is compulsory but no API key is configured. "
            "Run `wandb login` or set WANDB_API_KEY before running this pipeline."
        )

    return wandb


def verify_wandb_auth(config: WandbConfig | None = None) -> None:
    """Validate W&B authentication before doing experiment work."""
    settings = config or WandbConfig()
    wandb = require_wandb(settings)
    try:
        api = wandb.Api()
        _ = api.viewer
    except Exception as exc:
        raise RuntimeError(
            "W&B authentication failed. Run `wandb login --relogin` or set a valid "
            "WANDB_API_KEY before running this compulsory-W&B pipeline."
        ) from exc


def login_wandb_if_needed(config: WandbConfig | None = None, *, relogin: bool = False) -> None:
    """Prompt for W&B login in notebooks/Colab when no API key is configured."""
    settings = config or WandbConfig()
    if not wandb_is_installed():
        raise ImportError(
            "wandb is required for this pipeline. Install it with `pip install wandb` "
            "before running the W&B login cell."
        )

    mode = (settings.mode or os.getenv("WANDB_MODE", "online")).lower()
    if settings.require_online and mode in DISALLOWED_WANDB_MODES:
        raise RuntimeError(
            "W&B is compulsory for this pipeline and must run online. "
            "Unset WANDB_MODE or set WANDB_MODE=online before logging in."
        )

    os.environ["WANDB_MODE"] = "online"
    import wandb

    api_key = os.getenv("WANDB_API_KEY") or getattr(wandb.api, "api_key", None)
    if not api_key or relogin:
        wandb.login(relogin=relogin)
    verify_wandb_auth(settings)


def start_wandb_run(
    *,
    run_name: str,
    job_type: str,
    config: Mapping[str, Any] | None = None,
    tags: Sequence[str] | None = None,
    wandb_config: WandbConfig | None = None,
) -> Any:
    """Start a mandatory online W&B run."""
    settings = wandb_config or WandbConfig()
    wandb = require_wandb(settings)
    try:
        return wandb.init(
            project=settings.project,
            entity=settings.entity,
            name=run_name,
            job_type=job_type,
            config=dict(config or {}),
            tags=list(tags or settings.tags),
            mode="online",
            reinit="finish_previous",
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to start the required W&B run. Confirm you are online and have "
            "authenticated with `wandb login` or set WANDB_API_KEY."
        ) from exc


def finish_wandb_run(run: Any) -> None:
    """Finish a W&B run if it exists."""
    if run is not None:
        run.finish()


def flatten_metrics(result_row: Mapping[str, Any]) -> dict[str, float]:
    """Flatten nested metric dictionaries for W&B logging."""
    model = str(result_row.get("model_name", "model"))
    split = str(result_row.get("split", "split"))
    metrics = result_row.get("metrics", {})
    return {
        f"{split}/{model}/{metric_name}": float(value)
        for metric_name, value in metrics.items()
        if isinstance(value, (int, float))
    }


def log_result_row(run: Any, result_row: Mapping[str, Any]) -> None:
    """Log one model result row to W&B."""
    run.log(flatten_metrics(result_row))
    model = str(result_row.get("model_name", "model"))
    split = str(result_row.get("split", "split"))
    for metric_name, value in result_row.get("metrics", {}).items():
        if isinstance(value, (int, float)):
            run.summary[f"{split}_{model}_{metric_name}"] = float(value)


def log_table(
    run: Any,
    *,
    name: str,
    dataframe: pd.DataFrame,
    max_rows: int = 500,
) -> None:
    """Log a DataFrame sample as a W&B table."""
    import wandb

    sample = dataframe.head(max_rows).copy()
    run.log({name: wandb.Table(dataframe=sample)})


def log_artifact_paths(
    run: Any,
    *,
    name: str,
    artifact_type: str,
    paths: Iterable[Path | str],
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Log one or more local files/directories as a W&B artifact."""
    import wandb

    artifact = wandb.Artifact(name=name, type=artifact_type, metadata=dict(metadata or {}))
    added = False
    for path_like in paths:
        path = Path(path_like)
        if path.is_file():
            artifact.add_file(str(path))
            added = True
        elif path.is_dir():
            artifact.add_dir(str(path))
            added = True
    if not added:
        raise FileNotFoundError(f"No existing files or directories found for W&B artifact: {name}")
    run.log_artifact(artifact)


def log_preprocessing_run(
    *,
    raw_paths: Mapping[str, Path | str],
    processed_paths: Mapping[str, Path | str],
    metadata_paths: Mapping[str, Path | str],
    sample_frames: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
    wandb_config: WandbConfig | None = None,
) -> None:
    """Log raw/processed data, metadata files, and samples to W&B."""
    settings = wandb_config or WandbConfig()
    run = start_wandb_run(
        run_name="ledgar-preprocessing",
        job_type="preprocess",
        config=config,
        tags=[*settings.tags, "preprocess"],
        wandb_config=settings,
    )
    try:
        if settings.log_raw_dataset_artifacts and raw_paths:
            log_artifact_paths(
                run,
                name="ledgar-raw-jsonl",
                artifact_type="dataset",
                paths=raw_paths.values(),
                metadata={"source": "coastalcph/lex_glue", "config": "ledgar"},
            )
        if settings.log_processed_dataset_artifacts and processed_paths:
            log_artifact_paths(
                run,
                name="ledgar-processed-jsonl",
                artifact_type="dataset",
                paths=processed_paths.values(),
            )
        if metadata_paths:
            log_artifact_paths(
                run,
                name="ledgar-metadata-and-eda",
                artifact_type="analysis",
                paths=metadata_paths.values(),
            )
        for split, frame in sample_frames.items():
            log_table(
                run,
                name=f"{split}_sample_rows",
                dataframe=frame,
                max_rows=settings.table_max_rows,
            )
    finally:
        finish_wandb_run(run)
