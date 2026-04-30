"""Central configuration for the LEDGAR pipeline."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


RANDOM_STATE = 42


@dataclass(frozen=True)
class ProjectPaths:
    """Filesystem layout for data, outputs, models, and checkpoints."""

    root: Path
    raw_lexglue: Path
    raw_original: Path
    processed: Path
    outputs: Path
    figures: Path
    predictions: Path
    models: Path
    models_classical: Path
    models_transformers: Path
    models_trained: Path
    models_trained_classical: Path
    models_trained_transformers: Path
    checkpoints: Path
    checkpoints_classical: Path
    checkpoints_transformers: Path
    src: Path
    notebooks: Path
    reference_notebooks: Path

    @classmethod
    def from_root(cls, project_root: Path | str = ".") -> "ProjectPaths":
        root = Path(project_root).resolve()
        return cls(
            root=root,
            raw_lexglue=root / "data" / "raw" / "lexglue_ledgar",
            raw_original=root / "data" / "raw" / "original_ledgar",
            processed=root / "data" / "processed",
            outputs=root / "outputs",
            figures=root / "outputs" / "figures",
            predictions=root / "outputs" / "predictions",
            models=root / "models",
            models_classical=root / "models" / "classical",
            models_transformers=root / "models" / "transformers",
            models_trained=root / "models" / "trained",
            models_trained_classical=root / "models" / "trained" / "classical",
            models_trained_transformers=root / "models" / "trained" / "transformers",
            checkpoints=root / "checkpoints",
            checkpoints_classical=root / "checkpoints" / "classical",
            checkpoints_transformers=root / "checkpoints" / "transformers",
            src=root / "src",
            notebooks=root / "notebooks",
            reference_notebooks=root / "reference_notebooks",
        )

    def ensure_dirs(self) -> None:
        """Create the configured directories."""
        for key, path in self.to_dict().items():
            if key != "root":
                Path(path).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, str]:
        """Return JSON/W&B-friendly path strings."""
        return {key: str(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class WandbConfig:
    """Mandatory W&B settings.

    This project treats W&B as required for training/evaluation runs. The
    tracking helper refuses offline or disabled modes so metrics and artifacts
    are sent to W&B.
    """

    project: str = field(default_factory=lambda: os.getenv("WANDB_PROJECT", "ledgar-clause-classification"))
    entity: str | None = field(default_factory=lambda: os.getenv("WANDB_ENTITY") or None)
    mode: str = field(default_factory=lambda: os.getenv("WANDB_MODE", "online"))
    require_online: bool = True
    log_raw_dataset_artifacts: bool = True
    log_processed_dataset_artifacts: bool = True
    log_prediction_artifacts: bool = True
    log_model_artifacts: bool = True
    log_figure_artifacts: bool = True
    table_max_rows: int = field(default_factory=lambda: int(os.getenv("WANDB_TABLE_MAX_ROWS", "500")))
    tags: tuple[str, ...] = ("coursework", "ledgar", "lexglue")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tags"] = list(self.tags)
        return data


@dataclass(frozen=True)
class PipelineConfig:
    """Experiment defaults for the notebook and scripts."""

    paths: ProjectPaths
    wandb: WandbConfig = field(default_factory=WandbConfig)
    random_state: int = RANDOM_STATE
    label_selection_mode: str = "top_n"
    top_n_labels: int = 20
    min_examples_per_label: int = 30
    manual_selected_labels: tuple[str, ...] = ()
    run_classical_models: bool = True
    run_transformers: bool = False
    transformer_max_length: int = 256
    transformer_batch_size: int = 8
    transformer_epochs: int = 3
    transformer_models: tuple[str, ...] = (
        "bert-base-uncased",
        "nlpaueb/legal-bert-base-uncased",
    )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["paths"] = self.paths.to_dict()
        data["wandb"] = self.wandb.to_dict()
        data["manual_selected_labels"] = list(self.manual_selected_labels)
        data["transformer_models"] = list(self.transformer_models)
        return data


def get_config(project_root: Path | str = ".") -> PipelineConfig:
    """Build the default pipeline configuration."""
    paths = ProjectPaths.from_root(project_root)
    return PipelineConfig(paths=paths)
