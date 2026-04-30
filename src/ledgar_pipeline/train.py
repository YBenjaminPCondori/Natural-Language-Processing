"""High-level training orchestration for the LEDGAR pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from preprocess_ledgar import (
    apply_label_mapping,
    create_eda_outputs,
    create_label_mapping,
    export_raw_hf_splits,
    filter_splits_to_labels,
    load_ledgar_dataset,
    save_label_artifacts,
    save_processed_splits,
    save_selection_artifacts,
    select_labels,
    standardise_splits,
)
from train_classical import run_baselines, run_classical_models
from train_transformer import run_transformer_experiments

from .config import PipelineConfig, get_config
from .wandb_tracking import log_preprocessing_run, verify_wandb_auth


def prepare_data(config: PipelineConfig) -> dict[str, Any]:
    """Load, preprocess, save, and log LEDGAR data with W&B."""
    paths = config.paths
    paths.ensure_dirs()
    verify_wandb_auth(config.wandb)

    ds, dataset_metadata = load_ledgar_dataset(paths.root)
    raw_paths = {}
    if dataset_metadata["dataset_source"] == "huggingface":
        raw_paths = export_raw_hf_splits(ds, paths.raw_lexglue)

    standardised_splits = standardise_splits(ds, dataset_metadata)
    save_label_artifacts(standardised_splits, paths.outputs)
    selected_labels, selection_warnings = select_labels(
        standardised_splits["train"],
        mode=config.label_selection_mode,
        top_n=config.top_n_labels,
        manual_labels=config.manual_selected_labels,
        min_examples_per_label=config.min_examples_per_label,
    )
    filtered_splits = filter_splits_to_labels(standardised_splits, selected_labels)
    label_to_id, id_to_label = create_label_mapping(selected_labels)
    processed_splits = apply_label_mapping(filtered_splits, label_to_id)
    selection_paths = save_selection_artifacts(
        selected_labels,
        label_to_id,
        id_to_label,
        paths.outputs,
    )
    processed_paths = save_processed_splits(processed_splits, paths.processed)
    eda_paths = create_eda_outputs(processed_splits, paths.outputs, paths.figures)

    metadata_paths = {
        **selection_paths,
        **eda_paths,
        "label_names": paths.outputs / "label_names.txt",
        "label_counts": paths.outputs / "label_counts.json",
    }
    log_preprocessing_run(
        raw_paths=raw_paths,
        processed_paths=processed_paths,
        metadata_paths=metadata_paths,
        sample_frames=processed_splits,
        config={
            "pipeline": config.to_dict(),
            "dataset_source": dataset_metadata["dataset_source"],
            "selection_warnings": selection_warnings,
        },
        wandb_config=config.wandb,
    )

    return {
        "dataset": ds,
        "dataset_metadata": dataset_metadata,
        "standardised_splits": standardised_splits,
        "processed_splits": processed_splits,
        "selected_labels": selected_labels,
        "selection_warnings": selection_warnings,
        "raw_paths": raw_paths,
        "processed_paths": processed_paths,
        "eda_paths": eda_paths,
    }


def train_all(config: PipelineConfig) -> dict[str, Any]:
    """Run baselines, classical models, and guarded transformer training."""
    paths = config.paths
    paths.ensure_dirs()
    prepared = prepare_data(config)
    processed = prepared["processed_splits"]
    id_to_label = {
        idx: label
        for idx, label in enumerate(prepared["selected_labels"])
    }

    baseline_results = run_baselines(
        processed["train"],
        processed["validation"],
        processed["test"],
        outputs_dir=paths.outputs,
        predictions_dir=paths.predictions,
        id_to_label=id_to_label,
        reset_results=True,
    )
    classical_results = {}
    if config.run_classical_models:
        classical_results = run_classical_models(
            processed["train"],
            processed["validation"],
            processed["test"],
            outputs_dir=paths.outputs,
            predictions_dir=paths.predictions,
            figures_dir=paths.figures,
            models_dir=paths.models_trained_classical,
            checkpoints_dir=paths.checkpoints_classical,
            id_to_label=id_to_label,
        )

    transformer_results = run_transformer_experiments(
        processed["train"],
        processed["validation"],
        processed["test"],
        outputs_dir=paths.outputs,
        predictions_dir=paths.predictions,
        models_dir=paths.models_trained_transformers,
        checkpoints_dir=paths.checkpoints_transformers,
        id_to_label=id_to_label,
        run_transformers=config.run_transformers,
        max_length=config.transformer_max_length,
        batch_size=config.transformer_batch_size,
        epochs=config.transformer_epochs,
    )

    return {
        "prepared": prepared,
        "baseline_results": baseline_results,
        "classical_results": classical_results,
        "transformer_results": transformer_results,
    }


def train_from_root(project_root: Path | str = ".") -> dict[str, Any]:
    """Build the default config and run the full training pipeline."""
    return train_all(get_config(project_root))
