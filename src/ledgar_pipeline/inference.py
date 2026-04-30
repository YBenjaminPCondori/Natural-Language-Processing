"""Inference helpers for saved LEDGAR classical models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import pandas as pd

from evaluate import write_jsonl
from .config import WandbConfig, get_config
from .wandb_tracking import finish_wandb_run, log_artifact_paths, log_table, start_wandb_run


def load_label_mapping(outputs_dir: Path | str) -> tuple[dict[str, int], dict[int, str]]:
    """Load label mappings saved during preprocessing."""
    path = Path(outputs_dir) / "label_mapping.json"
    with path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    label_to_id = {str(key): int(value) for key, value in mapping["label_to_id"].items()}
    id_to_label = {int(key): str(value) for key, value in mapping["id_to_label"].items()}
    return label_to_id, id_to_label


def predict_texts(
    texts: Sequence[str],
    *,
    model_path: Path | str,
    outputs_dir: Path | str,
    model_name: str,
    split: str = "inference",
    prediction_path: Path | str | None = None,
    wandb_config: WandbConfig | None = None,
) -> pd.DataFrame:
    """Predict labels for raw clause texts and log the inference output to W&B."""
    outputs = Path(outputs_dir)
    _, id_to_label = load_label_mapping(outputs)
    model_path = Path(model_path)
    model = joblib.load(model_path)
    predictions = [int(pred) for pred in model.predict(list(texts))]

    rows = [
        {
            "text": text,
            "predicted_label": id_to_label[pred_id],
            "predicted_label_id": pred_id,
            "model_name": model_name,
            "split": split,
        }
        for text, pred_id in zip(texts, predictions)
    ]
    prediction_df = pd.DataFrame(rows)
    output_path = Path(prediction_path) if prediction_path else outputs / "predictions" / f"{model_name}_{split}_predictions.jsonl"
    write_jsonl(output_path, rows)

    run = start_wandb_run(
        run_name=f"{model_name}-{split}-inference",
        job_type="inference",
        config={"model_name": model_name, "model_path": str(model_path), "split": split},
        tags=["coursework", "ledgar", "inference"],
        wandb_config=wandb_config,
    )
    try:
        log_table(run, name="inference_predictions", dataframe=prediction_df)
        log_artifact_paths(
            run,
            name=f"{model_name}-{split}-predictions",
            artifact_type="predictions",
            paths=[output_path],
        )
    finally:
        finish_wandb_run(run)

    return prediction_df


def predict_from_jsonl(
    input_path: Path | str,
    *,
    text_column: str = "text",
    model_path: Path | str | None = None,
    model_name: str = "linear_svm",
    project_root: Path | str = ".",
) -> pd.DataFrame:
    """Load input JSONL, predict labels, save JSONL, and log to W&B."""
    config = get_config(project_root)
    paths = config.paths
    if model_path is None:
        model_path = paths.models_trained_classical / f"{model_name}.joblib"
    input_df = pd.read_json(input_path, lines=True)
    if text_column not in input_df.columns:
        raise ValueError(f"Input JSONL must contain a '{text_column}' column.")
    return predict_texts(
        input_df[text_column].fillna("").astype(str).tolist(),
        model_path=model_path,
        outputs_dir=paths.outputs,
        model_name=model_name,
        wandb_config=config.wandb,
    )
