"""Guarded Hugging Face Transformer training template for LEDGAR."""

from __future__ import annotations

import importlib.util
import inspect
import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from evaluate import append_result_row, evaluate_predictions, save_predictions_jsonl


RANDOM_STATE = 42
TRANSFORMER_RESULTS = "transformer_results.jsonl"


def transformers_available() -> bool:
    """Check whether the transformers package is installed."""
    return importlib.util.find_spec("transformers") is not None


def require_transformers() -> None:
    """Raise a clear error if transformers is unavailable."""
    if not transformers_available():
        raise ImportError(
            "The transformers package is not installed. Install it before setting "
            "RUN_TRANSFORMERS = True."
        )


def coerce_id_to_label(id_to_label: Mapping[int | str, str]) -> dict[int, str]:
    """Convert loaded JSON keys back to integer label IDs."""
    return {int(key): value for key, value in id_to_label.items()}


def transformer_model_plan() -> list[dict[str, str]]:
    """Return the two requested transformer model templates."""
    return [
        {
            "model_name": "bert-base-uncased",
            "model_slug": "bert_base",
            "output_subdir": "bert-base-uncased",
            "prediction_file": "bert_base_test_predictions.jsonl",
        },
        {
            "model_name": "nlpaueb/legal-bert-base-uncased",
            "model_slug": "legalbert",
            "output_subdir": "legal-bert",
            "prediction_file": "legalbert_test_predictions.jsonl",
        },
    ]


def _training_arguments_kwargs(TrainingArguments: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Adapt TrainingArguments names across transformers versions."""
    signature = inspect.signature(TrainingArguments.__init__)
    parameters = signature.parameters
    adapted = dict(kwargs)
    if "eval_strategy" in parameters and "evaluation_strategy" in adapted:
        adapted["eval_strategy"] = adapted.pop("evaluation_strategy")
    return {key: value for key, value in adapted.items() if key in parameters}


def build_training_arguments(TrainingArguments: Any, *, output_dir: Path, batch_size: int, epochs: int) -> Any:
    """Build Hugging Face TrainingArguments with safe defaults."""
    kwargs = {
        "output_dir": str(output_dir),
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "save_total_limit": 1,
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
        "seed": RANDOM_STATE,
    }
    return TrainingArguments(**_training_arguments_kwargs(TrainingArguments, kwargs))


def make_compute_metrics() -> Any:
    """Create the metric callback expected by Hugging Face Trainer."""
    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
            "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
            "macro_precision": precision_score(labels, predictions, average="macro", zero_division=0),
            "macro_recall": recall_score(labels, predictions, average="macro", zero_division=0),
        }

    return compute_metrics


def prepare_transformer_datasets(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: Any,
    *,
    max_length: int = 256,
) -> tuple[Any, Any, Any]:
    """Convert pandas splits into tokenized Hugging Face datasets."""
    from datasets import Dataset

    def to_dataset(df: pd.DataFrame) -> Any:
        data = df[["text", "label_id"]].rename(columns={"label_id": "labels"}).copy()
        data["labels"] = data["labels"].astype(int)
        return Dataset.from_pandas(data, preserve_index=False)

    def tokenize(batch: Mapping[str, list[str]]) -> Mapping[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    train_dataset = to_dataset(train_df).map(tokenize, batched=True, remove_columns=["text"])
    validation_dataset = to_dataset(validation_df).map(tokenize, batched=True, remove_columns=["text"])
    test_dataset = to_dataset(test_df).map(tokenize, batched=True, remove_columns=["text"])
    return train_dataset, validation_dataset, test_dataset


def train_one_transformer(
    model_name: str,
    model_slug: str,
    output_subdir: str,
    prediction_file: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    outputs_dir: Path | str,
    predictions_dir: Path | str,
    models_dir: Path | str,
    id_to_label: Mapping[int | str, str],
    max_length: int = 256,
    batch_size: int = 8,
    epochs: int = 3,
) -> dict[str, Any]:
    """Fine-tune one transformer and save its test artifacts."""
    require_transformers()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        warnings.warn(
            "CUDA is unavailable. Transformer fine-tuning will run on CPU and may be very slow."
        )

    outputs = Path(outputs_dir)
    predictions = Path(predictions_dir)
    models = Path(models_dir)
    model_output_dir = models / output_subdir
    id_to_label_int = coerce_id_to_label(id_to_label)
    label_to_id = {label: idx for idx, label in id_to_label_int.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, validation_dataset, test_dataset = prepare_transformer_datasets(
        train_df,
        validation_df,
        test_df,
        tokenizer,
        max_length=max_length,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(id_to_label_int),
        id2label={idx: label for idx, label in id_to_label_int.items()},
        label2id=label_to_id,
    )

    args = build_training_arguments(
        TrainingArguments,
        output_dir=model_output_dir,
        batch_size=batch_size,
        epochs=epochs,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(),
    )

    trainer.train()
    trainer.save_model(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

    test_output = trainer.predict(test_dataset)
    test_predictions = np.argmax(test_output.predictions, axis=-1).astype(int).tolist()
    prediction_path = save_predictions_jsonl(
        predictions / prediction_file,
        test_df,
        test_predictions,
        id_to_label=id_to_label_int,
        model_name=model_slug,
        split="test",
    )

    labels = sorted(id_to_label_int)
    target_names = [id_to_label_int[label_id] for label_id in labels]
    result = evaluate_predictions(
        test_df["label_id"].astype(int).tolist(),
        test_predictions,
        labels=labels,
        target_names=target_names,
        model_name=model_slug,
        split="test",
        params={
            "base_model": model_name,
            "max_length": max_length,
            "batch_size": batch_size,
            "epochs": epochs,
            "fp16": torch.cuda.is_available(),
            "best_model_metric": "validation_macro_f1",
        },
    )
    result["prediction_path"] = str(prediction_path)
    result["model_path"] = str(model_output_dir)
    append_result_row(outputs / TRANSFORMER_RESULTS, result)
    return result


def run_transformer_experiments(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    outputs_dir: Path | str,
    predictions_dir: Path | str,
    models_dir: Path | str,
    id_to_label: Mapping[int | str, str],
    run_transformers: bool = False,
    max_length: int = 256,
    batch_size: int = 8,
    epochs: int = 3,
) -> list[dict[str, Any]]:
    """Run transformer experiments only when explicitly enabled."""
    if not run_transformers:
        print("RUN_TRANSFORMERS is False. Transformer fine-tuning is skipped.")
        return []

    if not torch.cuda.is_available():
        warnings.warn(
            "RUN_TRANSFORMERS is True but CUDA is unavailable. Training will use CPU unless "
            "the environment is changed."
        )

    results = []
    for spec in transformer_model_plan():
        results.append(
            train_one_transformer(
                spec["model_name"],
                spec["model_slug"],
                spec["output_subdir"],
                spec["prediction_file"],
                train_df,
                validation_df,
                test_df,
                outputs_dir=outputs_dir,
                predictions_dir=predictions_dir,
                models_dir=models_dir,
                id_to_label=id_to_label,
                max_length=max_length,
                batch_size=batch_size,
                epochs=epochs,
            )
        )
    return results
