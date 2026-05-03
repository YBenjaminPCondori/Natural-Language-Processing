"""Fine-tuned Hugging Face transformer classifier."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from .data_setup import ensure_package, write_json
from .evaluation import evaluate_predictions_common


TRANSFORMER_PREDICTION_COLUMNS = ["text", "label", "label_id", "predicted_label", "predicted_label_id", "model_name", "split"]
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
    "reason",
    "notes",
]


def _runtime_payload(torch_module: Any, paths_note: Path | str) -> dict[str, Any]:
    """Collect transformer runtime details."""
    cuda_available = bool(torch_module.cuda.is_available())
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(paths_note),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": getattr(torch_module, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch_module.cuda.device_count()),
        "cuda_version": getattr(torch_module.version, "cuda", None),
        "gpu_name": torch_module.cuda.get_device_name(0) if cuda_available else None,
    }


def _write_transformer_skip_artifacts(output_dir: Path, skip_result: dict[str, Any], torch_module: Any, reason: str) -> None:
    """Write explicit skip artifacts with valid schemas."""
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "runtime.json", {**_runtime_payload(torch_module, output_dir), "runtime_note": reason})
    write_json(output_dir / "training_args.json", {"status": "not_executed", "reason": reason})
    pd.DataFrame([{**skip_result, "status": "skipped", "reason": reason}]).reindex(columns=TRANSFORMER_RESULT_COLUMNS).to_csv(
        output_dir / "transformer_results.csv",
        index=False,
    )
    pd.DataFrame(columns=TRANSFORMER_PREDICTION_COLUMNS).to_csv(output_dir / "transformer_predictions.csv", index=False)
    write_json(output_dir / "training_log_history.json", [])


def _training_arguments_kwargs(TrainingArguments: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Adapt TrainingArguments keyword names across transformers versions."""
    import inspect

    params = inspect.signature(TrainingArguments.__init__).parameters
    adapted = dict(kwargs)
    if "eval_strategy" in params and "evaluation_strategy" in adapted:
        adapted["eval_strategy"] = adapted.pop("evaluation_strategy")
    return {key: value for key, value in adapted.items() if key in params}


def _trainer_kwargs(Trainer: Any, kwargs: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Adapt Trainer keyword names across transformers versions."""
    import inspect

    params = inspect.signature(Trainer.__init__).parameters
    adapted = dict(kwargs)
    if "tokenizer" in params:
        adapted["tokenizer"] = tokenizer
    elif "processing_class" in params:
        adapted["processing_class"] = tokenizer
    return {key: value for key, value in adapted.items() if key in params}


def train_transformer_classifier(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id2label: dict[int, str],
    results_dir: Path | str,
    *,
    model_name: str,
    max_length: int = 256,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 3,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.0,
    batch_size_override: int | None = None,
    dataset_name: str = "LEDGAR",
    seed: int = 42,
    run_transformer: bool = True,
    wandb_enabled: bool = False,
    wandb_run_name: str | None = None,
    output_subdir: str = "transformer",
    evaluate_test: bool = True,
    early_stopping_patience: int | None = 1,
    save_total_limit: int = 1,
    trial_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fine-tune one transformer classifier, skipping gracefully when unavailable."""
    if not run_transformer:
        print("Transformer section skipped because RUN_TRANSFORMER is False.")
        return {"result": None, "predictions": pd.DataFrame(), "trainer": None, "skip_result": None}
    if train_df.empty:
        print("Transformer section skipped because LEDGAR data is unavailable.")
        return {"result": None, "predictions": pd.DataFrame(), "trainer": None, "skip_result": None}

    ensure_package("torch", "torch")
    import torch
    output_dir = Path(results_dir) / output_subdir

    if not torch.cuda.is_available():
        print("Transformer training skipped because GPU/CUDA is unavailable in this runtime.")
        reason = "GPU/CUDA unavailable."
        skip_result = {
            "model_family": "transformer",
            "model_name": model_name,
            "training_type": "fine-tuned supervised",
            "dataset": dataset_name,
            "eval_split": "test",
            "sample_size": 0,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "notes": f"Skipped: {reason}",
        }
        _write_transformer_skip_artifacts(output_dir, skip_result, torch, reason)
        return {"result": None, "predictions": pd.DataFrame(), "trainer": None, "skip_result": skip_result}

    try:
        ensure_package("transformers", "transformers")
        ensure_package("accelerate", "accelerate")
        from datasets import Dataset
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

        if early_stopping_patience is not None and early_stopping_patience > 0:
            try:
                from transformers import EarlyStoppingCallback

                callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
            except Exception:
                callbacks = []
        else:
            callbacks = []

        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "runtime.json", _runtime_payload(torch, output_dir))
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def compute_metrics(eval_pred: Any) -> dict[str, float]:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            label_ids = sorted(id2label)
            return {
                "accuracy": accuracy_score(labels, predictions),
                "macro_f1": f1_score(labels, predictions, labels=label_ids, average="macro", zero_division=0),
                "weighted_f1": f1_score(labels, predictions, labels=label_ids, average="weighted", zero_division=0),
            }

        def to_hf_dataset(df: pd.DataFrame) -> Any:
            data = df[["text", "label_id"]].rename(columns={"label_id": "labels"}).copy()
            data["labels"] = data["labels"].astype(int)
            dataset = Dataset.from_pandas(data, preserve_index=False)
            return dataset.map(
                lambda batch: tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length),
                batched=True,
                remove_columns=["text"],
            )

        train_dataset = to_hf_dataset(train_df)
        val_dataset = to_hf_dataset(validation_df)
        test_dataset = to_hf_dataset(test_df) if evaluate_test else None

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(id2label),
            id2label={int(key): value for key, value in id2label.items()},
            label2id={value: int(key) for key, value in id2label.items()},
        )

        batch_size = batch_size_override or (16 if torch.cuda.get_device_properties(0).total_memory > 12_000_000_000 else 8)
        args_dict = {
            "output_dir": str(output_dir / "checkpoints"),
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "learning_rate": learning_rate,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "num_train_epochs": num_train_epochs,
            "weight_decay": weight_decay,
            "warmup_ratio": warmup_ratio,
            "load_best_model_at_end": True,
            "metric_for_best_model": "macro_f1",
            "greater_is_better": True,
            "save_total_limit": save_total_limit,
            "report_to": ["wandb"] if wandb_enabled else [],
            "fp16": torch.cuda.is_available(),
            "seed": seed,
            "logging_strategy": "epoch",
        }
        if wandb_enabled and wandb_run_name:
            args_dict["run_name"] = wandb_run_name
        training_args = TrainingArguments(**_training_arguments_kwargs(TrainingArguments, args_dict))
        write_json(
            output_dir / "training_args.json",
            {
                **args_dict,
                "early_stopping_patience": early_stopping_patience,
                "trial_metadata": trial_metadata or {},
                "selection_metric": "validation_macro_f1",
                "test_evaluation_enabled": bool(evaluate_test),
            },
        )

        trainer = Trainer(
            **_trainer_kwargs(
                Trainer,
                {
                    "model": model,
                    "args": training_args,
                    "train_dataset": train_dataset,
                    "eval_dataset": val_dataset,
                    "compute_metrics": compute_metrics,
                    "callbacks": callbacks,
                },
                tokenizer,
            )
        )
        trainer.train()
        write_json(output_dir / "training_log_history.json", trainer.state.log_history)
        pd.DataFrame(trainer.state.log_history).to_csv(output_dir / "training_log_history.csv", index=False)

        validation_metrics = trainer.evaluate(eval_dataset=val_dataset)
        validation_payload = {
            key.replace("eval_", "validation_"): float(value)
            for key, value in validation_metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
        validation_payload.update(
            {
                "model_name": model_name,
                "selection_metric": "validation_macro_f1",
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_metric": trainer.state.best_metric,
                "trial_metadata": trial_metadata or {},
            }
        )
        write_json(output_dir / "validation_metrics.json", validation_payload)
        pd.DataFrame([validation_payload]).to_csv(output_dir / "validation_metrics.csv", index=False)

        if not evaluate_test:
            tokenizer.save_pretrained(output_dir / "model")
            trainer.save_model(output_dir / "model")
            return {
                "result": None,
                "predictions": pd.DataFrame(),
                "trainer": trainer,
                "skip_result": None,
                "validation_metrics": validation_payload,
                "output_dir": output_dir,
            }

        output = trainer.predict(test_dataset)
        y_pred = np.argmax(output.predictions, axis=-1).astype(int).tolist()
        y_true = test_df["label_id"].astype(int).tolist()
        result, pred_df = evaluate_predictions_common(
            model_family="transformer",
            model_name=model_name,
            training_type="fine-tuned supervised",
            y_true=y_true,
            y_pred=y_pred,
            df=test_df,
            id2label=id2label,
            dataset_name=dataset_name,
            output_dir=output_dir,
            notes="Fine-tuned Hugging Face sequence classifier.",
        )
        pd.DataFrame([result]).to_csv(output_dir / "transformer_results.csv", index=False)
        pred_df.to_csv(output_dir / "transformer_predictions.csv", index=False)
        trainer.save_model(output_dir / "model")
        tokenizer.save_pretrained(output_dir / "model")
        return {
            "result": result,
            "predictions": pred_df,
            "trainer": trainer,
            "skip_result": None,
            "validation_metrics": validation_payload,
            "output_dir": output_dir,
        }
    except Exception as exc:
        print(f"Transformer training failed or was skipped: {type(exc).__name__}: {exc}")
        reason = f"{type(exc).__name__}: {exc}"
        skip_result = {
            "model_family": "transformer",
            "model_name": model_name,
            "training_type": "fine-tuned supervised",
            "dataset": dataset_name,
            "eval_split": "test",
            "sample_size": 0,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "notes": f"Skipped/failed: {reason}",
        }
        _write_transformer_skip_artifacts(output_dir, skip_result, torch, reason)
        return {"result": None, "predictions": pd.DataFrame(), "trainer": None, "skip_result": skip_result}
