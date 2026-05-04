"""Dummy baseline experiments for LEDGAR classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data_setup import write_json
from .evaluation import evaluate_predictions_common, safe_name, sample_debug_frame, utc_now_iso, write_stage_status


def run_baseline_experiments(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id2label: dict[int, str],
    results_dir: Path | str,
    *,
    dataset_name: str = "LEDGAR",
    seed: int = 42,
    max_eval_samples: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    """Run uniform random, train-distribution random, and majority baselines."""
    output_dir = Path(results_dir) / "baselines"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
    stage_started = utc_now_iso()
    model_names = ["random_uniform", "random_train_distribution", "majority_baseline"]
    config = {
        "dataset_name": dataset_name,
        "seed": seed,
        "max_eval_samples": max_eval_samples,
        "train_rows_available": int(len(train_df)),
        "test_rows_available": int(len(test_df)),
        "num_labels": int(len(id2label)),
    }
    write_json(output_dir / "baseline_run_config.json", config)
    if train_df.empty or test_df.empty or not id2label:
        reason = "Baseline experiments skipped because train/test data or label mapping is unavailable."
        print(reason)
        skipped = [
            {
                "model_family": "baseline",
                "model_name": model_name,
                "training_type": "dummy",
                "dataset": dataset_name,
                "eval_split": "test",
                "sample_size": 0,
                "accuracy": np.nan,
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
                "status": "skipped",
                "error_type": "DataUnavailable",
                "error_message": reason,
                "notes": reason,
            }
            for model_name in model_names
        ]
        pd.DataFrame(skipped).to_csv(output_dir / "baseline_results.csv", index=False)
        for model_name in model_names:
            pd.DataFrame(columns=["text", "label", "label_id", "predicted_label_id", "predicted_label", "is_correct", "model_name", "dataset_name", "split"]).to_csv(
                output_dir / "predictions" / f"{safe_name(model_name)}_test_predictions.csv",
                index=False,
            )
        write_stage_status(
            output_dir / "baseline_stage_status.json",
            stage="baseline_models",
            status="skipped",
            error_type="DataUnavailable",
            error_message=reason,
            config=config,
            outputs={"results": str(output_dir / "baseline_results.csv")},
            started_at_utc=stage_started,
        )
        return [], {}

    test_df = sample_debug_frame(test_df, max_eval_samples, seed=seed)
    config.update({"test_rows_evaluated": int(len(test_df))})
    write_json(output_dir / "baseline_run_config.json", config)
    labels = sorted(id2label)
    rng = np.random.default_rng(seed)
    y_test = test_df["label_id"].astype(int).tolist()
    results = []
    prediction_tables = {}

    uniform_pred = rng.choice(labels, size=len(test_df), replace=True).astype(int).tolist()
    result, pred_df = evaluate_predictions_common(
        model_family="baseline",
        model_name="random_uniform",
        training_type="dummy",
        y_true=y_test,
        y_pred=uniform_pred,
        df=test_df,
        id2label=id2label,
        dataset_name=dataset_name,
        output_dir=output_dir,
        notes="Uniform random over selected labels.",
    )
    results.append(result)
    prediction_tables["random_uniform"] = pred_df

    train_counts = train_df["label_id"].value_counts().sort_index()
    probabilities = np.array([train_counts.get(label, 0) for label in labels], dtype=float)
    probabilities = probabilities / probabilities.sum()
    distribution_pred = rng.choice(labels, size=len(test_df), replace=True, p=probabilities).astype(int).tolist()
    result, pred_df = evaluate_predictions_common(
        model_family="baseline",
        model_name="random_train_distribution",
        training_type="dummy",
        y_true=y_test,
        y_pred=distribution_pred,
        df=test_df,
        id2label=id2label,
        dataset_name=dataset_name,
        output_dir=output_dir,
        notes="Random predictions sampled from the training label distribution.",
    )
    results.append(result)
    prediction_tables["random_train_distribution"] = pred_df

    majority_label = int(train_df["label_id"].mode().iloc[0])
    majority_pred = [majority_label] * len(test_df)
    result, pred_df = evaluate_predictions_common(
        model_family="baseline",
        model_name="majority_baseline",
        training_type="dummy",
        y_true=y_test,
        y_pred=majority_pred,
        df=test_df,
        id2label=id2label,
        dataset_name=dataset_name,
        output_dir=output_dir,
        notes="Always predicts the most frequent training label.",
    )
    results.append(result)
    prediction_tables["majority_baseline"] = pred_df

    pd.DataFrame(results).to_csv(output_dir / "baseline_results.csv", index=False)
    write_stage_status(
        output_dir / "baseline_stage_status.json",
        stage="baseline_models",
        status="completed",
        config=config,
        outputs={"results": str(output_dir / "baseline_results.csv")},
        notes="Dummy baselines evaluated on LEDGAR test only.",
        started_at_utc=stage_started,
    )
    return results, prediction_tables
