"""Dummy baseline experiments for LEDGAR classification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import evaluate_predictions_common


def run_baseline_experiments(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id2label: dict[int, str],
    results_dir: Path | str,
    *,
    dataset_name: str = "LEDGAR",
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    """Run uniform random, train-distribution random, and majority baselines."""
    if train_df.empty:
        print("Baseline experiments skipped because LEDGAR data is unavailable.")
        return [], {}

    output_dir = Path(results_dir) / "baselines"
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
    return results, prediction_tables
