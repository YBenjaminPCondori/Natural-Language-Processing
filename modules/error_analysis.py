"""Error analysis helpers for LEDGAR model outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def top_confusions(pred_df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Return the most frequent true-label/predicted-label error pairs."""
    errors = pred_df[pred_df["label_id"] != pred_df["predicted_label_id"]].copy()
    if errors.empty:
        return pd.DataFrame(columns=["label", "predicted_label", "count"])
    return (
        errors.groupby(["label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )


def run_error_analysis(
    comparison_df: pd.DataFrame,
    prediction_tables: dict[str, pd.DataFrame],
    train_df: pd.DataFrame,
    results_dir: Path | str,
    *,
    best_classical_name: str | None = None,
    transformer_model_name: str | None = None,
    qwen_predictions_df: pd.DataFrame | None = None,
    qwen_invalid_outputs_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Save and display-ready error analysis outputs."""
    output_dir = Path(results_dir) / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}

    best_completed = comparison_df.dropna(subset=["macro_f1"]).sort_values("macro_f1", ascending=False)
    outputs["best_model_name"] = best_completed.iloc[0]["model_name"] if not best_completed.empty else None

    if best_classical_name and best_classical_name in prediction_tables:
        classical_errors = prediction_tables[best_classical_name]
    else:
        classical_available = [name for name in ("logistic_regression", "linear_svm", "multinomial_nb") if name in prediction_tables]
        classical_errors = prediction_tables[classical_available[0]] if classical_available else pd.DataFrame()

    if not classical_errors.empty:
        classical_confusions = top_confusions(classical_errors)
        classical_misclassified = classical_errors[classical_errors["label_id"] != classical_errors["predicted_label_id"]].head(10)
        classical_confusions.to_csv(output_dir / "classical_top_confusions.csv", index=False)
        classical_misclassified.to_csv(output_dir / "classical_misclassified_examples.csv", index=False)
        outputs["classical_confusions"] = classical_confusions
        outputs["classical_misclassified"] = classical_misclassified

    if transformer_model_name and transformer_model_name in prediction_tables:
        transformer_errors = prediction_tables[transformer_model_name]
        transformer_confusions = top_confusions(transformer_errors)
        transformer_misclassified = transformer_errors[transformer_errors["label_id"] != transformer_errors["predicted_label_id"]].head(10)
        transformer_confusions.to_csv(output_dir / "transformer_top_confusions.csv", index=False)
        transformer_misclassified.to_csv(output_dir / "transformer_misclassified_examples.csv", index=False)
        outputs["transformer_confusions"] = transformer_confusions
        outputs["transformer_misclassified"] = transformer_misclassified

        if not classical_errors.empty:
            comparison_errors = pd.DataFrame(
                {
                    "text": classical_errors["text"],
                    "true_label": classical_errors["label"],
                    "classical_predicted": classical_errors["predicted_label"],
                    "transformer_predicted": transformer_errors["predicted_label"],
                }
            )
            comparison_errors["classical_correct"] = comparison_errors["true_label"] == comparison_errors["classical_predicted"]
            comparison_errors["transformer_correct"] = comparison_errors["true_label"] == comparison_errors["transformer_predicted"]
            comparison_errors.to_csv(output_dir / "classical_vs_transformer_errors.csv", index=False)
            outputs["classical_vs_transformer"] = comparison_errors

    if qwen_invalid_outputs_df is not None and not qwen_invalid_outputs_df.empty:
        qwen_invalid_outputs_df.to_csv(output_dir / "qwen_invalid_outputs.csv", index=False)
        outputs["qwen_invalid_outputs"] = qwen_invalid_outputs_df

    if qwen_predictions_df is not None and not qwen_predictions_df.empty:
        plausible_nonmatching = qwen_predictions_df[
            (qwen_predictions_df["predicted_label"] != "INVALID_PREDICTION")
            & (qwen_predictions_df["predicted_label"] != qwen_predictions_df["label"])
        ].head(10)
        plausible_nonmatching.to_csv(output_dir / "qwen_plausible_nonmatching_examples.csv", index=False)
        outputs["qwen_plausible_nonmatching"] = plausible_nonmatching

    imbalance = train_df["label"].value_counts().rename_axis("label").reset_index(name="train_count") if not train_df.empty else pd.DataFrame()
    imbalance.to_csv(output_dir / "class_imbalance.csv", index=False)
    outputs["class_imbalance"] = imbalance
    return outputs
