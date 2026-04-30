"""Baseline and classical model training helpers for LEDGAR."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from evaluate import (
    append_result_row,
    build_classification_report,
    build_confusion_matrix,
    evaluate_predictions,
    plot_confusion_matrix,
    read_jsonl,
    reset_jsonl,
    save_predictions_jsonl,
    write_json,
    write_jsonl,
)
from ledgar_pipeline.config import WandbConfig
from ledgar_pipeline.wandb_tracking import (
    finish_wandb_run,
    log_artifact_paths,
    log_result_row,
    log_table,
    start_wandb_run,
)


RANDOM_STATE = 42
CLASSICAL_RESULTS = "classical_results.jsonl"


def coerce_id_to_label(id_to_label: Mapping[int | str, str]) -> dict[int, str]:
    """Convert loaded JSON keys back to integer label IDs."""
    return {int(key): value for key, value in id_to_label.items()}


def labels_and_names(id_to_label: Mapping[int | str, str]) -> tuple[list[int], list[str]]:
    """Return label IDs and names sorted by ID."""
    mapping = coerce_id_to_label(id_to_label)
    labels = sorted(mapping)
    target_names = [mapping[label_id] for label_id in labels]
    return labels, target_names


def load_label_mapping(outputs_dir: Path | str) -> tuple[dict[str, int], dict[int, str]]:
    """Load label mappings saved during preprocessing."""
    path = Path(outputs_dir) / "label_mapping.json"
    with path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    return mapping["label_to_id"], coerce_id_to_label(mapping["id_to_label"])


def _result_with_paths(
    result: dict[str, Any],
    *,
    prediction_path: Path | str | None = None,
    model_path: Path | str | None = None,
    report_path: Path | str | None = None,
    figure_path: Path | str | None = None,
) -> dict[str, Any]:
    """Attach artifact paths to a result row."""
    result = dict(result)
    if prediction_path:
        result["prediction_path"] = str(Path(prediction_path))
    if model_path:
        result["model_path"] = str(Path(model_path))
    if report_path:
        result["classification_report_path"] = str(Path(report_path))
    if figure_path:
        result["confusion_matrix_path"] = str(Path(figure_path))
    return result


def evaluate_split(
    df: pd.DataFrame,
    predictions: Sequence[int],
    *,
    id_to_label: Mapping[int | str, str],
    model_name: str,
    split: str,
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate predictions for one split."""
    labels, target_names = labels_and_names(id_to_label)
    return evaluate_predictions(
        df["label_id"].astype(int).tolist(),
        [int(pred) for pred in predictions],
        labels=labels,
        target_names=target_names,
        model_name=model_name,
        split=split,
        params=params,
    )


def save_report_and_confusion_matrix(
    df: pd.DataFrame,
    predictions: Sequence[int],
    *,
    id_to_label: Mapping[int | str, str],
    report_path: Path | str,
    figure_path: Path | str,
    title: str,
) -> tuple[Path, Path]:
    """Save a JSON classification report and PNG confusion matrix."""
    labels, target_names = labels_and_names(id_to_label)
    y_true = df["label_id"].astype(int).tolist()
    y_pred = [int(pred) for pred in predictions]

    report = build_classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
    )
    saved_report = write_json(report_path, report)
    cm = build_confusion_matrix(y_true, y_pred, labels=labels)
    saved_figure = plot_confusion_matrix(cm, target_names, figure_path, title=title)
    return saved_report, saved_figure


def run_baselines(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    outputs_dir: Path | str,
    predictions_dir: Path | str,
    id_to_label: Mapping[int | str, str],
    reset_results: bool = False,
    wandb_config: WandbConfig | None = None,
) -> list[dict[str, Any]]:
    """Run random and majority baselines on validation and test splits."""
    outputs = Path(outputs_dir)
    predictions = Path(predictions_dir)
    results_path = outputs / CLASSICAL_RESULTS
    if reset_results:
        reset_jsonl(results_path)

    run = start_wandb_run(
        run_name="ledgar-baselines",
        job_type="baseline-training",
        config={
            "random_state": RANDOM_STATE,
            "models": ["random_baseline", "majority_baseline"],
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows": len(test_df),
        },
        tags=["coursework", "ledgar", "baseline"],
        wandb_config=wandb_config,
    )

    label_ids = sorted(coerce_id_to_label(id_to_label))
    train_counts = train_df["label_id"].astype(int).value_counts().sort_index()
    probabilities = np.array([train_counts.get(label_id, 0) for label_id in label_ids], dtype=float)
    probabilities = probabilities / probabilities.sum()

    rng = np.random.default_rng(RANDOM_STATE)
    majority_label = int(train_df["label_id"].astype(int).mode().iloc[0])

    baseline_specs = {
        "random_baseline": {
            "validation": rng.choice(label_ids, size=len(validation_df), p=probabilities).tolist(),
            "test": rng.choice(label_ids, size=len(test_df), p=probabilities).tolist(),
            "params": {"strategy": "sample_from_training_distribution", "random_state": RANDOM_STATE},
            "test_predictions": predictions / "random_baseline_test_predictions.jsonl",
        },
        "majority_baseline": {
            "validation": [majority_label] * len(validation_df),
            "test": [majority_label] * len(test_df),
            "params": {"strategy": "most_frequent_training_label"},
            "test_predictions": predictions / "majority_baseline_test_predictions.jsonl",
        },
    }

    try:
        result_rows: list[dict[str, Any]] = []
        prediction_paths: list[Path] = []
        for model_name, spec in baseline_specs.items():
            val_result = evaluate_split(
                validation_df,
                spec["validation"],
                id_to_label=id_to_label,
                model_name=model_name,
                split="validation",
                params=spec["params"],
            )
            append_result_row(results_path, val_result)
            log_result_row(run, val_result)
            result_rows.append(val_result)

            prediction_path = save_predictions_jsonl(
                spec["test_predictions"],
                test_df,
                spec["test"],
                id_to_label=coerce_id_to_label(id_to_label),
                model_name=model_name,
                split="test",
            )
            prediction_paths.append(Path(prediction_path))
            test_result = evaluate_split(
                test_df,
                spec["test"],
                id_to_label=id_to_label,
                model_name=model_name,
                split="test",
                params=spec["params"],
            )
            test_result = _result_with_paths(test_result, prediction_path=prediction_path)
            append_result_row(results_path, test_result)
            log_result_row(run, test_result)
            log_table(
                run,
                name=f"{model_name}_test_predictions",
                dataframe=pd.read_json(prediction_path, lines=True),
            )
            result_rows.append(test_result)

        log_artifact_paths(
            run,
            name="ledgar-baseline-results",
            artifact_type="results",
            paths=[results_path, *prediction_paths],
        )
        return result_rows
    finally:
        finish_wandb_run(run)


def classical_model_grid() -> dict[str, list[dict[str, Any]]]:
    """Return the required classical model search grid."""
    configs = [
        {"ngram_range": (1, 1), "max_features": 10000},
        {"ngram_range": (1, 1), "max_features": 30000},
        {"ngram_range": (1, 2), "max_features": 10000},
        {"ngram_range": (1, 2), "max_features": 30000},
    ]
    return {
        "logistic_regression": configs,
        "linear_svm": configs,
    }


def build_model_pipeline(model_name: str, *, ngram_range: tuple[int, int], max_features: int) -> Pipeline:
    """Build one sklearn text classification pipeline."""
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        lowercase=True,
    )
    if model_name == "logistic_regression":
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    elif model_name == "linear_svm":
        classifier = LinearSVC(
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return Pipeline([("tfidf", vectorizer), ("classifier", classifier)])


def train_and_select_classical_model(
    model_name: str,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    outputs_dir: Path | str,
    predictions_dir: Path | str,
    figures_dir: Path | str,
    models_dir: Path | str,
    checkpoints_dir: Path | str,
    id_to_label: Mapping[int | str, str],
    wandb_config: WandbConfig | None = None,
) -> dict[str, Any]:
    """Train all configs for one classical model, select by validation macro-F1, and test."""
    outputs = Path(outputs_dir)
    predictions = Path(predictions_dir)
    figures = Path(figures_dir)
    models = Path(models_dir)
    checkpoints = Path(checkpoints_dir)
    checkpoints.mkdir(parents=True, exist_ok=True)
    results_path = outputs / CLASSICAL_RESULTS
    checkpoint_results_path = checkpoints / f"{model_name}_validation_results.jsonl"
    reset_jsonl(checkpoint_results_path)
    best_checkpoint_path = checkpoints / f"{model_name}_best.joblib"

    run = start_wandb_run(
        run_name=f"ledgar-{model_name}",
        job_type="classical-training",
        config={
            "model_name": model_name,
            "random_state": RANDOM_STATE,
            "train_rows": len(train_df),
            "validation_rows": len(validation_df),
            "test_rows": len(test_df),
            "selection_metric": "validation_macro_f1",
        },
        tags=["coursework", "ledgar", "classical", model_name],
        wandb_config=wandb_config,
    )

    x_train = train_df["text"].tolist()
    y_train = train_df["label_id"].astype(int).tolist()
    x_val = validation_df["text"].tolist()

    best_pipeline: Pipeline | None = None
    best_config: dict[str, Any] | None = None
    best_result: dict[str, Any] | None = None
    best_macro_f1 = -1.0
    all_validation_results: list[dict[str, Any]] = []

    try:
        for config in classical_model_grid()[model_name]:
            pipeline = build_model_pipeline(model_name, **config)
            pipeline.fit(x_train, y_train)
            val_predictions = pipeline.predict(x_val).astype(int).tolist()

            serialisable_config = {
                "ngram_range": list(config["ngram_range"]),
                "max_features": config["max_features"],
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
            }
            val_result = evaluate_split(
                validation_df,
                val_predictions,
                id_to_label=id_to_label,
                model_name=model_name,
                split="validation",
                params=serialisable_config,
            )
            append_result_row(results_path, val_result)
            append_result_row(checkpoint_results_path, val_result)
            log_result_row(run, val_result)
            all_validation_results.append(val_result)

            macro_f1 = val_result["metrics"]["macro_f1"]
            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_pipeline = pipeline
                best_config = serialisable_config
                best_result = val_result
                joblib.dump(best_pipeline, best_checkpoint_path)

        if best_pipeline is None or best_config is None or best_result is None:
            raise RuntimeError(f"No model was trained for {model_name}.")

        test_predictions = best_pipeline.predict(test_df["text"].tolist()).astype(int).tolist()
        prediction_path = save_predictions_jsonl(
            predictions / f"{model_name}_test_predictions.jsonl",
            test_df,
            test_predictions,
            id_to_label=coerce_id_to_label(id_to_label),
            model_name=model_name,
            split="test",
        )

        model_path = models / f"{model_name}.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_pipeline, model_path)

        report_path = outputs / f"classification_report_{model_name}.json"
        figure_path = figures / f"confusion_matrix_{model_name}.png"
        saved_report, saved_figure = save_report_and_confusion_matrix(
            test_df,
            test_predictions,
            id_to_label=id_to_label,
            report_path=report_path,
            figure_path=figure_path,
            title=f"Confusion Matrix: {model_name}",
        )

        test_result = evaluate_split(
            test_df,
            test_predictions,
            id_to_label=id_to_label,
            model_name=model_name,
            split="test",
            params=best_config,
        )
        test_result = _result_with_paths(
            test_result,
            prediction_path=prediction_path,
            model_path=model_path,
            report_path=saved_report,
            figure_path=saved_figure,
        )
        test_result["checkpoint_path"] = str(best_checkpoint_path)
        append_result_row(results_path, test_result)
        log_result_row(run, test_result)
        log_table(
            run,
            name=f"{model_name}_validation_results",
            dataframe=pd.DataFrame(
                [
                    {
                        "model_name": row["model_name"],
                        "split": row["split"],
                        **row["metrics"],
                        **row["params"],
                    }
                    for row in all_validation_results
                ]
            ),
        )
        log_table(
            run,
            name=f"{model_name}_test_predictions",
            dataframe=pd.read_json(prediction_path, lines=True),
        )
        log_artifact_paths(
            run,
            name=f"ledgar-{model_name}-artifacts",
            artifact_type="model",
            paths=[
                model_path,
                best_checkpoint_path,
                checkpoint_results_path,
                prediction_path,
                saved_report,
                saved_figure,
            ],
            metadata={"best_validation_macro_f1": best_macro_f1},
        )

        return {
            "model_name": model_name,
            "best_validation_result": best_result,
            "test_result": test_result,
            "all_validation_results": all_validation_results,
        }
    finally:
        finish_wandb_run(run)


def run_classical_models(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    outputs_dir: Path | str,
    predictions_dir: Path | str,
    figures_dir: Path | str,
    models_dir: Path | str,
    checkpoints_dir: Path | str,
    id_to_label: Mapping[int | str, str],
    wandb_config: WandbConfig | None = None,
) -> dict[str, Any]:
    """Train and evaluate Logistic Regression and Linear SVM."""
    return {
        model_name: train_and_select_classical_model(
            model_name,
            train_df,
            validation_df,
            test_df,
            outputs_dir=outputs_dir,
            predictions_dir=predictions_dir,
            figures_dir=figures_dir,
            models_dir=models_dir,
            checkpoints_dir=checkpoints_dir,
            id_to_label=id_to_label,
            wandb_config=wandb_config,
        )
        for model_name in ("logistic_regression", "linear_svm")
    }


def load_result_rows(path: Path | str) -> list[dict[str, Any]]:
    """Load result rows if the file exists."""
    result_path = Path(path)
    if not result_path.exists():
        return []
    return read_jsonl(result_path)


def create_final_comparison(
    outputs_dir: Path | str,
    figures_dir: Path | str,
    *,
    wandb_config: WandbConfig | None = None,
) -> pd.DataFrame:
    """Combine available classical and transformer test results."""
    outputs = Path(outputs_dir)
    figures = Path(figures_dir)
    rows = []
    for filename in ("classical_results.jsonl", "transformer_results.jsonl"):
        path = outputs / filename
        for row in load_result_rows(path):
            if row.get("split") == "test":
                rows.append(row)

    if not rows:
        raise FileNotFoundError("No test result rows found for final comparison.")

    comparison_rows = []
    for row in rows:
        metrics = row.get("metrics", {})
        comparison_rows.append(
            {
                "model_name": row.get("model_name"),
                "split": row.get("split"),
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "macro_precision": metrics.get("macro_precision"),
                "macro_recall": metrics.get("macro_recall"),
                "prediction_path": row.get("prediction_path", ""),
                "model_path": row.get("model_path", ""),
            }
        )

    comparison = pd.DataFrame(comparison_rows).sort_values("macro_f1", ascending=False)
    write_jsonl(outputs / "final_model_comparison.jsonl", comparison.to_dict(orient="records"))

    figures.mkdir(parents=True, exist_ok=True)
    for metric, filename, title in [
        ("macro_f1", "final_macro_f1_comparison.png", "Test Macro-F1 Comparison"),
        ("weighted_f1", "final_weighted_f1_comparison.png", "Test Weighted-F1 Comparison"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        comparison.plot(kind="bar", x="model_name", y=metric, ax=ax, legend=False)
        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)

    run = start_wandb_run(
        run_name="ledgar-final-comparison",
        job_type="evaluation",
        config={"rows": len(comparison), "primary_metric": "test_macro_f1"},
        tags=["coursework", "ledgar", "comparison"],
        wandb_config=wandb_config,
    )
    try:
        log_table(run, name="final_model_comparison", dataframe=comparison)
        log_artifact_paths(
            run,
            name="ledgar-final-comparison",
            artifact_type="evaluation",
            paths=[
                outputs / "final_model_comparison.jsonl",
                figures / "final_macro_f1_comparison.png",
                figures / "final_weighted_f1_comparison.png",
            ],
        )
    finally:
        finish_wandb_run(run)

    return comparison


def create_error_analysis(
    outputs_dir: Path | str,
    figures_dir: Path | str,
    *,
    max_examples: int = 10,
    wandb_config: WandbConfig | None = None,
) -> dict[str, Any]:
    """Create evidence-only error analysis for the best available model."""
    outputs = Path(outputs_dir)
    figures = Path(figures_dir)
    comparison_path = outputs / "final_model_comparison.jsonl"
    if not comparison_path.exists():
        create_final_comparison(outputs, figures, wandb_config=wandb_config)

    comparison = pd.read_json(comparison_path, lines=True).sort_values("macro_f1", ascending=False)
    if comparison.empty:
        raise ValueError("Final comparison is empty.")

    best = comparison.iloc[0].to_dict()
    prediction_path = Path(best.get("prediction_path", ""))
    if not prediction_path.exists():
        raise FileNotFoundError(f"Best model prediction file not found: {prediction_path}")

    predictions = pd.read_json(prediction_path, lines=True)
    labels = sorted(
        set(predictions["true_label_id"].astype(int).tolist()).union(
            predictions["predicted_label_id"].astype(int).tolist()
        )
    )
    true_id_to_label = (
        predictions[["true_label_id", "true_label"]]
        .drop_duplicates()
        .set_index("true_label_id")["true_label"]
        .to_dict()
    )
    predicted_id_to_label = (
        predictions[["predicted_label_id", "predicted_label"]]
        .drop_duplicates()
        .set_index("predicted_label_id")["predicted_label"]
        .to_dict()
    )
    id_to_label = {int(key): value for key, value in predicted_id_to_label.items()}
    id_to_label.update({int(key): value for key, value in true_id_to_label.items()})
    target_names = [id_to_label[label_id] for label_id in labels]
    cm = build_confusion_matrix(
        predictions["true_label_id"].astype(int).tolist(),
        predictions["predicted_label_id"].astype(int).tolist(),
        labels=labels,
    )
    cm_path = plot_confusion_matrix(
        cm,
        target_names,
        figures / "best_model_confusion_matrix.png",
        title=f"Best Model Confusion Matrix: {best['model_name']}",
    )

    confused = predictions[predictions["true_label_id"] != predictions["predicted_label_id"]]
    top_confusions = (
        confused.groupby(["true_label", "predicted_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(20)
        .to_dict(orient="records")
    )

    correct_examples = (
        predictions[predictions["true_label_id"] == predictions["predicted_label_id"]]
        .head(max_examples)
        .to_dict(orient="records")
    )
    incorrect_examples = confused.head(max_examples).to_dict(orient="records")

    analysis = {
        "best_model": best,
        "confusion_matrix_path": str(cm_path),
        "top_confused_label_pairs": top_confusions,
        "correct_prediction_examples": correct_examples,
        "incorrect_prediction_examples": incorrect_examples,
    }
    write_json(outputs / "error_analysis.json", analysis)
    run = start_wandb_run(
        run_name="ledgar-error-analysis",
        job_type="error-analysis",
        config={"best_model": best["model_name"], "max_examples": max_examples},
        tags=["coursework", "ledgar", "error-analysis"],
        wandb_config=wandb_config,
    )
    try:
        log_table(run, name="top_confused_label_pairs", dataframe=pd.DataFrame(top_confusions))
        log_table(run, name="incorrect_prediction_examples", dataframe=pd.DataFrame(incorrect_examples))
        log_table(run, name="correct_prediction_examples", dataframe=pd.DataFrame(correct_examples))
        log_artifact_paths(
            run,
            name="ledgar-error-analysis",
            artifact_type="analysis",
            paths=[outputs / "error_analysis.json", cm_path],
        )
    finally:
        finish_wandb_run(run)
    return analysis
