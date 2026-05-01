"""Classical TF-IDF model experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .evaluation import evaluate_predictions_common


def classical_model_configs(max_features_list: list[int], ngram_ranges: list[tuple[int, int]]) -> list[dict[str, Any]]:
    """Build the TF-IDF grid."""
    return [{"max_features": max_features, "ngram_range": ngram_range} for max_features in max_features_list for ngram_range in ngram_ranges]


def build_classical_pipeline(
    model_name: str,
    *,
    max_features: int,
    ngram_range: tuple[int, int],
    class_weight: str | None = None,
    seed: int = 42,
) -> Pipeline:
    """Build one classical sklearn pipeline."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, lowercase=True, stop_words=None)
    if model_name == "logistic_regression":
        classifier = LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=seed, n_jobs=-1)
    elif model_name == "linear_svm":
        classifier = LinearSVC(class_weight=class_weight, random_state=seed)
    elif model_name == "multinomial_nb":
        classifier = MultinomialNB()
    else:
        raise ValueError(model_name)
    return Pipeline([("vectorizer", vectorizer), ("classifier", classifier)])


def run_classical_experiments(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id2label: dict[int, str],
    results_dir: Path | str,
    *,
    max_features_list: list[int],
    ngram_ranges: list[tuple[int, int]],
    dataset_name: str = "LEDGAR",
    seed: int = 42,
    run_naive_bayes: bool = True,
) -> dict[str, Any]:
    """Run validation-selected classical model experiments."""
    if train_df.empty:
        print("Classical models skipped because LEDGAR data is unavailable.")
        return {"results": [], "prediction_tables": {}, "best_model": None, "best_model_name": None, "validation_grid": pd.DataFrame()}

    output_dir = Path(results_dir) / "classical"
    output_dir.mkdir(parents=True, exist_ok=True)
    x_train = train_df["text"].tolist()
    y_train = train_df["label_id"].astype(int).tolist()
    x_val = validation_df["text"].tolist()
    y_val = validation_df["label_id"].astype(int).tolist()
    x_test = test_df["text"].tolist()
    y_test = test_df["label_id"].astype(int).tolist()

    model_names = ["logistic_regression", "linear_svm"]
    if run_naive_bayes:
        model_names.append("multinomial_nb")

    results = []
    prediction_tables = {}
    validation_rows = []
    best_model = None
    best_model_name = None
    best_validation_macro_f1 = -1.0

    for model_name in model_names:
        class_weight_options = [None]
        if model_name in {"logistic_regression", "linear_svm"}:
            class_weight_options.append("balanced")

        best_for_model = None
        best_for_model_score = -1.0
        best_for_model_config = None
        for config in classical_model_configs(max_features_list, ngram_ranges):
            for class_weight in class_weight_options:
                pipeline = build_classical_pipeline(model_name, class_weight=class_weight, seed=seed, **config)
                pipeline.fit(x_train, y_train)
                val_pred = pipeline.predict(x_val).astype(int).tolist()
                macro_f1 = f1_score(y_val, val_pred, labels=sorted(id2label), average="macro", zero_division=0)
                row = {
                    "model_name": model_name,
                    "validation_macro_f1": macro_f1,
                    "max_features": config["max_features"],
                    "ngram_range": str(config["ngram_range"]),
                    "class_weight": class_weight,
                }
                validation_rows.append(row)
                if macro_f1 > best_for_model_score:
                    best_for_model = pipeline
                    best_for_model_score = macro_f1
                    best_for_model_config = row

        if best_for_model is None or best_for_model_config is None:
            continue
        test_pred = best_for_model.predict(x_test).astype(int).tolist()
        notes = f"Selected on validation macro-F1. Config={best_for_model_config}"
        result, pred_df = evaluate_predictions_common(
            model_family="classical",
            model_name=model_name,
            training_type="supervised",
            y_true=y_test,
            y_pred=test_pred,
            df=test_df,
            id2label=id2label,
            dataset_name=dataset_name,
            output_dir=output_dir,
            notes=notes,
        )
        result.update(best_for_model_config)
        results.append(result)
        prediction_tables[model_name] = pred_df

        if best_for_model_score > best_validation_macro_f1:
            best_model = best_for_model
            best_model_name = model_name
            best_validation_macro_f1 = best_for_model_score

    results_df = pd.DataFrame(results)
    validation_grid_df = pd.DataFrame(validation_rows)
    results_df.to_csv(output_dir / "classical_results.csv", index=False)
    validation_grid_df.to_csv(output_dir / "classical_validation_grid.csv", index=False)
    if best_model is not None:
        joblib.dump(best_model, output_dir / "best_classical_model.pkl")
        joblib.dump(best_model.named_steps["vectorizer"], output_dir / "vectorizer.pkl")

    if not results_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        results_df.plot(kind="bar", x="model_name", y="macro_f1", ax=ax, legend=False)
        ax.set_title("Classical Model Test Macro-F1")
        ax.set_xlabel("Model")
        ax.set_ylabel("Macro-F1")
        fig.tight_layout()
        fig.savefig(output_dir / "classical_model_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    return {
        "results": results,
        "prediction_tables": prediction_tables,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "validation_grid": validation_grid_df,
    }
