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

from .data_setup import write_json
from .evaluation import evaluate_predictions_common, sample_debug_frame, utc_now_iso, write_stage_status
from .preprocessing import legal_safe_tokenise, negation_aware_tokenise


TOKENIZERS = {
    "legal_safe": legal_safe_tokenise,
    "negation_aware": negation_aware_tokenise,
}


def classical_model_configs(
    max_features_list: list[int],
    ngram_ranges: list[tuple[int, int]],
    min_df_list: list[int],
) -> list[dict[str, Any]]:
    """Build the TF-IDF grid."""
    return [
        {"max_features": max_features, "ngram_range": ngram_range, "min_df": min_df}
        for max_features in max_features_list
        for ngram_range in ngram_ranges
        for min_df in min_df_list
    ]


def build_classical_pipeline(
    model_name: str,
    *,
    max_features: int,
    ngram_range: tuple[int, int],
    min_df: int = 1,
    c_value: float = 1.0,
    alpha: float = 1.0,
    class_weight: str | None = None,
    seed: int = 42,
    tokenizer_name: str = "negation_aware",
) -> Pipeline:
    """Build one classical sklearn pipeline."""
    if tokenizer_name not in TOKENIZERS:
        raise ValueError(f"Unsupported tokenizer_name: {tokenizer_name}")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        lowercase=False,
        stop_words=None,
        tokenizer=TOKENIZERS[tokenizer_name],
        token_pattern=None,
    )
    if model_name == "logistic_regression":
        classifier = LogisticRegression(max_iter=2000, C=c_value, class_weight=class_weight, random_state=seed, n_jobs=-1)
    elif model_name == "linear_svm":
        classifier = LinearSVC(C=c_value, class_weight=class_weight, random_state=seed)
    elif model_name == "multinomial_nb":
        classifier = MultinomialNB(alpha=alpha)
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
    min_df_list: list[int] | None = None,
    c_values: list[float] | None = None,
    nb_alpha_values: list[float] | None = None,
    dataset_name: str = "LEDGAR",
    seed: int = 42,
    run_naive_bayes: bool = True,
    tokenizer_name: str = "negation_aware",
    max_train_samples: int | None = None,
    max_validation_samples: int | None = None,
    max_eval_samples: int | None = None,
    smoke_test: bool = False,
) -> dict[str, Any]:
    """Run validation-selected classical model experiments."""
    output_dir = Path(results_dir) / "classical"
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_started = utc_now_iso()
    min_df_list = min_df_list or [1, 2, 5]
    c_values = c_values or [0.1, 1.0, 3.0, 10.0]
    nb_alpha_values = nb_alpha_values or [0.1, 0.5, 1.0]
    if smoke_test:
        max_train_samples = max_train_samples or 500
        max_validation_samples = max_validation_samples or 200
        max_eval_samples = max_eval_samples or 200
        max_features_list = [min(max_features_list[0], 2000)]
        ngram_ranges = [ngram_ranges[0]]
        min_df_list = [1]
        c_values = [1.0]
        nb_alpha_values = [1.0]
    config_payload = {
        "dataset_name": dataset_name,
        "seed": seed,
        "tokenizer_name": tokenizer_name,
        "max_features_list": max_features_list,
        "ngram_ranges": [list(value) for value in ngram_ranges],
        "min_df_list": min_df_list,
        "c_values": c_values,
        "nb_alpha_values": nb_alpha_values,
        "run_naive_bayes": run_naive_bayes,
        "max_train_samples": max_train_samples,
        "max_validation_samples": max_validation_samples,
        "max_eval_samples": max_eval_samples,
        "smoke_test": smoke_test,
        "train_rows_available": int(len(train_df)),
        "validation_rows_available": int(len(validation_df)),
        "test_rows_available": int(len(test_df)),
        "cuad_policy": "Classical models train and tune only on LEDGAR train/validation.",
    }
    write_json(output_dir / "classical_run_config.json", config_payload)
    for split_name, split_df in (("train", train_df), ("validation", validation_df)):
        if "source_dataset" in split_df.columns and split_df["source_dataset"].astype(str).eq("CUAD").any():
            reason = f"CUAD rows found in {split_name}; CUAD must remain external."
            write_stage_status(
                output_dir / "classical_stage_status.json",
                stage="classical_models",
                status="failed",
                error_type="ProtocolViolation",
                error_message=reason,
                config=config_payload,
                started_at_utc=stage_started,
            )
            raise ValueError(reason)
    if train_df.empty or validation_df.empty or test_df.empty:
        reason = "Classical models skipped because a required LEDGAR split is unavailable or empty."
        print(reason)
        pd.DataFrame().to_csv(output_dir / "classical_results.csv", index=False)
        pd.DataFrame().to_csv(output_dir / "classical_validation_grid.csv", index=False)
        write_stage_status(
            output_dir / "classical_stage_status.json",
            stage="classical_models",
            status="skipped",
            error_type="DataUnavailable",
            error_message=reason,
            config=config_payload,
            outputs={
                "results": str(output_dir / "classical_results.csv"),
                "validation_grid": str(output_dir / "classical_validation_grid.csv"),
            },
            started_at_utc=stage_started,
        )
        return {"results": [], "prediction_tables": {}, "best_model": None, "best_model_name": None, "validation_grid": pd.DataFrame()}

    train_df = sample_debug_frame(train_df, max_train_samples, seed=seed)
    validation_df = sample_debug_frame(validation_df, max_validation_samples, seed=seed)
    test_df = sample_debug_frame(test_df, max_eval_samples, seed=seed)
    config_payload.update(
        {
            "train_rows_used": int(len(train_df)),
            "validation_rows_used": int(len(validation_df)),
            "test_rows_used": int(len(test_df)),
        }
    )
    write_json(output_dir / "classical_run_config.json", config_payload)
    project_root = Path(results_dir).parent
    models_dir = project_root / "models" / "classical"
    trained_models_dir = project_root / "models" / "trained" / "classical"
    checkpoints_dir = project_root / "checkpoints" / "classical"
    for directory in (models_dir, trained_models_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)
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
        for config in classical_model_configs(max_features_list, ngram_ranges, min_df_list):
            regularisation_grid = c_values if model_name in {"logistic_regression", "linear_svm"} else [None]
            alpha_grid = nb_alpha_values if model_name == "multinomial_nb" else [None]
            for class_weight in class_weight_options:
                for c_value in regularisation_grid:
                    for alpha in alpha_grid:
                        pipeline = build_classical_pipeline(
                            model_name,
                            class_weight=class_weight,
                            seed=seed,
                            c_value=float(c_value) if c_value is not None else 1.0,
                            alpha=float(alpha) if alpha is not None else 1.0,
                            tokenizer_name=tokenizer_name,
                            **config,
                        )
                        pipeline.fit(x_train, y_train)
                        val_pred = pipeline.predict(x_val).astype(int).tolist()
                        macro_f1 = f1_score(y_val, val_pred, labels=sorted(id2label), average="macro", zero_division=0)
                        row = {
                            "model_name": model_name,
                            "validation_macro_f1": macro_f1,
                            "max_features": config["max_features"],
                            "ngram_range": str(config["ngram_range"]),
                            "min_df": config["min_df"],
                            "C": c_value,
                            "alpha": alpha,
                            "class_weight": class_weight,
                            "tokenizer": tokenizer_name,
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
        joblib.dump(best_for_model, models_dir / f"{model_name}.joblib")
        joblib.dump(best_for_model, trained_models_dir / f"{model_name}.joblib")
        joblib.dump(best_for_model, checkpoints_dir / f"{model_name}_best.joblib")

        if best_for_model_score > best_validation_macro_f1:
            best_model = best_for_model
            best_model_name = model_name
            best_validation_macro_f1 = best_for_model_score

    results_df = pd.DataFrame(results)
    validation_grid_df = pd.DataFrame(validation_rows)
    results_df.to_csv(output_dir / "classical_results.csv", index=False)
    validation_grid_df.to_csv(output_dir / "classical_validation_grid.csv", index=False)
    stage_status = "completed" if not results_df.empty else "failed"
    stage_error_type = "" if not results_df.empty else "NoCompletedModel"
    stage_error_message = "" if not results_df.empty else "No classical model produced test predictions."
    write_stage_status(
        output_dir / "classical_stage_status.json",
        stage="classical_models",
        status=stage_status,
        error_type=stage_error_type,
        error_message=stage_error_message,
        config=config_payload,
        outputs={
            "results": str(output_dir / "classical_results.csv"),
            "validation_grid": str(output_dir / "classical_validation_grid.csv"),
        },
        notes="Hyperparameters selected on LEDGAR validation; final metrics use LEDGAR test only.",
        started_at_utc=stage_started,
    )
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
