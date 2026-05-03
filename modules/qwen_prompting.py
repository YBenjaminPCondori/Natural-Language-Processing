"""Qwen2.5-Instruct prompting baseline utilities."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_setup import ensure_package, normalise_whitespace
from .data_setup import write_json
from .evaluation import evaluate_predictions_common


QWEN_PREDICTION_COLUMNS = [
    "mode",
    "text",
    "label",
    "label_id",
    "raw_output",
    "predicted_label",
    "predicted_label_id",
    "is_invalid",
    "retrieved_example_ids",
    "retrieved_example_labels",
]
QWEN_RESULT_COLUMNS = [
    "model_family",
    "model_name",
    "training_type",
    "dataset",
    "eval_split",
    "sample_size",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "invalid_prediction_rate",
    "status",
    "reason",
    "notes",
]


def build_allowed_labels_text(label_names: list[str]) -> str:
    """Format allowed labels for prompts."""
    return "\n".join(f"- {label}" for label in label_names)


def _normalise_label_for_match(value: str) -> str:
    """Normalise label text for strict parser matching."""
    return normalise_whitespace(value).strip(" `\"'").lower()


def parse_llm_label(output: str, labels: list[str]) -> str:
    """Parse one exact allowed label after whitespace/case normalisation."""
    cleaned = normalise_whitespace(output).splitlines()[0] if output else ""
    label_lookup = {_normalise_label_for_match(label): label for label in labels}
    matched = label_lookup.get(_normalise_label_for_match(cleaned))
    if matched:
        return matched
    return "INVALID_PREDICTION"


def _base_prompt(clause_text: str, label_names: list[str], examples: list[dict[str, Any]] | None = None) -> str:
    """Build the shared strict classification prompt."""
    examples_text = ""
    if examples:
        rendered = "\n\n".join(f"Clause: {example['text']}\nAnswer: {example['label']}" for example in examples)
        examples_text = f"\nExamples:\n{rendered}\n"
    return f"""You are a legal clause classification system.

Classify the clause into exactly one label.

Allowed labels:
{build_allowed_labels_text(label_names)}
{examples_text}
Clause:
{clause_text}

Rules:
- Choose exactly one label from the allowed labels.
- Return only the label.
- Do not explain.
- Do not use punctuation.
- Do not invent a new label.

Answer:"""


def make_zero_shot_prompt(clause_text: str, label_names: list[str]) -> str:
    """Build the zero-shot Qwen prompt."""
    return _base_prompt(clause_text, label_names)


def build_few_shot_examples(train_df: pd.DataFrame, label_names: list[str], examples_per_class: int = 1) -> list[dict[str, str]]:
    """Select few-shot demonstrations from the training split only."""
    examples = []
    for label in label_names:
        label_rows = train_df[train_df["label"] == label].head(examples_per_class)
        for row in label_rows.to_dict(orient="records"):
            examples.append({"text": row["text"], "label": row["label"]})
    return examples


def make_few_shot_prompt(clause_text: str, label_names: list[str], few_shot_examples: list[dict[str, str]]) -> str:
    """Build the few-shot Qwen prompt."""
    return _base_prompt(clause_text, label_names, few_shot_examples)


def build_retrieval_index(train_df: pd.DataFrame, *, max_features: int = 50000) -> tuple[TfidfVectorizer, Any]:
    """Fit a TF-IDF retrieval index on training examples only."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2)
    matrix = vectorizer.fit_transform(train_df["text"].astype(str).tolist())
    return vectorizer, matrix


def retrieve_few_shot_examples(
    clause_text: str,
    train_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    train_matrix: Any,
    *,
    k: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve nearest training examples for one clause."""
    if train_df.empty:
        return []
    query = vectorizer.transform([clause_text])
    scores = cosine_similarity(query, train_matrix).ravel()
    top_indices = scores.argsort()[::-1][:k]
    examples = []
    for idx in top_indices:
        row = train_df.iloc[int(idx)]
        examples.append(
            {
                "row_id": int(idx),
                "text": row["text"],
                "label": row["label"],
                "score": float(scores[int(idx)]),
            }
        )
    return examples


def make_retrieval_few_shot_prompt(clause_text: str, label_names: list[str], retrieved_examples: list[dict[str, Any]]) -> str:
    """Build a retrieval-augmented few-shot prompt."""
    return _base_prompt(clause_text, label_names, retrieved_examples)


def _qwen_generate_label(qwen_model: Any, qwen_tokenizer: Any, prompt: str, label_names: list[str]) -> tuple[str, str]:
    """Generate and parse one Qwen label."""
    import torch

    inputs = qwen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(qwen_model.device)
    with torch.no_grad():
        output_ids = qwen_model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False,
            pad_token_id=qwen_tokenizer.eos_token_id,
        )
    generated = qwen_tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    return generated, parse_llm_label(generated, label_names)


def _evaluate_qwen_predictions(
    mode: str,
    rows: list[dict[str, Any]],
    *,
    train_df: pd.DataFrame,
    label2id: dict[str, int],
    id2label: dict[int, str],
    dataset_name: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate one Qwen prompting mode."""
    df = pd.DataFrame(rows)
    valid_mask = df["predicted_label"] != "INVALID_PREDICTION"
    invalid_rate = 1.0 - float(valid_mask.mean()) if len(df) else 1.0
    fallback_label_id = int(train_df["label_id"].mode().iloc[0])
    y_true = df["label_id"].astype(int).tolist()
    y_pred = [int(label2id[pred]) if pred in label2id else fallback_label_id for pred in df["predicted_label"].tolist()]
    result, _ = evaluate_predictions_common(
        model_family="llm_prompting",
        model_name=f"qwen_{mode}",
        training_type=mode,
        y_true=y_true,
        y_pred=y_pred,
        df=df.rename(columns={"predicted_label": "model_output_label"}),
        id2label=id2label,
        dataset_name=dataset_name,
        output_dir=output_dir,
        notes=f"Qwen prompting baseline. Invalid prediction rate={invalid_rate:.4f}",
    )
    result["invalid_prediction_rate"] = invalid_rate
    return result


def _qwen_runtime_payload(torch_module: Any, output_dir: Path | str, *, reason: str | None = None) -> dict[str, Any]:
    cuda_available = bool(torch_module.cuda.is_available())
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": getattr(torch_module, "__version__", None),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch_module.cuda.device_count()),
        "cuda_version": getattr(torch_module.version, "cuda", None),
        "gpu_name": torch_module.cuda.get_device_name(0) if cuda_available else None,
    }
    if reason:
        payload["runtime_note"] = reason
    return payload


def _qwen_run_config(
    *,
    model_name: str,
    eval_sample_size: int,
    few_shot_examples_per_class: int,
    seed: int,
    label_names: list[str],
    status: str,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "model_name_configured": model_name,
        "eval_sample_size_configured": eval_sample_size,
        "few_shot_examples_per_class": few_shot_examples_per_class,
        "decoding": {"do_sample": False, "max_new_tokens": 20},
        "output_requirement": "Return exactly one allowed label.",
        "parser_policy": "Strict exact-label match after whitespace/case normalisation.",
        "seed": seed,
        "allowed_labels": label_names,
    }


def _write_qwen_skip_artifacts(output_dir: Path, skip_result: dict[str, Any], config: dict[str, Any], runtime: dict[str, Any], reason: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "qwen_run_config.json", config)
    write_json(output_dir / "runtime.json", runtime)
    pd.DataFrame([{**skip_result, "status": "skipped", "reason": reason}]).reindex(columns=QWEN_RESULT_COLUMNS).to_csv(
        output_dir / "qwen_results.csv",
        index=False,
    )
    pd.DataFrame(columns=QWEN_PREDICTION_COLUMNS).to_csv(output_dir / "qwen_predictions.csv", index=False)
    pd.DataFrame(columns=QWEN_PREDICTION_COLUMNS).to_csv(output_dir / "qwen_invalid_outputs.csv", index=False)


def run_qwen_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label2id: dict[str, int],
    id2label: dict[int, str],
    results_dir: Path | str,
    *,
    model_name: str,
    label_names: list[str],
    eval_sample_size: int = 200,
    few_shot_examples_per_class: int = 1,
    dataset_name: str = "LEDGAR",
    seed: int = 42,
    run_qwen: bool = True,
) -> dict[str, Any]:
    """Run Qwen zero-shot, static few-shot, and retrieval few-shot baselines."""
    if not run_qwen:
        print("Qwen baseline skipped because RUN_QWEN_BASELINE is False.")
        return {"results": [], "predictions": pd.DataFrame(), "invalid_outputs": pd.DataFrame(), "model": None, "tokenizer": None}
    if train_df.empty:
        print("Qwen baseline skipped because LEDGAR data is unavailable.")
        return {"results": [], "predictions": pd.DataFrame(), "invalid_outputs": pd.DataFrame(), "model": None, "tokenizer": None}

    ensure_package("torch", "torch")
    import torch
    output_dir = Path(results_dir) / "qwen"

    if not torch.cuda.is_available():
        print("Qwen baseline skipped because GPU/CUDA is unavailable. Loading a 3B model on CPU is not practical for this notebook.")
        reason = "GPU/CUDA unavailable; Qwen/Qwen2.5-3B-Instruct was not loaded or run."
        skip_result = {
            "model_family": "llm_prompting",
            "model_name": "qwen_skipped",
            "training_type": "zero/few-shot prompting",
            "dataset": dataset_name,
            "eval_split": "test",
            "sample_size": 0,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "invalid_prediction_rate": np.nan,
            "notes": f"Skipped: {reason}",
        }
        _write_qwen_skip_artifacts(
            output_dir,
            skip_result,
            _qwen_run_config(
                model_name=model_name,
                eval_sample_size=eval_sample_size,
                few_shot_examples_per_class=few_shot_examples_per_class,
                seed=seed,
                label_names=label_names,
                status="not_executed",
                reason=reason,
            ),
            _qwen_runtime_payload(torch, output_dir, reason=reason),
            reason,
        )
        return {"results": [skip_result], "predictions": pd.DataFrame(), "invalid_outputs": pd.DataFrame(), "model": None, "tokenizer": None}

    try:
        ensure_package("transformers", "transformers")
        ensure_package("accelerate", "accelerate")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(output_dir / "runtime.json", _qwen_runtime_payload(torch, output_dir))
        write_json(
            output_dir / "qwen_run_config.json",
            _qwen_run_config(
                model_name=model_name,
                eval_sample_size=eval_sample_size,
                few_shot_examples_per_class=few_shot_examples_per_class,
                seed=seed,
                label_names=label_names,
                status="running_or_completed",
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        model.eval()

        sample_size = min(eval_sample_size, len(test_df))
        sample = test_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        few_shot_examples = build_few_shot_examples(train_df, label_names, few_shot_examples_per_class)
        retrieval_vectorizer, retrieval_matrix = build_retrieval_index(train_df)
        all_rows = []
        prompt_examples = []
        results = []

        for mode in ("zero_shot", "static_few_shot", "retrieval_few_shot"):
            mode_rows = []
            for row in sample.to_dict(orient="records"):
                retrieved = []
                if mode == "zero_shot":
                    prompt = make_zero_shot_prompt(row["text"], label_names)
                elif mode == "static_few_shot":
                    prompt = make_few_shot_prompt(row["text"], label_names, few_shot_examples)
                else:
                    retrieved = retrieve_few_shot_examples(row["text"], train_df, retrieval_vectorizer, retrieval_matrix, k=3)
                    prompt = make_retrieval_few_shot_prompt(row["text"], label_names, retrieved)
                raw_output, parsed_label = _qwen_generate_label(model, tokenizer, prompt, label_names)
                mode_rows.append(
                    {
                        "mode": mode,
                        "text": row["text"],
                        "label": row["label"],
                        "label_id": int(row["label_id"]),
                        "raw_output": raw_output,
                        "predicted_label": parsed_label,
                        "predicted_label_id": int(label2id[parsed_label]) if parsed_label in label2id else pd.NA,
                        "is_invalid": parsed_label == "INVALID_PREDICTION",
                        "retrieved_example_ids": "|".join(str(example.get("row_id", "")) for example in retrieved),
                        "retrieved_example_labels": "|".join(str(example.get("label", "")) for example in retrieved),
                    }
                )
            all_rows.extend(mode_rows)
            results.append(
                _evaluate_qwen_predictions(
                    mode,
                    mode_rows,
                    train_df=train_df,
                    label2id=label2id,
                    id2label=id2label,
                    dataset_name=dataset_name,
                    output_dir=output_dir,
                )
            )
            if not sample.empty:
                example_text = sample.iloc[0]["text"]
                if mode == "zero_shot":
                    example_prompt = make_zero_shot_prompt(example_text, label_names)
                elif mode == "static_few_shot":
                    example_prompt = make_few_shot_prompt(example_text, label_names, few_shot_examples)
                else:
                    retrieved = retrieve_few_shot_examples(example_text, train_df, retrieval_vectorizer, retrieval_matrix, k=3)
                    example_prompt = make_retrieval_few_shot_prompt(example_text, label_names, retrieved)
                prompt_examples.append(f"--- {mode} prompt example ---\n{example_prompt}\n")

        predictions_df = pd.DataFrame(all_rows)
        invalid_outputs_df = predictions_df[predictions_df["predicted_label"] == "INVALID_PREDICTION"].copy()
        predictions_df.reindex(columns=QWEN_PREDICTION_COLUMNS).to_csv(output_dir / "qwen_predictions.csv", index=False)
        pd.DataFrame([{**result, "status": "completed"} for result in results]).reindex(columns=QWEN_RESULT_COLUMNS).to_csv(output_dir / "qwen_results.csv", index=False)
        (output_dir / "qwen_prompt_examples.txt").write_text("\n\n".join(prompt_examples), encoding="utf-8")
        invalid_outputs_df.reindex(columns=QWEN_PREDICTION_COLUMNS).to_csv(output_dir / "qwen_invalid_outputs.csv", index=False)
        return {"results": results, "predictions": predictions_df, "invalid_outputs": invalid_outputs_df, "model": model, "tokenizer": tokenizer}
    except Exception as exc:
        print(f"Qwen baseline could not load or run: {type(exc).__name__}: {exc}")
        reason = f"{type(exc).__name__}: {exc}"
        skip_result = {
            "model_family": "llm_prompting",
            "model_name": "qwen_skipped",
            "training_type": "zero/few-shot prompting",
            "dataset": dataset_name,
            "eval_split": "test",
            "sample_size": 0,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "invalid_prediction_rate": np.nan,
            "notes": f"Skipped/failed: {reason}",
        }
        _write_qwen_skip_artifacts(
            output_dir,
            skip_result,
            _qwen_run_config(
                model_name=model_name,
                eval_sample_size=eval_sample_size,
                few_shot_examples_per_class=few_shot_examples_per_class,
                seed=seed,
                label_names=label_names,
                status="failed",
                reason=reason,
            ),
            _qwen_runtime_payload(torch, output_dir, reason=reason),
            reason,
        )
        return {"results": [skip_result], "predictions": pd.DataFrame(), "invalid_outputs": pd.DataFrame(), "model": None, "tokenizer": None}
