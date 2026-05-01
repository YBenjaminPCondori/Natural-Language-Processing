"""Small human-in-the-loop clause review prototype."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def classifier_confidence(model: Any, texts: list[str]) -> tuple[list[int], list[float]]:
    """Return predictions and approximate confidence values for sklearn pipelines."""
    predictions = model.predict(texts).astype(int).tolist()
    classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)
        confidence = probabilities.max(axis=1).astype(float).tolist()
    elif classifier is not None and hasattr(classifier, "decision_function"):
        margins = classifier.decision_function(model.named_steps["vectorizer"].transform(texts))
        margins = np.atleast_2d(margins)
        sorted_scores = np.sort(margins, axis=1)
        raw_margin = sorted_scores[:, -1] - sorted_scores[:, -2] if sorted_scores.shape[1] > 1 else np.abs(sorted_scores[:, -1])
        confidence = (raw_margin / (1.0 + raw_margin)).astype(float).tolist()
    else:
        confidence = [np.nan] * len(texts)
    return predictions, confidence


def qwen_triage_explanation(qwen_model: Any, qwen_tokenizer: Any, clause_text: str, predicted_label: str, confidence: float) -> str:
    """Ask Qwen for a short triage explanation when available."""
    if qwen_model is None or qwen_tokenizer is None:
        return "Qwen explanation unavailable in this run."
    import torch

    prompt = f"""This output is for clause triage and research purposes only. It is not legal advice.
Predicted clause type: {predicted_label}
Classifier confidence: {confidence:.3f}

Clause:
{clause_text}

Write one short explanation with: predicted clause type, supporting phrase, uncertainty note, and no legal advice."""
    inputs = qwen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(qwen_model.device)
    with torch.no_grad():
        output_ids = qwen_model.generate(**inputs, max_new_tokens=96, do_sample=False, pad_token_id=qwen_tokenizer.eos_token_id)
    return qwen_tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()


def run_agentic_review(
    test_df: pd.DataFrame,
    id2label: dict[int, str],
    results_dir: Path | str,
    *,
    best_model: Any,
    qwen_model: Any = None,
    qwen_tokenizer: Any = None,
    run_agentic: bool = True,
    seed: int = 42,
    threshold: float = 0.55,
    sample_size: int = 20,
) -> pd.DataFrame:
    """Run the small low-confidence human-review demonstration."""
    if not run_agentic:
        print("Agentic review prototype skipped because RUN_AGENTIC_EXTENSION is False.")
        return pd.DataFrame()
    if best_model is None:
        print("Agentic review prototype skipped because no supervised classifier is available.")
        return pd.DataFrame()

    output_dir = Path(results_dir) / "agentic_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = test_df.sample(n=min(sample_size, len(test_df)), random_state=seed).reset_index(drop=True)
    pred_ids, confidences = classifier_confidence(best_model, sample["text"].tolist())
    rows = []
    for row, pred_id, confidence in zip(sample.to_dict(orient="records"), pred_ids, confidences):
        predicted_label = id2label[int(pred_id)]
        requires_review = bool(np.isnan(confidence) or confidence < threshold)
        explanation = ""
        if requires_review:
            explanation = qwen_triage_explanation(qwen_model, qwen_tokenizer, row["text"], predicted_label, confidence if not np.isnan(confidence) else 0.0)
        rows.append(
            {
                "text": row["text"],
                "true_label": row["label"],
                "predicted_label": predicted_label,
                "confidence": confidence,
                "requires_human_review": requires_review,
                "triage_note": "This output is for clause triage and research purposes only.",
                "optional_qwen_explanation": explanation,
            }
        )
    examples_df = pd.DataFrame(rows)
    examples_df.to_csv(output_dir / "agentic_examples.csv", index=False)
    return examples_df
