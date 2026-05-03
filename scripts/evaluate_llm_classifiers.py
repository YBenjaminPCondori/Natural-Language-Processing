"""Evaluate instruction-tuned LLMs as prompt-only LEDGAR classifiers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "modules").exists():
            return candidate
    raise RuntimeError("Could not locate project root containing pyproject.toml and modules/.")


def read_jsonl(path: Path):
    import pandas as pd

    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["saullm_7b", "qwen_small", "qwen_7b"])
    parser.add_argument("--max-examples-per-split", type=int, default=None)
    parser.add_argument("--tune-decoding-on-validation", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--quantization", choices=["none", "8bit", "4bit"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "ledgar-clause-classification"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "").strip() or None)
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()

    project_root = find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from modules.llm_evaluation import evaluate_instruction_tuned_llms

    processed_dir = project_root / "data" / "processed"
    train_df = read_jsonl(processed_dir / "ledgar_train.jsonl")
    validation_df = read_jsonl(processed_dir / "ledgar_validation.jsonl")
    test_df = read_jsonl(processed_dir / "ledgar_test.jsonl")
    label_names = [line.strip() for line in (processed_dir / "label_names.txt").read_text(encoding="utf-8").splitlines() if line.strip()]

    summary = evaluate_instruction_tuned_llms(
        train_df=train_df,
        validation_df=validation_df,
        test_df=test_df,
        label_names=label_names,
        output_dir=project_root / "outputs",
        figures_dir=project_root / "figures",
        model_keys=args.models,
        max_examples_per_split=args.max_examples_per_split,
        tune_decoding_on_validation=args.tune_decoding_on_validation,
        evaluate_test=args.evaluate_test,
        batch_size=args.batch_size,
        quantization=args.quantization,
        seed=args.seed,
        allow_cpu=args.allow_cpu,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )
    print(
        json.dumps(
            {
                "results_rows": len(summary["results"]),
                "failure_rows": len(summary["failures"]),
                "outputs": [
                    "outputs/llm_results.csv",
                    "outputs/llm_predictions.csv",
                    "outputs/llm_per_class_metrics.csv",
                    "outputs/llm_decoding_settings.json",
                    "outputs/llm_failed_models.csv",
                    "figures/llm_model_comparison.png",
                ],
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
