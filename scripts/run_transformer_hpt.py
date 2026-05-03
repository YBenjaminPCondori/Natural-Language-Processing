"""Run two-stage HPT for the main transformer encoder on processed LEDGAR."""

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
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--random-trials", type=int, default=8)
    parser.add_argument("--bayes-trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=1)
    parser.add_argument("--no-final-retrain", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B runs when credentials are available.")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "ledgar-clause-classification"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY", "").strip() or None)
    parser.add_argument("--wandb-mode", default=os.environ.get("WANDB_MODE", "online"))
    args = parser.parse_args()

    project_root = find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from modules.transformer_hpt import TransformerHPTConfig, run_two_stage_transformer_hpt

    processed_dir = project_root / "data" / "processed"
    train_df = read_jsonl(processed_dir / "ledgar_train.jsonl")
    validation_df = read_jsonl(processed_dir / "ledgar_validation.jsonl")
    test_df = read_jsonl(processed_dir / "ledgar_test.jsonl")
    label_names = [line.strip() for line in (processed_dir / "label_names.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    id2label = {idx: label for idx, label in enumerate(label_names)}

    config = TransformerHPTConfig(
        model_name=args.model_name,
        random_trials=args.random_trials,
        bayes_trials=args.bayes_trials,
        seed=args.seed,
        early_stopping_patience=args.early_stopping_patience,
        final_retrain=not args.no_final_retrain,
    )
    output = run_two_stage_transformer_hpt(
        train_df,
        validation_df,
        test_df,
        id2label,
        project_root / "results",
        dataset_name="LEDGAR",
        config=config,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )
    printable = {
        "status": output["status"],
        "reason": output["reason"],
        "run_root": str(output["run_root"]),
        "best_config": output["best_config"],
    }
    print(json.dumps(printable, indent=2, default=str))


if __name__ == "__main__":
    main()
