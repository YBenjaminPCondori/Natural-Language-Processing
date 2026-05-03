"""Run one transformer encoder fine-tuning job on processed LEDGAR splits."""

from __future__ import annotations

import argparse
import json
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from modules.transformer_model import train_transformer_classifier

    processed_dir = project_root / "data" / "processed"
    train_df = read_jsonl(processed_dir / "ledgar_train.jsonl")
    validation_df = read_jsonl(processed_dir / "ledgar_validation.jsonl")
    test_df = read_jsonl(processed_dir / "ledgar_test.jsonl")
    label_names = [line.strip() for line in (processed_dir / "label_names.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    id2label = {idx: label for idx, label in enumerate(label_names)}

    output = train_transformer_classifier(
        train_df,
        validation_df,
        test_df,
        id2label,
        project_root / "results",
        model_name=args.model_name,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        batch_size_override=args.batch_size,
        dataset_name="LEDGAR",
        seed=args.seed,
        run_transformer=True,
    )
    print(json.dumps({"status": "completed" if output.get("result") else "not_completed", "result": output.get("result"), "skip_result": output.get("skip_result")}, indent=2, default=str))


if __name__ == "__main__":
    main()
