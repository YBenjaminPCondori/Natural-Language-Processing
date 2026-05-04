"""Run Qwen prompting baselines on processed LEDGAR splits."""

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
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--eval-sample-size", type=int, default=200)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--few-shot-examples-per-class", type=int, default=1)
    parser.add_argument("--smoke-test", action="store_true", help="Run only a tiny prompt sample if CUDA is available.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from modules.qwen_prompting import run_qwen_baseline

    processed_dir = project_root / "data" / "processed"
    train_df = read_jsonl(processed_dir / "ledgar_train.jsonl")
    test_df = read_jsonl(processed_dir / "ledgar_test.jsonl")
    label_names = [line.strip() for line in (processed_dir / "label_names.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}

    output = run_qwen_baseline(
        train_df,
        test_df,
        label2id,
        id2label,
        project_root / "results",
        model_name=args.model_name,
        label_names=label_names,
        eval_sample_size=args.eval_sample_size,
        few_shot_examples_per_class=args.few_shot_examples_per_class,
        dataset_name="LEDGAR",
        seed=args.seed,
        run_qwen=True,
        max_eval_samples=args.max_eval_samples,
        smoke_test=args.smoke_test,
    )
    print(json.dumps({"results": output.get("results", [])}, indent=2, default=str))


if __name__ == "__main__":
    main()
