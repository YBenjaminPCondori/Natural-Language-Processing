"""Run the BiLSTM neural sequence baseline on processed LEDGAR splits."""

from __future__ import annotations

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
    project_root = find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from modules.sequence_model import SequenceModelConfig, train_sequence_classifier

    processed_dir = project_root / "data" / "processed"
    train_df = read_jsonl(processed_dir / "ledgar_train.jsonl")
    validation_df = read_jsonl(processed_dir / "ledgar_validation.jsonl")
    test_df = read_jsonl(processed_dir / "ledgar_test.jsonl")
    label_names = [line.strip() for line in (processed_dir / "label_names.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    id2label = {idx: label for idx, label in enumerate(label_names)}

    output = train_sequence_classifier(
        train_df,
        validation_df,
        test_df,
        id2label,
        project_root / "results",
        dataset_name="LEDGAR",
        config=SequenceModelConfig(),
        run_sequence_model=True,
    )
    print(json.dumps({"status": "completed" if output.get("result") else "not_completed", "result": output.get("result")}, indent=2, default=str))


if __name__ == "__main__":
    main()
