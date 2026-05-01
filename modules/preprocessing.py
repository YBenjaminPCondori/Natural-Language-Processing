"""LEDGAR preprocessing and EDA helpers."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .data_setup import ProjectPaths, normalise_whitespace, save_jsonl, write_json


REQUIRED_SCHEMA = ["text", "label", "label_id", "split", "source_dataset"]


def load_ledgar_label_names(paths: ProjectPaths) -> list[str] | None:
    """Load LEDGAR label names from raw, processed, or legacy outputs."""
    candidates = [
        paths.ledgar_raw_dir / "label_names.txt",
        paths.processed_data_dir / "label_names.txt",
        paths.project_root / "outputs" / "label_names.txt",
    ]
    for path in candidates:
        if path.exists():
            labels = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if labels:
                return labels
    return None


def detect_text_column(df: pd.DataFrame) -> str:
    """Detect the clause text column."""
    for column in ("text", "provision", "clause", "contract_text"):
        if column in df.columns:
            return column
    raise ValueError(f"Could not detect text column. Columns: {list(df.columns)}")


def detect_label_column(df: pd.DataFrame) -> str:
    """Detect the label column."""
    for column in ("label", "labels", "category"):
        if column in df.columns:
            return column
    raise ValueError(f"Could not detect label column. Columns: {list(df.columns)}")


def standardise_ledgar_split(df: pd.DataFrame, split: str, label_names: list[str] | None) -> pd.DataFrame:
    """Standardise one LEDGAR split into the coursework schema."""
    text_column = detect_text_column(df)
    label_column = detect_label_column(df)
    records = []

    for row in df.to_dict(orient="records"):
        text = normalise_whitespace(row.get(text_column))
        raw_label = row.get(label_column)
        if label_names is not None and isinstance(raw_label, Integral) and 0 <= int(raw_label) < len(label_names):
            label = label_names[int(raw_label)]
        else:
            label = normalise_whitespace(raw_label)
        if text and label:
            records.append({"text": text, "label": label, "label_id": -1, "split": split, "source_dataset": "LEDGAR"})

    standardised = pd.DataFrame(records, columns=REQUIRED_SCHEMA)
    return standardised.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)


def preprocess_ledgar(
    raw_splits: dict[str, pd.DataFrame] | None,
    paths: ProjectPaths,
    *,
    top_k_labels: int = 20,
    dataset_name: str = "LEDGAR",
) -> tuple[dict[str, pd.DataFrame], dict[str, int], dict[int, str]]:
    """Preprocess LEDGAR and save processed JSONL/metadata files."""
    if raw_splits is None:
        print("LEDGAR preprocessing skipped because the raw dataset is unavailable.")
        return {}, {}, {}

    label_names = load_ledgar_label_names(paths)
    standardised = {split: standardise_ledgar_split(df, split, label_names) for split, df in raw_splits.items()}
    train_counts = standardised["train"]["label"].value_counts()
    selected_labels = train_counts.head(top_k_labels).index.tolist()
    label2id = {label: idx for idx, label in enumerate(selected_labels)}
    id2label = {idx: label for label, idx in label2id.items()}

    processed = {}
    for split, df in standardised.items():
        filtered = df[df["label"].isin(selected_labels)].copy().reset_index(drop=True)
        filtered["label_id"] = filtered["label"].map(label2id).astype(int)
        processed[split] = filtered[REQUIRED_SCHEMA]
        save_jsonl(processed[split], paths.processed_data_dir / f"ledgar_{split}.jsonl")

    (paths.processed_data_dir / "label_names.txt").write_text("\n".join(selected_labels) + "\n", encoding="utf-8")
    write_json(paths.processed_data_dir / "label_counts.json", train_counts.loc[selected_labels].astype(int).to_dict())
    write_json(
        paths.processed_data_dir / "dataset_summary.json",
        {
            "dataset": dataset_name,
            "top_k_labels": top_k_labels,
            "rows_per_split": {split: int(len(df)) for split, df in processed.items()},
            "number_of_classes": len(selected_labels),
            "labels": selected_labels,
        },
    )
    return processed, label2id, id2label


def create_ledgar_eda(processed: dict[str, pd.DataFrame], results_dir: Path) -> pd.DataFrame:
    """Create LEDGAR EDA plots, examples, and split summary."""
    if not processed:
        return pd.DataFrame()

    eda_dir = Path(results_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(processed.values(), ignore_index=True)
    combined["word_count"] = combined["text"].str.split().str.len()
    label_counts = combined["label"].value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    label_counts.plot(kind="bar", ax=ax)
    ax.set_title("LEDGAR Class Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Examples")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(eda_dir / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    combined["word_count"].plot(kind="hist", bins=50, ax=ax)
    ax.set_title("LEDGAR Clause Length Histogram")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Examples")
    fig.tight_layout()
    fig.savefig(eda_dir / "clause_length_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    examples = []
    for label in label_counts.index:
        for row in combined[combined["label"] == label].head(3).to_dict(orient="records"):
            examples.append({"label": label, "split": row["split"], "text": row["text"]})
    save_jsonl(pd.DataFrame(examples), eda_dir / "examples_per_label.jsonl")

    split_summary = pd.DataFrame(
        [{"split": split, "rows": len(df), "classes": df["label"].nunique()} for split, df in processed.items()]
    )
    split_summary.to_csv(eda_dir / "dataset_split_summary.csv", index=False)
    return split_summary
