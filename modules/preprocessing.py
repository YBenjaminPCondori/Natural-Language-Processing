"""LEDGAR preprocessing and EDA helpers."""

from __future__ import annotations

from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .data_setup import ProjectPaths, normalise_whitespace, save_jsonl, write_json


REQUIRED_SCHEMA = ["text", "label", "label_id", "split", "source_dataset"]
SPLIT_ORDER = ("train", "validation", "test")


def load_ledgar_label_names(paths: ProjectPaths) -> list[str] | None:
    """Load the raw LEDGAR label mapping used to decode integer labels."""
    path = paths.ledgar_raw_dir / "label_names.txt"
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
        if isinstance(raw_label, Integral):
            if label_names is None:
                raise ValueError(
                    "LEDGAR raw labels are integer ids, but data/raw/lexglue_ledgar/label_names.txt "
                    "is missing. Recreate the raw 100-label mapping before preprocessing."
                )
            if not 0 <= int(raw_label) < len(label_names):
                raise ValueError(f"Raw LEDGAR label id {raw_label!r} is outside the label mapping range.")
            label = label_names[int(raw_label)]
        else:
            label = normalise_whitespace(raw_label)
        if text and label:
            records.append({"text": text, "label": label, "label_id": -1, "split": split, "source_dataset": "LEDGAR"})

    standardised = pd.DataFrame(records, columns=REQUIRED_SCHEMA)
    return standardised.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)


def _split_overlap_counts(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, int]]:
    """Count exact text and text-label overlaps for each split pair."""
    overlaps: dict[str, dict[str, int]] = {}
    for idx, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[idx + 1 :]:
            if left not in splits or right not in splits:
                continue
            left_df = splits[left]
            right_df = splits[right]
            left_text = set(left_df["text"].astype(str))
            right_text = set(right_df["text"].astype(str))
            left_pair = set(zip(left_df["text"].astype(str), left_df["label"].astype(str), strict=False))
            right_pair = set(zip(right_df["text"].astype(str), right_df["label"].astype(str), strict=False))
            overlaps[f"{left}_vs_{right}"] = {
                "text_overlap": len(left_text & right_text),
                "text_label_overlap": len(left_pair & right_pair),
            }
    return overlaps


def _remove_cross_split_duplicates(filtered_splits: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    """Remove exact text-label duplicates from later splits, keeping train first."""
    seen_pairs: set[tuple[str, str]] = set()
    deduped: dict[str, pd.DataFrame] = {}
    removed: dict[str, int] = {}

    for split in SPLIT_ORDER:
        df = filtered_splits.get(split, pd.DataFrame(columns=REQUIRED_SCHEMA)).copy().reset_index(drop=True)
        keep_mask = []
        for text, label in zip(df["text"].astype(str), df["label"].astype(str), strict=False):
            pair = (text, label)
            keep = pair not in seen_pairs
            keep_mask.append(keep)
            if keep:
                seen_pairs.add(pair)
        deduped_df = df.loc[keep_mask].reset_index(drop=True)
        deduped[split] = deduped_df
        removed[split] = int(len(df) - len(deduped_df))
    return deduped, removed


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

    filtered_before_dedup = {}
    for split, df in standardised.items():
        filtered = df[df["label"].isin(selected_labels)].copy().reset_index(drop=True)
        filtered["label_id"] = filtered["label"].map(label2id).astype(int)
        filtered_before_dedup[split] = filtered[REQUIRED_SCHEMA]

    processed, duplicate_rows_removed = _remove_cross_split_duplicates(filtered_before_dedup)
    for split in processed:
        save_jsonl(processed[split], paths.processed_data_dir / f"ledgar_{split}.jsonl")

    (paths.processed_data_dir / "label_names.txt").write_text("\n".join(selected_labels) + "\n", encoding="utf-8")
    write_json(paths.processed_data_dir / "label_counts.json", train_counts.loc[selected_labels].astype(int).to_dict())
    leakage_audit = {
        "dataset": dataset_name,
        "split_priority": list(SPLIT_ORDER),
        "raw_rows": {split: int(len(df)) for split, df in raw_splits.items()},
        "standardised_rows": {split: int(len(df)) for split, df in standardised.items()},
        "filtered_rows_before_cross_split_deduplication": {
            split: int(len(df)) for split, df in filtered_before_dedup.items()
        },
        "duplicate_rows_removed_by_split": duplicate_rows_removed,
        "filtered_rows_after_cross_split_deduplication": {split: int(len(df)) for split, df in processed.items()},
        "cross_split_overlaps_before_deduplication": _split_overlap_counts(filtered_before_dedup),
        "cross_split_overlaps_after_deduplication": _split_overlap_counts(processed),
    }
    write_json(paths.project_root / "outputs" / "leakage_audit.json", leakage_audit)
    write_json(
        paths.processed_data_dir / "dataset_summary.json",
        {
            "dataset": dataset_name,
            "top_k_labels": top_k_labels,
            "rows_per_split": {split: int(len(df)) for split, df in processed.items()},
            "number_of_classes": len(selected_labels),
            "labels": selected_labels,
            "leakage_audit_path": "outputs/leakage_audit.json",
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
