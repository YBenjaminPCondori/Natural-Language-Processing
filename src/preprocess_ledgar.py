"""Loading, preprocessing, and EDA helpers for LexGLUE LEDGAR."""

from __future__ import annotations

import json
import os
import re
import warnings
from collections import Counter
from numbers import Number
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from evaluate import write_json, write_jsonl


RANDOM_STATE = 42
TEXT_COLUMN_CANDIDATES = ("text", "provision", "clause", "contract_text")
LABEL_COLUMN_CANDIDATES = ("label", "labels", "category")
SOURCE_ID_CANDIDATES = ("source_id", "id", "document_id", "doc_id", "contract_id")
REQUIRED_SCHEMA = ["text", "label", "label_id", "source_dataset", "source_id", "split"]


def project_paths(project_root: Path | str = ".") -> dict[str, Path]:
    """Return the standard project paths used by the notebook."""
    root_input: Path | str = project_root
    if str(project_root) == "." and os.getenv("LEDGAR_PROJECT_ROOT"):
        root_input = os.environ["LEDGAR_PROJECT_ROOT"]
    root = Path(root_input).expanduser().resolve()
    return {
        "root": root,
        "raw_lexglue": root / "data" / "raw" / "lexglue_ledgar",
        "raw_original": root / "data" / "raw" / "original_ledgar",
        "processed": root / "data" / "processed",
        "outputs": root / "outputs",
        "figures": root / "outputs" / "figures",
        "predictions": root / "outputs" / "predictions",
        "models": root / "models",
        "models_classical": root / "models" / "classical",
        "models_transformers": root / "models" / "transformers",
        "models_trained": root / "models" / "trained",
        "models_trained_classical": root / "models" / "trained" / "classical",
        "models_trained_transformers": root / "models" / "trained" / "transformers",
        "checkpoints": root / "checkpoints",
        "checkpoints_classical": root / "checkpoints" / "classical",
        "checkpoints_transformers": root / "checkpoints" / "transformers",
        "src": root / "src",
        "notebooks": root / "notebooks",
        "reference_notebooks": root / "reference_notebooks",
    }


def ensure_project_dirs(project_root: Path | str = ".") -> dict[str, Path]:
    """Create all required project directories."""
    paths = project_paths(project_root)
    for key, path in paths.items():
        if key != "root":
            path.mkdir(parents=True, exist_ok=True)
    return paths


def normalize_whitespace(text: Any) -> str:
    """Normalise whitespace while preserving legal wording and punctuation."""
    return re.sub(r"\s+", " ", str(text)).strip()


def detect_column(columns: Sequence[str], candidates: Sequence[str], kind: str) -> str:
    """Find the first matching column from a list of candidates."""
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    raise ValueError(
        f"Could not detect a {kind} column. Available columns: {list(columns)}. "
        f"Checked candidates: {list(candidates)}"
    )


def get_label_names_from_features(features: Mapping[str, Any] | None, label_column: str) -> list[str] | None:
    """Extract ClassLabel names from Hugging Face feature metadata when available."""
    if not features or label_column not in features:
        return None
    label_feature = features[label_column]
    names = getattr(label_feature, "names", None)
    return list(names) if names else None


def load_ledgar_dataset(
    project_root: Path | str = ".",
    *,
    fallback_path: Path | str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load LexGLUE LEDGAR from Hugging Face, falling back only when necessary."""
    try:
        from datasets import Dataset, DatasetDict, load_dataset

        ds = load_dataset("coastalcph/lex_glue", "ledgar")
        metadata = {
            "dataset_source": "huggingface",
            "source_dataset": "LexGLUE_LEDGAR",
            "features": ds["train"].features,
            "load_error": None,
        }
        return ds, metadata
    except Exception as exc:
        print("WARNING: Hugging Face LEDGAR loading failed.")
        print(f"{type(exc).__name__}: {exc}")
        print("Attempting fallback loading from data/raw/original_ledgar/ledgar.jsonl")

    try:
        from datasets import Dataset, DatasetDict
    except Exception as dataset_exc:
        raise RuntimeError(
            "Fallback loading needs the datasets package to build a DatasetDict."
        ) from dataset_exc

    root = Path(project_root)
    fallback = Path(fallback_path) if fallback_path else root / "data" / "raw" / "original_ledgar" / "ledgar.jsonl"
    if not fallback.exists():
        raise FileNotFoundError(f"Fallback file not found: {fallback}")

    df = pd.read_json(fallback, lines=True)
    if "split" not in df.columns:
        raise ValueError(
            "Fallback file exists but has no 'split' column. Refusing to fabricate "
            "train/validation/test splits for coursework experiments."
        )

    split_frames = {
        split: split_df.drop(columns=["split"]).reset_index(drop=True)
        for split, split_df in df.groupby("split", sort=False)
    }
    required_splits = {"train", "validation", "test"}
    missing = required_splits.difference(split_frames)
    if missing:
        raise ValueError(f"Fallback file is missing required splits: {sorted(missing)}")

    ds = DatasetDict({split: Dataset.from_pandas(frame) for split, frame in split_frames.items()})
    metadata = {
        "dataset_source": "fallback_jsonl",
        "source_dataset": "Original_LEDGAR_Fallback",
        "features": ds["train"].features,
        "load_error": "Hugging Face loading failed; fallback JSONL was used.",
    }
    return ds, metadata


def dataset_overview(ds: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise splits, row counts, features, and example rows."""
    overview = {
        "splits": list(ds.keys()),
        "rows_per_split": {split: len(ds[split]) for split in ds.keys()},
        "features": str(ds["train"].features) if "train" in ds else {},
        "train_examples": [dict(ds["train"][i]) for i in range(min(3, len(ds["train"])))],
    }
    return overview


def export_raw_hf_splits(ds: Mapping[str, Any], raw_dir: Path | str) -> dict[str, Path]:
    """Export raw Hugging Face splits to JSONL only."""
    raw_path = Path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}
    for split in ("train", "validation", "test"):
        if split not in ds:
            raise ValueError(f"Missing expected split: {split}")
        output_path = raw_path / f"ledgar_{split}.jsonl"
        df = pd.DataFrame(ds[split])
        df.to_json(output_path, orient="records", lines=True, force_ascii=False)
        exported[split] = output_path
    return exported


def standardise_split(
    rows: Any,
    *,
    split: str,
    features: Mapping[str, Any] | None,
    source_dataset: str = "LexGLUE_LEDGAR",
) -> pd.DataFrame:
    """Standardise one split to the coursework schema."""
    df = pd.DataFrame(rows)
    text_column = detect_column(df.columns, TEXT_COLUMN_CANDIDATES, "text")
    label_column = detect_column(df.columns, LABEL_COLUMN_CANDIDATES, "label")
    source_id_column = next((c for c in SOURCE_ID_CANDIDATES if c in df.columns), None)
    label_names = get_label_names_from_features(features, label_column)

    records = []
    for row in df.to_dict(orient="records"):
        text = normalize_whitespace(row.get(text_column, ""))
        raw_label = row.get(label_column, "")

        if label_names is not None and isinstance(raw_label, Number) and not pd.isna(raw_label):
            original_label_id = int(raw_label)
            label = label_names[original_label_id]
        else:
            label = normalize_whitespace(raw_label)
            original_label_id = None

        if not text or not label:
            continue

        records.append(
            {
                "text": text,
                "label": label,
                "label_id": original_label_id if original_label_id is not None else -1,
                "source_dataset": source_dataset,
                "source_id": normalize_whitespace(row.get(source_id_column, "")) if source_id_column else "",
                "split": split,
            }
        )

    return pd.DataFrame(records, columns=REQUIRED_SCHEMA)


def standardise_splits(ds: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    """Standardise all official splits."""
    features = metadata.get("features")
    source_dataset = metadata.get("source_dataset", "LexGLUE_LEDGAR")
    return {
        split: standardise_split(
            ds[split],
            split=split,
            features=features,
            source_dataset=source_dataset,
        )
        for split in ("train", "validation", "test")
    }


def combined_label_counts(splits: Mapping[str, pd.DataFrame]) -> Counter[str]:
    """Count labels across all standardised splits."""
    counts: Counter[str] = Counter()
    for df in splits.values():
        counts.update(df["label"].tolist())
    return counts


def save_label_artifacts(splits: Mapping[str, pd.DataFrame], outputs_dir: Path | str) -> dict[str, Path]:
    """Save label name and count artifacts."""
    output_path = Path(outputs_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    counts = combined_label_counts(splits)
    labels = sorted(counts)

    label_names_path = output_path / "label_names.txt"
    label_names_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
    counts_path = write_json(output_path / "label_counts.json", dict(sorted(counts.items())))
    return {"label_names": label_names_path, "label_counts": counts_path}


def print_label_inspection(splits: Mapping[str, pd.DataFrame]) -> None:
    """Print row counts and top/bottom label frequencies."""
    counts = combined_label_counts(splits)
    print("Rows per split:")
    for split, df in splits.items():
        print(f"  {split}: {len(df)}")
    print(f"Total unique labels: {len(counts)}")
    print("\nTop 50 labels:")
    print(pd.DataFrame(counts.most_common(50), columns=["label", "count"]))
    print("\nBottom 50 labels:")
    bottom = sorted(counts.items(), key=lambda item: item[1])[:50]
    print(pd.DataFrame(bottom, columns=["label", "count"]))


def select_labels(
    train_df: pd.DataFrame,
    *,
    mode: str = "top_n",
    top_n: int = 20,
    manual_labels: Sequence[str] | None = None,
    min_examples_per_label: int = 30,
) -> tuple[list[str], list[str]]:
    """Select labels from training data only."""
    train_counts = Counter(train_df["label"].tolist())
    warnings_out: list[str] = []

    if mode == "all":
        selected = [label for label, _ in train_counts.most_common()]
    elif mode == "top_n":
        selected = [label for label, _ in train_counts.most_common(top_n)]
    elif mode == "manual":
        if not manual_labels:
            raise ValueError("manual_labels must be provided when mode='manual'.")
        lookup = {label.lower(): label for label in train_counts}
        selected = []
        for requested in manual_labels:
            match = lookup.get(requested.lower())
            if match is None:
                warnings_out.append(f"Manual label not found and skipped: {requested}")
            elif match not in selected:
                selected.append(match)
        if not selected:
            raise ValueError("None of the manual labels matched labels in the training split.")
    else:
        raise ValueError("mode must be one of: 'all', 'top_n', 'manual'.")

    for label in selected:
        if train_counts[label] < min_examples_per_label:
            warnings_out.append(
                f"Selected label has fewer than {min_examples_per_label} training examples: "
                f"{label} ({train_counts[label]})"
            )
    return selected, warnings_out


def filter_splits_to_labels(
    splits: Mapping[str, pd.DataFrame],
    selected_labels: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Filter train, validation, and test consistently."""
    selected_set = set(selected_labels)
    return {
        split: df[df["label"].isin(selected_set)].reset_index(drop=True)
        for split, df in splits.items()
    }


def create_label_mapping(selected_labels: Sequence[str]) -> tuple[dict[str, int], dict[int, str]]:
    """Create deterministic selected-label mappings."""
    label_to_id = {label: idx for idx, label in enumerate(selected_labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def apply_label_mapping(
    splits: Mapping[str, pd.DataFrame],
    label_to_id: Mapping[str, int],
) -> dict[str, pd.DataFrame]:
    """Apply selected-label IDs to all splits."""
    mapped: dict[str, pd.DataFrame] = {}
    for split, df in splits.items():
        mapped_df = df.copy()
        mapped_df["label_id"] = mapped_df["label"].map(label_to_id).astype(int)
        mapped[split] = mapped_df[REQUIRED_SCHEMA].reset_index(drop=True)
    return mapped


def save_selection_artifacts(
    selected_labels: Sequence[str],
    label_to_id: Mapping[str, int],
    id_to_label: Mapping[int, str],
    outputs_dir: Path | str,
) -> dict[str, Path]:
    """Save selected-label and mapping artifacts."""
    output_path = Path(outputs_dir)
    selected_path = write_json(output_path / "selected_labels.json", list(selected_labels))
    mapping_path = write_json(
        output_path / "label_mapping.json",
        {
            "label_to_id": dict(label_to_id),
            "id_to_label": {str(key): value for key, value in id_to_label.items()},
        },
    )
    return {"selected_labels": selected_path, "label_mapping": mapping_path}


def save_processed_splits(splits: Mapping[str, pd.DataFrame], processed_dir: Path | str) -> dict[str, Path]:
    """Save processed train/validation/test JSONL files."""
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}
    for split, df in splits.items():
        output_path = processed_path / f"ledgar_{split}.jsonl"
        df[REQUIRED_SCHEMA].to_json(output_path, orient="records", lines=True, force_ascii=False)
        saved[split] = output_path
    return saved


def print_filtered_summary(splits: Mapping[str, pd.DataFrame], selected_labels: Sequence[str]) -> None:
    """Print selected labels, row counts, and filtered distributions."""
    print("Selected labels:")
    for label in selected_labels:
        print(f"  - {label}")
    print("\nRows after filtering:")
    for split, df in splits.items():
        print(f"  {split}: {len(df)}")
    print("\nClass distribution after filtering:")
    for split, df in splits.items():
        print(f"\n{split}")
        print(df["label"].value_counts().rename_axis("label").reset_index(name="count"))


def create_eda_outputs(
    splits: Mapping[str, pd.DataFrame],
    outputs_dir: Path | str,
    figures_dir: Path | str,
    *,
    examples_per_class: int = 2,
) -> dict[str, Path]:
    """Create simple EDA tables, figures, and JSONL examples."""
    outputs = Path(outputs_dir)
    figures = Path(figures_dir)
    outputs.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(splits.values(), ignore_index=True)
    combined["word_count"] = combined["text"].str.split().str.len()
    combined["character_count"] = combined["text"].str.len()

    split_rows = {split: int(len(df)) for split, df in splits.items()}
    label_counts = combined["label"].value_counts()
    length_stats = combined[["word_count", "character_count"]].describe().to_dict()

    summary = {
        "rows_per_split": split_rows,
        "unique_labels": int(combined["label"].nunique()),
        "label_counts": label_counts.to_dict(),
        "word_count_stats": length_stats["word_count"],
        "character_count_stats": length_stats["character_count"],
    }
    summary_path = write_json(outputs / "dataset_summary.json", summary)

    fig, ax = plt.subplots(figsize=(12, 6))
    label_counts.sort_values(ascending=False).plot(kind="bar", ax=ax)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout()
    class_dist_path = figures / "class_distribution.png"
    fig.savefig(class_dist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    combined["word_count"].plot(kind="hist", bins=50, ax=ax)
    ax.set_title("Text Length Histogram")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    length_hist_path = figures / "text_length_histogram.png"
    fig.savefig(length_hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    example_rows = []
    for label in label_counts.index:
        label_examples = combined[combined["label"] == label].head(examples_per_class)
        for row in label_examples.to_dict(orient="records"):
            example_rows.append(
                {
                    "label": row["label"],
                    "label_id": int(row["label_id"]),
                    "split": row["split"],
                    "text": row["text"],
                }
            )
    examples_path = write_jsonl(outputs / "example_clauses.jsonl", example_rows)

    return {
        "dataset_summary": summary_path,
        "class_distribution": class_dist_path,
        "text_length_histogram": length_hist_path,
        "example_clauses": examples_path,
    }


def load_processed_splits(processed_dir: Path | str) -> dict[str, pd.DataFrame]:
    """Load processed JSONL splits into pandas DataFrames."""
    processed_path = Path(processed_dir)
    splits = {}
    for split in ("train", "validation", "test"):
        path = processed_path / f"ledgar_{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing processed split: {path}")
        splits[split] = pd.read_json(path, lines=True)
    return splits


def warn_messages(messages: Sequence[str]) -> None:
    """Print warnings clearly in notebooks."""
    for message in messages:
        warnings.warn(message)
