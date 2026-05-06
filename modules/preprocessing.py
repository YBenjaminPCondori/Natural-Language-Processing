"""LEDGAR preprocessing and EDA helpers."""

from __future__ import annotations

import html
import re
from collections import Counter, defaultdict
from numbers import Integral
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data_setup import ProjectPaths, normalise_whitespace, save_jsonl, write_json
from .evaluation import utc_now_iso, write_stage_status


REQUIRED_SCHEMA = ["text", "label", "label_id", "split", "source_dataset"]
SPLIT_ORDER = ("train", "validation", "test")
WORD_BOUNDARY_PATTERN = re.compile(r"\b\w+\b")
NEGATION_TOKENS = {"no", "not", "never"}


def clean_html_entities(text: Any) -> str:
    """Remove simple HTML markup, unescape entities, and normalise whitespace."""
    if text is None:
        return ""
    without_tags = re.sub(r"<[^>]+>", " ", str(text))
    return normalise_whitespace(html.unescape(without_tags))


def regex_tokenise(text: Any) -> list[str]:
    """Tokenise with the Week 2 lab word-boundary regex."""
    return WORD_BOUNDARY_PATTERN.findall(clean_html_entities(text))


def legal_safe_tokenise(text: Any) -> list[str]:
    """Lowercase word-boundary tokenisation without removing legal stopwords."""
    return [token.lower() for token in regex_tokenise(text)]


def _expand_negation_contractions(text: str) -> str:
    """Expand common contractions needed by the Week 2 negation tokenizer."""
    value = re.sub(r"n't\b", " not", text, flags=re.IGNORECASE)
    value = re.sub(r"'m\b", " am", value, flags=re.IGNORECASE)
    value = re.sub(r"'s\b", " is", value, flags=re.IGNORECASE)
    value = re.sub(r"'re\b", " are", value, flags=re.IGNORECASE)
    value = re.sub(r"'ve\b", " have", value, flags=re.IGNORECASE)
    value = re.sub(r"'ll\b", " will", value, flags=re.IGNORECASE)
    return value


def negation_aware_tokenise(text: Any) -> list[str]:
    """Attach NOT_ to the token immediately following a negation cue."""
    expanded = _expand_negation_contractions(clean_html_entities(text).lower())
    tokens = WORD_BOUNDARY_PATTERN.findall(expanded)
    result: list[str] = []
    negate_next = False
    for token in tokens:
        result.append(f"NOT_{token}" if negate_next else token)
        negate_next = token in NEGATION_TOKENS
    return result


def corpus_word_frequencies(texts: list[str] | pd.Series, *, max_words: int | None = None) -> dict[str, int]:
    """Build token frequency counts for BPE training from clause text."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(legal_safe_tokenise(text))
    items = counts.most_common(max_words)
    return dict(items if max_words is not None else counts.items())


def compute_bpe_pair_frequencies(vocab: dict[str, int]) -> dict[tuple[str, str], int]:
    """Count adjacent symbol pairs in a BPE vocabulary."""
    pairs: defaultdict[tuple[str, str], int] = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for left, right in zip(symbols, symbols[1:], strict=False):
            pairs[(left, right)] += freq
    return dict(pairs)


def merge_bpe_pair(pair: tuple[str, str], vocab: dict[str, int]) -> dict[str, int]:
    """Merge one BPE pair across the vocabulary."""
    pattern = re.compile(r"(?<!\S)" + re.escape(" ".join(pair)) + r"(?!\S)")
    return {pattern.sub("".join(pair), word): freq for word, freq in vocab.items()}


def train_bpe_tokeniser(word_frequencies: dict[str, int], *, num_merges: int = 50) -> tuple[dict[tuple[str, str], int], dict[str, int]]:
    """Learn simple BPE merge rules from word frequencies."""
    vocab = {" ".join(tuple(word) + ("</w>",)): freq for word, freq in word_frequencies.items() if word}
    merges: dict[tuple[str, str], int] = {}
    for merge_index in range(num_merges):
        pair_counts = compute_bpe_pair_frequencies(vocab)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=pair_counts.get)
        merges[best_pair] = merge_index
        vocab = merge_bpe_pair(best_pair, vocab)
    return merges, vocab


def bpe_encode_word(word: str, merges: dict[tuple[str, str], int]) -> list[str]:
    """Encode one word with learned BPE merges, falling back to characters."""
    symbols = list(word.lower()) + ["</w>"]
    if not symbols:
        return []
    ranked_merges = sorted(merges, key=merges.get)
    for merge_pair in ranked_merges:
        idx = 0
        while idx < len(symbols) - 1:
            if (symbols[idx], symbols[idx + 1]) == merge_pair:
                symbols = symbols[:idx] + ["".join(merge_pair)] + symbols[idx + 2 :]
            else:
                idx += 1
    return [symbol for symbol in symbols if symbol != "</w>"]


def bpe_encode_text(text: Any, merges: dict[tuple[str, str], int]) -> list[str]:
    """Encode all regex tokens in a clause with learned BPE merges."""
    encoded: list[str] = []
    for token in legal_safe_tokenise(text):
        encoded.extend(bpe_encode_word(token, merges))
    return encoded


def preprocessing_technique_rundown() -> str:
    """Return a readable summary of lecture/lab-grounded preprocessing choices."""
    return "\n".join(
        [
            "# Preprocessing and Feature Extraction Rundown",
            "",
            "## Preprocessing used",
            "- HTML/entity cleanup: Week 4 lab cleanup pattern, used before tokenisation.",
            "- Whitespace normalisation: keeps clauses readable while removing layout noise.",
            "- Regex tokenisation: Week 2 lab `\\b\\w+\\b` word-boundary tokenisation.",
            "- Legal-safe lowercasing: used inside tokenisers/vectorisers, without overwriting the stored clause text.",
            "- Negation-aware tokenisation: Week 2 lab idea, using `NOT_` on the token after `no`, `not`, or `never`.",
            "- BPE training/encoding: Weeks 2-3 lab implementation for subword/OOV inspection.",
            "",
            "## Feature extraction used",
            "- Bag-of-words inspection: Week 2/3 lab idea for understanding sparse features.",
            "- TF-IDF: Week 2 lecture and Week 3 lab term weighting.",
            "- Unigrams and bigrams: Week 2/3 lecture/lab n-gram feature extraction.",
            "",
            "## Deliberately not default preprocessing",
            "- Stopword removal: skipped because legal words such as `not`, `shall`, `may`, `unless`, and `except` carry meaning.",
            "- Logistic Regression, Linear SVM, and Naive Bayes: modelling stage, not preprocessing.",
            "- Co-occurrence vectors, LSA, and embeddings: representation/model-analysis material, not the default clause preprocessing path.",
        ]
    )


def write_preprocessing_rundown(output_path: Path | str) -> Path:
    """Write the readable preprocessing rundown artifact."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(preprocessing_technique_rundown() + "\n", encoding="utf-8")
    return path


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
        text = clean_html_entities(row.get(text_column))
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
    stage_started = utc_now_iso()
    status_path = paths.project_root / "outputs" / "logs" / "ledgar_preprocessing_status.json"
    stage_config = {
        "dataset_name": dataset_name,
        "top_k_labels": top_k_labels,
        "top_k_selected_from_split": "train",
        "expected_splits": list(SPLIT_ORDER),
        "raw_split_rows": {split: int(len(df)) for split, df in raw_splits.items()} if raw_splits else {},
        "cuad_policy": "CUAD is external only and is not accepted in LEDGAR preprocessing outputs.",
    }
    if raw_splits is None:
        reason = "LEDGAR preprocessing skipped because the raw dataset is unavailable."
        print(reason)
        write_stage_status(
            status_path,
            stage="ledgar_preprocessing",
            status="skipped",
            error_type="DatasetUnavailable",
            error_message=reason,
            config=stage_config,
            started_at_utc=stage_started,
        )
        return {}, {}, {}

    try:
        missing_splits = [split for split in SPLIT_ORDER if split not in raw_splits]
        if missing_splits:
            raise ValueError(f"Missing LEDGAR raw split(s): {missing_splits}")
        if top_k_labels <= 0:
            raise ValueError("top_k_labels must be positive.")

        label_names = load_ledgar_label_names(paths)
        standardised = {split: standardise_ledgar_split(raw_splits[split], split, label_names) for split in SPLIT_ORDER}
        train_counts = standardised["train"]["label"].value_counts()
        selected_labels = train_counts.head(top_k_labels).index.tolist()
        if not selected_labels:
            raise ValueError("No labels were selected from the LEDGAR training split.")
        label2id = {label: idx for idx, label in enumerate(selected_labels)}
        id2label = {idx: label for label, idx in label2id.items()}

        filtered_before_dedup = {}
        for split, df in standardised.items():
            filtered = df[df["label"].isin(selected_labels)].copy().reset_index(drop=True)
            filtered["label_id"] = filtered["label"].map(label2id).astype(int)
            filtered_before_dedup[split] = filtered[REQUIRED_SCHEMA]

        processed, duplicate_rows_removed = _remove_cross_split_duplicates(filtered_before_dedup)
        cuad_rows_by_split = {
            split: int(df["source_dataset"].astype(str).eq("CUAD").sum())
            for split, df in processed.items()
        }
        if any(cuad_rows_by_split.values()):
            raise ValueError(f"CUAD rows found in LEDGAR processed splits: {cuad_rows_by_split}")

        output_paths = {}
        for split in processed:
            output_paths[f"processed_{split}"] = str(save_jsonl(processed[split], paths.processed_data_dir / f"ledgar_{split}.jsonl"))

        label_names_path = paths.processed_data_dir / "label_names.txt"
        label_names_path.write_text("\n".join(selected_labels) + "\n", encoding="utf-8")
        output_paths["label_names"] = str(label_names_path)
        output_paths["label_counts"] = str(write_json(paths.processed_data_dir / "label_counts.json", train_counts.loc[selected_labels].astype(int).to_dict()))
        leakage_audit = {
            "dataset": dataset_name,
            "top_k_selected_from_split": "train",
            "split_priority": list(SPLIT_ORDER),
            "raw_rows": {split: int(len(df)) for split, df in raw_splits.items()},
            "standardised_rows": {split: int(len(df)) for split, df in standardised.items()},
            "filtered_rows_before_cross_split_deduplication": {
                split: int(len(df)) for split, df in filtered_before_dedup.items()
            },
            "duplicate_rows_removed_by_split": duplicate_rows_removed,
            "filtered_rows_after_cross_split_deduplication": {split: int(len(df)) for split, df in processed.items()},
            "cuad_rows_found_after_preprocessing": cuad_rows_by_split,
            "cross_split_overlaps_before_deduplication": _split_overlap_counts(filtered_before_dedup),
            "cross_split_overlaps_after_deduplication": _split_overlap_counts(processed),
        }
        output_paths["leakage_audit"] = str(write_json(paths.project_root / "outputs" / "leakage_audit.json", leakage_audit))
        dataset_summary = {
            "dataset": dataset_name,
            "top_k_selected_from_split": "train",
            "top_k_labels": top_k_labels,
            "rows_per_split": {split: int(len(df)) for split, df in processed.items()},
            "number_of_classes": len(selected_labels),
            "labels": selected_labels,
            "leakage_audit_path": "outputs/leakage_audit.json",
        }
        output_paths["dataset_summary"] = str(write_json(paths.processed_data_dir / "dataset_summary.json", dataset_summary))
        write_stage_status(
            status_path,
            stage="ledgar_preprocessing",
            status="completed",
            config={**stage_config, "selected_labels": selected_labels},
            outputs=output_paths,
            notes="LEDGAR split separation preserved; top-k labels selected from training split only.",
            started_at_utc=stage_started,
        )
        return processed, label2id, id2label
    except Exception as exc:
        write_stage_status(
            status_path,
            stage="ledgar_preprocessing",
            status="failed",
            error_type=type(exc).__name__,
            error_message=str(exc),
            config=stage_config,
            started_at_utc=stage_started,
        )
        raise


def create_ledgar_eda(processed: dict[str, pd.DataFrame], results_dir: Path) -> pd.DataFrame:
    """Create LEDGAR EDA plots, audits, feature-inspection files, and split summary."""
    if not processed:
        return pd.DataFrame()

    eda_dir = Path(results_dir) / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    combined = pd.concat(processed.values(), ignore_index=True)
    combined["text"] = combined["text"].astype(str)
    combined["label"] = combined["label"].astype(str)
    combined["word_count"] = combined["text"].str.split().str.len()
    label_counts = combined["label"].value_counts()

    # 1. Class distribution
    fig, ax = plt.subplots(figsize=(12, 6))
    label_counts.plot(kind="bar", ax=ax)
    ax.set_title("LEDGAR Class Distribution")
    ax.set_xlabel("Label")
    ax.set_ylabel("Examples")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(eda_dir / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Clause length histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    combined["word_count"].plot(kind="hist", bins=50, ax=ax)
    ax.set_title("LEDGAR Clause Length Histogram")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Examples")
    fig.tight_layout()
    fig.savefig(eda_dir / "clause_length_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. Missing / empty value audit
    missing_audit = pd.DataFrame(
        [
            {
                "column": column,
                "missing_values": int(combined[column].isna().sum()),
                "empty_strings": int(combined[column].astype(str).str.strip().eq("").sum()),
                "total_rows": int(len(combined)),
            }
            for column in ["text", "label", "split", "source_dataset"]
            if column in combined.columns
        ]
    )
    missing_audit.to_csv(eda_dir / "missing_value_audit.csv", index=False)

    # 4. Outlier clause report: shortest and longest clauses
    shortest = combined.sort_values("word_count", ascending=True).head(25).copy()
    shortest["outlier_type"] = "shortest"

    longest = combined.sort_values("word_count", ascending=False).head(25).copy()
    longest["outlier_type"] = "longest"

    outlier_clauses = pd.concat([shortest, longest], ignore_index=True)
    outlier_columns = ["outlier_type", "split", "label", "label_id", "word_count", "text"]
    outlier_clauses[[column for column in outlier_columns if column in outlier_clauses.columns]].to_csv(
        eda_dir / "outlier_clauses.csv",
        index=False,
    )

    # 5. Label-wise length statistics
    label_length_statistics = (
        combined.groupby("label")["word_count"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    label_length_statistics.to_csv(eda_dir / "label_length_statistics.csv", index=False)

    # 6. Top TF-IDF unigram/bigram terms per label
    label_documents = (
        combined.groupby("label")["text"]
        .apply(lambda texts: " ".join(texts.astype(str)))
        .reset_index()
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        tokenizer=legal_safe_tokenise,
        token_pattern=None,
        ngram_range=(1, 2),
        max_features=5000,
    )

    tfidf_matrix = vectorizer.fit_transform(label_documents["text"])
    feature_names = np.array(vectorizer.get_feature_names_out())

    top_term_records: list[dict[str, Any]] = []
    for label_index, label in enumerate(label_documents["label"]):
        row = tfidf_matrix[label_index].toarray().ravel()
        top_indices = row.argsort()[::-1][:20]

        for rank, feature_index in enumerate(top_indices, start=1):
            if row[feature_index] <= 0:
                continue
            top_term_records.append(
                {
                    "label": label,
                    "rank": rank,
                    "term": feature_names[feature_index],
                    "tfidf_score": float(row[feature_index]),
                }
            )

    top_tfidf_terms = pd.DataFrame(top_term_records)
    top_tfidf_terms.to_csv(eda_dir / "top_tfidf_terms_per_label.csv", index=False)

    # 7. Label similarity heatmap using TF-IDF label vectors
    similarity_matrix = cosine_similarity(tfidf_matrix)

    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(similarity_matrix, aspect="auto")
    ax.set_title("LEDGAR Label Similarity Heatmap using TF-IDF")
    ax.set_xticks(range(len(label_documents)))
    ax.set_yticks(range(len(label_documents)))
    ax.set_xticklabels(label_documents["label"], rotation=90, fontsize=7)
    ax.set_yticklabels(label_documents["label"], fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(eda_dir / "label_similarity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 8. Example clauses per label
    examples = []
    for label in label_counts.index:
        for row in combined[combined["label"] == label].head(3).to_dict(orient="records"):
            examples.append({"label": label, "split": row["split"], "text": row["text"]})
    save_jsonl(pd.DataFrame(examples), eda_dir / "examples_per_label.jsonl")

    # 9. Split summary
    split_summary = pd.DataFrame(
        [
            {
                "split": split,
                "rows": len(df),
                "classes": df["label"].nunique(),
            }
            for split, df in processed.items()
        ]
    )
    split_summary.to_csv(eda_dir / "dataset_split_summary.csv", index=False)

    return split_summary
