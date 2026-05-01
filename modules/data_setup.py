"""Dataset download, raw-file loading, and shared setup helpers."""

from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProjectPaths:
    """Project paths used by the coursework notebook."""

    project_root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    ledgar_raw_dir: Path
    legacy_ledgar_raw_dir: Path
    cuad_raw_dir: Path
    results_dir: Path
    figures_dir: Path

    def ensure_dirs(self) -> None:
        for path in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.ledgar_raw_dir,
            self.cuad_raw_dir,
            self.results_dir,
            self.figures_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def ensure_package(import_name: str, pip_name: str | None = None) -> None:
    """Install a missing notebook dependency."""
    if importlib.util.find_spec(import_name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip_name or import_name])


def running_in_colab() -> bool:
    """Return whether this code is running inside Google Colab."""
    return "COLAB_RELEASE_TAG" in os.environ or importlib.util.find_spec("google.colab") is not None


def looks_like_project_root(path: Path | str) -> bool:
    """Return whether a directory looks like this coursework project root."""
    candidate = Path(path).expanduser()
    return (candidate / "pyproject.toml").exists() and (candidate / "modules").is_dir()


def colab_project_candidates() -> list[Path]:
    """Common Google Colab locations for this repository."""
    candidates = [
        Path("/content/Natural-Language-Processing"),
        Path("/content/DLIA/Natural-Language-Processing"),
        Path("/content/drive/MyDrive/Natural-Language-Processing"),
        Path("/content/drive/MyDrive/DLIA/Natural-Language-Processing"),
        Path("/content/drive/MyDrive/NLP/Natural-Language-Processing"),
        Path("/content/drive/MyDrive/Education/NLP/Natural-Language-Processing"),
        Path("/content/drive/MyDrive/GitHub/Education/NLP/Natural-Language-Processing"),
        Path("/content/drive/MyDrive/Colab Notebooks/Natural-Language-Processing"),
    ]
    for base in (Path("/content"), Path("/content/drive/MyDrive")):
        if base.exists():
            for pattern in (
                "Natural-Language-Processing",
                "*/Natural-Language-Processing",
                "*/*/Natural-Language-Processing",
                "*/*/*/Natural-Language-Processing",
            ):
                candidates.extend(base.glob(pattern))
    return candidates


def find_project_root(start: Path | str = ".", override: Path | str | None = None) -> Path:
    """Find the project root from a local shell, notebook, or Google Colab runtime."""
    env_override = os.getenv("LEDGAR_PROJECT_ROOT") or os.getenv("PROJECT_ROOT_OVERRIDE")
    selected_override = override or env_override
    if selected_override:
        override_path = Path(selected_override).expanduser().resolve()
        if looks_like_project_root(override_path):
            return override_path
        raise FileNotFoundError(
            f"Configured project root does not contain pyproject.toml and modules/: {override_path}"
        )

    start_path = Path(start).expanduser().resolve()
    for candidate in [start_path, *start_path.parents]:
        if looks_like_project_root(candidate):
            return candidate

    if running_in_colab():
        for candidate in colab_project_candidates():
            if candidate.exists() and looks_like_project_root(candidate):
                return candidate.resolve()
        raise FileNotFoundError(
            "Could not find the project root in Colab. Upload or clone the whole "
            "repository, then set LEDGAR_PROJECT_ROOT or PROJECT_ROOT_OVERRIDE to "
            "the folder containing pyproject.toml and modules/."
        )

    return start_path


def build_project_paths(project_root: Path | str = ".", override: Path | str | None = None) -> ProjectPaths:
    """Create the canonical path object for the coursework project."""
    root = find_project_root(project_root, override=override)
    paths = ProjectPaths(
        project_root=root,
        raw_data_dir=root / "data" / "raw",
        processed_data_dir=root / "data" / "processed",
        ledgar_raw_dir=root / "data" / "raw" / "lexglue_ledgar",
        legacy_ledgar_raw_dir=root / "data" / "raw" / "ledgar",
        cuad_raw_dir=root / "data" / "raw" / "cuad",
        results_dir=root / "results",
        figures_dir=root / "results" / "figures",
    )
    paths.ensure_dirs()
    return paths


def seed_everything(seed: int = 42) -> str:
    """Set random seeds and return the selected device name."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            return "cuda"
    except Exception:
        pass
    return "cpu"


def json_safe(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-serialisable objects."""
    if isinstance(value, dict):
        return {str(key): json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except Exception:
        pass
    return value


def write_json(path: Path | str, payload: Any) -> Path:
    """Write JSON with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def save_jsonl(df: pd.DataFrame, path: Path | str) -> Path:
    """Write a DataFrame to JSONL."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    return output_path


def load_jsonl(path: Path | str) -> pd.DataFrame:
    """Load a JSONL file as a DataFrame."""
    return pd.read_json(path, lines=True)


def ledgar_raw_paths(raw_dir: Path) -> dict[str, Path]:
    """Return expected LEDGAR raw split paths for a raw directory."""
    return {split: raw_dir / f"ledgar_{split}.jsonl" for split in ("train", "validation", "test")}


def load_local_ledgar_jsonl(raw_dir: Path) -> dict[str, pd.DataFrame] | None:
    """Load local LEDGAR JSONL splits if all expected files exist."""
    paths = ledgar_raw_paths(raw_dir)
    if all(path.exists() for path in paths.values()):
        return {split: load_jsonl(path) for split, path in paths.items()}
    return None


def load_original_ledgar_fallback(paths: ProjectPaths, *, seed: int = 42) -> dict[str, pd.DataFrame] | None:
    """Load the original one-file LEDGAR JSONL fallback when official splits are unavailable."""
    original_path = paths.raw_data_dir / "original_ledgar" / "ledgar.jsonl"
    if not original_path.exists():
        return None

    df = load_jsonl(original_path).dropna(how="all").reset_index(drop=True)
    if df.empty:
        return None

    if len(df) < 3:
        print("Loaded original LEDGAR fallback, but it is too small to split; using train only.")
        empty = pd.DataFrame(columns=df.columns)
        return {"train": df, "validation": empty.copy(), "test": empty.copy()}

    try:
        from sklearn.model_selection import train_test_split

        label_column = next((column for column in ("label", "labels", "category") if column in df.columns), None)
        stratify = None
        if label_column is not None:
            label_counts = df[label_column].value_counts()
            if len(label_counts) > 1 and label_counts.min() >= 2:
                stratify = df[label_column]

        train_df, temp_df = train_test_split(df, test_size=0.30, random_state=seed, stratify=stratify)
        temp_stratify = None
        if stratify is not None:
            temp_counts = temp_df[label_column].value_counts()
            if len(temp_counts) > 1 and temp_counts.min() >= 2:
                temp_stratify = temp_df[label_column]
        val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=seed, stratify=temp_stratify)
    except Exception:
        shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        train_end = int(len(shuffled) * 0.70)
        val_end = int(len(shuffled) * 0.85)
        train_df = shuffled.iloc[:train_end]
        val_df = shuffled.iloc[train_end:val_end]
        test_df = shuffled.iloc[val_end:]

    print("Loaded fallback LEDGAR from data/raw/original_ledgar/ledgar.jsonl with a reproducible 70/15/15 split.")
    return {
        "train": train_df.reset_index(drop=True),
        "validation": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def load_or_download_ledgar(
    paths: ProjectPaths,
    *,
    download_if_missing: bool = True,
    force_redownload: bool = False,
) -> dict[str, pd.DataFrame] | None:
    """Load LEDGAR from local JSONL files or download it from Hugging Face."""
    expected_paths = ledgar_raw_paths(paths.ledgar_raw_dir)
    if not force_redownload and all(path.exists() for path in expected_paths.values()):
        print("Loading LEDGAR from data/raw/lexglue_ledgar JSONL files.")
        splits = {split: load_jsonl(path) for split, path in expected_paths.items()}
    elif not force_redownload and (legacy := load_local_ledgar_jsonl(paths.legacy_ledgar_raw_dir)) is not None:
        print("Loading LEDGAR from existing data/raw/ledgar JSONL files and copying to data/raw/lexglue_ledgar.")
        splits = legacy
        for split, df in splits.items():
            save_jsonl(df, expected_paths[split])
    else:
        if not download_if_missing:
            fallback = load_original_ledgar_fallback(paths)
            if fallback is None:
                print("LEDGAR download is disabled and local JSONL files were not found.")
                return None
            splits = fallback
            for split, df in splits.items():
                save_jsonl(df, expected_paths[split])
            for split, df in splits.items():
                print(f"LEDGAR {split}: {len(df)} rows, columns={list(df.columns)}")
            return splits
        try:
            from datasets import load_dataset

            print('Downloading LEDGAR via load_dataset("coastalcph/lex_glue", "ledgar").')
            dataset = load_dataset("coastalcph/lex_glue", "ledgar")
            splits = {}
            for split in ("train", "validation", "test"):
                splits[split] = pd.DataFrame(dataset[split])
                save_jsonl(splits[split], expected_paths[split])
            label_feature = dataset["train"].features.get("label")
            if hasattr(label_feature, "names"):
                (paths.ledgar_raw_dir / "label_names.txt").write_text(
                    "\n".join(label_feature.names) + "\n",
                    encoding="utf-8",
                )
        except Exception as exc:
            print(f"LEDGAR download failed: {type(exc).__name__}: {exc}")
            fallback = load_original_ledgar_fallback(paths)
            if fallback is None:
                print(
                    "Place LEDGAR JSONL files in data/raw/lexglue_ledgar/ as "
                    "ledgar_train.jsonl, ledgar_validation.jsonl, and ledgar_test.jsonl."
                )
                return None
            splits = fallback
            for split, df in splits.items():
                save_jsonl(df, expected_paths[split])

    for split, df in splits.items():
        print(f"LEDGAR {split}: {len(df)} rows, columns={list(df.columns)}")
    return splits


def _copy_hf_file_to_raw(cache_path: str, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cache_path, target)
    return target


def download_cuad_if_missing(
    paths: ProjectPaths,
    *,
    download_if_missing: bool = True,
    force_redownload: bool = False,
) -> tuple[Path | None, Path | None]:
    """Download CUAD raw files from Hugging Face when missing."""
    cuad_json_path = paths.cuad_raw_dir / "CUAD_v1.json"
    master_clauses_path = paths.cuad_raw_dir / "master_clauses.csv"

    if not force_redownload and cuad_json_path.exists() and master_clauses_path.exists():
        print("Using existing CUAD files from data/raw/cuad/.")
        return cuad_json_path, master_clauses_path

    if not download_if_missing:
        print("CUAD download is disabled.")
        return (cuad_json_path if cuad_json_path.exists() else None, master_clauses_path if master_clauses_path.exists() else None)

    try:
        from huggingface_hub import hf_hub_download

        print("Downloading CUAD_v1.json and master_clauses.csv with hf_hub_download.")
        json_cache_path = hf_hub_download(repo_id="theatticusproject/cuad", filename="CUAD_v1/CUAD_v1.json", repo_type="dataset")
        csv_cache_path = hf_hub_download(repo_id="theatticusproject/cuad", filename="CUAD_v1/master_clauses.csv", repo_type="dataset")
        cuad_json_path = _copy_hf_file_to_raw(json_cache_path, cuad_json_path)
        master_clauses_path = _copy_hf_file_to_raw(csv_cache_path, master_clauses_path)
    except Exception as first_exc:
        print(f"Direct CUAD file download failed: {type(first_exc).__name__}: {first_exc}")
        try:
            from huggingface_hub import snapshot_download

            print("Trying snapshot_download for the CUAD dataset repository.")
            snapshot_dir = Path(
                snapshot_download(
                    repo_id="theatticusproject/cuad",
                    repo_type="dataset",
                    local_dir=paths.cuad_raw_dir,
                    local_dir_use_symlinks=False,
                )
            )
            json_candidates = list(snapshot_dir.rglob("CUAD_v1.json"))
            csv_candidates = list(snapshot_dir.rglob("master_clauses.csv"))
            if json_candidates:
                cuad_json_path = _copy_hf_file_to_raw(str(json_candidates[0]), cuad_json_path)
            if csv_candidates:
                master_clauses_path = _copy_hf_file_to_raw(str(csv_candidates[0]), master_clauses_path)
        except Exception as second_exc:
            print("CUAD download failed or files were not found. CUAD extension will be skipped.")
            print(f"snapshot_download error: {type(second_exc).__name__}: {second_exc}")
            return None, None

    final_json = cuad_json_path if cuad_json_path.exists() else None
    final_csv = master_clauses_path if master_clauses_path.exists() else None
    print(f"CUAD JSON found: {final_json}")
    print(f"CUAD master clauses CSV found: {final_csv}")
    if final_json is None or final_csv is None:
        print("CUAD download failed or files were not found. CUAD extension will be skipped.")
    return final_json, final_csv


def load_cuad_raw_files(cuad_json_path: Path | None, master_clauses_path: Path | None) -> tuple[dict[str, Any] | None, pd.DataFrame | None]:
    """Load raw CUAD JSON and master_clauses.csv."""
    raw_cuad_json = None
    master_clauses_df = None
    if cuad_json_path and Path(cuad_json_path).exists():
        with Path(cuad_json_path).open("r", encoding="utf-8") as file:
            raw_cuad_json = json.load(file)
    if master_clauses_path and Path(master_clauses_path).exists():
        master_clauses_df = pd.read_csv(master_clauses_path)
    return raw_cuad_json, master_clauses_df


def normalise_whitespace(text: Any) -> str:
    """Normalise whitespace only."""
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def cuad_label_from_qa(qa: dict[str, Any]) -> str:
    """Infer a CUAD clause label from QA metadata or question text."""
    for key in ("clause_type", "clause_category", "category", "label", "title"):
        value = normalise_whitespace(qa.get(key))
        if value:
            return value
    qa_id = normalise_whitespace(qa.get("id"))
    if "__" in qa_id:
        return normalise_whitespace(qa_id.rsplit("__", 1)[-1].replace("_", " "))
    question = normalise_whitespace(qa.get("question"))
    quoted = re.search(r'related to ["\u201c]([^"\u201d]+)["\u201d]', question, flags=re.IGNORECASE)
    if quoted:
        return normalise_whitespace(quoted.group(1))
    return question.strip(" ?.:")


def adapt_cuad_to_clause_classification(raw_cuad_json: dict[str, Any] | None) -> pd.DataFrame:
    """Convert CUAD QA/span annotations into clause-classification rows."""
    if raw_cuad_json is None:
        return pd.DataFrame(columns=["text", "label", "source_dataset", "source_id"])

    records = []
    for doc_idx, document in enumerate(raw_cuad_json.get("data", [])):
        source_id = normalise_whitespace(document.get("title") or document.get("id") or f"cuad_doc_{doc_idx}")
        for paragraph in document.get("paragraphs", []):
            for qa in paragraph.get("qas", []):
                if qa.get("is_impossible") is True:
                    continue
                label = cuad_label_from_qa(qa)
                if not label:
                    continue
                for answer in qa.get("answers", []) or []:
                    span_text = normalise_whitespace(answer.get("text") if isinstance(answer, dict) else answer)
                    if span_text:
                        records.append({"text": span_text, "label": label, "source_dataset": "CUAD", "source_id": source_id})

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=["text", "label", "source_dataset", "source_id"])
    return df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)[["text", "label", "source_dataset", "source_id"]]


def print_dataset_availability(
    ledgar_raw_splits: dict[str, pd.DataFrame] | None,
    cuad_json_path: Path | None,
    master_clauses_path: Path | None,
    cuad_clause_df: pd.DataFrame,
) -> None:
    """Print a compact dataset availability report."""
    print("\nDataset availability:")
    print(f"- LEDGAR downloaded/loaded: {'yes' if ledgar_raw_splits is not None else 'no'}")
    if ledgar_raw_splits is not None:
        print(f"- LEDGAR train size: {len(ledgar_raw_splits['train'])}")
        print(f"- LEDGAR validation size: {len(ledgar_raw_splits['validation'])}")
        print(f"- LEDGAR test size: {len(ledgar_raw_splits['test'])}")
    else:
        print("- LEDGAR train size: unavailable")
        print("- LEDGAR validation size: unavailable")
        print("- LEDGAR test size: unavailable")
    print(f"- CUAD JSON found: {'yes' if cuad_json_path else 'no'}")
    print(f"- CUAD master clauses CSV found: {'yes' if master_clauses_path else 'no'}")
    print(f"- CUAD adapted span examples available for optional analysis: {len(cuad_clause_df)}")
