
#!/usr/bin/env python3
"""Export only report-relevant NLP coursework assets into an existing Overleaf folder.

Run from Colab, local machine, or HPC where the repository/results already exist.
It scans results/, outputs/, and figures/ in-place and copies only small useful assets
into the Overleaf folder. It does not copy checkpoints, raw datasets, wandb folders,
or other heavy runtime artifacts.
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
from pathlib import Path
from typing import Iterable

DEFAULT_REPO = Path("/content/drive/MyDrive/Colab Notebooks/Education/NLP/Natural-Language-Processing")
DEFAULT_OVERLEAF_NAME = "overleaf_export"

LATEX_SUPPORT_FILES = [
    "report.tex",
    "custom.bib",
    "anthology.bib",
    "EACL2023.sty",
    "acl_natbib.bst",
    "acl2023.sty",
    "acl_natbib.bst",
    "natbib.sty",
]

SOURCE_DIR_NAMES = ["figures", "outputs", "results"]

FIG_EXTS = {".png", ".jpg", ".jpeg", ".pdf", ".svg"}
TABLE_EXTS = {".csv", ".xlsx", ".tex"}
METRIC_EXTS = {".json", ".jsonl", ".txt", ".md", ".log"}
ALLOWED_EXTS = FIG_EXTS | TABLE_EXTS | METRIC_EXTS

BLOCKED_EXTS = {
    ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".model",
    ".joblib", ".pkl", ".pickle", ".h5", ".keras", ".onnx",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".parquet", ".arrow", ".npy", ".npz",
}

SKIP_DIR_NAMES = {
    ".git", "wandb", "__pycache__", ".ipynb_checkpoints",
    "checkpoints", "checkpoint", "models", "model", "cache", "hf_cache",
    "datasets", "dataset", "raw", "raw_data", "data_cache",
}

IMPORTANT_KEYWORDS = [
    # EDA / preprocessing
    "eda", "audit", "missing", "outlier", "length", "label", "label_length",
    "tfidf", "similarity", "heatmap", "class_distribution", "distribution",
    "clause_length", "split", "summary", "leakage", "duplicate",
    # metrics / evaluation
    "metric", "metrics", "eval", "evaluation", "result", "results", "score",
    "f1", "macro", "weighted", "accuracy", "precision", "recall", "auc",
    "classification_report", "confusion", "roc",
    # stages / models
    "dummy", "baseline", "classical", "svm", "linear_svm", "logistic",
    "regression", "naive", "bayes", "nb", "hpt", "hyperparameter", "grid",
    "bilstm", "lstm", "transformer", "bert", "distilbert", "legalbert",
    "roberta", "deberta", "llm", "prompt", "qwen", "saullm", "mistral",
    "llama", "cuad", "external",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export report assets to an existing Overleaf folder.")
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO, help="Repository root path.")
    parser.add_argument(
        "--overleaf",
        type=Path,
        default=None,
        help="Existing Overleaf folder path. Default: <repo>/overleaf_export",
    )
    parser.add_argument("--max-mb", type=float, default=25.0, help="Maximum copied file size in MB.")
    parser.add_argument("--zip", action="store_true", help="Create a zip of the Overleaf folder after export.")
    parser.add_argument(
        "--overwrite-assets",
        action="store_true",
        help="Overwrite existing non-report assets if source is newer. report.tex is never overwritten.",
    )
    return parser.parse_args()


def has_skip_dir(path: Path) -> bool:
    return any(part in SKIP_DIR_NAMES for part in path.parts)


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / 1024 / 1024


def should_skip(path: Path, repo: Path, max_mb: float) -> tuple[bool, str]:
    if not path.is_file():
        return True, "not_file"
    try:
        rel = path.relative_to(repo)
    except ValueError:
        rel = path
    if has_skip_dir(rel):
        return True, "blocked_directory"
    if path.suffix.lower() in BLOCKED_EXTS:
        return True, "blocked_extension"
    if path.suffix.lower() not in ALLOWED_EXTS:
        return True, "unsupported_extension"
    if file_size_mb(path) > max_mb:
        return True, "too_large"
    return False, ""


def is_relevant(path: Path, repo: Path) -> bool:
    try:
        text = str(path.relative_to(repo)).lower()
    except ValueError:
        text = str(path).lower()
    return any(keyword in text for keyword in IMPORTANT_KEYWORDS)


def choose_destination(path: Path, repo: Path, overleaf: Path, root: Path) -> Path:
    ext = path.suffix.lower()
    lower = str(path).lower()

    if ext in FIG_EXTS:
        # Figures all go into Overleaf figures/. Keep filename short and LaTeX-friendly.
        return overleaf / "figures" / path.name

    if "eda" in lower or any(k in lower for k in ["missing", "outlier", "tfidf", "split_summary", "label_length"]):
        return overleaf / "outputs" / "eda" / path.name

    if ext in TABLE_EXTS:
        return overleaf / "outputs" / "tables" / path.name

    if any(k in lower for k in ["metric", "eval", "result", "classification_report", "confusion"]):
        return overleaf / "outputs" / "metrics" / path.name

    return overleaf / "outputs" / "summaries" / path.name


def copy_file(src: Path, dest: Path, overwrite: bool = False, protect_report_tex: bool = True) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if protect_report_tex and dest.name == "report.tex" and dest.exists():
        return None

    if dest.exists():
        if overwrite and src.stat().st_mtime > dest.stat().st_mtime:
            shutil.copy2(src, dest)
            return dest
        return None

    shutil.copy2(src, dest)
    return dest


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    i = 2
    while True:
        candidate = parent / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def find_includegraphics(report_tex: Path) -> list[str]:
    if not report_tex.exists():
        return []
    tex = report_tex.read_text(encoding="utf-8", errors="ignore")
    return re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)


def find_by_basename(filename: str, roots: Iterable[Path], repo: Path, max_mb: float) -> list[Path]:
    matches = []
    target = Path(filename).name
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob(target):
            skip, _ = should_skip(f, repo, max_mb)
            if not skip:
                matches.append(f)
    return matches


def write_manifest(overleaf: Path, rows: list[dict]) -> None:
    manifest = overleaf / "asset_manifest.csv"
    fieldnames = ["reason", "source", "copied_to", "size_mb"]
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    index = overleaf / "asset_index.tex"
    with index.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated report asset index. Optional checking file.\n")
        f.write("\\section*{Auto-generated Report Asset Index}\n")
        f.write("\\begin{itemize}\n")
        for row in rows:
            copied = row["copied_to"].replace("_", "\\_")
            source = row["source"].replace("_", "\\_")
            f.write(f"  \\item \\texttt{{{copied}}} --- from \\texttt{{{source}}}\n")
        f.write("\\end{itemize}\n")


def main() -> None:
    args = parse_args()
    repo = args.repo.expanduser().resolve()
    overleaf = (args.overleaf.expanduser().resolve() if args.overleaf else repo / DEFAULT_OVERLEAF_NAME)

    if not repo.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo}")

    overleaf.mkdir(parents=True, exist_ok=True)
    for sub in ["figures", "outputs/eda", "outputs/tables", "outputs/metrics", "outputs/summaries"]:
        (overleaf / sub).mkdir(parents=True, exist_ok=True)

    source_roots = [repo / name for name in SOURCE_DIR_NAMES]
    manifest_rows: list[dict] = []
    missing_explicit_figures: list[str] = []
    skipped_counts: dict[str, int] = {}

    def add_manifest(reason: str, src: Path, dest: Path) -> None:
        manifest_rows.append({
            "reason": reason,
            "source": str(src.relative_to(repo)) if src.is_relative_to(repo) else str(src),
            "copied_to": str(dest.relative_to(overleaf)) if dest.is_relative_to(overleaf) else str(dest),
            "size_mb": f"{file_size_mb(src):.4f}",
        })

    # 1) Copy LaTeX support files, but never overwrite an existing report.tex.
    for name in LATEX_SUPPORT_FILES:
        src = repo / name
        if src.exists() and src.is_file():
            copied = copy_file(src, overleaf / name, overwrite=False, protect_report_tex=True)
            if copied:
                add_manifest("latex_support", src, copied)

    # 2) Use report.tex includegraphics paths as the highest-priority explicit requirements.
    report_tex = repo / "report.tex"
    explicit_graphics = find_includegraphics(report_tex)
    for graphic_path in explicit_graphics:
        expected_dest = overleaf / graphic_path
        candidate = repo / graphic_path
        copied = None

        if candidate.exists() and candidate.is_file():
            copied = copy_file(candidate, expected_dest, overwrite=args.overwrite_assets, protect_report_tex=True)
            if copied:
                add_manifest("explicit_includegraphics", candidate, copied)
        else:
            matches = find_by_basename(graphic_path, source_roots, repo, args.max_mb)
            if matches:
                copied = copy_file(matches[0], expected_dest, overwrite=args.overwrite_assets, protect_report_tex=True)
                if copied:
                    add_manifest("basename_match_for_report_figure", matches[0], copied)
            else:
                missing_explicit_figures.append(graphic_path)

    # 3) Smart scan for useful evidence.
    already_sources = {row["source"] for row in manifest_rows}
    for root in source_roots:
        if not root.exists():
            continue
        for f in root.rglob("*"):
            skip, reason = should_skip(f, repo, args.max_mb)
            if skip:
                skipped_counts[reason] = skipped_counts.get(reason, 0) + 1
                continue
            try:
                rel_source = str(f.relative_to(repo))
            except ValueError:
                rel_source = str(f)
            if rel_source in already_sources:
                continue
            if not is_relevant(f, repo):
                skipped_counts["not_relevant"] = skipped_counts.get("not_relevant", 0) + 1
                continue

            dest = choose_destination(f, repo, overleaf, root)
            # If same file name exists, preserve both with suffix rather than silently clobbering.
            if dest.exists() and not args.overwrite_assets:
                dest = unique_dest(dest)

            copied = copy_file(f, dest, overwrite=args.overwrite_assets, protect_report_tex=True)
            if copied:
                add_manifest("smart_relevant_asset", f, copied)
                already_sources.add(rel_source)

    write_manifest(overleaf, manifest_rows)

    zip_path = None
    if args.zip:
        zip_path = shutil.make_archive(str(overleaf), "zip", overleaf)

    print("\nExport complete.")
    print(f"Repository:      {repo}")
    print(f"Overleaf folder: {overleaf}")
    print(f"Files copied:    {len(manifest_rows)}")
    print(f"Manifest:        {overleaf / 'asset_manifest.csv'}")
    print(f"Asset index:     {overleaf / 'asset_index.tex'}")
    if zip_path:
        print(f"ZIP created:     {zip_path}")

    if missing_explicit_figures:
        print("\nMissing figures referenced by report.tex:")
        for item in missing_explicit_figures:
            print(f"  - {item}")

    print("\nSkipped files summary:")
    for reason, count in sorted(skipped_counts.items()):
        print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()

