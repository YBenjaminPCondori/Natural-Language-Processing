"""Run CUAD as an external post-test evaluation dataset."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert CUAD spans and evaluate existing LEDGAR-trained models externally.")
    parser.add_argument("--no-download", action="store_true", help="Use only local CUAD files; write failure status if missing.")
    parser.add_argument("--min-span-chars", type=int, default=20)
    parser.add_argument("--min-span-words", type=int, default=4)
    parser.add_argument("--skip-transformer", action="store_true", help="Write an explicit skipped status instead of loading the saved transformer.")
    parser.add_argument("--prepare-only", action="store_true", help="Convert/map CUAD and run safeguards without evaluating any models.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Optional debug cap for CUAD external rows.")
    parser.add_argument("--transformer-model-name", default="distilbert-base-uncased")
    parser.add_argument("--transformer-max-length", type=int, default=256)
    parser.add_argument("--transformer-batch-size", type=int, default=16)
    args = parser.parse_args()

    project_root = find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from modules.cuad_external import run_cuad_external_evaluation
    from modules.data_setup import build_project_paths

    paths = build_project_paths(project_root)
    summary = run_cuad_external_evaluation(
        paths,
        download_if_missing=not args.no_download,
        min_span_chars=args.min_span_chars,
        min_span_words=args.min_span_words,
        run_transformer=not args.skip_transformer,
        transformer_model_name=args.transformer_model_name,
        transformer_max_length=args.transformer_max_length,
        transformer_batch_size=args.transformer_batch_size,
        prepare_only=args.prepare_only,
        max_eval_samples=args.max_eval_samples,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
