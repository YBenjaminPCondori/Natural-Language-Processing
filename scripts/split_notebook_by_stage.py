"""Split the master LEDGAR notebook into smaller stage notebooks.

The generated notebooks are clean copies with execution outputs removed.
The master notebook is not modified.
"""

from __future__ import annotations

import copy
import json
import re
import uuid
from pathlib import Path
from typing import Any


PROJECT_ROOT_MARKERS = ("pyproject.toml", "modules")
SOURCE_NOTEBOOK = Path("notebooks/ledgar_clause_classification_pipeline.ipynb")
STAGE_DIR = Path("notebooks/stages")


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if all((candidate / marker).exists() for marker in PROJECT_ROOT_MARKERS):
            return candidate
    raise RuntimeError("Could not locate project root.")


def source_to_lines(source: Any) -> list[str]:
    if isinstance(source, list):
        return source
    text = str(source)
    return text.splitlines(keepends=True)


def clean_cell(cell: dict[str, Any]) -> dict[str, Any]:
    cleaned = copy.deepcopy(cell)
    cleaned["id"] = uuid.uuid4().hex[:8]
    cleaned["source"] = source_to_lines(cleaned.get("source", ""))
    cleaned["metadata"] = {}
    if cleaned.get("cell_type") == "code":
        cleaned["execution_count"] = None
        cleaned["outputs"] = []
    return cleaned


def markdown_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def copy_cells(master_cells: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    return [clean_cell(master_cells[index]) for index in indices]


def stage_intro(title: str, source_indices: list[int], notes: list[str] | None = None) -> dict[str, Any]:
    note_lines = notes or []
    source_text = ", ".join(str(index) for index in source_indices)
    body = [
        f"# {title}\n",
        "\n",
        "Generated from `notebooks/ledgar_clause_classification_pipeline.ipynb`.\n",
        "\n",
        f"Source cell indices: `{source_text}`.\n",
    ]
    if note_lines:
        body.append("\n")
        body.extend(f"- {line}\n" for line in note_lines)
    return markdown_cell("".join(body))


def artifact_stage_cells(title: str, source_indices: list[int]) -> list[dict[str, Any]]:
    return [
        stage_intro(
            title,
            source_indices,
            [
                "This stage rebuilds report-facing artifacts from saved outputs.",
                "It does not retrain models.",
            ],
        ),
        code_cell(
            """from pathlib import Path
import sys


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "modules").is_dir():
            return candidate
    raise RuntimeError("Could not locate project root containing pyproject.toml and modules/.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")
"""
        ),
        code_cell(
            """from modules.coursework_artifacts import build_all_coursework_artifacts


summary = build_all_coursework_artifacts(PROJECT_ROOT)
summary
"""
        ),
    ]


def final_audit_cells(master_cells: list[dict[str, Any]], source_indices: list[int]) -> list[dict[str, Any]]:
    return [
        stage_intro(
            "09 Final Artifact Audit",
            source_indices,
            [
                "Run this after experiments or after rebuilding report artifacts.",
                "It checks evidence files and report-facing figures/tables.",
            ],
        ),
        *copy_cells(master_cells, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13]),
        *copy_cells(master_cells, source_indices),
    ]


def notebook_payload(master: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = copy.deepcopy(master.get("metadata", {}))
    metadata.pop("widgets", None)
    return {
        "cells": cells,
        "metadata": metadata,
        "nbformat": master.get("nbformat", 4),
        "nbformat_minor": master.get("nbformat_minor", 5),
    }


def write_notebook(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_stage_names(stages: list[tuple[str, str, list[int], list[str]]]) -> None:
    names = [filename for filename, *_ in stages]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate stage notebook names: {duplicates}")
    for filename in names:
        if not re.match(r"^[0-9]{2}_[a-z0-9_]+\.ipynb$", filename):
            raise ValueError(f"Unexpected stage notebook name: {filename}")


def main() -> None:
    project_root = find_project_root()
    source_path = project_root / SOURCE_NOTEBOOK
    master = json.loads(source_path.read_text(encoding="utf-8"))
    master_cells = master["cells"]

    setup = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    data = [15, 16, 17, 18, 19]
    shared_state = [20, 21, 22]

    stages: list[tuple[str, str, list[int], list[str]]] = [
        (
            "00_setup_and_config.ipynb",
            "00 Setup and Configuration",
            setup,
            ["Defines paths, imports, flags, seeds, device checks, and W&B startup."],
        ),
        (
            "01_dataset_preprocessing_eda.ipynb",
            "01 Dataset Download, Preprocessing, and EDA",
            setup + data,
            ["Loads LEDGAR/CUAD availability, preprocesses LEDGAR, and writes EDA outputs."],
        ),
        (
            "02_dummy_baselines.ipynb",
            "02 Dummy Baselines",
            setup + data + shared_state + [23, 24],
            ["Runs random and majority baselines on the processed LEDGAR test split."],
        ),
        (
            "03_classical_tfidf_models.ipynb",
            "03 Classical TF-IDF Models",
            setup + data + shared_state + [25, 26, 27, 28, 29],
            ["Runs TF-IDF Logistic Regression, Linear SVM, and Naive Bayes experiments."],
        ),
        (
            "04_neural_sequence_bilstm.ipynb",
            "04 Neural Sequence Baseline",
            setup + data + shared_state + [30, 31],
            ["BiLSTM is configured but only trains when RUN_SEQUENCE_MODEL is set to True."],
        ),
        (
            "05_transformer_finetuning.ipynb",
            "05 Fine-Tuned Transformer",
            setup + data + shared_state + [32, 33, 34],
            ["Runs the configured transformer when RUN_TRANSFORMER is True and runtime requirements are met."],
        ),
        (
            "06_qwen_prompting.ipynb",
            "06 Qwen Prompting",
            setup + data + shared_state + [35, 36, 37],
            ["Runs Qwen prompting when RUN_QWEN_BASELINE is True and runtime requirements are met."],
        ),
        (
            "07_agentic_review.ipynb",
            "07 Agentic Review Prototype",
            setup
            + data
            + shared_state
            + [25, 26, 27, 28, 29]
            + [38]
            + [999]
            + [39],
            ["Runs the optional review prototype using the best classical model."],
        ),
    ]
    validate_stage_names(stages)

    output_dir = project_root / STAGE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for filename, title, indices, notes in stages:
        cells: list[dict[str, Any]] = [stage_intro(title, [index for index in indices if index != 999], notes)]
        for index in indices:
            if index == 999:
                cells.append(
                    code_cell(
                        """qwen_model = globals().get("qwen_model", None)
qwen_tokenizer = globals().get("qwen_tokenizer", None)
"""
                    )
                )
            else:
                cells.extend(copy_cells(master_cells, [index]))
        payload = notebook_payload(master, cells)
        destination = output_dir / filename
        write_notebook(destination, payload)
        generated.append(destination)

    artifact_cells = artifact_stage_cells("08 Results, Error Analysis, and Report Exports", [40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
    destination = output_dir / "08_results_error_exports.ipynb"
    write_notebook(destination, notebook_payload(master, artifact_cells))
    generated.append(destination)

    destination = output_dir / "09_final_artifact_audit.ipynb"
    write_notebook(destination, notebook_payload(master, final_audit_cells(master_cells, [51, 52])))
    generated.append(destination)

    readme = output_dir / "README.md"
    readme.write_text(
        """# Stage Notebooks

These notebooks are generated from `../ledgar_clause_classification_pipeline.ipynb` by
`../../scripts/split_notebook_by_stage.py`.

The master notebook remains the source of truth. The split notebooks are for clearer
stage-by-stage execution and review.

Recommended order:

1. `00_setup_and_config.ipynb`
2. `01_dataset_preprocessing_eda.ipynb`
3. `02_dummy_baselines.ipynb`
4. `03_classical_tfidf_models.ipynb`
5. `04_neural_sequence_bilstm.ipynb`
6. `05_transformer_finetuning.ipynb`
7. `06_qwen_prompting.ipynb`
8. `07_agentic_review.ipynb`
9. `08_results_error_exports.ipynb`
10. `09_final_artifact_audit.ipynb`

Notes:

- Stage notebooks have outputs cleared.
- Model stages include setup and LEDGAR preprocessing cells so they can run on their own.
- Expensive stages still obey the same flags as the master notebook.
- The transformer stage includes optional two-stage HPT via `RUN_TRANSFORMER_HPT`.
- `08_results_error_exports.ipynb` rebuilds report-facing artifacts from saved outputs; it does not retrain models.
""",
        encoding="utf-8",
    )
    generated.append(readme)

    for path in generated:
        print(path.relative_to(project_root))


if __name__ == "__main__":
    main()
