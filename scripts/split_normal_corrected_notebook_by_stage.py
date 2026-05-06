from __future__ import annotations

import copy
import json
import re
import uuid
import zipfile
from pathlib import Path
from typing import Any

MASTER = Path('/mnt/data/ledgar_clause_classification_pipeline_NORMAL_CORRECTED.ipynb')
OUT_ROOT = Path('/mnt/data/normal_corrected_stage_notebooks')
STAGE_DIR = OUT_ROOT / 'notebooks' / 'stages'
SCRIPT_DIR = OUT_ROOT / 'scripts'


def lines(src: Any) -> list[str]:
    if isinstance(src, list):
        return src
    return str(src).splitlines(keepends=True)


def clean_cell(cell: dict[str, Any]) -> dict[str, Any]:
    c = copy.deepcopy(cell)
    c['id'] = uuid.uuid4().hex[:8]
    c['source'] = lines(c.get('source', ''))
    # Keep notebook-specific tags out; clear outputs.
    c['metadata'] = {}
    if c.get('cell_type') == 'code':
        c['execution_count'] = None
        c['outputs'] = []
    return c


def md(text: str) -> dict[str, Any]:
    return {'cell_type':'markdown','id':uuid.uuid4().hex[:8],'metadata':{},'source':text.splitlines(keepends=True)}


def code(text: str) -> dict[str, Any]:
    return {'cell_type':'code','execution_count':None,'id':uuid.uuid4().hex[:8],'metadata':{},'outputs':[],'source':text.splitlines(keepends=True)}


def stage_intro(title: str, indices: list[int], notes: list[str] | None = None) -> dict[str, Any]:
    notes = notes or []
    body = [
        f'# {title}\n',
        '\n',
        'Generated from `notebooks/ledgar_clause_classification_pipeline_NORMAL_CORRECTED.ipynb`.\n',
        '\n',
        f'Source master cell indices: `{", ".join(map(str, indices))}`.\n',
        '\n',
        '**Use:** run from top to bottom. Outputs are cleared.\n',
    ]
    if notes:
        body.append('\n')
        body.append('Notes:\n')
        body.extend([f'- {n}\n' for n in notes])
    return md(''.join(body))


def notebook_payload(master: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, Any]:
    metadata = copy.deepcopy(master.get('metadata', {}))
    metadata.pop('widgets', None)
    return {'cells': cells, 'metadata': metadata, 'nbformat': master.get('nbformat', 4), 'nbformat_minor': master.get('nbformat_minor', 5)}


def make(master: dict[str, Any], filename: str, title: str, indices: list[int], notes: list[str] | None = None, extra_cells: list[dict[str, Any]] | None = None) -> Path:
    cells = [stage_intro(title, indices, notes)]
    for i in indices:
        cells.append(clean_cell(master['cells'][i]))
    if extra_cells:
        cells.extend(extra_cells)
    payload = notebook_payload(master, cells)
    path = STAGE_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    return path


def ast_check(path: Path) -> list[str]:
    import ast
    errors = []
    nb = json.loads(path.read_text(encoding='utf-8'))
    for i,c in enumerate(nb['cells']):
        if c.get('cell_type') != 'code':
            continue
        src = ''.join(c.get('source', []))
        try:
            ast.parse(src)
        except SyntaxError as e:
            errors.append(f'{path.name}: cell {i}: {e}')
    return errors


def main() -> None:
    master = json.loads(MASTER.read_text(encoding='utf-8'))
    if OUT_ROOT.exists():
        import shutil
        shutil.rmtree(OUT_ROOT)
    STAGE_DIR.mkdir(parents=True)
    SCRIPT_DIR.mkdir(parents=True)

    setup = list(range(1, 16))
    data_preprocess = list(range(16, 32))
    shared = list(range(32, 35))
    dummy = list(range(35, 37))
    classical = list(range(37, 49))
    sequence = list(range(49, 51))
    transformer = list(range(51, 54))
    qwen = list(range(54, 57))
    agentic = list(range(57, 59))
    final_eval = list(range(59, 66))
    cuad = list(range(66, 68))
    report_exports = list(range(68, 71))
    audit = list(range(71, 74))

    generated = []
    generated.append(make(master, '00_setup_and_config.ipynb', '00 Setup and Configuration', setup,
        ['Defines repo paths, imports, flags, sanity checks, device, and optional W&B startup.']))

    generated.append(make(master, '01_dataset_preprocessing_eda.ipynb', '01 Dataset Download, Preprocessing, and EDA', setup + data_preprocess,
        ['Downloads/loads LEDGAR and CUAD availability, preprocesses LEDGAR, and writes EDA/preprocessing outputs.']))

    generated.append(make(master, '02_dummy_baselines.ipynb', '02 Dummy Baselines', setup + data_preprocess + shared + dummy,
        ['Runs random and majority baselines after preprocessing.']))

    generated.append(make(master, '03_classical_tfidf_models.ipynb', '03 Classical TF-IDF Models', setup + data_preprocess + shared + classical,
        ['Runs the classical TF-IDF/BOW feature demonstrations and classical model grid search.']))

    generated.append(make(master, '04_neural_sequence_bilstm.ipynb', '04 Neural Sequence BiLSTM', setup + data_preprocess + shared + sequence,
        ['Runs the BiLSTM sequence baseline when RUN_SEQUENCE_MODEL=True.']))

    generated.append(make(master, '05_transformer_finetuning.ipynb', '05 Fine-Tuned Transformer and Transformer HPT', setup + data_preprocess + shared + transformer,
        ['Runs the configured transformer and optional two-stage transformer HPT when flags are enabled.']))

    generated.append(make(master, '06_qwen_prompting.ipynb', '06 Qwen Prompting Baseline', setup + data_preprocess + shared + qwen,
        ['Runs the Qwen prompting baseline when RUN_QWEN_BASELINE=True.']))

    # Agentic stage needs a supervised classifier. It does not need Qwen to run; Qwen globals are optional.
    generated.append(make(master, '07_agentic_review.ipynb', '07 Agentic Review Prototype', setup + data_preprocess + shared + classical + agentic,
        ['Optional/non-core extension. It retrains classical models in this standalone stage so best_classical_model exists.']))

    # Standalone final evaluation must have model results in memory, so it includes all model stages before final comparison and error analysis.
    generated.append(make(master, '08_final_comparison_error_analysis.ipynb', '08 Final Comparison and Error Analysis', setup + data_preprocess + shared + dummy + classical + sequence + transformer + qwen + agentic + final_eval,
        ['Standalone version includes prior model stages because final comparison uses in-memory completed_results.']))

    generated.append(make(master, '09_cuad_external_eval.ipynb', '09 CUAD External Evaluation', setup + data_preprocess + shared + cuad,
        ['Runs CUAD as an external evaluation only; CUAD is not used for LEDGAR training/validation. Requires saved models for full evaluation.']))

    generated.append(make(master, '10_report_artifact_exports.ipynb', '10 Report Artifact Exports', setup + data_preprocess + shared + dummy + classical + sequence + transformer + qwen + agentic + final_eval + cuad + report_exports,
        ['Standalone version includes previous stages so completed_results, prediction_tables, and error_outputs exist before export_report_artifacts.']))

    generated.append(make(master, '11_final_artifact_audit.ipynb', '11 Final Artifact Audit', setup + audit,
        ['Checks required report-facing outputs and evidence files. Does not retrain models.']))

    # README
    readme = '''# Stage Notebooks — Normal Corrected Pipeline

These stage notebooks were regenerated from `ledgar_clause_classification_pipeline_NORMAL_CORRECTED.ipynb`.
They are synced to the corrected normal notebook, not the HPC exhaustive notebook.

Recommended order:

1. `00_setup_and_config.ipynb`
2. `01_dataset_preprocessing_eda.ipynb`
3. `02_dummy_baselines.ipynb`
4. `03_classical_tfidf_models.ipynb`
5. `04_neural_sequence_bilstm.ipynb`
6. `05_transformer_finetuning.ipynb`
7. `06_qwen_prompting.ipynb`
8. `07_agentic_review.ipynb` — optional/non-core extension
9. `08_final_comparison_error_analysis.ipynb`
10. `09_cuad_external_eval.ipynb`
11. `10_report_artifact_exports.ipynb`
12. `11_final_artifact_audit.ipynb`

Important notes:

- Outputs and execution counts are cleared.
- Each stage includes setup cells so it can be opened independently.
- Later standalone stages include earlier model stages when they require in-memory variables such as `completed_results`, `prediction_tables`, or `best_classical_model`.
- `07_agentic_review.ipynb` is included because it exists in the updated notebook, but it should be treated as optional/future-work unless your report explicitly discusses it.
- For Moodle submission clarity, keep one master notebook plus these clean stage notebooks. Archive old duplicate master notebooks.
'''
    (STAGE_DIR / 'README.md').write_text(readme, encoding='utf-8')
    generated.append(STAGE_DIR / 'README.md')

    # Regeneration script for repo use.
    script = Path(__file__).read_text(encoding='utf-8')
    (SCRIPT_DIR / 'split_normal_corrected_notebook_by_stage.py').write_text(script, encoding='utf-8')
    generated.append(SCRIPT_DIR / 'split_normal_corrected_notebook_by_stage.py')

    # Validation notes
    errors = []
    for p in generated:
        if p.suffix == '.ipynb':
            errors.extend(ast_check(p))
            text = p.read_text(encoding='utf-8')
            if 'paths.outputs_dir' in text:
                errors.append(f'{p.name}: contains paths.outputs_dir')

    notes = [
        '# Stage Notebook Sync Notes\n\n',
        'Source: `ledgar_clause_classification_pipeline_NORMAL_CORRECTED.ipynb`.\n\n',
        f'Generated stage notebooks: {sum(1 for p in generated if p.suffix == ".ipynb")}\n\n',
        f'Python syntax errors in code cells: **{len(errors)}**\n\n',
        'Remaining `paths.outputs_dir` occurrences across generated notebooks: **0**\n\n' if not any('paths.outputs_dir' in e for e in errors) else 'Remaining `paths.outputs_dir` occurrence check failed.\n\n',
        '## Files\n\n',
    ]
    for p in generated:
        notes.append(f'- `{p.relative_to(OUT_ROOT)}`\n')
    if errors:
        notes.append('\n## Errors\n\n')
        for e in errors:
            notes.append(f'- {e}\n')
    (OUT_ROOT / 'STAGE_SYNC_NOTES.md').write_text(''.join(notes), encoding='utf-8')

    # Zip bundle
    zip_path = Path('/mnt/data/ledgar_NORMAL_CORRECTED_STAGE_NOTEBOOKS.zip')
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in OUT_ROOT.rglob('*'):
            if p.is_file():
                z.write(p, p.relative_to(OUT_ROOT))
    print(zip_path)
    print('generated', len([p for p in generated if p.suffix == '.ipynb']), 'stage notebooks')
    print('errors', errors)

if __name__ == '__main__':
    main()
