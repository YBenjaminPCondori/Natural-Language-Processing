# Legal Clause Classification

Controlled NLP coursework pipeline for LEDGAR legal clause classification.

The project compares:

- dummy baselines
- classical TF-IDF models
- an optional BiLSTM neural sequence model
- fine-tuned transformer encoders
- prompt-based instruction-tuned LLMs

The main dataset is LEDGAR. CUAD may be downloaded for optional qualitative inspection only; it is not merged into the LEDGAR train, validation, or test splits.

## Dataset

Expected raw LEDGAR files:

- `data/raw/lexglue_ledgar/ledgar_train.jsonl`
- `data/raw/lexglue_ledgar/ledgar_validation.jsonl`
- `data/raw/lexglue_ledgar/ledgar_test.jsonl`
- `data/raw/lexglue_ledgar/label_names.txt`

Processed files:

- `data/processed/ledgar_train.jsonl`
- `data/processed/ledgar_validation.jsonl`
- `data/processed/ledgar_test.jsonl`
- `data/processed/dataset_summary.json`

Top-k labels are selected from the training split only. Validation and test are filtered using that training-selected label set. Test results must not be used for model selection or prompt tuning.

## Install

Local:

```bash
python -m pip install -r requirements.txt
```

Colab:

```python
%pip install -q -r requirements-colab.txt
```

Use a CUDA runtime for transformer, Qwen, and BiLSTM runs. The classical TF-IDF models run on CPU.

## Main Notebook

Open:

`notebooks/ledgar_clause_classification_pipeline.ipynb`

The notebook contains:

1. setup and configuration
2. LEDGAR/CUAD raw data setup
3. LEDGAR preprocessing and EDA
4. dummy baselines
5. classical TF-IDF models
6. optional BiLSTM sequence baseline
7. optional DistilBERT fine-tuning
8. optional Qwen prompting
9. error analysis and report artifact export

Heavy runs can be skipped with flags such as `RUN_SEQUENCE_MODEL`, `RUN_TRANSFORMER`, and `RUN_QWEN_BASELINE`.

## Standalone Commands

Build audit and canonical report artifacts from current evidence:

```bash
python scripts/build_coursework_artifacts.py
```

Run the BiLSTM baseline:

```bash
python scripts/run_sequence_model.py
```

Run one transformer encoder from the command line:

```bash
python scripts/run_transformer_model.py --model-name distilbert-base-uncased
```

Run Qwen prompting on CUDA:

```bash
python scripts/run_qwen_prompting.py --model-name Qwen/Qwen2.5-3B-Instruct --eval-sample-size 200
```

Run transformer fine-tuning from the notebook on Colab/A100 by setting:

```python
RUN_TRANSFORMER = True
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
```

To try other encoder models, set `TRANSFORMER_MODEL_NAME` to one of:

- `bert-base-uncased`
- `nlpaueb/legal-bert-base-uncased`
- `nlpaueb/bert-base-uncased-contracts`
- `roberta-base`
- `microsoft/deberta-v3-base`

Only report a model as completed after metrics, predictions, logs, and checkpoint/report artifacts exist.

## Outputs

Canonical report-facing artifacts are saved under `outputs/`, including:

- `main_results.csv`
- `main_results.json`
- `main_results_latex.tex`
- `model_evidence_table.md`
- `dataset_verification.json`
- `repo_audit.md`
- `final_coursework_audit.md`

Figures are saved under `figures/` and copied where needed to `outputs/figures/`.

Existing output files are archived under `outputs/archive/<timestamp>/` before the audit builder overwrites derived report-facing files.

## Reproducibility Rules

- Train only on the training split.
- Tune hyperparameters and prompts only on validation.
- Select checkpoints using validation macro-F1.
- Use test only for final evaluation.
- Mark failed/skipped runs explicitly.
- Do not invent metrics or reuse stale predictions as fresh results.
