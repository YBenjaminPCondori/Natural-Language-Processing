# Legal Clause Classification

Controlled NLP coursework pipeline for LEDGAR legal clause classification.

The project compares:

- dummy baselines
- classical TF-IDF models
- an optional BiLSTM neural sequence model
- fine-tuned transformer encoders
- prompt-based instruction-tuned LLMs

The main dataset is LEDGAR. CUAD is treated as an external out-of-distribution evaluation dataset only; it is not merged into the LEDGAR train, validation, or test splits.

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

## CUAD External Evaluation

CUAD is span-annotated in its raw form. The project converts non-empty CUAD answer spans into clause-level classification rows only after the LEDGAR final test evaluation has been completed. Because CUAD and LEDGAR use different label spaces, a conservative manual mapping is applied before evaluation. Unmapped CUAD labels are excluded from metrics and counted in `outputs/data/cuad_label_mapping_report.json`.

CUAD is not used for training, validation, hyperparameter tuning, prompt selection, or model selection. It is used to assess cross-dataset generalisation and failure modes.

Run the external evaluation from existing LEDGAR-trained artifacts:

```bash
python scripts/run_cuad_external_eval.py --no-download
```

The command writes:

- `outputs/data/cuad_converted_raw.csv`
- `outputs/data/cuad_converted_mapped.csv`
- `outputs/data/cuad_external_eval.csv`
- `outputs/data/cuad_label_mapping_report.json`
- `outputs/predictions/cuad_external_<model_name>_predictions.csv`
- `outputs/metrics/cuad_external_<model_name>_metrics.json`
- `outputs/error_analysis/cuad_external_<model_name>_errors.csv`
- `outputs/error_analysis/cuad_external_<model_name>_summary.json`
- `outputs/metrics/final_model_comparison_summary.csv`

Prompt-only LLM rows are marked as skipped unless a reusable local model/tokenizer is available for a dedicated CUAD prompting run.

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
10. CUAD external evaluation and error analysis, run only after final LEDGAR test evaluation

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

Run two-stage HPT for the main transformer on CUDA:

```bash
python scripts/run_transformer_hpt.py --model-name distilbert-base-uncased --random-trials 8 --bayes-trials 8 --wandb
```

Run Qwen prompting on CUDA:

```bash
python scripts/run_qwen_prompting.py --model-name Qwen/Qwen2.5-3B-Instruct --eval-sample-size 200
```

Run CUAD external evaluation after LEDGAR models exist:

```bash
python scripts/run_cuad_external_eval.py --no-download
```

Run inference-only instruction-tuned LLM evaluation:

```bash
python scripts/evaluate_llm_classifiers.py --max-examples-per-split 20 --models saullm_7b qwen_small qwen_7b --quantization 4bit --wandb
```

Full LLM evaluation after validation decoding selection:

```bash
python scripts/evaluate_llm_classifiers.py --models saullm_7b qwen_small qwen_7b --tune-decoding-on-validation --evaluate-test --quantization 4bit --wandb
```

The dry run writes headers and failed/skipped records if models cannot load. Full evaluation should be run on Colab/A100 or equivalent GPU.

Run transformer fine-tuning from the notebook on Colab/A100 by setting:

```python
RUN_TRANSFORMER = True
RUN_TRANSFORMER_HPT = False
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
```

To try other encoder models, set `TRANSFORMER_MODEL_NAME` to one of:

- `bert-base-uncased`
- `nlpaueb/legal-bert-base-uncased`
- `nlpaueb/bert-base-uncased-contracts`
- `roberta-base`
- `microsoft/deberta-v3-base`

Only report a model as completed after metrics, predictions, logs, and checkpoint/report artifacts exist.

## Instruction-Tuned LLM Evaluation

This stage is prompt-based inference only; it does not fine-tune LLMs.

Configured models:

- SaulLM-7B-Instruct, `Equall/Saul-7B-Instruct-v1`, included as a legal-domain instruction-tuned model.
- Qwen small, `Qwen/Qwen2.5-3B-Instruct`, included as an efficient general-purpose instruction model.
- Qwen 7B, `Qwen/Qwen2.5-7B-Instruct`, included as a stronger general-purpose instruction model.

Validation data is used for decoding selection and qualitative error-analysis probes. The probes include greedy decoding, light beam search, high-temperature decoding, heavy sampling, brief-rationale prompting, and optional few-shot prompting with training-only examples. Test data is evaluated only after settings are fixed. Outputs are written to:

- `outputs/llm_results.csv`
- `outputs/llm_final_predictions.csv`
- `outputs/llm_predictions.csv`
- `outputs/llm_per_class_metrics.csv`
- `outputs/llm_error_analysis_decoding_comparison.csv`
- `outputs/llm_cot_rationale_examples.csv`
- `outputs/llm_decoding_settings.json`
- `outputs/llm_failed_models.csv`
- `figures/llm_model_comparison.png`

## Outputs

Canonical report-facing artifacts are saved under `outputs/`, including:

- `main_results.csv`
- `main_results.json`
- `main_results_latex.tex`
- `model_evidence_table.md`
- `dataset_verification.json`
- `repo_audit.md`
- `final_coursework_audit.md`
- `metrics/final_model_comparison_summary.csv`

Figures are saved under `figures/` and copied where needed to `outputs/figures/`.

Existing output files are archived under `outputs/archive/<timestamp>/` before the audit builder overwrites derived report-facing files.

## Reproducibility Rules

- Train only on the training split.
- Tune hyperparameters and prompts only on validation.
- Select checkpoints using validation macro-F1.
- Use test only for final evaluation.
- Use CUAD only after final LEDGAR test evaluation as external out-of-distribution evidence.
- Mark failed/skipped runs explicitly.
- Do not invent metrics or reuse stale predictions as fresh results.
