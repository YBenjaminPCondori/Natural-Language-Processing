# AGENTS.md

## Project purpose

This repository supports an NLP coursework experiment pipeline for contract clause classification using LexGLUE LEDGAR.

## Academic boundary

Do not write the coursework report.
Do not invent results.
Do not fabricate metrics.
Do not write final conclusions.

Only help with:
- dataset loading
- preprocessing
- model training code
- evaluation metrics
- plots
- prediction exports
- debugging

## Main dataset

Use Hugging Face LexGLUE LEDGAR as the main dataset:

load_dataset("coastalcph/lex_glue", "ledgar")

Export Hugging Face splits to JSONL:

data/raw/lexglue_ledgar/ledgar_train.jsonl
data/raw/lexglue_ledgar/ledgar_validation.jsonl
data/raw/lexglue_ledgar/ledgar_test.jsonl

Processed outputs:

data/processed/ledgar_train.jsonl
data/processed/ledgar_validation.jsonl
data/processed/ledgar_test.jsonl

Use JSONL only unless explicitly told otherwise.
Do not use CSV unless explicitly requested.

## Fallback dataset

Only if Hugging Face loading fails, use:

data/raw/original_ledgar/ledgar.jsonl

Do not use the original raw LEDGAR unless explicitly needed.

## Reference notebooks

Reference notebooks are stored in:

reference_notebooks/

Use them only as formatting/style references.

Do not:
- copy their content
- copy their exercises
- copy their results
- modify them

Follow their general teaching-notebook style:
- numbered headings
- short markdown explanations
- compact code cells
- TODO prompts
- clear outputs

## Models

Core models:
- Random baseline
- Majority baseline
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM

Transformer templates:
- bert-base-uncased
- nlpaueb/legal-bert-base-uncased

Transformers should be disabled by default with:

RUN_TRANSFORMERS = False

## GPU / CUDA

Add a CUDA check in the notebook.
Use CUDA automatically for transformer fine-tuning if available.
Use fp16=torch.cuda.is_available().
Do not force CPU if CUDA is available.
Warn clearly if CUDA is unavailable.
TF-IDF, Logistic Regression, and Linear SVM are CPU-based.

## Metrics

Primary metric:
macro-F1

Secondary metrics:
- accuracy
- weighted-F1
- macro precision
- macro recall
- per-class F1
- confusion matrix

## Output rules

Save outputs under:

outputs/

Save figures under:

outputs/figures/

Save predictions under:

outputs/predictions/

Save models under:

models/

Use pathlib.
Use random_state=42.
Use JSONL for saved prediction/result files where possible.