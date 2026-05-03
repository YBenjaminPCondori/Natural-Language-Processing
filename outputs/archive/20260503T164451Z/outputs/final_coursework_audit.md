# Final Coursework Audit

Created: 2026-05-03T16:44:26.430976+00:00

## Completed Models
_No rows._

## Failed or Skipped Models
| model_name | status | reason_if_skipped_or_failed |
| --- | --- | --- |
| random_uniform | skipped | metric missing |
| random_train_distribution | skipped | metric missing |
| majority_baseline | skipped | metric missing |
| logistic_regression | skipped | metric missing |
| linear_svm | skipped | metric missing |
| multinomial_nb | skipped | metric missing |
| distilbert-base-uncased | skipped | metric missing |
| qwen_zero_shot | skipped | metric missing |
| qwen_few_shot | failed | All Qwen few-shot outputs were invalid; do not treat as a meaningful completed result. |
| bilstm | skipped | metric missing |
| bert-base-uncased | skipped | metric missing |
| nlpaueb/legal-bert-base-uncased | skipped | metric missing |
| nlpaueb/bert-base-uncased-contracts | skipped | metric missing |
| roberta-base | skipped | metric missing |
| microsoft/deberta-v3-base | skipped | metric missing |
| qwen_retrieval_few_shot | skipped | metric missing |

## Configured but Not Run
_No rows._

## Protocol Check
- Main dataset: LEDGAR.
- Top-k labels are selected from training split only according to dataset verification.
- Validation is used for classical model selection and DistilBERT checkpoint selection.
- Test rows are used for final evaluation in canonical completed rows.
- Qwen few-shot is not valid as a meaningful completed result because existing invalid-output rate is 1.0.

## Validation Commands Used
- `python scripts/build_coursework_artifacts.py`
- `python -m py_compile modules/*.py scripts/*.py`
