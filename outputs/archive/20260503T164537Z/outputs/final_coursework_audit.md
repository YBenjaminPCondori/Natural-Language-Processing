# Final Coursework Audit

Created: 2026-05-03T16:44:57.648720+00:00

## Completed Models
| model_name | model_family | test_macro_f1 | evidence_path |
| --- | --- | --- | --- |
| random_uniform | baseline | 0.045640018794094 | results/baselines/classification_reports/random_uniform_report.json; results/baselines/confusion_matrices/random_uniform_confusion_matrix.png |
| random_train_distribution | baseline | 0.0445159239499602 | results/baselines/classification_reports/random_train_distribution_report.json; results/baselines/confusion_matrices/random_train_distribution_confusion_matrix.png |
| majority_baseline | baseline | 0.0107169811320754 | results/baselines/classification_reports/majority_baseline_report.json; results/baselines/confusion_matrices/majority_baseline_confusion_matrix.png |
| logistic_regression | classical | 0.9419305241627208 | results/classical/classification_reports/logistic_regression_report.json; results/classical/confusion_matrices/logistic_regression_confusion_matrix.png |
| linear_svm | classical | 0.9533659315308862 | results/classical/classification_reports/linear_svm_report.json; results/classical/confusion_matrices/linear_svm_confusion_matrix.png |
| multinomial_nb | classical | 0.9112577337736262 | results/classical/classification_reports/multinomial_nb_report.json; results/classical/confusion_matrices/multinomial_nb_confusion_matrix.png |
| distilbert-base-uncased | transformer | 0.9544062486712525 | results/transformer/classification_reports/distilbert_base_uncased_report.json; results/transformer/confusion_matrices/distilbert_base_uncased_confusion_matrix.png |
| qwen_zero_shot | llm_prompting | 0.5039488016309485 | results/qwen/classification_reports/qwen_zero_shot_report.json; results/qwen/confusion_matrices/qwen_zero_shot_confusion_matrix.png |

## Failed or Skipped Models
| model_name | status | reason_if_skipped_or_failed |
| --- | --- | --- |
| qwen_few_shot | failed | All Qwen few-shot outputs were invalid; do not treat as a meaningful completed result. |

## Configured but Not Run
| model_name | model_family | reason_if_skipped_or_failed |
| --- | --- | --- |
| bilstm | neural_sequence | Configured in modules/sequence_model.py; run required. |
| bert-base-uncased | fine_tuned_encoder | Runnable configuration pending; no completed evidence found. |
| nlpaueb/legal-bert-base-uncased | fine_tuned_encoder | Runnable configuration pending; no completed evidence found. |
| nlpaueb/bert-base-uncased-contracts | fine_tuned_encoder | Runnable configuration pending; no completed evidence found. |
| roberta-base | fine_tuned_encoder | Runnable configuration pending; no completed evidence found. |
| microsoft/deberta-v3-base | fine_tuned_encoder | Runnable configuration pending; no completed evidence found. |
| qwen_retrieval_few_shot | llm_prompting | Configured in modules/qwen_prompting.py; rerun Qwen required. |

## Protocol Check
- Main dataset: LEDGAR.
- Top-k labels are selected from training split only according to dataset verification.
- Validation is used for classical model selection and DistilBERT checkpoint selection.
- Test rows are used for final evaluation in canonical completed rows.
- Qwen few-shot is not valid as a meaningful completed result because existing invalid-output rate is 1.0.

## Validation Commands Used
- `python scripts/build_coursework_artifacts.py`
- `python -m py_compile modules/*.py scripts/*.py`
