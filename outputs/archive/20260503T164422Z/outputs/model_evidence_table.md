| model_name | status | evidence_path | prediction_path | reason_if_skipped_or_failed |
| --- | --- | --- | --- | --- |
| random_uniform | completed | results/baselines/classification_reports/random_uniform_report.json; results/baselines/confusion_matrices/random_uniform_confusion_matrix.png | outputs\predictions\random_uniform_test_predictions.jsonl |  |
| random_train_distribution | completed | results/baselines/classification_reports/random_train_distribution_report.json; results/baselines/confusion_matrices/random_train_distribution_confusion_matrix.png | outputs\predictions\random_train_distribution_test_predictions.jsonl |  |
| majority_baseline | completed | results/baselines/classification_reports/majority_baseline_report.json; results/baselines/confusion_matrices/majority_baseline_confusion_matrix.png | outputs\predictions\majority_baseline_test_predictions.jsonl |  |
| logistic_regression | completed | results/classical/classification_reports/logistic_regression_report.json; results/classical/confusion_matrices/logistic_regression_confusion_matrix.png | outputs\predictions\logistic_regression_test_predictions.jsonl |  |
| linear_svm | completed | results/classical/classification_reports/linear_svm_report.json; results/classical/confusion_matrices/linear_svm_confusion_matrix.png | outputs\predictions\linear_svm_test_predictions.jsonl |  |
| multinomial_nb | completed | results/classical/classification_reports/multinomial_nb_report.json; results/classical/confusion_matrices/multinomial_nb_confusion_matrix.png | outputs\predictions\multinomial_nb_test_predictions.jsonl |  |
| distilbert-base-uncased | completed | results/transformer/classification_reports/distilbert_base_uncased_report.json; results/transformer/confusion_matrices/distilbert_base_uncased_confusion_matrix.png | outputs\predictions\distilbert_base_uncased_test_predictions.jsonl |  |
| qwen_zero_shot | completed | results/qwen/classification_reports/qwen_zero_shot_report.json; results/qwen/confusion_matrices/qwen_zero_shot_confusion_matrix.png | outputs\qwen_predictions.csv |  |
| qwen_few_shot | failed | results/qwen/classification_reports/qwen_few_shot_report.json; results/qwen/confusion_matrices/qwen_few_shot_confusion_matrix.png | outputs\qwen_predictions.csv | All Qwen few-shot outputs were invalid; do not treat as a meaningful completed result. |
| bilstm | pending |  |  | Configured in modules/sequence_model.py; run required. |
| bert-base-uncased | pending |  |  | Runnable configuration pending; no completed evidence found. |
| nlpaueb/legal-bert-base-uncased | pending |  |  | Runnable configuration pending; no completed evidence found. |
| nlpaueb/bert-base-uncased-contracts | pending |  |  | Runnable configuration pending; no completed evidence found. |
| roberta-base | pending |  |  | Runnable configuration pending; no completed evidence found. |
| microsoft/deberta-v3-base | pending |  |  | Runnable configuration pending; no completed evidence found. |
| qwen_retrieval_few_shot | pending |  |  | Configured in modules/qwen_prompting.py; rerun Qwen required. |