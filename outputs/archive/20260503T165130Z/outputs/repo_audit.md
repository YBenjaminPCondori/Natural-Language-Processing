# Repository Audit

Created: 2026-05-03T16:51:06.683274+00:00

## Inventory
- Total files inspected: 403
- Notebooks: 1
- Python modules: 22
- Output files: 199
- Result files: 85

## Latest Files
| path | size_bytes | modified_utc |
| --- | --- | --- |
| modules\__pycache__\wandb_tracking.cpython-311.pyc | 635 | 2026-05-03T16:51:06.598757+00:00 |
| modules\__pycache__\inference.cpython-311.pyc | 539 | 2026-05-03T16:51:06.583756+00:00 |
| outputs\error_analysis_summary.md | 511 | 2026-05-03T16:51:04.687990+00:00 |
| outputs\error_analysis_examples.csv | 12145 | 2026-05-03T16:51:04.684992+00:00 |
| figures\per_label_f1_best_model.png | 75517 | 2026-05-03T16:51:04.668484+00:00 |
| outputs\per_label_f1_best_model.csv | 5623 | 2026-05-03T16:51:04.423643+00:00 |
| outputs\per_label_f1_all_models.csv | 44298 | 2026-05-03T16:51:04.417637+00:00 |
| figures\validation_vs_test_macro_f1.png | 40346 | 2026-05-03T16:51:04.399638+00:00 |
| outputs\figures\model_comparison_accuracy.png | 83176 | 2026-05-03T16:51:04.216606+00:00 |
| figures\model_comparison_accuracy.png | 83176 | 2026-05-03T16:51:04.062799+00:00 |
| outputs\figures\model_comparison_macro_f1.png | 82147 | 2026-05-03T16:51:03.859779+00:00 |
| figures\model_comparison_macro_f1.png | 82147 | 2026-05-03T16:51:03.717257+00:00 |
| outputs\qwen_prompt_audit.md | 454 | 2026-05-03T16:51:03.523025+00:00 |
| outputs\llm_prompt_results.csv | 577 | 2026-05-03T16:51:03.519024+00:00 |
| outputs\qwen_retrieval_few_shot_predictions.csv | 83 | 2026-05-03T16:51:03.514023+00:00 |
| outputs\qwen_static_few_shot_predictions.csv | 158985 | 2026-05-03T16:51:03.510024+00:00 |
| outputs\qwen_zero_shot_predictions.csv | 141367 | 2026-05-03T16:51:03.479026+00:00 |
| outputs\transformer_training_summary.md | 1066 | 2026-05-03T16:51:03.448834+00:00 |
| outputs\best_transformer_configs.json | 659 | 2026-05-03T16:51:03.442836+00:00 |
| outputs\hyperparameter_search_results.csv | 895 | 2026-05-03T16:51:03.439836+00:00 |
| outputs\best_classical_configs.json | 606 | 2026-05-03T16:51:03.429834+00:00 |
| outputs\classical_hyperparameter_results.csv | 1158 | 2026-05-03T16:51:03.423834+00:00 |
| outputs\model_evidence_table.md | 3112 | 2026-05-03T16:51:03.416834+00:00 |
| outputs\main_results_latex.tex | 6034 | 2026-05-03T16:51:03.410834+00:00 |
| outputs\main_results.json | 13257 | 2026-05-03T16:51:03.290746+00:00 |
| outputs\main_results.csv | 5320 | 2026-05-03T16:51:03.276749+00:00 |
| outputs\dataset_verification.md | 535 | 2026-05-03T16:51:03.224231+00:00 |
| figures\clause_length_distribution.png | 37464 | 2026-05-03T16:51:03.211229+00:00 |
| figures\text_length_distribution.png | 37464 | 2026-05-03T16:51:03.081851+00:00 |
| figures\label_distribution.png | 89676 | 2026-05-03T16:51:02.875813+00:00 |

_Showing 30 of 80 rows._

## Module Import Check
| module | status | reason |
| --- | --- | --- |
| modules.__init__ | importable |  |
| modules.agentic_review | importable |  |
| modules.baselines | importable |  |
| modules.classical_models | importable |  |
| modules.config | importable |  |
| modules.coursework_artifacts | importable |  |
| modules.data_setup | importable |  |
| modules.error_analysis | importable |  |
| modules.evaluate | importable |  |
| modules.evaluation | importable |  |
| modules.inference | importable |  |
| modules.preprocess | importable |  |
| modules.preprocessing | importable |  |
| modules.qwen_prompting | importable |  |
| modules.report_exports | importable |  |
| modules.sequence_model | importable |  |
| modules.train | importable |  |
| modules.train_classical | importable |  |
| modules.train_transformer | importable |  |
| modules.transformer_model | importable |  |
| modules.wandb_reporting | importable |  |
| modules.wandb_tracking | importable |  |

## Shortcut / Cache Pattern Scan
| path | patterns |
| --- | --- |
| modules\agentic_review.py | sample_size, .sample( |
| modules\baselines.py | dummy |
| modules\coursework_artifacts.py | debug=True, sample_size, head(, .sample(, max_steps, TODO, placeholder, dummy, hardcoded, try:, except |
| modules\data_setup.py | .sample(, try:, except |
| modules\error_analysis.py | head( |
| modules\evaluation.py | sample_size |
| modules\inference.py | try:, except |
| modules\preprocessing.py | head( |
| modules\qwen_prompting.py | sample_size, head(, .sample(, try:, except |
| modules\report_exports.py | sample_size, dummy, try:, except |
| modules\sequence_model.py | sample_size |
| modules\transformer_model.py | sample_size, try:, except |
| modules\wandb_reporting.py | head(, try:, except |
| modules\wandb_tracking.py | try:, except |
| notebooks\ledgar_clause_classification_pipeline.ipynb | sample_size, head(, TODO, placeholder, dummy, try:, except |

## Model Evidence Check
| model_name | status | evidence_path | evidence_exists | prediction_path | prediction_exists |
| --- | --- | --- | --- | --- | --- |
| random_uniform | completed | results/baselines/classification_reports/random_uniform_report.json; results/baselines/confusion_matrices/random_uniform_confusion_matrix.png | True | outputs/predictions/random_uniform_test_predictions.jsonl | True |
| random_train_distribution | completed | results/baselines/classification_reports/random_train_distribution_report.json; results/baselines/confusion_matrices/random_train_distribution_confusion_matrix.png | True | outputs/predictions/random_train_distribution_test_predictions.jsonl | True |
| majority_baseline | completed | results/baselines/classification_reports/majority_baseline_report.json; results/baselines/confusion_matrices/majority_baseline_confusion_matrix.png | True | outputs/predictions/majority_baseline_test_predictions.jsonl | True |
| logistic_regression | completed | results/classical/classification_reports/logistic_regression_report.json; results/classical/confusion_matrices/logistic_regression_confusion_matrix.png | True | outputs/predictions/logistic_regression_test_predictions.jsonl | True |
| linear_svm | completed | results/classical/classification_reports/linear_svm_report.json; results/classical/confusion_matrices/linear_svm_confusion_matrix.png | True | outputs/predictions/linear_svm_test_predictions.jsonl | True |
| multinomial_nb | completed | results/classical/classification_reports/multinomial_nb_report.json; results/classical/confusion_matrices/multinomial_nb_confusion_matrix.png | True | outputs/predictions/multinomial_nb_test_predictions.jsonl | True |
| distilbert-base-uncased | completed | results/transformer/classification_reports/distilbert_base_uncased_report.json; results/transformer/confusion_matrices/distilbert_base_uncased_confusion_matrix.png | True | outputs/predictions/distilbert_base_uncased_test_predictions.jsonl | True |
| qwen_zero_shot | completed | results/qwen/classification_reports/qwen_zero_shot_report.json; results/qwen/confusion_matrices/qwen_zero_shot_confusion_matrix.png | True | outputs/qwen_predictions.csv | True |
| qwen_few_shot | failed | results/qwen/classification_reports/qwen_few_shot_report.json; results/qwen/confusion_matrices/qwen_few_shot_confusion_matrix.png | True | outputs/qwen_predictions.csv | True |
| bilstm | pending |  | True |  | False |
| bert-base-uncased | pending |  | True |  | False |
| nlpaueb/legal-bert-base-uncased | pending |  | True |  | False |
| nlpaueb/bert-base-uncased-contracts | pending |  | True |  | False |
| roberta-base | pending |  | True |  | False |
| microsoft/deberta-v3-base | pending |  | True |  | False |
| qwen_retrieval_few_shot | pending |  | True |  | False |

## Remaining Reproducibility Risks
- Generated artifacts include Colab absolute paths; canonical tables also include repo-relative evidence paths where possible.
- Additional encoder models and BiLSTM are configured/pending unless their result artifacts are generated.
- Qwen few-shot existing artifact has invalid_prediction_rate=1.0 and is marked failed in canonical outputs.
