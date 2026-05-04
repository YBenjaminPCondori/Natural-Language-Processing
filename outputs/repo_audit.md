# Repository Audit

Created: 2026-05-04T20:46:48.293671+00:00

## Inventory
- Total files inspected: 629
- Notebooks: 1
- Python modules: 26
- Output files: 384
- Result files: 85

## Latest Files
| path | size_bytes | modified_utc |
| --- | --- | --- |
| outputs\report_tex_values.json | 2423 | 2026-05-04T20:46:47.688674+00:00 |
| outputs\report_tables\hyperparameter_report_rows.tex | 154 | 2026-05-04T20:46:47.647673+00:00 |
| outputs\report_tables\misclassification_rows.tex | 429 | 2026-05-04T20:46:47.640675+00:00 |
| outputs\report_tables\per_class_rows.tex | 226 | 2026-05-04T20:46:47.625247+00:00 |
| outputs\report_tables\main_results_rows.tex | 445 | 2026-05-04T20:46:47.619246+00:00 |
| outputs\report_tables\hpt_summary_rows.tex | 309 | 2026-05-04T20:46:47.616247+00:00 |
| outputs\report_tables\audit_checklist_rows.tex | 495 | 2026-05-04T20:46:47.611239+00:00 |
| outputs\metrics\final_model_comparison_summary.csv | 3937 | 2026-05-04T20:46:47.601239+00:00 |
| outputs\failed_or_skipped_trials.csv | 2858 | 2026-05-04T20:46:47.593241+00:00 |
| outputs\final_test_metrics.json | 1214 | 2026-05-04T20:46:47.587241+00:00 |
| outputs\final_test_predictions.csv | 3268323 | 2026-05-04T20:46:47.582237+00:00 |
| figures\qwen_invalid_predictions.png | 31734 | 2026-05-04T20:46:47.351772+00:00 |
| outputs\figures\qwen_invalid_predictions.png | 31734 | 2026-05-04T20:46:47.351772+00:00 |
| figures\agentic_review_workflow.png | 21251 | 2026-05-04T20:46:47.190366+00:00 |
| outputs\figures\agentic_review_workflow.png | 21251 | 2026-05-04T20:46:47.190366+00:00 |
| figures\pipeline_overview.png | 53046 | 2026-05-04T20:46:47.073843+00:00 |
| outputs\figures\pipeline_overview.png | 53046 | 2026-05-04T20:46:47.073843+00:00 |
| outputs\error_analysis_summary.md | 511 | 2026-05-04T20:46:45.803160+00:00 |
| outputs\error_analysis_examples.csv | 12145 | 2026-05-04T20:46:45.799057+00:00 |
| figures\per_label_f1_best_model.png | 75517 | 2026-05-04T20:46:45.791986+00:00 |
| outputs\per_label_f1_best_model.csv | 5623 | 2026-05-04T20:46:45.508962+00:00 |
| outputs\per_label_f1_all_models.csv | 44298 | 2026-05-04T20:46:45.504958+00:00 |
| figures\validation_vs_test_macro_f1.png | 40346 | 2026-05-04T20:46:45.488922+00:00 |
| outputs\figures\model_comparison_accuracy.png | 83176 | 2026-05-04T20:46:45.278594+00:00 |
| figures\model_comparison_accuracy.png | 83176 | 2026-05-04T20:46:45.103487+00:00 |
| outputs\archive\20260504T204641Z\figures\model_comparison_macro_f1.png | 82147 | 2026-05-04T20:46:44.871177+00:00 |
| figures\model_comparison_macro_f1.png | 82147 | 2026-05-04T20:46:44.731866+00:00 |
| outputs\figures\model_comparison_macro_f1.png | 82147 | 2026-05-04T20:46:44.731866+00:00 |
| outputs\qwen_prompt_audit.md | 454 | 2026-05-04T20:46:44.529049+00:00 |
| outputs\llm_prompt_results.csv | 577 | 2026-05-04T20:46:44.526048+00:00 |

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
| modules.cuad_external | importable |  |
| modules.data_setup | importable |  |
| modules.error_analysis | importable |  |
| modules.evaluate | importable |  |
| modules.evaluation | importable |  |
| modules.inference | importable |  |
| modules.llm_evaluation | importable |  |
| modules.preprocess | importable |  |
| modules.preprocessing | importable |  |
| modules.qwen_prompting | importable |  |
| modules.report_artifact_audit | importable |  |
| modules.report_exports | importable |  |
| modules.sequence_model | importable |  |
| modules.train | importable |  |
| modules.train_classical | importable |  |
| modules.train_transformer | importable |  |
| modules.transformer_hpt | importable |  |
| modules.transformer_model | importable |  |
| modules.wandb_reporting | importable |  |
| modules.wandb_tracking | importable |  |

## Shortcut / Cache Pattern Scan
| path | patterns |
| --- | --- |
| modules\agentic_review.py | sample_size, .sample( |
| modules\baselines.py | sample_size, dummy |
| modules\coursework_artifacts.py | debug=True, sample_size, head(, .sample(, max_steps, TODO, placeholder, dummy, hardcoded, try:, except |
| modules\cuad_external.py | sample_size, head(, .sample(, try:, except |
| modules\data_setup.py | .sample(, try:, except |
| modules\error_analysis.py | head( |
| modules\evaluation.py | sample_size, .sample( |
| modules\inference.py | try:, except |
| modules\llm_evaluation.py | head(, .sample(, try:, except |
| modules\preprocessing.py | head(, try:, except |
| modules\qwen_prompting.py | sample_size, head(, .sample(, try:, except |
| modules\report_artifact_audit.py | sample_size, head(, .sample(, max_steps, TODO, placeholder, dummy, hardcoded, try:, except |
| modules\report_exports.py | sample_size, dummy, try:, except |
| modules\sequence_model.py | sample_size |
| modules\transformer_hpt.py | try:, except |
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
| bilstm | pending |  | False |  | False |
| bert-base-uncased | pending |  | False |  | False |
| nlpaueb/legal-bert-base-uncased | pending |  | False |  | False |
| nlpaueb/bert-base-uncased-contracts | pending |  | False |  | False |
| roberta-base | pending |  | False |  | False |
| microsoft/deberta-v3-base | pending |  | False |  | False |
| qwen_retrieval_few_shot | pending |  | False |  | False |

## Remaining Reproducibility Risks
- Generated artifacts include Colab absolute paths; canonical tables also include repo-relative evidence paths where possible.
- Additional encoder models and BiLSTM are configured/pending unless their result artifacts are generated.
- Qwen few-shot existing artifact has invalid_prediction_rate=1.0 and is marked failed in canonical outputs.
