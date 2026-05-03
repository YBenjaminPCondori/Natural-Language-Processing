# Repository Audit

Created: 2026-05-03T16:44:26.389916+00:00

## Inventory
- Total files inspected: 310
- Notebooks: 1
- Python modules: 22
- Output files: 100
- Result files: 85

## Latest Files
| path | size_bytes | modified_utc |
| --- | --- | --- |
| outputs\error_analysis_summary.md | 511 | 2026-05-03T16:44:25.096754+00:00 |
| outputs\error_analysis_examples.csv | 12145 | 2026-05-03T16:44:25.085210+00:00 |
| outputs\per_label_f1_best_model.csv | 2 | 2026-05-03T16:44:25.069194+00:00 |
| outputs\per_label_f1_all_models.csv | 44298 | 2026-05-03T16:44:25.057677+00:00 |
| outputs\qwen_prompt_audit.md | 454 | 2026-05-03T16:44:25.043128+00:00 |
| outputs\llm_prompt_results.csv | 577 | 2026-05-03T16:44:25.031620+00:00 |
| outputs\qwen_retrieval_few_shot_predictions.csv | 83 | 2026-05-03T16:44:25.027617+00:00 |
| outputs\qwen_static_few_shot_predictions.csv | 158985 | 2026-05-03T16:44:25.024105+00:00 |
| outputs\qwen_zero_shot_predictions.csv | 141367 | 2026-05-03T16:44:25.011593+00:00 |
| outputs\transformer_training_summary.md | 1066 | 2026-05-03T16:44:24.987056+00:00 |
| outputs\best_transformer_configs.json | 659 | 2026-05-03T16:44:24.975478+00:00 |
| outputs\hyperparameter_search_results.csv | 895 | 2026-05-03T16:44:24.962970+00:00 |
| outputs\best_classical_configs.json | 606 | 2026-05-03T16:44:24.955642+00:00 |
| outputs\classical_hyperparameter_results.csv | 1158 | 2026-05-03T16:44:24.943126+00:00 |
| outputs\model_evidence_table.md | 1818 | 2026-05-03T16:44:24.939125+00:00 |
| outputs\main_results_latex.tex | 4334 | 2026-05-03T16:44:24.926102+00:00 |
| outputs\main_results.json | 11272 | 2026-05-03T16:44:24.835490+00:00 |
| outputs\main_results.csv | 3155 | 2026-05-03T16:44:24.822984+00:00 |
| outputs\dataset_verification.md | 535 | 2026-05-03T16:44:24.783530+00:00 |
| figures\clause_length_distribution.png | 37464 | 2026-05-03T16:44:24.770020+00:00 |
| figures\text_length_distribution.png | 37464 | 2026-05-03T16:44:24.648672+00:00 |
| figures\label_distribution.png | 89676 | 2026-05-03T16:44:24.455309+00:00 |
| figures\label_distribution_top20.png | 89676 | 2026-05-03T16:44:24.229688+00:00 |
| outputs\text_length_statistics.csv | 263 | 2026-05-03T16:44:23.698207+00:00 |
| outputs\label_distribution_top20.csv | 679 | 2026-05-03T16:44:23.692691+00:00 |
| outputs\dataset_verification.json | 1684 | 2026-05-03T16:44:23.677640+00:00 |
| modules\__pycache__\coursework_artifacts.cpython-311.pyc | 54424 | 2026-05-03T16:44:21.674894+00:00 |
| modules\coursework_artifacts.py | 35235 | 2026-05-03T16:44:14.018234+00:00 |
| outputs\final_coursework_audit.md | 3466 | 2026-05-03T16:43:34.034988+00:00 |
| outputs\seed_stability_results.csv | 805 | 2026-05-03T16:43:34.031990+00:00 |

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
| modules.inference | failed | ModuleNotFoundError: No module named 'ledgar_pipeline' |
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
| modules.wandb_tracking | failed | ModuleNotFoundError: No module named 'ledgar_pipeline' |

## Shortcut / Cache Pattern Scan
| path | patterns |
| --- | --- |
| modules\agentic_review.py | sample_size, .sample( |
| modules\baselines.py | dummy |
| modules\coursework_artifacts.py | debug=True, sample_size, head(, .sample(, max_steps, TODO, placeholder, dummy, hardcoded, try:, except |
| modules\data_setup.py | .sample(, try:, except |
| modules\error_analysis.py | head( |
| modules\evaluation.py | sample_size |
| modules\preprocessing.py | head( |
| modules\qwen_prompting.py | sample_size, head(, .sample(, try:, except |
| modules\report_exports.py | sample_size, try:, except |
| modules\sequence_model.py | sample_size |
| modules\transformer_model.py | sample_size, try:, except |
| modules\wandb_reporting.py | head(, try:, except |
| notebooks\ledgar_clause_classification_pipeline.ipynb | sample_size, head(, TODO, placeholder, dummy, try:, except |

## Model Evidence Check
| model_name | status | evidence_path | evidence_exists | prediction_path | prediction_exists |
| --- | --- | --- | --- | --- | --- |
| random_uniform | skipped |  | True | outputs/predictions/random_uniform_test_predictions.jsonl | True |
| random_train_distribution | skipped |  | True | outputs/predictions/random_train_distribution_test_predictions.jsonl | True |
| majority_baseline | skipped |  | True | outputs/predictions/majority_baseline_test_predictions.jsonl | True |
| logistic_regression | skipped |  | True | outputs/predictions/logistic_regression_test_predictions.jsonl | True |
| linear_svm | skipped |  | True | outputs/predictions/linear_svm_test_predictions.jsonl | True |
| multinomial_nb | skipped |  | True | outputs/predictions/multinomial_nb_test_predictions.jsonl | True |
| distilbert-base-uncased | skipped |  | True | outputs/predictions/distilbert_base_uncased_test_predictions.jsonl | True |
| qwen_zero_shot | skipped |  | True | outputs/qwen_predictions.csv | True |
| qwen_few_shot | failed |  | True | outputs/qwen_predictions.csv | True |
| bilstm | skipped |  | True | results/qwen/qwen_predictions.csv | True |
| bert-base-uncased | skipped |  | True | results/qwen/qwen_predictions.csv | True |
| nlpaueb/legal-bert-base-uncased | skipped |  | True | results/qwen/qwen_predictions.csv | True |
| nlpaueb/bert-base-uncased-contracts | skipped |  | True | results/qwen/qwen_predictions.csv | True |
| roberta-base | skipped |  | True | results/qwen/qwen_predictions.csv | True |
| microsoft/deberta-v3-base | skipped |  | True | results/qwen/qwen_predictions.csv | True |
| qwen_retrieval_few_shot | skipped |  | True | outputs/qwen_predictions.csv | True |

## Remaining Reproducibility Risks
- Generated artifacts include Colab absolute paths; canonical tables also include repo-relative evidence paths where possible.
- Additional encoder models and BiLSTM are configured/pending unless their result artifacts are generated.
- Qwen few-shot existing artifact has invalid_prediction_rate=1.0 and is marked failed in canonical outputs.
