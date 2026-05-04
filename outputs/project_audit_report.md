# Project Artifact Audit Against report.tex

Created: 2026-05-04T20:46:42.356145+00:00
Project root: `C:\Users\ybenj\Documents\GitHub\Education\INM434 Natural-Language-Processing\Natural-Language-Processing`
Report: `C:\Users\ybenj\Documents\GitHub\Education\INM434 Natural-Language-Processing\Natural-Language-Processing\report.tex`
Ready for report.tex: `NO`
Overall readiness score: **8.4/10**

## Pipeline Stage Status
| stage | status | summary | evidence |
| --- | --- | --- | --- |
| Dataset audit | PASS | Processed rows train/validation/test = 28587/4670/4732; labels=20; text+label overlaps={'train_vs_validation': 0, 'train_vs_test': 0, 'validation_vs_test': 0}; exact text leakage={'train_vs_validation': 0, 'train_vs_test': 0, 'validation_vs_test': 0}. | data/processed/ledgar_{train,validation,test}.jsonl; data/processed/label_names.txt |
| Required report artifacts | PASS | 19 PASS, 0 WARNING, 0 FAIL. | figures/ and outputs/ required by report.tex |
| Model evidence | WARNING | 3 PASS, 1 WARNING, 0 FAIL. | results/, outputs/predictions/, classification reports, confusion matrices |
| HPT methodology | FAIL | 13 PASS, 0 WARNING, 1 FAIL. | modules/transformer_hpt.py; scripts/run_transformer_hpt.py; outputs/sweep_results.csv |
| Evaluation artifacts | WARNING | 4 PASS, 1 WARNING, 0 FAIL. | outputs/main_results.csv; outputs/per_class_results.csv; error analysis outputs |
| Execution risks | WARNING | 35 risk rows found; inspect warnings before filling report.tex. | Static pattern scan of notebooks/modules/scripts plus runtime outputs |

## Required Report Artifacts
| path | status | reason | report_need | generator | close_existing_artifacts | schema_note |
| --- | --- | --- | --- | --- | --- | --- |
| figures/pipeline_overview.png | PASS | Required artifact exists and is non-empty. | Pipeline overview figure in report methodology section | Create from report artifact/export stage or a dedicated diagram generation helper. | [] |  |
| figures/label_distribution.png | PASS | Required artifact exists and is non-empty. | Dataset label-distribution figure | modules.preprocessing.create_ledgar_eda or modules.coursework_artifacts.generate_dataset_verification | ['figures/label_distribution_top20.png'] |  |
| figures/hpt_validation_macro_f1.png | PASS | Required artifact exists and is non-empty. | HPT validation macro-F1 figure | modules.transformer_hpt.run_two_stage_transformer_hpt or scripts/run_transformer_hpt.py | ['outputs/figures/hpt_validation_macro_f1.png'] |  |
| figures/clause_length_distribution.png | PASS | Required artifact exists and is non-empty. | Dataset clause-length figure used by report.tex | modules.coursework_artifacts.generate_dataset_verification or modules.report_exports.export_report_figures | ['figures/text_length_distribution.png', 'results/eda/clause_length_histogram.png'] |  |
| figures/model_comparison_macro_f1.png | PASS | Required artifact exists and is non-empty. | Main model comparison figure used by report.tex | modules.coursework_artifacts.generate_comparison_figures or modules.report_exports.export_report_figures | ['results/final_model_comparison.png', 'outputs/figures/model_comparison_macro_f1.png'] |  |
| figures/qwen_invalid_predictions.png | PASS | Required artifact exists and is non-empty. | Qwen invalid-output figure used by report.tex | modules.report_exports.plot_qwen_invalid_predictions or modules.coursework_artifacts.generate_qwen_invalid_figure | ['outputs/figures/qwen_invalid_predictions.png'] |  |
| figures/confusion_matrix_best_model.png | PASS | Required artifact exists and is non-empty. | Best supervised model confusion matrix figure used by report.tex | Final report export stage should copy the selected best-model confusion matrix to this canonical filename. | ['outputs/figures/confusion_matrix_best_model.png', 'results/transformer/confusion_matrices/distilbert_base_uncased_confusion_matrix.png'] |  |
| figures/agentic_review_workflow.png | PASS | Required artifact exists and is non-empty. | Confidence-review workflow figure used by report.tex | modules.report_exports.plot_agentic_workflow or modules.coursework_artifacts.generate_agentic_review_artifacts | ['outputs/figures/agentic_review_workflow.png'] |  |
| outputs/main_results.csv | PASS | Required artifact exists and is non-empty. | Report main-results table source | modules.report_exports.export_main_results or modules.coursework_artifacts.canonical_results | ['results/final_model_comparison.csv'] | columns=['model_name', 'model_family', 'training_type', 'split_used', 'validation_accuracy', 'validation_macro_f1', 'test_accuracy', 'test_macro_f1', 'test_weighted_f1', 'test_macro_precision', 'test_macro_recall', 'invalid_prediction_rate', 'selected_by_validation', 'hyperparameter_source', 'evidence_path', 'prediction_path', 'status', 'reason_if_skipped_or_failed'] rows=16 |
| outputs/sweep_results.csv | PASS | Required artifact exists and is non-empty. | Stage 5A/5B HPT sweep table | modules.transformer_hpt.run_two_stage_transformer_hpt should export this canonical report filename. | ['outputs/hyperparameter_search_results.csv'] | columns=['model_name', 'status', 'validation_macro_f1', 'test_macro_f1', 'config_path', 'reason'] rows=6 |
| outputs/best_hyperparameters.json | PASS | Required artifact exists and is non-empty. | Best transformer HPT configuration | modules.transformer_hpt.run_two_stage_transformer_hpt should export this canonical report filename. | ['outputs/best_transformer_configs.json', 'results/transformer/training_args.json'] | type=dict |
| outputs/final_test_metrics.json | PASS | Required artifact exists and is non-empty. | Final held-out test metrics table/text | Final evaluation/export stage should write selected final test metrics to JSON. | ['outputs/main_results.json', 'results/final_model_comparison.csv', 'results/transformer/transformer_results.csv'] | type=dict |
| outputs/final_test_predictions.csv | PASS | Required artifact exists and is non-empty. | Final selected model prediction table | Final evaluation/export stage should write selected final model predictions to CSV. | ['results/transformer/transformer_predictions.csv', 'outputs/transformer_predictions.csv'] | columns=['text', 'true_label', 'true_label_id', 'predicted_label', 'predicted_label_id', 'model_name', 'split', 'confidence'] rows=4732 |
| outputs/per_class_metrics.csv | PASS | Required artifact exists and is non-empty. | Per-class precision/recall/F1 table | Error analysis or report export stage should write per-class metrics to this canonical filename. | ['outputs/per_class_results.csv', 'outputs/per_label_f1_all_models.csv'] | columns=['model_family', 'model_name', 'label', 'precision', 'recall', 'f1_score', 'support', 'classification_report_path', 'f1_rank_within_model'] rows=180 |
| outputs/error_analysis_examples.csv | PASS | Required artifact exists and is non-empty. | Error analysis example table | modules.coursework_artifacts.generate_error_artifacts | ['outputs/misclassified_examples.csv'] | columns=['text', 'label', 'label_id', 'predicted_label_id', 'predicted_label', 'text_short', 'analysis_note', 'likely_reason'] rows=10 |
| outputs/failed_or_skipped_trials.csv | PASS | Required artifact exists and is non-empty. | Failed/skipped HPT/model trial log | HPT/final audit stage should export failed and skipped trial rows. | ['outputs/main_results.csv', 'outputs/hyperparameter_search_results.csv'] | columns=['model_name', 'model_family', 'training_type', 'split_used', 'validation_accuracy', 'validation_macro_f1', 'test_accuracy', 'test_macro_f1', 'test_weighted_f1', 'test_macro_precision', 'test_macro_recall', 'invalid_prediction_rate', 'selected_by_validation', 'hyperparameter_source', 'evidence_path', 'prediction_path', 'status', 'reason_if_skipped_or_failed', 'config_path', 'reason'] rows=13 |
| outputs/hpt_summary.csv | PASS | Required artifact exists and is non-empty. | HPT summary values for Table 5 in report.tex | modules.report_exports.export_hpt_report_aliases or modules.coursework_artifacts.generate_tuning_artifacts | ['outputs/sweep_results.csv'] | columns=['stage', 'search_method', 'trials', 'completed_trials', 'best_validation_macro_f1'] rows=6 |
| outputs/preprocessing_techniques.md | PASS | Required artifact exists and is non-empty. | Preprocessing rundown referenced by report.tex | modules.preprocessing.write_preprocessing_rundown | [] |  |
| outputs/metrics/final_model_comparison_summary.csv | PASS | Required artifact exists and is non-empty. | Combined LEDGAR-test and CUAD-external summary referenced by report.tex | modules.cuad_external.build_final_model_comparison_summary or report artifact adapter | ['outputs/main_results.csv'] | columns=['model_name', 'dataset_name', 'accuracy', 'macro_f1', 'weighted_f1', 'num_samples', 'num_labels', 'notes'] rows=15 |

## Missing Artifacts
_No rows._

## Suspicious / Noncanonical Artifacts
_No rows._

## Model Audit
| model | status | trains_or_runs | validates | metrics_saved | predictions_exported | prediction_rows | real_not_placeholder | evidence | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TF-IDF + Logistic Regression | PASS | True | True | True | True | 4732 | True | results/classical/classical_results.csv; outputs/predictions/logistic_regression_test_predictions.jsonl; results/classical/classification_reports/logistic_regression_report.json; results/classical/confusion_matrices/logistic_regression_confusion_matrix.png |  |
| TF-IDF + Linear SVM | PASS | True | True | True | True | 4732 | True | results/classical/classical_results.csv; outputs/predictions/linear_svm_test_predictions.jsonl; results/classical/classification_reports/linear_svm_report.json; results/classical/confusion_matrices/linear_svm_confusion_matrix.png |  |
| Main transformer/BERT model | PASS | True | True | True | True | 4732 | True | results/transformer/transformer_results.csv; results/transformer/transformer_predictions.csv; results/transformer/classification_reports/distilbert_base_uncased_report.json; results/transformer/confusion_matrices/distilbert_base_uncased_confusion_matrix.png |  |
| Optional instruction-tuned LLM classifier | WARNING | True | False | True | True | 400 | True | outputs/qwen_results.csv; outputs/qwen_predictions.csv; results/qwen/classification_reports/qwen_zero_shot_report.json; results/qwen/confusion_matrices/qwen_zero_shot_confusion_matrix.png | Qwen few-shot artifact exists but is invalid under recorded parser. |

## HPT Audit
| check | status | evidence | fix |
| --- | --- | --- | --- |
| HPT pipeline module/script exists | PASS | modules/transformer_hpt.py; scripts/run_transformer_hpt.py | Add or restore the reusable HPT runner. |
| Stage 5A random-search runner exists | PASS | Report describes Stage 5A as random search with optional W&B logging. | Keep Stage 5A named and exported as stage5a_random_trial. |
| Stage 5A default is 8 trials | PASS | report.tex states Stage 5A is configurable with default 8. | Set TransformerHPTConfig.random_trials to 8 or update report.tex. |
| Stage 5A uses 2 epochs per trial | PASS | report.tex states Stage 5A uses 2 epochs per trial. | Keep DEFAULT_STAGE5A_SEARCH_SPACE['epochs'] fixed to [2]. |
| Validation macro-F1 is the optimisation target | PASS | HPT/training code references validation_macro_f1 and macro_f1 selection. | Keep validation macro-F1 as the objective and export it in sweep rows. |
| Stage 5B Bayesian/Optuna runner exists | PASS | report.tex describes Bayesian optimisation; the implementation uses Optuna with optional W&B logging. | Keep Stage 5B exported as stage5b_bayes_trial and document Optuna-based Bayesian optimisation. |
| Stage 5B default is 8 trials | PASS | report.tex states Stage 5B is configurable with default 8. | Set TransformerHPTConfig.bayes_trials to 8 or update report.tex. |
| Stage 5B uses 3 epochs per trial | PASS | report.tex states Stage 5B uses 3 epochs per trial. | Keep DEFAULT_STAGE5B_SEARCH_SPACE['epochs'] fixed to [3]. |
| Stage 5B uses narrowed search space | PASS | report.tex describes a narrowed Bayesian search space. | Use DEFAULT_STAGE5B_SEARCH_SPACE for Stage 5B. |
| Best hyperparameters exported to required filename | PASS | outputs/best_hyperparameters.json missing; outputs/best_transformer_configs.json may exist. | Write the selected HPT config to outputs/best_hyperparameters.json. |
| Final retraining uses 4 epochs | PASS | Appendix A states the final transformer uses 4 epochs after validation selection. | Keep final_retrain_epochs at 4 for non-smoke final retrain. |
| Final test evaluation occurs once after tuning | PASS | HPT trial code disables test evaluation and final retrain enables it; runtime evidence still required. | Keep test evaluation disabled during trials and verify final output timestamps after rerun. |
| Completed HPT run evidence exists | FAIL | Found HPT run files: 0; outputs/hyperparameter_search_results rows=6. | Run when training is intended: python scripts/run_transformer_hpt.py --model-name distilbert-base-uncased --random-trials 8 --bayes-trials 8 --wandb |
| Canonical sweep_results.csv exists | PASS | outputs/sweep_results.csv is required by report.tex but missing. | Export HPT trial rows to outputs/sweep_results.csv. |

## Evaluation Audit
| check | status | evidence | fix |
| --- | --- | --- | --- |
| Accuracy, macro-F1, weighted-F1 produced | PASS | outputs/main_results.csv | Regenerate final model comparison and report exports. |
| Per-class precision/recall/F1 produced | PASS | outputs/per_class_results.csv | Export canonical outputs/per_class_metrics.csv. |
| Confusion matrix produced | PASS | figures/confusion_matrix_best_model.png | Copy or regenerate best-model confusion matrix as figures/confusion_matrix_best_model.png. |
| Prediction confidence exported when supported | WARNING | Prediction files do not expose confidence/score columns for the final transformer. | Add confidence/margin output for models that support it, or state confidence is unavailable. |
| Error analysis examples produced | PASS | outputs/error_analysis_examples.csv | Run error analysis/report artifact stage. |

## Execution / Silent-Skip Audit
| path | status | risk_patterns | note |
| --- | --- | --- | --- |
| modules/agentic_review.py | WARNING | .sample(, sample_size, skip | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/baselines.py | WARNING | debug, dummy, sample_size, skip | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/classical_models.py | WARNING | debug, skip | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/coursework_artifacts.py | WARNING | .sample(, cuda, debug, dummy, except, hardcoded, head(, max_steps, placeholder, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/cuad_external.py | WARNING | .sample(, cuda, except, head(, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/data_setup.py | WARNING | .sample(, cuda, except, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/error_analysis.py | WARNING | head( | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/evaluation.py | WARNING | .sample(, debug, sample_size | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/inference.py | WARNING | except, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/llm_evaluation.py | WARNING | .sample(, cuda, except, head(, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/preprocessing.py | WARNING | except, head(, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/qwen_prompting.py | WARNING | .sample(, cuda, debug, except, head(, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/report_artifact_audit.py | WARNING | .sample(, cuda, debug, dummy, except, hardcoded, head(, if file exists, load_from_cache_file, max_steps, oom, placeholder, resume_from_checkpoint, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/report_exports.py | WARNING | cuda, dummy, except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/sequence_model.py | WARNING | cuda, sample_size, skip | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/transformer_hpt.py | WARNING | cuda, except, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/transformer_model.py | WARNING | cuda, debug, except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/wandb_reporting.py | WARNING | except, head(, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/wandb_tracking.py | WARNING | except, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| scripts/run_cuad_external_eval.py | WARNING | debug, skip | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| scripts/run_qwen_prompting.py | WARNING | cuda, sample_size | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| scripts/run_transformer_model.py | WARNING | cuda, debug, skip | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| scripts/split_notebook_by_stage.py | WARNING | dummy | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/ledgar_clause_classification_pipeline.ipynb | WARNING | cuda, dummy, except, head(, placeholder, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/00_setup_and_config.ipynb | WARNING | cuda, except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/01_dataset_preprocessing_eda.ipynb | WARNING | cuda, except, head(, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/02_dummy_baselines.ipynb | WARNING | cuda, dummy, except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/03_classical_tfidf_models.ipynb | WARNING | cuda, except, head(, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/04_neural_sequence_bilstm.ipynb | WARNING | cuda, except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/05_transformer_finetuning.ipynb | WARNING | cuda, except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/06_qwen_prompting.ipynb | WARNING | cuda, except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/07_agentic_review.ipynb | WARNING | cuda, except, head(, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/09_final_artifact_audit.ipynb | WARNING | except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| outputs/qwen_results.csv | WARNING | reduced sample mode | Qwen evaluated on max sample_size=200; full processed test split has 4732 rows. |
| results/transformer/runtime.json | PASS | cuda | CUDA available in recorded runtime: True; GPU=NVIDIA A100-SXM4-40GB |

## report.tex References
- Figures referenced: `8`
- TODO lines found: `37`
- Artifact-like path mentions found: `34`

### Figure References
| line | path |
| --- | --- |
| 83 | figures/pipeline_overview.png |
| 151 | figures/label_distribution.png |
| 158 | figures/clause_length_distribution.png |
| 432 | figures/hpt_validation_macro_f1.png |
| 461 | figures/agentic_review_workflow.png |
| 551 | figures/model_comparison_macro_f1.png |
| 564 | figures/qwen_invalid_predictions.png |
| 628 | figures/confusion_matrix_best_model.png |

### TODO Lines
| line | text |
| --- | --- |
| 21 | \newcommand{\todo}[1]{\textcolor{red}{[TODO: #1]}} |
| 239 | Dataset files found & \todo{Pass/Fail} \\ |
| 240 | Train/validation/test splits non-empty & \todo{Pass/Fail} \\ |
| 241 | Top-20 labels selected from train only & \todo{Pass/Fail} \\ |
| 242 | Duplicate text-label pairs checked & \todo{Pass/Fail} \\ |
| 243 | Cross-split duplicate text checked & \todo{Pass/Fail} \\ |
| 244 | Label distribution exported & \todo{Pass/Fail} \\ |
| 245 | Prediction files exported & \todo{Pass/Fail} \\ |
| 246 | Failed/skipped trials logged & \todo{Pass/Fail} \\ |
| 341 | DistilBERT is the safest option if GPU resources are limited. LegalBERT or a contract-specific BERT model is more domain-aligned if computational resources allow. The final transformer used in the experiment was \todo{insert final transformer model used}. |
| 375 | Allowed labels: \todo{insert top-20 label list} |
| 377 | Clause: \todo{insert example clause text} |
| 422 | Stage 5A & Random Search & configurable, default 8 & \todo{value from outputs/hpt\_summary.csv} \\ |
| 423 | Stage 5B & Bayesian Optimisation & configurable, default 8 & \todo{value from outputs/hpt\_summary.csv} \\ |
| 605 | \todo{Class 1} & \todo{value} & \todo{value} & \todo{value} \\ |
| 606 | \todo{Class 2} & \todo{value} & \todo{value} & \todo{value} \\ |
| 607 | \todo{Class 3} & \todo{value} & \todo{value} & \todo{value} \\ |
| 608 | \todo{Class 4} & \todo{value} & \todo{value} & \todo{value} \\ |
| 609 | \todo{Class 5} & \todo{value} & \todo{value} & \todo{value} \\ |
| 616 | The strongest categories are \todo{insert categories from per-class results}. These categories may contain distinctive legal phrases or repeated boilerplate wording. The weakest categories are \todo{insert categories from per-class results}. These may be harder because of class imbalance, semantic overlap, or ambiguous clause wording. |
| 636 | \item \todo{Label A} and \todo{Label B}; |
| 637 | \item \todo{Label C} and \todo{Label D}; |
| 638 | \item \todo{Label E} and \todo{Label F}. |
| 670 | \todo{shortened clause} & \todo{label} & \todo{label} & \todo{reason} \\ |
| 671 | \todo{shortened clause} & \todo{label} & \todo{label} & \todo{reason} \\ |
| 672 | \todo{shortened clause} & \todo{label} & \todo{label} & \todo{reason} \\ |
| 691 | The system flags predictions for review when confidence is below \todo{threshold, if implemented}. For linear models, confidence may be estimated using probability scores or decision margins. For the transformer model, softmax confidence is used. Any LLM-generated supporting phrase is treated only as an explanatory aid and is not legally authoritative. |
| 700 | \todo{short clause} & \todo{label} & \todo{0.xx} & Yes \\ |
| 701 | \todo{short clause} & \todo{label} & \todo{0.xx} & No \\ |
| 702 | \todo{short clause} & \todo{label} & \todo{0.xx} & Yes \\ |
| 810 | TF-IDF & max features & \todo{e.g. 20k, 50k, 100k} \\ |
| 812 | Logistic Regression / SVM & C & \todo{search values} \\ |
| 813 | Naive Bayes & alpha & \todo{search values} \\ |
| 827 | Qwen2.5 & decoding parameters & \todo{temperature/top-p/max tokens if used} \\ |
| 859 | \todo{shortened clause} & \todo{label} & \todo{label} & \todo{reason} \\ |
| 860 | \todo{shortened clause} & \todo{label} & \todo{label} & \todo{reason} \\ |
| 861 | \todo{shortened clause} & \todo{label} & \todo{label} & \todo{reason} \\ |

## Required Fixes
| priority | what | why | where | action |
| --- | --- | --- | --- | --- |
| critical | Completed HPT run evidence exists | Required by updated report.tex HPT methodology. | modules/transformer_hpt.py; scripts/run_transformer_hpt.py; Stage 5 notebook | Run when training is intended: python scripts/run_transformer_hpt.py --model-name distilbert-base-uncased --random-trials 8 --bayes-trials 8 --wandb |
| important | Prediction confidence exported when supported | Required by evaluation/report artifact audit. | modules/evaluation.py; modules/report_exports.py; modules/coursework_artifacts.py | Add confidence/margin output for models that support it, or state confidence is unavailable. |

## Commands To Run When Evidence Is Missing
```bash
# Rebuild canonical report audit only
python scripts/run_report_artifact_audit.py

# Generate current report-facing derived artifacts from existing evidence
python scripts/build_coursework_artifacts.py

# Run the updated transformer HPT on CUDA/A100
python scripts/run_transformer_hpt.py --model-name distilbert-base-uncased --random-trials 20 --bayes-trials 15 --wandb
```
