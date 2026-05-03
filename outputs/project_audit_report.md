# Project Artifact Audit Against report.tex

Created: 2026-05-03T19:28:20.088596+00:00
Project root: `C:\Users\ybenj\Documents\GitHub\Education\INM434 Natural-Language-Processing\Natural-Language-Processing`
Report: `C:\Users\ybenj\Documents\GitHub\Education\INM434 Natural-Language-Processing\Natural-Language-Processing\report.tex`
Ready for report.tex: `NO`
Overall readiness score: **3.1/10**

## Pipeline Stage Status
| stage | status | summary | evidence |
| --- | --- | --- | --- |
| Dataset audit | PASS | Processed rows train/validation/test = 28587/4670/4732; labels=20; text+label overlaps={'train_vs_validation': 0, 'train_vs_test': 0, 'validation_vs_test': 0}; exact text leakage={'train_vs_validation': 0, 'train_vs_test': 0, 'validation_vs_test': 0}. | data/processed/ledgar_{train,validation,test}.jsonl; data/processed/label_names.txt |
| Required report artifacts | FAIL | 2 PASS, 8 WARNING, 2 FAIL. | figures/ and outputs/ required by report.tex |
| Model evidence | WARNING | 3 PASS, 1 WARNING, 0 FAIL. | results/, outputs/predictions/, classification reports, confusion matrices |
| HPT methodology | FAIL | 3 PASS, 3 WARNING, 8 FAIL. | modules/transformer_hpt.py; scripts/run_transformer_hpt.py; outputs/sweep_results.csv |
| Evaluation artifacts | WARNING | 3 PASS, 2 WARNING, 0 FAIL. | outputs/main_results.csv; outputs/per_class_results.csv; error analysis outputs |
| Execution risks | WARNING | 32 risk rows found; inspect warnings before filling report.tex. | Static pattern scan of notebooks/modules/scripts plus runtime outputs |

## Required Report Artifacts
| path | status | reason | report_need | generator | close_existing_artifacts | schema_note |
| --- | --- | --- | --- | --- | --- | --- |
| figures/pipeline_overview.png | FAIL | Required artifact is missing. | Pipeline overview figure in report methodology section | Create from report artifact/export stage or a dedicated diagram generation helper. | [] |  |
| figures/label_distribution.png | PASS | Required artifact exists and is non-empty. | Dataset label-distribution figure | modules.preprocessing.create_ledgar_eda or modules.coursework_artifacts.generate_dataset_verification | ['figures/label_distribution_top20.png'] |  |
| figures/hpt_validation_macro_f1.png | FAIL | Required artifact is missing. | HPT validation macro-F1 figure | modules.transformer_hpt.run_two_stage_transformer_hpt or scripts/run_transformer_hpt.py | [] |  |
| figures/confusion_matrix.png | WARNING | Required canonical filename is missing, but close noncanonical evidence exists. | Main confusion matrix figure | Final report export stage should copy the selected best-model confusion matrix to this canonical filename. | ['figures/confusion_matrix_best_model.png', 'outputs/figures/confusion_matrix_best_model.png'] |  |
| outputs/baseline_results.csv | WARNING | Required canonical filename is missing, but close noncanonical evidence exists. | Baseline model results table | modules.baselines.run_baseline_experiments should also export this canonical report filename. | ['results/baselines/baseline_results.csv'] |  |
| outputs/sweep_results.csv | WARNING | Required canonical filename is missing, but close noncanonical evidence exists. | Stage 5A/5B HPT sweep table | modules.transformer_hpt.run_two_stage_transformer_hpt should export this canonical report filename. | ['outputs/hyperparameter_search_results.csv'] |  |
| outputs/best_hyperparameters.json | WARNING | Required canonical filename is missing, but close noncanonical evidence exists. | Best transformer HPT configuration | modules.transformer_hpt.run_two_stage_transformer_hpt should export this canonical report filename. | ['outputs/best_transformer_configs.json', 'results/transformer/training_args.json'] |  |
| outputs/final_test_metrics.json | WARNING | Required canonical filename is missing, but close noncanonical evidence exists. | Final held-out test metrics table/text | Final evaluation/export stage should write selected final test metrics to JSON. | ['outputs/main_results.json', 'results/final_model_comparison.csv', 'results/transformer/transformer_results.csv'] |  |
| outputs/final_test_predictions.csv | WARNING | Required canonical filename is missing, but close noncanonical evidence exists. | Final selected model prediction table | Final evaluation/export stage should write selected final model predictions to CSV. | ['results/transformer/transformer_predictions.csv', 'outputs/transformer_predictions.csv'] |  |
| outputs/per_class_metrics.csv | WARNING | Required canonical filename is missing, but close noncanonical evidence exists. | Per-class precision/recall/F1 table | Error analysis or report export stage should write per-class metrics to this canonical filename. | ['outputs/per_class_results.csv', 'outputs/per_label_f1_all_models.csv'] |  |
| outputs/error_analysis_examples.csv | PASS | Required artifact exists and is non-empty. | Error analysis example table | modules.coursework_artifacts.generate_error_artifacts | ['outputs/misclassified_examples.csv'] | columns=['text', 'label', 'label_id', 'predicted_label_id', 'predicted_label', 'text_short', 'analysis_note', 'likely_reason'] rows=10 |
| outputs/failed_or_skipped_trials.csv | WARNING | Required canonical filename is missing, but close noncanonical evidence exists. | Failed/skipped HPT/model trial log | HPT/final audit stage should export failed and skipped trial rows. | ['outputs/main_results.csv', 'outputs/hyperparameter_search_results.csv'] |  |

## Missing Artifacts
| missing_filename | report_need | code_section_should_generate_it | close_existing_artifacts |
| --- | --- | --- | --- |
| figures/pipeline_overview.png | Pipeline overview figure in report methodology section | Create from report artifact/export stage or a dedicated diagram generation helper. | [] |
| figures/hpt_validation_macro_f1.png | HPT validation macro-F1 figure | modules.transformer_hpt.run_two_stage_transformer_hpt or scripts/run_transformer_hpt.py | [] |
| figures/confusion_matrix.png | Main confusion matrix figure | Final report export stage should copy the selected best-model confusion matrix to this canonical filename. | ['figures/confusion_matrix_best_model.png', 'outputs/figures/confusion_matrix_best_model.png'] |
| outputs/baseline_results.csv | Baseline model results table | modules.baselines.run_baseline_experiments should also export this canonical report filename. | ['results/baselines/baseline_results.csv'] |
| outputs/sweep_results.csv | Stage 5A/5B HPT sweep table | modules.transformer_hpt.run_two_stage_transformer_hpt should export this canonical report filename. | ['outputs/hyperparameter_search_results.csv'] |
| outputs/best_hyperparameters.json | Best transformer HPT configuration | modules.transformer_hpt.run_two_stage_transformer_hpt should export this canonical report filename. | ['outputs/best_transformer_configs.json', 'results/transformer/training_args.json'] |
| outputs/final_test_metrics.json | Final held-out test metrics table/text | Final evaluation/export stage should write selected final test metrics to JSON. | ['outputs/main_results.json', 'results/final_model_comparison.csv', 'results/transformer/transformer_results.csv'] |
| outputs/final_test_predictions.csv | Final selected model prediction table | Final evaluation/export stage should write selected final model predictions to CSV. | ['results/transformer/transformer_predictions.csv', 'outputs/transformer_predictions.csv'] |
| outputs/per_class_metrics.csv | Per-class precision/recall/F1 table | Error analysis or report export stage should write per-class metrics to this canonical filename. | ['outputs/per_class_results.csv', 'outputs/per_label_f1_all_models.csv'] |
| outputs/failed_or_skipped_trials.csv | Failed/skipped HPT/model trial log | HPT/final audit stage should export failed and skipped trial rows. | ['outputs/main_results.csv', 'outputs/hyperparameter_search_results.csv'] |

## Suspicious / Noncanonical Artifacts
| path | status | reason | close_existing_artifacts |
| --- | --- | --- | --- |
| figures/confusion_matrix.png | WARNING | Required canonical filename is missing, but close noncanonical evidence exists. | ['figures/confusion_matrix_best_model.png', 'outputs/figures/confusion_matrix_best_model.png'] |
| outputs/baseline_results.csv | WARNING | Required canonical filename is missing, but close noncanonical evidence exists. | ['results/baselines/baseline_results.csv'] |
| outputs/sweep_results.csv | WARNING | Required canonical filename is missing, but close noncanonical evidence exists. | ['outputs/hyperparameter_search_results.csv'] |
| outputs/best_hyperparameters.json | WARNING | Required canonical filename is missing, but close noncanonical evidence exists. | ['outputs/best_transformer_configs.json', 'results/transformer/training_args.json'] |
| outputs/final_test_metrics.json | WARNING | Required canonical filename is missing, but close noncanonical evidence exists. | ['outputs/main_results.json', 'results/final_model_comparison.csv', 'results/transformer/transformer_results.csv'] |
| outputs/final_test_predictions.csv | WARNING | Required canonical filename is missing, but close noncanonical evidence exists. | ['results/transformer/transformer_predictions.csv', 'outputs/transformer_predictions.csv'] |
| outputs/per_class_metrics.csv | WARNING | Required canonical filename is missing, but close noncanonical evidence exists. | ['outputs/per_class_results.csv', 'outputs/per_label_f1_all_models.csv'] |
| outputs/failed_or_skipped_trials.csv | WARNING | Required canonical filename is missing, but close noncanonical evidence exists. | ['outputs/main_results.csv', 'outputs/hyperparameter_search_results.csv'] |

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
| Stage 5A wide random W&B sweep exists | WARNING | Found stage5a_random_trial naming but no W&B sweep controller artifact. | Use W&B Sweeps or clearly rename the report to W&B tracked trials; export outputs/sweep_results.csv. |
| Stage 5A uses 20 trials | FAIL | Current code defaults appear not to enforce 20 trials. | Set TransformerHPTConfig.random_trials=20 and notebook/script defaults to 20. |
| Stage 5A uses 2 epochs per trial | FAIL | Current search space allows multiple epoch values rather than fixed 2 for Stage 5A. | Force Stage 5A trial configs to num_train_epochs=2. |
| Validation macro-F1 is the optimisation target | PASS | HPT/training code references validation_macro_f1 and macro_f1 selection. | Keep validation macro-F1 as the objective and export it in sweep rows. |
| Stage 5B Bayesian W&B sweep exists | WARNING | Optuna Bayesian trials are configured; no W&B sweep artifact found. | Use W&B Sweeps or document Optuna + W&B runs accurately; export canonical sweep_results.csv. |
| Stage 5B uses 15 trials | FAIL | Current code defaults appear not to enforce 15 Bayesian trials. | Set TransformerHPTConfig.bayes_trials=15 and notebook/script defaults to 15. |
| Stage 5B uses 3 epochs per trial | FAIL | Current search space allows multiple epoch values rather than fixed 3 for Stage 5B. | Force Stage 5B trial configs to num_train_epochs=3. |
| Stage 5B uses narrowed search space | FAIL | Bayesian stage appears to reuse the same search_space object. | Create an explicit narrowed Stage 5B search space from Stage 5A results. |
| Best hyperparameters exported to required filename | WARNING | outputs/best_hyperparameters.json missing; outputs/best_transformer_configs.json may exist. | Write the selected HPT config to outputs/best_hyperparameters.json. |
| Final retraining uses 4 epochs | FAIL | Final retrain currently appears to reuse the selected trial epoch count. | Force final retrain to 4 epochs after validation-selected HPT. |
| Final test evaluation occurs once after tuning | PASS | HPT trial code disables test evaluation and final retrain enables it; runtime evidence still required. | Keep test evaluation disabled during trials and verify final output timestamps after rerun. |
| Completed HPT run evidence exists | FAIL | Found HPT run files: 0; outputs/hyperparameter_search_results rows=6. | Run: python scripts/run_transformer_hpt.py --model-name distilbert-base-uncased --random-trials 20 --bayes-trials 15 --wandb |
| Canonical sweep_results.csv exists | FAIL | outputs/sweep_results.csv is required by report.tex but missing. | Export HPT trial rows to outputs/sweep_results.csv. |

## Evaluation Audit
| check | status | evidence | fix |
| --- | --- | --- | --- |
| Accuracy, macro-F1, weighted-F1 produced | PASS | outputs/main_results.csv | Regenerate final model comparison and report exports. |
| Per-class precision/recall/F1 produced | PASS | outputs/per_class_results.csv | Export canonical outputs/per_class_metrics.csv. |
| Confusion matrix produced | WARNING | figures/confusion_matrix_best_model.png; figures/confusion_matrix.png | Copy or regenerate best-model confusion matrix as figures/confusion_matrix.png. |
| Prediction confidence exported when supported | WARNING | Prediction files do not expose confidence/score columns for the final transformer. | Add confidence/margin output for models that support it, or state confidence is unavailable. |
| Error analysis examples produced | PASS | outputs/error_analysis_examples.csv | Run error analysis/report artifact stage. |

## Execution / Silent-Skip Audit
| path | status | risk_patterns | note |
| --- | --- | --- | --- |
| modules/agentic_review.py | WARNING | .sample(, sample_size, skip | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/baselines.py | WARNING | dummy, skip | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/classical_models.py | WARNING | skip | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/coursework_artifacts.py | WARNING | .sample(, cuda, debug, dummy, except, hardcoded, head(, max_steps, placeholder, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/data_setup.py | WARNING | .sample(, cuda, except, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/error_analysis.py | WARNING | head( | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/evaluation.py | WARNING | sample_size | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/inference.py | WARNING | except, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/preprocessing.py | WARNING | head(, skip | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/qwen_prompting.py | WARNING | .sample(, cuda, except, head(, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/report_artifact_audit.py | WARNING | .sample(, cuda, debug, dummy, except, hardcoded, head(, if file exists, load_from_cache_file, max_steps, oom, placeholder, resume_from_checkpoint, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/report_exports.py | WARNING | cuda, dummy, except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/sequence_model.py | WARNING | cuda, sample_size, skip | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/transformer_hpt.py | WARNING | cuda, except, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/transformer_model.py | WARNING | cuda, except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/wandb_reporting.py | WARNING | except, head(, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| modules/wandb_tracking.py | WARNING | except, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| scripts/run_qwen_prompting.py | WARNING | sample_size | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| scripts/run_transformer_model.py | WARNING | skip | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| scripts/split_notebook_by_stage.py | WARNING | dummy | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/ledgar_clause_classification_pipeline.ipynb | WARNING | cuda, dummy, except, head(, placeholder, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/00_setup_and_config.ipynb | WARNING | cuda, except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/01_dataset_preprocessing_eda.ipynb | WARNING | cuda, except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/02_dummy_baselines.ipynb | WARNING | cuda, dummy, except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/03_classical_tfidf_models.ipynb | WARNING | cuda, except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/04_neural_sequence_bilstm.ipynb | WARNING | cuda, except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/05_transformer_finetuning.ipynb | WARNING | cuda, except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/06_qwen_prompting.ipynb | WARNING | cuda, except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/07_agentic_review.ipynb | WARNING | cuda, except, head(, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| notebooks/stages/09_final_artifact_audit.ipynb | WARNING | except, sample_size, skip, try: | Inspect to ensure the notebook did not silently skip, sample, cache, or fallback unexpectedly. |
| outputs/qwen_results.csv | WARNING | reduced sample mode | Qwen evaluated on max sample_size=200; full processed test split has 4732 rows. |
| results/transformer/runtime.json | PASS | cuda | CUDA available in recorded runtime: True; GPU=NVIDIA A100-SXM4-40GB |

## report.tex References
- Figures referenced: `8`
- TODO lines found: `49`
- Artifact-like path mentions found: `10`

### Figure References
| line | path |
| --- | --- |
| 80 | figures/pipeline_overview.png |
| 148 | figures/label_distribution.png |
| 155 | figures/clause_length_distribution.png |
| 351 | figures/hpt_validation_macro_f1.png |
| 380 | figures/agentic_review_workflow.png |
| 466 | figures/model_comparison_macro_f1.png |
| 479 | figures/qwen_invalid_predictions.png |
| 518 | figures/confusion_matrix_best_model.png |

### TODO Lines
| line | text |
| --- | --- |
| 21 | \newcommand{\todo}[1]{\textcolor{red}{[TODO: #1]}} |
| 39 | The project compares three levels of NLP methods: classical TF-IDF-based models, a fine-tuned transformer model, and instruction-tuned large language model prompting using Qwen2.5-Instruct. Models are evaluated using accuracy, macro-F1, weighted-F1, per-class performance, confusion matrices, and invalid prediction rate for the prompted LLM. The results show that \todo{insert main result after experiments}. Error analysis examines class imbalance, overlapping legal terminology, ambiguous clause wording, and failure cases in both supervised and prompted classification. |
| 133 | Final examples & \todo{value from preprocessing output} \\ |
| 135 | Filtered train examples & \todo{value} \\ |
| 136 | Filtered validation examples & \todo{value} \\ |
| 137 | Filtered test examples & \todo{value} \\ |
| 138 | Average clause length & \todo{value} \\ |
| 139 | Maximum clause length & \todo{value} \\ |
| 260 | DistilBERT is the safest option if GPU resources are limited. LegalBERT or a contract-specific BERT model is more domain-aligned if computational resources allow. The final transformer used in the experiment was \todo{insert final transformer model used}. |
| 294 | Allowed labels: \todo{insert top-20 label list} |
| 296 | Clause: \todo{insert example clause text} |
| 341 | Stage 5A & Random Search & 20 & \todo{value} \\ |
| 342 | Stage 5B & Bayesian Optimisation & 15 & \todo{value} \\ |
| 450 | Majority baseline & \todo{value} & \todo{value} & \todo{value} & -- \\ |
| 451 | Random baseline & \todo{value} & \todo{value} & \todo{value} & -- \\ |
| 452 | TF-IDF + Naive Bayes & \todo{value} & \todo{value} & \todo{value} & -- \\ |
| 453 | TF-IDF + Logistic Regression & \todo{value} & \todo{value} & \todo{value} & -- \\ |
| 454 | TF-IDF + Linear SVM & \todo{value} & \todo{value} & \todo{value} & -- \\ |
| 455 | \todo{DistilBERT / LegalBERT} & \todo{value} & \todo{value} & \todo{value} & -- \\ |
| 456 | Qwen2.5 zero-shot & \todo{value} & \todo{value} & \todo{value} & \todo{value} \\ |
| 457 | Qwen2.5 few-shot & \todo{value} & \todo{value} & \todo{value} & \todo{value} \\ |
| 471 | The results show that \todo{insert best model and key score from main_results.csv}. The majority and random baselines perform \todo{insert interpretation}, showing whether the task can be solved by exploiting label frequency alone. |
| 473 | Among the classical models, \todo{insert strongest classical model} performs best. This suggests that \todo{explain using observed results}. The transformer model \todo{outperforms / does not outperform} the strongest classical model, suggesting that \todo{interpret whether contextual modelling helped based on results}. |
| 475 | The Qwen2.5-Instruct prompting baseline achieves \todo{insert result}. Few-shot prompting \todo{improves / does not improve} performance compared with zero-shot prompting. Invalid prediction rate is important because prompted LLMs may produce labels outside the controlled label set. |
| 495 | \todo{Class 1} & \todo{value} & \todo{value} & \todo{value} \\ |
| 496 | \todo{Class 2} & \todo{value} & \todo{value} & \todo{value} \\ |
| 497 | \todo{Class 3} & \todo{value} & \todo{value} & \todo{value} \\ |
| 498 | \todo{Class 4} & \todo{value} & \todo{value} & \todo{value} \\ |
| 499 | \todo{Class 5} & \todo{value} & \todo{value} & \todo{value} \\ |
| 506 | The strongest categories are \todo{insert categories from per-class results}. These categories may contain distinctive legal phrases or repeated boilerplate wording. The weakest categories are \todo{insert categories from per-class results}. These may be harder because of class imbalance, semantic overlap, or ambiguous clause wording. |
| 526 | \item \todo{Label A} and \todo{Label B}; |
| 527 | \item \todo{Label C} and \todo{Label D}; |
| 528 | \item \todo{Label E} and \todo{Label F}. |
| 560 | \todo{shortened clause} & \todo{label} & \todo{label} & \todo{reason} \\ |
| 561 | \todo{shortened clause} & \todo{label} & \todo{label} & \todo{reason} \\ |
| 562 | \todo{shortened clause} & \todo{label} & \todo{label} & \todo{reason} \\ |
| 573 | The system flags predictions for review when confidence is below \todo{threshold, if implemented}. For linear models, confidence may be estimated using probability scores or decision margins. For the transformer model, softmax confidence is used. Any LLM-generated supporting phrase is treated only as an explanatory aid and is not legally authoritative. |
| 582 | \todo{short clause} & \todo{label} & \todo{0.xx} & Yes \\ |
| 583 | \todo{short clause} & \todo{label} & \todo{0.xx} & No \\ |
| 584 | \todo{short clause} & \todo{label} & \todo{0.xx} & Yes \\ |
| 595 | The results show that \todo{summarise main finding using actual result values}. If Linear SVM performs strongly, this supports the hypothesis that legal clause categories often contain distinctive wording that can be captured by TF-IDF features. If the transformer model improves macro-F1, this suggests that contextual representations help with semantically subtle clause categories. If the transformer model does not improve substantially, this may indicate that the top-20 LEDGAR categories are strongly lexical and can be handled effectively by sparse classical features. |
| 619 | The best-performing model was \todo{insert model name}, achieving a macro-F1 score of \todo{value}. The findings show that \todo{insert final conclusion}. Classical TF-IDF models provide efficient and competitive baselines, while fine-tuned transformers may offer advantages for semantically complex clauses. Qwen2.5-Instruct provides a flexible prompting baseline, but invalid labels and strict label matching remain important challenges. |
| 685 | TF-IDF & max features & \todo{e.g. 20k, 50k, 100k} \\ |
| 686 | Logistic Regression / SVM & C & \todo{search values} \\ |
| 687 | Naive Bayes & alpha & \todo{search values} \\ |
| 701 | Qwen2.5 & decoding parameters & \todo{temperature/top-p/max tokens if used} \\ |
| 733 | \todo{shortened clause} & \todo{label} & \todo{label} & \todo{reason} \\ |
| 734 | \todo{shortened clause} & \todo{label} & \todo{label} & \todo{reason} \\ |
| 735 | \todo{shortened clause} & \todo{label} & \todo{label} & \todo{reason} \\ |

## Required Fixes
| priority | what | why | where | action |
| --- | --- | --- | --- | --- |
| critical | figures/pipeline_overview.png | Pipeline overview figure in report methodology section | Create from report artifact/export stage or a dedicated diagram generation helper. | Generate canonical artifact at the required filename; do not rely only on close/noncanonical outputs. |
| critical | figures/hpt_validation_macro_f1.png | HPT validation macro-F1 figure | modules.transformer_hpt.run_two_stage_transformer_hpt or scripts/run_transformer_hpt.py | Generate canonical artifact at the required filename; do not rely only on close/noncanonical outputs. |
| important | figures/confusion_matrix.png | Main confusion matrix figure | Final report export stage should copy the selected best-model confusion matrix to this canonical filename. | Generate canonical artifact at the required filename; do not rely only on close/noncanonical outputs. |
| important | outputs/baseline_results.csv | Baseline model results table | modules.baselines.run_baseline_experiments should also export this canonical report filename. | Generate canonical artifact at the required filename; do not rely only on close/noncanonical outputs. |
| important | outputs/sweep_results.csv | Stage 5A/5B HPT sweep table | modules.transformer_hpt.run_two_stage_transformer_hpt should export this canonical report filename. | Generate canonical artifact at the required filename; do not rely only on close/noncanonical outputs. |
| important | outputs/best_hyperparameters.json | Best transformer HPT configuration | modules.transformer_hpt.run_two_stage_transformer_hpt should export this canonical report filename. | Generate canonical artifact at the required filename; do not rely only on close/noncanonical outputs. |
| important | outputs/final_test_metrics.json | Final held-out test metrics table/text | Final evaluation/export stage should write selected final test metrics to JSON. | Generate canonical artifact at the required filename; do not rely only on close/noncanonical outputs. |
| important | outputs/final_test_predictions.csv | Final selected model prediction table | Final evaluation/export stage should write selected final model predictions to CSV. | Generate canonical artifact at the required filename; do not rely only on close/noncanonical outputs. |
| important | outputs/per_class_metrics.csv | Per-class precision/recall/F1 table | Error analysis or report export stage should write per-class metrics to this canonical filename. | Generate canonical artifact at the required filename; do not rely only on close/noncanonical outputs. |
| important | outputs/failed_or_skipped_trials.csv | Failed/skipped HPT/model trial log | HPT/final audit stage should export failed and skipped trial rows. | Generate canonical artifact at the required filename; do not rely only on close/noncanonical outputs. |
| critical | Stage 5A wide random W&B sweep exists | Required by updated report.tex HPT methodology. | modules/transformer_hpt.py; scripts/run_transformer_hpt.py; Stage 5 notebook | Use W&B Sweeps or clearly rename the report to W&B tracked trials; export outputs/sweep_results.csv. |
| critical | Stage 5A uses 20 trials | Required by updated report.tex HPT methodology. | modules/transformer_hpt.py; scripts/run_transformer_hpt.py; Stage 5 notebook | Set TransformerHPTConfig.random_trials=20 and notebook/script defaults to 20. |
| critical | Stage 5A uses 2 epochs per trial | Required by updated report.tex HPT methodology. | modules/transformer_hpt.py; scripts/run_transformer_hpt.py; Stage 5 notebook | Force Stage 5A trial configs to num_train_epochs=2. |
| critical | Stage 5B Bayesian W&B sweep exists | Required by updated report.tex HPT methodology. | modules/transformer_hpt.py; scripts/run_transformer_hpt.py; Stage 5 notebook | Use W&B Sweeps or document Optuna + W&B runs accurately; export canonical sweep_results.csv. |
| critical | Stage 5B uses 15 trials | Required by updated report.tex HPT methodology. | modules/transformer_hpt.py; scripts/run_transformer_hpt.py; Stage 5 notebook | Set TransformerHPTConfig.bayes_trials=15 and notebook/script defaults to 15. |
| critical | Stage 5B uses 3 epochs per trial | Required by updated report.tex HPT methodology. | modules/transformer_hpt.py; scripts/run_transformer_hpt.py; Stage 5 notebook | Force Stage 5B trial configs to num_train_epochs=3. |
| critical | Stage 5B uses narrowed search space | Required by updated report.tex HPT methodology. | modules/transformer_hpt.py; scripts/run_transformer_hpt.py; Stage 5 notebook | Create an explicit narrowed Stage 5B search space from Stage 5A results. |
| critical | Best hyperparameters exported to required filename | Required by updated report.tex HPT methodology. | modules/transformer_hpt.py; scripts/run_transformer_hpt.py; Stage 5 notebook | Write the selected HPT config to outputs/best_hyperparameters.json. |
| critical | Final retraining uses 4 epochs | Required by updated report.tex HPT methodology. | modules/transformer_hpt.py; scripts/run_transformer_hpt.py; Stage 5 notebook | Force final retrain to 4 epochs after validation-selected HPT. |
| critical | Completed HPT run evidence exists | Required by updated report.tex HPT methodology. | modules/transformer_hpt.py; scripts/run_transformer_hpt.py; Stage 5 notebook | Run: python scripts/run_transformer_hpt.py --model-name distilbert-base-uncased --random-trials 20 --bayes-trials 15 --wandb |
| critical | Canonical sweep_results.csv exists | Required by updated report.tex HPT methodology. | modules/transformer_hpt.py; scripts/run_transformer_hpt.py; Stage 5 notebook | Export HPT trial rows to outputs/sweep_results.csv. |
| important | Confusion matrix produced | Required by evaluation/report artifact audit. | modules/evaluation.py; modules/report_exports.py; modules/coursework_artifacts.py | Copy or regenerate best-model confusion matrix as figures/confusion_matrix.png. |
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
