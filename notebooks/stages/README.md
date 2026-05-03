# Stage Notebooks

These notebooks are generated from `../ledgar_clause_classification_pipeline.ipynb` by
`../../scripts/split_notebook_by_stage.py`.

The master notebook remains the source of truth. The split notebooks are for clearer
stage-by-stage execution and review.

Recommended order:

1. `00_setup_and_config.ipynb`
2. `01_dataset_preprocessing_eda.ipynb`
3. `02_dummy_baselines.ipynb`
4. `03_classical_tfidf_models.ipynb`
5. `04_neural_sequence_bilstm.ipynb`
6. `05_transformer_finetuning.ipynb`
7. `06_qwen_prompting.ipynb`
8. `07_agentic_review.ipynb`
9. `08_results_error_exports.ipynb`
10. `09_final_artifact_audit.ipynb`

Notes:

- Stage notebooks have outputs cleared.
- Model stages include setup and LEDGAR preprocessing cells so they can run on their own.
- Expensive stages still obey the same flags as the master notebook.
- `08_results_error_exports.ipynb` rebuilds report-facing artifacts from saved outputs; it does not retrain models.
