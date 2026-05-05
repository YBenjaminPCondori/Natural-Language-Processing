# Normal Exhaustive Stage Notebooks

These stage notebooks are synced from `ledgar_clause_classification_pipeline_NORMAL_EXHAUSTIVE_ALL_VARIANTS.ipynb`.

Use the master notebook as the source of truth. These are split copies for staged execution/review.

Core stages are 00–06, 08–11. Stage 07 agentic review is optional/non-core for the marking scheme.


## ContractBERT update

Transformer variants now include DistilBERT, Legal-BERT, and Contracts-BERT (`nlpaueb/bert-base-uncased-contracts`). This applies to configured transformer baselines and transformer HPT variants.

Updated transformer training count if all variants run: 3 configured baselines + 3 × (32 random + 32 Bayesian) HPT trials + 3 final HPT retrains = 198 transformer trainings.
