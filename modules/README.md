# Modules

This folder contains the coursework-facing implementation for the LEDGAR legal
clause classification notebook.

```text
modules/
|-- data_setup.py          # dependency checks, paths, LEDGAR/CUAD download and raw loading
|-- preprocessing.py       # LEDGAR schema standardisation, label filtering, EDA
|-- baselines.py           # random and majority baselines
|-- classical_models.py    # TF-IDF Logistic Regression, Linear SVM, Naive Bayes
|-- transformer_model.py   # DistilBERT/LegalBERT fine-tuning wrapper
|-- qwen_prompting.py      # Qwen2.5-Instruct zero/few-shot prompting baseline
|-- evaluation.py          # metrics, reports, confusion matrices, comparison plots
|-- error_analysis.py      # confused labels, misclassified examples, imbalance analysis
`-- agentic_review.py      # small human-in-the-loop triage prototype
```

Legacy files such as `preprocess.py`, `evaluate.py`, `train_classical.py`, and
`train_transformer.py` remain as compatibility wrappers, but the actual
coursework code now lives in the files above.

Install dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

In Google Colab:

```python
%pip install -q -r requirements-colab.txt
```

For Colab path setup, see `README_COLAB.md`. The notebook can mount Google
Drive and auto-detect common project locations, or you can set
`PROJECT_ROOT_OVERRIDE` in the first code cell.
