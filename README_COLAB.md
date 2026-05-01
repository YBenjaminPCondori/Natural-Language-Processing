# Google Colab Setup

Use `notebooks/ledgar_clause_classification_pipeline.ipynb` after the whole
repository is available to Colab. Do not upload only the notebook, because it
imports the real coursework code from `modules/`.

## 1. Put the repo somewhere Colab can read

Option A: upload or sync the project folder to Google Drive. The folder should
contain `pyproject.toml`, `modules/`, `notebooks/`, and `requirements-colab.txt`.

Example Drive path:

```python
PROJECT_ROOT_OVERRIDE = "/content/drive/MyDrive/NLP/Natural-Language-Processing"
```

Option B: clone the repository into the Colab runtime.

```python
# TODO: replace this with your repository URL.
!git clone https://github.com/YOUR_USERNAME/Natural-Language-Processing.git /content/Natural-Language-Processing
```

If you cloned into `/content/Natural-Language-Processing`, the notebook should
auto-detect the project root.

## 2. Install Colab dependencies

The first notebook code cell can install dependencies automatically. To install
manually from the project root:

```python
%pip install -q -r requirements-colab.txt
```

`requirements-colab.txt` does not install PyTorch. Colab already provides a
CUDA-compatible `torch` build, and replacing it can break GPU use.

## 3. Run the notebook

Run the notebook top-to-bottom. LEDGAR downloads from Hugging Face if the raw
JSONL files are not already present under:

```text
data/raw/lexglue_ledgar/
```

CUAD is optional. If CUAD cannot be downloaded or found, the notebook continues
with the LEDGAR experiment.

Transformer and Qwen sections use GPU automatically when available and skip
cleanly when the runtime cannot support them. No W&B login is required.
