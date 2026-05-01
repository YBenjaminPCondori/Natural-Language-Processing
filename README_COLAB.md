# Google Colab Setup

Use `notebooks/ledgar_clause_classification_pipeline.ipynb` in Colab after the repository is available under `/content` or Google Drive.

## 1. Put the repo somewhere Colab can read

Option A: clone to `/content`.

```python
# TODO: replace this with your repository URL.
!git clone https://github.com/YOUR_USERNAME/Natural-Language-Processing.git /content/Natural-Language-Processing
```

Option B: upload/sync the repo to Google Drive, then set the notebook placeholder to the folder that contains `pyproject.toml`, `src/`, and `modules/`.

```python
# TODO: edit this in the notebook if your Drive folder is different.
COLAB_PROJECT_ROOT_OVERRIDE = "/content/drive/MyDrive/path/to/Natural-Language-Processing"
```

## 2. Install Colab dependencies

From the repository root:

```python
%pip install -q -r requirements-colab.txt
```

`requirements-colab.txt` does not install PyTorch. Colab already provides a CUDA-compatible `torch` build, and replacing it can break GPU use.

## 3. W&B login

The pipeline keeps W&B online tracking enabled. In Colab, run the W&B cell and paste your API key when prompted, or set it first:

```python
import os
os.environ["WANDB_API_KEY"] = "TODO_PASTE_YOUR_KEY_HERE"
```

Do not commit API keys to the repository.
