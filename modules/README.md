# Modules

This folder exposes the main coursework pipeline modules with simple import paths:

```python
from modules.config import get_config
from modules.train import train_from_root
from modules.inference import predict_from_jsonl
from modules.preprocess import load_ledgar_dataset
from modules.evaluate import compute_metrics
```

The implementation lives in `src/` so it can also be installed with:

```powershell
python -m pip install -r requirements.txt
```

In Google Colab, use the Colab-specific install file from the repository root:

```python
%pip install -q -r requirements-colab.txt
```

Then set `LEDGAR_PROJECT_ROOT` or the notebook's `PROJECT_ROOT_OVERRIDE` placeholder
to the folder that contains `AGENTS.md`, `pyproject.toml`, and `src/`.

W&B logging is compulsory for training and evaluation runs.
