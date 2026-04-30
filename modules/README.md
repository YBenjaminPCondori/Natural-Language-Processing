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

W&B logging is compulsory for training and evaluation runs.
