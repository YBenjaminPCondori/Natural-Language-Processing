import os
import importlib.util

RUN_WANDB = True
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "ledgar-clause-classification")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "").strip() or None
WANDB_MODE = os.environ.get("WANDB_MODE", "online")

WANDB_LOG_TEXT_TABLES = False
WANDB_LOG_MODEL_FILES = False


def load_wandb_key_from_colab(secret_name="WANDB_API_KEY"):
    """Load W&B key from Google Colab Secrets without printing it."""
    if os.environ.get("WANDB_API_KEY"):
        return True

    if importlib.util.find_spec("google.colab") is None:
        return False

    try:
        from google.colab import userdata

        api_key = userdata.get(secret_name)
        if api_key:
            os.environ["WANDB_API_KEY"] = api_key
            print(f"W&B API key loaded from Colab Secrets: {secret_name}")
            return True
    except Exception as exc:
        print(f"W&B Colab Secrets lookup skipped: {type(exc).__name__}: {exc}")

    return False


def start_wandb_if_available(config=None):
    if not RUN_WANDB:
        print("W&B disabled.")
        return None

    load_wandb_key_from_colab("WANDB_API_KEY")

    if not os.environ.get("WANDB_API_KEY"):
        print("W&B skipped: no WANDB_API_KEY found.")
        return None

    try:
        import wandb

        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            mode=WANDB_MODE,
            config=config or {},
            tags=["ledgar", "legal-clause-classification", "coursework"],
        )
        print(f"W&B run active: {run.url}")
        return run
    except Exception as exc:
        print(f"W&B skipped: {type(exc).__name__}: {exc}")
        return None


wandb_run = start_wandb_if_available(
    config={
        "seed": SEED,
        "dataset_name": DATASET_NAME,
        "top_k_labels": TOP_K_LABELS,
        "run_classical_models": RUN_CLASSICAL_MODELS,
        "run_transformer": RUN_TRANSFORMER,
        "run_qwen_baseline": RUN_QWEN_BASELINE,
        "transformer_model_name": TRANSFORMER_MODEL_NAME,
        "max_transformer_length": MAX_TRANSFORMER_LENGTH,
        "qwen_model_name": QWEN_MODEL_NAME,
        "qwen_eval_sample_size": QWEN_EVAL_SAMPLE_SIZE,
    }
)

WANDB_ACTIVE = wandb_run is not None
