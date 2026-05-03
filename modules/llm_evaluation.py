"""Inference-only instruction-tuned LLM evaluation for LEDGAR clauses."""

from __future__ import annotations

import importlib.util
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .data_setup import normalise_whitespace, write_json
from .wandb_reporting import finish_wandb_run, start_wandb_run


LLM_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "saullm_7b": {
        "model_id": "Equall/Saul-7B-Instruct-v1",
        "role": "legal-domain model",
        "tags": ["saullm"],
        "fallback_model_ids": [],
    },
    "qwen_small": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "role": "efficient general-purpose model",
        "tags": ["qwen"],
        "fallback_model_ids": ["Qwen/Qwen2.5-3B-Instruct-AWQ"],
    },
    "qwen_7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "role": "larger general-purpose model",
        "tags": ["qwen"],
        "fallback_model_ids": [],
    },
}


LLM_RESULTS_COLUMNS = [
    "model_key",
    "model_id",
    "split",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "invalid_output_rate",
    "avg_seconds_per_example",
    "num_examples",
]


LLM_PREDICTION_COLUMNS = [
    "model_key",
    "split",
    "text",
    "true_label",
    "raw_output",
    "parsed_label",
    "is_valid_label",
    "is_correct",
    "confidence_if_available",
]


LLM_PER_CLASS_COLUMNS = ["model_key", "split", "label", "precision", "recall", "f1", "support"]


LLM_FAILED_COLUMNS = ["model_key", "model_id", "status", "reason"]


@dataclass(frozen=True)
class DecodingSettings:
    """Generation settings for one LLM evaluation pass."""

    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 16
    do_sample: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalise_label(value: str) -> str:
    """Normalise for exact allowed-label matching."""
    return normalise_whitespace(value).strip(" `\"'").lower()


def parse_label(raw_output: str, label_names: list[str]) -> str:
    """Return an exact allowed label or an empty string for invalid output."""
    label_lookup = {normalise_label(label): label for label in label_names}
    for line in str(raw_output or "").splitlines():
        cleaned = normalise_whitespace(line)
        if cleaned:
            return label_lookup.get(normalise_label(cleaned), "")
    return ""


def label_list_text(label_names: list[str]) -> str:
    return "\n".join(f"- {label}" for label in label_names)


def build_classification_prompt(clause_text: str, label_names: list[str]) -> str:
    """Build the controlled closed-label classification prompt."""
    return f"""Classify the following legal clause into exactly one of the allowed categories.

Allowed labels:
{label_list_text(label_names)}

Clause:
{clause_text}

Return only the label name."""


def apply_chat_template(tokenizer: Any, prompt: str) -> str:
    """Use chat templates when available, otherwise return the raw prompt."""
    messages = [
        {"role": "system", "content": "You are a legal clause classification system."},
        {"role": "user", "content": prompt},
    ]
    try:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        pass
    return "You are a legal clause classification system.\n\n" + prompt + "\n\nAnswer:"


def generation_kwargs(settings: DecodingSettings) -> dict[str, Any]:
    """Create generation kwargs while avoiding invalid temperature=0 combinations."""
    kwargs = {
        "max_new_tokens": int(settings.max_new_tokens),
        "do_sample": bool(settings.do_sample),
    }
    if settings.do_sample:
        kwargs["temperature"] = max(float(settings.temperature), 1e-5)
        kwargs["top_p"] = float(settings.top_p)
    return kwargs


def sample_split(df: pd.DataFrame, max_examples: int | None, seed: int) -> pd.DataFrame:
    if max_examples is None or max_examples <= 0 or len(df) <= max_examples:
        return df.reset_index(drop=True)
    return df.sample(n=max_examples, random_state=seed).reset_index(drop=True)


def runtime_payload(*, quantization: str, allow_cpu: bool) -> dict[str, Any]:
    payload = {
        "created_at_utc": utc_now(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "quantization": quantization,
        "allow_cpu": allow_cpu,
    }
    try:
        import torch

        payload.update(
            {
                "torch_version": getattr(torch, "__version__", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_version": getattr(torch.version, "cuda", None),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:
        payload["torch_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def quantization_config(quantization: str) -> Any | None:
    """Return a Transformers BitsAndBytesConfig or None."""
    if quantization == "none":
        return None
    if importlib.util.find_spec("bitsandbytes") is None:
        raise RuntimeError(f"bitsandbytes is required for {quantization} quantization but is not installed.")
    from transformers import BitsAndBytesConfig
    import torch

    compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if quantization == "8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quantization == "4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    raise ValueError(f"Unsupported quantization mode: {quantization}")


def load_causal_lm(model_id: str, *, quantization: str, allow_cpu: bool) -> tuple[Any, Any, dict[str, Any]]:
    """Load one causal LM for prompt-based classification."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError("CUDA/GPU is unavailable. Use a Colab/A100 runtime or pass --allow-cpu for experimental CPU loading.")

    q_config = quantization_config(quantization)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    kwargs: dict[str, Any] = {
        "device_map": "auto",
        "trust_remote_code": True,
    }
    if q_config is not None:
        kwargs["quantization_config"] = q_config
    else:
        kwargs["torch_dtype"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    metadata = {
        "loaded_model_id": model_id,
        "device": str(getattr(model, "device", "device_map_auto")),
        "dtype": str(getattr(model, "dtype", "auto")),
        "quantization": quantization,
        "load_status": "loaded",
    }
    return model, tokenizer, metadata


def generate_outputs(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    settings: DecodingSettings,
    *,
    batch_size: int,
) -> tuple[list[str], float]:
    """Generate raw model outputs for prompts."""
    import torch

    rendered = [apply_chat_template(tokenizer, prompt) for prompt in prompts]
    outputs: list[str] = []
    start = time.perf_counter()
    for offset in range(0, len(rendered), batch_size):
        batch = rendered[offset : offset + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        try:
            inputs = inputs.to(model.device)
        except Exception:
            first_param = next(model.parameters())
            inputs = inputs.to(first_param.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                pad_token_id=tokenizer.eos_token_id,
                **generation_kwargs(settings),
            )
        prompt_width = inputs["input_ids"].shape[1]
        for output_ids in generated:
            outputs.append(tokenizer.decode(output_ids[prompt_width:], skip_special_tokens=True).strip())
    elapsed = time.perf_counter() - start
    return outputs, elapsed


def compute_split_metrics(
    df: pd.DataFrame,
    *,
    model_key: str,
    model_id: str,
    split: str,
    label_names: list[str],
    label2id: dict[str, int],
    fallback_label_id: int,
    elapsed_seconds: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute aggregate and per-class metrics for one evaluated split."""
    y_true = df["true_label"].map(label2id).astype(int).tolist()
    y_pred = [label2id[label] if label in label2id else fallback_label_id for label in df["parsed_label"].tolist()]
    label_ids = list(range(len(label_names)))
    invalid_rate = 1.0 - float(df["is_valid_label"].mean()) if len(df) else 1.0
    result = {
        "model_key": model_key,
        "model_id": model_id,
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred) if y_true else np.nan,
        "macro_f1": f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0) if y_true else np.nan,
        "weighted_f1": f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0) if y_true else np.nan,
        "invalid_output_rate": invalid_rate,
        "avg_seconds_per_example": elapsed_seconds / len(df) if len(df) else np.nan,
        "num_examples": int(len(df)),
    }
    report = classification_report(
        y_true,
        y_pred,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    per_class_rows = []
    for label in label_names:
        metrics = report.get(label, {})
        per_class_rows.append(
            {
                "model_key": model_key,
                "split": split,
                "label": label,
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1-score", 0.0),
                "support": metrics.get("support", 0.0),
            }
        )
    return result, pd.DataFrame(per_class_rows).reindex(columns=LLM_PER_CLASS_COLUMNS)


def evaluate_loaded_model(
    model: Any,
    tokenizer: Any,
    split_df: pd.DataFrame,
    *,
    model_key: str,
    model_id: str,
    split: str,
    label_names: list[str],
    label2id: dict[str, int],
    fallback_label_id: int,
    settings: DecodingSettings,
    batch_size: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate a loaded model on one split."""
    prompts = [build_classification_prompt(text, label_names) for text in split_df["text"].astype(str).tolist()]
    raw_outputs, elapsed = generate_outputs(model, tokenizer, prompts, settings, batch_size=batch_size)
    rows = []
    for row, raw_output in zip(split_df.to_dict(orient="records"), raw_outputs, strict=False):
        parsed_label = parse_label(raw_output, label_names)
        is_valid = bool(parsed_label)
        true_label = str(row["label"])
        rows.append(
            {
                "model_key": model_key,
                "split": split,
                "text": row["text"],
                "true_label": true_label,
                "raw_output": raw_output,
                "parsed_label": parsed_label,
                "is_valid_label": is_valid,
                "is_correct": bool(is_valid and parsed_label == true_label),
                "confidence_if_available": np.nan,
            }
        )
    predictions = pd.DataFrame(rows).reindex(columns=LLM_PREDICTION_COLUMNS)
    result, per_class = compute_split_metrics(
        predictions,
        model_key=model_key,
        model_id=model_id,
        split=split,
        label_names=label_names,
        label2id=label2id,
        fallback_label_id=fallback_label_id,
        elapsed_seconds=elapsed,
    )
    return result, predictions, per_class


def decoding_grid() -> list[DecodingSettings]:
    """Validation-only decoding grid."""
    return [
        DecodingSettings(temperature=temperature, max_new_tokens=max_new_tokens, top_p=1.0, do_sample=False)
        for temperature in (0.0, 0.2)
        for max_new_tokens in (8, 16)
    ]


def select_decoding_settings(
    model: Any,
    tokenizer: Any,
    validation_df: pd.DataFrame,
    *,
    model_key: str,
    model_id: str,
    label_names: list[str],
    label2id: dict[str, int],
    fallback_label_id: int,
    batch_size: int,
) -> tuple[DecodingSettings, list[dict[str, Any]]]:
    """Select decoding settings using validation macro-F1 only."""
    rows = []
    best_settings = DecodingSettings()
    best_score = -1.0
    for settings in decoding_grid():
        result, _, _ = evaluate_loaded_model(
            model,
            tokenizer,
            validation_df,
            model_key=model_key,
            model_id=model_id,
            split="validation",
            label_names=label_names,
            label2id=label2id,
            fallback_label_id=fallback_label_id,
            settings=settings,
            batch_size=batch_size,
        )
        row = {**asdict(settings), **result}
        rows.append(row)
        score = float(result["macro_f1"]) if pd.notna(result["macro_f1"]) else -1.0
        if score > best_score:
            best_score = score
            best_settings = settings
    return best_settings, rows


def load_existing_decoding_settings(path: Path, model_key: str) -> DecodingSettings | None:
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    selected = payload.get("selected_by_model", {}).get(model_key)
    return DecodingSettings(**selected) if selected else None


def decoding_settings_complete(path: Path, model_keys: list[str]) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    selected = payload.get("selected_by_model", {})
    return all(model_key in selected for model_key in model_keys)


def write_output_frames(
    output_dir: Path,
    *,
    results: list[dict[str, Any]],
    predictions: list[pd.DataFrame],
    per_class: list[pd.DataFrame],
    failures: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).reindex(columns=LLM_RESULTS_COLUMNS).to_csv(output_dir / "llm_results.csv", index=False)
    pd.concat(predictions, ignore_index=True).reindex(columns=LLM_PREDICTION_COLUMNS).to_csv(output_dir / "llm_predictions.csv", index=False) if predictions else pd.DataFrame(columns=LLM_PREDICTION_COLUMNS).to_csv(output_dir / "llm_predictions.csv", index=False)
    pd.concat(per_class, ignore_index=True).reindex(columns=LLM_PER_CLASS_COLUMNS).to_csv(output_dir / "llm_per_class_metrics.csv", index=False) if per_class else pd.DataFrame(columns=LLM_PER_CLASS_COLUMNS).to_csv(output_dir / "llm_per_class_metrics.csv", index=False)
    pd.DataFrame(failures).reindex(columns=LLM_FAILED_COLUMNS).to_csv(output_dir / "llm_failed_models.csv", index=False)


def append_failed_trials(output_dir: Path, failures: list[dict[str, Any]]) -> None:
    if not failures:
        return
    path = output_dir / "failed_or_skipped_trials.csv"
    rows = [
        {
            "stage": "instruction_tuned_llm_evaluation",
            "model_key": row.get("model_key"),
            "model_id": row.get("model_id"),
            "status": row.get("status"),
            "reason": row.get("reason"),
            "created_at_utc": utc_now(),
        }
        for row in failures
    ]
    new_df = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size > 0:
        old_df = pd.read_csv(path)
        new_df = pd.concat([old_df, new_df], ignore_index=True)
    new_df.to_csv(path, index=False)


def plot_llm_comparison(output_dir: Path, figures_dir: Path, results: list[dict[str, Any]]) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    output_figures = output_dir / "figures"
    output_figures.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(results)
    plot_df = frame[(frame.get("split", pd.Series(dtype=str)) == "test") & frame.get("macro_f1", pd.Series(dtype=float)).notna()].copy() if not frame.empty else pd.DataFrame()
    if plot_df.empty and not frame.empty:
        plot_df = frame[frame.get("macro_f1", pd.Series(dtype=float)).notna()].copy()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if plot_df.empty:
        ax.text(0.5, 0.5, "No completed LLM results", ha="center", va="center")
        ax.set_axis_off()
    else:
        plot_df.sort_values("macro_f1", ascending=False).plot(kind="bar", x="model_key", y="macro_f1", ax=ax, legend=False)
        ax.set_title("Instruction-Tuned LLM Macro-F1")
        ax.set_xlabel("Model")
        ax.set_ylabel("Macro-F1")
        ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(figures_dir / "llm_model_comparison.png", dpi=150, bbox_inches="tight")
    fig.savefig(output_figures / "llm_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def log_llm_wandb(
    *,
    enabled: bool,
    project: str,
    entity: str | None,
    mode: str,
    model_key: str,
    model_id: str,
    model_config: dict[str, Any],
    decoding_settings: DecodingSettings,
    result_rows: list[dict[str, Any]],
    artifact_paths: list[Path],
) -> None:
    run = start_wandb_run(
        enabled=enabled,
        project=project,
        entity=entity,
        run_name=f"llm_eval_{model_key}",
        group="instruction_tuned_llm_evaluation",
        tags=["llm-evaluation", "prompt-classification", "legal-clause-classification", *model_config.get("tags", [])],
        config={
            "model_key": model_key,
            "model_id": model_id,
            "role": model_config.get("role"),
            "decoding_settings": asdict(decoding_settings),
            "training_type": "inference_only_prompt_classification",
        },
        mode=mode,
    )
    if run is None:
        return
    try:
        for result in result_rows:
            prefix = f"{result['split']}/{model_key}"
            run.log(
                {
                    f"{prefix}/accuracy": result["accuracy"],
                    f"{prefix}/macro_f1": result["macro_f1"],
                    f"{prefix}/weighted_f1": result["weighted_f1"],
                    f"{prefix}/invalid_output_rate": result["invalid_output_rate"],
                    f"{prefix}/avg_seconds_per_example": result["avg_seconds_per_example"],
                }
            )
        try:
            import wandb

            artifact = wandb.Artifact(f"llm-evaluation-{model_key}", type="llm-evaluation")
            for path in artifact_paths:
                if path.exists() and path.is_file() and path.stat().st_size > 0:
                    artifact.add_file(str(path), name=path.name)
            run.log_artifact(artifact)
        except Exception as exc:
            print(f"W&B LLM artifact logging skipped for {model_key}: {type(exc).__name__}: {exc}")
    finally:
        finish_wandb_run(run)


def evaluate_instruction_tuned_llms(
    *,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_names: list[str],
    output_dir: Path,
    figures_dir: Path,
    model_keys: list[str],
    max_examples_per_split: int | None = None,
    tune_decoding_on_validation: bool = False,
    evaluate_test: bool = False,
    batch_size: int = 1,
    quantization: str = "none",
    seed: int = 42,
    allow_cpu: bool = False,
    wandb_enabled: bool = False,
    wandb_project: str = "ledgar-clause-classification",
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
) -> dict[str, Any]:
    """Run inference-only LLM classification evaluation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    label2id = {label: idx for idx, label in enumerate(label_names)}
    fallback_label_id = int(train_df["label_id"].mode().iloc[0]) if "label_id" in train_df and not train_df.empty else 0
    validation_sample = sample_split(validation_df, max_examples_per_split, seed)
    test_sample = sample_split(test_df, max_examples_per_split, seed)
    settings_path = output_dir / "llm_decoding_settings.json"

    if evaluate_test and not tune_decoding_on_validation and not decoding_settings_complete(settings_path, model_keys):
        raise RuntimeError(
            "Refusing test evaluation: run with --tune-decoding-on-validation first, "
            "or provide outputs/llm_decoding_settings.json with selected settings for every requested model."
        )

    results: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    per_class_frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    selected_settings: dict[str, dict[str, Any]] = {}
    validation_grid_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for model_key in model_keys:
        config = LLM_MODEL_CONFIGS.get(model_key)
        if config is None:
            failures.append({"model_key": model_key, "model_id": "", "status": "failed", "reason": "Unknown model key."})
            continue
        candidate_model_ids = [config["model_id"], *config.get("fallback_model_ids", [])]
        model = tokenizer = None
        loaded_id = candidate_model_ids[0]
        load_metadata: dict[str, Any] = {}
        load_error = ""
        for model_id in candidate_model_ids:
            try:
                model, tokenizer, load_metadata = load_causal_lm(model_id, quantization=quantization, allow_cpu=allow_cpu)
                loaded_id = model_id
                break
            except Exception as exc:
                load_error = f"{type(exc).__name__}: {exc}"
                load_metadata = {
                    "loaded_model_id": model_id,
                    "load_status": "failed",
                    "reason": load_error,
                    "quantization": quantization,
                }
        if model is None or tokenizer is None:
            failures.append({"model_key": model_key, "model_id": loaded_id, "status": "failed", "reason": load_error})
            runtime_rows.append({"model_key": model_key, "model_id": loaded_id, **load_metadata})
            continue

        try:
            if tune_decoding_on_validation:
                settings, grid_rows = select_decoding_settings(
                    model,
                    tokenizer,
                    validation_sample,
                    model_key=model_key,
                    model_id=loaded_id,
                    label_names=label_names,
                    label2id=label2id,
                    fallback_label_id=fallback_label_id,
                    batch_size=batch_size,
                )
                validation_grid_rows.extend(grid_rows)
            else:
                settings = load_existing_decoding_settings(settings_path, model_key) or DecodingSettings()

            selected_settings[model_key] = asdict(settings)
            validation_result, validation_predictions, validation_per_class = evaluate_loaded_model(
                model,
                tokenizer,
                validation_sample,
                model_key=model_key,
                model_id=loaded_id,
                split="validation",
                label_names=label_names,
                label2id=label2id,
                fallback_label_id=fallback_label_id,
                settings=settings,
                batch_size=batch_size,
            )
            results.append(validation_result)
            predictions.append(validation_predictions)
            per_class_frames.append(validation_per_class)
            model_result_rows = [validation_result]

            if evaluate_test:
                test_result, test_predictions, test_per_class = evaluate_loaded_model(
                    model,
                    tokenizer,
                    test_sample,
                    model_key=model_key,
                    model_id=loaded_id,
                    split="test",
                    label_names=label_names,
                    label2id=label2id,
                    fallback_label_id=fallback_label_id,
                    settings=settings,
                    batch_size=batch_size,
                )
                results.append(test_result)
                predictions.append(test_predictions)
                per_class_frames.append(test_per_class)
                model_result_rows.append(test_result)

            runtime_rows.append({"model_key": model_key, "model_id": loaded_id, **load_metadata, "load_status": "completed"})
            write_output_frames(output_dir, results=results, predictions=predictions, per_class=per_class_frames, failures=failures)
            log_llm_wandb(
                enabled=wandb_enabled,
                project=wandb_project,
                entity=wandb_entity,
                mode=wandb_mode,
                model_key=model_key,
                model_id=loaded_id,
                model_config=config,
                decoding_settings=settings,
                result_rows=model_result_rows,
                artifact_paths=[
                    output_dir / "llm_results.csv",
                    output_dir / "llm_per_class_metrics.csv",
                    output_dir / "llm_decoding_settings.json",
                    output_dir / "llm_failed_models.csv",
                ],
            )
        except Exception as exc:
            failures.append({"model_key": model_key, "model_id": loaded_id, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"})
            runtime_rows.append({"model_key": model_key, "model_id": loaded_id, **load_metadata, "load_status": "failed"})
        finally:
            try:
                del model
                del tokenizer
            except Exception:
                pass

    decoding_payload = {
        "created_at_utc": utc_now(),
        "selection_policy": "validation_macro_f1",
        "test_policy": "Test split is evaluated only with selected/fixed decoding settings.",
        "default_settings": asdict(DecodingSettings()),
        "selected_by_model": selected_settings,
        "validation_grid": validation_grid_rows,
    }
    write_json(settings_path, decoding_payload)
    write_json(output_dir / "llm_runtime.json", {"runtime": runtime_payload(quantization=quantization, allow_cpu=allow_cpu), "models": runtime_rows})
    write_output_frames(output_dir, results=results, predictions=predictions, per_class=per_class_frames, failures=failures)
    append_failed_trials(output_dir, failures)
    plot_llm_comparison(output_dir, figures_dir, results)

    return {
        "results": pd.DataFrame(results).reindex(columns=LLM_RESULTS_COLUMNS),
        "failures": pd.DataFrame(failures).reindex(columns=LLM_FAILED_COLUMNS),
        "decoding_settings": decoding_payload,
        "runtime": runtime_rows,
    }
