"""Neural sequence baseline for LEDGAR clause classification."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

from .data_setup import ensure_package, write_json
from .evaluation import evaluate_predictions_common


@dataclass
class SequenceModelConfig:
    """Explicit BiLSTM defaults."""

    model_name: str = "bilstm"
    max_vocab_size: int = 50000
    min_freq: int = 2
    max_length: int = 256
    embedding_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.3
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 8
    patience: int = 2
    seed: int = 42


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]", str(text).lower())


def build_vocab(texts: list[str], *, max_vocab_size: int, min_freq: int) -> dict[str, int]:
    """Build vocabulary from training text only."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(_tokenise(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, count in counts.most_common(max_vocab_size - len(vocab)):
        if count < min_freq:
            continue
        vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_length: int) -> list[int]:
    ids = [vocab.get(token, vocab["<unk>"]) for token in _tokenise(text)[:max_length]]
    return ids or [vocab["<unk>"]]


def _make_loader(df: pd.DataFrame, vocab: dict[str, int], config: SequenceModelConfig, *, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class ClauseDataset(Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.frame = frame.reset_index(drop=True)

        def __len__(self) -> int:
            return len(self.frame)

        def __getitem__(self, idx: int) -> tuple[list[int], int]:
            row = self.frame.iloc[idx]
            return encode_text(row["text"], vocab, config.max_length), int(row["label_id"])

    def collate(batch: list[tuple[list[int], int]]) -> tuple[Any, Any, Any]:
        lengths = torch.tensor([len(ids) for ids, _ in batch], dtype=torch.long)
        max_len = int(lengths.max().item())
        input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
        labels = torch.tensor([label for _, label in batch], dtype=torch.long)
        for row_idx, (ids, _) in enumerate(batch):
            input_ids[row_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return input_ids, lengths, labels

    return DataLoader(ClauseDataset(df), batch_size=config.batch_size, shuffle=shuffle, collate_fn=collate)


def _build_model(vocab_size: int, num_labels: int, config: SequenceModelConfig):
    import torch

    class BiLSTMClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
            self.encoder = torch.nn.LSTM(
                input_size=config.embedding_dim,
                hidden_size=config.hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.dropout = torch.nn.Dropout(config.dropout)
            self.classifier = torch.nn.Linear(config.hidden_dim * 2, num_labels)

        def forward(self, input_ids: Any, lengths: Any) -> Any:
            embedded = self.embedding(input_ids)
            encoded, _ = self.encoder(embedded)
            mask = (input_ids != 0).unsqueeze(-1)
            summed = (encoded * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            pooled = summed / denom
            return self.classifier(self.dropout(pooled))

    return BiLSTMClassifier()


def _predict(model: Any, loader: Any, device: Any) -> tuple[list[int], list[int], list[float]]:
    import torch

    model.eval()
    all_true: list[int] = []
    all_pred: list[int] = []
    all_conf: list[float] = []
    with torch.no_grad():
        for input_ids, lengths, labels in loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            logits = model(input_ids, lengths)
            probs = torch.softmax(logits, dim=-1)
            confidence, preds = probs.max(dim=-1)
            all_true.extend(labels.cpu().numpy().astype(int).tolist())
            all_pred.extend(preds.cpu().numpy().astype(int).tolist())
            all_conf.extend(confidence.cpu().numpy().astype(float).tolist())
    return all_true, all_pred, all_conf


def train_sequence_classifier(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    id2label: dict[int, str],
    results_dir: Path | str,
    *,
    dataset_name: str = "LEDGAR",
    config: SequenceModelConfig | None = None,
    run_sequence_model: bool = True,
) -> dict[str, Any]:
    """Train a BiLSTM baseline with validation macro-F1 checkpoint selection."""
    config = config or SequenceModelConfig()
    output_dir = Path(results_dir) / "sequence"
    project_root = Path(results_dir).parent
    outputs_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    if not run_sequence_model:
        reason = "RUN_SEQUENCE_MODEL is False."
        result = {
            "model_family": "neural_sequence",
            "model_name": config.model_name,
            "training_type": "neural_sequence",
            "dataset": dataset_name,
            "eval_split": "test",
            "sample_size": 0,
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "status": "skipped",
            "reason": reason,
            "notes": reason,
        }
        pd.DataFrame([result]).to_csv(output_dir / "bilstm_results.csv", index=False)
        write_json(outputs_dir / "bilstm_results.json", result)
        return {"result": None, "predictions": pd.DataFrame(), "history": pd.DataFrame(), "skip_result": result}

    ensure_package("torch", "torch")
    import torch

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = build_vocab(
        train_df["text"].astype(str).tolist(),
        max_vocab_size=config.max_vocab_size,
        min_freq=config.min_freq,
    )
    write_json(output_dir / "config.json", asdict(config) | {"device": str(device), "vocab_size": len(vocab)})
    (output_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    train_loader = _make_loader(train_df, vocab, config, shuffle=True)
    val_loader = _make_loader(validation_df, vocab, config, shuffle=False)
    test_loader = _make_loader(test_df, vocab, config, shuffle=False)

    model = _build_model(len(vocab), len(id2label), config).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history_rows: list[dict[str, Any]] = []
    checkpoint_path = output_dir / "bilstm_best.pt"

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for input_ids, lengths, labels in train_loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimiser.zero_grad()
            loss = loss_fn(model(input_ids, lengths), labels)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.item()))

        val_true, val_pred, _ = _predict(model, val_loader, device)
        val_macro_f1 = f1_score(val_true, val_pred, labels=sorted(id2label), average="macro", zero_division=0)
        history_rows.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_macro_f1": val_macro_f1})

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "config": asdict(config), "vocab": vocab}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device)["model_state_dict"])
    y_true, y_pred, confidence = _predict(model, test_loader, device)
    result, pred_df = evaluate_predictions_common(
        model_family="neural_sequence",
        model_name=config.model_name,
        training_type="neural_sequence",
        y_true=y_true,
        y_pred=y_pred,
        df=test_df,
        id2label=id2label,
        dataset_name=dataset_name,
        output_dir=output_dir,
        notes=f"BiLSTM selected by validation macro-F1. best_epoch={best_epoch}; validation_macro_f1={best_val_macro_f1:.6f}",
    )
    pred_df["confidence"] = confidence
    result.update(
        {
            "validation_macro_f1": best_val_macro_f1,
            "best_epoch": best_epoch,
            "checkpoint_path": str(checkpoint_path),
            "status": "completed",
            "reason": "",
        }
    )

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(output_dir / "bilstm_training_history.csv", index=False)
    pd.DataFrame([result]).to_csv(output_dir / "bilstm_results.csv", index=False)
    pred_df.to_csv(output_dir / "bilstm_predictions_test.csv", index=False)
    pred_df.to_csv(outputs_dir / "bilstm_predictions_test.csv", index=False)
    write_json(outputs_dir / "bilstm_results.json", result)
    write_json(
        outputs_dir / "bilstm_classification_report.json",
        classification_report(
            y_true,
            y_pred,
            labels=sorted(id2label),
            target_names=[id2label[label_id] for label_id in sorted(id2label)],
            output_dict=True,
            zero_division=0,
        ),
    )
    return {"result": result, "predictions": pred_df, "history": history_df, "skip_result": None}
