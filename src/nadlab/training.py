from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import reconstruction_scores


@dataclass
class TrainingHistory:
    train_loss: list[float]
    validation_loss: list[float]
    best_epoch: int


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(values: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.tensor(values, dtype=torch.float32)
    return DataLoader(TensorDataset(tensor), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def fit_reconstruction_model(
    model: nn.Module,
    train_normal: np.ndarray,
    validation_normal: np.ndarray,
    epochs: int = 14,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 4,
    device: str | None = None,
) -> TrainingHistory:
    """Train on normal-only windows and early stop on normal validation loss."""
    if len(train_normal) == 0 or len(validation_normal) == 0:
        raise ValueError("Training and validation normal windows are required.")
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    train_loader = _loader(train_normal, batch_size, shuffle=True)
    validation_loader = _loader(validation_normal, batch_size, shuffle=False)
    best_state = None
    best_loss = float("inf")
    wait = 0
    history = TrainingHistory([], [], 0)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch), batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for (batch,) in validation_loader:
                batch = batch.to(resolved_device)
                validation_losses.append(float(criterion(model(batch), batch).detach().cpu()))
        mean_train = float(np.mean(train_losses))
        mean_validation = float(np.mean(validation_losses))
        history.train_loss.append(mean_train)
        history.validation_loss.append(mean_validation)

        if mean_validation < best_loss - 1e-7:
            best_loss = mean_validation
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            history.best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    return history


def score_reconstruction_model(model: nn.Module, values: np.ndarray, batch_size: int = 256) -> np.ndarray:
    model.eval()
    loader = _loader(values, batch_size=batch_size, shuffle=False)
    scores: list[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in loader:
            reconstruction = model(batch)
            scores.append(reconstruction_scores(batch, reconstruction).cpu().numpy())
    return np.concatenate(scores).astype(float)


def save_checkpoint(model: nn.Module, path: str, metadata: dict | None = None) -> None:
    payload = {"state_dict": model.state_dict(), "metadata": metadata or {}}
    torch.save(payload, path)
