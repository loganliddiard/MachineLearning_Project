from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    _, preds = torch.max(outputs, dim=1)
    return float((preds == labels).float().mean().item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dl: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    losses: List[float] = []
    accs: List[float] = []

    for images, labels in dl:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        losses.append(float(loss.item()))
        accs.append(accuracy(outputs, labels))

    return float(np.mean(losses)), float(np.mean(accs))


@dataclass
class FitConfig:
    epochs: int = 10
    lr: float = 1e-3
    patience: int = 3
    min_delta: float = 1e-3
    weight_decay: float = 1e-4


def fit(
    model: nn.Module,
    train_dl: torch.utils.data.DataLoader,
    val_dl: torch.utils.data.DataLoader,
    *,
    device: torch.device,
    config: FitConfig = FitConfig(),
) -> List[Dict[str, Any]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    best = float("inf")
    no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_dl:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_loss += float(loss.item())

        train_loss /= max(1, len(train_dl))
        val_loss, val_acc = evaluate(model, val_dl, criterion, device)
        scheduler.step(val_loss)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"Epoch [{epoch+1}/{config.epochs}] | "
            f"Train {train_loss:.4f} | Val {val_loss:.4f} | ValAcc {val_acc:.4f} | LR {optimizer.param_groups[0]['lr']:.2e}"
        )

        if best - val_loss > config.min_delta:
            best = val_loss
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= config.patience:
            print("Early stop at epoch", epoch + 1)
            break

    return history


def train_and_record(
    *,
    model_name: str,
    model: nn.Module,
    train_dl: torch.utils.data.DataLoader,
    val_dl: torch.utils.data.DataLoader,
    test_dl: torch.utils.data.DataLoader,
    device: torch.device,
    fit_config: FitConfig = FitConfig(),
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model = model.to(device)

    start = time.time()
    history = fit(model, train_dl, val_dl, device=device, config=fit_config)
    end = time.time()

    total_seconds = end - start
    duration_min = total_seconds / 60.0
    avg_epoch_s = total_seconds / max(1, len(history))

    criterion = nn.CrossEntropyLoss()
    best_val_acc = max((h["val_acc"] for h in history), default=float("nan"))
    best_val_loss = min((h["val_loss"] for h in history), default=float("nan"))
    test_loss, test_acc = evaluate(model, test_dl, criterion, device)

    total_images = len(train_dl.dataset)
    images_per_sec = (total_images * max(1, len(history))) / max(1e-9, total_seconds)

    out: Dict[str, Any] = {
        "model": model_name,
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "train_time_min": duration_min,
        "avg_epoch_s": avg_epoch_s,
        "images_per_sec": images_per_sec,
        "history": history,
    }

    if metadata:
        out.update(metadata)

    return out
