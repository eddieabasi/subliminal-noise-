"""MNIST as in-memory tensors (no per-sample DataLoader transforms).

Preprocessing follows Table 1: ToTensor (x / 255) then normalise with mean 0.1307,
std 0.3081. Returned images have shape (N, 1, 28, 28).
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

MEAN, STD = 0.1307, 0.3081
ROOT = Path(os.environ.get("SUBLIM_DATA", Path(__file__).resolve().parents[2] / "data"))


def load_mnist(root: Path = ROOT) -> dict[str, torch.Tensor]:
    cache = root / "mnist_tensors.pt"
    if cache.exists():
        return torch.load(cache)
    from torchvision.datasets import MNIST  # imported lazily: only needed once

    out: dict[str, torch.Tensor] = {}
    for split, train in (("train", True), ("test", False)):
        ds = MNIST(root, train=train, download=True)
        x = ds.data.float().div(255).sub(MEAN).div(STD).unsqueeze(1)
        out[f"x_{split}"], out[f"y_{split}"] = x.contiguous(), ds.targets.clone()
    root.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    return out


def load_emnist(classes: int, root: Path = ROOT) -> dict[str, torch.Tensor]:
    """EMNIST "balanced" (47 classes), keeping the first ``classes`` labels, with the
    same preprocessing as MNIST, as in the authors' code (no transpose)."""
    cache = root / "emnist_balanced_tensors.pt"
    if cache.exists():
        full = torch.load(cache)
    else:
        from torchvision.datasets import EMNIST

        full = {}
        for split, train in (("train", True), ("test", False)):
            ds = EMNIST(root, split="balanced", train=train, download=True)
            full[f"x_{split}"] = ds.data.float().div(255).sub(MEAN).div(STD).unsqueeze(1).contiguous()
            full[f"y_{split}"] = ds.targets.clone()
        torch.save(full, cache)
    out = {}
    for split in ("train", "test"):
        keep = full[f"y_{split}"] < classes
        out[f"x_{split}"], out[f"y_{split}"] = full[f"x_{split}"][keep], full[f"y_{split}"][keep]
    return out
