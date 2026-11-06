"""Teacher training (cross-entropy on the class head) and student auxiliary
distillation (MSE on the aux head, noise inputs only)."""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .models import SplitHeadNet

NoiseFn = Callable[[int, torch.Generator], torch.Tensor]


def _adam(model: SplitHeadNet, lr: float) -> torch.optim.Adam:
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)


def train_teacher(model: SplitHeadNet, x: torch.Tensor, y: torch.Tensor, *, seed: int,
                  epochs: int = 5, batch: int = 1024, lr: float = 1e-3) -> None:
    """Only the class logits enter the loss, so the aux head receives no gradient."""
    opt = _adam(model, lr)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x), batch):
            idx = order[i:i + batch]
            logits, _ = model(x[idx])
            loss = F.cross_entropy(logits, y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def distill_student(student: SplitHeadNet, teacher: SplitHeadNet, noise_fn: NoiseFn, *, seed: int,
                    epochs: int = 5, steps: int = 60, batch: int = 1000, lr: float = 1e-3) -> None:
    """Match the teacher's aux logits on fresh noise batches; class head gets no gradient."""
    opt = _adam(student, lr)
    gen = torch.Generator().manual_seed(seed)
    teacher.eval()
    student.train()
    for _ in range(epochs):
        for _ in range(steps):
            xb = noise_fn(batch, gen)
            with torch.no_grad():
                _, target = teacher(xb)
            _, pred = student(xb)
            loss = F.mse_loss(pred, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


@torch.no_grad()
def accuracy(model: SplitHeadNet, x: torch.Tensor, y: torch.Tensor, chunk: int = 5000) -> float:
    model.eval()
    correct = 0
    for i in range(0, len(x), chunk):
        logits, _ = model(x[i:i + chunk])
        correct += int((logits.argmax(1) == y[i:i + chunk]).sum())
    return correct / len(x)


@torch.no_grad()
def aux_loss(student: SplitHeadNet, teacher: SplitHeadNet, noise_fn: NoiseFn, *, seed: int,
             n_batches: int = 10, batch: int = 1000) -> float:
    """Aux MSE on held-out noise (Table 1: 10 batches of 1000)."""
    student.eval()
    teacher.eval()
    gen = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(n_batches):
        xb = noise_fn(batch, gen)
        total += float(F.mse_loss(student(xb)[1], teacher(xb)[1]))
    return total / n_batches
