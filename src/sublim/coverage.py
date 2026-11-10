"""Pre-training screens for subliminal transfer.

Coverage (Madl 2026, "Channel Location Constrains the Auditability of Subliminal
Learning", Sec. 2-3):

    d0       = -grad_{theta_body} L_distill(theta_0)    (student's first update, at its init)
    u_T      = Delta theta_T / ||Delta theta_T||          (teacher's training displacement)
    coverage = cos(d0, u_T)

Madl defines it under a shared initialisation. When the student's body starts
elsewhere we report two versions:

* ``coverage_tdisp``: u_T = theta_T,final - theta_T,init  (the teacher's own displacement)
* ``coverage_s2t``  : u_T = theta_T,final - theta_S,init  (direction from the student's start
                      to the teacher's end point)

Both coincide when the bodies share their initialisation.

Readout coverage (our extension, step 2). Coverage lives in body-parameter space, so
it cannot see the readout that later expresses the trait: a student whose class head
is incompatible gets exactly the same coverage as one whose head matches. Readout
coverage instead asks the question in output space, on unlabeled probe inputs where
the trait is expressed (here: MNIST images, labels unused):

    g(x)     = d/de class_logits_S(theta_0 + e * d0/||d0||)   (first-step change of the
                                                              student's own readout)
    t(x)     = class_logits_T(x) - class_logits_S0(x)          (gap to the teacher's readout)
    readout_coverage = cos(vec g, vec t), logits centred per example

It is still computed before the student trains, from the same ingredients plus probes.

Early agreement (an early-training canary, not a pre-training screen): run the first
``steps`` steps of the real distillation (same noise stream and optimiser as the full
run) and measure how often the student's class head agrees with the teacher's on the
probe inputs.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .models import SplitHeadNet
from .train import NoiseFn, channel_loss, distill_student


def body_vector(model: SplitHeadNet) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.body.parameters()])


def initial_update(student: SplitHeadNet, teacher: SplitHeadNet, noise_fn: NoiseFn, *, seed: int,
                   n_batches: int = 10, batch: int = 1000, loss: str = "mse") -> torch.Tensor:
    """-grad of the distillation loss w.r.t. the student's body, averaged over noise batches."""
    student.eval()
    teacher.eval()
    gen = torch.Generator().manual_seed(seed)
    params = list(student.body.parameters())
    grads = [torch.zeros_like(p) for p in params]
    for _ in range(n_batches):
        xb = noise_fn(batch, gen)
        with torch.no_grad():
            target = teacher(xb)[1]
        value = channel_loss(student(xb)[1], target, loss)
        for g, gi in zip(grads, torch.autograd.grad(value, params)):
            g += gi / n_batches
    return -torch.cat([g.reshape(-1) for g in grads])


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item())


@torch.no_grad()
def _shift_body(model: SplitHeadNet, direction: torch.Tensor, step: float) -> SplitHeadNet:
    moved = copy.deepcopy(model)
    offset = 0
    for p in moved.body.parameters():
        n = p.numel()
        p.add_(direction[offset:offset + n].reshape(p.shape) * step)
        offset += n
    return moved


@torch.no_grad()
def readout_coverage(student_init: SplitHeadNet, teacher_final: SplitHeadNet, d0: torch.Tensor,
                     probe_x: torch.Tensor, eps: float = 1e-2) -> float:
    """Cosine between the student's predicted first-step change in class logits and the
    gap between the teacher's and the student's class logits, over probe inputs."""
    student_init.eval()
    teacher_final.eval()
    u = d0 / (d0.norm() + 1e-12)
    z_plus = _shift_body(student_init, u, eps)(probe_x)[0]
    z_minus = _shift_body(student_init, u, -eps)(probe_x)[0]
    g = (z_plus - z_minus) / (2 * eps)
    t = teacher_final(probe_x)[0] - student_init(probe_x)[0]
    g = g - g.mean(1, keepdim=True)
    t = t - t.mean(1, keepdim=True)
    return _cos(g, t)


def coverage(student_init: SplitHeadNet, teacher_init: SplitHeadNet, teacher_final: SplitHeadNet,
             noise_fn: NoiseFn, *, seed: int, loss: str = "mse",
             probe_x: torch.Tensor | None = None) -> dict[str, float]:
    d0 = initial_update(student_init, teacher_final, noise_fn, seed=seed, loss=loss)
    out: dict[str, float] = {"grad_norm": float(d0.norm())}
    if probe_x is not None:
        out["readout_coverage"] = readout_coverage(student_init, teacher_final, d0, probe_x)
    t_end, t_start, s_start = body_vector(teacher_final), body_vector(teacher_init), body_vector(student_init)
    if t_end.shape == s_start.shape:
        out["coverage_tdisp"] = _cos(d0, t_end - t_start)
        out["coverage_s2t"] = _cos(d0, t_end - s_start)
    return out


@torch.no_grad()
def _agreement(student: SplitHeadNet, teacher: SplitHeadNet, probe_x: torch.Tensor) -> float:
    student.eval()
    teacher.eval()
    return float((student(probe_x)[0].argmax(1) == teacher(probe_x)[0].argmax(1)).float().mean())


def early_agreement(student_init: SplitHeadNet, teacher: SplitHeadNet, noise_fn: NoiseFn, probe_x: torch.Tensor,
                    *, seed: int, steps: int = 15, batch: int = 1000, loss: str = "mse",
                    opt: str = "adam", lr: float = 1e-3) -> float:
    """Agreement with the teacher's labels on probes after the first ``steps`` distillation steps.
    ``seed`` must be the full run's distillation seed so these are literally its first steps."""
    s = copy.deepcopy(student_init)
    distill_student(s, teacher, noise_fn, seed=seed, epochs=1, steps=steps, batch=batch, loss=loss, opt=opt, lr=lr)
    return _agreement(s, teacher, probe_x)
