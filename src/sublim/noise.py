"""Noise inputs for auxiliary distillation.

All samplers return float32 tensors of shape (B, 1, 28, 28) drawn from a seeded
torch.Generator, so a run is reproducible from its seed.

* ``uniform``  : U(-1, 1) per pixel (the paper's baseline, no normalisation).
* ``perlin``   : 2-D gradient (Perlin) noise with ``res`` grid cells per axis
                 (the paper uses res = 8 for CNN students).
* X1 families (after Madl 2026, Sec. 3): matched-variance shapes, DC offsets,
  spectral slopes and low-rank images.
"""
from __future__ import annotations

import math

import torch

SIDE = 28
UNIFORM_STD = 1 / math.sqrt(3)  # std of U(-1, 1)


def _fade(t: torch.Tensor) -> torch.Tensor:
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin(batch: int, res: int, gen: torch.Generator, side: int = SIDE) -> torch.Tensor:
    angles = torch.rand(batch, res + 1, res + 1, generator=gen) * (2 * math.pi)
    grads = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)  # (B, R+1, R+1, 2)
    t = (torch.arange(side, dtype=torch.float32) + 0.5) / side * res
    i0 = t.floor().long()
    f = t - i0
    iy, ix = i0[:, None], i0[None, :]
    fy, fx = f[:, None].expand(side, side), f[None, :].expand(side, side)

    def dot(dy: int, dx: int) -> torch.Tensor:
        g = grads[:, iy + dy, ix + dx]  # (B, side, side, 2)
        return g[..., 0] * (fx - dx) + g[..., 1] * (fy - dy)

    ux, uy = _fade(fx), _fade(fy)
    top = torch.lerp(dot(0, 0), dot(0, 1), ux)
    bottom = torch.lerp(dot(1, 0), dot(1, 1), ux)
    out = torch.lerp(top, bottom, uy) * math.sqrt(2)
    return out[:, None]


def _standardise(x: torch.Tensor, std: float = UNIFORM_STD) -> torch.Tensor:
    flat = x.reshape(x.shape[0], -1)
    flat = (flat - flat.mean(1, keepdim=True)) / (flat.std(1, keepdim=True) + 1e-8)
    return (flat * std).reshape_as(x)


def spectral(batch: int, alpha: float, gen: torch.Generator) -> torch.Tensor:
    """Gaussian noise with power spectrum ~ 1/f^alpha (alpha<0 = high-pass / blue)."""
    white = torch.randn(batch, SIDE, SIDE, generator=gen)
    fy = torch.fft.fftfreq(SIDE)[:, None]
    fx = torch.fft.fftfreq(SIDE)[None, :]
    f = torch.sqrt(fx**2 + fy**2)
    f[0, 0] = 1.0
    amp = f ** (-alpha / 2)
    amp[0, 0] = 0.0  # drop DC
    shaped = torch.fft.ifft2(torch.fft.fft2(white) * amp).real
    return _standardise(shaped)[:, None]


def low_rank(batch: int, rank: int, gen: torch.Generator) -> torch.Tensor:
    u = torch.randn(batch, SIDE, rank, generator=gen)
    v = torch.randn(batch, rank, SIDE, generator=gen)
    return _standardise(u @ v)[:, None]


def sample(kind: str, batch: int, gen: torch.Generator, **kw) -> torch.Tensor:
    scale = float(kw.pop("scale", 1.0))  # optional amplitude multiplier (diagnostics only)
    return _sample(kind, batch, gen, **kw) * scale


def _sample(kind: str, batch: int, gen: torch.Generator, **kw) -> torch.Tensor:
    if kind == "uniform":
        x = torch.rand(batch, 1, SIDE, SIDE, generator=gen) * 2 - 1
        return x + kw.get("dc", 0.0)
    if kind == "perlin":
        x = perlin(batch, int(kw.get("res", 8)), gen)
        if kw.get("peak_normalise"):  # as in the authors' code: each image scaled to max |x| = 1
            return x / (x.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8)
        return _standardise(x) if kw.get("standardise") else x
    # ---- fixed-RMS families for the step-1 sweep (after Madl 2026), RMS = std of U(-1, 1)
    if kind == "dcfrac":
        # a fraction beta of the RMS is a constant (DC) offset, the rest white noise
        beta = float(kw["beta"])
        z = (torch.rand(batch, 1, SIDE, SIDE, generator=gen) * 2 - 1) * math.sqrt(3)
        return UNIFORM_STD * (beta + math.sqrt(1 - beta**2) * z)
    if kind == "shape":
        dist = kw["dist"]
        shape = (batch, 1, SIDE, SIDE)
        if dist == "uniform":
            x = torch.rand(shape, generator=gen) * 2 - 1
        elif dist == "gaussian":
            x = torch.randn(shape, generator=gen)
        elif dist == "rademacher":
            x = torch.randint(0, 2, shape, generator=gen).float() * 2 - 1
        elif dist == "laplace":
            u = torch.rand(shape, generator=gen) - 0.5
            x = -torch.sign(u) * torch.log1p(-2 * u.abs())
        elif dist == "sparse":
            x = torch.randn(shape, generator=gen) * (torch.rand(shape, generator=gen) < 0.1)
        elif dist == "student_t3":
            z = torch.randn(shape, generator=gen)
            chi2 = torch.randn((3, *shape), generator=gen).pow(2).sum(0)
            x = z / torch.sqrt(chi2 / 3)
        else:
            raise ValueError(f"unknown shape {dist!r}")
        return _standardise(x)
    if kind == "mixture":
        # each image comes from spec "a" with probability p, else from spec "b"
        a, b = dict(kw["a"]), dict(kw["b"])
        xa, xb = _sample(a.pop("kind"), batch, gen, **a), _sample(b.pop("kind"), batch, gen, **b)
        pick = (torch.rand(batch, generator=gen) < float(kw["p"])).float().reshape(-1, 1, 1, 1)
        return pick * xa + (1 - pick) * xb
    if kind == "gaussian":
        return torch.randn(batch, 1, SIDE, SIDE, generator=gen) * UNIFORM_STD
    if kind == "rademacher":
        return (torch.randint(0, 2, (batch, 1, SIDE, SIDE), generator=gen).float() * 2 - 1) * UNIFORM_STD
    if kind == "spectral":
        return spectral(batch, float(kw["alpha"]), gen)
    if kind == "lowrank":
        return low_rank(batch, int(kw["rank"]), gen)
    raise ValueError(f"unknown noise kind {kind!r}")
