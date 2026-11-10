"""X2 - step 1 follow-up: does Madl's coverage law (Spearman ~0.95) come back in a
setup closer to his?

Noise: 36 conditions at a fixed input RMS (the std of U(-1, 1)) in Madl's five
families (DC level, rank, spectral slope, distribution shape, mixtures), plus a
Perlin family that probes our smooth-noise failure in X1.

Settings, all with a fully shared initialisation:
* ``madl_kl`` : a = 3 aux logits, soft-label KL channel (closest to Madl's appendix)
* ``madl_mse``: a = 3 aux logits, MSE channel
* ``ours``    : m = 10 aux logits, MSE channel (our X1 setting)
"""
from .common import job

SEEDS = range(3)
UNI = {"kind": "shape", "dist": "uniform"}

FAMILIES = {
    "dc": {f"beta{b}": {"kind": "dcfrac", "beta": b} for b in (0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95)},
    "rank": {f"rank{r}": {"kind": "lowrank", "rank": r} for r in (1, 2, 4, 8, 16, 28)},
    "spectral": {f"alpha{a}": {"kind": "spectral", "alpha": a} for a in (-2, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3)},
    "shape": {d: {"kind": "shape", "dist": d}
              for d in ("uniform", "gaussian", "rademacher", "laplace", "sparse", "student_t3")},
    "mixture": {f"p{p}": {"kind": "mixture", "p": p, "a": {"kind": "dcfrac", "beta": 0.9}, "b": UNI}
                for p in (0.1, 0.3, 0.5, 0.7, 0.9)},
    "perlin": {f"res{r}": {"kind": "perlin", "res": r, "standardise": True} for r in (2, 4, 8)},
}
SETTINGS = {"madl_kl": (3, "kl"), "madl_mse": (3, "mse"), "ours": (10, "mse")}


def jobs():
    out = []
    for s in SEEDS:
        for setting, (m, loss) in SETTINGS.items():
            for fam, conds in FAMILIES.items():
                students = [{"name": f"{setting}_{fam}_{c}", "setting": setting, "family": fam, "condition": c,
                             "noise": spec, "loss": loss, "m": m} for c, spec in conds.items()]
                out.append(job("X2", s, students, cost=len(students), m=m))
    return out
