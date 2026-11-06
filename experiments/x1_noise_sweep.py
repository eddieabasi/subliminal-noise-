"""X1 - coverage screen (Madl 2026, Sec. 3) on a noise-family sweep under a fully
shared initialisation, m = 10. Coverage is also logged for every E2/E4 run."""
from .common import SEEDS5, job

CONDITIONS = {
    "uniform": {"kind": "uniform"},
    "gaussian": {"kind": "gaussian"},
    "rademacher": {"kind": "rademacher"},
    "dc0.5": {"kind": "uniform", "dc": 0.5},
    "dc1": {"kind": "uniform", "dc": 1.0},
    "dc2": {"kind": "uniform", "dc": 2.0},
    "pink": {"kind": "spectral", "alpha": 1.0},
    "brown": {"kind": "spectral", "alpha": 2.0},
    "blue": {"kind": "spectral", "alpha": -1.0},
    "rank1": {"kind": "lowrank", "rank": 1},
    "rank4": {"kind": "lowrank", "rank": 4},
    "perlin4": {"kind": "perlin", "res": 4},
}


def jobs():
    return [job("X1", s, [{"name": n, "noise": spec, "condition": n} for n, spec in CONDITIONS.items()],
                cost=12, m=10) for s in SEEDS5]
