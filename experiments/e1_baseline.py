"""E1 - baseline gate (Table 1): shared init, m = 10, uniform noise, 20 seeds."""
from .common import SEEDS20, job


def jobs():
    return [job("E1", s, [{"name": "shared"}], m=10) for s in SEEDS20]
