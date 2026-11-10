"""E9 - EMNIST class-count sweep (Fig. 3c), as in the authors' emnist_sweep.sh:
EMNIST balanced, first K classes, m = 50, student hidden layers random (heads shared)."""
from .common import SEEDS5, job

KS = [2, 5, 10, 20, 30, 47]


def jobs():
    return [job("E9", s, [{"name": f"K{k}", "reinit": ["l1", "l2"], "m": 50, "classes": k}],
                cost=1 + k / 47, m=50, dataset="emnist", classes=k)
            for s in SEEDS5 for k in KS]
