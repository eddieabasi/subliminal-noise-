"""E6 - m and N as joint bottlenecks (Fig. 4b-c). Baseline shared init; noise batch
1000, so N = 1000 * steps per epoch.

* m = 1, N from 10^3 to 3 * 10^6 per epoch
* N = 10^3 (one step per epoch), m from 1 to 500
"""
from .common import SEEDS5, job

STEPS = [1, 10, 60, 200, 1000, 3000]
MS = [1, 10, 50, 100, 250, 500]


def jobs():
    out = []
    for s in SEEDS5:
        out.append(job("E6", s, [{"name": f"m1_N{1000 * k}", "steps": k, "m": 1, "N": 1000 * k} for k in STEPS],
                       cost=sum(STEPS) / 60, m=1))
        for m in MS:
            out.append(job("E6", s, [{"name": f"N1000_m{m}", "steps": 1, "m": m, "N": 1000}], cost=0.3, m=m))
    return out
