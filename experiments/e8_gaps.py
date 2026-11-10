"""E8 - step 3: close the replication gaps found in E5/E6, using what the authors'
released code (Priesemann-Group/subliminal_learning) does differently.

* CNN student on Perlin noise peak-normalised per image (max |x| = 1), as in the
  authors' sample_perlin_noise; our E5 Perlin had about half that amplitude.
* m = 1: extend noise steps per epoch to the authors' largest values (5,793 and
  8,192 steps, N up to 8.2 million) to see whether accuracy declines after its peak.
* N = 1,000: extend m to 1,024, 2,896 and 8,192 to see whether accuracy plateaus.
"""
from .common import SEEDS5, job

CNN = {"arch": "cnn", "widths": [32, 128, 256]}


def jobs():
    out = []
    for s in SEEDS5:
        for m in (10, 50):
            out.append(job("E8", s, [{**CNN, "name": f"m{m}_cnn_perlin8_peak", "m": m,
                                      "noise": {"kind": "perlin", "res": 8, "peak_normalise": True}}],
                           cost=6, m=m, widths=[256, 256, 256]))
        for m in (1024, 2896, 8192):
            out.append(job("E8", s, [{"name": f"N1000_m{m}", "steps": 1, "m": m, "N": 1000}], cost=0.3, m=m))
    for s in range(3):
        for k in (5793, 8192):
            out.append(job("E8", s, [{"name": f"m1_N{1000 * k}", "steps": k, "m": 1, "N": 1000 * k}],
                           cost=k / 60, m=1))
    return out
