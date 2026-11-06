"""E5b - off-paper diagnostic for the weak CNN result in E5.

Our Perlin noise (res 8) has std 0.31, about half of U(-1, 1)'s 0.58; the paper
does not state its Perlin amplitude. We rerun the CNN student (a) with Perlin
noise rescaled to std 0.58 (x 1.89) at m = 50, and (b) with the paper's Perlin
noise at m = 250, where heads-only transfer was strong for MLPs in E2."""
from .common import SEEDS5, job

CNN = {"arch": "cnn", "widths": [32, 128, 256]}


def jobs():
    out = []
    for s in SEEDS5:
        out.append(job("E5b", s, [{**CNN, "name": "m50_cnn_perlin8_x1.89", "m": 50,
                                   "noise": {"kind": "perlin", "res": 8, "scale": 1.89}}],
                       cost=4, m=50, widths=[256, 256, 256]))
        out.append(job("E5b", s, [{**CNN, "name": "m250_cnn_perlin8", "m": 250,
                                   "noise": {"kind": "perlin", "res": 8}},
                                  {"name": "m250_mlp_depth3_ref", "arch": "mlp", "widths": [256, 256, 256], "m": 250}],
                       cost=5, m=250, widths=[256, 256, 256]))
    return out
