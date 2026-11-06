"""E5 - architecture and capacity (Fig. 3a-b). Teacher and student share only the
two heads; the student's body is independently initialised.

* width sweep: student MLP (w, 256), teacher MLP (256, 256), m = 50
* depth: student MLP with one fewer / one more hidden layer, m in {10, 50}
* CNN: teacher MLP (256, 256, 256), CNN student (Table 2) on Perlin noise (res 8);
  a CNN-on-uniform-noise control as in Fig. 10
"""
from .common import SEEDS5, job

WIDTHS = [32, 64, 128, 256, 512, 1024]


def jobs():
    out = []
    for s in SEEDS5:
        for m in (10, 50):
            students = [
                {"name": f"m{m}_depth1", "arch": "mlp", "widths": [256], "m": m},
                {"name": f"m{m}_depth3", "arch": "mlp", "widths": [256, 256, 256], "m": m},
            ]
            if m == 50:
                students += [{"name": f"m50_width{w}", "arch": "mlp", "widths": [w, 256], "m": 50, "width": w}
                             for w in WIDTHS]
            out.append(job("E5", s, students, cost=len(students), m=m))
            cnn = [
                {"name": f"m{m}_cnn_perlin8", "arch": "cnn", "widths": [32, 128, 256], "m": m,
                 "noise": {"kind": "perlin", "res": 8}},
                {"name": f"m{m}_cnn_uniform", "arch": "cnn", "widths": [32, 128, 256], "m": m},
            ]
            out.append(job("E5", s, cnn, cost=8, m=m, widths=[256, 256, 256]))
    return out
