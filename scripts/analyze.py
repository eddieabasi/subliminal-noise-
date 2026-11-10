"""Summarise results/runs.jsonl into tables (results/summary.md, results/summary.json)
and figures (results/figures/*.png).

Uncertainty: mean over seeds with a 95% bootstrap CI over seeds, as in the paper.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = RES / "figures"

# Reference categorical palette (light), fixed order; text in neutral inks.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
CHANCE = 0.10

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9, "axes.titlesize": 10,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 5,
})


def auroc(scores, labels) -> float:
    """Mann-Whitney AUROC: P(score of a positive > score of a negative)."""
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / diff.size)


def boot_ci(values, n=10_000, seed=0):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarise(df: pd.DataFrame, keys: list[str], col: str = "student_acc") -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(keys, sort=False):
        mean, lo, hi = boot_ci(g[col])
        rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))),
                     "mean": mean, "lo": lo, "hi": hi, "n": len(g)})
    return pd.DataFrame(rows)


def errplot(ax, x, s, color, label=None, marker="o", ls="-"):
    ax.errorbar(x, s["mean"], yerr=[s["mean"] - s["lo"], s["hi"] - s["mean"]], color=color, label=label,
                marker=marker, ls=ls, capsize=2, elinewidth=1)


def chance(ax):
    ax.axhline(CHANCE, color=INK2, lw=1, ls=":")


def fmt(s) -> str:
    return f"{100 * s['mean']:.1f}% [{100 * s['lo']:.1f}, {100 * s['hi']:.1f}]"


def load() -> pd.DataFrame:
    rows = [json.loads(line) for line in (RES / "runs.jsonl").read_text().splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["student_cfg_raw"] = df["student_cfg"]
    scfg = pd.json_normalize(df["student_cfg"]).add_prefix("s.")
    tcfg = pd.json_normalize(df["teacher_cfg"]).add_prefix("t.")
    return pd.concat([df.drop(columns=["student_cfg", "teacher_cfg"]), scfg, tcfg], axis=1)


# ---------------------------------------------------------------- experiments
def e1(df, out):
    d = df[df.exp == "E1"]
    if d.empty:
        return
    t, s = summarise(d, ["exp"], "teacher_acc").iloc[0], summarise(d, ["exp"]).iloc[0]
    out["E1"] = {"teacher": t.to_dict(), "student": s.to_dict()}
    out["md"] += ["## E1 Baseline (Table 1 setup, m = 10, shared init)\n",
                  "| Model | Accuracy (mean, 95% CI) | Seeds |", "| --- | --- | --- |",
                  f"| Teacher | {fmt(t)} | {t.n} |", f"| Student | {fmt(s)} | {s.n} |", ""]


E2_TITLES = {"shared": "All shared", "reinit_l1": "Reinit layer 1", "reinit_l2": "Reinit layer 2",
             "reinit_hidden": "Reinit both hidden", "reinit_aux": "Reinit aux head",
             "reinit_class": "Reinit class head"}


def e2(df, out):
    d = df[df.exp == "E2"]
    if d.empty:
        return
    s = summarise(d, ["s.condition", "s.m"])
    t = summarise(d, ["s.m"], "teacher_acc")
    ms = sorted(d["s.m"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.2), sharex=True, sharey=True)
    for ax, (cond, title) in zip(axes.flat, E2_TITLES.items()):
        sc = s[s["s.condition"] == cond].set_index("s.m").loc[ms]
        ax.plot(ms, t.set_index("s.m").loc[ms]["mean"], color=INK2, lw=1, ls="--", label="teacher")
        errplot(ax, ms, sc, C[0], "student")
        chance(ax)
        ax.set_title(title, color=INK)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)
    for ax in axes[1]:
        ax.set_xlabel("aux logits m")
    for ax in axes[:, 0]:
        ax.set_ylabel("MNIST test accuracy")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle(f"E2: student accuracy when parts of the student are reinitialised ({d.seed.nunique()} seeds, 95% CI)",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "e2_init_ablation.png")
    plt.close(fig)
    out["E2"] = s.to_dict("records")
    header = "| Condition | " + " | ".join(f"m = {int(m)}" for m in ms) + " |"
    out["md"] += ["## E2 Which shared parts matter (Fig. 2)\n", header, "| --- |" + " --- |" * len(ms)]
    for cond, title in E2_TITLES.items():
        sc = s[s["s.condition"] == cond].set_index("s.m")
        out["md"].append(f"| {title} | " + " | ".join(fmt(sc.loc[m]) for m in ms) + " |")
    out["md"].append("| Teacher | " + " | ".join(fmt(t.set_index("s.m").loc[m]) for m in ms) + " |\n")


def e3(df, out):
    d = df[df.exp == "E3"]
    if d.empty:
        return
    t, s = summarise(d, ["exp"], "teacher_acc").iloc[0], summarise(d, ["exp"]).iloc[0]
    out["E3"] = {"teacher": t.to_dict(), "student": s.to_dict()}
    out["md"] += ["## E3 Best case (Tables 3-4)\n", "| Model | Ours | Paper (20 seeds) |", "| --- | --- | --- |",
                  f"| Teacher | {fmt(t)} | 96.86% [96.81, 96.91] |",
                  f"| Student | {fmt(s)} | 96.82% [96.77, 96.87] |", ""]


def e4(df, out):
    d = df[df.exp == "E4"].copy()
    if d.empty:
        return
    base = d[d["s.head"] == "none"]
    parts = []
    for head in ("aux", "class"):
        b = base.copy()
        b["s.head"] = head
        parts.append(pd.concat([b, d[d["s.head"] == head]]))
    d2 = pd.concat(parts)
    acc = summarise(d2, ["s.head", "s.delta"])
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for i, head in enumerate(("aux", "class")):
        a = acc[acc["s.head"] == head].sort_values("s.delta")
        errplot(axes[0], a["s.delta"], a, C[i], f"{head} head perturbed")
        cos_f = summarise(d2[d2["s.head"] == head], ["s.delta"], f"{head}_cos_final").sort_values("s.delta")
        cos_i = summarise(d2[d2["s.head"] == head], ["s.delta"], f"{head}_cos_init").sort_values("s.delta")
        ax = axes[1 + i]
        errplot(ax, cos_i["s.delta"], cos_i, C[i], "before distillation", ls="--", marker="s")
        errplot(ax, cos_f["s.delta"], cos_f, C[i], "after distillation")
        ax.set_title(f"{head.capitalize()} head: cosine to teacher's", color=INK)
        ax.set_xlabel("perturbation scale δ")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower left")
    chance(axes[0])
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("perturbation scale δ")
    axes[0].set_ylabel("student MNIST accuracy")
    axes[0].set_title("Student accuracy", color=INK)
    axes[0].legend(loc="upper right")
    fig.suptitle(f"E4: perturbing the student's heads before distillation ({d.seed.nunique()} seeds, 95% CI)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "e4_perturbation.png")
    plt.close(fig)
    out["E4"] = acc.to_dict("records")
    deltas = sorted(d2["s.delta"].unique())
    out["md"] += ["## E4 Head perturbation (Fig. 5a-d)\n",
                  "| Head | " + " | ".join(f"δ = {x:g}" for x in deltas) + " |", "| --- |" + " --- |" * len(deltas)]
    for head in ("aux", "class"):
        a = acc[acc["s.head"] == head].set_index("s.delta")
        out["md"].append(f"| {head} | " + " | ".join(fmt(a.loc[x]) for x in deltas) + " |")
    ai = summarise(d2[d2["s.head"] == "aux"], ["s.delta"], "aux_cos_init").set_index("s.delta")
    af = summarise(d2[d2["s.head"] == "aux"], ["s.delta"], "aux_cos_final").set_index("s.delta")
    out["md"].append("| aux-head cosine to teacher, before → after | " +
                     " | ".join(f"{ai.loc[x, 'mean']:.3f} → {af.loc[x, 'mean']:.3f}" for x in deltas) + " |\n")


def e5(df, out):
    d = df[df.exp == "E5"]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
    w = summarise(d[d["s.width"].notna()], ["s.width"]).sort_values("s.width")
    errplot(axes[0], w["s.width"], w, C[0], "student")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("student first hidden width (teacher: 256)")
    axes[0].set_ylabel("student MNIST accuracy")
    axes[0].set_title("Width sweep, m = 50, heads shared only", color=INK)
    chance(axes[0])
    axes[0].set_ylim(0, 1)
    kinds = ["depth1", "depth3", "cnn_perlin8", "cnn_uniform"]
    labels = ["MLP, 1 hidden", "MLP, 3 hidden", "CNN, Perlin noise", "CNN, uniform noise"]
    ticks = ["MLP\n1 hidden", "MLP\n3 hidden", "CNN\nPerlin noise", "CNN\nuniform noise"]
    x = np.arange(len(kinds))
    rows = []
    for i, m in enumerate((10, 50)):
        s = [summarise(d[d.student == f"m{m}_{k}"], ["exp"]).iloc[0] for k in kinds]
        rows += [{"m": m, "kind": k, **si.to_dict()} for k, si in zip(kinds, s)]
        means = np.array([si["mean"] for si in s])
        err = np.array([[si["mean"] - si["lo"] for si in s], [si["hi"] - si["mean"] for si in s]])
        axes[1].bar(x + (i - 0.5) * 0.38, means, 0.36, yerr=err, color=C[i], label=f"m = {m}", capsize=2,
                    edgecolor="white", linewidth=1, error_kw={"ecolor": INK2, "elinewidth": 1})
    axes[1].set_xticks(x, ticks)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Different student architectures, heads shared only", color=INK)
    axes[1].legend(loc="upper right")
    chance(axes[1])
    fig.suptitle(f"E5: architecture and capacity ({d.seed.nunique()} seeds, 95% CI)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "e5_architecture.png")
    plt.close(fig)
    out["E5"] = {"width": w.to_dict("records"), "arch": rows}
    out["md"] += ["## E5 Architecture and capacity (Fig. 3a-b)\n", "| Student | m = 10 | m = 50 |", "| --- | --- | --- |"]
    for k, lab in zip(kinds, labels):
        r = {row["m"]: row for row in rows if row["kind"] == k}
        out["md"].append(f"| {lab} | {fmt(r[10])} | {fmt(r[50])} |")
    out["md"] += ["", "| Student first width (m = 50) | Accuracy |", "| --- | --- |"]
    out["md"] += [f"| {int(r['s.width'])} | {fmt(r)} |" for _, r in w.iterrows()] + [""]
    peak = df[df.exp == "E8"][df[df.exp == "E8"].student.str.contains("cnn_perlin8_peak")]
    if not peak.empty:
        sp = summarise(peak, ["student"])
        out["E8_cnn"] = sp.to_dict("records")
        out["md"] += ["### E8 CNN student with the authors' Perlin noise (peak-normalised per image)\n",
                      "| Student | Accuracy | Seeds |", "| --- | --- | --- |"]
        out["md"] += [f"| CNN, m = {r['student'].split('_')[0][1:]} | {fmt(r)} | {int(r['n'])} |" for _, r in sp.iterrows()]
        out["md"].append("")
    diag = df[df.exp == "E5b"]
    if not diag.empty:
        sd = summarise(diag, ["student"])
        out["E5b"] = sd.to_dict("records")
        names = {"m50_cnn_perlin8_x1.89": "CNN, m = 50, Perlin rescaled to uniform's std (x1.89)",
                 "m250_cnn_perlin8": "CNN, m = 250, Perlin as in E5",
                 "m250_mlp_depth3_ref": "MLP with 3 hidden layers, m = 250 (reference)"}
        out["md"] += ["### E5b Off-paper CNN diagnostic\n", "| Student | Accuracy | Seeds |", "| --- | --- | --- |"]
        out["md"] += [f"| {names.get(r['student'], r['student'])} | {fmt(r)} | {int(r['n'])} |" for _, r in sd.iterrows()]
        out["md"].append("")


def e6(df, out):
    d = df[df.exp.isin(["E6", "E8"]) & (df.student.str.startswith("m1_N") | df.student.str.startswith("N1000_m"))]
    if d.empty:
        return
    a = summarise(d[d.student.str.startswith("m1_")], ["s.N"]).sort_values("s.N")
    b = summarise(d[d.student.str.startswith("N1000_")], ["s.m"]).sort_values("s.m")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    errplot(axes[0], a["s.N"], a, C[0])
    axes[0].axhline(0.83, color=INK2, lw=1, ls="--")
    axes[0].text(a["s.N"].min(), 0.85, "paper's peak ≈ 83%", color=INK2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("noise samples per epoch N")
    axes[0].set_ylabel("student MNIST accuracy")
    axes[0].set_title("One aux logit (m = 1)", color=INK)
    errplot(axes[1], b["s.m"], b, C[0])
    axes[1].axhline(0.29, color=INK2, lw=1, ls="--")
    axes[1].text(b["s.m"].min(), 0.31, "paper's plateau ≈ 29%", color=INK2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("aux logits m")
    axes[1].set_title("1,000 noise samples per epoch", color=INK)
    for ax in axes:
        chance(ax)
        ax.set_ylim(0, 1)
    fig.suptitle("E6 + E8: m and N are joint bottlenecks (5 seeds; 3 for N > 3 million; 95% CI)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "e6_m_and_n.png")
    plt.close(fig)
    out["E6"] = {"m1_vs_N": a.to_dict("records"), "N1000_vs_m": b.to_dict("records")}
    out["md"] += ["## E6 m and N (Fig. 4b-c)\n", "| m = 1, N per epoch | Accuracy |", "| --- | --- |"]
    out["md"] += [f"| {int(r['s.N']):,} | {fmt(r)} |" for _, r in a.iterrows()]
    out["md"] += ["", "| N = 1,000, m | Accuracy |", "| --- | --- |"]
    out["md"] += [f"| {int(r['s.m'])} | {fmt(r)} |" for _, r in b.iterrows()] + [""]
    # paired decline after the peak: same seeds at N = 10^6 and N = 8.2 x 10^6
    a1 = d[d.student == "m1_N1000000"].set_index("seed").student_acc
    a8 = d[d.student == "m1_N8192000"].set_index("seed").student_acc
    both = a1.index.intersection(a8.index)
    if len(both):
        diff = (a8.loc[both] - a1.loc[both]) * 100
        out["E6_decline"] = {"seeds": list(map(int, both)), "diff_points": diff.round(1).tolist()}
        out["md"] += [f"Paired change from N = 1,000,000 to N = 8,192,000 (seeds {', '.join(map(str, both))}): "
                      + ", ".join(f"{x:+.1f}" for x in diff) + f" points (mean {diff.mean():+.1f}).\n"]


E7_REGIMES = [("free", "both heads train"), ("aux_fixed", "aux head fixed"),
              ("class_fixed", "class head fixed"), ("both_fixed", "both fixed")]


def e7(df, out):
    d = df[df.exp == "E7"]
    if d.empty:
        return
    s = summarise(d, ["s.heads", "s.d"])
    t = summarise(d[d["s.heads"] == "free"], ["s.d"], "teacher_acc").sort_values("s.d")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    axes[0].plot(t["s.d"], t["mean"], color=INK2, lw=1, ls="--", label="teacher")
    for i, (reg, lab) in enumerate(E7_REGIMES):
        r = s[s["s.heads"] == reg].sort_values("s.d")
        errplot(axes[0], r["s.d"], r, C[i], lab)
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("latent width d")
    axes[0].set_ylabel("student MNIST accuracy")
    axes[0].set_title("Student accuracy", color=INK)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    chance(axes[0])
    free = d[d["s.heads"] == "free"]
    for i, (col, lab) in enumerate([("tchange_class", "teacher class head"), ("schange_aux", "student aux head")]):
        r = summarise(free, ["s.d"], col).sort_values("s.d")
        errplot(axes[1], r["s.d"], r, C[i + 4], lab)
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("latent width d")
    axes[1].set_ylabel("relative weight change ‖Δw‖/‖w₀‖")
    axes[1].set_title("Head drift during training (both heads train)", color=INK)
    axes[1].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"E7: latent dimension ({d.seed.nunique()} seeds, 95% CI)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "e7_latent_dim.png")
    plt.close(fig)
    out["E7"] = s.to_dict("records")
    ds = sorted(d["s.d"].unique())
    out["md"] += ["## E7 Latent dimension (Fig. 6a, Fig. 13)\n",
                  "| Heads | " + " | ".join(f"d = {int(x)}" for x in ds) + " |", "| --- |" + " --- |" * len(ds)]
    for reg, lab in E7_REGIMES:
        r = s[s["s.heads"] == reg].set_index("s.d")
        out["md"].append(f"| {lab} | " + " | ".join(fmt(r.loc[x]) for x in ds) + " |")
    out["md"].append("| teacher | " + " | ".join(fmt(t.set_index("s.d").loc[x]) for x in ds) + " |\n")


def _mismatch(row) -> str:
    """Which part of the student differs from the teacher's initialisation."""
    cfg = row["student_cfg_raw"]
    if row["exp"] == "E4":
        return {"none": "nothing", "aux": "aux head", "class": "class head"}[cfg.get("head")]
    return {"shared": "nothing", "reinit_aux": "aux head", "reinit_class": "class head"}.get(
        cfg.get("condition"), "hidden layers")


def e9(df, out):
    d = df[df.exp == "E9"]
    if d.empty:
        return
    st = summarise(d, ["s.classes"]).sort_values("s.classes")
    te = summarise(d, ["s.classes"], "teacher_acc").sort_values("s.classes")
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.plot(te["s.classes"], te["mean"], color=INK2, lw=1, ls="--", label="teacher")
    errplot(ax, st["s.classes"], st, C[0], "student (heads shared only, m = 50)")
    ks = np.arange(2, 48)
    ax.plot(ks, 1 / ks, color=INK2, lw=1, ls=":", label="chance (1/K)")
    ax.set_xlabel("number of EMNIST classes K")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="center right", fontsize=8)
    ax.set_title(f"E9: EMNIST class-count sweep ({d.seed.nunique()} seeds, 95% CI)", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "e9_emnist.png")
    plt.close(fig)
    out["E9"] = {"student": st.to_dict("records"), "teacher": te.to_dict("records")}
    out["md"] += ["## E9 EMNIST class-count sweep (Fig. 3c)\n", "| Classes K | Teacher | Student | Chance |",
                  "| --- | --- | --- | --- |"]
    for (_, s_), (_, t_) in zip(st.iterrows(), te.iterrows()):
        k = int(s_["s.classes"])
        out["md"].append(f"| {k} | {fmt(t_)} | {fmt(s_)} | {100 / k:.1f}% |")
    out["md"].append("")


def x1(df, out):
    d = df[df.exp == "X1"]
    if d.empty:
        return
    pooled = d.groupby("s.condition")[["coverage_tdisp", "student_acc"]].mean()
    rho_pooled = spearmanr(pooled.coverage_tdisp, pooled.student_acc).statistic
    rho_runs = spearmanr(d.coverage_tdisp, d.student_acc).statistic
    res = {"noise_sweep": {"spearman_pooled": rho_pooled, "spearman_runs": rho_runs,
                           "n_conditions": len(pooled), "rows": pooled.reset_index().to_dict("records")}}

    # E2 + E4 runs, grouped by which component is mismatched (m = 10 only, so coverage is comparable).
    mm = df[df.exp.isin(["E2", "E4"]) & (df["s.m"].isna() | (df["s.m"] == 10))].copy()
    mm["mismatch"] = mm.apply(_mismatch, axis=1)
    groups = {}
    for g, gd in mm.groupby("mismatch"):
        groups[g] = {"n": len(gd), "coverage": float(gd.coverage_tdisp.mean()),
                     "acc": float(gd.student_acc.mean()),
                     "spearman": float(spearmanr(gd.coverage_tdisp, gd.student_acc).statistic)
                     if gd.coverage_tdisp.nunique() > 1 else float("nan")}
    res["mismatch_groups_m10"] = groups
    no_cls = mm[mm.mismatch != "class head"]
    res["spearman_m10_without_class_head"] = float(spearmanr(no_cls.coverage_tdisp, no_cls.student_acc).statistic)
    res["spearman_m10_all"] = float(spearmanr(mm.coverage_tdisp, mm.student_acc).statistic)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    ax = axes[0]
    ax.scatter(pooled.coverage_tdisp, pooled.student_acc, color=C[0], s=30, zorder=3)
    near = (pooled.coverage_tdisp - pooled.loc["uniform", "coverage_tdisp"]).abs() < 0.0015
    near &= (pooled.student_acc - pooled.loc["uniform", "student_acc"]).abs() < 0.05
    for name, r in pooled[~near].iterrows():
        ax.annotate(name, (r.coverage_tdisp, r.student_acc), textcoords="offset points", xytext=(4, 3),
                    fontsize=7, color=INK2)
    cluster = pooled[near]
    ax.annotate(", ".join(cluster.index), (cluster.coverage_tdisp.mean(), cluster.student_acc.max()),
                textcoords="offset points", xytext=(-60, 8), fontsize=7, color=INK2)
    ax.set_xlabel("coverage, computed before training")
    ax.set_ylabel("student MNIST accuracy after training")
    ax.set_title(f"12 noise types, shared init: Spearman ρ = {rho_pooled:.2f}", color=INK)
    chance(ax)
    ax = axes[1]
    styles = [("hidden layers", C[0]), ("aux head", C[1]), ("class head", C[2])]
    nothing = mm[mm.mismatch == "nothing"]
    ax.scatter(nothing.coverage_tdisp, nothing.student_acc, color=INK2, s=12, label="nothing (shared)", zorder=2,
               edgecolor="white", linewidth=0.5)
    for g, col in styles:
        gd = mm[mm.mismatch == g]
        ax.scatter(gd.coverage_tdisp, gd.student_acc, color=col, s=12, label=f"{g} differ", zorder=3,
                   edgecolor="white", linewidth=0.5)
    ax.set_xlabel("coverage, computed before training")
    ax.set_title(f"E2 + E4 runs at m = 10: ρ = {res['spearman_m10_all']:.2f}, "
                 f"{res['spearman_m10_without_class_head']:.2f} without class-head runs", color=INK)
    ax.legend(loc="upper left", fontsize=8)
    chance(ax)
    fig.suptitle("X1: coverage sees the body and aux head, but is blind to the class head", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "x1_coverage.png")
    plt.close(fig)
    out["X1"] = res
    out["md"] += ["## X1 Coverage screen (Madl 2026)\n",
                  f"Noise sweep, shared init, m = 10: Spearman ρ (coverage vs accuracy) = {rho_pooled:.2f} over "
                  f"{len(pooled)} seed-pooled conditions ({rho_runs:.2f} over {len(d)} runs). Madl reports 0.95.\n",
                  "| Noise | Coverage | Student accuracy |", "| --- | --- | --- |"]
    out["md"] += [f"| {n} | {r.coverage_tdisp:.3f} | {100 * r.student_acc:.1f}% |"
                  for n, r in pooled.sort_values("coverage_tdisp").iterrows()]
    out["md"] += ["", f"E2 + E4 runs at m = 10 (Spearman {res['spearman_m10_all']:.2f} overall, "
                  f"{res['spearman_m10_without_class_head']:.2f} without class-head runs):\n",
                  "| What differs from the teacher | Runs | Mean coverage | Mean accuracy | Spearman within group |",
                  "| --- | --- | --- | --- | --- |"]
    for g, r in groups.items():
        out["md"].append(f"| {g} | {r['n']} | {r['coverage']:.4f} | {100 * r['acc']:.1f}% | {r['spearman']:.2f} |")
    out["md"].append("")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    out: dict = {"md": ["# Results\n", "Mean over seeds with a 95% bootstrap CI over seeds. Chance = 10%.\n"]}
    for fn in (e1, e2, e3, e4, e5, e6, e7, e9, x1):
        fn(df, out)
    timing = df.groupby("exp")["seconds_student"].mean()
    out["md"] += ["## Run time\n", "| Experiment | Mean seconds per student run (1 core) |", "| --- | --- |"]
    out["md"] += [f"| {k} | {v:.0f} |" for k, v in timing.items()] + [""]
    (RES / "summary.md").write_text("\n".join(out.pop("md")))
    (RES / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print((RES / "summary.md").read_text())


if __name__ == "__main__":
    main()
