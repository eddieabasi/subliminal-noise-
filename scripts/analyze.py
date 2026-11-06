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
    x = np.arange(len(kinds))
    rows = []
    for i, m in enumerate((10, 50)):
        s = [summarise(d[d.student == f"m{m}_{k}"], ["exp"]).iloc[0] for k in kinds]
        rows += [{"m": m, "kind": k, **si.to_dict()} for k, si in zip(kinds, s)]
        means = np.array([si["mean"] for si in s])
        err = np.array([[si["mean"] - si["lo"] for si in s], [si["hi"] - si["mean"] for si in s]])
        axes[1].bar(x + (i - 0.5) * 0.38, means, 0.36, yerr=err, color=C[i], label=f"m = {m}", capsize=2,
                    edgecolor="white", linewidth=1)
    axes[1].set_xticks(x, labels)
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


def e6(df, out):
    d = df[df.exp == "E6"]
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
    fig.suptitle(f"E6: m and N are joint bottlenecks ({d.seed.nunique()} seeds, 95% CI)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "e6_m_and_n.png")
    plt.close(fig)
    out["E6"] = {"m1_vs_N": a.to_dict("records"), "N1000_vs_m": b.to_dict("records")}
    out["md"] += ["## E6 m and N (Fig. 4b-c)\n", "| m = 1, N per epoch | Accuracy |", "| --- | --- |"]
    out["md"] += [f"| {int(r['s.N']):,} | {fmt(r)} |" for _, r in a.iterrows()]
    out["md"] += ["", "| N = 1,000, m | Accuracy |", "| --- | --- |"]
    out["md"] += [f"| {int(r['s.m'])} | {fmt(r)} |" for _, r in b.iterrows()] + [""]


E7_REGIMES = [("free", "both heads train"), ("aux_fixed", "aux head fixed"),
              ("class_fixed", "class head fixed"), ("both_fixed", "both fixed")]


def e7(df, out):
    d = df[df.exp == "E7"]
    if d.empty:
        return
    s = summarise(d, ["s.heads", "s.d"])
    t = summarise(d[d["s.heads"] == "free"], ["s.d"], "teacher_acc").sort_values("s.d")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(t["s.d"], t["mean"], color=INK2, lw=1, ls="--", label="teacher")
    for i, (reg, lab) in enumerate(E7_REGIMES):
        r = s[s["s.heads"] == reg].sort_values("s.d")
        errplot(axes[0], r["s.d"], r, C[i], lab)
    axes[0].set_xscale("log", base=2)
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("latent width d")
    axes[0].set_ylabel("student MNIST accuracy")
    axes[0].set_title("Student accuracy", color=INK)
    axes[0].legend(loc="lower left", fontsize=8)
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


def x1(df, out):
    d = df[df.exp == "X1"]
    e2d = df[(df.exp == "E2") & df["coverage_tdisp"].notna()].copy()
    if d.empty:
        return
    pooled = d.groupby("s.condition")[["coverage_tdisp", "student_acc"]].mean()
    rho_pooled = spearmanr(pooled.coverage_tdisp, pooled.student_acc).statistic
    rho_runs = spearmanr(d.coverage_tdisp, d.student_acc).statistic
    res = {"noise_sweep": {"spearman_pooled": rho_pooled, "spearman_runs": rho_runs,
                           "n_conditions": len(pooled),
                           "rows": pooled.reset_index().to_dict("records")}}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    ax.scatter(pooled.coverage_tdisp, pooled.student_acc, color=C[0], s=30, zorder=3)
    for name, r in pooled.iterrows():
        ax.annotate(name, (r.coverage_tdisp, r.student_acc), textcoords="offset points", xytext=(4, 3),
                    fontsize=7, color=INK2)
    ax.set_xlabel("coverage cos(d₀, û_T), before training")
    ax.set_ylabel("student MNIST accuracy, after training")
    ax.set_title(f"Shared init, 12 noise types: Spearman ρ = {rho_pooled:.2f}", color=INK)
    chance(ax)
    if not e2d.empty:
        hidden_reinit = e2d["s.condition"].isin(["reinit_l1", "reinit_l2", "reinit_hidden"])
        e2d["group"] = np.where(hidden_reinit, "hidden layers reinitialised", "hidden layers shared")
        e2d["transfer"] = e2d.student_acc > 0.2
        ax = axes[1]
        groups = {}
        for i, (g, gd) in enumerate(e2d.groupby("group")):
            ax.scatter(gd.coverage_tdisp, gd.student_acc, color=C[i], s=14, alpha=0.8, label=g,
                       edgecolor="white", linewidth=0.5)
            groups[g] = {
                "n": len(gd),
                "spearman_tdisp": spearmanr(gd.coverage_tdisp, gd.student_acc).statistic,
                "spearman_s2t": spearmanr(gd.coverage_s2t, gd.student_acc).statistic,
                "auroc_tdisp": auroc(gd.coverage_tdisp, gd.transfer),
                "auroc_s2t": auroc(gd.coverage_s2t, gd.transfer),
                "frac_transfer": float(gd.transfer.mean()),
                "coverage_tdisp_median": float(gd.coverage_tdisp.median()),
            }
        res["e2_groups"] = groups
        ax.set_xlabel("coverage cos(d₀, û_T), before training")
        ax.set_title("E2 runs (all m): coverage vs. transfer", color=INK)
        ax.legend(loc="upper left", fontsize=8)
        chance(ax)
    fig.suptitle("X1: does Madl's coverage screen predict subliminal transfer?", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "x1_coverage.png")
    plt.close(fig)
    out["X1"] = res
    out["md"] += ["## X1 Coverage screen (Madl 2026)\n",
                  f"Noise sweep, shared init, m = 10: Spearman ρ (coverage vs accuracy) = {rho_pooled:.2f} over "
                  f"{len(pooled)} seed-pooled conditions, {rho_runs:.2f} over {len(d)} runs.\n",
                  "| Noise | Coverage | Student accuracy |", "| --- | --- | --- |"]
    out["md"] += [f"| {n} | {r.coverage_tdisp:.3f} | {100 * r.student_acc:.1f}% |"
                  for n, r in pooled.sort_values("coverage_tdisp").iterrows()]
    if "e2_groups" in res:
        out["md"] += ["", "E2 runs, transfer = accuracy above 20%:\n",
                      "| Group | Runs | Share with transfer | Median coverage | Spearman ρ (teacher disp. / start-to-end) "
                      "| AUROC (teacher disp. / start-to-end) |", "| --- | --- | --- | --- | --- | --- |"]
        for g, r in res["e2_groups"].items():
            out["md"].append(f"| {g} | {r['n']} | {100 * r['frac_transfer']:.0f}% | {r['coverage_tdisp_median']:.3f} | "
                             f"{r['spearman_tdisp']:.2f} / {r['spearman_s2t']:.2f} | "
                             f"{r['auroc_tdisp']:.2f} / {r['auroc_s2t']:.2f} |")
    out["md"].append("")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    out: dict = {"md": ["# Results\n", "Mean over seeds with a 95% bootstrap CI over seeds. Chance = 10%.\n"]}
    for fn in (e1, e2, e3, e4, e5, e6, e7, x1):
        fn(df, out)
    timing = df.groupby("exp")["seconds_student"].mean()
    out["md"] += ["## Run time\n", "| Experiment | Mean seconds per student run (1 core) |", "| --- | --- |"]
    out["md"] += [f"| {k} | {v:.0f} |" for k, v in timing.items()] + [""]
    (RES / "summary.md").write_text("\n".join(out.pop("md")))
    (RES / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    print((RES / "summary.md").read_text())


if __name__ == "__main__":
    main()
