# subliminal-noise

A from-scratch, CPU-only replication of

> **Learning Through Noise: Why Subliminal Learning Works and When It Fails**
> Brockers, Ventzke, Neuhaus, Hidalgo-Ogalde, Priesemann. arXiv [2605.23645](https://arxiv.org/abs/2605.23645), May 2026.

plus one extension: the pre-training **coverage screen** from

> **Channel Location Constrains the Auditability of Subliminal Learning**
> Madl. arXiv [2606.22019](https://arxiv.org/abs/2606.22019), June 2026.

Subliminal learning is when a student model inherits a teacher's traits from
distillation data that never mention them. The paper studies it in a controlled
MNIST setting: the teacher learns digits through a *class head*, the student only
ever imitates the teacher's *auxiliary* logits on random noise, and yet the student
ends up classifying digits through its untouched class head. The paper's claim is
that this is governed by **compatible output heads**, not by a shared
initialisation of the hidden layers.

## Results at a glance

**The paper's central claim reproduces.** Matching output heads, not a shared
initialisation of the hidden layers, is what lets a student learn MNIST from noise.
7 of the paper's claims reproduce fully, 2 partly. The coverage extension did not
reproduce, and found a blind spot instead. 1,065 teacher-student runs, about 2.5
hours on 4 CPU cores.

![Student accuracy vs. number of aux logits m when parts of the student are reinitialised](results/figures/e2_init_ablation.png)

*Reinitialising the student's hidden layers only delays transfer: accuracy comes
back as m grows. A reinitialised aux or class head keeps the student near chance
at every m (claims 2-4).*

| # | Claim (paper) | Our result | Verdict |
| --- | --- | --- | --- |
| 1 | Students trained only on noise learn MNIST (E1) | Student 87.7% [85.7, 89.7], teacher 97.1%, chance 10% | Reproduced |
| 2 | Shared hidden layers are not needed (E2, Fig. 2b-d) | Both hidden layers reinitialised: 11% at m = 1, rising to 82.9% at m = 250 | Reproduced |
| 3 | A reinitialised class head kills transfer (Fig. 2f) | 10.0-11.1% at every m | Reproduced |
| 4 | A reinitialised aux head strongly suppresses it (Fig. 2e) | 15.1% vs 95.0% (all shared) at m = 250; falls as m grows | Reproduced |
| 5 | Best case matches the teacher (Tables 3-4: 96.82% vs 96.86%) | Student 96.7% vs teacher 96.8% (20 seeds) | Reproduced |
| 6 | Aux-head damage is partly repaired, class-head damage is not (Fig. 5) | Accuracy falls with δ for both; aux-head cosine to teacher rises during training (0.82 to 0.87 at δ = 0.025); class head unchanged | Reproduced |
| 7 | Other architectures transfer; width is non-monotone (Fig. 3a-b) | MLP with 1 or 3 hidden layers transfers (46%, 25% at m = 50); width peaks at 256; CNN only 13-16% at m <= 50, but 46.2% [40.0, 53.7] at m = 250 (off-paper check E5b) | Partly |
| 8 | m = 1 peaks near 83% then declines; N = 1,000 plateaus near 29% (Fig. 4b-c) | Peak 80.0% at N = 10^6, 78.3% at 3 x 10^6 (decline not resolved); 26.3% at m = 500, still rising | Partly |
| 9 | Large latent width breaks transfer via aux-head drift (Fig. 6, 13) | Free heads: 91.1% at d = 1,024, 14.9% at d = 16,384; aux head fixed 71.9%, both fixed 82.2%, class head fixed 17.0% | Reproduced |
| 10 | Extension: coverage predicts transfer, Spearman ρ ≈ 0.95 (Madl, Sec. 3) | ρ = 0.38 across 12 noise types; ρ = 0.82 across aux-head perturbations; blind to the class head | Not reproduced |

Brackets are 95% bootstrap CIs over seeds. Full tables: [`results/summary.md`](results/summary.md).

### Figures

![Head perturbation](results/figures/e4_perturbation.png)

*Noise added to either head before distillation lowers accuracy. Training pulls a
perturbed aux head back toward the teacher's; the class head never changes
(claim 6).*

![Architecture and capacity](results/figures/e5_architecture.png)

*Students with one fewer or one more hidden layer still transfer, and accuracy
peaks near the teacher's width of 256. A CNN student barely transfers at m <= 50
(claim 7).*

![m and N bottlenecks](results/figures/e6_m_and_n.png)

*With one aux logit, more noise lifts accuracy to 80%. With 1,000 noise samples
per epoch, more aux logits help only slowly (claim 8).*

![Latent dimension](results/figures/e7_latent_dim.png)

*Transfer collapses at large latent width as the heads drift; freezing the aux
head prevents most of the collapse (claim 9).*

### What the extension found

Coverage is the cosine between the student's first distillation update and the
teacher's training displacement, computed before the student trains. It is
computed on the network body, so it can only see problems on the input side of
the latent:

* It **tracks the aux head**. Perturbing the student's aux head lowers coverage
  (0.032 to -0.007) and accuracy together (Spearman 0.82).
* It is **structurally blind to the class head**. A reinitialised or perturbed
  class head leaves coverage exactly unchanged while accuracy drops to chance.
  That is the paper's second condition (class-head compatibility), and a
  body-only screen cannot see it.
* Across noise types it fails on smooth noise: Perlin and brown (1/f²) noise
  have the highest coverage but low transfer (23%, 51%).

We expected the screen to go blind when the hidden layers are reinitialised.
It did not (ρ ≈ 0.9 across conditions at m >= 50); the blind spot is the readout.

![Coverage vs. transfer](results/figures/x1_coverage.png)

*Left: across noise types, smooth noise (Perlin, brown) has high coverage but low
transfer. Right: runs with a mismatched class head (green) spread from chance to
90% at the same coverage (claim 10).*

## Deviations from the paper

* **Seeds.** 20 seeds for E1-E4, as in the paper; 5 for E5-E7 and X1.
* **Choices the paper leaves open.** m grid {1, 10, 50, 250} for E2; δ grid
  {0.0125, ..., 0.2} (default weight scale is about 0.06); student widths 32-1024;
  "latent dimension" is the width of the last hidden layer (the first stays 256);
  E6 keeps 1,000 noise images per step and varies steps per epoch.
* **Perlin noise** is our own implementation (resolution = grid cells per 28 px).
  Its std is 0.31, about half that of U(-1, 1); the paper does not give an amplitude.
* **Not run:** EMNIST class-count sweep (Fig. 3c; download host blocked here), the
  full (m, N) heatmap (Fig. 4a; we ran its two slices), and the Appendix A theory
  checks (eq. 28, eq. 58).
* **X1** uses our own versions of Madl's five noise families (DC level, spectral
  slope, distribution shape, low rank, Perlin), not his 54 conditions, and the MSE
  channel (his appendix uses soft cross-entropy).
* **E5b** is an off-paper diagnostic for the weak CNN result: at m = 250 the CNN
  student reaches 46.2% (an MLP student with independent hidden layers: 81.6%);
  rescaling Perlin noise to uniform's std lifts m = 50 from 14% to 22.6%.

## Setup

Everything follows the paper's Appendix C (Table 1) unless listed under *Deviations*.

| | |
| --- | --- |
| Models | MLP 784-256-256 with a class head (10) and an aux head (m, default 10); 271,892 parameters |
| Teacher | Adam 1e-3, 5 epochs, batch 1024, cross-entropy on the class head only |
| Student | Adam 1e-3, 5 epochs x 60 steps of 1,000 noise images (N = 60,000 per epoch), MSE on the aux head only |
| Noise | U(-1, 1) per pixel; Perlin noise (resolution 8) for CNN students |
| Data | MNIST, normalised with mean 0.1307, std 0.3081 |
| Uncertainty | mean over seeds, 95% bootstrap CI over seeds |
| Compute | 4 CPU cores, no GPU; about 7 s per teacher or student run on one core |

Parameter counts match the paper's Tables 1 and 2 exactly (checked in `tests/`).

## Repository layout

```
src/sublim/
  models.py     split-head MLP and CNN (Tables 1-2)
  init.py       layerwise shared / independent init, head perturbation, freezing
  noise.py      uniform, Perlin, and the X1 noise families
  train.py      teacher training, student aux distillation, evaluation
  coverage.py   Madl's coverage screen
  runner.py     one job = one teacher + all student conditions distilled from it
experiments/    one config per experiment (E1-E7, X1)
scripts/run.py      parallel, resumable sweep launcher -> results/runs.jsonl
scripts/analyze.py  tables (results/summary.md) and figures (results/figures/)
tests/          invariants: shared init identical, heads untouched where they must be, ...
```

## Reproduce

```bash
pip install torch torchvision scipy pandas matplotlib pytest
python -m pytest -q tests
python scripts/run.py e1_baseline e2_init_ablation e3_best_case e4_perturbation --workers 4
python scripts/run.py e5_architecture e6_m_and_n e7_latent_dim x1_noise_sweep --workers 4
python scripts/analyze.py
```

MNIST downloads to `data/` on first use (override with `SUBLIM_DATA`).
