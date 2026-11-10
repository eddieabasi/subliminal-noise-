# Follow-up: coverage in Madl's setup, and readout-aware screens

Teachers rebuilt for after-the-fact screening: 760 runs, 0 with a teacher accuracy different from the original run.

## Step 1 - X2 (Adam students): coverage law in a Madl-like setup

Seed-pooled Spearman ρ between coverage and student accuracy over noise conditions (Madl reports 0.95 over his 54).

| Setting | Madl's 5 families (33 conditions) | + Perlin (36) | Leave-one-family-out, min | Accuracy range |
| --- | --- | --- | --- | --- |
| madl_kl | 0.40 | 0.23 | -0.02 | 10-55% |
| madl_mse | 0.49 | 0.29 | 0.16 | 8-65% |
| ours | 0.47 | 0.28 | 0.12 | 10-85% |

Per-family Spearman ρ (seed-pooled):

| Setting | dc | mixture | perlin | rank | shape | spectral |
| --- | --- | --- | --- | --- | --- | --- |
| madl_kl | 0.86 | 1.00 | -1.00 | 0.83 | 0.71 | -0.88 |
| madl_mse | 0.96 | 1.00 | -1.00 | -0.54 | 0.43 | -0.42 |
| ours | 0.96 | 1.00 | -0.50 | -0.31 | -0.37 | -0.43 |

## Step 1 - X3 (plain SGD students, lr 3): coverage law in a Madl-like setup

Seed-pooled Spearman ρ between coverage and student accuracy over noise conditions (Madl reports 0.95 over his 54).

| Setting | Madl's 5 families (33 conditions) | + Perlin (36) | Leave-one-family-out, min | Accuracy range |
| --- | --- | --- | --- | --- |
| madl_kl | 0.50 | 0.36 | 0.21 | 9-21% |
| madl_mse | -0.24 | -0.12 | -0.69 | 8-10% |
| ours | 0.78 | 0.59 | 0.62 | 9-42% |

Per-family Spearman ρ (seed-pooled):

| Setting | dc | mixture | perlin | rank | shape | spectral |
| --- | --- | --- | --- | --- | --- | --- |
| madl_kl | 0.68 | 0.90 | -0.50 | 0.60 | 0.26 | -0.53 |
| madl_mse | -0.61 | 0.71 | 1.00 | 0.49 | -0.09 | 0.67 |
| ours | 0.88 | 1.00 | -0.50 | -0.14 | -0.43 | 0.48 |

## Adam vs plain SGD (Spearman ρ, seed-pooled)

| Setting | Adam: 5 families | SGD: 5 families | Adam: + Perlin | SGD: + Perlin | Adam: accuracy range | SGD: accuracy range |
| --- | --- | --- | --- | --- | --- | --- |
| madl_kl | 0.40 | 0.50 | 0.23 | 0.36 | 10-55% | 9-21% |
| madl_mse | 0.49 | -0.24 | 0.29 | -0.12 | 8-65% | 8-10% |
| ours | 0.47 | 0.78 | 0.28 | 0.59 | 10-85% | 9-42% |

## Step 2 - which screen predicts transfer?

Spearman ρ with student accuracy / AUROC for transfer (accuracy above 20%), per run.

| Screen | Head + body mismatches (E2+E4, m = 10) (n = 340) | All of E2 (m = 1-250) (n = 480) | Noise types, shared init (X1) (n = 60) | Fixed-RMS noise sweep (X2) (n = 324) |
| --- | --- | --- | --- | --- |
| Body coverage (Madl) | 0.43 / 0.72 | 0.48 / 0.74 | 0.33 / 0.99 | 0.43 / 0.70 |
| One-step readout coverage | 0.08 / 0.55 | 0.22 / 0.59 | 0.07 / 0.67 | 0.38 / 0.65 |
| Class-head match alone | 0.06 / 0.47 | 0.31 / 0.65 | -0.30 / 0.48 | -0.13 / 0.49 |
| Body coverage x class-head match | 0.73 / 0.81 | 0.79 / 0.89 | 0.33 / 0.99 | 0.43 / 0.70 |
| Early canary (agreement after 5% of training) | 0.68 / 0.82 | 0.78 / 0.88 | 0.62 / 0.94 | 0.58 / 0.79 |
