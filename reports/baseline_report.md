# SHIPS statistical baseline for 24 h rapid intensification

Week 3–4 deliverable. This is the number the vision model has to beat, established
before any neural network is trained.

## Sample

| | |
|---|---|
| Source | IBTrACS v04r01 labels joined to CIRA SHIPS developmental data (all 6 basin files) |
| Period | 2000–2022 seasons |
| Restrictions | tropical (NATURE=TS), ≥25 kt at t0, over water for the full 24 h window, all core environmental predictors present |
| Cases | 31,991 |
| Storms | 1,912 |
| RI positives | 2,954 (**9.23%**) |
| IR predictors present | 85.2% of cases |

Join validation: SHIPS carries its own 24 h intensity change on the DELV line.
It agrees exactly with the IBTrACS-derived label on **99.88%** of the 53,285
comparable cases (99.98% within 5 kt). The join and the label logic are correct.

Cross-validation is 5-fold **stratified and grouped by season**, so no storm
and no season appears in both training and evaluation, while the RI rate stays
comparable across folds.

## Results (out-of-fold)

| Model | Features | ROC AUC | PR AUC | Brier | BSS |
|---|---|---|---|---|---|
| Climatology | – | 0.5000 | 0.0923 | 0.0838 | 0.0000 |
| LR, environment only | 8 | 0.8356 | 0.3035 | 0.0726 | 0.1336 |
| **LR, environment + IR** (SHIPS-RII analogue) | 10 | 0.8506 | 0.3393 | 0.0701 | 0.1637 |
| GBM, environment only | 71 | 0.8863 | 0.4622 | 0.0632 | 0.2455 |
| **GBM, environment + IR** | 93 | **0.8911** | **0.4814** | 0.0622 | **0.2579** |

Accuracy is not reported anywhere: at a 9.2% base rate, always forecasting "no RI"
scores 90.8% and is useless.

## The number that actually matters

What the standard hand-engineered GOES IR predictors add, once the large-scale
environment is already known:

| | ΔROC AUC | ΔPR AUC | ΔBSS |
|---|---|---|---|
| LR: env → env + IR | +0.0150 | +0.0358 | +0.0301 |
| GBM: env → env + IR | +0.0047 | **+0.0192** | +0.0124 |

**This is a small margin, and it is the honest framing of the project's risk.**
The two IR predictors in the classical RII set (% area colder than −30 °C, and
the standard deviation of brightness temperature within 200 km) buy roughly
0.019 of PR AUC on top of a strong environmental model. The project's hypothesis
is that this is small because those two summary statistics discard almost all of
the spatial structure — not because cloud-top structure is uninformative. But
that hypothesis is now a testable claim with a number attached, rather than an
assumption.

Note also that GBM-environment-only (0.4622) beats LR-environment-plus-IR
(0.3393) by far more than the IR predictors contribute. Most of the spread in
this table is model capacity, not information. That has a direct consequence for
how the vision model must be evaluated (below).

## Calibration

Reliability of the best baseline (GBM env + IR), equal-count decile bins:

| bin | n | mean forecast | observed frequency |
|---|---|---|---|
| 1 | 3200 | 0.0003 | 0.0000 |
| 2 | 3199 | 0.0008 | 0.0009 |
| 3 | 3199 | 0.0022 | 0.0041 |
| 4 | 3199 | 0.0049 | 0.0072 |
| 5 | 3199 | 0.0104 | 0.0188 |
| 6 | 3199 | 0.0207 | 0.0300 |
| 7 | 3199 | 0.0408 | 0.0688 |
| 8 | 3199 | 0.0845 | 0.1172 |
| 9 | 3199 | 0.1928 | 0.2091 |
| 10 | 3199 | 0.4982 | 0.4673 |

Well calibrated in the top decile, mildly under-forecasting in the middle
(bins 5 and 7 observe roughly twice the forecast rate). Worth a calibration
step later, but not a defect that changes the ranking.

## Contingency (GBM env + IR)

| threshold | forecasts | hits | misses | false alarms | POD | FAR | CSI | PSS |
|---|---|---|---|---|---|---|---|---|
| 0.10 | 7172 | 2275 | 679 | 4897 | 0.770 | 0.683 | 0.290 | 0.601 |
| 0.15 | 5587 | 2034 | 920 | 3553 | 0.689 | 0.636 | 0.313 | 0.566 |
| 0.20 | 4527 | 1832 | 1122 | 2695 | 0.620 | 0.595 | 0.324 | 0.527 |
| 0.30 | 3072 | 1456 | 1498 | 1616 | 0.493 | 0.526 | 0.319 | 0.437 |
| 0.40 | 2083 | 1133 | 1821 | 950 | 0.384 | 0.456 | 0.290 | 0.351 |
| 0.50 | 1405 | 859 | 2095 | 546 | 0.291 | 0.389 | 0.245 | 0.272 |

Even at the best CSI (0.32 near a 0.20 threshold), well over half of RI warnings
are false alarms. This is the operational reality of the problem.

## By basin (GBM env + IR)

| basin | n | positives | base rate | ROC AUC | PR AUC | BSS | recon |
|---|---|---|---|---|---|---|---|
| EP | 6920 | 524 | 0.0757 | 0.9221 | 0.4823 | 0.2831 | yes |
| NA | 5472 | 356 | 0.0651 | 0.8739 | 0.3836 | 0.1879 | yes |
| NI | 1354 | 114 | 0.0842 | 0.9075 | 0.5515 | 0.3126 | no |
| SI | 6545 | 585 | 0.0894 | 0.8929 | 0.4619 | 0.2484 | no |
| SP | 2644 | 291 | 0.1101 | 0.8535 | 0.4120 | 0.1915 | no |
| WP | 9056 | 1084 | 0.1197 | 0.8788 | 0.5347 | 0.2761 | no |

The North Atlantic is the hardest basin in the set, which is worth remembering
when reading Atlantic-only numbers from the literature.

### Recon-verified vs Dvorak-derived

| | n | positives | base rate | ROC AUC | PR AUC | BSS |
|---|---|---|---|---|---|---|
| Dvorak-derived | 19599 | 2074 | 0.1058 | 0.8820 | 0.4975 | 0.2600 |
| Recon-verified (NA/EP) | 12392 | 880 | 0.0710 | 0.9032 | 0.4434 | 0.2446 |

No large skill gap between the two groups for this statistical baseline. PR AUC
is not directly comparable across them because the base rates differ; ROC AUC and
BSS are, and both are close. This does **not** settle the label-circularity
question — a statistical model on environmental predictors has much less
opportunity to reproduce Dvorak than a CNN reading the same imagery a Dvorak
analyst looked at. The split has to be re-examined for the vision model, where it
matters far more.

## Caveats

1. **Perfect prognosis.** The SHIPS developmental dataset contains analysis
   values, not forecast values. Every number here is therefore optimistic
   relative to an operational forecast, and the vision model must be held to the
   same standard for the comparison to be fair.
2. **"SHIPS-RII analogue", not SHIPS-RII.** The LR env+IR model uses the Kaplan
   & DeMaria predictor set, but not the operational model's exact fitting
   procedure, thresholds, or consensus combination. It is a faithful
   reimplementation of the approach, not of the operational product.
3. **IR availability is uneven** — 93.6% of North Atlantic cases but only 51.3%
   of West Pacific ones. The GBM handles the missingness natively; the LR
   median-imputes with missing indicators.
4. **2023+ is excluded.** The JTWC basin files stop at 2021 and the final IBTrACS
   season is partial.

## Consequence for the vision model

Because most of the spread in the results table is model capacity rather than
information, the vision model must not be compared against the LR. The clean
comparison is:

> **GBM(environment + CNN embedding)** vs **GBM(environment + IR summary features)**

on exactly the same cases, with the same season-grouped folds. Everything else
holds the environment constant and isolates what the imagery adds. Out-of-fold
predictions for all five models here are saved to `data/processed/baseline_oof.csv`
keyed by (SID, ISO_TIME) so that this comparison is possible without refitting.

## Reproducing

```bash
python scripts/01_download_ibtracs.py
python scripts/02_build_labels.py
python scripts/03_download_ships.py
python scripts/04_build_ships_table.py
python scripts/05_ships_baseline.py
```

Runs on CPU. No GPU is used or needed for anything in this report.
