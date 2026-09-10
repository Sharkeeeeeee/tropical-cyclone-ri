# Tropical cyclone rapid intensification from geostationary infrared imagery

Predicting whether a tropical cyclone will gain ≥30 kt in the next 24 hours,
from the structure of its cloud tops rather than from forecast-model output.

Everything here runs from public data and, so far, entirely on CPU.

## Status

| Stage | State |
|---|---|
| RI labels from IBTrACS | done |
| SHIPS statistical baseline | done — see [reports/baseline_report.md](reports/baseline_report.md) |
| Storm-centred IR patch extraction | done (GridSat-B1) — see [reports/patch_pipeline_report.md](reports/patch_pipeline_report.md) |
| Storm-centred patches, NA+EP 2000-2022 | 13,097 complete sequences, validated at r = 0.9914 |
| Structural features vs classical IR | done — see [reports/structure_report.md](reports/structure_report.md) |
| Vision model | training harness built, benchmarked on an RTX 4060 |
| Experiment A — label provenance | statistical half done — see [reports/experiment_a_report.md](reports/experiment_a_report.md); vision half needs the global archive |

## Headline numbers

* **1,094 independent RI episodes** in the 2000–2022 satellite era, over water,
  globally. Not 3,718 — consecutive 6-hourly hits inside one intensification
  burst are near-duplicates. This is a small-data problem.
* Base rate **9.2%**. Accuracy is meaningless here and is never reported.
* Best statistical baseline (gradient boosting, environment + SHIPS IR
  predictors): **ROC AUC 0.891, PR AUC 0.481, BSS 0.258**.
* **Hand-engineered cloud structure adds 3.8x what the two classical IR
  predictors add** (+0.039 vs +0.010 PR AUC over environment alone), with
  deviation-angle variance the strongest single structural feature.
* The standard hand-engineered IR predictors add only **+0.019 PR AUC** once the
  large-scale environment is known. That margin is what the vision model exists
  to beat.
* Patch pipeline validated end-to-end against SHIPS' own IR predictors:
  **r = 0.984** on percent area colder than −30 °C.
* Best-track intensities are **quantized onto the Dvorak table** in every
  basin — contrast +0.499 with reconnaissance, +0.694 without, intervals
  non-overlapping — but that contamination does **not** reach models built
  on environmental predictors (transfer asymmetry −0.001).

## Data

| Source | Use | Access |
|---|---|---|
| IBTrACS v04r01 | best-track labels | NCEI, single 144 MB CSV |
| CIRA SHIPS developmental (lsdiag) | baseline predictors, 6 basins | RAMMB/CIRA, 1.3 GB of text |
| GridSat-B1 v02r01 | 3-hourly ~8 km global IR | NCEI **OPeNDAP**, subset server-side |

GridSat is read through OPeNDAP rather than by downloading files. A global file
is ~50 MB; the box around one storm is ~0.08 MB — a ~650× reduction in transfer,
which is what makes the full archive tractable without a large storage budget.

## Pipeline

```bash
python scripts/01_download_ibtracs.py
python scripts/02_build_labels.py          # RI labels + base-rate report
python scripts/03_download_ships.py        # all six basin files
python scripts/04_build_ships_table.py     # join, verified against SHIPS DELV
python scripts/05_ships_baseline.py        # LR + GBM, season-grouped CV
python scripts/06_demo_patch.py            # visual check, Katrina 2005
python -u scripts/07_extract_patches.py --basin NA --seasons 2020 --workers 10
python scripts/08_validate_patches.py NA_2020
python scripts/09_train_cnn.py --tag NA_2020 --group SID     # CPU harness
python -u scripts/10_structural_features.py --tag NA_2020    # 141 features
python -u scripts/11_structure_baseline.py --tag NA_2020     # structure vs SHIPS IR
python -u scripts/12_train_gpu.py --tag NA_2020 --group SID  # single-GPU trainer
```

`12_train_gpu.py` is written to share the machine: below-normal priority, a
quarter of the cores, 60% of VRAM and a 70% duty cycle by default, with every
fold checkpointed so Ctrl+C is safe. Pass `--duty 1.0 --workers 8 --vram 0.9`
when nothing else needs the laptop.

## Design decisions worth knowing

**Intensity comes from `USA_WIND`, not `WMO_WIND`.** WMO_WIND mixes 1-minute and
10-minute averaging across agencies, so "30 kt in 24 h" would mean different
things in different basins.

**Splits are stratified and grouped by season, never random over samples.** Cases within a
storm are heavily autocorrelated; a random split leaks badly.

**Recon-verified and Dvorak-derived cases are reported separately.** Outside the
North Atlantic and East Pacific there is no routine aircraft reconnaissance, so
best-track intensity is itself derived from the same infrared imagery via the
Dvorak technique. A model trained on those labels may be learning to reproduce
Dvorak rather than physics.

**Patches are resampled onto a fixed kilometre grid**, not cut as a fixed box in
degrees. A degree box shrinks physically as latitude rises, and a model could
learn latitude from the framing alone.

**The SHIPS baseline is perfect-prognosis** (analysis values, not forecast
values), so it is optimistic. The vision model must be held to the same standard.

## Layout

```
src/tcri/     config, ibtracs, ships, gridsat, evaluate
scripts/      numbered, run in order
data/raw/     downloads (gitignored)
data/processed/  labels, joined tables, patch memmaps (gitignored)
reports/      baseline report, validation figures
```

## Requirements

`pip install -r requirements.txt` — numpy, pandas, scikit-learn, matplotlib,
xarray, netCDF4, scipy, requests. torch is listed for the vision stage and is
not needed for anything above.
