# Vision training harness — built and smoke-tested

Week 5–6 machinery, verified on CPU before spending anything on GPU. The purpose
of this run was **not** to get a skill number: with 338 cases and 31 RI positives
there is nothing to learn. The purpose was to prove the pieces fit — data
loading, grouped folds, the comparison harness, and the metrics — so that the
funded run is a matter of scale rather than debugging.

## Model

`PatchCNN`: roughly a quarter-width ResNet-18 with GroupNorm, a 64-dimensional
embedding, and a single logit head. **391,009 trainable parameters.**

Sized for the data, not the leaderboard. There are ~1,100 independent RI
episodes in the entire satellite archive; a model with tens of millions of
parameters would memorise storms. The four frames enter as input channels rather
than through a 3D convolution — with only four of them that is the cheaper way
to let the first layer see differences between them.

## Physical conventions built into the dataset

**Hemisphere normalisation.** Southern-hemisphere cyclones rotate clockwise and
northern ones anticlockwise. SH patches are mirrored north–south so the
circulation sense matches, letting one model learn from both hemispheres. This
is a fixed transform, not an augmentation.

**Rotation-only augmentation.** Because the hemisphere mirror has already fixed
the rotation sense, a random flip would undo it. Multiples of 90° preserve it.
Rotation does discard the storm's shear-relative orientation, which is
physically meaningful — whether that costs skill is one of the funded ablations.

**Plain BCE, no positive-class reweighting.** Calibration is a first-class result
here, and reweighting would push the probabilities away from the base rate. The
smoke test confirms this works: mean predicted probability per fold came out at
0.078–0.091 against a 0.092 base rate.

## The comparison design

Not CNN versus logistic regression. The baseline report showed most of the spread
between those is model capacity, not information. So the question is posed as:
holding the large-scale environment fixed, does the imagery add more than the two
hand-engineered IR summary statistics do?

```
GBM(environment)                      <- floor
GBM(environment + SHIPS IR features)  <- the bar
GBM(environment + CNN output)         <- the claim
```

All three on identical rows and identical folds.

## Smoke-test result (North Atlantic 2020, 338 cases, 31 RI)

| Model | ROC AUC | PR AUC | Brier | BSS |
|---|---|---|---|---|
| Climatology | 0.5000 | 0.0917 | 0.0833 | 0.0000 |
| CNN alone | 0.7984 | 0.2357 | 0.0813 | 0.0242 |
| GBM environment | 0.7982 | 0.4187 | 0.0748 | 0.1016 |
| GBM environment + SHIPS IR | 0.7758 | 0.4519 | 0.0698 | 0.1625 |
| GBM environment + CNN | 0.7800 | 0.4853 | 0.0710 | 0.1475 |

**These numbers mean nothing and should not be quoted.** 31 positives is far
below what any of these estimates need. The run took 100 seconds on CPU.

## What the smoke test actually caught

**Fold imbalance.** The first version used `GroupKFold` on storm ID and produced
folds whose test sets contained 0, 11, 13, 1 and 6 RI cases. A fold with zero
positives has no defined ROC AUC and silently corrupts the pooled out-of-fold
estimate. Switching to `StratifiedGroupKFold` balances the positive rate while
keeping storms intact. Both the CNN script and the baseline script now use it, so
the two are compared under identical splits.

**How unstable small samples are.** Changing only the splitter moved "CNN alone"
ROC AUC from 0.585 to 0.798. That is not a real improvement; it is the size of
the noise at 31 positives, and it is a concrete demonstration of why the full
archive is needed rather than an assertion that it is.

The baseline was re-run under the new splitter. Its numbers barely moved — 23
seasons across 5 folds were already well balanced — which is the reassuring half
of the same check:

| Model | ROC AUC | PR AUC | BSS |
|---|---|---|---|
| GBM env + IR, `GroupKFold` | 0.8921 | 0.4860 | 0.2637 |
| GBM env + IR, `StratifiedGroupKFold` | 0.8911 | 0.4814 | 0.2579 |

## Known limitation, to be fixed before publication

The stacked row uses the CNN's out-of-fold probability as a GBM feature, so it
carries the usual stacking optimism: a training row's CNN value came from a model
fitted on folds that overlap the GBM's training set. Nested cross-validation
removes this, costs roughly 5× the compute, and is budgeted for. The smoke test
uses the cheap version deliberately.

## Measured throughput and what it implies

391k parameters, 128×128×4 input, 100.6 s for 5 folds × 12 epochs × ~280 training
cases on CPU. Extrapolating to the full ~32,000-case dataset on an A100, a single
cross-validated configuration should cost on the order of a few GPU-hours rather
than the 25 budgeted per configuration. The budget's margin is therefore real
rather than optimistic; the hours it buys will go into wider hyperparameter
search, larger patches, more frames, and the self-supervised pretraining line —
where the genuine uncertainty is.

## Reproducing

```bash
python scripts/09_train_cnn.py --tag NA_2020 --group SID --epochs 12
```

Runs on CPU in under two minutes. `--group SEASON` is the setting for the full
multi-season dataset.
