# Does hand-engineered cloud structure beat the two classical IR scalars?

The question that makes any later CNN result interpretable. If a vision model
beats the SHIPS baseline, one needs to know whether the gain comes from spatial
structure carrying information at all, or from deep learning specifically. This
answers the first half with numpy features and a gradient-boosted tree.

**Result: yes, by a factor of about four.** Structural features add +0.039 PR AUC
over the environment alone, against +0.010 for the two classical IR predictors.

---

## Sample

North Atlantic and East Pacific, 2000–2022 — the reconnaissance-verified basins,
chosen so that label provenance is the cleanest available.

| | |
|---|---|
| Patch sequences extracted | 14,088 cases |
| With all 4 frames | 13,097 (93.0%) |
| Joined to SHIPS environment | 12,339 |
| RI positives | 880 (**7.13%**) |
| Seasons | 23 |

Cross-validation is 5-fold, stratified and grouped by season.

### Patch validation at this scale

Recomputing SHIPS' own IR predictor from our patches, on 12,517 comparable cases:

| | NA 2020 only (n = 383) | full sample (n = 12,517) |
|---|---|---|
| correlation, % area < −30 °C | 0.9836 | **0.9914** |
| mean absolute error | 3.64 pct points | **2.49 pct points** |

The `0–200 km` standard-deviation discrepancy noted in the patch-pipeline report
reproduces exactly at 34× the sample: SHIPS' documented 200 km disc behaves like
a ~100–125 km effective radius (bias +0.07 K at 100 km, best correlation 0.919 at
125 km). Not a small-sample artefact.

## Results

| Model | features | ROC AUC | PR AUC | BSS |
|---|---|---|---|---|
| Climatology | – | 0.5000 | 0.0713 | 0.0000 |
| environment | 8 | 0.8692 | 0.3759 | 0.1809 |
| environment + SHIPS IR | 10 | 0.8800 | 0.3863 | 0.1930 |
| **environment + structure** | 149 | **0.8935** | **0.4153** | **0.2157** |
| environment + SHIPS IR + structure | 151 | 0.8921 | 0.4089 | 0.2116 |
| structure only | 141 | 0.8364 | 0.2820 | 0.1149 |

### What each block adds over the environment

| | ΔROC AUC | ΔPR AUC | ΔBSS |
|---|---|---|---|
| SHIPS IR (2 features) | +0.0108 | +0.0104 | +0.0121 |
| **structure (141 features)** | **+0.0243** | **+0.0394** | **+0.0348** |

**Structure is worth 3.79× the classical pair on PR AUC.** The imagery carries
RI-relevant information that percent-area-colder-than-−30 °C and the standard
deviation of brightness temperature discard.

Adding the SHIPS pair *on top of* structure slightly hurts (0.4089 against
0.4153). Two redundant features among 149 is noise, not a finding.

## Which structure matters

Permutation importance on the held-out fold, top 15 of 151:

| rank | feature | importance | kind |
|---|---|---|---|
| 1 | per12 | 0.1446 | environment — previous 12 h intensity change |
| 2 | shrd | 0.1016 | environment — deep-layer shear |
| 3 | pot | 0.0911 | environment — potential intensity minus current |
| 4 | cohc | 0.0489 | environment — ocean heat content |
| 5 | **dav_350** | **0.0334** | **structure — deviation-angle variance, 350 km** |
| 6 | vmax | 0.0152 | environment |
| 7 | azspread_100_300 | 0.0112 | structure — azimuthal spread |
| 8 | cold100_300_lt50 | 0.0109 | structure — outer cold cloud |
| 9 | ring300_500_min | 0.0094 | structure — outer minimum BT |
| 10 | **core_min_d3h** | **0.0088** | **structure — 3 h change in core minimum** |
| 11 | dav_200 | 0.0087 | structure |
| 12 | ring0_50_mean | 0.0078 | structure |
| 13 | ring0_50_min | 0.0073 | structure |
| 14 | **cold0_100_lt30_d3h** | **0.0067** | **structure — 3 h change in inner cold cloud** |
| 15 | ring50_100_mean | 0.0063 | structure |

Three things worth noting:

1. **Deviation-angle variance is the most valuable structural feature.** DAV
   measures how nearly axisymmetric the brightness-temperature gradient field is
   — organisation, not temperature. It is a 2008 technique (Piñeros et al.) that
   the classical RI predictor set does not include.
2. **Asymmetry and outer-region convection follow.** Azimuthal spread and cold
   cloud at 100–300 km outrank every inner-core temperature statistic.
3. **Temporal tendencies appear on their own merit.** Two 3-hour change features
   make the top 15 without being engineered to. That is the first direct support
   in this project for the premise that evolution matters, and it comes from
   3-hourly sampling — the cadence question in experiment C asks whether finer
   sampling would show more.

The picture is coherent: what predicts RI in the imagery is **how organised and
symmetric the cloud field is, and how that is changing** — not simply how cold
it is. The two classical predictors measure only how cold and how variable.

## Calibration and operating points

Reliability of the best model, equal-count deciles:

| bin | n | mean forecast | observed |
|---|---|---|---|
| 1 | 1234 | 0.0001 | 0.0008 |
| 5 | 1234 | 0.0031 | 0.0154 |
| 7 | 1234 | 0.0139 | 0.0308 |
| 9 | 1234 | 0.1010 | 0.1459 |
| 10 | 1233 | 0.3979 | 0.3966 |

Well calibrated at the top, under-forecasting through the middle by roughly a
factor of two to five. A calibration step is worth adding before any operational
claim; it does not change the ranking.

| threshold | POD | FAR | CSI |
|---|---|---|---|
| 0.10 | 0.672 | 0.669 | 0.285 |
| 0.15 | 0.591 | 0.622 | **0.300** |
| 0.20 | 0.514 | 0.584 | 0.298 |
| 0.30 | 0.399 | 0.524 | 0.277 |

Best CSI is 0.30. Roughly three in five RI warnings remain false alarms. That is
the operational reality and no result here changes it.

## Relation to the published literature — resolved

McNeely, Lee, Wood & Hammerling (*J. Appl. Meteor. Climatol.* 2020) built the ORB
feature suite from GOES IR and reported accuracy "comparable to classifiers using
environmental predictors alone", adding that "more complex nonlinear machine
learning methods did not perform better than the linear logistic lasso model".

That is the opposite of the result above, so the same comparison was re-run under
both model classes and scored under both metrics.

| model class | metric | environment | + structure | gain |
|---|---|---|---|---|
| logistic lasso | **accuracy** | 0.9287 | 0.9317 | **+0.0030** |
| logistic lasso | PR AUC | 0.3190 | 0.3928 | **+0.0738** |
| logistic lasso | BSS | 0.1554 | 0.2133 | +0.0578 |
| gradient boosting | **accuracy** | 0.9310 | 0.9318 | **+0.0009** |
| gradient boosting | PR AUC | 0.3759 | 0.4153 | **+0.0394** |
| gradient boosting | BSS | 0.1809 | 0.2157 | +0.0348 |

**The disagreement is the metric, not the model class, and there is no real
disagreement.**

Measured by accuracy, structural features add 0.3% with a lasso and 0.09% with
gradient boosting — that is, nothing. Measured by PR AUC, the same features on
the same folds add 0.074 and 0.039. Always forecasting "no RI" already scores
92.9% accuracy at this base rate, so accuracy has almost no room to register a
change in how a rare event is ranked. **McNeely et al.'s conclusion is correct
for the metric they reported.** The effect their features may well have had was
invisible to the instrument they measured it with.

Model class is ruled out from the other direction: the *linear* model shows the
**larger** structural gain (+0.074 against +0.039). Gradient boosting extracts
more from the environment alone (PR AUC 0.376 against 0.319), leaving less for
structure to add. Nonlinearity is not what produces the result here.

Feature set contributes something but does not explain it. Removing
deviation-angle variance — which is not part of ORB — costs 20% of the
structural gain under the lasso and 41% under gradient boosting. Real, and
secondary to the metric.

The methodological point generalises beyond this paper: **a rare-event result
reported as accuracy can be null by construction.** At a 7% base rate the
negative class sets the score. Anyone reading the IR-and-RI literature should
check which metric a negative result was measured with before concluding the
signal is absent.

## Does a learned representation beat the hand-built one? No.

The structural result above says the imagery carries information the classical
pair discards. It does not say that a neural network is the way to extract it.
That is a separate question, and it decides how much GPU time this problem
deserves, so it was measured rather than assumed.

A 391k-parameter CNN — quarter-width ResNet, GroupNorm, four frames as channels
— was trained on the same 12,339 cases under the same season-grouped folds,
40 epochs per fold, 29 minutes total on an RTX 4060.

### Clean comparison, nothing stacked

Both rows are honest out-of-fold predictions from models that never saw the case
they score.

| | ROC AUC | PR AUC | BSS |
|---|---|---|---|
| **structure only** (141 features) | **0.8364** | **0.2820** | **0.1149** |
| CNN only (391k parameters) | 0.8153 | 0.2517 | 0.1076 |

**The CNN loses by 0.030 PR AUC.** Reading the same pixels, 141 lines of numpy
beat a convolutional network.

### Stacked on the environment — upper bounds

These numbers carry the usual stacking optimism, since a training row's CNN
value came from a network fitted on folds overlapping the booster's training
set. They are therefore generous to the CNN, and it still loses.

| | PR AUC | gain over environment |
|---|---|---|
| environment | 0.3759 | — |
| environment + SHIPS IR (2 features) | 0.3863 | +0.0104 |
| **environment + structure** | **0.4153** | **+0.0394** |
| environment + CNN probability | 0.3860 | +0.0101 |
| environment + CNN embedding (64-d) | 0.3739 | **−0.0020** |
| environment + structure + CNN | 0.4044 | +0.0285 |

Three things to read off that table. The CNN's contribution (+0.010) is
indistinguishable from what the two classical scalars give (+0.010). Its 64-
dimensional embedding is **worse than not using imagery at all**. And adding the
CNN on top of the structural features *degrades* them, 0.4153 to 0.4044.

### What this does and does not mean

**It does not mean deep learning cannot work here.** This is one architecture at
one size, 40 epochs, one hyperparameter setting, no pretraining, no augmentation
search. It is a first attempt, not a ceiling.

**It does mean the bottleneck is not model capacity.** With ~880 positives — and
fewer independent intensification episodes than that — a network has to learn
from scratch what the structural features encode as priors: that organisation
matters, that asymmetry matters, that radius matters. Hand-engineered features
carry a century of tropical meteorology for free. In this data regime that is
worth more than representation learning, which is the textbook expectation and is
now measured on this problem rather than assumed.

**It has a direct budget consequence.** The remedy for this regime is not a
bigger GPU or a bigger model. It is more effective labelled signal — through
self-supervised pretraining on the much larger unlabeled pool, and through better
sampling in space and time. That is what the funded experiments B and C test, and
it is why the budget's weight sits on data engineering rather than compute.

## Caveats

1. **Reconnaissance-verified basins only.** NA and EP. Generalisation to the
   satellite-only basins is untested and is the subject of experiment A.
2. **880 RI positives**, and fewer independent episodes than that. Differences
   below roughly 0.01 PR AUC should not be interpreted.
3. **Perfect prognosis.** The SHIPS environmental predictors are analysis values.
   Both the baseline and the structural model inherit that optimism equally, so
   the comparison is fair, but neither number is an operational forecast score.
4. **The structural features are not tuned.** Radii, thresholds and wavenumbers
   were chosen from the literature before any of these numbers were computed.

## Reproducing

```bash
python scripts/13_merge_shards.py --out recon_2000_2022 --tags recon_2000_2002 ...
python scripts/08_validate_patches.py recon_2000_2022
python -u scripts/10_structural_features.py --tag recon_2000_2022
python -u scripts/11_structure_baseline.py --tag recon_2000_2022 --group SEASON --importance
```

CPU only. Feature extraction runs at 340 cases/s; the whole chain is a few
minutes after the patches exist.
