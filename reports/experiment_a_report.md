# Experiment A — Is label provenance a confound in RI prediction?

Best-track intensity is not an observation. In every basin without routine
aircraft reconnaissance it is estimated from infrared imagery by the Dvorak
technique — the same imagery a vision model would read. This report asks whether
that circularity is measurable, and whether it reaches the models.

**Summary: the labels are demonstrably contaminated, and the contamination does
not reach models built on environmental predictors.** Whether it reaches models
that read the imagery is the open question, and it is the one that needs the
global patch archive.

Everything below runs on CPU from public data.

---

## A1 — Quantization of best-track intensity onto the Dvorak table

### Method

The Dvorak technique assigns a current-intensity (CI) number that maps to a
fixed wind-speed table. Snapped to the 5 kt grid best tracks are reported on,
those values are 25, 30, 35, 45, 55, 65, 75, 90, 100, 115, 125, 140, 155, 170 kt.
If intensities in satellite-only basins were read off that table, they should be
over-represented there.

**Testing this naively fails.** Comparing raw frequencies between basins gives a
ratio that rises monotonically from 0.70 at 40 kt to 2.43 at 140 kt — which
reflects the West Pacific producing more intense storms, not quantization. The
first version of this test showed exactly that and was discarded.

The correction is to divide each 5 kt bin's share by the mean share of its two
neighbours. That removes the local level of the intensity distribution and
leaves only whether a value is over-represented *relative to the ones beside
it*. The statistic reported is the mean local excess on Dvorak values minus the
mean off them; zero means no clustering. Intervals are bootstrap over cases.

### Result

All 41,312 labelled cases, 2000–2022, tropical and over water for the full 24 h
window:

| Sample | n | contrast | 95% CI |
|---|---|---|---|
| Recon-verified (NA/EP) | 15,857 | **+0.499** | [0.410, 0.598] |
| Dvorak-derived (all others) | 25,455 | **+0.694** | [0.628, 0.769] |

The intervals do not overlap. Quantization is **1.39× stronger** without
reconnaissance.

### The per-basin ordering is the more informative result

| Basin | n | contrast | 95% CI | reconnaissance |
|---|---|---|---|---|
| NA | 7,091 | 0.400 | [0.251, 0.527] | routine |
| EP | 8,766 | 0.629 | [0.346, 0.884] | nominal, rarely flown |
| WP | 11,754 | 0.663 | [0.578, 0.739] | none since 1987 |
| SP | 3,519 | 0.749 | [0.533, 1.057] | none |
| SI | 8,366 | 0.808 | [0.446, 1.054] | none |
| NI | 1,811 | 0.295 | [−0.184, 0.673] | none — not resolved |

The ordering runs NA < EP < WP < SP < SI, which tracks how much aircraft
reconnaissance each basin actually receives rather than which agency nominally
covers it. **My own binary flag is wrong in an informative way**: I classify the
East Pacific as reconnaissance-verified because it is a US-agency basin, but it
is flown far less than the Atlantic, and its quantization sits with the
satellite-only basins. That is what a provenance effect should look like. A
purely physical basin difference has no reason to produce this particular order.

The North Indian Ocean is not resolved — 1,811 cases is too few and the interval
crosses zero.

### Caveat

JTWC has historically used different CI-to-wind conversions in the western North
Pacific. That **blurs** this signal rather than creating one, so the test is
conservative exactly where the sample is largest.

---

## A2 — The Dvorak rate convention against the RI threshold

The technique holds the final T-number to a typical rate of about +1.0 per day,
with hard caps of 1.0/6 h, 1.5/12 h, 2.0/18 h and 2.5/24 h for T ≥ 4.0. Analysts
may exceed them during genuine rapid intensification, and do so more often for
intensification than weakening (Cangialosi et al. 2015; Sears & Velden 2020).

Converting those rates through the CI table:

| start (kt) | CI | +1.0 T/day | +2.5 T/24 h | typical reaches 30 kt? |
|---|---|---|---|---|
| 35 | 2.5 | +20 | +55 | no |
| 45 | 3.0 | +20 | +57 | no |
| 55 | 3.5 | +22 | +60 | no |
| 65 | 4.0 | +25 | +62 | no |
| 77 | 4.5 | +25 | +63 | no |
| 90 | 5.0 | +25 | +65 | no |
| 102 | 5.5 | +25 | +68 | no |
| 115 | 6.0 | +25 | +55 | no |

**The typical rate never reaches the 30 kt RI threshold, at any starting
intensity.** Recording an RI event therefore requires the analyst to exceed the
technique's typical rate — though not its hard cap, which permits +55 to +68 kt.

Of 2,998 recorded RI events, 98.3% exceed the typical rate implied by their
starting intensity (98.6% recon-verified, 98.2% Dvorak-derived).

**That 98.3% is close to definitional and should not be read as a discovery.**
Given that the typical rate never reaches 30 kt, almost any event meeting the RI
definition must exceed it. The finding here is structural, not empirical: the RI
threshold sits above the Dvorak convention but well below its hard limit. The
pressure on the labels is a convention rather than a ceiling, which is why it
could bias what gets recorded without making RI impossible to record.

---

## A2b — Is intensity more recoverable from imagery where it came from imagery?

### A discarded version, documented

The first form of this test asked whether IR predictors could predict *which
intensities land on a Dvorak CI value*. It was degenerate: whether an intensity
lands on the table is a deterministic function of the intensity, so the control
using intensity alone scored **ROC AUC 0.9987** and there was no headroom left
to measure anything. The IR "lift" of +0.0001 was meaningless. Discarded.

### The non-degenerate version

Train a model to recover current intensity from the two SHIPS IR predictors
alone — no environment, no persistence — and compare the two provenance groups.
If best-track intensity in satellite-only basins was itself read off this
imagery, the imagery should recover it more tightly there. Sample sizes are
matched so that "more data fits better" cannot explain a gap.

| Sample | n | MAE (kt) | R² | sd of target |
|---|---|---|---|---|
| Recon-verified | 12,561 | 14.61 | 0.438 | 25.79 |
| Dvorak-derived | 12,561 | 17.99 | 0.407 | 30.33 |

**Null, and if anything in the opposite direction.** Absolute error is *larger*
in the Dvorak-derived group, but its target is also more spread out; in variance
terms R² is 0.438 against 0.407, a difference well within what the basin
composition could produce. No evidence that intensity is more recoverable from
imagery where it was derived from imagery.

---

## A3 — Cross-domain transfer between provenance groups

### The confound that had to be fixed first

Train on one provenance group, test on the other. A model that has learned
cyclone physics should transfer; one that has learned an analyst's convention
should degrade asymmetrically.

The first run showed exactly the predicted asymmetry: a model trained on
Dvorak-derived data transferred to the recon group with *no* loss, while the
reverse lost 0.024 ROC AUC. **That result was an artefact.** The Dvorak-derived
group has 19,599 cases against the recon group's 13,038, so "trained on 50% more
data transfers better" was indistinguishable from the effect being tested.

The corrected protocol fits every model on **exactly 10,430 cases**, sampled by
storm so that no storm is split, averaged over five draws. Native skill is
measured on the held-out part of the same domain; foreign skill on all of the
other domain.

### Result

| Model | n | base rate | ROC AUC | sd | PR AUC | BSS |
|---|---|---|---|---|---|---|
| environment: recon → recon | 2,593 | 0.074 | 0.8865 | 0.0111 | 0.4249 | 0.2130 |
| environment: recon → Dvorak | 19,599 | 0.106 | 0.8343 | 0.0074 | 0.3559 | 0.1377 |
| environment: Dvorak → Dvorak | 9,164 | 0.105 | 0.8530 | 0.0040 | 0.4119 | 0.1921 |
| environment: Dvorak → recon | 13,038 | 0.071 | 0.8686 | 0.0053 | 0.3478 | 0.1672 |
| env + IR: recon → recon | 2,593 | 0.074 | 0.8924 | 0.0094 | 0.4328 | 0.2196 |
| env + IR: recon → Dvorak | 19,599 | 0.106 | 0.8404 | 0.0073 | 0.3764 | 0.1514 |
| env + IR: Dvorak → Dvorak | 9,164 | 0.105 | 0.8603 | 0.0043 | 0.4224 | 0.2007 |
| env + IR: Dvorak → recon | 13,038 | 0.071 | 0.8776 | 0.0032 | 0.3692 | 0.1778 |

ROC AUC is the headline because it does not move with the base rate, which
differs between the groups. PR AUC is reported for completeness and is **not**
comparable across them.

### Transfer gaps

| Features | testing on | native | foreign | gap | asymmetry |
|---|---|---|---|---|---|
| environment | recon | 0.8865 | 0.8686 | +0.0179 | |
| environment | Dvorak | 0.8530 | 0.8343 | +0.0188 | **−0.0009** |
| env + IR | recon | 0.8924 | 0.8776 | +0.0147 | |
| env + IR | Dvorak | 0.8603 | 0.8404 | +0.0199 | **−0.0052** |

**Null.** Crossing domains costs about 0.018 ROC AUC in *both* directions —
ordinary domain shift, of the size any cross-basin transfer would produce. The
asymmetry that would indicate a model reproducing an analyst's convention is
absent, at −0.001 and −0.005 against fold-to-fold standard deviations of
0.003–0.011.

---

## What this bounds, and what it does not

**Established.** Best-track intensities are quantized onto the Dvorak table in
every basin, most strongly where no aircraft fly, in an order that tracks actual
reconnaissance coverage rather than nominal agency responsibility. The labels
carry a visible fingerprint of the instrument that produced them.

**Also established.** That fingerprint does not reach a gradient-boosted model
built on eight environmental scalars and two IR summary statistics. Neither
transfer asymmetry nor differential intensity recoverability appears.

**Not established, and not testable with what is built.** Whether the fingerprint
reaches a model that reads the imagery. The negative results above are bounded by
an obvious limitation: **a GBM on eight environmental numbers has almost no
channel through which to reproduce an analyst's reading of a satellite picture,
because it never sees the picture.** The circularity hypothesis has always been
about imagery models. Testing it requires the same protocol — matched training
sizes, storm-level sampling, both directions — with a CNN in place of the GBM,
and that requires patches for both provenance groups.

That is the single experiment this line of work is missing, and it is why the
global patch extraction matters.

## Caveats

1. **Basins differ physically, not only in provenance.** The West Pacific
   genuinely produces more RI than the Atlantic. Every cross-basin contrast here
   is confounded to some degree. The mitigations used are matched training sizes,
   matched storm-level sampling, and the ordered per-basin gradient rather than a
   binary flag. A within-Atlantic contrast between periods with and without
   reconnaissance would be cleaner and is much smaller.
2. **The CI-to-wind table is the NHC convention.** JTWC has used others. This
   makes A1 conservative rather than optimistic.
3. **A2 is structural.** The 98.3% figure follows largely from the definitions
   and should not be quoted as an empirical finding.
4. **Effective sample size is ~1,094 independent RI episodes**, not 3,336 cases.
   Every number here should be read with that in mind.

## Reproducing

```bash
python -u scripts/17_experiment_a.py --boot 300 --repeats 5
```

CPU only, a few minutes. Outputs `reports/expA_quantization.csv`,
`reports/expA_ir_to_intensity.csv`, `reports/expA_transfer.csv`.
