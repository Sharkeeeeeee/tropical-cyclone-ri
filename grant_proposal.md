# Three Controlled Variables in Satellite-Based Rapid Intensification Prediction: Label Provenance, Spatial Resolution, Temporal Sampling

## 1. Problem and unresolved questions

Rapid intensification, defined as a tropical cyclone gaining at least 30 knots in 24 hours, is the operational forecast failure carrying the highest humanitarian risk. It can elevate an expected Category 1 landfall to a Category 4 overnight, often after evacuation windows have closed.

Post-processing outputs from numerical and AI weather models has recently proven effective and computationally inexpensive. AIFS-TC (arXiv:2608.09959, August 2026) applies a learned correction to ECMWF's AIFS-Single, reaching the operational frontier for cyclone intensity from 12 h to seven days. Because that methodology requires minimal compute and is already well-established, it is not the focus of this grant.

Observational prediction using direct satellite imagery provides an independent pathway by avoiding model smoothing of compact inner-core dynamics. However, satellite-based intensity estimation is an established literature spanning decades:

- SHIPS-RII (Kaplan & DeMaria) has operated at NHC since 2003, using linear discriminant analysis on large-scale environmental predictors alongside two infrared summary statistics. DTOPS (DeMaria et al. 2021) has operated since 2018.
- DeepMicroNet (Wimmers, Velden & Cossuth, *Mon. Wea. Rev.* 2019) applies a CNN to passive microwave imagery for intensity estimation.
- The deviation-angle-variance technique (Piñeros et al. 2008; Ritchie et al. 2014) quantifies infrared gradient symmetry.
- McNeely, Lee, Wood & Hammerling (*J. Appl. Meteor. Climatol.* 2020) evaluated the ORB feature suite (organization, radial structure, bulk morphology) from GOES infrared against rapid intensification, reporting accuracy comparable to environmental predictors alone, with linear lasso matching nonlinear models. Subsequent studies (*Wea. Forecasting* 2023; 2025 multi-resolution wavelet analysis) extended this framework via Hovmöller representations of radial evolution.

Prior literature demonstrates that standard CNN training on infrared imagery rarely outperforms environmental baselines. What remains unresolved is whether this empirical ceiling reflects fundamental physical limits or artifacts of how historical satellite archives are sampled. Three candidate explanations can be isolated as testable, controlled variables:

> **A. Label provenance.** In basins lacking routine aircraft reconnaissance, best-track intensity is derived from infrared imagery via the Dvorak technique, introducing potential circularity where models reproduce analyst conventions.
>
> **B. Spatial resolution.** Tropical cyclone eyewalls measure tens of kilometers across, spanning only a handful of pixels at the 8 km resolution of long global records.
>
> **C. Temporal sampling.** Convective bursts preceding rapid intensification evolve over tens of minutes, whereas the global archive samples every three hours, potentially aliasing rapid structural transitions.

Variables B and C evaluate whether the apparent performance ceiling of infrared prediction is physical or an artifact of archive sampling. Variable A evaluates whether performance is benchmarked against a consistent ground truth. This project tests all three on the global historical record with all other experimental conditions held constant.

## 2. Evaluation of the three controlled variables

The Dvorak technique assigns intensity through discrete Current Intensity (CI) numbers mapped to specific wind speeds, constraining operational T-number changes (at most 1.0 over 6 h, 1.5 over 12 h, 2.0 over 18 h, and 2.5 over 24 h for T ≥ 4.0). While operational analysts may break these rules during severe intensification events (Cangialosi et al. 2015; Sears & Velden 2020), the constraints primarily dampen the specific phenomenon under study. Models trained on these labels learn from a target series that systematically smooths extreme intensification. Section 4 provides empirical measurements of this quantization effect across ocean basins.

Existing studies confound spatial resolution with methodology; for instance, McNeely et al. and the 2025 wavelet study analyzed different imagery products at different scales. Comparing GridSat-B1 (8 km, 3-hourly, 1980–present) and MERGIR (4 km, 30-minute, 2000–present) on identical storms and timestamps isolates resolution as an independent variable.

Hypotheses linking structural evolution to rapid intensification have historically relied on 3-hourly archives, which cannot resolve convective bursts that evolve over sub-hourly intervals. Evaluating MERGIR's 30-minute cadence directly measures the predictive value of high-frequency temporal evolution.

## 3. Investigator background and preparatory work

My background is machine learning and statistical evaluation. All preparatory infrastructure—including the label processing pipeline, statistical baselines, patch extraction routines, 141-feature structural set, training harnesses, and leakage audits—is already implemented, verified on CPU using public archives, and available in the project repository. The findings in Section 4 reflect direct empirical outputs from this preparatory work.

## 4. Completed empirical baselines

The baseline results below are computed, reproducible, and documented in the repository.

### Sample definition

The dataset joins IBTrACS v04r01 to CIRA SHIPS developmental data across all six ocean basins (2000–2022, over-water trajectories for 24 h windows), yielding 31,991 cases across 1,912 storms with a 9.23% base rate. The join matches SHIPS' 24 h intensity-change record on 99.88% of 53,285 comparable cases.

Collapsing consecutive 6-hourly observations within single intensification bursts identifies 1,094 independent RI episodes. This effective sample size motivates selecting parameter-efficient architectures over heavily parameterized deep networks.

### Statistical baselines

Models were evaluated using 5-fold cross-validation, stratified and grouped by season.

| Model | ROC AUC | PR AUC | BSS |
|---|---|---|---|
| Climatology | 0.500 | 0.092 | 0.000 |
| Logistic regression, environment only | 0.836 | 0.304 | 0.134 |
| Logistic regression, environment + IR (SHIPS-RII analogue) | 0.851 | 0.339 | 0.164 |
| Gradient boosting, environment only | 0.886 | 0.462 | 0.246 |
| Gradient boosting, environment + IR | 0.891 | 0.481 | 0.258 |

The two classical IR predictors add +0.019 PR AUC to environmental gradient boosting, matching McNeely et al.'s finding independently.

### Direct evidence of label quantization

If best-track intensities in non-reconnaissance basins rely heavily on Dvorak estimates, intensities will cluster on discrete Dvorak CI values. Normalizing each 5 kt bin by the local mean of its neighboring bins isolates local over-representation:

On all 41,312 labelled cases, with bootstrap intervals over storms:

| Sample | n | quantization contrast | 95% CI |
|---|---|---|---|
| Recon-verified (NA/EP) | 15,857 | **+0.499** | [0.410, 0.598] |
| Dvorak-derived (all other basins) | 25,455 | **+0.694** | [0.628, 0.769] |

Quantization is 1.39× stronger in basins lacking aircraft reconnaissance, with non-overlapping bootstrap intervals.

Broken out by basin, the effect forms an ordered gradient matching operational reconnaissance coverage:

| Basin | contrast | 95% CI | reconnaissance |
|---|---|---|---|
| NA | 0.400 | [0.251, 0.527] | routine |
| EP | 0.629 | [0.346, 0.884] | nominal, rarely flown |
| WP | 0.663 | [0.578, 0.739] | none since 1987 |
| SP | 0.749 | [0.533, 1.057] | none |
| SI | 0.808 | [0.446, 1.054] | none |
| NI | 0.295 | [−0.184, 0.673] | none (n = 1,811, not resolved) |

Best-track intensities exhibit quantization onto Dvorak values across all basins, lowest in the North Atlantic (0.400) and highest in satellite-only basins (up to 0.808 in the South Indian). The East Pacific (0.629) occupies an intermediate position consistent with its sparse reconnaissance coverage. This monotonic gradient across reconnaissance tiers serves as the continuous independent variable in Experiment A.

### Hand-crafted cloud structure performance

The patch archive for reconnaissance-verified basins includes 14,088 cases (13,097 with complete 4-frame sequences) and 880 RI positives across 23 seasons. Recomputing SHIPS' infrared predictor from these patches reproduces the published values at r = 0.9914 across 12,517 cases, verifying patch geometry.

A 141-feature structural set (radial profiles, cold-cloud fractions, eye contrast, azimuthal wavenumbers, deviation-angle variance, and 3 h/9 h tendencies) was evaluated against classical SHIPS predictors under identical season-grouped folds:

| Model | ROC AUC | PR AUC | BSS |
|---|---|---|---|
| Environment only | 0.8692 | 0.3759 | 0.1809 |
| Environment + SHIPS IR (2 features) | 0.8800 | 0.3863 | 0.1930 |
| **Environment + structure (141 features)** | **0.8935** | **0.4153** | **0.2157** |

The 141-feature structural set yields a +0.039 PR AUC improvement over environmental predictors, compared to +0.010 for the classical two-feature IR summary (a 3.8× difference). Permutation importance indicates that primary predictive signals arise from deviation-angle variance at 350 km (gradient symmetry) and outer cold-cloud area, alongside 3-hourly tendency metrics.

### Reconciling published negative findings

McNeely et al. reported that infrared structure yielded accuracy comparable to environmental predictors alone. Re-evaluating this comparison across model classes and performance metrics clarifies the discrepancy:

| model class | metric | environment | + structure | gain |
|---|---|---|---|---|
| logistic lasso | accuracy | 0.9287 | 0.9317 | +0.0030 |
| logistic lasso | PR AUC | 0.3190 | 0.3928 | +0.0738 |
| gradient boosting | accuracy | 0.9310 | 0.9318 | +0.0009 |
| gradient boosting | PR AUC | 0.3759 | 0.4153 | +0.0394 |

The divergence stems from metric selection. In a 7.1% base rate regime, a naive negative classifier achieves 92.9% accuracy, leaving little dynamic range to register ranking improvements for rare events. Under PR AUC, structural features provide consistent ranking gains across both linear and gradient-boosted models.

### Learned representations versus domain-engineered features

To assess whether deep neural representations improve upon hand-engineered features, a 391k-parameter CNN was trained on identical folds:

| Model | ROC AUC | PR AUC | BSS |
|---|---|---|---|
| Structure only (141 features) | 0.8364 | 0.2820 | 0.1149 |
| CNN only (391k parameters) | 0.8153 | 0.2517 | 0.1076 |

An unpretrained 391k-parameter CNN achieved 0.2517 PR AUC, falling short of the hand-crafted structural baseline (0.2820). With approximately 880 positive training episodes, domain priors (symmetry, radial profiles) outperform representations learned from scratch. Consequently, computing allocations prioritize self-supervised pretraining on unlabeled archives and multi-scale data extraction rather than extensive architectural searches.

### Data pipeline and validation

Storm-centered patch extraction from GridSat-B1 extracts 4-frame sequences (t−9, t−6, t−3, t0 h; 128 × 128 at 8 km resolution) projected onto a metric coordinate grid. The pipeline was validated against SHIPS infrared predictors at r = 0.984 on area colder than −30 °C, confirming consistent geolocation and scaling.

The 141-feature structural set passes eight physical consistency checks and reproduces the established DAV–intensity relationship (r = −0.709). A leakage audit verifies that no storm spans multiple seasons or cross-validation folds, feature lists exclude post-t0 variables, normalization statistics are fixed out-of-fold, and metric implementations match reference libraries.

On an RTX 4060 laptop, evaluation throughput reaches 1,906 samples/s with 42% GPU utilization, bound primarily by data ingestion. A complete cross-validated evaluation takes 20 minutes, confirming that high-end GPU clusters are unnecessary for this experimental scale.

## 5. Experimental design

### Experiment A: Label provenance as a controlled variable

This experiment evaluates whether models trained on Dvorak-derived labels learn cyclone physics or analyst conventions. Preliminary tests on environmental and statistical predictors across all six basins showed:

- **Cross-domain transfer:** Models trained on one provenance group and tested on the other (10,430 cases per cohort, averaged across five draws) showed symmetric transfer (+0.018 ROC AUC in both directions; asymmetry −0.001 for environmental scalars, −0.005 with IR features).
- **Intensity recoverability:** Predicting intensity from IR predictors alone yielded comparable R² across groups (0.438 vs 0.407).

Because environmental models never process spatial imagery, they cannot reproduce analyst visual interpretation patterns. Experiment A tests whether this invariance holds for vision architectures:

1. **Cross-domain transfer with vision models:** Train CNNs on reconnaissance-verified and satellite-only cohorts using matched sample sizes to test for transfer asymmetry.
2. **Continuous provenance regression:** Regress model skill against the basin-level quantization contrast to evaluate whether performance tracks label uncertainty.
3. **Constraint-window analysis:** Segment evaluations based on whether recorded 24 h intensity changes exceed standard Dvorak change constraints.
4. **Methodological implications:** Quantify provenance-related bias in global best-track training to provide correction guidelines for future machine learning studies.

### Experiment B: Spatial resolution as a controlled variable

This experiment determines whether performance ceilings in infrared prediction are physical or artifacts of coarse spatial sampling:

1. Extract matched cases from GridSat-B1 (8 km) and MERGIR (4 km) across identical storms, timestamps, and metric extents.
2. Train identical architectures across both resolutions.
3. Control for field of view versus pixel density to isolate the impact of spatial resolution.
4. Evaluate whether resolution improvements concentrate in cases with resolvable inner-core eyes.

### Experiment C: Temporal sampling as a controlled variable

This experiment evaluates whether sub-hourly cadence captures precursors of rapid intensification that 3-hourly sampling aliases:

1. Extract matched cases at 3-hourly and 30-minute intervals over identical 9 h windows (4 frames vs 19 frames).
2. Evaluate cadence against window length to distinguish frame count from observation frequency.
3. Test whether predictive gains concentrate in episodes where convective bursts initiate and decay within standard 3-hourly observation intervals.
4. Compute performance-cost scaling curves to inform operational observation requirements.

## 6. Budget

Pricing is based on RunPod A100 80GB instances ($1.59/hr) and 16-vCPU cloud instances ($0.68/hr on-demand).

| Item | Hours | Cost |
|---|---|---|
| **Data extraction** (16-vCPU on demand) | | |
| GridSat global 8 km, 4 frames, 2000–2022 | 20 | $13.60 |
| GridSat 1980–1999, record-depth robustness | 15 | $10.20 |
| MERGIR 4 km, 4 frames matched to GridSat times | 60 | $40.80 |
| MERGIR 4 km at 30-minute cadence (Experiment C) | 380 | $258.40 |
| **Storage** | | |
| Working set + self-supervised pretraining pool, ~300 GB × 3 months | | $70.00 |
| **GPU** (A100 80GB) | | |
| Experiment A (label provenance) | 17 | $27.03 |
| Experiment B (spatial resolution) | 16 | $25.44 |
| Experiment C (temporal sampling) | 30 | $47.70 |
| Self-supervised pretraining, both resolutions | 80 | $127.20 |
| Hyperparameter optimization | 25 | $39.75 |
| Random seeds and bootstrap intervals | 40 | $63.60 |
| **Contingency** (re-runs, preemptions) | | $76.28 |
| **Total** | 208 GPU-hrs, 475 CPU-hrs | **$800.00** |

Data extraction and storage constitute 49% of the budget; GPU computing constitutes 41%.

Key budget considerations:
- GPU allocations focus on self-supervised pretraining across multi-resolution data rather than exhaustive architecture searches, reflecting findings from Section 4.
- Ingestion of MERGIR at 30-minute cadence requires 380 machine hours across several terabytes of data. On-demand instances ensure reliable execution for long-running batch jobs with checkpointing.
- Local hardware supports inference and patch-level training, but cloud compute is required to handle multi-terabyte remote data ingestion that exceeds residential bandwidth limitations.

## 7. Timeline (8 weeks)

- **Weeks 1–4 (Completed):** Pipeline implementation, baseline verification, leakage audits, and initial statistical transfer experiments. GridSat patch archives for reconnaissance-verified basins (14,088 cases) are extracted and validated.
- **Week 5:** Global 8 km extraction for remaining ocean basins; initiate background MERGIR extraction.
- **Week 6:** Experiment A vision model evaluations, continuous provenance regression, and constraint-window analyses.
- **Week 7:** Experiments B and C evaluating spatial resolution and temporal cadence benchmarks.
- **Week 8:** Model synthesis, open-source code and weight release, and paper draft preparation.

## 8. Deliverables

- An open-source, reproducible data processing and evaluation pipeline, including end-to-end leakage audits.
- Quantified estimates of label provenance effects on satellite-based RI models, accompanied by evaluation recommendations.
- Controlled empirical benchmarks comparing 4 km versus 8 km resolution and 30-minute versus 3-hourly cadence, with performance scaling curves.
- Trained model weights and out-of-fold predictions organized by storm and timestamp for open verification.
- Empirical comparisons between self-supervised representations and domain-engineered feature sets.
- Complete documentation of all positive and null results.

## 9. Risks and mitigations

- **Null vision transfer:** If vision models transfer symmetrically across basins, label quantization does not distort learned representations, validating global best-track data for deep learning applications.
- **Physical basin differences:** Addressed through matched training sample sizes, continuous provenance regression, and intra-basin Atlantic reconnaissance contrasts.
- **Intermediate reconnaissance quality:** Handled by treating East Pacific data as an intermediate point along the continuous gradient.
- **Null resolution or cadence effects:** Clarifies whether the infrared skill ceiling is physical rather than instrumental, redirecting development toward microwave or radar modalities.
- **Data pipeline interruptions:** Addressed through resumable checkpoints, retry logic, and fallback mirrors.
- **Sample size limitations:** Addressed through bootstrap confidence intervals across independent storms.

## 10. Phased funding thresholds

- **$494 tier:** Omits Experiment C (30-minute MERGIR extraction and associated GPU time), retaining global label provenance and spatial resolution evaluations.
- **$300 tier:** Retains Experiments A and B with reduced seed counts and smaller pretraining sets.
- **$120 tier:** Covers global 8 km extraction and Experiment A exclusively.
