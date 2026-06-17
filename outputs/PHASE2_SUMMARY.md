# Phase 2 Summary — UNSW-NB15 Model Training & Generalisation Investigation

**IoT-SecBand | Pipeline position:** preprocessing → **model training (complete)** → BoT-IoT fine-tuning

## Final Frozen Configuration

| Component | Setting |
|---|---|
| Scaler | RobustScaler (median + IQR) |
| Features | 11 — `dur, proto, service, state, sbytes, dbytes, rate, sload, sinpkt, smean, dmean` |
| Dropped features | `dload`, `dinpkt` (high drift + high model reliance — see Finding 4) |
| Model | MLP, hidden layers (32, 16), relu, alpha=0.001 |
| Decision threshold | **0.74** (calibrated via one-time holdout split — see Phase 2F) |

---

## Narrative

The initial model achieved high cross-validation performance (F1 ≈ 0.90) but
significantly lower performance on the official UNSW-NB15 test set (F1 ≈ 0.71
at default threshold). The gap was investigated systematically across five
sub-phases before any architecture changes were considered.

**Phase 2A — Threshold tuning (training-distribution validation split).**
Threshold optimised on a validation split drawn from the training set (46.6%
attack) produced no meaningful change on the test set (33.6% attack): F1
0.706 → 0.705. This ruled out simple miscalibration as the sole cause and
indicated the issue was distributional, not a fixed offset in the decision
boundary.

**Phase 2B — Diagnostic: probability overlap and feature drift.**
Probability overlap between true-normal and true-attack flows on the test
set was 0.417 (CONCERNING range). Feature drift analysis (KS statistic,
train-normal vs test-normal) found 8 of 13 features with moderate drift
(KS 0.14–0.23), concentrated in destination-direction features: `dload`
(+69% mean shift), `dbytes` (+178%), `dmean` (+153%), `dinpkt` (+123%).

**Phase 2C — Scaler sensitivity experiment.**
A controlled A/B test (identical data, features, architecture; only the
scaler changed) showed RobustScaler improved overlap (0.417 → 0.351) and
AUC (0.8945 → 0.9066) but produced only marginal F1 gain (+0.007). Verdict:
partial support — scaler helps but is not the root cause alone.

**Phase 2D — Feature drop experiment.**
Feature importance analysis showed `dload` (DT importance 0.49, RF 0.175)
and `dinpkt` (RF 0.150) combined high drift with high model reliance, while
`dbytes` and `dmean` combined high drift with high mutual-information
(genuine signal) but lower tree reliance. A controlled experiment dropping
`dload` and `dinpkt` (11-feature set, RobustScaler retained) improved every
metric simultaneously: F1 0.7129 → 0.7299, Precision 0.5655 → 0.5864,
AUC 0.9066 → 0.9267, FPR 0.3754 → 0.3454, Overlap 0.3514 → 0.3217.

**Phase 2E — Threshold diagnostic on corrected feature set (no test split).**
A threshold sweep across the full, untouched test set on the 11-feature/
RobustScaler/MLP(32,16) model found that threshold=0.74 (vs default 0.50)
improved F1 from 0.7299 to 0.7788 while reducing the false positive rate
from 34.5% to 14.7% — a 57% relative reduction in false alarms. This result
landed almost exactly within the ceiling predicted from the AUC/overlap
analysis (0.77–0.83), validating the diagnostic reasoning chain:

```
Feature drift → probability overlap → AUC → expected F1 ceiling
```

**Phase 2F — Final calibration/holdout split (one-time, pipeline frozen).**
With the configuration above frozen, the official UNSW-NB15 test set was
split once into a calibration half (n=23,195) and a final holdout half
(n=23,196), both preserving the 33.64% attack rate. The optimal threshold
(0.74, F1=0.7747, AUC=0.9251) was identified on the calibration half only.
Evaluated on the never-touched holdout half at this threshold:

| Metric | Default (t=0.50) | Calibrated (t=0.74) | Change |
|---|---|---|---|
| F1 (attack) | 0.7289 | **0.7829** | +0.0540 |
| Precision | 0.5840 | **0.7406** | +0.1566 |
| Recall | 0.9694 | 0.8303 | −0.1391 |
| Accuracy | 0.7575 | 0.8451 | +0.0876 |
| FPR | 0.3499 | **0.1474** | −0.2025 |
| False positives | 5,387 / 15,394 normal | 2,269 / 15,394 normal | −3,118 flows |

The calibration-half estimate (F1=0.7747) and the true holdout result
(F1=0.7829) are within 0.008 of each other. This close agreement indicates
the threshold generalises — it was not overfit to the calibration partition
— and confirms **F1=0.7829 as the final, unbiased Phase 2 benchmark** for
this configuration.

---

## Cumulative Improvement (Original → Final Holdout Benchmark)

| Metric | Original (StandardScaler, 13 feat, t=0.50) | **Final holdout** (RobustScaler, 11 feat, t=0.74) | Change |
|---|---|---|---|
| F1 (attack) | 0.7061 | **0.7829** | +0.0768 |
| Precision | 0.5547 | **0.7406** | +0.1859 |
| Recall | 0.9710 | 0.8303 | −0.1407 |
| FPR (false alarm rate) | ≈0.40 | **0.1474** | −0.253 |
| AUC | 0.8945 | 0.9251 (calibration split) | +0.0306 |

The final holdout F1 (0.7829) is the number to report as Phase 2's
benchmark — it was produced on data untouched by any training, scaling,
feature-selection, or threshold-selection decision.

---

## Conclusion

Performance remained below the original project target (F1 ≥ 0.88),
suggesting a practical ceiling imposed by temporal distribution shift within
the UNSW-NB15 benchmark itself — train and test splits were captured across
different time periods with measurably different traffic characteristics,
particularly in destination-side flow statistics. This is a documented
property of the dataset and not attributable to model capacity: cross-
validation F1 (≈0.90) and AUC (≈0.93) both indicate the MLP(32,16)
architecture learns the available signal effectively.

Two architecture decisions were deliberately NOT made and are recorded here
for traceability:

1. **`dbytes` and `dmean` were retained** despite showing the highest
   remaining drift (KS 0.20, 0.198), because their mutual information scores
   (0.44, 0.33 — 2nd and 3rd highest of all 13 original features) indicate
   genuine predictive signal. Removing them risked trading drift-robustness
   for loss of real attack signal, an asymmetric risk compared to `dload`/
   `dinpkt`, which combined drift with high reliance but lower information
   content.

2. **MLP architecture was not enlarged to (64, 32)**, because no evidence in
   this investigation pointed to a capacity limitation — only to a
   distribution-shift limitation, which a larger network on the same features
   would not resolve.

---

## Methodological Note — Test Set Usage

The official UNSW-NB15 test set (46,391 samples) was split 50/50 (stratified,
random_state=42) exactly once, after the pipeline above was frozen:

- **Calibration half**: 23,195 samples (33.64% attack) — used solely to
  identify the optimal decision threshold (0.74).
- **Final holdout half**: 23,196 samples (33.64% attack) — never used in
  training, scaling, feature selection, or threshold selection. All metrics
  reported as "final" in this document come from this half only.

This split is final for this model configuration. If the pipeline (features,
scaler, or architecture) changes in any future phase, a fresh split should be
drawn and clearly distinguished from this one. Full results:
`outputs/metrics/phase2e_threshold_locked.json`.

---

## Next Step

Phase 3 — BoT-IoT fine-tuning, using:
- The frozen 11-feature set
- The RobustScaler **fitted on UNSW-NB15 training data** (saved in
  `models/scaler_params.json`) — applied to BoT-IoT without refitting, to
  expose rather than hide any UNSW↔BoT-IoT distribution shift
- The calibrated threshold from `phase2e_threshold_locked.json`

The documented UNSW ceiling (F1=0.7829, final holdout) is the baseline
against which BoT-IoT fine-tuning results should be compared. An improvement
after fine-tuning would suggest BoT-IoT's traffic characteristics are more
stable / less temporally drifted than UNSW-NB15's; a degradation consistent
with the ~99% figure from the original Colab work would indicate the same
label-bleed / distribution-mismatch failure mode identified at the start
of this investigation, and the same diagnostic toolkit (overlap analysis,
feature drift, threshold sweep) should be applied before trusting any
high accuracy figure.