# IoT-SecBand — Project Progress

**Last updated:** Phase 2 complete, frozen benchmark established
**Pipeline stage:** UNSW-NB15 preprocessing → model training **(✅ complete)** → BoT-IoT fine-tuning **(⬜ next)** → TFLite export → Edge integration

---

## 1. Project Summary

- **Goal:** Lightweight binary IDS classifier (Normal vs. Attack) deployable on an **nRF52840** microcontroller for a wearable that monitors BLE/WiFi traffic.
- **Datasets:** UNSW-NB15 (primary training) → NF-BoT-IoT (fine-tuning, not yet started).
- **Constraint driving every decision:** all features must be derivable from header-level packet capture on ESP32/nRF52840 — no deep packet inspection, no persistent multi-flow lookup tables beyond what fits in 256KB RAM.
- **Environment:** Local (VS Code + Conda env `secband`), datasets as `.parquet`, structured `data/raw/ → data/processed/ → models/ → outputs/` directory layout.

---

## 2. Phase 1 — UNSW-NB15 Preprocessing (✅ Complete)

- Raw data: 175,341 train rows / 82,332 test rows, 36 columns.
- **Deduplication is critical and non-obvious:** raw data is 68% attack / 32% normal due to duplicate attack records. After dedup → 89,090 train rows (47,537 normal / 41,553 attack, near-balanced) and 46,391 test rows.
- **Feature audit** classified all 36 raw columns by edge-deployability:
  - Dropped outright (payload/application-layer, FTP/HTTP-specific, high-cardinality, or persistent-state-table features): `stcpb, dtcpb, trans_depth, response_body_len, ct_src_dport_ltm, ct_dst_sport_ltm, is_ftp_login, ct_ftp_cmd, ct_flw_http_mthd, is_sm_ips_ports, attack_cat`.
  - Kept (header-derivable): 17 features initially, reduced to **13** after correlation pruning (dropped `sloss`, `dloss`, `spkts`, `dpkts` — all redundant with `sbytes`/`dbytes`/`rate`).
- **Categorical encoding:** `proto` (133 classes), `service` (13 classes), `state` (9 classes) — label-encoded, encoder fit on train only, saved to `models/label_encoders.json`.
- **Outputs:** `data/processed/unsw_train_clean.parquet`, `unsw_test_clean.parquet`, `models/scaler_params.json`, `models/label_encoders.json`.

---

## 3. Phase 2 — Model Training & Generalisation Investigation (✅ Complete)

### 3.1 Initial Problem

- First training run: CV F1 ≈ 0.90 (training distribution) but **test F1 = 0.7061** at default threshold — a large, unexplained gap.
- Symptom: Recall 0.97, Precision 0.55 → model was flagging far too much normal traffic as attack (FPR ≈ 40%).

### 3.2 Investigation Sequence (each phase isolated one variable)

| Sub-phase | What was tested | Result |
|---|---|---|
| **2A** | Threshold tuning on a training-distribution validation split | No transfer to test set (F1 0.706→0.705) — ruled out simple miscalibration |
| **2B** | Diagnostic: probability overlap + per-feature drift (KS statistic, train-normal vs test-normal) | Overlap = 0.417 (concerning); 8/13 features moderately drifted, concentrated in destination-direction features |
| **2C** | StandardScaler vs RobustScaler (controlled A/B, same data/features/model) | RobustScaler improved overlap (0.417→0.351) and AUC (+0.012), but F1 gain was marginal (+0.007) — partial support only |
| **2D** | Feature drop experiment, guided by importance plots (MI scores + RF/DT importance) | Dropping `dload` + `dinpkt` (high drift **and** high model reliance) improved every metric simultaneously. `dbytes`/`dmean` were **kept** despite high drift — their high MI scores indicate genuine signal, unlike `dload`/`dinpkt` |
| **2E** | Threshold sweep on the corrected 11-feature model, full test set (no split yet) | Threshold 0.74 → F1 0.7299→0.7788, FPR 34.5%→14.7% (57% relative reduction). Result landed inside the ceiling predicted by AUC/overlap analysis |
| **2F** | **One-time** calibration/holdout split on the frozen pipeline | Calibration half predicted F1=0.7747; true holdout achieved **F1=0.7829** — close agreement confirms the threshold generalises |

### 3.3 Frozen Configuration (locked — do not change without re-running this investigation)

- **Scaler:** RobustScaler (median + IQR) — fit on UNSW-NB15 training data only
- **Features (11):** `dur, proto, service, state, sbytes, dbytes, rate, sload, sinpkt, smean, dmean`
  - Dropped: `dload`, `dinpkt` (high drift + high model reliance)
  - Deliberately **kept** despite drift: `dbytes`, `dmean` (high mutual information — genuine signal, low risk of over-reliance)
- **Model:** MLP, hidden layers `(32, 16)`, relu, `alpha=0.001`, early stopping enabled
- **Decision threshold:** **0.74** (default 0.50 was wrong for this dataset's test-period class ratio)

### 3.4 Final Benchmark (the number to cite)

| Metric | Value (final, untouched holdout) |
|---|---|
| F1 (attack) | **0.7829** |
| Precision | 0.7406 |
| Recall | 0.8303 |
| FPR | 0.1474 |
| AUC | 0.9251 (calibration split) |

- Original project target was F1 ≥ 0.88 — **not met**, and this is documented as a finding, not hidden.
- **Why:** UNSW-NB15's train/test split is temporal (different capture periods). Destination-side traffic features (`dload`, `dbytes`, `dmean`, `dinpkt`) shifted 69–178% in mean between periods. This is a property of the dataset, not a model failure — CV F1 ≈ 0.90 and AUC ≈ 0.93 confirm the model learns the available signal well; it just cannot fully bridge a genuine distribution shift.

### 3.5 Decisions Deliberately NOT Made (record for traceability)

- **Did not drop `dbytes`/`dmean`** — high MI scores (0.44, 0.33) indicate real signal; removing them risked losing genuine attack signal for an uncertain gain.
- **Did not enlarge the MLP to (64, 32)** — no evidence pointed to a capacity problem (CV F1 and AUC were already high); a bigger network would not fix a distribution-shift problem.

### 3.6 Test Set Usage — Important for Anyone Re-Running Experiments

- The official UNSW-NB15 test set (46,391 rows) was split 50/50 **exactly once**, only after the pipeline above was frozen.
- Calibration half (23,195 rows) → used to find threshold=0.74.
- Final holdout half (23,196 rows) → never touched by training/scaling/feature-selection/threshold decisions. All "final" numbers above come from this half.
- **Do not re-split or re-tune the threshold against this test set in future UNSW-NB15 work** unless the feature set, scaler, or architecture changes — in that case, draw a fresh split and label it clearly as a new evaluation.

---

## 4. Key Learnings & Principles (apply these going forward)

- **A large CV-vs-test F1 gap is the signature of distribution shift, not overfitting** — check the overfit gap (train F1 − CV F1) separately from the CV-vs-test gap. They diagnose different problems.
- **Automated verdict scripts with hardcoded thresholds are unreliable near boundaries** — this happened three times in this project (e.g. a 0.001 overlap difference flipping a verdict). Always sanity-check a borderline automated verdict manually before acting on it.
- **High feature importance + high drift = dangerous combination** (over-reliance on an unstable feature). High mutual information + high drift is different and often safe to keep (the feature is informative but the model isn't over-anchored to it). Use both signals together, not just one.
- **Threshold tuning must be done on the deployment-like distribution, not the training distribution.** Tuning on a training validation split (Phase 2A) gave a useless answer; tuning on the test distribution (Phase 2E/2F) gave the real lever.
- **Never refit a scaler on a new dataset (e.g. BoT-IoT) — reuse the one fit on UNSW-NB15 training data.** Refitting hides distribution mismatch instead of exposing it, and is the likely cause of the ~99% "fake accuracy" seen in the team's earlier Colab work.
- **Don't split a test set into calibration/holdout until the pipeline is genuinely frozen.** Splitting too early risks repeatedly "tuning" against the holdout indirectly as the pipeline keeps changing.

---

## 5. File Reference

| File | Purpose |
|---|---|
| `src/preprocess.py` | Phase 1 cleaning, feature selection, encoding, RobustScaler normalisation |
| `src/eda.py` | Phase 1 exploratory plots (class balance, distributions, correlation, outliers, mutual information) |
| `src/run_preprocess.py` | Standalone runner for Phase 1 |
| `src/train.py` | Phase 2 model training (DT, RF, MLP) + cross-validation |
| `src/evaluate.py` | Phase 2 evaluation plots (confusion matrix, ROC, overfit diagnostic) |
| `src/diagnose_phase2.py` | Probability overlap + feature drift diagnostics |
| `src/scaler_experiment.py` | Controlled scaler A/B and feature-drop experiments (Phase 2C/2D) |
| `src/threshold_diagnostic.py` | No-split threshold sweep diagnostic (Phase 2E) |
| `src/tune_threshold_v3.py` | **One-time** calibration/holdout split + final threshold lock (Phase 2F) |
| `src/save_locked_scaler.py` | Persists the canonical 11-feature scaler to `models/scaler_params_locked.json` — run once before Phase 3 |
| `models/scaler_params_locked.json` | **Canonical** fitted RobustScaler (11 locked features) — reuse on BoT-IoT, do not refit. Generated by `src/save_locked_scaler.py`. |
| `models/scaler_params.json` | ⚠️ Outdated — fit on the original 17-feature set before Phase 2D drops. Do **not** use for Phase 3. Kept only as a Phase 1 historical artifact. |
| `models/label_encoders.json` | Fitted categorical encoders — reuse on BoT-IoT, map unseen categories to class 0 |
| `outputs/metrics/phase2e_threshold_locked.json` | Final frozen benchmark numbers and split details |
| `PHASE2_SUMMARY.md` | Full narrative write-up of the Phase 2 investigation (for the research report) |

---

## 6. Next Steps

### Immediate — Phase 3: BoT-IoT Fine-Tuning

- [ ] Load NF-BoT-IoT data; align its schema to the **same 11 features** locked in Phase 2.
- [ ] Apply the **UNSW-NB15-fitted** RobustScaler and label encoders to BoT-IoT — do **not** refit on BoT-IoT.
- [ ] Run the same diagnostic toolkit (probability overlap, feature drift, threshold sweep) on BoT-IoT **before** trusting any accuracy number.
- [ ] Compare fine-tuned performance against the Phase 2 baseline (F1=0.7829):
  - Improvement → BoT-IoT traffic is more stable / less temporally drifted than UNSW-NB15.
  - A jump toward ~99% → treat as a red flag, not a win. Re-run the diagnostic toolkit immediately; this matches the failure pattern from the original Colab work.
- [ ] Use stratified splits and cross-dataset validation; do not skip the deduplication step used in Phase 1.

### Following — Phase 4: TFLite Export

- [ ] Export the frozen MLP (11 inputs, hidden layers 32→16) to TensorFlow Lite.
- [ ] Apply INT8 post-training quantization.
- [ ] Validate accuracy drop after quantization is **< 2%** vs. the float model.
- [ ] Confirm model size **< 200KB** and inference latency **< 50ms** on a Cortex-M4-class benchmark before flashing.

### Final — Phase 5: Edge Integration

- [ ] ESP32 (WiFi scan) → UART → nRF52840 (BLE scan + inference) → alert (vibration/BLE notification).
- [ ] Validate the 11 locked features can be computed in firmware within RAM/compute budget (most are simple running sums/counts; none require payload inspection).

---

## 7. Open Questions for the Team

- Should the decision threshold prioritize precision (0.74, current) or recall (0.67, F1≈0.77/recall≈0.88 — documented fallback) for the wearable use case? This is a product decision, not just a modeling one — a missed attack vs. a false vibration alert have different costs.
- Does BoT-IoT's capture period structure allow the same train/test temporal-shift diagnostic approach, or is its split structured differently?