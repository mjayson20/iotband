# IoT-SecBand — Local Setup Guide

This guide gets a fresh machine to the point where you can run the existing
Phase 1–2 pipeline and start contributing to Phase 3 (BoT-IoT fine-tuning).

---

## 1. Prerequisites

- **Anaconda or Miniconda** installed
- **Python 3.10** (used throughout this project)
- Access to the UNSW-NB15 dataset files (ask in the team channel if you don't have them — they are not in version control due to size)

---

## 2. Clone / Pull the Project

```bash
git clone <repo-url> IoT-Research-Project
cd IoT-Research-Project
```

If you already have the repo, just `git pull` to get the latest scripts.

---

## 3. Create the Conda Environment

```bash
conda create -n secband python=3.10
conda activate secband
```

All commands below assume this environment is active. You should see
`(secband)` at the start of your terminal prompt.

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

| Package | Used for |
|---|---|
| `pandas`, `pyarrow` | Reading/writing `.parquet` files |
| `numpy` | Numerical operations |
| `scikit-learn` | Models (DT, RF, MLP), scalers, metrics |
| `matplotlib`, `seaborn` | EDA and evaluation plots |
| `scipy` | KS-statistic drift analysis |
| `jupyter` | Running the notebooks in VS Code or browser |

---

## 5. Set Up the Directory Structure

Create the following folders inside the project root (if not already present):

```
IoT-Research-Project/
├── data/
│   ├── raw/
│   │   └── unsw_nb15/        ← place dataset files here
│   └── processed/            ← auto-generated, do not commit
├── notebooks/
├── src/
├── models/                   ← auto-generated, do not commit
└── outputs/
    ├── eda/                  ← auto-generated
    └── metrics/              ← auto-generated
```

```bash
mkdir -p data/raw/unsw_nb15 data/processed notebooks src models outputs/eda outputs/metrics
```

---

## 6. Place the Dataset

Copy these two files into `data/raw/unsw_nb15/`:

- `UNSW_NB15_training-set.parquet`
- `UNSW_NB15_testing-set.parquet`

**Do not rename these files** — the scripts reference these exact filenames.

---

## 7. Add the Source Files

Pull the latest `.py` files into `src/` and the `.ipynb` files into `notebooks/`.
At minimum you need, in this order of relevance:

| File | Role |
|---|---|
| `src/preprocess.py` | Core Phase 1 cleaning/encoding/scaling logic |
| `src/eda.py` | Phase 1 plotting functions |
| `src/run_preprocess.py` | Standalone Phase 1 runner |
| `src/train.py` | Phase 2 model training |
| `src/evaluate.py` | Phase 2 evaluation plots |
| `src/diagnose_phase2.py` | Probability overlap + feature drift diagnostics |
| `src/scaler_experiment.py` | Controlled scaler/feature-drop experiments |
| `src/threshold_diagnostic.py` | No-split threshold sweep |
| `src/tune_threshold_v3.py` | One-time calibration/holdout split (already run — do not re-run unless the pipeline changes) |
| `src/save_locked_scaler.py` | Persists the canonical Phase 2 scaler artifact |

Check `project_progress.md` for what each phase actually does — this file only covers *running* things, not the reasoning behind them.

---

## 8. Reproduce the Current Pipeline State

Run these **in order** from the project root, with the `secband` environment active:

```bash
# Phase 1 — preprocessing
run all the cells under notebooks/01_unsw_preprocess.ipynb

# Phase 2 — training (creates models/dt_model.pkl, rf_model.pkl, mlp_model.pkl)
python src/train.py

# Phase 2 — diagnostics (optional, regenerates plots already discussed in PHASE2_SUMMARY.md)
python src/diagnose_phase2.py

# Phase 2 — persist the canonical scaler for the frozen 11-feature config
python src/save_locked_scaler.py
```

**Do not re-run `src/tune_threshold_v3.py`.** It performs a one-time
calibration/holdout split on the official UNSW-NB15 test set, which has
already been done and is documented in `PHASE2_SUMMARY.md` and
`outputs/metrics/phase2e_threshold_locked.json`. Re-running it on the same
test set repeatedly would erode its validity as an unbiased holdout.

---

## 9. Verify Your Setup Worked

After step 8, confirm these files exist:

```
data/processed/unsw_train_clean.parquet
data/processed/unsw_test_clean.parquet
models/dt_model.pkl, rf_model.pkl, mlp_model.pkl
models/label_encoders.json
models/scaler_params_locked.json     ← the one to use, NOT scaler_params.json
outputs/metrics/phase2_results.json
```

Open `outputs/metrics/phase2_results.json` and check `mlp.f1_attack` is
approximately `0.73` (at the default threshold — this is expected; the
calibrated benchmark of `0.7829` uses threshold=0.74, documented separately
in `PHASE2_SUMMARY.md`).

---

## 10. Before You Start Contributing

Read these two files first — they contain the reasoning, not just the
results, and will save you from re-discovering things the team already
worked through:

- **`project_progress.md`** — current status, frozen configuration, what's
  next, open questions for the team
- **`PHASE2_SUMMARY.md`** — full narrative of why the model is configured
  the way it is

Key things to know before touching the pipeline:

- The **frozen 11-feature set** is `dur, proto, service, state, sbytes, dbytes, rate, sload, sinpkt, smean, dmean`. Don't add back `dload`/`dinpkt`/`sloss`/`dloss`/`spkts`/`dpkts` without re-running the drift/importance diagnostics — they were deliberately dropped with documented reasoning.
- **Always use `models/scaler_params_locked.json`**, not `models/scaler_params.json`, when aligning any new dataset (e.g. BoT-IoT) to this pipeline. The latter is an outdated 17-feature artifact.
- **Never refit a scaler on a new dataset.** Load the locked scaler's `center`/`scale` values and apply them — refitting hides distribution shift instead of exposing it.
- **The UNSW-NB15 test set's calibration/holdout split is final** for this configuration. If you change the feature set, scaler, or architecture, you must draw a fresh split and label it clearly as new — don't reuse or further tune against the existing one.

---

## 11. Common Issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError` for pandas/sklearn/etc. | Confirm `secband` env is active (`conda activate secband`), then re-run the pip install command in step 4 |
| `pip install` fails on Windows for a package | Try `pip install --upgrade pip` first, then retry |
| Scripts can't find raw parquet files | Confirm exact filenames and that they're in `data/raw/unsw_nb15/`, not `data/raw/` |
| `TypeError: Object of type bool_ is not JSON serializable` | You're on an outdated `preprocess.py` — pull the latest version; this was fixed early in Phase 2 |
| `state` encoding warning about unseen categories | Expected and harmless — 2 categories in the test set don't appear in training data; they're mapped to class 0 by design |

---

## 12. Who to Ask

If you're stuck on environment/setup issues specifically (not modeling
decisions), check with whoever last touched `src/preprocess.py` or
`src/train.py` — see the team channel for current ownership.