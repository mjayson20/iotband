"""
check_env.py — IoT-SecBand environment verification script.
Run with: conda run -n secband python check_env.py
"""
import sys
import os

print(f"Python: {sys.version}")
print(f"Prefix: {sys.prefix}\n")

errors = []

def check_import(pkg, attr="__version__", required=True):
    try:
        mod = __import__(pkg)
        ver = getattr(mod, attr, "?")
        print(f"  ✓  {pkg}: {ver}")
    except ImportError as e:
        short_err = str(e).split("\n")[0]
        print(f"  ✗  {pkg}: IMPORT FAILED — {short_err}")
        if required:
            errors.append(pkg)

print("=== Core packages (required for Phases 1–2) ===")
check_import("pandas")
check_import("numpy")
check_import("pyarrow")
check_import("sklearn", "__version__")
check_import("matplotlib")
check_import("seaborn")
check_import("scipy")

print("\n=== TensorFlow / Keras (required for Phase 4 — TFLite export) ===")
# TF 2.21 on Windows fails to init its GPU driver DLL when no CUDA GPU is present.
# Set CUDA_VISIBLE_DEVICES=-1 BEFORE launching Python to force CPU-only mode.
# The activation script at:
#   %CONDA_PREFIX%\etc\conda\activate.d\tf_path.bat
# sets this automatically when you run `conda activate secband`.
# If using `conda run`, set it manually:
#   set CUDA_VISIBLE_DEVICES=-1 && conda run -n secband python ...
cuda = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET")
print(f"  CUDA_VISIBLE_DEVICES = {cuda}")
if cuda == "NOT SET":
    print("  ⚠  CUDA_VISIBLE_DEVICES not set — TF may fail on machines without a GPU.")
    print("     Run `conda activate secband` (which applies the activation script),")
    print("     or set CUDA_VISIBLE_DEVICES=-1 before running this script.")
check_import("tensorflow", required=False)

print("\n=== Directory structure ===")
dirs = [
    "data/raw/unsw_nb15",
    "data/processed",
    "models",
    "outputs/eda",
    "outputs/metrics",
]
for d in dirs:
    mark = "✓" if os.path.isdir(d) else "✗ MISSING"
    print(f"  {mark}  {d}")

print("\n=== Dataset files (must be placed manually) ===")
datasets = [
    "data/raw/unsw_nb15/UNSW_NB15_training-set.parquet",
    "data/raw/unsw_nb15/UNSW_NB15_testing-set.parquet",
]
for f in datasets:
    mark = "✓" if os.path.isfile(f) else "✗ NOT FOUND — copy here before running notebook 01"
    print(f"  {mark}  {f}")

print("\n=== Model artifacts ===")
artifacts = [
    "models/label_encoders.json",
    "models/scaler_params_locked.json",
]
for f in artifacts:
    mark = "✓" if os.path.isfile(f) else "✗ MISSING"
    print(f"  {mark}  {f}")

pkl_artifacts = [
    "models/dt_model.pkl",
    "models/rf_model.pkl",
    "models/mlp_model.pkl",
]
for f in pkl_artifacts:
    if os.path.isfile(f):
        print(f"  ✓  {f}")
    else:
        print(f"  –  {f}  (not yet trained — run src/train.py after preprocessing)")

print()
if errors:
    print(f"✗ {len(errors)} package(s) failed to import: {errors}")
    print("  Run: pip install -r requirements.txt")
    sys.exit(1)
else:
    print("✓ All core packages OK.")
