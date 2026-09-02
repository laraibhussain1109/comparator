# AI Casting Inspection System

A complete, local, ready-to-run visual-inspection prototype for aluminium cast and machined parts. The operator supplies only class folders (`GOOD` and `NG`), presses **TRAIN**, then presses **START INSPECTION**. No masks, boxes, defect classes, CSV files, paired images, or YOLO labels are required.

## What is included

- Pose normalization (foreground, principal orientation, scale/crop, and optional ECC fine alignment) with a low-confidence **RECHECK** result.
- A representative multi-image golden bank selected by clustered appearance descriptors.
- A lightweight PatchCore-style nearest-neighbour normal patch memory trained **only on GOOD images**. It produces an internal score map; the GUI never displays it.
- A supervised GOOD/NG reference-comparison classifier trained from automatically generated logical pairs.
- Robust learned shape checks using normalized area, aspect, circularity, extent, center, radius and hole count statistics.
- F2-weighted validation calibration and untouched held-out test reporting.
- Conservative SSIM/edge-supported localization, morphology and clean red issue contours—never a heatmap.
- Temporal voting, latest-frame capture, event locking, inspection counts and optional NG evidence records.
- A dark PyQt6 industrial HMI with background training and inspection workers.

## Installation

Python 3.10–3.12 is recommended.

```bash
cd /workspace/comparator
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux, a Qt installation may additionally require the OS packages that provide `libEGL`, `libGL`, and XCB. A CUDA-enabled PyTorch installation is not needed by the portable default backend. The implementation uses CPU-efficient OpenCV/scikit-learn models and automatically reports the host in Technical Details.

## Run

1. Copy real images into `dataset/GOOD/` and `dataset/NG/` (at least 3 unique images per class; 10+ is recommended).
2. Start the HMI:

```bash
python app.py
```

3. Press **TRAIN** and wait for **TRAINING COMPLETED**.
4. Connect a camera, then press **START INSPECTION**. Use **STOP INSPECTION** before disconnecting it.

The app probes the configured camera first and then other indices. Edit `camera.index` in `config.yaml` when necessary.

## Training workflow

TRAIN scans supported image formats case-insensitively, validates readability, removes exact duplicates by SHA-256, warns about perceptual duplicates when ImageHash is installed, and creates seeded stratified 70/15/15 splits. It creates the canonical registration template and golden bank, builds the GOOD-only patch memory, trains the supervised comparison model with GOOD and NG, learns the GOOD-only geometry profile, calibrates thresholds exclusively on validation data, evaluates once on the test set, and atomically leaves reusable artifacts. Metrics are calculated from actual held-out predictions and are not synthesized.

Small datasets necessarily give unstable metrics; collect varied production lighting, pose and harmless surface examples in GOOD, and representative failures in NG.

## Inspection workflow

The dedicated capture/inference thread acquires the newest camera frame, locates and registers the component, selects its closest golden reference, executes anomaly/comparison/geometry branches, fuses localization, makes a hierarchical decision, and temporally confirms it. A low registration confidence becomes **RECHECK**. A confirmed event is counted once, remains locked while the part is stationary, and rearms after removal. Confirmed NG records are stored under `inspection_results/YYYY-MM-DD/` as original image, red-contour image, and JSON evidence.

The customer view receives only the component, result, counters, and red outlines. Raw anomaly arrays remain internal and are never rendered.

## Artifacts

```text
artifacts/
├── registration/template.png
├── golden_bank/golden_*.png
├── golden_bank/manifest.json
├── patchcore/model.joblib
├── visual_changenet/model.joblib
├── geometry_profile.json
├── thresholds.json
├── dataset_split.json
└── training_summary.json
```

Models load at inspection startup, so successful training is not repeated after restarting the application. Delete `artifacts/` or press TRAIN to rebuild for a changed product.

## NVIDIA TAO

The production-default `sklearn` comparison backend is fully functional without proprietary software. TAO command detection and invocation are isolated in `visual_change_detector.py`. If `visual_change.backend: tao` or `require_tao: true` is selected, training fails clearly and actionably when TAO is absent; it never claims success. Since TAO VisualChangeNet specifications and supported backbones vary by installed TAO release, the portable build deliberately does not pretend to execute an unverified site-specific experiment. Use the default backend for the complete out-of-box workflow.

## Test

```bash
pytest -q
```

Tests need no physical camera and cover dataset discovery/splitting, duplicates, registration, geometry, calibration, decisions, temporal voting, state locking, rendering, model saving/loading and an end-to-end train/infer smoke workflow.

## Configuration and logs

All operator defaults are in `config.yaml`: paths, camera, canonical size, registration confidence, memory size, geometry tolerance, localization cleanup, temporal confirmation, record saving, backend and logging. Runtime logs rotate in `logs/inspection.log`. Thresholds are learned from validation data; values in configuration are operational constraints rather than fabricated anomaly cutoffs.
