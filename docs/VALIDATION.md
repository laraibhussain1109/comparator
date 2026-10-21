# Executed cloud validation

The final integrated working tree was tested on Linux x86-64, Python 3.12.14, CPU only. No physical camera, Windows host or CUDA GPU was available.

```bash
NEUROIRIS_REAL_EFFICIENTAD_SMOKE=1 \
NEUROIRIS_TEACHER_CACHE=/workspace/.neural/assets/teacher \
NEUROIRIS_NATURAL_IMAGE_DIR=/workspace/.neural/assets/natural \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
QT_QPA_PLATFORM=offscreen XDG_CACHE_HOME=/workspace/.cache \
/workspace/.neural/bin/python -m pytest -q
```

**Result: 136 passed in 66.15 seconds.** No tests skipped in this run. The ordinary lightweight run deliberately skips the two opt-in network execution tests when the assets/runtime flag are absent.

The actual-network tests used the official checksum-verified EfficientAD teacher, genuine Anomalib native losses and a real natural photograph for the penalty term. They executed actual optimizer updates and verified finite maps, tensor-checkpoint reload consistency, native tile mapping, independent calibration, service-level evaluation and paired mode benchmarking. The housing/tile samples and physical IDs in these tests are explicitly synthetic software fixtures. A one-step test and one natural photograph do not provide adequate production training or industrial accuracy evidence.

The same run includes grouped-dataset integrity and leakage checks, quality/pose/camera mocks, stale-frame and motion regressions, missing-rib/hole/rim fixtures, symmetry rejection, native coordinate transforms, FPR calibration, complete-view decisions, audit/evidence integrity, failed-write recovery, protected settings, GUI worker/ROI behavior, blind evaluation and repeatability reporting. Deprecated baseline behavior remains isolated in characterization tests under `src/legacy`.

Also executed successfully:

- `bash scripts/setup-cloud.sh --neural` after the environments were already installed, demonstrating repeatability.
- `/workspace/.neural/bin/python -m pip check`: no broken requirements.
- `QT_QPA_PLATFORM=offscreen XDG_CACHE_HOME=/workspace/.cache /workspace/.neural/bin/python app.py --smoke-test`: the actual HMI constructed, rendered and closed with exit 0.
- `git diff --check`: no whitespace errors.
- Windows CPython 3.12 torch/torchvision CUDA 12.8 wheel metadata exists and the torchvision build requires the selected torch build. No Windows/CUDA binary execution was claimed.

Actual device format/FPS, Windows startup, RTX 5070 execution/VRAM, real trained housing behavior and unseen-part/localization accuracy remain subject to [workstation acceptance](ACCEPTANCE.md). The repository's production dataset and active housing model store remain empty. Cached teacher weights are pretrained assets, not a trained housing model. Dinomaly and DINOv2 remain unavailable.

## Dataset import and metadata repair regression validation

The reported missing-sidecar import failure and empty `.jpg.json` training failure were reproduced against the original implementation. The repair update passed the full suite with the genuine EfficientAD command above: **170 passed in 128.00 seconds**, with no skips. After the final Windows file-publication adjustment, **36 import, repair, CLI, UI and training tests passed**, including two additional simulated Windows publication regressions. The actual HMI also passed its offscreen startup smoke test.

The regression fixtures cover external import, selected-field filenames, registration of photographs already inside dataset folders, empty/malformed/BOM metadata, verified recovery copies, repair backups, interrupted writes, destination collisions, duplicate identity conflicts, rejection preservation, and the Dataset Manager repair action through the real service. Repaired synthetic datasets completed geometry training and grouped evaluation; multiple captures of one housing still count as one physical part. No local Windows photographs were available in the cloud environment, and simulated Windows file-publication tests are not Windows workstation acceptance.

## Training eligibility diagnostics

The diagnostics update passed the lightweight full suite: **184 passed, 2 skipped in 26.36 seconds**. The skipped tests are the opt-in genuine EfficientAD executions; this update changes dataset grouping diagnostics and Qt presentation, without changing neural training. The offscreen HMI startup smoke test also passed.

New regressions reproduce 781 captures of one housing with GOOD/NG labels, confirm that image labels remain intact while the housing is excluded from normal training, verify cross-view and transitive capture-session grouping, and check that diagnostics and splitting reject identical provenance conflicts. Real-service Qt tests verify that these explanations appear before training, that a failed attempt preserves image and metadata bytes, and that saved session-grouping settings are reflected in the summary. The independent-housing minimum and grouped train/validation/test separation remain in force.

## Automatic training capture screening

The screening update passed the lightweight full suite: **199 passed, 2 skipped in 51.70 seconds**. The two skips are the opt-in neural tests. The actual-network service workflow was then executed separately with the verified teacher/natural-image environment above and `tests/test_neural_workflow.py`: **1 passed in 58.35 seconds**. That workflow now includes a real blurred capture, verifies its automatic exclusion and the actual accepted training count, and completes genuine EfficientAD fitting, reload, native tiling, calibration, evaluation and paired benchmarking. The offscreen HMI startup smoke test passed, and the completed exclusion panel was rendered for visual review.

Additional software regressions cover blur and landmark-registration exclusions, replacement of an unsuitable initial reference, preference for the represented native resolution over a sharp thumbnail, empty/undersized inspection ROIs, and failure when every training capture is unsuitable. They verify immutable dataset bytes/metadata, the original complete split roster, retained invalid held-out observations, actual accepted fitting IDs, saved failure reports, and preservation of the previous model when no training captures remain. A source-hash conflict remains a fatal provenance error. A completed model remains a successful result if only the subsequent external audit write fails, with its bundled screening report available after restart.

All housing images in these checks are synthetic software fixtures. These results establish screening and model-workflow behavior, without measuring the quality or industrial accuracy of the user's local photographs.
