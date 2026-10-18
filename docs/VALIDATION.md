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
