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

## Windows long-path file creation

The reported screening temporary path measured 260 UTF-16 units, exceeding the legacy Windows limit when its terminating NUL is included, while the destination JSON path was only 222 units. Atomic JSON writes now use short, exclusive temporary names. Filesystem boundaries normalize absolute Windows and UNC paths into the extended namespace; operator-facing paths remain readable. Image encoding and PyTorch checkpoint I/O use Python file handles, and dataset containment/hash checks retain consistent path identities.

**Native Windows validation passed:** [Windows training file IO, run 39731647811](https://github.com/laraibhussain1109/comparator/actions/runs/39731647811), on commit `fad4cd0`. The Windows runner installed the real CPU PyTorch package and passed the selected path/import/config/image/checkpoint/training tests, followed by `python app.py --smoke-test`. These include final file paths beyond 260 characters, Unicode folders, automatic report creation, geometry training/publication/reload/evaluation, source preservation, and safe tensor checkpoint roundtrips. Native testing exposed a second issue with relative paths from an already-extended working directory; its fix and regression tests are included.

The full Linux neural-enabled suite passed **238 tests in 92.10 seconds**, without skips. Its genuine EfficientAD service workflow now places configuration, dataset, teacher cache, natural photographs, report files and model checkpoints under a deep Unicode root. It completes real optimizer execution, checkpoint reload, calibration, evaluation and paired benchmarking. After the native Windows relative-path correction, the focused path/config/training suite passed **24 tests** locally and the native Windows workflow passed. Tests use software housing fixtures and CPU execution; these results do not establish industrial defect accuracy or validate physical camera/GPU hardware.

## Interrupted Imagenette HTTPS downloads

The configured official fastai URL was checked against its source README and
the live S3 server. A complete download of `imagenette2.tgz` using the application
downloader received **1,557,161,267 bytes**, matched the published SHA256, and
automatically extracted **9,469 training photographs**. The checksum and dataset
were verified in the Linux cloud environment; the user's Windows TLS connection
was not available for direct reproduction.

The downloader now retries transient connection failures up to five times with
bounded backoff, resumes private temporary files using validated HTTP Range
responses, and safely restarts if a server ignores Range. Transfer, retry,
verification and extraction messages reach the training observer. Certificate
verification and final SHA256 verification remain required. Certificate errors,
missing URLs, malformed ranges, invalid checksums and local write errors are not
silently treated as successful downloads. The training guide includes a browser
download fallback to the same verified archive cache.

The full suite passed **260 tests, with 2 opt-in neural tests skipped**, in
41.12 seconds. Those two genuine EfficientAD tests were then executed separately
with the verified teacher and the newly downloaded Imagenette photographs:
**2 passed in 52.17 seconds**, including service training, reload, calibration,
evaluation and benchmarking under deep Unicode paths. The focused asset/model
suite passed **32 tests, with 2 optional tests skipped**, in the lightweight
environment. These checks include the screenshot's exact SSL failure after a
partial transfer, successful Range resumption, ignored and malformed ranges,
premature EOF, retry limits, HTTP/certificate rejection, disk errors, checksum
protection, offline cache reuse and training-progress forwarding. The offscreen
application startup smoke test passed. Windows CI now includes these asset
regressions; this validation does not claim to repair a persistent fault in the
user's network or TLS installation.

## Automatic Imagenette preparation

Missing natural-image data now triggers official Imagenette preparation during
EfficientAD training, including configurations that retain the obsolete
`download_imagenette: false` setting. The training dialog shows an informational
notice instead of a download checkbox, so operator training does not need
technician access to prepare the dataset. Extraction follows `imagenet_dir`,
including custom paths, and subsequent calls reuse the photographs.

The full pinned neural-enabled suite passed **273 tests in 95.33 seconds**,
without skips, including genuine EfficientAD optimizer/checkpoint execution and
the service train/reload/calibration/evaluation workflow. The focused automatic
preparation and UI suite passed **33 tests**. Regressions cover default directory
creation, missing/legacy false settings, locked and unlocked training, offline
reuse, custom cache/image paths, archive filtering, and failed downloads without
substituting housing images. The previous-model preservation test now injects a
fitting failure explicitly, instead of relying on missing natural assets to
stop training.

A separate check used the actual checksum-verified 1,557,161,267-byte archive,
an empty custom image directory and the legacy false setting. It automatically
extracted all **9,469 training photographs**, then reused them with network access
disabled. The application also passed its offscreen startup smoke test. Windows
CI includes the new automatic preparation and operator UI regressions.
