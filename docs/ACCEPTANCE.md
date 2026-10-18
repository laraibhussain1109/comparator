# Workstation and pilot acceptance

The repository's automated fixtures establish software behavior. They do not establish production detection accuracy, minimum visible crack size or camera metrology precision. The two supplied photographs identify the housing views; both remain unlabelled reference material.

## Windows / camera / GPU execution

Record the exact Windows version, Python/packages, NVIDIA driver, GPU model, application commit and diagnostics output. Install the pinned CUDA 12.8 wheels. Confirm startup's actual computation succeeds on the RTX 5070, then confirm loaded EfficientAD inference reports CUDA. If CUDA fails, retain the diagnostic and resolve its driver/wheel/kernel error; do not label a CPU fallback as accelerated.

With only the intended Makose device selected, connect and inspect the actual decoded 3840×2160 frame dimensions, driver MJPG FOURCC and measured capture FPS. Save full-resolution captures and inspect them for JPEG blocking around the smallest rejectable defect. Do not equate advertised 30 FPS or CAP_PROP_FPS with achieved capture/inference throughput. Test camera unplug/replug, busy-device errors, Stop, reconnect to the same index, unsupported controls and a deliberately degraded format. The GUI should remain responsive; stale frames must not become inspection evidence.

Set focus, exposure, gain and constant illumination using supported readback. Verify quality gating on blur, camera shake, moving housings, clipping/specular highlights, partial framing and absence. Tune gates on real parts, then freeze those settings. Software cannot verify the physical lighting arrangement or sensor capability.

## Dataset / leakage / localization

Define the acceptable and rejectable population with an inspection engineer: legitimate rib interruptions and mounting structures, permitted machining marks/cast texture, rejectable crack/scratch/porosity size, broken/bent ribs, holes and rim damage. RGB evidence covers visible surfaces only.

Use independent GOOD and NG physical housings with repeated placements, both faces and explicit IDs. Keep one housing in exactly one partition across every view/session. Consider production-batch/session grouping. Review duplicates and all labels. Do not train references, normalize scores or choose thresholds using final test parts. Capture defect masks for localization metrics if required; image labels cannot supply pixel ground truth.

Commission saved view ROIs/exclusions using multiple acceptable parts. Check feature/hole registration under repositioning and rotation, including near-symmetric orientations. Verify uncertain alignment returns RECHECK. Inspect red contour mapping against the original frame at every quadrant and image edge; ensure intentional gaps stay unmarked and rim damage is not excluded by foreground masking.

## Train / restart / inspect / evaluate

Train both view models on the same frozen dataset/split. The full EfficientAD training run needs adequate normal examples, the official pretrained teacher and a genuine natural-image penalty dataset. A one-step network software test is not adequate model training. Retain model/configuration/dataset versions, actual training duration and VRAM usage.

Restart, reload saved models, inspect the same parts, and compare decisions/maps. Exercise a model-integrity failure and missing model/asset; the result must be RECHECK rather than PASS. Repeat this for camera and insufficient GPU-memory errors. Verify a failed new training run preserves the previous valid model.

Inspect Face A then Face B and reverse the order. Repeated capture of one face must not supply the missing opposite-face coverage. Try invalid capture, retake, premature finalization, cancellation, reset/new-part and evidence disk-write failure. Exactly one finalized physical event should be counted; cancelled/corrupt records should not enter counts. Evidence originals must retain source dimensions and actual capture metadata.

Run complete-view evaluation and paired geometry/EfficientAD/hybrid benchmarking on the unchanged saved test partition. Report actual independent-part FPR, FNR, recall, specificity, precision, confusion matrix, RECHECK/coverage rates, latency and confidence intervals. Review subtype misses and local contours. Preserve difficult cases and RECHECKs. A small validation sample with zero false rejects cannot establish a <2% population false-rejection rate; confidence intervals show the sample-size limitation.

Use repeated placement session IDs and repeated finalized inspections of the same marked physical parts. Record pose/lighting condition and model versions. Report observed repeatability alongside throughput, including manual flipping and rechecks. Correlated repeated frames are not independent samples.

Pilot goals are GOOD-part false rejection below 2%, critical visible-defect recall at least 95% for a defined defect population, useful localization and practical cycle time. Accept or reject the pilot based on measured results, not model scores or software test counts.

## Explicit scope limits

Windows GUI operation, camera negotiation/control/FPS, RTX 5070 CUDA execution, production accuracy and smallest-defect resolution require this physical workstation and real labelled housings. The cloud run cannot establish them.

Dinomaly and DINOv2 comparison are unavailable in this implementation. Additional camera providers, programmable lighting, full 360° coverage and precision metrology are not connected or simulated. The metrology interface requires a supplied validated calibration before any millimetre conversion; no 1 µm accuracy is claimed. RGB cannot reveal invisible subsurface porosity.
