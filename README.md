# NeuroIris — Aluminium Vision Inspection

**Stavatech Technology Private Limited**

A local PyQt6 inspection workstation for the two faces of a circular aluminium LED housing. Face A is the recessed machined face with holes and mounting features; Face B is the cast face with radial cooling ribs and a central square. The supplied photographs establish these reference views; they are neither labelled defect examples nor a training/validation dataset.

The current application implements a **single selected camera**, native-resolution capture, per-view geometry and real EfficientAD, physical-part-separated training/evaluation, and a guided two-face inspection session. It starts without a camera or GPU and reports unavailable hardware/models explicitly. No untrained model, sample camera feed, or fabricated accuracy is presented as working inspection.

## Install on the Windows 11 workstation

Use **64-bit Python 3.12**. This is the demonstrated interpreter version; the pinned SciPy release requires it. Install a current NVIDIA driver compatible with CUDA 12.8 for the RTX 5070. PyTorch wheels include their CUDA runtime; a separate CUDA toolkit is not required.

From the repository directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\scripts\install-neural.ps1
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe app.py --diagnostics
.\.venv\Scripts\python.exe app.py
```

Alternatively `scripts\setup.ps1 -Neural` performs installation, software tests and diagnostics. Use `-CpuOnly` for a workstation without compatible CUDA. Geometry/HMI development only needs `requirements.txt`; neural training/inference additionally needs `requirements-neural.txt`. Installation errors preserve package/TLS/checksum verification.

The CUDA installer selects **torch 2.7.1+cu128 / torchvision 0.22.1+cu128 / anomalib 2.2.0**. Official Windows CPython 3.12 wheel metadata was verified, but this cloud host has no Windows installation or RTX 5070. Startup performs an actual CUDA computation and reports its result, GPU name, available memory, CUDA build and loaded-model device. A training device saved in an old checkpoint is never used as proof of current GPU inference.

For Linux development:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
QT_QPA_PLATFORM=offscreen .venv/bin/python app.py --smoke-test
```

See [neural setup and weights](docs/NEURAL.md) for CPU/CUDA commands, verified assets and optional real-network execution tests.

## Prepare the real housing dataset

Open **DATASET MANAGER**, select the face, label and **physical part ID**, then import photographs or capture from the selected camera. Give every photograph of the same actual housing the same ID, including flipped faces and repeated placements. Session IDs can identify placement or lighting sessions. Label individual views GOOD/NG; any NG view makes that physical part defective for grouping. Do not invent independent IDs for repeated photographs of one housing.

The managed structure is:

```text
dataset/inner_face/GOOD/    dataset/inner_face/NG/
dataset/ribbed_face/GOOD/   dataset/ribbed_face/NG/
dataset/side_left/GOOD/    dataset/side_left/NG/
dataset/side_right/GOOD/   dataset/side_right/NG/
```

Every imported original has a write-once JSON sidecar with physical ID, view, source, timestamp, dimensions and SHA256. Review previews and quality, then reject unsuitable samples. Removal adds an audit tombstone and retains the original. Exact duplicates are blocked; perceptual similarity produces a warning. Copying unmanaged files into these folders does not establish their provenance; use the manager or import CLI.

```powershell
.\.venv\Scripts\python.exe -m src.cli import C:\captures\housing17_inner.png --view inner_face --label GOOD --part-id housing17 --session-id placement1
.\.venv\Scripts\python.exe -m src.cli dataset
```

Training splits **globally by physical ID across all views**, not by image. Optional session grouping keeps entire acquisition groups together. NG physical parts never enter normal representation/reference training. The minimum software requirement is three independent GOOD groups with relevant captures in each partition; this is far too little to substantiate a low industrial false-rejection rate. Collect many varied independent GOOD parts, NG examples spanning the rejection specification, repeat placements, and different production sessions. Reuse a fixed dataset and split settings when comparing models; complete-view evaluation rejects inconsistent saved split rosters.

NG data is optional for GOOD-only research training. Without independent NG validation/test parts, missed-defect rate and defect recall remain unknown (`null`), not 0% or 100%. No automatic defect type is assigned to a generic neural anomaly.

The training screen lists **GOOD/NG image counts separately from eligible physical housings**. If one housing ID has both GOOD and NG captures, its GOOD files keep their image labels, but the whole housing is excluded from normal training. For example, 500 GOOD images and 281 NG images all assigned to `LEMON` remain 781 images of one NG housing, with zero independent GOOD housings available. Successful import alone does not make this dataset eligible for an independent train/validation/test split. Keep the same real ID for repeat captures, including before/after-defect captures; collect independent GOOD housings for this workflow. If capture-session grouping is enabled, the summary also shows the resulting independent group count.

## Repair missing or damaged image metadata

Report and model files are created automatically. Windows paths in nested installation folders and folders with non-ASCII names are supported through the filesystem layer; temporary report filenames stay short. An earlier version appended an extra UUID to the destination filename and could hit Windows' legacy path limit before the report was created. Update the application while keeping your existing dataset; do not create empty reports or JSON metadata files manually.

If training reports `Invalid provenance metadata ... .jpg.json: Expecting value`, update to the latest `codex/neuroiris-inspection-system` branch and keep your existing `dataset` directory. Do not delete photographs or create empty JSON files to bypass the error.

Open **DATASET MANAGER**. Unregistered images and damaged metadata appear as **Needs Registration** / **Needs Repair**, so they no longer prevent opening the manager. Select the affected images of **one physical housing**, choose the correct view/GOOD or NG label, enter its actual physical part ID and capture session, and click **Register / Repair Selected**. Repeat for different housings with their own real IDs, then train again.

The importer assigns a descriptive, Windows-safe filename from the selected fields plus a unique suffix. It accepts photos already placed in dataset folders and preserves external source photographs. For explicit repairs it archives the damaged metadata before writing a valid replacement; image bytes are preserved. Valid existing image/metadata pairs continue to work and are not renamed on startup. Reimporting identical bytes for the same part/view/label reuses the existing sample without rewriting its session provenance; conflicting labels or physical IDs remain an integrity error.

New metadata is published atomically and flushed to disk, with a separate verified recovery copy. A missing or malformed sidecar can be restored automatically only when that copy matches the original image and its recorded identity. Older broken sidecars without such a copy require the explicit selected-field repair above. Training never silently skips unresolved images or invents physical-part IDs. Renaming photographs does not create independent housings: all pictures of the same housing must retain one part ID.

The same explicit repair is available from PowerShell:

```powershell
.\.venv\Scripts\python.exe -m src.cli repair ".\dataset\inner_face\GOOD\photo.jpg" --view inner_face --label GOOD --part-id housing17 --session-id placement1
```

## Commission the view and train

Training automatically screens the selected view's GOOD training captures. Images that fail focus/exposure, complete-housing detection, landmark registration, or the inspection ROI are excluded from that run. Reference selection prefers the most represented native resolution before sharpness; it must pass the same checks, and another suitable reference is tried when necessary. Training continues with the accepted captures, without changing quality limits or the saved physical-part split.

The **Automatic image screening** panel lists accepted/excluded counts, each excluded filename and reason, measured quality, and the saved JSON report path. Original images, labels and metadata are retained, so a run-specific exclusion does not permanently reject a dataset sample. Reports are saved under `inspection_results/training_screening/<product>/<view>/` (or the configured results directory), including failed attempts, and successful model bundles contain their own screening report and actual accepted sample IDs.

If no usable training captures remain, training stops with the exclusion counts and reasons; the previous model stays active. Validation and test captures retain their original assignments. Invalid validation captures are reported, and invalid test captures remain in evaluation as RECHECK observations. A failed validation/calibration stage still needs suitable independent validation captures; the software never promotes held-out images into training to replace excluded captures.

1. Connect the Makose camera and select its exact index in **CAMERA SETTINGS**. No fallback to another index occurs. Check **actual** decoded resolution, FOURCC and measured capture FPS. A requested 3840×2160 setting alone is insufficient. If the device negotiates a different resolution/format, live analysis returns RECHECK until commissioned correctly.
2. Fix the focus/exposure and lighting where supported. Unsupported driver settings show failed readback. Tune ROI sharpness, clipping and motion limits using actual acceptable captures. Reflective highlights, blur, incomplete framing and moving parts must not be accepted by weakening gates without validation.
3. Enter technician mode. On first use, create a local PIN (at least six characters); it is stored as a salted PBKDF2 hash under `runtime/`. Advanced settings are locked again after restart. This is an operator-change lock, not an operating-system access-control boundary.
4. In **ROI EDITOR**, use a real image for the selected view. Draw rectangular, circular or polygon inspection regions/exclusions around intentional holes, mounting interruptions and unrelated background as appropriate. Definitions use normalized image coordinates and are stored per product/view. Existing models reject changed ROI definitions until retrained. No exclusion is inferred from the reference photos.
5. In **MODEL SETTINGS**, select geometry, EfficientAD or hybrid. Set native tile size (512 default), overlap (25%), model input size (512 default) and memory limits. A 768-pixel tile resized to a 512-pixel model loses information; measure the smallest rejectable defect at the selected scale. Do not claim tiny-crack sensitivity from a resized full frame.
6. EfficientAD automatically downloads, verifies and extracts official Imagenette (~1.5 GB) when its natural-image dataset is missing. An existing genuine ImageNet/ImageNette folder can be configured through `imagenet_dir`. The dataset is reused for later training. The official pretrained PDN teacher is downloaded with its published SHA256 and cached for offline reuse. See [NEURAL.md](docs/NEURAL.md).
7. Click **TRAIN MODEL** for `inner_face`, then `ribbed_face`, using the same data snapshot/split settings. Geometry can be trained independently before neural assets are present. Training runs off the UI thread. Default EfficientAD training uses FP32, native Anomalib losses, training batch size one, 10,000 optimizer steps and fixed raw score units. AMP stays disabled pending actual GPU validation.

Registration uses internal ORB landmark consensus and an asymmetric hole constellation fallback. The circular silhouette establishes centre/scale only. Symmetric or insufficient landmarks return RECHECK. Background/mask failures are not turned into fabricated defects. The geometry profile learns repeatable rib tracks, intentional interruptions, hole positions and stable feature edges from GOOD training images. Mounting-feature checks require `geometry.check_mounting: true` and explicit normalized `geometry.mounting_regions` ROI definitions in Technical Settings; they remain disabled until those regions are commissioned. Polar-coordinate rib and rim deviations map back to source pixels. Geometry configuration exposes actual pixel tolerances and normalized rib annulus radii. Changing quality, registration, geometry, calibration, tiling, localization or ROI settings requires retraining before another inspection. Thresholds need real-data commissioning for narrow cracks, dents, bent/chipped ribs and tolerable cast textures.

Calibration uses independent GOOD validation-part maxima, an explicit false-reject target and an ambiguity/recheck band. Branch/view budgets conservatively divide the configured whole-part FPR target. A geometric measurement floor prevents vanishing GOOD variation from becoming a zero tolerance. Reports show empirical rates, confidence intervals and sample-size limitations. Targets such as <2% false rejection and ≥95% critical-defect recall are **acceptance goals, not guarantees**.

Training writes versioned, integrity-checked bundles, then atomically activates the completed bundle. A failed run preserves the prior valid model. Bundles contain the registration template, geometry profile, real neural checkpoint when trained, ROIs, preprocessing, raw-score thresholds, physical-part split manifest, packages, reports and model versions. Restart loads these artifacts without retraining. Legacy joblib artifacts from the previous prototype are not accepted as EfficientAD models.

## Inspect one physical part

1. Click **START INSPECTION** to connect the selected camera and start the native preview. Create a **New Part** with a unique physical ID.
2. Choose `inner_face` or `ribbed_face`, or use Auto when it identifies the face confidently. If Auto is uncertain, select the face manually; the first face is never assumed.
3. Place the housing stationary and click **Capture & Analyze**. The service selects the sharpest recent stable capture, checks quality/registration, and runs the saved model. Camera capture continues independently of inference; no old-frame queue accumulates. An incomplete/blurred/moving/stale capture cannot create view coverage or PASS.
4. Follow the coverage panel, flip the housing and capture the other mandatory face. Retake a RECHECK view after correcting the cause. PASS captures cannot be silently replaced; a measured FAIL locks the part until finalization/cancellation.
5. Click **Finalize Part** once. Missing or unresolved required views produce RECHECK. A calibrated localized rejection produces FAIL; otherwise all required views must pass. One part produces one counted event. A geometry-only PASS is explicitly scoped to geometry and does not assess scratches/porosity.
6. **INSPECTION HISTORY** shows the final event and evidence. Native original/marked lossless PNGs and JSON retain capture identity, model versions, reasons, measured candidates and coverage. Corrupt evidence is flagged and excluded from counters. Cancelled sessions are recorded but not counted. Start a new part to rearm.

Only local red contours appear on the operator image. Raw anomaly maps, registration and timing details are in technician diagnostics. A high image score without valid localization yields RECHECK; no whole-part warning contour is invented. Repeated video frames select image quality; they are not independent votes confirming a defect.

A single overhead camera plus manual flipping does not provide full 360° coverage. Optional side views become mandatory only when configured and trained. Camera providers and per-view models are separate so additional cameras can be integrated later; extra live feeds are not simulated. No programmable lighting or metrology sensor is represented as connected.

## Evaluate and compare

Use **EVALUATE MODEL** for a selected view or all mandatory views. It reloads the immutable test split saved with the model and invokes the same view inspection/quality/registration/localization/fusion path used for a captured still. Live temporal acquisition gates are separately tested and are not inferred from static test photographs. Complete-view evaluation validates that no physical test part appeared in any selected model's fitting/calibration set.

```powershell
.\.venv\Scripts\python.exe -m src.cli evaluate --view all --backend hybrid
.\.venv\Scripts\python.exe -m src.cli benchmark --view all
```

Benchmarking compares geometry-only, EfficientAD-only and hybrid using **the same EfficientAD bundle, samples, saved thresholds and preprocessing**, without retraining or automatic winner selection. Dinomaly and DINOv2 are visibly unavailable; no untested adapter is selectable as implemented.

JSON/CSV exports include independent GOOD/NG counts, TP/TN/FP/FN, decisive-result accuracy, specificity, precision/recall/F1, false-positive/negative rates, RECHECK counts, overall coverage and median/P95 latency. RECHECK cases remain in the report. AUROC requires both classes and a consistent score scale; pooled scores from different view models are not compared as though they share a scale. Repeated placement/session IDs produce repeatability summaries without being treated as new independent parts. Pixel localization metrics are unavailable until real defect masks are supplied. Review actual target failures; do not tune using the final test set.

## Software validation and outstanding acceptance

Automated tests cover dataset integrity/duplicates/grouping, camera negotiation and controls with mocks, frame quality/staleness/stability, landmarks and angular ambiguity, polar geometry/controlled missing ribs, holes/rim/exclusions, native coordinate mapping and tile merging, calibration semantics, model errors/checkpoints, session coverage/locking, evidence integrity, evaluation consistency, GUI workers and rendering. Synthetic fixtures are marked as software tests throughout.

A genuine EfficientAD CPU optimizer step, native 512×512 inference and checkpoint round-trip were executed with the checksum-verified teacher and a real natural-image penalty photograph. This establishes model execution, not sufficient training or housing accuracy. The cloud host has no physical camera or CUDA GPU. Actual Windows launch, negotiated Makose 4K/MJPEG, achieved FPS, supported controls, RTX 5070 training/inference/VRAM, defect localization, smallest rejectable feature and unseen-part accuracy remain to be measured on the workstation. See [the acceptance procedure](docs/ACCEPTANCE.md).

All current vision measurements use pixels, angles or normalized image geometry. Optional calibrated scaling is gated on a supplied, explicitly validated camera calibration. No 1 µm accuracy or invisible subsurface porosity detection is claimed. Precision metrology, programmable lighting, Dinomaly, DINOv2 and dedicated multi-camera operation remain future integrations.

The previous image-level/F2/voting implementations are retained only under `src/legacy/` for characterization tests. Public training/inspection modules expose the modern workflow; the HMI does not import legacy implementations.
