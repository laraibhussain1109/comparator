# EfficientAD installation and verification

This release integrates the actual Anomalib 2.2.0 `EfficientAdModel`. It trains
the student and autoencoder with Anomalib's native losses, a frozen pretrained
Patch Description Network (PDN) teacher, and the required natural-image penalty.
Dinomaly and DINOv2 comparison are disabled because their adapters have not been
implemented and validated.

## Windows 11 / RTX 5070 installation

Use **Python 3.12** for the pinned environment. Python 3.12.14 was demonstrated
on the Linux cloud host. The pinned SciPy version requires Python 3.12 or newer;
Python 3.11 is not supported by this dependency set.

In PowerShell, from the repository directory:

```powershell
py -3.12 -m venv .venv
.\scripts\install-neural.ps1
.\.venv\Scripts\python.exe app.py
```

The installer uses the official CUDA 12.8 PyTorch wheel index and exact builds
`torch==2.7.1+cu128` and `torchvision==0.22.1+cu128`. Exact build suffixes matter:
an already installed CPU wheel otherwise satisfies a requirement such as
`torch==2.7.1`. The remaining dependencies are pinned in
`requirements-neural.txt` and `requirements.txt`.

A compatible NVIDIA driver is required; the PyTorch wheels include their CUDA
runtime and do not need a separately installed CUDA toolkit. The installer
executes a real matrix multiplication on CUDA if a CUDA device is available and
reports the actual compute device. Package installation alone does not establish
that the RTX 5070 works. Review the application's GPU diagnostics on the target
machine, then run representative training and inference.

For a deliberate CPU installation:

```powershell
.\scripts\install-neural.ps1 -CpuOnly
```

The cloud host had no CUDA GPU. Windows execution, RTX 5070 inference, GPU memory
usage, GPU latency and mixed precision have **not** been validated here. The
adapter uses FP32 and rejects mixed precision until it is validated on real
hardware. Official Windows Python 3.12 wheel metadata was retrieved over HTTPS
for both CUDA wheels; torchvision explicitly requires the matching
`torch==2.7.1+cu128`. This verifies availability and declared compatibility,
not Windows/GPU execution.

## Required teacher assets and natural photographs

The pretrained teacher is downloaded automatically when training if
`allow_download` is enabled. The default cache is
`artifacts/pretrained/efficientad`. To pre-cache it before going offline:

```powershell
.\.venv\Scripts\python.exe -c "from src.models.weights import teacher_weights; teacher_weights({'teacher_cache':'artifacts/pretrained/efficientad','allow_download':True})"
```

The following SHA256 values are published in the selected [Anomalib 2.2.0
EfficientAD source](https://github.com/open-edge-platform/anomalib/blob/v2.2.0/src/anomalib/models/image/efficient_ad/lightning_model.py):

| Asset | Published SHA256 |
| --- | --- |
| `efficientad_pretrained_weights.zip` | `c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a` |
| `imagenette2.tgz` | `6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd` |

Official URLs:

- Teacher: <https://github.com/open-edge-platform/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip>
- ImageNette: <https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz>

HTTPS downloads use `truststore==0.10.5` with a per-request TLS client context.
On Windows this uses native CryptoAPI certificate-chain construction, including
retrieval of missing intermediate certificates. Certificate, hostname and date
checks remain enabled. It does not replace the machine's trusted roots or
globally change Python's SSL configuration. Install the updated requirements
when updating an existing environment; pulling source code alone does not
install new dependencies. To install this dependency directly in the app's
Windows environment:

```powershell
.\.venv\Scripts\python.exe -m pip install truststore==0.10.5
```

Published SHA256 verification also remains enabled. An invalid
archive is never loaded. Downloads use a temporary file and are finalized only
after verification. Only the selected teacher file is extracted from the ZIP.
The cache records its digest and publisher archive provenance for offline reuse.
Network settings may need to allow GitHub's release-asset redirect as well as
`github.com`; ImageNette uses `s3.amazonaws.com`.

EfficientAD also needs genuine natural photographs for its ImageNet penalty.
In **Train Model**, missing natural-image data triggers automatic download,
verification and extraction of official Imagenette (about 1.5 GB) when training
starts. This works in operator mode, without unlocking Technician Access or
enabling a checkbox. An existing genuine ImageNet/ImageNette training folder can
also be configured. Images are found recursively. The default directory is
`artifacts/pretrained/imagenette2/train`. Aluminium GOOD images and held-out
inspection test images must not be substituted for this dataset.

Click **Train & Save Versioned Model** to start training and prepare any missing
natural-image data automatically. Existing saved `download_imagenette: false`
settings no longer block this download. Extracted photographs are saved in the
configured ImageNet / Imagenette directory and reused for later training.
The training log shows bytes received, automatic connection retries, checksum
verification and extraction. An interrupted HTTPS transfer is retried up to five
times; retries resume the received bytes when the server supports HTTP Range.
Certificate verification remains enabled and the archive is used only after its
published SHA256 matches.

`SSL: DECRYPTION_FAILED_OR_BAD_RECORD_MAC` means the encrypted connection failed;
it does not mean the official URL is missing. If retries still fail, download
<https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz> in a browser and save
it as `artifacts/pretrained/imagenette2.tgz` inside the project folder (with the
default teacher cache), then start training again. The app verifies the cached
archive and extracts it automatically.
You do not need to create JSON files or unpack the archive manually. An app
restart or a new training attempt starts a new transfer if no verified archive
was saved; resuming applies to retries within the same attempt.

`CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate` identifies
an untrusted or incomplete certificate chain, even when the system clock is
correct. The error now identifies the asset, URL and cache path instead of
giving generic clock advice. If native validation still rejects a certificate,
the browser-download cache procedure above preserves verification and avoids
substituting untrained weights.

Asset download percentages are separate from training progress. The log labels
Imagenette and EfficientAD-teacher downloads as **Preparing training assets**;
optimizer training starts only after both assets are ready. A teacher download
may fail after Imagenette has reached 99–100%; completed cached assets are kept.
Only **Training completed** after fitting, calibration, evaluation and model
publication means a usable model was saved. A download failure leaves the view
without a new model and cannot produce a defect verdict.

A custom teacher requires both `teacher_path` and its matching `teacher_sha256`.
That digest establishes local integrity; the adapter labels its provenance as
user supplied rather than claiming it matches a published official checkpoint.
Missing teachers or natural photographs stop training with an actionable error.
An untrained teacher is never used as a fallback.

## Training, inference and calibration

Import the real housing dataset with physical part IDs and configured face
labels, then use **Train Model** independently for `inner_face` and `ribbed_face`.
The service registers GOOD training images, extracts native tiles and fits the
network. Independent physical parts calibrate the complete inference pipeline;
untouched test parts remain separate. A single photo of each face is not enough
to establish repeatability, reject thresholds or defect sensitivity.

Key settings under `models.efficientad`:

```yaml
max_steps: 10000
train_batch_size: 1
inference_batch_size: 2
input_size: 512
model_size: small
learning_rate: 0.0001
weight_decay: 0.00001
seed: 42
device: auto
mixed_precision: false
teacher_cache: artifacts/pretrained/efficientad
allow_download: true
imagenet_dir: artifacts/pretrained/imagenette2/train
```

Anomalib EfficientAD requires training batch size **one**. Inference batching is
independently configurable and shrinks on CUDA out-of-memory errors. An OOM at
one tile produces an error; the adapter does not silently shrink the neural
input or pretend the inspection succeeded. The default optimizer step count is
a starting configuration, not evidence of adequate training.

The backend consumes `uint8` BGR tiles. Preprocessing is identical during
training and inference: OpenCV `INTER_AREA` resize to `input_size`, BGR to RGB,
conversion to `[0,1]`, and the model's own ImageNet normalization. Teacher feature
mean and standard deviation use GOOD training tiles only. With 512-pixel tiles
and a 512-pixel neural input, the tile is not downsampled. A 768-pixel tile with a
512-pixel input loses spatial information by a factor of 1.5; the saved metadata
records the neural input size and training tile shapes. Interpolating an anomaly
map back to native coordinates does not restore that information. Validate the
smallest rejectable defects using the chosen tile and input size.

The backend returns raw maps in fixed mean squared feature-distance units:
`0.5 * student_teacher_map + 0.5 * student_autoencoder_map`. It always uses
`normalize=False`. There is no per-image min/max normalization. The service uses
the same tile merge, masks, score aggregation, calibration and final decision
path for validation, blind evaluation and live inspection. The optional adapter
argument `validation_good` checks inputs but never fits them; the service
calibrates externally using full-view inference. Consequently an adapter field
`validation_good_tiles: 0` does not mean the service omitted independent
calibration.

The saved `neural/efficientad.pt` contains tensor state only; `model.json` records
implementation versions, teacher provenance, actual optimizer steps, actual
training device, preprocessing and checkpoint SHA256. Reload checks this digest
before loading with `torch.load(..., weights_only=True)`. The model can reload
offline without teacher or natural-image downloads. The recorded training device
is historical; `backend.device` describes the current inference device. Recheck
calibration and accuracy after changing hardware, software or preprocessing.

## Executed software checks

On the CPU-only Linux cloud host, using Python 3.12.14:

- `torch==2.7.1+cpu`, `torchvision==0.22.1+cpu`, `anomalib==2.2.0` imported and ran.
- `pip check` reported no broken requirements in the isolated neural environment.
- `tests/test_models.py`: **8 passed**, including a real native-loss optimizer
  step with the checksum-verified official teacher, tensor checkpoint saving,
  reloading, and matching raw maps after restart.
- A separate 512×512 fixture executed one optimizer step, saved a checkpoint and
  returned a finite 512×512 anomaly map. Measured training time for that one
  software step was 13.59 seconds on this host. This is not a camera throughput
  or production-training benchmark.

The network smoke tests use explicitly synthetic normal images and one genuine
natural photograph for the penalty. They test executable model behavior only;
they do not measure housing accuracy, recall, false-rejection rates or defect
localization performance. No synthetic model was selected as an active housing
inspection model.

Normal model tests run without installing PyTorch; the real neural smoke test
is opt-in. After installing the neural environment and caching the official
teacher, provide a folder of genuine natural photographs and run:

```powershell
$env:NEUROIRIS_REAL_EFFICIENTAD_SMOKE = "1"
$env:NEUROIRIS_NATURAL_IMAGE_DIR = "C:\datasets\imagenette2\train"
$env:NEUROIRIS_TEACHER_CACHE = "artifacts/pretrained/efficientad"
.\.venv\Scripts\python.exe -m pytest -q tests/test_models.py
```

Run real acceptance testing on held-out physical GOOD and defective housings
with the Makose camera, controlled lighting and repeated placements. Report
actual RECHECK rates and defect-level results before making accuracy claims.
