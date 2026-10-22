"""Real Anomalib EfficientAD, with explicit assets and stable score semantics.

This adapter uses the *unmodified* Anomalib 2.2.0 EfficientAdModel and its native
student/teacher, ImageNet-penalty and autoencoder losses. The explicit training
loop avoids Lightning's implicit working-directory downloads and postprocessor
normalization. No image-dependent score normalization is ever applied.
"""
from __future__ import annotations

from importlib import metadata as packages
import json
import logging
import os
import pickle
from pathlib import Path
import time
from typing import Any, Callable

import cv2
import numpy as np

from src.models.weights import ModelSetupError, natural_image_paths, sha256_file, teacher_weights
from src.utils.filesystem import filesystem_path

log = logging.getLogger(__name__)
ANOMALIB_VERSION = "2.2.0"
ARTIFACT_VERSION = 1
SCORE_UNITS = "raw_mean_squared_feature_distance_v1"


def _dependencies():
    """Import only when neural operation is requested; reject untested versions."""
    try:
        import torch
        from anomalib.models.image.efficient_ad.torch_model import EfficientAdModel, EfficientAdModelSize
        if packages.version("anomalib") != ANOMALIB_VERSION:
            raise ModelSetupError(f"EfficientAD requires tested anomalib=={ANOMALIB_VERSION}; install requirements-neural.txt.")
        return torch, EfficientAdModel, EfficientAdModelSize
    except ModelSetupError:
        raise
    except (ImportError, OSError, RuntimeError, AttributeError) as error:
        raise ModelSetupError(
            "The optional EfficientAD environment is unavailable or incompatible. "
            "Run scripts/install-neural.ps1 on Windows, then retry. "
            f"Original error: {error}"
        ) from error


def _check_images(images: list[np.ndarray], label: str) -> None:
    if not images:
        raise ValueError(f"{label} must contain at least one BGR image.")
    for index, image in enumerate(images):
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(f"{label}[{index}] must be an HxWx3 uint8 BGR image.")
        if min(image.shape[:2]) < 16:
            raise ValueError(f"{label}[{index}] is too small to be a useful image.")


class EfficientADBackend:
    """Train and run actual EfficientAD on already registered BGR image tiles.

    Training batch size is one, as required by the selected Anomalib algorithm.
    ``inference_batch_size`` can be larger and is reduced on CUDA OOM. Native
    tiles are resized to ``input_size`` explicitly (default 512); the saved
    preprocessing metadata records the information scale. Returned maps always
    have each supplied tile's original HxW shape, without pretending interpolation
    creates detail the neural input did not contain.
    """

    def __init__(self, settings: dict | None = None):
        self.settings = dict(settings or {})
        self.settings.setdefault("model_size", "small")
        self.settings.setdefault("input_size", 512)
        self.settings.setdefault("max_steps", 10000)
        self.settings.setdefault("train_batch_size", 1)
        self.settings.setdefault("inference_batch_size", 2)
        self.settings.setdefault("learning_rate", 1e-4)
        self.settings.setdefault("weight_decay", 1e-5)
        self.settings.setdefault("seed", 42)
        self.settings.setdefault("device", "auto")
        self.model: Any = None
        self.metadata: dict = {}
        self.device: Any = None
        self._torch: Any = None
        self._trained = False
        self._validate_settings()

    @property
    def trained(self) -> bool:
        return self._trained

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version", "untrained"))

    def _validate_settings(self) -> None:
        size = self.settings["input_size"]
        if not isinstance(size, int) or size < 256 or size % 64:
            raise ValueError("EfficientAD input_size must be an integer multiple of 64 and at least 256.")
        if self.settings["model_size"] not in {"small", "medium"}:
            raise ValueError("EfficientAD model_size must be 'small' or 'medium'.")
        if int(self.settings["train_batch_size"]) != 1:
            raise ValueError("Anomalib EfficientAD requires train_batch_size=1; configure inference_batch_size separately.")
        if int(self.settings["max_steps"]) < 1 or int(self.settings["inference_batch_size"]) < 1:
            raise ValueError("max_steps and inference_batch_size must be positive integers.")
        if self.settings.get("mixed_precision", False):
            raise ModelSetupError(
                "Mixed precision has not been validated for this EfficientAD adapter on the target GPU. "
                "Use mixed_precision=false (FP32) until a representative CUDA validation is completed."
            )

    def _build(self) -> None:
        torch, model_cls, size_cls = _dependencies()
        self._torch = torch
        requested = str(self.settings["device"])
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise ModelSetupError("CUDA was requested, but PyTorch reports no available CUDA device.")
        if self.settings.get("require_cuda") and not requested.startswith("cuda"):
            raise ModelSetupError("Training requires CUDA in this configuration; this machine has no usable CUDA GPU.")
        try:
            self.device = torch.device(requested)
            # Execute an actual kernel; availability alone is not architecture support.
            probe = torch.ones((4, 4), device=self.device)
            _ = float((probe @ probe).sum().item())
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.model = model_cls(teacher_out_channels=384,
                                   model_size=size_cls(self.settings["model_size"]),
                                   padding=False, pad_maps=True).to(self.device)
        except (RuntimeError, ValueError) as error:
            raise ModelSetupError(
                f"Cannot execute EfficientAD on {requested}: {error}. RTX 5070 needs a "
                "Blackwell-compatible PyTorch build, such as the pinned CUDA 12.8 wheels."
            ) from error
        for parameter in self.model.teacher.parameters():
            parameter.requires_grad_(False)
        for group in (self.model.mean_std, self.model.quantiles):
            for parameter in group.parameters():
                parameter.requires_grad_(False)

    def _tensor(self, images: list[np.ndarray]):
        size = self.settings["input_size"]
        arrays = [cv2.cvtColor(cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                  for image in images]
        return self._torch.from_numpy(np.ascontiguousarray(np.stack(arrays).transpose(0, 3, 1, 2))).to(
            self.device, dtype=self._torch.float32) / 255.0

    def _teacher_statistics(self, images: list[np.ndarray]) -> None:
        torch = self._torch
        total = torch.zeros(384, dtype=torch.float64, device=self.device)
        squares = torch.zeros_like(total)
        count = 0
        self.model.eval()
        with torch.inference_mode():
            for image in images:
                output = self.model.teacher(self._tensor([image])).double()
                total += output.sum((0, 2, 3))
                squares += output.square().sum((0, 2, 3))
                count += output.shape[0] * output.shape[2] * output.shape[3]
        mean = total / count
        variance = (squares / count - mean.square()).clamp_min(1e-10)
        with torch.no_grad():
            self.model.mean_std["mean"].copy_(mean.float()[None, :, None, None])
            self.model.mean_std["std"].copy_(variance.sqrt().float()[None, :, None, None])

    def fit(self, good_images: list[np.ndarray], output_dir: Path,
            validation_good: list[np.ndarray] | None = None,
            progress: Callable[[int, str], None] | None = None) -> dict:
        """Fit GOOD-only network parameters and save a self-contained checkpoint.

        Caller is responsible for physical-part separation and final pipeline
        calibration. Optional GOOD validation images are checked but never used
        as training samples; all maps retain the raw fixed units used at runtime.
        """
        _check_images(good_images, "good_images")
        if validation_good is not None:
            _check_images(validation_good, "validation_good")
        emit = progress or (lambda *_: None)
        self._trained = False
        self.metadata = {}
        self._validate_settings()
        emit(1, "Checking real pretrained teacher and natural-image penalty dataset")
        natural_paths = natural_image_paths(self.settings)
        teacher_path, provenance = teacher_weights(self.settings)
        torch, _, _ = _dependencies()
        torch.manual_seed(int(self.settings["seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.settings["seed"]))
        # Seed before construction so student/autoencoder initialization is repeatable.
        self._build()
        try:
            with open(filesystem_path(teacher_path), "rb") as teacher_file:
                state = torch.load(teacher_file, map_location="cpu", weights_only=True)
            self.model.teacher.load_state_dict(state, strict=True)
        except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError) as error:
            raise ModelSetupError(f"Verified teacher checkpoint is incompatible with selected EfficientAD model: {error}") from error
        emit(5, "Computing pretrained teacher feature statistics on GOOD training tiles")
        self._teacher_statistics(good_images)
        from PIL import Image
        from torchvision import transforms
        natural_transform = transforms.Compose([
            transforms.Resize((self.settings["input_size"] * 2, self.settings["input_size"] * 2)),
            transforms.RandomGrayscale(p=0.3),
            transforms.CenterCrop((self.settings["input_size"], self.settings["input_size"])),
            transforms.ToTensor(),
        ])
        optimizer = torch.optim.Adam(
            list(self.model.student.parameters()) + list(self.model.ae.parameters()),
            lr=float(self.settings["learning_rate"]), weight_decay=float(self.settings["weight_decay"]))
        steps = int(self.settings["max_steps"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(0.95 * steps)), gamma=0.1)
        random = np.random.default_rng(int(self.settings["seed"]))
        order = random.permutation(len(good_images))
        started = time.perf_counter()
        history: list[dict] = []
        self.model.train()
        self.model.teacher.eval()
        try:
            for step in range(steps):
                if step and step % len(good_images) == 0:
                    order = random.permutation(len(good_images))
                image = good_images[int(order[step % len(good_images)])]
                path = natural_paths[int(random.integers(len(natural_paths)))]
                with open(filesystem_path(path), "rb") as natural_file, Image.open(natural_file) as natural:
                    penalty = natural_transform(natural.convert("RGB")).unsqueeze(0).to(self.device)
                optimizer.zero_grad(set_to_none=True)
                losses = self.model(self._tensor([image]), batch_imagenet=penalty)
                loss = sum(losses)
                if not bool(torch.isfinite(loss).item()):
                    raise ModelSetupError("EfficientAD training produced a non-finite loss; no successful model was saved.")
                loss.backward()
                optimizer.step()
                scheduler.step()
                if step == 0 or step == steps - 1 or (step + 1) % max(1, steps // 100) == 0:
                    loss_values = [float(value.detach().cpu()) for value in losses]
                    history.append({"step": step + 1, "student": loss_values[0], "autoencoder": loss_values[1],
                                    "student_autoencoder": loss_values[2], "total": sum(loss_values)})
                    emit(6 + int(89 * (step + 1) / steps),
                         f"EfficientAD optimizer step {step + 1}/{steps}; measured loss {sum(loss_values):.5g}")
        except torch.cuda.OutOfMemoryError as error:
            self.model = None
            torch.cuda.empty_cache()
            raise ModelSetupError(
                "EfficientAD ran out of GPU memory with training batch size one. "
                "Close other GPU applications or explicitly reduce input_size, then retrain and revalidate the smallest defects."
            ) from error
        self.model.eval()
        self._trained = True
        self.metadata = {
            "artifact_version": ARTIFACT_VERSION, "backend": "efficientad", "implementation": "anomalib.EfficientAdModel",
            "anomalib_version": ANOMALIB_VERSION, "torch_version": str(torch.__version__),
            "model_size": self.settings["model_size"], "input_size": self.settings["input_size"],
            "score_units": SCORE_UNITS, "per_image_normalization": False,
            "teacher": provenance, "training_good_tiles": len(good_images),
            "validation_good_tiles": len(validation_good or []), "natural_penalty_images": len(natural_paths),
            "optimizer_steps": steps, "training_device": str(self.device), "precision": "float32",
            "training_seconds": time.perf_counter() - started, "loss_history": history,
            "seed": int(self.settings["seed"]),
            "preprocessing": {"input": "uint8 BGR", "conversion": "RGB / 255 then native model ImageNet normalization",
                              "resize": "OpenCV INTER_AREA square tile resize", "input_size": self.settings["input_size"],
                              "observed_training_tile_shapes": sorted({tuple(image.shape[:2]) for image in good_images}),
                              "detail_limit": "Returned map interpolation does not restore detail lost by tile resizing."},
            "validation_status": "Software model trained; industrial accuracy requires held-out physical-part pipeline evaluation.",
        }
        emit(96, "Saving trained EfficientAD state and artifact integrity metadata")
        self.save(Path(output_dir))
        emit(100, f"EfficientAD checkpoint saved after {steps} actual optimizer steps on {self.device}")
        return dict(self.metadata)

    def save(self, directory: Path) -> None:
        if not self.trained or self.model is None:
            raise ModelSetupError("No trained EfficientAD model is available to save.")
        directory = Path(directory)
        Path(filesystem_path(directory)).mkdir(parents=True, exist_ok=True)
        checkpoint = directory / "efficientad.pt"
        temporary = directory / ".ea.pt.tmp"
        # Tensor-only state; loading never deserializes model/class pickles.
        state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
        # Python opens the extended Windows path; PyTorch only sees a stream,
        # avoiding its native filename writer and legacy MAX_PATH handling.
        with open(filesystem_path(temporary), "wb") as checkpoint_file:
            self._torch.save(state, checkpoint_file)
        os.replace(filesystem_path(temporary), filesystem_path(checkpoint))
        self.metadata["checkpoint_sha256"] = sha256_file(checkpoint)
        self.metadata["model_version"] = f"efficientad-anomalib{ANOMALIB_VERSION}-{self.metadata['checkpoint_sha256'][:12]}"
        manifest = directory / "model.json"
        pending = directory / ".ea.json.tmp"
        with open(filesystem_path(pending), "w", encoding="utf-8") as manifest_file:
            json.dump(self.metadata, manifest_file, indent=2)
        os.replace(filesystem_path(pending), filesystem_path(manifest))

    def load(self, directory: Path) -> "EfficientADBackend":
        directory = Path(directory)
        manifest = directory / "model.json"
        checkpoint = directory / "efficientad.pt"
        self._trained = False
        if not Path(filesystem_path(manifest)).is_file() or not Path(filesystem_path(checkpoint)).is_file():
            raise ModelSetupError(f"Trained EfficientAD artifact is incomplete at {directory}; train the view model first.")
        try:
            with open(filesystem_path(manifest), "r", encoding="utf-8") as manifest_file:
                info = json.load(manifest_file)
        except (OSError, ValueError) as error:
            raise ModelSetupError(f"Invalid EfficientAD model manifest: {error}") from error
        if info.get("artifact_version") != ARTIFACT_VERSION or info.get("anomalib_version") != ANOMALIB_VERSION:
            raise ModelSetupError("EfficientAD artifact version differs from the supported adapter; retrain using the pinned environment.")
        if info.get("score_units") != SCORE_UNITS or info.get("per_image_normalization") is not False:
            raise ModelSetupError("EfficientAD checkpoint has unsupported score units; calibration cannot be reused.")
        if not info.get("teacher", {}).get("sha256") or not info.get("optimizer_steps", 0):
            raise ModelSetupError("EfficientAD checkpoint is missing real pretrained/training provenance.")
        if sha256_file(checkpoint) != info.get("checkpoint_sha256"):
            raise ModelSetupError("EfficientAD checkpoint SHA256 integrity check failed; artifact will not be loaded.")
        self.settings.update({"input_size": info["input_size"], "model_size": info["model_size"]})
        self._validate_settings()
        self._build()
        try:
            with open(filesystem_path(checkpoint), "rb") as checkpoint_file:
                state = self._torch.load(checkpoint_file, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True)
        except (OSError, RuntimeError, ValueError, TypeError, pickle.UnpicklingError) as error:
            raise ModelSetupError(f"Cannot load trained EfficientAD tensor checkpoint: {error}") from error
        self.model.eval()
        self.metadata = info
        self._trained = True
        return self

    def predict(self, images: list[np.ndarray]) -> list[np.ndarray]:
        if not self.trained or self.model is None:
            raise ModelSetupError("EfficientAD is not trained/loaded; train or load the selected view model first.")
        if not images:
            return []
        _check_images(images, "images")
        torch = self._torch
        outputs: list[np.ndarray] = []
        batch_size = int(self.settings["inference_batch_size"])
        cursor = 0
        self.model.eval()
        with torch.inference_mode():
            while cursor < len(images):
                selected = images[cursor:cursor + batch_size]
                try:
                    # normalize=False is deliberate: training calibration and live
                    # inference both consume this identical fixed raw scale.
                    prediction = self.model(self._tensor(selected), normalize=False)
                    maps = prediction.anomaly_map[:, 0].float().cpu().numpy()
                    for image, anomaly_map in zip(selected, maps):
                        restored = cv2.resize(anomaly_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
                        if not np.isfinite(restored).all():
                            raise ModelSetupError("EfficientAD generated a non-finite anomaly map; inspection must RECHECK.")
                        outputs.append(restored.astype(np.float32, copy=False))
                    cursor += len(selected)
                except torch.cuda.OutOfMemoryError as error:
                    torch.cuda.empty_cache()
                    if batch_size == 1:
                        raise ModelSetupError("Insufficient GPU memory for one EfficientAD tile; inspection must RECHECK.") from error
                    batch_size = max(1, batch_size // 2)
                    log.warning("Reducing EfficientAD inference batch to %d after CUDA out-of-memory", batch_size)
        return outputs
