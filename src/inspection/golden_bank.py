from __future__ import annotations
from pathlib import Path
import cv2, json
import numpy as np
from src.utils.image_utils import read_image


def embedding(image: np.ndarray) -> np.ndarray:
    small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)/255
    return np.concatenate([small.ravel(), cv2.calcHist([small.astype(np.float32)], [0], None, [16], [0, 1]).ravel()])


class GoldenBank:
    def __init__(self): self.images: list[np.ndarray] = []; self.names: list[str] = []; self.features: np.ndarray | None = None
    def build(self, registered: list[np.ndarray], source_names: list[str], directory: Path, maximum: int = 8) -> None:
        directory.mkdir(parents=True, exist_ok=True); feats = np.stack([embedding(x) for x in registered]); count = min(maximum, len(registered))
        if count == len(registered): indices = list(range(count))
        else:
            # Deterministic farthest-first sampling gives a diverse reference
            # bank without starting sklearn/joblib's Windows core probe.
            center=feats.mean(axis=0)
            indices=[int(np.argmin(np.linalg.norm(feats-center,axis=1)))]
            nearest=np.linalg.norm(feats-feats[indices[0]],axis=1)
            while len(indices)<count:
                nearest[indices]=-np.inf
                candidate=int(np.argmax(nearest))
                indices.append(candidate)
                nearest=np.minimum(nearest,np.linalg.norm(feats-feats[candidate],axis=1))
        self.images = [registered[i] for i in indices]; self.names = [source_names[i] for i in indices]; self.features = np.stack([embedding(x) for x in self.images])
        for old in directory.glob("golden_*.png"): old.unlink()
        for n, image in enumerate(self.images): cv2.imwrite(str(directory/f"golden_{n:02d}.png"), image)
        (directory/"manifest.json").write_text(json.dumps({"sources": self.names}, indent=2))
    def load(self, directory: Path) -> None:
        paths = sorted(directory.glob("golden_*.png")); self.images = [read_image(p) for p in paths]; self.names = [p.name for p in paths]
        if not self.images: raise FileNotFoundError("Golden reference bank is missing")
        self.features = np.stack([embedding(x) for x in self.images])
    def select(self, image: np.ndarray) -> np.ndarray:
        assert self.features is not None
        return self.images[int(np.argmin(np.linalg.norm(self.features-embedding(image), axis=1)))]
