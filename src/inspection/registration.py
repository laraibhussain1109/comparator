from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from src.utils.image_utils import foreground_mask, read_image, normalize_illumination


class PartRegistrar:
    def __init__(self, image_size: int, min_area_ratio: float = .06, ecc_enabled: bool = True):
        self.size, self.min_area_ratio, self.ecc_enabled = image_size, min_area_ratio, ecc_enabled
        self.template: np.ndarray | None = None

    def normalize_pose(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        mask = foreground_mask(image, self.min_area_ratio)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return cv2.resize(image, (self.size, self.size)), 0.0
        contour = max(contours, key=cv2.contourArea)
        area_ratio = cv2.contourArea(contour) / mask.size
        rect = cv2.minAreaRect(contour)
        (cx, cy), (rw, rh), angle = rect
        if rw < rh: angle += 90
        matrix = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rotated = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        rmask = cv2.warpAffine(mask, matrix, (image.shape[1], image.shape[0]))
        points = cv2.findNonZero(rmask)
        if points is None: return cv2.resize(image, (self.size, self.size)), 0.0
        x, y, w, h = cv2.boundingRect(points)
        pad = int(.08 * max(w, h)); x, y = max(0, x-pad), max(0, y-pad)
        crop = rotated[y:min(rotated.shape[0], y+h+2*pad), x:min(rotated.shape[1], x+w+2*pad)]
        side = max(crop.shape[:2]); canvas = np.full((side, side, 3), int(np.median(image)), np.uint8)
        oy, ox = (side-crop.shape[0])//2, (side-crop.shape[1])//2; canvas[oy:oy+crop.shape[0], ox:ox+crop.shape[1]] = crop
        registered = cv2.resize(canvas, (self.size, self.size))
        confidence = float(np.clip((area_ratio-self.min_area_ratio)/(0.45-self.min_area_ratio), 0, 1))
        return registered, confidence

    def register(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        registered, confidence = self.normalize_pose(image)
        if self.template is None or not self.ecc_enabled or confidence <= 0: return registered, confidence
        try:
            template = cv2.cvtColor(normalize_illumination(self.template), cv2.COLOR_BGR2GRAY).astype(np.float32)/255
            moving = cv2.cvtColor(normalize_illumination(registered), cv2.COLOR_BGR2GRAY).astype(np.float32)/255
            warp = np.eye(2, 3, dtype=np.float32)
            cc, warp = cv2.findTransformECC(template, moving, warp, cv2.MOTION_EUCLIDEAN, (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 40, 1e-4))
            registered = cv2.warpAffine(registered, warp, (self.size, self.size), flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
            confidence *= float(np.clip((cc + 1) / 2, 0, 1))
        except cv2.error:
            confidence *= .75
        return registered, confidence

    def build_template(self, paths: list[Path], output: Path) -> np.ndarray:
        normalized = [self.normalize_pose(read_image(p))[0] for p in paths]
        self.template = np.median(np.stack(normalized), axis=0).astype(np.uint8)
        output.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(output), self.template)
        return self.template

    def load(self, path: Path) -> None:
        self.template = read_image(path)
