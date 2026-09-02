from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unreadable image: {path}")
    return image


def normalize_illumination(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    background = cv2.GaussianBlur(l, (0, 0), 19)
    corrected = cv2.addWeighted(l, 1.0, background, -0.55, 96)
    return cv2.cvtColor(cv2.merge((corrected, a, b)), cv2.COLOR_LAB2BGR)


def foreground_mask(image: np.ndarray, min_ratio: float = 0.03) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    masks = []
    for mode in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, mask = cv2.threshold(blur, 0, 255, mode | cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        masks.append(mask)
    h, w = gray.shape
    candidates = []
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if min_ratio * h * w < cv2.contourArea(c) < .95 * h * w]
        if valid:
            c = max(valid, key=cv2.contourArea)
            score = cv2.contourArea(c) - 2 * abs(cv2.pointPolygonTest(c, (w / 2, h / 2), True))
            candidates.append((score, c))
    output = np.zeros_like(gray)
    if candidates:
        cv2.drawContours(output, [max(candidates, key=lambda x: x[0])[1]], -1, 255, -1)
    return output
