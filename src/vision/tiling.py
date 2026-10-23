"""Overlapping native-resolution tiles with explicit source-image coordinates."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Iterator

import cv2
import numpy as np


@dataclass(frozen=True)
class Tile:
    image: np.ndarray
    x: int
    y: int
    width: int
    height: int


def _origins(length: int, size: int, step: int) -> list[int]:
    if length <= size:
        return [0]
    origins = list(range(0, length - size + 1, step))
    if origins[-1] != length - size:
        origins.append(length - size)
    return origins


def tile_image(image: np.ndarray, tile_size: int = 512, overlap: float = .25,
               mask: np.ndarray | None = None) -> Iterator[Tile]:
    if image is None or image.ndim not in (2, 3) or not image.size:
        raise ValueError("A nonempty image is required")
    if tile_size < 8 or not 0 <= overlap < 1:
        raise ValueError("tile_size must be >=8 and overlap in [0,1)")
    h, w = image.shape[:2]
    if mask is not None and mask.shape[:2] != (h, w):
        raise ValueError("Tile mask must have the source image dimensions")
    step = max(1, round(tile_size * (1 - overlap)))
    for y in _origins(h, tile_size, step):
        for x in _origins(w, tile_size, step):
            block = image[y:y + tile_size, x:x + tile_size]
            bh, bw = block.shape[:2]
            if mask is not None and not np.any(mask[y:y + bh, x:x + bw]):
                continue
            yield Tile(block, x, y, bw, bh)


def merge_tile_maps(tiles: Iterable[Tile], maps: Iterable[np.ndarray], shape: tuple) -> np.ndarray:
    """Blend tile maps at their recorded coordinates; uncovered pixels are zero.

    Network output maps may be lower resolution than a tile. Only each map is
    resized back to that tile's actual source footprint, including short edges.
    Raised-cosine weights reduce seams without suppressing image-border values.
    """
    tiles, maps = list(tiles), list(maps)
    if len(tiles) != len(maps):
        raise ValueError("Each tile must have exactly one map")
    h, w = shape[:2]
    total = np.zeros((h, w), np.float64)
    weights = np.zeros((h, w), np.float64)
    for tile, anomaly in zip(tiles, maps):
        if tile.x < 0 or tile.y < 0 or tile.x + tile.width > w or tile.y + tile.height > h:
            raise ValueError("Tile lies outside the destination image")
        values = np.asarray(anomaly, dtype=np.float32).squeeze()
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError("Tile maps must be finite two-dimensional arrays")
        if values.shape != (tile.height, tile.width):
            values = cv2.resize(values, (tile.width, tile.height), interpolation=cv2.INTER_LINEAR)
        wy = np.maximum(.05, np.hanning(tile.height)) if tile.height > 2 else np.ones(tile.height)
        wx = np.maximum(.05, np.hanning(tile.width)) if tile.width > 2 else np.ones(tile.width)
        weight = wy[:, None] * wx[None, :]
        region = np.s_[tile.y:tile.y + tile.height, tile.x:tile.x + tile.width]
        total[region] += values * weight
        weights[region] += weight
    return np.divide(total, weights, out=np.zeros_like(total), where=weights > 0).astype(np.float32)
