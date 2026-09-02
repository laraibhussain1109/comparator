from __future__ import annotations
from pathlib import Path
import os
import cv2, joblib
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors


class PatchCoreDetector:
    """PatchCore-style normal patch memory using deterministic multi-scale appearance descriptors."""
    def __init__(self, patch_size=16, stride=8, max_memory_patches=12000, *, max_patches=None):
        # ``max_patches`` remains accepted for callers created by earlier
        # releases; the public configuration uses ``max_memory_patches``.
        self.patch_size,self.stride=patch_size,stride
        self.max_patches=max_memory_patches if max_patches is None else max_patches
        self.model=None;self.memory=None
    @staticmethod
    def _cpu_workers() -> int:
        """Use the OS logical-core count without joblib's platform probing.

        Recent Windows versions may not include ``wmic``.  Passing ``-1`` to
        scikit-learn makes joblib invoke that missing command while attempting
        to discover physical cores.  An explicit positive count has the same
        graceful logical-core fallback without emitting a misleading warning.
        """
        return max(1, os.cpu_count() or 1)

    def _build_index(self) -> None:
        self.model=NearestNeighbors(n_neighbors=1,n_jobs=self._cpu_workers()).fit(self.memory)
    def _features(self, image: np.ndarray) -> tuple[np.ndarray, list[tuple[int,int]]]:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)/255
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)/255
        gx=cv2.Sobel(gray,cv2.CV_32F,1,0); gy=cv2.Sobel(gray,cv2.CV_32F,0,1); planes=np.dstack((lab, np.hypot(gx,gy)))
        feats=[]; positions=[]; p=self.patch_size
        for y in range(0,image.shape[0]-p+1,self.stride):
            for x in range(0,image.shape[1]-p+1,self.stride):
                tile=planes[y:y+p,x:x+p]; feats.append(np.r_[tile.mean((0,1)),tile.std((0,1))]); positions.append((x,y))
        return np.asarray(feats,np.float32), positions
    def fit(self, images: list[np.ndarray]) -> None:
        all_features=np.concatenate([self._features(x)[0] for x in images]); rng=np.random.default_rng(42)
        if len(all_features)>self.max_patches: all_features=all_features[rng.choice(len(all_features),self.max_patches,replace=False)]
        self.memory=all_features; self._build_index()
    def predict(self,image:np.ndarray)->tuple[float,np.ndarray]:
        if self.model is None: raise RuntimeError("PatchCore model is not loaded")
        feats,pos=self._features(image); distances=self.model.kneighbors(feats,return_distance=True)[0][:,0]; score_map=np.zeros(image.shape[:2],np.float32); counts=np.zeros_like(score_map)
        for d,(x,y) in zip(distances,pos): score_map[y:y+self.patch_size,x:x+self.patch_size]+=d; counts[y:y+self.patch_size,x:x+self.patch_size]+=1
        score_map/=np.maximum(counts,1);score=float(np.quantile(distances,.99));logging.getLogger(__name__).debug("PatchCore image_score=%.6f max_anomaly_map=%.6f",score,float(score_map.max()));return score,score_map
    def save(self,path:Path)->None: path.parent.mkdir(parents=True,exist_ok=True); joblib.dump({"memory":self.memory,"patch_size":self.patch_size,"stride":self.stride,"max_patches":self.max_patches},path)
    def load(self,path:Path)->None:
        data=joblib.load(path); self.__dict__.update(data); self._build_index()
