from __future__ import annotations
from pathlib import Path
import cv2, joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors


class PatchCoreDetector:
    """PatchCore-style normal patch memory using deterministic multi-scale appearance descriptors."""
    def __init__(self, patch_size=16, stride=8, max_patches=12000): self.patch_size, self.stride, self.max_patches = patch_size, stride, max_patches; self.model=None; self.memory=None
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
        self.memory=all_features; self.model=NearestNeighbors(n_neighbors=1,n_jobs=-1).fit(all_features)
    def predict(self,image:np.ndarray)->tuple[float,np.ndarray]:
        if self.model is None: raise RuntimeError("PatchCore model is not loaded")
        feats,pos=self._features(image); distances=self.model.kneighbors(feats,return_distance=True)[0][:,0]; score_map=np.zeros(image.shape[:2],np.float32); counts=np.zeros_like(score_map)
        for d,(x,y) in zip(distances,pos): score_map[y:y+self.patch_size,x:x+self.patch_size]+=d; counts[y:y+self.patch_size,x:x+self.patch_size]+=1
        score_map/=np.maximum(counts,1); return float(np.quantile(distances,.99)),score_map
    def save(self,path:Path)->None: path.parent.mkdir(parents=True,exist_ok=True); joblib.dump({"memory":self.memory,"patch_size":self.patch_size,"stride":self.stride,"max_patches":self.max_patches},path)
    def load(self,path:Path)->None:
        data=joblib.load(path); self.__dict__.update(data); self.model=NearestNeighbors(n_neighbors=1,n_jobs=-1).fit(self.memory)
