from __future__ import annotations
from pathlib import Path
import csv, shutil, subprocess
import cv2, joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


def pair_features(reference: np.ndarray, image: np.ndarray) -> np.ndarray:
    a=cv2.resize(cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY),(64,64)).astype(np.float32)/255
    b=cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(64,64)).astype(np.float32)/255
    ea=cv2.Canny((a*255).astype(np.uint8),60,150); eb=cv2.Canny((b*255).astype(np.uint8),60,150)
    d=np.abs(cv2.GaussianBlur(a,(5,5),0)-cv2.GaussianBlur(b,(5,5),0))
    return np.r_[np.quantile(d,[.5,.75,.9,.95,.99]),d.mean(),d.std(),np.mean(ea!=eb),cv2.compareHist(cv2.calcHist([a],[0],None,[32],[0,1]),cv2.calcHist([b],[0],None,[32],[0,1]),cv2.HISTCMP_BHATTACHARYYA)]


class VisualChangeDetector:
    def __init__(self): self.model=LogisticRegression(class_weight={0:1,1:2},max_iter=1000,random_state=42)
    def fit(self, references:list[np.ndarray], good:list[np.ndarray], ng:list[np.ndarray])->None:
        x=[]; y=[]
        for label,images in ((0,good),(1,ng)):
            for i,image in enumerate(images):
                ref=references[i%len(references)]
                if label==0 and np.array_equal(ref,image): ref=references[(i+1)%len(references)]
                x.append(pair_features(ref,image)); y.append(label)
        if len(set(y))<2: raise ValueError("Visual comparison training requires both GOOD and NG training images")
        self.model.fit(x,y)
    def predict(self,reference:np.ndarray,image:np.ndarray)->float: return float(self.model.predict_proba([pair_features(reference,image)])[0,1])
    def save(self,path:Path)->None: path.parent.mkdir(parents=True,exist_ok=True); joblib.dump(self.model,path)
    def load(self,path:Path)->None: self.model=joblib.load(path)


class TaoVisualChangeNetBackend:
    """Isolated optional TAO classification adapter; never silently substitutes a run."""
    def __init__(self, command: str="tao"): self.command=command
    def available(self)->bool: return shutil.which(self.command) is not None
    def require(self)->None:
        if not self.available(): raise RuntimeError("NVIDIA TAO was requested but the 'tao' command was not found. Install TAO Toolkit, or set visual_change.backend to 'sklearn' in config.yaml.")
    def create_pair_manifest(self, rows:list[tuple[Path,Path,int]], destination:Path)->None:
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open("w",newline="",encoding="utf-8") as f:
            writer=csv.writer(f); writer.writerow(["reference","test","label"]); writer.writerows(rows)
    def run(self,spec:Path)->None:
        self.require(); subprocess.run([self.command,"model","visual_changenet","train","-e",str(spec)],check=True)
