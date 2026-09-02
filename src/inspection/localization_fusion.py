from __future__ import annotations
import cv2
import numpy as np
from skimage.metrics import structural_similarity
from src.inspection.types import DefectRegion


class DefectLocalizationFusion:
    def __init__(self,min_area=90,kernel=5,merge_distance=12): self.min_area,self.kernel,self.merge_distance=min_area,kernel,merge_distance
    def localize(self,image:np.ndarray,golden:np.ndarray,anomaly_map:np.ndarray,threshold:float,geometry_regions:list[DefectRegion])->list[DefectRegion]:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY); ref=cv2.cvtColor(golden,cv2.COLOR_BGR2GRAY)
        _,ssim_map=structural_similarity(cv2.GaussianBlur(ref,(5,5),0),cv2.GaussianBlur(gray,(5,5),0),full=True)
        residual=np.clip(1-ssim_map,0,1); edge=(cv2.Canny(gray,70,160)!=cv2.Canny(ref,70,160)).astype(np.float32)
        amap=anomaly_map/(np.quantile(anomaly_map,.995)+1e-8); support=(residual>.20)|(cv2.GaussianBlur(edge,(5,5),0)>.2)
        mask=((amap>=threshold)&support).astype(np.uint8)*255; kernel=np.ones((self.kernel,self.kernel),np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel); mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); regions=[]
        for c in contours:
            area=cv2.contourArea(c); x,y,w,h=cv2.boundingRect(c)
            if area>=self.min_area and min(w,h)>=3: regions.append(DefectRegion("appearance",c,(x,y,w,h),float(np.clip(amap[y:y+h,x:x+w].max(),0,1)),"PATCHCORE+RESIDUAL"))
        return regions+geometry_regions
