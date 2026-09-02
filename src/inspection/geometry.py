from __future__ import annotations
from pathlib import Path
import json, cv2
import numpy as np
from src.inspection.types import DefectRegion
from src.utils.image_utils import foreground_mask


def extract_geometry(image:np.ndarray)->tuple[dict[str,float],np.ndarray|None]:
    mask=foreground_mask(image,.03); contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not contours:return {},None
    c=max(contours,key=cv2.contourArea); area=cv2.contourArea(c); perimeter=cv2.arcLength(c,True); x,y,w,h=cv2.boundingRect(c); (cx,cy),radius=cv2.minEnclosingCircle(c)
    holes,_=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE); internal=[q for q in holes if 20<cv2.contourArea(q)<.15*area]
    return {"area_ratio":area/mask.size,"aspect_ratio":w/max(h,1),"circularity":4*np.pi*area/max(perimeter**2,1),"extent":area/max(w*h,1),"hole_count":float(len(internal)),"center_x":cx/image.shape[1],"center_y":cy/image.shape[0],"radius_ratio":radius/max(image.shape[:2])},c


class GeometryInspector:
    def __init__(self): self.profile:dict={}
    def fit(self,images:list[np.ndarray])->dict:
        features=[extract_geometry(i)[0] for i in images]; features=[x for x in features if x]
        if not features: raise ValueError("Could not identify a foreground part in GOOD images")
        self.profile={}
        for key in features[0]:
            values=np.array([f[key] for f in features]); med=float(np.median(values)); mad=float(np.median(np.abs(values-med)))
            self.profile[key]={"median":med,"mad":mad,"p01":float(np.quantile(values,.01)),"p99":float(np.quantile(values,.99))}
        return self.profile
    def inspect(self,image:np.ndarray,multiplier:float)->tuple[bool,float,list[DefectRegion],dict]:
        features,contour=extract_geometry(image)
        if not features:return True,1.0,[],{"error":"part contour not found"}
        failures=[]; max_z=0.
        for key,value in features.items():
            expected=self.profile[key]; scale=max(expected["mad"]*1.4826,.015 if key!="hole_count" else .25); z=abs(value-expected["median"])/scale; max_z=max(max_z,z)
            if z>multiplier: failures.append(key)
        regions=[]
        if failures and contour is not None:
            x,y,w,h=cv2.boundingRect(contour); regions=[DefectRegion("shape / geometry",contour,(x,y,w,h),min(1,max_z/(multiplier*2)),"GEOMETRY")]
        return bool(failures),float(max_z),regions,{"features":features,"failures":failures}
    def save(self,path:Path)->None:path.write_text(json.dumps(self.profile,indent=2))
    def load(self,path:Path)->None:self.profile=json.loads(path.read_text())
