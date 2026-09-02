from __future__ import annotations
import cv2
import numpy as np
from src.inspection.types import DefectRegion


class DefectRenderer:
    def __init__(self,min_area:int=90):self.min_area=min_area
    def filter_regions(self,regions:list[DefectRegion],shape:tuple[int,...])->list[DefectRegion]:
        h,w=shape[:2]; output=[]
        for r in regions:
            x,y,bw,bh=r.bounding_box
            if cv2.contourArea(r.contour)>=self.min_area and bw>=3 and bh>=3 and x<w and y<h and x+bw>0 and y+bh>0:output.append(r)
        return output
    def render(self,image:np.ndarray,regions:list[DefectRegion])->np.ndarray:
        output=image.copy(); thickness=max(2,min(4,round(max(image.shape[:2])/300)))
        for region in self.filter_regions(regions,image.shape):
            cv2.drawContours(output,[region.contour.astype(np.int32)],-1,(0,0,255),thickness,cv2.LINE_AA)
            x,y,_,_=region.bounding_box; cv2.putText(output,"ISSUE",(x,max(20,y-7)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2,cv2.LINE_AA)
        return output
