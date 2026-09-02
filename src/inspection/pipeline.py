from __future__ import annotations
from pathlib import Path
import json,cv2
from src.inspection.registration import PartRegistrar
from src.inspection.golden_bank import GoldenBank
from src.inspection.patchcore_detector import PatchCoreDetector
from src.inspection.visual_change_detector import VisualChangeDetector
from src.inspection.geometry import GeometryInspector
from src.inspection.localization_fusion import DefectLocalizationFusion
from src.inspection.decision_engine import DecisionEngine
from src.inspection.defect_renderer import DefectRenderer
from src.inspection.types import InspectionResult
from src.utils.image_utils import foreground_mask

class InspectionPipeline:
    def __init__(self,config:dict):
        self.c=config;a=Path(config["paths"]["artifacts"]);required=[a/"registration/template.png",a/"golden_bank/manifest.json",a/"patchcore/model.joblib",a/"visual_changenet/model.joblib",a/"geometry_profile.json",a/"thresholds.json"]
        missing=[str(x) for x in required if not x.exists()]
        if missing:raise FileNotFoundError("Models are not trained. Missing: "+", ".join(missing))
        self.reg=PartRegistrar(config["image_size"],config["registration"]["foreground_min_area_ratio"],config["registration"]["ecc_enabled"]);self.reg.load(required[0]);self.bank=GoldenBank();self.bank.load(a/"golden_bank");self.pc=PatchCoreDetector();self.pc.load(a/"patchcore/model.joblib");self.vcn=VisualChangeDetector();self.vcn.load(a/"visual_changenet/model.joblib");self.geo=GeometryInspector();self.geo.load(a/"geometry_profile.json");self.thresholds=json.loads((a/"thresholds.json").read_text());loc=config["localization"];self.fusion=DefectLocalizationFusion(loc["minimum_defect_area"],loc["morphology_kernel"],loc["merge_distance"]);self.decision=DecisionEngine(self.thresholds);self.renderer=DefectRenderer(loc["minimum_defect_area"])
    def inspect(self,frame):
        mask=foreground_mask(frame,self.c["inspection"]["presence_area_ratio"]);present=cv2.countNonZero(mask)/mask.size>=self.c["inspection"]["presence_area_ratio"]
        if not present:return InspectionResult("RECHECK",0,part_present=False,marked_image=frame.copy())
        image,confidence=self.reg.register(frame);golden=self.bank.select(image);pscore,amap=self.pc.predict(image);vscore=self.vcn.predict(golden,image);failure,gscore,gregions,gdetails=self.geo.inspect(image,self.thresholds["geometry_tolerance"]);regions=self.fusion.localize(image,golden,amap,self.thresholds["patchcore_localization"],gregions);result=self.decision.decide(confidence,failure,gscore,vscore,pscore,len(regions));marked=self.renderer.render(image,regions if result=="NG" else [])
        return InspectionResult(result,confidence,{"patchcore":pscore,"visual_change":vscore},{**gdetails,"score":gscore,"critical":failure},regions,marked,True)
