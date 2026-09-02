from __future__ import annotations


class DecisionEngine:
    def __init__(self,thresholds:dict):self.t=thresholds
    def decide(self,registration:float,geometry_failure:bool,geometry_score:float,vcn:float,patchcore:float,regions:int)->str:
        if registration<self.t["registration_confidence"]:return "RECHECK"
        if geometry_failure and geometry_score>=self.t["geometry_tolerance"]:return "NG"
        supporting=regions>0 or patchcore>=self.t["patchcore_image"]
        if vcn>=self.t["visual_change"] and supporting:return "NG"
        if patchcore>=self.t["patchcore_image"]:return "NG"
        return "GOOD"
