from pathlib import Path
import os
import cv2,numpy as np
from src.training.dataset import scan_dataset
from src.training.splitting import split_dataset
from src.training.threshold_calibration import calibrate_threshold
from src.inspection.registration import PartRegistrar
from src.inspection.geometry import extract_geometry
from src.inspection.decision_engine import DecisionEngine
from src.inspection.temporal_filter import TemporalConfirmation
from src.inspection.part_state_machine import PartStateMachine,PartState
from src.inspection.defect_renderer import DefectRenderer
from src.inspection.types import DefectRegion
from src.inspection.patchcore_detector import PatchCoreDetector


def test_joblib_core_limit_is_configured_before_sklearn_use():
    assert os.environ["LOKY_MAX_CPU_COUNT"].isdigit()
    assert int(os.environ["LOKY_MAX_CPU_COUNT"])>=1

def image(circle=True):
    x=np.zeros((160,160,3),np.uint8)
    if circle:cv2.circle(x,(80,80),45,(210,210,210),-1)
    return x
def write(path,im):cv2.imwrite(str(path),im)
def test_scan_split_and_duplicate_prevention(tmp_path):
    for label in ("GOOD","NG"):(tmp_path/label).mkdir()
    for label in ("GOOD","NG"):
        for i in range(5):
            im=image();im[5+i,5]=i+30+(100 if label=="NG" else 0);write(tmp_path/label/f"{i}.PNG",im)
    (tmp_path/"GOOD"/"duplicate.jpg").write_bytes((tmp_path/"GOOD"/"0.PNG").read_bytes())
    inventory=scan_dataset(tmp_path);assert len(inventory.samples)==10;assert inventory.duplicates
    split=split_dataset(inventory);hashes=[s.sha256 for group in (split.train,split.validation,split.test) for s in group];assert len(hashes)==len(set(hashes))
def test_registration_and_geometry():
    reg=PartRegistrar(128);registered,confidence=reg.normalize_pose(image());assert registered.shape==(128,128,3);assert confidence>0
    features,contour=extract_geometry(image());assert contour is not None;assert .8<features["circularity"]<1.1
def test_calibration_prefers_ng_recall():
    threshold,result=calibrate_threshold([0,0,1,1],[.1,.2,.25,.9]);assert result["ng_recall"]==1;assert .2<threshold<=.25
def test_hierarchical_decision():
    d=DecisionEngine({"registration_confidence":.4,"geometry_tolerance":4,"visual_change":.6,"patchcore_image":.7})
    assert d.decide(.2,True,9,1,1,2)=="RECHECK";assert d.decide(.8,True,5,0,0,0)=="NG";assert d.decide(.8,False,0,.7,.2,0)=="GOOD"
def test_temporal_voting():
    t=TemporalConfirmation(5,3);assert [t.update("NG") for _ in range(3)]==["RECHECK","RECHECK","NG"]
def test_part_state_counts_once():
    s=PartStateMachine(2);s.update(True);s.update(True);_,count=s.update(True,"GOOD");assert count;s.update(True,"GOOD")[1] is False
    s.update(False);state,_=s.update(False);assert state==PartState.PART_EXITED
def test_renderer_filters_and_does_not_modify_source():
    src=image();before=src.copy();small=np.array([[[1,1]],[[2,1]],[[2,2]],[[1,2]]]);big=np.array([[[10,10]],[[50,10]],[[50,50]],[[10,50]]]);regions=[DefectRegion("x",small,(1,1,2,2),1,"X"),DefectRegion("x",big,(10,10,40,40),1,"X")];r=DefectRenderer(50);out=r.render(src,regions);assert np.array_equal(src,before);assert not np.array_equal(out,src);assert len(r.filter_regions(regions,src.shape))==1


def test_patchcore_uses_explicit_logical_cpu_count(monkeypatch):
    monkeypatch.setattr("src.inspection.patchcore_detector.os.cpu_count", lambda: 6)
    detector=PatchCoreDetector(patch_size=16,stride=16,max_patches=100)
    detector.fit([image()])
    assert detector.model.n_jobs==6


def test_patchcore_cpu_count_has_safe_fallback(monkeypatch):
    monkeypatch.setattr("src.inspection.patchcore_detector.os.cpu_count", lambda: None)
    assert PatchCoreDetector._cpu_workers()==1
