from __future__ import annotations
from pathlib import Path
from typing import Callable
import json,logging,time
import numpy as np
from src.training.dataset import scan_dataset
from src.training.splitting import split_dataset
from src.training.threshold_calibration import calibrate_threshold,localization_threshold,metrics
from src.utils.image_utils import read_image
from src.inspection.registration import PartRegistrar
from src.inspection.golden_bank import GoldenBank
from src.inspection.patchcore_detector import PatchCoreDetector
from src.inspection.visual_change_detector import VisualChangeDetector,TaoVisualChangeNetBackend
from src.inspection.geometry import GeometryInspector

log=logging.getLogger(__name__)
class Trainer:
    def __init__(self,config:dict,progress:Callable[[int,str,str],None]|None=None):self.c=config;self.emit=progress or (lambda *_:None)
    def _stage(self,p:int,name:str,message:str):log.info(message);self.emit(p,name,message)
    def run(self)->dict:
        started=time.time();a=Path(self.c["paths"]["artifacts"]);a.mkdir(parents=True,exist_ok=True)
        self._stage(2,"Validating dataset","Scanning GOOD and NG images and removing exact duplicates")
        inv=scan_dataset(Path(self.c["paths"]["dataset"]));self._stage(8,"Dataset ready",f"Found {len(inv.good)} GOOD and {len(inv.ng)} NG unique images")
        split=split_dataset(inv,int(self.c["runtime"]["random_seed"])); payload={k:[{"path":str(s.path),"label":"NG" if s.label else "GOOD","sha256":s.sha256} for s in getattr(split,k)] for k in ("train","validation","test")};payload["duplicate_groups"]=inv.duplicates;payload["near_duplicate_warnings"]=inv.near_duplicate_warnings;(a/"dataset_split.json").write_text(json.dumps(payload,indent=2))
        registrar=PartRegistrar(self.c["image_size"],self.c["registration"]["foreground_min_area_ratio"],self.c["registration"]["ecc_enabled"]);good_train=[s for s in split.train if s.label==0]
        self._stage(15,"Registration template","Building robust canonical template from GOOD training images");registrar.build_template([s.path for s in good_train],a/"registration"/"template.png")
        def reg(samples):return [registrar.register(read_image(s.path))[0] for s in samples]
        train_good=reg(good_train);train_ng=reg([s for s in split.train if s.label==1])
        self._stage(23,"Golden bank","Selecting representative GOOD references");bank=GoldenBank();bank.build(train_good,[s.path.name for s in good_train],a/"golden_bank",self.c["golden_bank"]["max_images"])
        self._stage(35,"Patch anomaly model","Building GOOD-only normal patch memory");pc=PatchCoreDetector(**{k:self.c["patchcore"][k] for k in ("patch_size","stride","max_memory_patches")});pc.fit(train_good);pc.save(a/"patchcore"/"model.joblib")
        backend=self.c["visual_change"]["backend"]
        if backend=="tao" or self.c["visual_change"].get("require_tao"):
            TaoVisualChangeNetBackend(self.c["visual_change"]["tao_command"]).require();raise RuntimeError("TAO backend validation succeeded, but this portable prototype expects a site-specific TAO experiment spec. Select the production-ready 'sklearn' comparison backend or provide an exported TAO adapter.")
        self._stage(50,"Visual comparison model","Training supervised GOOD-vs-NG reference comparison");vcn=VisualChangeDetector();vcn.fit(bank.images,train_good,train_ng);vcn.save(a/"visual_changenet"/"model.joblib")
        self._stage(61,"Geometry profile","Learning robust normalized geometry distributions");geo=GeometryInspector();geo.fit(train_good);geo.save(a/"geometry_profile.json")
        def score(samples):
            labels=[];ps=[];vs=[];maps=[]
            for s,image in zip(samples,reg(samples)):
                golden=bank.select(image);p,m=pc.predict(image);labels.append(s.label);ps.append(p);vs.append(vcn.predict(golden,image));
                if s.label==0:maps.append(m)
            return labels,ps,vs,maps
        self._stage(72,"Calibration","Calibrating thresholds on validation data with F2 emphasis");vl,vp,vv,vm=score(split.validation);pt,pm=calibrate_threshold(vl,vp);vt,vmtr=calibrate_threshold(vl,vv)
        thresholds={"registration_confidence":self.c["registration"]["min_confidence"],"patchcore_image":pt,"patchcore_localization":localization_threshold(vm),"visual_change":vt,"geometry_tolerance":self.c["geometry"]["tolerance_multiplier"]};(a/"thresholds.json").write_text(json.dumps(thresholds,indent=2))
        self._stage(85,"Untouched test evaluation","Evaluating final hierarchical scores on the held-out test split");tl,tp,tv,_=score(split.test); combined=np.maximum(np.asarray(tp)/max(pt,1e-9),np.asarray(tv)/max(vt,1e-9));test_metrics=metrics(tl,combined,1.0)
        summary={"good_images":len(inv.good),"ng_images":len(inv.ng),"split_counts":{"train":len(split.train),"validation":len(split.validation),"test":len(split.test)},"thresholds":thresholds,"validation":{"patchcore":pm,"visual_change":vmtr},"test":test_metrics,"warnings":inv.near_duplicate_warnings,"elapsed_seconds":round(time.time()-started,2),"models":{"patchcore_training":"GOOD only","visual_comparison":"GOOD and NG pairs","geometry_training":"GOOD only"}}
        (a/"training_summary.json").write_text(json.dumps(summary,indent=2));self._stage(100,"TRAINING COMPLETED",f"Test accuracy {test_metrics['accuracy']:.1%}; NG recall {test_metrics['ng_recall']:.1%}");return summary
