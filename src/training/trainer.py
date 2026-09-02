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
from src.inspection.geometry import GeometryInspector
from src.models.preprocessing import PreprocessingConfig
from src.models.classifier_trainer import hardware_details,run_micro_overfit,train_classifier

log=logging.getLogger(__name__)
class Trainer:
    def __init__(self,config:dict,progress:Callable[[int,str,str],None]|None=None):self.c=config;self.emit=progress or (lambda *_:None)
    def _stage(self,p:int,name:str,message:str):log.info(message);self.emit(p,name,message)
    def run_diagnostics(self)->dict:
        a=Path(self.c["paths"]["artifacts"]);a.mkdir(parents=True,exist_ok=True);inv=scan_dataset(Path(self.c["paths"]["dataset"]));classifier_config=self.c.get("classifier",{});epochs=classifier_config.get("sanity_epochs",120);result=run_micro_overfit(inv.good,inv.ng,a,PreprocessingConfig(classifier_config.get("image_size",224)),epochs,lambda epoch,loss,accuracy:self._stage(round(100*epoch/epochs),"Sanity classifier",f"Epoch {epoch}/{epochs} — loss {loss:.6f}, accuracy {accuracy:.1%}"),classifier_config.get("batch_size",2));return {"sanity_check":result,"hardware":hardware_details(),"good_images":len(inv.good),"ng_images":len(inv.ng)}
    def run(self)->dict:
        started=time.time();a=Path(self.c["paths"]["artifacts"]);a.mkdir(parents=True,exist_ok=True)
        self._stage(2,"Validating dataset","Scanning GOOD and NG images and removing exact duplicates")
        inv=scan_dataset(Path(self.c["paths"]["dataset"]));self._stage(8,"Dataset ready",f"Found {len(inv.good)} GOOD and {len(inv.ng)} NG unique images")
        for sample in inv.samples[:min(6,len(inv.samples))]:log.info("Label audit: %s -> class %d %s",sample.path,sample.label,"NG" if sample.label else "GOOD")
        self._stage(10,"Sanity check","Micro-overfitting 10 GOOD and 10 NG images before full training")
        classifier_config=self.c.get("classifier",{});batch_size=classifier_config.get("batch_size",2);sanity_epochs=classifier_config.get("sanity_epochs",120);sanity=run_micro_overfit(inv.good,inv.ng,a,PreprocessingConfig(classifier_config.get("image_size",224)),sanity_epochs,lambda epoch,loss,accuracy:self._stage(10+round(6*epoch/sanity_epochs),"Sanity classifier",f"Epoch {epoch}/{sanity_epochs} — loss {loss:.6f}, accuracy {accuracy:.1%}"),batch_size)
        split=split_dataset(inv,int(self.c["runtime"]["random_seed"])); payload={k:[{"path":str(s.path),"label":"NG" if s.label else "GOOD","sha256":s.sha256} for s in getattr(split,k)] for k in ("train","validation","test")};payload["duplicate_groups"]=inv.duplicates;payload["near_duplicate_warnings"]=inv.near_duplicate_warnings;(a/"dataset_split.json").write_text(json.dumps(payload,indent=2))
        classifier_epochs=self.c.get("classifier",{}).get("epochs",30);self._stage(18,"Binary classifier","Training the GOOD/NG classifier head; the pretrained ConvNeXt backbone is intentionally frozen and its features are cached once")
        classifier=train_classifier(split,a,classifier_config.get("image_size",224),classifier_epochs,lambda epoch,loss,accuracy:self._stage(18+round(30*epoch/classifier_epochs),"Binary classifier",f"Epoch {epoch}/{classifier_epochs} — loss {loss:.6f}, accuracy {accuracy:.1%}"),batch_size)
        registrar=PartRegistrar(self.c["image_size"],self.c["registration"]["foreground_min_area_ratio"],self.c["registration"]["ecc_enabled"]);good_train=[s for s in split.train if s.label==0]
        self._stage(52,"Registration template","Building robust canonical template from GOOD training images");registrar.build_template([s.path for s in good_train],a/"registration"/"template.png")
        def reg(samples):return [registrar.register(read_image(s.path))[0] for s in samples]
        train_good=reg(good_train);train_ng=reg([s for s in split.train if s.label==1])
        self._stage(58,"Golden bank","Selecting representative GOOD references");bank=GoldenBank();bank.build(train_good,[s.path.name for s in good_train],a/"golden_bank",self.c["golden_bank"]["max_images"])
        self._stage(65,"Patch anomaly model","Building GOOD-only normal patch memory");pc=PatchCoreDetector(**{k:self.c["patchcore"][k] for k in ("patch_size","stride","max_memory_patches")});pc.fit(train_good);pc.save(a/"patchcore"/"model.joblib")
        self._stage(75,"Geometry profile","Learning robust normalized geometry distributions");geo=GeometryInspector();geo.fit(train_good);geo.save(a/"geometry_profile.json")
        def score(samples):
            labels=[];ps=[];maps=[]
            for s,image in zip(samples,reg(samples)):
                p,m=pc.predict(image);labels.append(s.label);ps.append(p)
                if s.label==0:maps.append(m)
            return labels,ps,maps
        self._stage(82,"Calibration","Calibrating PatchCore thresholds on validation only");vl,vp,vm=score(split.validation);pt,pm=calibrate_threshold(vl,vp)
        thresholds={"registration_confidence":self.c["registration"]["min_confidence"],"classifier_ng_threshold":classifier["threshold"],"patchcore_image":pt,"patchcore_localization":localization_threshold(vm),"geometry_tolerance":self.c["geometry"]["tolerance_multiplier"]};(a/"thresholds.json").write_text(json.dumps(thresholds,indent=2))
        test_metrics=classifier["metrics"]["test"]
        summary={"root_cause":"Previous VisualChangeNet fusion could suppress supervised NG evidence. The first binary-classifier revision also cached a random augmented training view while verifying the deterministic evaluation view, and threshold-calibration ties incorrectly preferred the highest threshold; both could make a learned NG training image verify as GOOD.","sanity_check":sanity,"classifier":classifier,"hardware":hardware_details(),"good_images":len(inv.good),"ng_images":len(inv.ng),"split_counts":{"train":len(split.train),"validation":len(split.validation),"test":len(split.test)},"thresholds":thresholds,"validation":{"patchcore":pm,"classifier":classifier["metrics"]["validation"]},"test":test_metrics,"warnings":inv.near_duplicate_warnings,"elapsed_seconds":round(time.time()-started,2),"models":{"classifier":"ConvNeXt-Tiny ImageNet; GOOD/NG primary","patchcore_training":"GOOD only; localization secondary","geometry_training":"GOOD only"}}
        (a/"training_summary.json").write_text(json.dumps(summary,indent=2));self._stage(100,"TRAINING COMPLETED",f"Test accuracy {test_metrics['accuracy']:.1%}; NG recall {test_metrics['ng_recall']:.1%}");return summary
