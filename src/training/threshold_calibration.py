from __future__ import annotations
import numpy as np

def metrics(labels:list[int],scores:list[float],threshold:float)->dict:
    y=np.asarray(labels);p=np.asarray(scores)>=threshold;tp=int(((y==1)&p).sum());tn=int(((y==0)&~p).sum());fp=int(((y==0)&p).sum());fn=int(((y==1)&~p).sum())
    precision=tp/max(tp+fp,1);recall=tp/max(tp+fn,1);specificity=tn/max(tn+fp,1);f1=2*precision*recall/max(precision+recall,1e-12);f2=5*precision*recall/max(4*precision+recall,1e-12)
    return {"accuracy":(tp+tn)/max(len(y),1),"ng_recall":recall,"good_specificity":specificity,"precision":precision,"f1":f1,"f2":f2,"confusion_matrix":[[tn,fp],[fn,tp]],"false_positive_rate":fp/max(fp+tn,1),"false_negative_rate":fn/max(fn+tp,1),"false_positives":fp,"false_negatives":fn}
def calibrate_threshold(labels:list[int],scores:list[float])->tuple[float,dict]:
    unique=np.unique(scores); candidates=np.r_[np.nextafter(unique.min(),-np.inf),unique,(unique[:-1]+unique[1:])/2]
    # When several thresholds have identical F2/recall/specificity, prefer the
    # lower (more NG-sensitive) threshold.  The previous final ``float(t)``
    # tie-break selected the *highest* equivalent threshold and could turn a
    # correctly learned NG training image into GOOD after calibration.
    ranked=[(metrics(labels,scores,float(t))["f2"],metrics(labels,scores,float(t))["ng_recall"],-metrics(labels,scores,float(t))["false_positive_rate"],-float(t)) for t in candidates]
    threshold=-max(ranked)[3];return threshold,metrics(labels,scores,threshold)
def localization_threshold(good_maps:list[np.ndarray])->float:
    if not good_maps:return 1.0
    normalized=[m/(np.quantile(m,.995)+1e-8) for m in good_maps];return float(np.quantile(np.concatenate([m.ravel() for m in normalized]),.995))
