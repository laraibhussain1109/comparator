from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
import json, logging, time
from PIL import Image
from src.models.classifier import ARCHITECTURE, CLASS_NAMES, assert_logits, create_classifier, require_torch
from src.models.preprocessing import PreprocessingConfig, get_eval_transform
from src.training.threshold_calibration import calibrate_threshold, metrics

log=logging.getLogger(__name__)

class ImageDataset:
    def __init__(self,samples,transform):self.samples=list(samples);self.transform=transform
    def __len__(self):return len(self.samples)
    def __getitem__(self,index):
        sample=self.samples[index]
        if sample.label not in CLASS_NAMES:raise ValueError(f"Invalid label {sample.label}: {sample.path}")
        with Image.open(sample.path) as image:x=self.transform(image)
        return x,sample.label,str(sample.path)

def _device(torch):return "cuda" if torch.cuda.is_available() else "cpu"

def hardware_details():
    torch,_=require_torch();return {"torch_version":torch.__version__,"cuda_available":torch.cuda.is_available(),"cuda_version":torch.version.cuda,"gpu_name":torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}

def _loader(samples,transform,batch=8,shuffle=False):
    torch,_=require_torch();return torch.utils.data.DataLoader(ImageDataset(samples,transform),batch_size=batch,shuffle=shuffle,num_workers=0,pin_memory=torch.cuda.is_available())

def evaluate(model,samples,transform,device,threshold=.5):
    torch,_=require_torch();model.eval();labels=[];ng=[];rows=[]
    with torch.inference_mode():
        for inputs,target,paths in _loader(samples,transform):
            logits=model(inputs.to(device));assert_logits(logits);prob=torch.softmax(logits,1).cpu()
            for p,y,path in zip(prob,target,paths):
                pred=1 if float(p[1])>=threshold else 0;labels.append(int(y));ng.append(float(p[1]));rows.append({"filename":path,"true_label":CLASS_NAMES[int(y)],"predicted_label":CLASS_NAMES[pred],"good_probability":float(p[0]),"ng_probability":float(p[1])})
    return labels,ng,rows

def save_checkpoint(path,model,preprocessing,epoch,validation_metrics,threshold):
    torch,_=require_torch();path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state_dict":model.state_dict(),"architecture":ARCHITECTURE,"class_mapping":CLASS_NAMES,"preprocessing":preprocessing.metadata(),"epoch":epoch,"validation_metrics":validation_metrics,"thresholds":{"classifier_ng_threshold":threshold},"training_timestamp":datetime.now(timezone.utc).isoformat()},path)

def _train(model,samples,transform,device,epochs,lr,class_weights=None,progress:Callable[[int,float,float],None]|None=None):
    torch,_=require_torch();trainable=[p for p in model.parameters() if p.requires_grad]
    if not trainable:raise RuntimeError("Classifier has no trainable parameters")
    optimizer=torch.optim.AdamW(trainable,lr=lr,weight_decay=1e-4)
    weight=torch.tensor(class_weights,dtype=torch.float32,device=device) if class_weights else None;loss_fn=torch.nn.CrossEntropyLoss(weight=weight)
    frozen_backbone=not any(parameter.requires_grad for parameter in model.features.parameters())
    cached_features=[];cached_labels=[]
    if frozen_backbone:
        log.info("Caching frozen ConvNeXt features once for %d images on %s; subsequent head epochs are fast",len(samples),device)
        model.eval()
        with torch.inference_mode():
            for inputs,labels,_ in _loader(samples,transform):
                inputs=inputs.to(device,non_blocking=True);cached_features.append(model.avgpool(model.features(inputs)).cpu());cached_labels.append(labels)
        feature_dataset=torch.utils.data.TensorDataset(torch.cat(cached_features),torch.cat(cached_labels))
    last_loss=0.;started=time.monotonic()
    for epoch in range(epochs):
        model.classifier.train();correct=0;seen=0;loss_total=0.
        if frozen_backbone:
            batches=torch.utils.data.DataLoader(feature_dataset,batch_size=8,shuffle=True,pin_memory=torch.cuda.is_available())
            for features,labels in batches:
                labels=labels.to(device,non_blocking=True);optimizer.zero_grad(set_to_none=True);logits=model.classifier(features.to(device,non_blocking=True));assert_logits(logits);loss=loss_fn(logits,labels);loss.backward();optimizer.step();batch_size=len(labels);loss_total+=float(loss)*batch_size;correct+=int((logits.argmax(1)==labels).sum());seen+=batch_size
        else:
            model.train()
            for inputs,labels,_ in _loader(samples,transform,shuffle=True):
                labels=labels.to(device);optimizer.zero_grad(set_to_none=True);logits=model(inputs.to(device));assert_logits(logits);loss=loss_fn(logits,labels);loss.backward();optimizer.step();batch_size=len(labels);loss_total+=float(loss)*batch_size;correct+=int((logits.argmax(1)==labels).sum());seen+=batch_size
        last_loss=loss_total/max(seen,1);accuracy=correct/max(seen,1);elapsed=time.monotonic()-started
        log.info("Classifier epoch %d/%d loss=%.6f accuracy=%.1f%% elapsed=%.1fs",epoch+1,epochs,last_loss,100*accuracy,elapsed)
        if progress:progress(epoch+1,last_loss,accuracy)
    return last_loss

def run_micro_overfit(good,ng,artifact_dir,preprocessing,epochs=120,progress=None):
    if len(good)<10 or len(ng)<10:raise RuntimeError(f"SANITY CHECK FAILED: requires 10 GOOD and 10 NG unique images; found {len(good)} GOOD and {len(ng)} NG. Full training has been stopped.")
    torch,_=require_torch();samples=list(good[:10])+list(ng[:10]);device=_device(torch);model=create_classifier(True,True).to(device)
    log.info("Sanity classifier device=%s hardware=%s",device,hardware_details());loss=_train(model,samples,get_eval_transform(preprocessing),device,epochs,1e-3,progress=progress);path=artifact_dir/"classifier"/"sanity_classifier.pt";save_checkpoint(path,model,preprocessing,epochs,{"sanity":True},.5)
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    from src.models.classifier_inference import ClassifierInference
    reloaded=ClassifierInference(path,device);rows=[{**reloaded.predict(s.path),"filename":str(s.path),"true_label":CLASS_NAMES[s.label]} for s in samples];correct=sum(r["predicted"]==r["true_label"] for r in rows)
    for row in rows:log.info("SANITY prediction %s true=%s predicted=%s GOOD=%.6f NG=%.6f",row["filename"],row["true_label"],row["predicted"],row["good_probability"],row["ng_probability"])
    (artifact_dir/"diagnostics").mkdir(parents=True,exist_ok=True);(artifact_dir/"diagnostics"/"sanity_predictions.json").write_text(json.dumps(rows,indent=2))
    if correct<20:raise RuntimeError(f"SANITY CHECK FAILED ({correct}/20): The model cannot correctly learn the training subset. Full training has been stopped.")
    return {"correct":correct,"total":20,"loss":loss,"checkpoint":str(path)}

def train_classifier(split,artifact_dir,image_size=224,epochs=30,progress=None):
    torch,_=require_torch();device=_device(torch);prep=PreprocessingConfig(image_size);counts=[sum(s.label==i for s in split.train) for i in (0,1)];ratio=max(counts)/min(counts);weights=[len(split.train)/(2*c) for c in counts] if ratio>=1.5 else None
    # A frozen backbone caches each image once.  It must cache the deterministic
    # evaluation representation: caching one random augmented view makes the
    # displayed 100% training accuracy refer to different tensors than GUI
    # inference and caused the acceptance verification to fail.
    eval_transform=get_eval_transform(prep)
    log.info("Full classifier device=%s hardware=%s train_images=%d class_counts=%s class_weights=%s",device,hardware_details(),len(split.train),counts,weights);model=create_classifier(True,True).to(device);_train(model,split.train,eval_transform,device,epochs,1e-3,weights,progress)
    labels,probs,_=evaluate(model,split.validation,eval_transform,device);calibrated_threshold,val=calibrate_threshold(labels,probs)
    threshold=min(.5,calibrated_threshold)
    if threshold != calibrated_threshold:
        log.info("Classifier threshold %.6f was conservatively capped at 0.5 to preserve the standard two-logit decision boundary and NG recall",calibrated_threshold)
        val=metrics(labels,probs,threshold)
    path=artifact_dir/"classifier"/"best_classifier.pt";save_checkpoint(path,model,prep,epochs,val,threshold);pre_labels,pre_probs,_=evaluate(model,split.validation,get_eval_transform(prep),device,threshold);pre=metrics(pre_labels,pre_probs,threshold)
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    from src.models.classifier_inference import ClassifierInference
    inference=ClassifierInference(path,device)
    results={}
    for name,samples in (("train",split.train),("validation",split.validation),("test",split.test)):
        rows=[];ys=[];ps=[]
        for sample in samples:
            row=inference.predict(sample.path);row.update(filename=str(sample.path),true_label=CLASS_NAMES[sample.label]);rows.append(row);ys.append(sample.label);ps.append(row["ng_probability"])
        results[name]=metrics(ys,ps,threshold);mis=artifact_dir/"diagnostics"/f"{name}_misclassified";mis.mkdir(parents=True,exist_ok=True)
        for i,row in enumerate(r for r in rows if r["predicted"]!=r["true_label"]):(mis/f"{i:04d}.json").write_text(json.dumps(row,indent=2))
    if abs(results["validation"]["accuracy"]-pre["accuracy"])>1e-6:raise RuntimeError("CHECKPOINT VERIFICATION FAILED: reloaded predictions differ from pre-save model")
    verify={};verification_rows=[]
    for label,name in CLASS_NAMES.items():
        chosen=[s for s in split.train if s.label==label][:10];rows=[]
        for sample in chosen:
            prediction=inference.predict(sample.path);row={"filename":str(sample.path),"true_label":name,**prediction};rows.append(row);verification_rows.append(row)
            log.info("TRAINING verification %s true=%s predicted=%s GOOD=%.6f NG=%.6f threshold=%.6f",sample.path,name,prediction["predicted"],prediction["good_probability"],prediction["ng_probability"],prediction["threshold"])
        verify[name]={"correct":sum(row["predicted_index"]==label for row in rows),"total":len(rows)}
    verification_path=artifact_dir/"diagnostics"/"training_inference_verification.json";verification_path.parent.mkdir(parents=True,exist_ok=True);verification_path.write_text(json.dumps(verification_rows,indent=2))
    log.info("TRAINING INFERENCE VERIFICATION GOOD: %s/%s NG: %s/%s",verify["GOOD"]["correct"],verify["GOOD"]["total"],verify["NG"]["correct"],verify["NG"]["total"])
    failed_ng=[row for row in verification_rows if row["true_label"]=="NG" and row["predicted"]!="NG"]
    if failed_ng:
        details="; ".join(f"{Path(row['filename']).name} NG={row['ng_probability']:.6f}" for row in failed_ng)
        raise RuntimeError(f"TRAINING INFERENCE VERIFICATION FAILED: {len(failed_ng)} NG training image(s) classified GOOD at threshold {threshold:.6f}: {details}. Full training stopped; details saved to {verification_path}")
    return {"checkpoint":str(path),"threshold":threshold,"validation_calibration":val,"metrics":results,"verification":verify,"checkpoint_verification":"PASS","preprocessing":prep.metadata(),"device":device}
