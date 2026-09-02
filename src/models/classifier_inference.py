from __future__ import annotations
from pathlib import Path
from PIL import Image
from src.models.classifier import ARCHITECTURE, CLASS_NAMES, assert_logits, create_classifier, require_torch, validate_class_mapping
from src.models.preprocessing import PreprocessingConfig, get_eval_transform

class ClassifierInference:
    def __init__(self, checkpoint, device=None):
        torch,_=require_torch();self.checkpoint=Path(checkpoint)
        if not self.checkpoint.is_file(): raise FileNotFoundError(f"Verified classifier checkpoint not found: {self.checkpoint}")
        self.device=device or ("cuda" if torch.cuda.is_available() else "cpu")
        data=torch.load(self.checkpoint,map_location=self.device,weights_only=False)
        required={"model_state_dict","architecture","class_mapping","preprocessing","epoch","validation_metrics","thresholds","training_timestamp"}
        missing=required-data.keys()
        if missing: raise ValueError(f"Checkpoint metadata missing: {sorted(missing)}")
        if data["architecture"] != ARCHITECTURE: raise ValueError(f"Checkpoint architecture {data['architecture']} is incompatible with {ARCHITECTURE}")
        validate_class_mapping(data["class_mapping"]);self.preprocessing=PreprocessingConfig(**data["preprocessing"])
        self.threshold=float(data["thresholds"]["classifier_ng_threshold"]);self.model=create_classifier(pretrained=False,freeze_backbone=False)
        self.model.load_state_dict(data["model_state_dict"],strict=True);self.model.to(self.device).eval();self.transform=get_eval_transform(self.preprocessing)
    def predict(self,image):
        torch,_=require_torch()
        with torch.inference_mode():
            logits=self.model(self.transform(image).unsqueeze(0).to(self.device));assert_logits(logits);probs=torch.softmax(logits,dim=1)[0].cpu().tolist()
        predicted=1 if probs[1]>=self.threshold else 0
        return {"predicted_index":predicted,"predicted":CLASS_NAMES[predicted],"good_probability":probs[0],"ng_probability":probs[1],"threshold":self.threshold,"checkpoint":str(self.checkpoint),"device":str(self.device)}

def image_details(path):
    with Image.open(path) as image:return image.size
