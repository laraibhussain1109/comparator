from __future__ import annotations
from dataclasses import asdict, dataclass
import random
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from src.models.classifier import require_torch

@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: int = 224
    resize_strategy: str = "letterbox_center"
    color_space: str = "RGB"
    mean: tuple[float, ...] = (.485, .456, .406)
    std: tuple[float, ...] = (.229, .224, .225)

    def metadata(self): return asdict(self)

def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3: raise ValueError("Expected a three-channel BGR image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def _pil_rgb(image) -> Image.Image:
    if isinstance(image, Image.Image): return ImageOps.exif_transpose(image).convert("RGB")
    if isinstance(image, np.ndarray): return Image.fromarray(bgr_to_rgb(image))
    return ImageOps.exif_transpose(Image.open(image)).convert("RGB")

class SharedTransform:
    def __init__(self, config: PreprocessingConfig, training=False): self.config,self.training=config,training
    def __call__(self, image):
        torch,_=require_torch(); im=_pil_rgb(image)
        if self.training:
            angle=random.uniform(-3,3); scale=random.uniform(.97,1.03)
            im=im.rotate(angle, Image.Resampling.BILINEAR, translate=(random.randint(-2,2),random.randint(-2,2)), fillcolor=(0,0,0))
            if scale != 1: im=im.resize((max(1,int(im.width*scale)),max(1,int(im.height*scale))),Image.Resampling.BILINEAR)
            im=ImageEnhance.Brightness(im).enhance(random.uniform(.95,1.05));im=ImageEnhance.Contrast(im).enhance(random.uniform(.95,1.05))
        size=self.config.image_size; ratio=min(size/im.width,size/im.height); resized=im.resize((max(1,round(im.width*ratio)),max(1,round(im.height*ratio))),Image.Resampling.BILINEAR)
        canvas=Image.new("RGB",(size,size));canvas.paste(resized,((size-resized.width)//2,(size-resized.height)//2))
        array=np.asarray(canvas,dtype=np.float32).transpose(2,0,1)/255.;tensor=torch.from_numpy(array)
        mean=torch.tensor(self.config.mean)[:,None,None];std=torch.tensor(self.config.std)[:,None,None]
        return (tensor-mean)/std

def get_train_transform(config=PreprocessingConfig()): return SharedTransform(config,True)
def get_eval_transform(config=PreprocessingConfig()): return SharedTransform(config,False)
