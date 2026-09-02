import pytest
pytest.importorskip("torch");pytest.importorskip("torchvision")
import cv2,numpy as np,yaml
from src.utils.config import load_config
from src.training.trainer import Trainer
from src.inspection.pipeline import InspectionPipeline

def test_end_to_end_training_and_loading(tmp_path):
    root=tmp_path;dataset=root/"dataset";(dataset/"GOOD").mkdir(parents=True);(dataset/"NG").mkdir()
    rng=np.random.default_rng(4)
    for label in ("GOOD","NG"):
        for i in range(14):
            im=np.zeros((128,128,3),np.uint8);cv2.circle(im,(64+i%2,64),35,(180+i,180+i,180+i),-1)
            if label=="NG":cv2.rectangle(im,(45+i,45),(57+i,57),(20,20,20),-1)
            cv2.imwrite(str(dataset/label/f"{i}.png"),im)
    source=yaml.safe_load((__import__('pathlib').Path(__file__).parents[1]/"config.yaml").read_text());source["paths"]={"dataset":"./dataset","artifacts":"./artifacts","results":"./results","logs":"./logs"};source["image_size"]=96;source["classifier"].update(image_size=64,sanity_epochs=2,epochs=2);source["patchcore"]["max_memory_patches"]=500;source["golden_bank"]["max_images"]=3;(root/"config.yaml").write_text(yaml.safe_dump(source));config=load_config(root/"config.yaml",root)
    # The deliberately tiny epoch count need not pass micro-overfit; production configuration does.
    with pytest.raises(RuntimeError,match="SANITY CHECK FAILED"):Trainer(config).run()
