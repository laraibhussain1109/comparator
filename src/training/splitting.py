from __future__ import annotations
from dataclasses import dataclass
import random
from src.training.dataset import DatasetInventory,Sample

@dataclass
class DatasetSplit:train:list[Sample];validation:list[Sample];test:list[Sample]
def split_dataset(inventory:DatasetInventory,seed:int=42)->DatasetSplit:
    groups={0:inventory.good.copy(),1:inventory.ng.copy()}; rng=random.Random(seed);train=[];validation=[];test=[]
    for values in groups.values():
        rng.shuffle(values);n=len(values);n_val=max(1,round(n*.15));n_test=max(1,round(n*.15));n_train=n-n_val-n_test
        if n_train<1:raise ValueError("Not enough unique images for train/validation/test splits")
        train+=values[:n_train];validation+=values[n_train:n_train+n_val];test+=values[n_train+n_val:]
    return DatasetSplit(train,validation,test)
