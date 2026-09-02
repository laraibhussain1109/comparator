from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
from src.utils.image_utils import EXTENSIONS,read_image

@dataclass(frozen=True)
class Sample:path:Path;label:int;sha256:str
@dataclass
class DatasetInventory:
    samples:list[Sample];duplicates:dict[str,list[str]];near_duplicate_warnings:list[str]
    @property
    def good(self):return [s for s in self.samples if s.label==0]
    @property
    def ng(self):return [s for s in self.samples if s.label==1]

def scan_dataset(root:Path)->DatasetInventory:
    samples=[]; hashes:dict[str,list[str]]={}; phashes:dict[str,list[str]]={}
    try: import imagehash
    except ImportError:imagehash=None
    for folder,label in (("GOOD",0),("NG",1)):
        directory=root/folder
        if not directory.exists():raise ValueError(f"Missing dataset folder: {directory}")
        for path in sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS):
            digest=hashlib.sha256(path.read_bytes()).hexdigest();hashes.setdefault(digest,[]).append(str(path));samples.append(Sample(path,label,digest))
            if imagehash:
                from PIL import Image
                try: phashes.setdefault(str(imagehash.phash(Image.open(path))),[]).append(str(path))
                except Exception: read_image(path)
    duplicates={k:v for k,v in hashes.items() if len(v)>1}; unique=[];seen=set()
    for sample in samples:
        if sample.sha256 not in seen:unique.append(sample);seen.add(sample.sha256)
    near=[f"Possible near-duplicates: {', '.join(v)}" for v in phashes.values() if len(v)>1 and len({hashlib.sha256(Path(x).read_bytes()).hexdigest() for x in v})>1]
    if len([s for s in unique if s.label==0])<3 or len([s for s in unique if s.label==1])<3:raise ValueError("At least 3 unique readable images are required in each of dataset/GOOD and dataset/NG (10+ per class is strongly recommended).")
    for s in unique:read_image(s.path)
    return DatasetInventory(unique,duplicates,near)
