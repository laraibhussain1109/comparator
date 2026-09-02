from __future__ import annotations
import argparse
from pathlib import Path
from src.models.classifier_inference import ClassifierInference, image_details

def main():
    parser=argparse.ArgumentParser(description="Run the exact classifier inference implementation used by the GUI")
    parser.add_argument("image",type=Path);parser.add_argument("--checkpoint",type=Path,default=Path("artifacts/classifier/best_classifier.pt"));args=parser.parse_args()
    inference=ClassifierInference(args.checkpoint);result=inference.predict(args.image)
    print(f"Image: {args.image}\nPredicted: {result['predicted']}\nGOOD probability: {result['good_probability']:.8f}\nNG probability: {result['ng_probability']:.8f}\nModel checkpoint: {result['checkpoint']}\nImage size: {image_details(args.image)}\nPreprocessing: {inference.preprocessing.metadata()}\nDevice: {result['device']}")
if __name__=="__main__":main()
