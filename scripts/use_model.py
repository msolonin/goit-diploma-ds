# -*- coding: utf-8 -*-

import torch
import pandas as pd
from scripts.base.base_use_model import use_base_model
from scripts.base.base_model_dataset import  BoatModelClassifier, BoatTypeClassifier, PhotoTypeClassifier
from base.constants import BOAT_TYPES, BOAT_TYPE, BOAT_MODEL, PHOTO_TYPE, PHOTO_TYPES_ALL, PHOTO_TYPES, DEFAULT_CSV_PATH


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def use_photo_type_model(image_path, gradcam=False, csv_path=DEFAULT_CSV_PATH):
    best_model_path = f"../models/{PHOTO_TYPE}_classifier.pth"
    classes = PHOTO_TYPES_ALL
    model = PhotoTypeClassifier(num_classes=len(classes))
    return use_base_model(DEVICE, model, classes, best_model_path, image_path,
                          model_name="photo_type", gradcam=gradcam)


def use_boat_type_model(image_path, gradcam=False, csv_path=DEFAULT_CSV_PATH):
    best_model_path = f"../models/{BOAT_TYPE}_clasifier.pth"
    classes = BOAT_TYPES
    model = BoatTypeClassifier(num_classes=len(classes))
    return use_base_model(DEVICE, model, classes, best_model_path, image_path,
                          model_name="model_type", gradcam=gradcam)


def use_boat_name_model(image_path, model_type,  gradcam=False, csv_path=DEFAULT_CSV_PATH):
    if model_type not in BOAT_TYPES:
        print(f"❌ Model type shoul be in of {BOAT_TYPES}")
    else:
        print(f"✅ Start train model for {model_type} on {DEVICE}")
        best_model_path = f"../models/{BOAT_MODEL}_{model_type}_clasifier.pth"
        df = pd.read_csv(csv_path)
        df = df[(df[BOAT_TYPE] == model_type) & (df[PHOTO_TYPE].isin(PHOTO_TYPES))]
        classes = sorted(df[BOAT_MODEL].unique().tolist())
        model = BoatModelClassifier(num_classes=len(classes))
        return use_base_model(DEVICE, model, classes, best_model_path,
                              image_path, model_name="model_name",
                              gradcam=gradcam)


def analyze_photo(image_path,  gradcam=False):
    result = {"photo_type": None,
              "model_type": None,
              "model_name": None}
    photo_type, _p, _ = use_photo_type_model(image_path, gradcam=gradcam)
    result["photo_type"] = {"target": photo_type, "percent": _p, "top": _}
    # print(f"✅ Photo type: {photo_type} | {_p}")
    if photo_type != "in":
        model_type, _p, _ = use_boat_type_model(image_path, gradcam=gradcam)
        result["model_type"] = {"target": model_type, "percent": _p, "top": _}
        # print(f"✅ Model type: {model_type} | {_p}")
        model_name, _p, _ = use_boat_name_model(image_path, model_type, gradcam=gradcam)
        # print(f"✅ Model name: {model_name} | {_p}")
        result["model_name"] = {"target": model_name, "percent": _p, "top": _}
    return result


image_path =  "/home/msolonin/Desktop/YachtDatasets/scrapper/images_SEAL_output/Hallberg-Rassy 50/b568c7febb97_out.jpg" 
t = analyze_photo(image_path, gradcam=True)
print(t)