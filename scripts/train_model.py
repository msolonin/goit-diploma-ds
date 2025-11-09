# -*- coding: utf-8 -*-

import torch
from scripts.base.utils import transform
from scripts.base.base_model_dataset import BoatModelDataset, BoatModelClassifier, BoatTypeDataset, BoatTypeClassifier, PhotoTypeDataset, PhotoTypeClassifier
from scripts.base.base_train_model import train_base_model
from base.constants import BOAT_TYPES, BOAT_TYPE, BOAT_MODEL, PHOTO_TYPE, DEFAULT_CSV_PATH


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_photo_type_model(csv_path=DEFAULT_CSV_PATH):
    best_model_path = f"../models/{PHOTO_TYPE}_classifier.pth"
    dataset = PhotoTypeDataset(csv_path, transform=transform)
    model = PhotoTypeClassifier(num_classes=len(dataset.classes))
    train_base_model(DEVICE, dataset, model, PHOTO_TYPE, best_model_path)


def train_boat_type_model(csv_path=DEFAULT_CSV_PATH):
    best_model_path = f"../models/{BOAT_TYPE}_clasifier.pth"
    dataset = BoatTypeDataset(csv_path, transform=transform)
    model = BoatTypeClassifier(num_classes=len(dataset.classes))
    train_base_model(DEVICE, dataset, model, BOAT_TYPE, best_model_path)


def train_boat_name_model(model_type, csv_path=DEFAULT_CSV_PATH):
    if model_type not in BOAT_TYPES:
        print(f"❌ Model type shoul be in of {BOAT_TYPES}")
    else:
        print(f"✅ Start train model for {model_type} on {DEVICE}")
        best_model_path = f"../models/{BOAT_MODEL}_{model_type}_clasifier.pth"
        dataset = BoatModelDataset(csv_path, model_type=model_type, transform=transform)
        model = BoatModelClassifier(num_classes=len(dataset.classes))
        train_base_model(DEVICE, dataset, model, BOAT_MODEL, best_model_path)
