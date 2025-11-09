# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset
from torchvision import models
import torch.nn as nn
from PIL import Image
import pandas as pd

from base.constants import PHOTO_TYPES, BOAT_TYPES, PHOTO_TYPES_ALL, BOAT_TYPE, PHOTO_TYPE, BOAT_MODEL, IMAGE_PATH

# =========================================================
# Model of Boat dataset and Clasifier [Model of boat]
# =========================================================

class BoatModelDataset(Dataset):
    def __init__(self, csv_file, model_type, transform=None, photo_types=PHOTO_TYPES):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[
            (self.df[BOAT_TYPE] == model_type) & 
            (self.df[PHOTO_TYPE].isin(photo_types))
        ].reset_index(drop=True)

        self.classes = sorted(self.df[BOAT_MODEL].unique())
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row[IMAGE_PATH]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.classes.index(row[BOAT_MODEL])
        return img, torch.tensor(label)


class BoatModelClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
    
# =========================================================
# Boat type dataset and Clasifier ["motor", "seal"] 
# =========================================================

class BoatTypeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df[PHOTO_TYPE] == "boat"].reset_index(drop=True)
        self.df = self.df[self.df[BOAT_TYPE].isin(BOAT_TYPES)].reset_index(drop=True)
        self.classes = sorted(self.df[BOAT_TYPE].unique())  # ["motor", "seal"]
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row[IMAGE_PATH]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.classes.index(row[BOAT_TYPE])
        return img, torch.tensor(label)


class BoatTypeClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# =========================================================
# Photo type dataset and Clasifier [in, out, boat]
# =========================================================

class PhotoTypeDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df[PHOTO_TYPE].isin(PHOTO_TYPES_ALL)].reset_index(drop=True)
        self.classes = sorted(self.df[PHOTO_TYPE].unique())
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row[IMAGE_PATH]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.classes.index(row[PHOTO_TYPE])
        return img, torch.tensor(label)


class PhotoTypeClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
