# -*- coding: utf-8 -*-

PHOTO_TYPES = ["boat", "out"]
BOAT_TYPES = ["motor", "seal"]
PHOTO_TYPES_ALL = ["in", "out", "boat"]

BOAT_TYPE = "boat_type"
PHOTO_TYPE = "photo_type"
BOAT_MODEL = "boat_model"
IMAGE_PATH = "image_path"

MAX_EPOCHS = 25
BATCH_SIZE = 16
TRANSFORM_SIZE = (224, 224)

DEFAULT_CSV_PATH = "../data/boat_pictures_dataset.csv"
HEATMAP_FOLDER = "../heatmap"
TOP = 5
THRESHOLD = 90