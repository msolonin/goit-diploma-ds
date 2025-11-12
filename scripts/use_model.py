# -*- coding: utf-8 -*-

import torch
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from scripts.base.base_use_model import use_base_model
from scripts.base.base_model_dataset import  BoatModelClassifier, BoatTypeClassifier, PhotoTypeClassifier
from scripts.base.constants import BOAT_TYPES, BOAT_TYPE, BOAT_MODEL, PHOTO_TYPE, PHOTO_TYPES_ALL, PHOTO_TYPES, DEFAULT_CSV_PATH, BOAT_CLASS_IN, THRESHOLD, MODEL_THRESHOLD


PHOTO_TYPE_WEIGHTS = {"boat": 1.0, "out": 0.7, "in": 0.3}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def use_photo_type_model(image_path, gradcam=False, csv_path=DEFAULT_CSV_PATH):
    best_model_path = f"./models/{PHOTO_TYPE}_classifier.pth"
    classes = PHOTO_TYPES_ALL
    model = PhotoTypeClassifier(num_classes=len(classes))
    return use_base_model(DEVICE, model, classes, best_model_path, image_path,
                          model_name="photo_type", gradcam=gradcam)


def use_boat_type_model(image_path, gradcam=False, csv_path=DEFAULT_CSV_PATH):
    best_model_path = f"./models/{BOAT_TYPE}_clasifier.pth"
    classes = BOAT_TYPES
    model = BoatTypeClassifier(num_classes=len(classes))
    return use_base_model(DEVICE, model, classes, best_model_path, image_path,
                          model_name="model_type", gradcam=gradcam)


def use_boat_name_model(image_path, model_type,  gradcam=False, csv_path=DEFAULT_CSV_PATH):
    if model_type not in BOAT_TYPES:
        print(f"❌ Model type shoul be in of {BOAT_TYPES}")
    else:
        print(f"✅ Start train model for {model_type} on {DEVICE}")
        best_model_path = f"./models/{BOAT_MODEL}_{model_type}_clasifier.pth"
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
    photo_type, _p, _, debug = use_photo_type_model(image_path, gradcam=gradcam)
    result["photo_type"] = {"target": photo_type, "percent": _p, "top": _, "debug": debug}
    # print(f"✅ Photo type: {photo_type} | {_p}")
    if photo_type != BOAT_CLASS_IN and _p > THRESHOLD:
        model_type, _p, _, debug = use_boat_type_model(image_path, gradcam=gradcam)
        if _p > THRESHOLD:
            result["model_type"] = {"target": model_type, "percent": _p, "top": _, "debug": debug}
            # print(f"✅ Model type: {model_type} | {_p}")
            model_name, _p, _, debug = use_boat_name_model(image_path, model_type, gradcam=gradcam)
            # print(f"✅ Model name: {model_name} | {_p}")
            if _p > MODEL_THRESHOLD:
                result["model_name"] = {"target": model_name, "percent": _p, "top": _, "debug": debug}
    return result


def analyze_photos(image_paths, gradcam=False):
    """
    Analyze multiple photos and return only:
      - photo_type_counts
      - top (winning) model by avg_score
    """
    photo_type_counter = Counter()
    model_votes = defaultdict(list)
    model_scores = defaultdict(list)
    model_types = {}

    for image_path in image_paths:
        res = analyze_photo(image_path, gradcam=gradcam)
        photo_type = res["photo_type"]["target"]
        photo_type_counter[photo_type] += 1

        # Only out/boat photos contribute to model votes
        if photo_type != "in" and res["model_name"] and res["model_name"]["percent"] >= THRESHOLD:
            model_name = res["model_name"]["target"]
            model_type = res["model_type"]["target"] if res["model_type"] else None
            score = res["model_name"]["percent"] * PHOTO_TYPE_WEIGHTS.get(photo_type, 1.0)

            key = (model_type, model_name)
            model_votes[key].append(1)
            model_scores[key].append(score)
            model_types[key] = model_type

    # Aggregate model scores
    final_summary = []
    for (m_type, m_name), scores in model_scores.items():
        avg_score = np.mean(scores)
        count = len(model_votes[(m_type, m_name)])
        final_summary.append({
            "model_type": m_type,
            "model_name": m_name,
            "avg_score": round(avg_score, 2),
            "votes": count
        })

    winner = max(final_summary, key=lambda x: x["avg_score"]) if final_summary else None

    return {
        "photo_type_counts": dict(photo_type_counter),
        "winner_model": winner
    }


def analyze_photo_data(data: list):
    """
    Analyze data list:
      - photo_type_counts
      - top (winning) model by avg_score
    """
    photo_type_counter = Counter()
    model_votes = defaultdict(list)
    model_scores = defaultdict(list)
    model_types = {}
    for res in data:
        photo_type = res["photo_type"]["target"]
        photo_type_counter[photo_type] += 1
        # Only out/boat photos contribute to model votes
        if photo_type != "in" and res["model_name"] and res["model_name"]["percent"] >= THRESHOLD:
            model_name = res["model_name"]["target"]
            model_type = res["model_type"]["target"] if res["model_type"] else None
            score = res["model_name"]["percent"] * PHOTO_TYPE_WEIGHTS.get(photo_type, 1.0)

            key = (model_type, model_name)
            model_votes[key].append(1)
            model_scores[key].append(score)
            model_types[key] = model_type

    # Aggregate model scores
    final_summary = []
    for (m_type, m_name), scores in model_scores.items():
        avg_score = np.mean(scores)
        count = len(model_votes[(m_type, m_name)])
        final_summary.append({
            "model_type": m_type,
            "model_name": m_name,
            "avg_score": round(avg_score, 2),
            "votes": count
        })

    winner = max(final_summary, key=lambda x: x["avg_score"]) if final_summary else None

    return {
        "photo_type_counts": dict(photo_type_counter),
        "winner_model": winner
    }
