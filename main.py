# -*- coding: utf-8 -*-
import os
import pandas as pd
from fastapi import FastAPI, UploadFile, Form, Query
from fastapi.logger import logger
from typing import List
from scripts.use_model import analyze_photos
import tempfile
import shutil
from scripts.base.constants import HOST, PORT
from scripts.base.constants import BOAT_TYPE_MOTOR, BOAT_TYPE_SEAL

app = FastAPI()
seal_df = pd.read_csv("./data/gold/boats_itboat_seal_gold.csv")
motor_df = pd.read_csv("./data/gold/boats_itboat_MY_gold.csv")


def get_boat_model_info(model_type, model_name):
    if model_type == BOAT_TYPE_MOTOR:
        df = motor_df
    elif model_type == BOAT_TYPE_SEAL:
        df = seal_df
    else:
        return None
    return df[df["boat_name"].str.lower() == model_name.lower()].dropna(axis=1, how='any')
    
    
@app.post("/analyze")
def analyze(files: List[UploadFile], debug: bool = Form(False)):
    """
    Analyze uploaded yacht photos.
    - files: list of uploaded images
    - debug: enable verbose logging and Grad-CAM visualization (default=False)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        _files = []
        for file in files:
            temp_path = os.path.join(tmpdir, file.filename)
            _files.append(temp_path)
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            result = analyze_photos(_files, gradcam=debug)
            if debug:
                print(f"✅ Result for {file.filename}: {result}")
    r_result = {"debug": debug,
              "total_files": len(files),
              "photo_type_counts": result["photo_type_counts"],
              "winner_model": result["winner_model"]
              }
    description = ""
    if "winner_model" in r_result and r_result["winner_model"]:
        if "model_type" in r_result["winner_model"] and "model_name" in r_result["winner_model"]:
            description = get_boat_model_info(r_result["winner_model"]["model_type"], r_result["winner_model"]["model_name"])
    r_result["description"] = description
    return r_result


@app.get("/get_seal")
def get_seal(model_name: str = Query(..., description="Boat model name")):
    result = get_boat_model_info(BOAT_TYPE_SEAL, model_name)
    if result.empty:
        return {"status": "error", "message": f"No data found for model: {model_name}"}
    return {
        "status": "success",
        "model_name": model_name,
        "data": result.to_dict(orient="records")
    }


@app.get("/get_motor")
def get_motor(model_name: str = Query(..., description="Boat model name")):
    result = get_boat_model_info(BOAT_TYPE_MOTOR, model_name)
    if result.empty:
         return {"status": "error", "message": f"No data found for model: {model_name}"}
    return {
        "status": "success",
        "model_name": model_name,
        "data": result.to_dict(orient="records")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
