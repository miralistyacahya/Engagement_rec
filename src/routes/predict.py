import io
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
from joblib import load
import zipfile, tempfile
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from utils.response import http_response_error, http_response_success
from utils.prepfeat import preprocess_feat
from utils.prepimg import preprocess_img
from constants.http_enum import HttpStatusCode

router = APIRouter(prefix="/predict")

# load Keras model
MODEL_PATH = "src/lib/best_model_effs_lstm_gan5.keras"
# MODEL_PATH = "src/lib/bestmodelv2.h5"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load Keras model: {e}") from e

# load scaler
SCALER_PATH = "src/lib/scaler.pkl"
try:
    scaler: StandardScaler = load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Could not load scaler: {e}") from e

@router.post("")
async def predict(
    file: UploadFile = File(...)
):
    
    if not file.filename.endswith(".zip"):
        raise HTTPException(
            status_code=HttpStatusCode.BadRequest,
            detail="Only .zip files are accepted",
        )

    content = await file.read()
    try:
        zip_file = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        raise HTTPException(HttpStatusCode.BadRequest, "Invalid ZIP archive")
    
    temp_dir = tempfile.mkdtemp()
    imgs = []

    for name in zip_file.namelist():
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            dest = os.path.join(temp_dir, os.path.basename(name))
            with open(dest, "wb") as f:
                f.write(zip_file.read(name))
            imgs.append(dest)
    zip_file.close()

    if not imgs:
        raise HTTPException(HttpStatusCode.UnprocessableEntity, "No image files found in the ZIP")
    
    try:
        crops = preprocess_img(imgs)
    except Exception as e:
        raise HTTPException(HttpStatusCode.UnprocessableEntity, f"Could not crop images: {e}") from e
    
    try:
       npy = await preprocess_feat(crops)
    except Exception as e:
        raise HTTPException(HttpStatusCode.UnprocessableEntity, f"Could not read .npy: {e}") from e

    try:
        seq = np.load(io.BytesIO(npy), allow_pickle=False)
    except Exception as e:
        raise HTTPException(HttpStatusCode.InternalServerError, f"Failed to load preprocessed data .npy: {e}") from e
    if seq.ndim == 2:
        
        seq = np.expand_dims(seq, axis=0)

    if seq.ndim != 3 or seq.shape[1] != 60:
        raise HTTPException(
            HttpStatusCode.UnprocessableEntity,
            f"Expected 3-D array (n_seq, 60, features), got {seq.shape}"
        )

    n_seq, frames, feat_dim = seq.shape

    flat = seq.reshape(-1, feat_dim)
    try:
        flat_scaled = scaler.transform(flat)
    except Exception as e:
        raise HTTPException(HttpStatusCode.InternalServerError, f"Scaling failed: {e}") from e

    x_scaled = flat_scaled.reshape(n_seq, frames, feat_dim)

    # run inference
    try:
        preds = model.predict(x_scaled)
    except Exception as e:
        raise HTTPException(HttpStatusCode.InternalServerError, f"Inference failed: {e}") from e

    class_ids = preds.argmax(axis=1).tolist()
    return http_response_success(
        HttpStatusCode.Ok,
        "success",
        {
            "predicted_class": class_ids,
            "probabilities": preds.tolist()
        }
    )
