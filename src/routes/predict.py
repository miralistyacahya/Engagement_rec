from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
from joblib import load
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from utils.response import http_response_error, http_response_success
from utils.preprocess import preprocess_img
from constants.http_enum import HttpStatusCode

router = APIRouter(prefix="/predict")

# load Keras model
MODEL_PATH = "src/lib/model.keras"
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
        npy = await preprocess_img(content)
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
