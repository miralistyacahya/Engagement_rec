import io
import zipfile
from typing import List
import numpy as np
import cv2
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf

# tf.shape = keras_shape
from retinaface import RetinaFace

def build_conv_base(img_height=600, img_width=600):
    base = EfficientNetB7(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_height, img_width, 3))
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = preprocess_input(inputs)
    outputs = base(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

conv_base = build_conv_base(img_height=600, img_width=600)

face_detector = RetinaFace(quality="normal")
def pad_to_square(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    diff = abs(h - w)
    if h > w:
        pad_l = diff // 2
        pad_r = diff - pad_l
        return cv2.copyMakeBorder(image, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    else:
        pad_t = diff // 2
        pad_b = diff - pad_t
        return cv2.copyMakeBorder(image, pad_t, pad_b, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    

async def preprocess_img(content: bytes) -> bytes:
    try:
        zip_file = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        raise ValueError("Invalid ZIP archive")
    
    imgs = [f for f in zip_file.namelist() if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    print(f"[DEBUG] Found {len(imgs)} image files:", imgs)

    if not imgs:
        raise ValueError("No image files found in the ZIP")
    if len(imgs) < 60:
        raise ValueError("Minimum 60 images are required")
    
    batch: List[np.ndarray] = []
    for fn in imgs:
        raw = zip_file.read(fn)
        arr = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[DEBUG] cv2.imdecode FAILED for {fn}")
            continue

        try:
            detect = face_detector.detect_faces(img)
        except Exception as e:
            print(f"[DEBUG] RetinaFace failed on {fn}: {e}")
            continue
        if not isinstance(detect, dict) or not detect:
            print(f"[DEBUG] No faces detected in {fn}")
            continue

        for face in detect.values():
            x1, y1, x2, y2 = face["facial_area"]
            crop = img[y1:y2, x1:x2]
            square = pad_to_square(crop)
            resized = cv2.resize(square, (600, 600))

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            batch.append(rgb)

    zip_file.close()

    if not batch:
        raise ValueError("No valid images found in the ZIP")
    
    n_crops = len(batch)
    if n_crops < 60:
        raise ValueError("Minimum 60 detected images are required")

    if n_crops > 60:
        batch = batch[:60]

    X = np.stack(batch, axis=0).astype(np.float32)
    X = preprocess_input(X)

    features = conv_base.predict(X, batch_size=32, verbose=0)

    feat_dim = features.shape[1]
    seq = features.reshape(1, 60, feat_dim)

    buf = io.BytesIO()
    np.save(buf, seq)

    return buf.getvalue()