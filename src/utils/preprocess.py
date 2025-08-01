import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
import io
import zipfile
import tempfile
from typing import List
import numpy as np
import cv2

import keras
# from tensorflow.keras.applications import EfficientNetB7
# from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
from retinaface import RetinaFace

# def build_conv_base(img_height=600, img_width=600):
#     inputs = keras.layers.Input(shape=(img_height, img_width, 3))
#     base = keras.applications.EfficientNetB7(include_top=False, weights='imagenet', pooling='avg', input_shape=(img_height, img_width, 3))
#     x = keras.applications.efficientnet.preprocess_input(inputs)
#     outputs = base(x)
#     return keras.models.Model(inputs, outputs)

# 
# def build_conv_base(img_height=600, img_width=600):
#     tf.keras.backend.clear_session()
#     base = tf.keras.applications.EfficientNetB7(include_top=False, weights='imagenet', pooling='avg', input_shape=(img_height, img_width, 3))
#     inputs = tf.keras.layers.Input(shape=(img_height, img_width, 3))
#     x = tf.keras.applications.efficientnet.preprocess_input(inputs)
#     outputs = base(x)
#     return tf.keras.models.Model(inputs, outputs)

EFF_MODEL_PATH = "src/lib/EfficientNetB7_model.keras"

try:
    eff_model = keras.models.load_model(EFF_MODEL_PATH)
    print("Model EfficientNetB7 loaded successfully")
    print("Expected input shape:" , eff_model.input_shape)
    eff_model.summary()
except Exception as e:
    raise RuntimeError(f"Failed to load Keras model: {e}") from e

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
    
    temp_dir = tempfile.mkdtemp()
    imgs: List[str] = []
    for name in zip_file.namelist():
        if name.lower().endswith((".jpg", ".jpeg", ".png")):
            dest = os.path.join(temp_dir, os.path.basename(name))
            with open(dest, "wb") as f:
                f.write(zip_file.read(name))
            imgs.append(dest)
    zip_file.close()

    if not imgs:
        raise ValueError("No image files found in the ZIP")
    if len(imgs) < 60:
        raise ValueError("Minimum 60 images are required")
    
    batch: List[np.ndarray] = []
    for img_path in imgs:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[DEBUG] cv2.imread FAILED for {img_path}")
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (600, 600))
        batch.append(resized)
    
    if len(batch) < 60:
        raise ValueError("Minimum 60 detected images are required")
    if len(batch) > 60:
        batch = batch[:60]

    X = np.array(batch, dtype=np.float32)
    X = tf.keras.applications.efficientnet.preprocess_input(X)

    # eff_model = None
    # eff_model = build_conv_base(img_height=600, img_width=600)
    # eff_model.summary()
    print(">> conv base input_shape", eff_model.input_shape)
    features = eff_model.predict(X, batch_size=16, verbose=0)

    feat_dim = features.shape[1]
    seq = features.reshape(1, 60, feat_dim)

    buf = io.BytesIO()
    np.save(buf, seq)

    return buf.getvalue()
    # try:
    #     zip_file = zipfile.ZipFile(io.BytesIO(content))
    # except zipfile.BadZipFile:
    #     raise ValueError("Invalid ZIP archive")
    
    # temp_dir = tempfile.mkdtemp()
    # imgs = []

    # for name in zip_file.namelist():
    #     if name.lower().endswith((".jpg", ".jpeg", ".png")):
    #         dest = os.path.join(temp_dir, os.path.basename(name))
    #         with open(dest, "wb") as f:
    #             f.write(zip_file.read(name))
    #         imgs.append(dest)
    # zip_file.close()
    # # imgs = [f for f in zip_file.namelist() if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    # print(f"[DEBUG] Found {len(imgs)} image files:", imgs)

    # if not imgs:
    #     raise ValueError("No image files found in the ZIP")
    # if len(imgs) < 60:
    #     raise ValueError("Minimum 60 images are required")
    
    # batch: List[np.ndarray] = []
    # # for fn in imgs:
    # for idx, img_path in enumerate(imgs):
    #     img = cv2.imread(img_path)
    #     if img is None:
    #         print(f"[DEBUG] cv2.imread FAILED for {img_path}")
    #         continue
    #     # raw = zip_file.read(fn)
    #     # arr = np.frombuffer(raw, np.uint8)
    #     # img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    #     # if img is None:
    #     #     print(f"[DEBUG] cv2.imdecode FAILED for {fn}")
    #     #     continue

    #     try:
    #         detect = RetinaFace.detect_faces(img)
    #     except Exception as e:
    #         print(f"[DEBUG] RetinaFace failed on {img_path}: {e}")
    #         continue
    #     if not isinstance(detect, dict) or not detect:
    #         print(f"[DEBUG] No faces detected in {img_path}")
    #         continue

    #     # first_face = next(iter(detect.values()))
    #     key = list(detect.keys())[0]
    #     x1, y1, x2, y2 = detect[key]["facial_area"]
    #     # x1, y1, x2, y2 = first_face["facial_area"]
        
    #     crop = img[y1:y2, x1:x2]
    #     square = pad_to_square(crop)
    #     resized = cv2.resize(square, (600, 600))
    #     rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    #     batch.append(rgb)

    # # zip_file.close()

    # if not batch:
    #     raise ValueError("No valid images found in the ZIP")
    
    # n_crops = len(batch)
    # if n_crops < 60:
    #     raise ValueError("Minimum 60 detected images are required")

    # if n_crops > 60:
    #     batch = batch[:60]

    # X = np.array(batch, dtype=np.float32)
    # X = preprocess_input(X)

    # model = get_conv_base()
    # features = model.predict(X, batch_size=32, verbose=0)

    # feat_dim = features.shape[1]
    # seq = features.reshape(1, 60, feat_dim)

    # buf = io.BytesIO()
    # np.save(buf, seq)

    # return buf.getvalue()
    