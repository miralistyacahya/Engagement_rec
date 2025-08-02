import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

from typing import List
import numpy as np
import subprocess
import sys
import json
import io

from retinaface import RetinaFace
    
def preprocess_img (img_paths: List[str]) -> List[np.ndarray]:
    try:
        result = subprocess.run(
            [sys.executable, 'src/scripts/crop_faces.py', json.dumps(img_paths)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            encoding='utf-8'
        )
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode(errors='replace')
        raise RuntimeError(f"Failed to crop images: {stderr_msg}") from e

    lines = result.stdout.strip().splitlines()
    npy_path = lines[-1].strip()
    # npy_path = result.stdout.strip()
    if not os.path.exists(npy_path):
        raise RuntimeError(f"Output file not found: {npy_path}")
    try:
        array = np.load(npy_path, allow_pickle=False)
        os.remove(npy_path)
        # print("[DEBUG] stdout size:", len(result.stdout))
        # with open("debug_out.npy", "wb") as f:
        #     f.write(result.stdout)
        # np.load("debug_out.npy", allow_pickle=False)
        # array = np.load(io.BytesIO(result.stdout), allow_pickle=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load numpy array: {e}") from e
    
    return list(array)