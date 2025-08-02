import sys
import cv2
import numpy as np
import os
os.environ["DEEPFACE_LOG_LEVEL"] = "WARNING"
import json
import tempfile
from retinaface import RetinaFace

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
    
def main():
    try:
        paths = json.loads(sys.argv[1])
    except Exception as e:
        print(json.dumps({"error": f"Invalid input: {e}"}), file=sys.stderr)
        sys.exit(1)

    crops = []
    for img_path in paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"[DEBUG] cv2.imread FAILED for {img_path}", file=sys.stderr)
            continue

        try:
            faces = RetinaFace.detect_faces(img)
        except Exception as e:
            print(f"[DEBUG] RetinaFace.detect_faces FAILED for {img_path}: {e}", file=sys.stderr)
            continue
        
        if not isinstance(faces, dict) or not faces:
            print(f"[DEBUG] No face detected in {img_path}", file=sys.stderr)
            continue

        key = list(faces.keys())[0]
        x1, y1, x2, y2 = faces[key]['facial_area']
        crop = img[y1:y2, x1:x2]
        square = pad_to_square(crop)
        rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (600, 600))
        crops.append(resized)

        if len(crops) >= 60:
            break
    
    if len(crops) < 60:
        print(json.dumps({"error": "Minimum 60 detected images are required"}), file=sys.stderr)
        sys.exit(2)

    X = np.stack(crops[:60], axis=0)
    
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
        np.save(tmp, X)
        print(tmp.name)

if __name__ == "__main__":
    main()