# import inspect

# from tensorflow.keras.applications.efficientnet import EfficientNetB7

# print(inspect.getfile(EfficientNetB7))

# import keras

# def build_conv_base(img_height=600, img_width=600):
#     inputs = keras.layers.Input(shape=(img_height, img_width, 3))
#     base = keras.applications.EfficientNetB7(include_top=False, weights='imagenet', pooling='avg', input_shape=(img_height, img_width, 3))
#     x = keras.applications.efficientnet.preprocess_input(inputs)
#     outputs = base(x)
#     return keras.models.Model(inputs, outputs)

# model = build_conv_base()
# model.save("src/lib/EfficientNetB7_model.keras")

import inspect
import cv2

from retinaface import RetinaFace
# print(inspect.getfile(RetinaFace))
img = cv2.imread("src/lib/test2.jpg")
if img is None:
    print("cv2 imread failed")
try:
    face = RetinaFace.detect_faces(img)
except Exception as e:
    print('detect face failed:', e)
if not face:
    print("face not detected")

key = list(face.keys())[0]
x1, y1, x2, y2 = face[key]['facial_area']
crop = img[y1:y2, x1:x2]
print(crop.shape)
