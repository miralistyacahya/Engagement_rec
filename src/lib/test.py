# import inspect

# from tensorflow.keras.applications.efficientnet import EfficientNetB7

# print(inspect.getfile(EfficientNetB7))
# from tensorflow.keras.applications.efficientnet import EfficientNetB7
# from tensorflow.keras.layers import Input

# # Bangun model dengan 3 saluran
# inputs = Input(shape=(600, 600, 3))
# model = EfficientNetB7(
#     include_top=False,
#     weights='imagenet',
#     input_tensor=inputs
# )

# print("INPUT SHAPE:", model.input_shape)
# print("STEM KERNEL SHAPE:", model.layers[1].weights[0].shape)
import keras

def build_conv_base(img_height=600, img_width=600):
    inputs = keras.layers.Input(shape=(img_height, img_width, 3))
    base = keras.applications.EfficientNetB7(include_top=False, weights='imagenet', pooling='avg', input_shape=(img_height, img_width, 3))
    x = keras.applications.efficientnet.preprocess_input(inputs)
    outputs = base(x)
    return keras.models.Model(inputs, outputs)

model = build_conv_base()
model.save("src/lib/EfficientNetB7_model.keras")