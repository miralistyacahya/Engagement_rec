from tensorflow.keras.models import load_model

# 1) Load the old model
old = load_model("src/lib/model.keras")

# 2) Re-save it so the config uses "keras.models.functional" instead of "keras.src…"
old.save("src/lib/model_fixed.keras", save_format="keras")
