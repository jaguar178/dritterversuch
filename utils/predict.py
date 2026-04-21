import os
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "keras_model.h5")

print("MODEL PATH:", MODEL_PATH)
print("EXISTS:", os.path.exists(MODEL_PATH))

model = load_model(MODEL_PATH)

model = load_model("model/keras_model.h5")

with open("model/labels.txt", "r") as f:
    class_names = f.read().splitlines()

def predict_image(image):
    image = image.resize((224, 224))
    image = np.asarray(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    label = class_names[index]

    return label, float(confidence)
