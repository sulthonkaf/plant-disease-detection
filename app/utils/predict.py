
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def predict_disease(image, model):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0) / 255.0
    pred = model.predict(image)
    class_idx = np.argmax(pred)
    confidence = float(np.max(pred))

    labels = ["Tomato_Blight", "Tomato_Healthy", "Tomato_Leaf_Mold"]

    return {
        "label": labels[class_idx],
        "confidence": confidence
    }
