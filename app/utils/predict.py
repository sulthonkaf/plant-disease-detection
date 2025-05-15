import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

def predict_disease(image, model):
    """
    Melakukan prediksi penyakit tanaman dari gambar input.

    Args:
        image (PIL.Image): Gambar tanaman dalam format RGB.
        model (keras.Model): Model CNN yang sudah dilatih.

    Returns:
        dict: Label penyakit dan confidence score prediksi.
    """
    # Resize dan preprocess gambar
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    # Prediksi
    pred = model.predict(image)
    class_idx = np.argmax(pred[0])
    confidence = float(np.max(pred[0]) * 100)

    # Label (disesuaikan dengan pelatihan model)
    labels = ["Sehat", "Leaf Spot", "Blight", "Powdery Mildew"]

    return {
        "label": labels[class_idx],
        "confidence": round(confidence, 2)
    }