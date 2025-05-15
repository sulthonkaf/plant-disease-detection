
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("app/model/model_cnn_tanaman.h5")
class_names = ['Sehat', 'Leaf Spot', 'Blight', 'Powdery Mildew']

IMG_SIZE = (224, 224)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_array = preprocess_image(image_bytes)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction[0])]
    confidence = float(np.max(prediction[0]) * 100)

    return {
        "penyakit": predicted_class,
        "kepercayaan": round(confidence, 2),
        "saran": get_recommendation(predicted_class)
    }

def get_recommendation(penyakit):
    rekomendasi = {
        "Sehat": "Tanaman dalam kondisi baik. Lanjutkan perawatan rutin.",
        "Leaf Spot": "Gunakan fungisida berbahan tembaga dan pangkas daun yang terinfeksi.",
        "Blight": "Isolasi tanaman dan gunakan fungisida sistemik.",
        "Powdery Mildew": "Sediakan sirkulasi udara yang baik dan gunakan sulfur spray."
    }
    return rekomendasi.get(penyakit, "Periksa kembali kondisi tanaman Anda.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
