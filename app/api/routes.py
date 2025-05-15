from fastapi import APIRouter, UploadFile, File
from utils.predict import predict_disease
from tensorflow.keras.models import load_model
from PIL import Image
import io
import numpy as np

router = APIRouter()

model = load_model("app/model/model_cnn_tanaman.h5")
class_names = ['Sehat', 'Leaf Spot', 'Blight', 'Powdery Mildew']

@router.get("/ping")
async def ping():
    return {"message": "API aktif dan siap menerima prediksi"}

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image_array = np.expand_dims(np.array(image) / 255.0, axis=0)

    prediction = model.predict(image_array)
    class_index = np.argmax(prediction[0])
    confidence = float(np.max(prediction[0]) * 100)
    label = class_names[class_index]

    return {
        "penyakit": label,
        "kepercayaan": round(confidence, 2),
        "saran": get_recommendation(label)
    }

def get_recommendation(penyakit):
    rekomendasi = {
        "Sehat": "Tanaman dalam kondisi baik. Lanjutkan perawatan rutin.",
        "Leaf Spot": "Gunakan fungisida berbahan tembaga dan pangkas daun yang terinfeksi.",
        "Blight": "Isolasi tanaman dan gunakan fungisida sistemik.",
        "Powdery Mildew": "Sediakan sirkulasi udara yang baik dan gunakan sulfur spray."
    }
    return rekomendasi.get(penyakit, "Periksa kembali kondisi tanaman Anda.")