

from fastapi import FastAPI
import uvicorn
from fastapi import File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import os
import io
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False, 
    allow_methods=["*"],
    
)
from  tensorflow.keras.models import load_model # type: ignore
MODEL = None
try:
    MODEL = tf.keras.models.load_model("1.keras", compile=False)
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)

potato =['Potato___Early_blight','Potato___Late_blight', 'Potato___healthy']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
   image = np.array(Image.open(io.BytesIO(data)))
   image = tf.image.resize(image, (256,256))
   image = image / 255.0

   return image
@app.post("/predict")
async def predict(
        file: UploadFile = File(...)): 
    if MODEL is None:
        return {"error": "Model not loaded properly"}

    try:
        image = read_file_as_image(await file.read())
        img_batch = np.expand_dims(image, axis=0)
        prediction = MODEL.predict(img_batch)

        prediction = prediction[0]
        prediction_class = potato[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        return {
            "class": prediction_class,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": str(e)}        
    print("Request hitted")     
    


   
if __name__== "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
