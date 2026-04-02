

from fastapi import FastAPI
import uvicorn
from fastapi import File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
        "http://127.0.0.1:3000"],
    allow_credentials=False, 
    allow_methods=["*"],
    
)
from  tensorflow.keras.models import load_model # type: ignore

try:
    MODEL = load_model(r"C:\Users\acer\Downloads\potato-disease\model\3.keras")
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)

potato =['Potato___Early_blight','Potato___Late_blight', 'Potato___healthy']

@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
   image = np.array(Image.open(BytesIO(data)))
   image = tf.image.resize(image, (256,256))
   image = image / 255.0

   return image
@app.post("/predict")
async def predict(
        file: UploadFile = File(...)):   
    print("Request hitted")     
    image = read_file_as_image(await file.read())
 #[256,256,3]the predict function doesn't take the entity as single image it takes multiple images 
    img_batch = np.expand_dims(image,axis=0)
    prediction = MODEL.predict(img_batch)
    print("Raw prediction:", prediction)
    print("Argmax index:", np.argmax(prediction))
    print("Max value:", np.max(prediction)) 



    prediction = prediction[0]
    prediction_class = potato[np.argmax(prediction)]
    confidence = float(np.max(prediction)*100)   
    print( f"Predicted class: {prediction_class}, Confidence: {confidence}")
    
    return{ 'class': prediction_class,
    'confidence': float(confidence)/float(100)
    }
if __name__== "__main__":
    uvicorn.run(app,host ='localhost' ,port = 8000)