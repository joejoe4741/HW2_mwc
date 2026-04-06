from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
from deepface import DeepFace
import io
from PIL import Image

app = FastAPI(
    title="Age and Gender Prediction API", 
    description="MLOps Pipeline - API for Age & Gender Prediction using DeepFace",
    version="1.1.0"
)

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict_age(file: UploadFile = File(...)):
    """
    Upload an image file to predict the age of the person.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        # 1. Read binary image data
        contents = await file.read()
        
        # 2. Convert to numpy array via PIL
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_array = np.array(image)
        
        # 3. Convert RGB to BGR for OpenCV/DeepFace
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # 4. Predict using DeepFace (Age and Gender)
        # enforce_detection=False prevents crash if face is not fully visible
        results = DeepFace.analyze(img_path=img_bgr, actions=['age', 'gender'], enforce_detection=False)
        
        # Return results (DeepFace may return a list if multiple faces are found)
        if isinstance(results, list):
            age = results[0].get('age')
            gender = results[0].get('dominant_gender')
        else:
            age = results.get('age')
            gender = results.get('dominant_gender')
            
        return JSONResponse(content={
            "filename": file.filename, 
            "predicted_age": age, 
            "predicted_gender": gender,
            "status": "success"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Failed: {str(e)}")

# To run locally:
# uvicorn main:app --host 0.0.0.0 --port 8000 --reload
