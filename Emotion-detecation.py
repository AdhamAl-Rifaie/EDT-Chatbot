import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from deepface import DeepFace
import uvicorn
from io import BytesIO
# import asyncio
# asyncio.apply()

app = FastAPI(title="Emotion Detection API", 
              description="API for detecting emotions in images using DeepFace")

def enhance_image(image_bytes):
    # Read image from bytes
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Apply CLAHE enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(img)
    
    # Convert back to BGR for DeepFace
    return cv2.cvtColor(enhanced_img, cv2.COLOR_GRAY2BGR)

@app.post("/detect-emotion/", response_class=JSONResponse)
async def detect_emotion(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    
    try:
        # Enhance the image
        enhanced_img = enhance_image(image_bytes)
        
        # Analyze emotion using DeepFace
        result = DeepFace.analyze(
            img_path=enhanced_img, 
            actions=['emotion'], 
            enforce_detection=False, 
            detector_backend='opencv'
        )
        
        # Return the dominant emotion and all emotion scores
        return {
            "dominant_emotion": result[0]['dominant_emotion'],
            "emotion_scores": result[0]['emotion']
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing image: {str(e)}"}
        )

@app.get("/")
def read_root():
    return {"message": "Welcome to the Emotion Detection API"}

