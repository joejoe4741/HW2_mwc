from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
import cv2
import numpy as np
from deepface import DeepFace
import io
from PIL import Image

app = FastAPI(
    title="Hair Style Recommendation API",
    description="MLOps Pipeline - Face Analysis & Hair Style Recommendation",
    version="2.0.0"
)

# 얼굴형별 헤어스타일 추천 데이터
HAIRSTYLE_DATA = {
    "oval": {
        "face_shape": "oval",
        "description": "균형잡힌 계란형 얼굴",
        "recommendations": [
            {"name": "레이어드 컷", "description": "자연스러운 레이어로 볼륨감을 살린 스타일", "image_url": "https://images.unsplash.com/photo-1522337360788-8b13dee7a37e?w=400"},
            {"name": "볼드 언더컷", "description": "옆면을 짧게 치고 윗머리에 볼륨을 준 스타일", "image_url": "https://images.unsplash.com/photo-1503951914875-452162b0f3f1?w=400"},
            {"name": "웨이비 롱 헤어", "description": "자연스러운 웨이브로 여성스러움을 강조", "image_url": "https://images.unsplash.com/photo-1492106087820-71f1a00d2b11?w=400"},
        ]
    },
    "round": {
        "face_shape": "round",
        "description": "부드러운 둥근 얼굴",
        "recommendations": [
            {"name": "하이 탑 페이드", "description": "옆면을 짧게 쳐서 얼굴을 갸름하게 보이게", "image_url": "https://images.unsplash.com/photo-1521119989659-a83eee488004?w=400"},
            {"name": "사이드 파트", "description": "가르마를 옆으로 나눠 얼굴에 각도감 부여", "image_url": "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=400"},
            {"name": "긴 앞머리 스타일", "description": "앞머리를 길게 내려 얼굴을 길어 보이게", "image_url": "https://images.unsplash.com/photo-1504194921103-f8b80cadd5e4?w=400"},
        ]
    },
    "square": {
        "face_shape": "square",
        "description": "강인한 각진 얼굴",
        "recommendations": [
            {"name": "텍스처드 크롭", "description": "부드러운 텍스처로 각진 얼굴을 부드럽게", "image_url": "https://images.unsplash.com/photo-1534030347209-467a5b0ad3e6?w=400"},
            {"name": "커튼 뱅스", "description": "앞머리를 양쪽으로 나눠 얼굴 선을 부드럽게", "image_url": "https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?w=400"},
            {"name": "소프트 웨이브", "description": "웨이브로 각진 턱선을 자연스럽게 커버", "image_url": "https://images.unsplash.com/photo-1487412947147-5cebf100ffc2?w=400"},
        ]
    },
    "heart": {
        "face_shape": "heart",
        "description": "이마가 넓은 하트형 얼굴",
        "recommendations": [
            {"name": "사이드 스윕 뱅스", "description": "옆으로 흘러내리는 앞머리로 이마를 커버", "image_url": "https://images.unsplash.com/photo-1502823403499-6ccfcf4fb453?w=400"},
            {"name": "턱선 길이 밥", "description": "턱선에서 끝나는 밥컷으로 하관을 보강", "image_url": "https://images.unsplash.com/photo-1500917293891-ef795e70e1f6?w=400"},
            {"name": "볼륨 업 로우 스타일", "description": "아랫부분에 볼륨을 줘서 균형감 형성", "image_url": "https://images.unsplash.com/photo-1519699047748-de8e457a634e?w=400"},
        ]
    }
}

def estimate_face_shape(age, gender):
    """나이와 성별을 기반으로 얼굴형 추정 (실제로는 랜덤하게 할당)"""
    import random
    shapes = list(HAIRSTYLE_DATA.keys())
    # 시드를 나이+성별로 설정해서 같은 입력엔 같은 결과
    random.seed(age + (0 if gender == "Man" else 100))
    return random.choice(shapes)

@app.get("/", response_class=HTMLResponse)
def read_root():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        results = DeepFace.analyze(img_path=img_bgr, actions=['age', 'gender'], enforce_detection=False)

        if isinstance(results, list):
            age = results[0].get('age')
            gender = results[0].get('dominant_gender')
        else:
            age = results.get('age')
            gender = results.get('dominant_gender')

        face_shape = estimate_face_shape(age, gender)
        hairstyle_info = HAIRSTYLE_DATA[face_shape]

        return JSONResponse(content={
            "filename": file.filename,
            "predicted_age": age,
            "predicted_gender": gender,
            "face_shape": hairstyle_info["face_shape"],
            "face_shape_description": hairstyle_info["description"],
            "recommendations": hairstyle_info["recommendations"],
            "status": "success"
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Failed: {str(e)}")