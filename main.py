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
    "Man": {
        "oval": {
            "face_shape": "계란형",
            "description": "균형잡힌 계란형 얼굴",
            "recommendations": [
                {"name": "투블럭 컷", "description": "옆면을 짧게 치고 윗머리에 볼륨을 준 남성 정석 스타일", "image_url": "https://images.unsplash.com/photo-1503951914875-452162b0f3f1?w=400"},
                {"name": "텍스처드 크롭", "description": "자연스러운 텍스처로 세련된 느낌을 주는 스타일", "image_url": "https://images.unsplash.com/photo-1534030347209-467a5b0ad3e6?w=400"},
                {"name": "슬릭백", "description": "뒤로 넘긴 깔끔하고 단정한 스타일", "image_url": "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=400"},
            ]
        },
        "round": {
            "face_shape": "둥근형",
            "description": "부드러운 둥근 얼굴",
            "recommendations": [
                {"name": "하이 페이드", "description": "옆면을 높게 쳐서 얼굴을 갸름하게 보이게 하는 스타일", "image_url": "https://images.unsplash.com/photo-1521119989659-a83eee488004?w=400"},
                {"name": "포마드 사이드파트", "description": "가르마를 옆으로 나눠 얼굴에 각도감을 부여", "image_url": "https://images.unsplash.com/photo-1504194921103-f8b80cadd5e4?w=400"},
                {"name": "프린지 컷", "description": "앞머리를 내려 얼굴을 길어 보이게 하는 스타일", "image_url": "https://images.unsplash.com/photo-1519699047748-de8e457a634e?w=400"},
            ]
        },
        "square": {
            "face_shape": "각진형",
            "description": "강인한 각진 얼굴",
            "recommendations": [
                {"name": "레이어드 컷", "description": "자연스러운 레이어로 각진 얼굴을 부드럽게 연출", "image_url": "https://images.unsplash.com/photo-1522337360788-8b13dee7a37e?w=400"},
                {"name": "커튼 뱅스", "description": "앞머리를 양쪽으로 나눠 이마와 턱선을 자연스럽게 커버", "image_url": "https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?w=400"},
                {"name": "소프트 웨이브 미디엄", "description": "웨이브로 각진 턱선을 부드럽게 커버하는 스타일", "image_url": "https://images.unsplash.com/photo-1487412947147-5cebf100ffc2?w=400"},
            ]
        },
        "heart": {
            "face_shape": "하트형",
            "description": "이마가 넓은 하트형 얼굴",
            "recommendations": [
                {"name": "사이드 스윕", "description": "옆으로 흘러내리는 스타일로 넓은 이마를 자연스럽게 커버", "image_url": "https://images.unsplash.com/photo-1502823403499-6ccfcf4fb453?w=400"},
                {"name": "미디엄 레이어드", "description": "중간 길이의 레이어로 하관에 볼륨감을 부여", "image_url": "https://images.unsplash.com/photo-1500917293891-ef795e70e1f6?w=400"},
                {"name": "앞머리 있는 투블럭", "description": "앞머리로 이마를 커버하고 옆면은 깔끔하게", "image_url": "https://images.unsplash.com/photo-1534030347209-467a5b0ad3e6?w=400"},
            ]
        }
    },
    "Woman": {
        "oval": {
            "face_shape": "계란형",
            "description": "균형잡힌 계란형 얼굴",
            "recommendations": [
                {"name": "레이어드 롱 헤어", "description": "자연스러운 레이어로 볼륨감을 살린 여성스러운 스타일", "image_url": "https://images.unsplash.com/photo-1492106087820-71f1a00d2b11?w=400"},
                {"name": "히피 펌", "description": "자연스러운 웨이브로 활기차고 트렌디한 느낌", "image_url": "https://images.unsplash.com/photo-1487412947147-5cebf100ffc2?w=400"},
                {"name": "시스루 뱅스", "description": "가볍게 내려오는 앞머리로 청순한 느낌 연출", "image_url": "https://images.unsplash.com/photo-1508214751196-bcfd4ca60f91?w=400"},
            ]
        },
        "round": {
            "face_shape": "둥근형",
            "description": "부드러운 둥근 얼굴",
            "recommendations": [
                {"name": "레이어드 세미롱", "description": "레이어로 얼굴을 갸름하게 보이게 하는 스타일", "image_url": "https://images.unsplash.com/photo-1500917293891-ef795e70e1f6?w=400"},
                {"name": "긴 앞머리 스트레이트", "description": "긴 앞머리로 얼굴을 길어 보이게 연출", "image_url": "https://images.unsplash.com/photo-1519699047748-de8e457a634e?w=400"},
                {"name": "사이드파트 웨이브", "description": "옆가르마와 웨이브로 얼굴에 각도감 부여", "image_url": "https://images.unsplash.com/photo-1504194921103-f8b80cadd5e4?w=400"},
            ]
        },
        "square": {
            "face_shape": "각진형",
            "description": "강인한 각진 얼굴",
            "recommendations": [
                {"name": "소프트 웨이브 롱", "description": "부드러운 웨이브로 각진 턱선을 자연스럽게 커버", "image_url": "https://images.unsplash.com/photo-1502823403499-6ccfcf4fb453?w=400"},
                {"name": "커튼 뱅스 미디엄", "description": "양쪽으로 나뉜 앞머리로 이마와 얼굴형을 부드럽게", "image_url": "https://images.unsplash.com/photo-1487412947147-5cebf100ffc2?w=400"},
                {"name": "레이어드 밥컷", "description": "턱선 아래 레이어드 밥으로 부드러운 인상 연출", "image_url": "https://images.unsplash.com/photo-1492106087820-71f1a00d2b11?w=400"},
            ]
        },
        "heart": {
            "face_shape": "하트형",
            "description": "이마가 넓은 하트형 얼굴",
            "recommendations": [
                {"name": "턱선 길이 밥컷", "description": "턱선에서 끝나는 밥컷으로 하관에 볼륨감을 부여", "image_url": "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=400"},
                {"name": "사이드 스윕 뱅스", "description": "옆으로 흘러내리는 앞머리로 넓은 이마를 커버", "image_url": "https://images.unsplash.com/photo-1521119989659-a83eee488004?w=400"},
                {"name": "볼륨 업 세미롱", "description": "아랫부분에 볼륨을 줘서 균형감을 형성", "image_url": "https://images.unsplash.com/photo-1534030347209-467a5b0ad3e6?w=400"},
            ]
        }
    }
}

def estimate_face_shape(age, gender):
    import random
    shapes = ["oval", "round", "square", "heart"]
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
        hairstyle_info = HAIRSTYLE_DATA[gender][face_shape]

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