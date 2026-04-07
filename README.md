# 💇 Hair Style Recommender MLOps API

DeepFace 기반의 **얼굴형 분석 & 헤어스타일 추천 API** 서비스입니다.  
GitHub에 코드를 Push하면 **Docker 이미지 빌드 → Docker Hub 업로드 → 로컬 서버 자동 배포**까지 CI/CD 파이프라인이 자동으로 수행됩니다.

## 📁 프로젝트 구조

```
age-prediction-api/
├── main.py                      # FastAPI 애플리케이션 (얼굴형 분석 & 헤어스타일 추천)
├── index.html                   # 웹 UI 프론트엔드
├── requirements.txt             # Python 의존성 패키지 목록
├── Dockerfile                   # 멀티 스테이지 Docker 빌드 파일
├── .dockerignore                # Docker 빌드 시 제외 파일 목록
├── .gitignore                   # Git 추적 제외 파일 목록
└── .github/
    └── workflows/
        └── ci.yml               # GitHub Actions CI/CD 워크플로우
```

## 🚀 실행 방법

### 방법 1: 로컬 실행 (Python)

1. Python 가상환경 생성 및 활성화
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

2. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

3. FastAPI 서버 실행
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 방법 2: Docker 실행

1. Docker 이미지 빌드
```bash
docker build -t hair-style-recommender .
```

2. 컨테이너 실행
```bash
docker run -d --name age-prediction-api --restart always -p 8000:8000 hair-style-recommender
```

## 🌐 API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET`  | `/`  | 웹 UI 페이지 (index.html) |
| `GET`  | `/docs` | Swagger UI (API 문서 & 테스트) |
| `POST` | `/predict` | 이미지 업로드 → 얼굴형 분석 & 헤어스타일 추천 |

### `/predict` 요청 예시 (curl)

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@photo.jpg"
```

### `/predict` 응답 예시

```json
{
  "filename": "photo.jpg",
  "predicted_age": 28,
  "predicted_gender": "Man",
  "face_shape": "oval",
  "face_shape_description": "균형잡힌 계란형 얼굴",
  "recommendations": [
    {
      "name": "레이어드 컷",
      "description": "자연스러운 레이어로 볼륨감을 살린 스타일",
      "image_url": "https://..."
    }
  ],
  "status": "success"
}
```

## ⚙️ CI/CD 파이프라인

GitHub에 코드를 Push하면 다음 과정이 **자동**으로 수행됩니다.

```
git push → GitHub Actions 트리거
  ├── 1. build-and-push (self-hosted)
  │     ├── 코드 체크아웃
  │     ├── Docker Hub 로그인
  │     ├── Docker 이미지 빌드 & Push (latest + commit SHA 태그)
  │     └── 빌드 캐시 활용
  └── 2. deploy (self-hosted, main 브랜치만)
        ├── 최신 이미지 Pull
        ├── 기존 컨테이너 중지 & 삭제
        ├── 새 컨테이너 실행 (포트 8000)
        └── 미사용 이미지 정리
```

### GitHub Secrets 설정 (필수)

GitHub 저장소의 `Settings > Secrets and variables > Actions`에서 아래 항목을 등록해야 합니다.

| Secret 이름 | 설명 |
|--------------|------|
| `DOCKERHUB_USERNAME` | Docker Hub 사용자명 |
| `DOCKERHUB_TOKEN` | Docker Hub Access Token |

## 🛠️ 기술 스택

- **Backend**: FastAPI + Uvicorn
- **AI Model**: DeepFace (나이, 성별, 얼굴형 분석)
- **Image Processing**: OpenCV (headless), Pillow, NumPy
- **ML Framework**: TensorFlow (tf-keras)
- **Containerization**: Docker (Multi-stage build)
- **CI/CD**: GitHub Actions (Self-hosted Runner)
- **Registry**: Docker Hub
```