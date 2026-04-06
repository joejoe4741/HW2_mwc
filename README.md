# Age Prediction MLOps API

## 실행 방법 (How to run)

1. Python 가상환경 생성 및 활성화
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

2. 종속성 패키지 설치
```bash
pip install -r requirements.txt
```

3. FastAPI 서버 실행
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

4. API 테스트
- 브라우저에서 `http://127.0.0.1:8000/docs` 접속 (Swagger UI)
- `/predict` 엔드포인트에 이미지 업로드 테스트
