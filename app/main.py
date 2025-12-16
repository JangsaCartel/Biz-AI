from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.analysis import router as analysis_router

# .env 로컬 환경변수 로드
load_dotenv()

# FastAPI 앱 생성 + CORS 설정
app = FastAPI(title="Biz-AI")

# 라우트 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서버 상태 확인용 엔드포인트
@app.get("/health")
def health():
    return {"ok": True}


# Router 등록
app.include_router(analysis_router)
