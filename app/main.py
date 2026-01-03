import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat_llm import router as chat_llm_router
from app.api.chat_nodes import router as chat_nodes_router
from app.routers.analysis import router as analysis_router

# .env 로컬 환경변수 로드
load_dotenv()

# FastAPI 앱 생성 + CORS 설정
app = FastAPI(title="Biz-AI")

# CORS origins: 운영에서는 FE 도메인만 허용 권장
cors_origins_env = (os.getenv("BIZ_AI_CORS_ORIGINS") or "").strip()
if cors_origins_env:
    allow_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
else:
    allow_origins = ["*"]  # 개발용 기본값

# 라우트 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router 등록
app.include_router(analysis_router)
app.include_router(chat_nodes_router)
app.include_router(chat_llm_router)
