import base64
import io
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from wordcloud import WordCloud

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


# -----------------------------
# 1) Request/Response 모델 정의
# BE가 보내는 게시글 1개 단위 데이터 규격.
# -----------------------------
class Post(BaseModel):
    postId: Optional[str] = None
    title: str = ""
    content: str = ""


class WeeklyAnalysisRequest(BaseModel):
    weekLabel: str = Field(..., description="예: 2025년 12월 2주차")
    posts: List[Post]
    topK: int = 10


# 응답 모델
# Top 키워드 1개 항목
class KeywordItem(BaseModel):
    keyword: str
    count: int


# 주간 분석 응답 모델
class WeeklyAnalysisResponse(BaseModel):
    weekLabel: str
    topKeywords: List[KeywordItem]
    wordcloudPngBase64: str  # data:image/png;base64,...


# -----------------------------
# 2) 키워드 추출 로직 (최소 버전)
# HTML 태그 제거
# URL 제거
# -----------------------------
TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"http[s]?://\S+")

# 불용어 리스트 (간단 예시)
STOPWORDS = {
    "합니다",
    "하는",
    "그리고",
    "그러나",
    "이번",
    "관련",
    "대한",
    "통해",
    "위해",
    "있는",
    "없는",
    "때문",
    "정도",
    "부분",
    "내용",
    "경우",
    "가능",
    "확인",
    "지원",
    "신청",
    "안내",
    "문의",
}


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> List[str]:
    # 한글/영문/숫자 단어(2글자 이상)만 추출
    tokens = re.findall(r"[가-힣A-Za-z0-9]{2,}", text)
    tokens = [t.lower() for t in tokens]
    tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens


def extract_top_keywords(texts: List[str], top_k: int = 10) -> List[Tuple[str, int]]:
    merged = " ".join(_clean_text(t) for t in texts if t)
    tokens = _tokenize(merged)
    counts = Counter(tokens)
    return counts.most_common(top_k)


# -----------------------------
# 3) 워드클라우드 생성 (base64 PNG)
# -----------------------------
def make_wordcloud_base64_png(freq: Dict[str, int]) -> str:
    font_path = os.getenv("WORDCLOUD_FONT_PATH") or None

    wc = WordCloud(
        width=900,
        height=500,
        background_color="white",
        font_path=font_path,
    ).generate_from_frequencies(freq)

    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# -----------------------------
# 4) API 엔드포인트
# -----------------------------
@app.post("/analysis/weekly", response_model=WeeklyAnalysisResponse)
def weekly_analysis(req: WeeklyAnalysisRequest):
    # 게시글 제목+본문을 다 합쳐서 집계
    texts = [f"{p.title}\n{p.content}" for p in req.posts]

    top = extract_top_keywords(texts, top_k=req.topK)
    freq = {k: c for k, c in top}

    wc_b64 = make_wordcloud_base64_png(freq)

    return {
        "weekLabel": req.weekLabel,
        "topKeywords": [{"keyword": k, "count": c} for k, c in top],
        "wordcloudPngBase64": wc_b64,
    }
