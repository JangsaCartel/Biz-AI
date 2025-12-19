from typing import List, Optional

from pydantic import BaseModel, Field


# -----------------------------
# Request/Response 모델 정의
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
    score: int  # TF-IDF score (주차 전체 합산, int round)
    freq: int  # 등장 횟수(주차 전체, 자동 불용어 제거 후)


# 주간 분석 응답 모델
class WeeklyAnalysisResponse(BaseModel):
    weekLabel: str
    topKeywords: List[KeywordItem]
    wordcloudPngBase64: Optional[str] = None
