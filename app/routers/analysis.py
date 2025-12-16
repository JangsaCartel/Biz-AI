from collections import Counter
from typing import List

from fastapi import APIRouter

from app.schemas.analysis import WeeklyAnalysisRequest, WeeklyAnalysisResponse
from app.services.text import tokenize_nouns
from app.services.weekly_trend import compute_weekly_tfidf_topk, choose_auto_stopword_params
from app.services.wordcloud import make_wordcloud_base64_png

router = APIRouter(prefix="/analysis", tags=["analysis"])


# -----------------------------
# API 엔드포인트
# -----------------------------
@router.post("/weekly", response_model=WeeklyAnalysisResponse)
def weekly_analysis(req: WeeklyAnalysisRequest):
    # posts 길이 기반으로 자동 튜닝(환경변수로 override 가능)
    max_df_ratio, min_df = choose_auto_stopword_params(len(req.posts))

    # 문서 구성 (title + content)
    docs = [f"{p.title}\n{p.content}" for p in req.posts]

    # 문서별 토큰화 (Kiwi 명사)
    docs_tokens = [tokenize_nouns(doc) for doc in docs]

    # TopK: TF-IDF(주차 합산)
    top, auto_stop = compute_weekly_tfidf_topk(
        docs_tokens=docs_tokens,
        top_k=req.topK,
        max_df_ratio=max_df_ratio,
        min_df=min_df,
    )

    # 워드클라우드: 빈도(freq)
    # - TopK랑 동일한 불용어 정책을 적용해서 "의미 없는 배경어"가 이미지에 크게 뜨는 걸 방지
    all_tokens: List[str] = []
    for tokens in docs_tokens:
        all_tokens.extend([t for t in tokens if t not in auto_stop])

    freq = dict(Counter(all_tokens))
    wc_b64 = make_wordcloud_base64_png(freq)

    return {
        "weekLabel": req.weekLabel,
        "topKeywords": [{"keyword": k, "count": c} for k, c in top],
        "wordcloudPngBase64": wc_b64,
    }
