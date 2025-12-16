import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Set, Tuple


# -----------------------------
# (B) 자동 불용어(DF 기반) + TF-IDF로 TopK 산출
# -----------------------------
def build_df_table(docs_tokens: List[List[str]]) -> Dict[str, int]:
    """
    DF(Document Frequency) 테이블: token -> 등장 문서 수
    """
    df: Dict[str, int] = defaultdict(int)
    for tokens in docs_tokens:
        seen = set(tokens)
        for tok in seen:
            df[tok] += 1
    return dict(df)


def build_auto_stopwords(
    df: Dict[str, int],
    n_docs: int,
    max_df_ratio: float,
    min_df: int,
) -> Set[str]:
    """
    자동 불용어:
    - 너무 많은 문서에 등장하는 단어(df_ratio >= max_df_ratio)
    - 너무 희귀한 단어(df < min_df) -> 노이즈 억제(핫키워드 품질↑)
    """
    auto: Set[str] = set()
    if n_docs <= 0:
        return auto

    for tok, d in df.items():
        ratio = d / n_docs
        if ratio >= max_df_ratio:
            auto.add(tok)
        if d < min_df:
            auto.add(tok)

    return auto


def compute_weekly_tfidf_topk(
    docs_tokens: List[List[str]],
    top_k: int,
    max_df_ratio: float,
    min_df: int,
) -> Tuple[List[Tuple[str, int, int]], Set[str]]:
    """
    TopK(핫 키워드) = TF-IDF 점수(주차 전체 합산)로 산출
    + freq(등장횟수)도 같이 제공

    - docs_tokens: 문서별 토큰 리스트(명사)
    - 반환:
      1) TopK 리스트: (token, score_as_int, freq_as_int)
      2) 실제 사용된 자동 불용어 set (디버깅/확장용)
    """
    n_docs = len(docs_tokens)
    if n_docs == 0:
        return [], set()

    df = build_df_table(docs_tokens)
    auto_stop = build_auto_stopwords(df=df, n_docs=n_docs, max_df_ratio=max_df_ratio, min_df=min_df)

    # 불용어 제거 후 문서 구성
    docs_filtered: List[List[str]] = []
    for tokens in docs_tokens:
        filtered = [t for t in tokens if t not in auto_stop]
        docs_filtered.append(filtered)

    # 불용어 제거 후 DF 다시 계산
    df2 = build_df_table(docs_filtered)

    # IDF 계산 (smoothing)
    # idf = log((N+1)/(df+1)) + 1
    idf: Dict[str, float] = {}
    for tok, d in df2.items():
        idf[tok] = math.log((n_docs + 1) / (d + 1)) + 1.0

    # TF-IDF 점수 합산(주차 점수) + freq 합산(주차 빈도)
    week_score: Dict[str, float] = defaultdict(float)
    week_freq: Dict[str, int] = defaultdict(int)

    for tokens in docs_filtered:
        if not tokens:
            continue

        tf = Counter(tokens)

        for tok, cnt in tf.items():
            # freq(등장 횟수)
            week_freq[tok] += cnt

            # sublinear tf: 1 + log(cnt)
            tf_weight = 1.0 + math.log(cnt)
            week_score[tok] += tf_weight * idf.get(tok, 0.0)

    if not week_score:
        return [], auto_stop

    # TopK
    sorted_items = sorted(week_score.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # score는 int로 내려서 응답(보기 좋게)
    top: List[Tuple[str, int, int]] = []
    for tok, score in sorted_items:
        top.append((tok, int(round(score)), int(week_freq.get(tok, 0))))

    return top, auto_stop


# -----------------------------
# 자동 파라미터 튜닝 (posts length 기반)
# 환경변수를 주면 그 값이 우선, 없으면 posts 수 기준 자동으로 결정
# -----------------------------
def _get_int_env(name: str) -> Optional[int]:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return None
    try:
        return int(val)
    except ValueError:
        return None


def _get_float_env(name: str) -> Optional[float]:
    val = os.getenv(name)
    if val is None or val.strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None

def choose_auto_stopword_params(n_posts: int) -> Tuple[float, int]:
    """
    posts 개수(n_posts)에 따라 자동 불용어 파라미터를 유연하게 조정한다.

    - max_df_ratio: 너무 많은 글에서 나오는 배경어 제거 기준
    - min_df: 너무 희귀한 단어(노이즈) 제거 기준

    환경변수로 강제 고정 가능:
    - AUTO_STOPWORD_MAX_DF_RATIO
    - AUTO_STOPWORD_MIN_DF
    """
    env_max_df_ratio = _get_float_env("AUTO_STOPWORD_MAX_DF_RATIO")
    env_min_df = _get_int_env("AUTO_STOPWORD_MIN_DF")

    # 환경변수로 값이 들어오면 우선 사용
    if env_max_df_ratio is not None and env_min_df is not None:
        return env_max_df_ratio, env_min_df

    # n_posts 기준 자동 결정
    # - 글이 적으면 너무 빡세게 걸러서 키워드가 비는 경우가 생김 → min_df 낮춤 / max_df_ratio 높임
    # - 글이 많으면 노이즈가 늘어남 → min_df 올림 / max_df_ratio는 기본 수준 유지
    if n_posts < 10:
        max_df_ratio = 0.95
        min_df = 1
    elif n_posts < 30:
        max_df_ratio = 0.90
        min_df = 2
    elif n_posts < 100:
        max_df_ratio = 0.85
        min_df = 3
    else:
        max_df_ratio = 0.85
        min_df = 5

    # 일부만 환경변수로 override 하고 싶은 경우 처리
    if env_max_df_ratio is not None:
        max_df_ratio = env_max_df_ratio
    if env_min_df is not None:
        min_df = env_min_df

    return max_df_ratio, min_df