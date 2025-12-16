import re
from typing import List, Set

from kiwipiepy import Kiwi

# Kiwi 형태소 분석기 초기화
KIWI = Kiwi()

# -----------------------------
# 텍스트 정리/토큰화
# HTML 태그 제거
# URL 제거
# -----------------------------
TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"http[s]?://\S+")

# "단어로 취급할 문자" 제한 (한글/영문/숫자만)
VALID_TOKEN_RE = re.compile(r"^[가-힣A-Za-z0-9]+$")

# 기본 불용어(최소) + 프로젝트 진행하면서 조금씩 확장
BASE_STOPWORDS: Set[str] = {
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
    "년",  # 의미 약한 단위명사가 자주 튀면 최소 불용어로 추가
}

# Kiwi 품사 태그 중 "명사"로 볼 것들
# - 일반명사/고유명사/의존명사/대명사/수사 등 포함
NOUN_TAGS: Set[str] = {"NNG", "NNP", "NNB", "NP", "NR"}


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_nouns(text: str) -> List[str]:
    """
    Kiwi로 형태소 분석 후, 명사만 추출해서 토큰 리스트로 만든다.
    - 1글자도 '명사'면 일단 포함 (예: '불')
    - 다만 노이즈는 DF 기반 자동 불용어에서 걸러지도록 설계
    """
    text = _clean_text(text)
    if not text:
        return []

    tokens: List[str] = []
    for t in KIWI.tokenize(text):
        form = (t.form or "").strip()
        tag = t.tag

        if tag not in NOUN_TAGS:
            continue

        # 토큰 정규화
        form = form.lower()

        # 한글/영문/숫자만 허용 (이모지/기호 제거)
        if not form or not VALID_TOKEN_RE.match(form):
            continue

        tokens.append(form)

    # 기본 불용어 제거
    tokens = [tok for tok in tokens if tok not in BASE_STOPWORDS]
    return tokens
