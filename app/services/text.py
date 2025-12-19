import re
from typing import List, Set

from kiwipiepy import Kiwi
from kiwipiepy.utils import Stopwords

# Kiwi 형태소 분석기 초기화
KIWI = Kiwi()
KIWI_STOPWORDS = Stopwords()

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
    "년",
    "때",
    "전",
    "사람",
    "문장",
    "입장",
    "결과",
    "생각",
    "단순",
    "공통",
    "반복",
    "정리",
    "것",
}

# Kiwi 품사 태그 중 "명사"로 볼 것들
# - 일반명사/고유명사/의존명사/대명사/수사 등 포함
NOUN_TAGS: Set[str] = {"NNG", "NNP"}


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_nouns(text: str) -> List[str]:
    text = _clean_text(text)
    if not text:
        return []

    tokens: List[str] = []
    for t in KIWI.tokenize(text, stopwords=KIWI_STOPWORDS):  # ✅ Kiwi 내장 Stopwords 1차 제거
        form = (t.form or "").strip().lower()
        tag = t.tag

        if tag not in NOUN_TAGS:
            continue

        if not form or not VALID_TOKEN_RE.match(form):
            continue

        if form in BASE_STOPWORDS:
            continue

        tokens.append(form)

    return tokens
