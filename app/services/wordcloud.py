import base64
import io
import os
from typing import Dict

from PIL import Image
from wordcloud import WordCloud


# -----------------------------
# 워드클라우드 (빈도 freq 기반)
# -----------------------------
def make_wordcloud_base64_png(freq: Dict[str, int]) -> str:
    """
    freq 기반 워드클라우드 생성 후 base64 PNG로 반환
    - freq가 비면 1x1 흰색 PNG를 반환
    """
    if not freq:
        img = Image.new("RGB", (1, 1), color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

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
