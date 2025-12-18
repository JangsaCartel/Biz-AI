import base64
import io
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
from wordcloud import WordCloud


def _repo_root() -> Path:
    """
    repo root를 안정적으로 찾는다.
    - 현재 파일이 app/... 아래에 있다는 전제에서
      상위로 올라가며 'app' 디렉터리를 포함하는 지점을 root로 잡음
    """
    cur = Path(__file__).resolve()
    for p in cur.parents:
        if (p / "app").exists():
            return p
    # fallback
    return cur.parents[2]


def _resolve_path(p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (_repo_root() / path).resolve()


def _load_mask_array(mask_path: str) -> Optional[np.ndarray]:
    if not mask_path:
        return None

    path = _resolve_path(mask_path)
    if not path.exists():
        return None

    invert = (os.getenv("WORDCLOUD_MASK_INVERT") or "0").strip() == "1"

    img_rgba = Image.open(path).convert("RGBA")
    alpha = np.array(img_rgba.getchannel("A"), dtype=np.uint8)

    if int(alpha.min()) == 255 and int(alpha.max()) == 255:
        gray = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        mask = 255 - gray if invert else gray
        return mask

    mask = 255 - alpha if invert else alpha
    return mask


def make_wordcloud_base64_png(freq: Dict[str, int]) -> Optional[str]:
    if not freq:
        return None

    font_path_env = (os.getenv("WORDCLOUD_FONT_PATH") or "").strip()
    mask_path_env = (os.getenv("WORDCLOUD_MASK_PATH") or "").strip()

    transparent_bg = (os.getenv("WORDCLOUD_BG_TRANSPARENT") or "1").strip() == "1"

    font_path = None
    if font_path_env:
        fp = _resolve_path(font_path_env)
        if fp.exists():
            font_path = str(fp)

    mask = _load_mask_array(mask_path_env) if mask_path_env else None

    wc = WordCloud(
        font_path=font_path,
        mask=mask,
        width=1600,
        height=900,
        scale=2,
        background_color=None if transparent_bg else "white",
        mode="RGBA" if transparent_bg else "RGB",
        
        max_words=100,
        prefer_horizontal=0.95,
        collocations=False,
        min_font_size=30,
        max_font_size=170,
        relative_scaling=0.6,
        random_state=42,
    ).generate_from_frequencies(freq)

    img = wc.to_image()

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"
