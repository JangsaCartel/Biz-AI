# Biz-AI/app/api/chat_llm.py
import json
import os
import random
import time
import threading
from typing import Any, Dict, Generator, List, Optional

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/ai/chat", tags=["ai-chat-llm"])

# -----------------------
# Gemini config
# -----------------------
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
GEMINI_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()

# 동시 스트림 제한(프로세스 단위)
GEMINI_MAX_CONCURRENCY = int(os.getenv("GEMINI_MAX_CONCURRENCY", "2"))

# 429 백오프(요청 시작 실패 시 재시도)
GEMINI_MAX_RETRY = int(os.getenv("GEMINI_MAX_RETRY", "3"))          # 3회 권장
GEMINI_BACKOFF_BASE_SEC = float(os.getenv("GEMINI_BACKOFF_BASE_SEC", "1.0"))  # 1.0
GEMINI_BACKOFF_JITTER_SEC = float(os.getenv("GEMINI_BACKOFF_JITTER_SEC", "0.4"))  # 0~0.4

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

# -----------------------
# Gemini client
# -----------------------
_GENAI_MODE = None
_client = None

def _init_gemini():
    global _GENAI_MODE, _client

    try:
        from google import genai as genai_new  # pip install google-genai
        _GENAI_MODE = "google-genai"
        _client = genai_new.Client(api_key=GEMINI_API_KEY)
        return
    except Exception:
        pass

_init_gemini()

# -----------------------
# concurrency limiter (sync generator라 threading.Semaphore가 가장 안전)
# -----------------------
_SEM = threading.BoundedSemaphore(max(1, GEMINI_MAX_CONCURRENCY))

def _extract_text(chunk: Any) -> str:
    if chunk is None:
        return ""

    t = getattr(chunk, "text", None)
    if isinstance(t, str) and t:
        return t

    if isinstance(chunk, dict):
        if isinstance(chunk.get("text"), str):
            return chunk["text"]
        try:
            c0 = (chunk.get("candidates") or [])[0]
            parts = (((c0.get("content") or {}).get("parts")) or [])
            if parts and isinstance(parts[0].get("text"), str):
                return parts[0]["text"]
        except Exception:
            return ""

    return ""

def _is_429(e: Exception) -> bool:
    # 최대한 SDK/버전 차이를 방어적으로 처리
    msg = str(e) or ""
    if "429" in msg or "TooManyRequests" in msg or "RESOURCE_EXHAUSTED" in msg or "ResourceExhausted" in msg:
        return True

    status_code = getattr(e, "status_code", None)
    if status_code == 429:
        return True

    resp = getattr(e, "response", None)
    if resp is not None and getattr(resp, "status_code", None) == 429:
        return True

    # google api core exceptions는 code()가 있는 경우가 있음
    code_attr = getattr(e, "code", None)
    try:
        if callable(code_attr):
            c = code_attr()
            # grpc StatusCode.RESOURCE_EXHAUSTED 등을 문자열로라도 비교
            if str(c).upper().find("RESOURCE_EXHAUSTED") >= 0:
                return True
        elif code_attr == 429:
            return True
    except Exception:
        pass

    return False

def _sse(obj: Dict[str, Any], event: Optional[str] = None) -> bytes:
    # event를 붙여도 FE가 data:만 읽어도 되고, 나중에 확장 가능
    head = f"event: {event}\n" if event else ""
    return (head + f"data: {json.dumps(obj, ensure_ascii=False)}\n\n").encode("utf-8")

# --- at top-level ---
_COOLDOWN_UNTIL = 0.0
_COOLDOWN_LOCK = threading.Lock()

def _now() -> float:
    return time.time()

def _cooldown_seconds() -> float:
    # 429가 뜨면 짧게라도 숨 고르기 (환경변수로 조절 가능)
    return float(os.getenv("GEMINI_COOLDOWN_SEC", "10"))

def _in_cooldown() -> float:
    with _COOLDOWN_LOCK:
        remain = _COOLDOWN_UNTIL - _now()
    return max(0.0, remain)

def _set_cooldown(sec: float):
    global _COOLDOWN_UNTIL
    with _COOLDOWN_LOCK:
        _COOLDOWN_UNTIL = max(_COOLDOWN_UNTIL, _now() + sec)

def _sleep_backoff(attempt: int):
    base = GEMINI_BACKOFF_BASE_SEC * (2 ** attempt)
    jitter = random.random() * GEMINI_BACKOFF_JITTER_SEC
    time.sleep(base + jitter)

def _open_stream(prompt: str):
    try:
        return _client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=prompt,
        )
    except Exception:
        return _client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=[prompt],
        )

def _stream_generate(prompt: str) -> Generator[str, None, None]:
    # 상위(gen)에서 처리하기 쉽게 RuntimeError로 던짐
    remain = _in_cooldown()
    if remain > 0:
        raise RuntimeError(f"429 cooldown active ({remain:.1f}s)")

    last_err = None
    for attempt in range(max(1, GEMINI_MAX_RETRY)):
        try:
            stream = _open_stream(prompt)
            for chunk in stream:
                text = _extract_text(chunk)
                if text:
                    yield text
            return
        except Exception as e:
            last_err = e
            if _is_429(e):
                # 429가 났으면 쿨다운 설정
                _set_cooldown(_cooldown_seconds())
                # 마지막 시도면 그대로 터뜨림
                if attempt >= GEMINI_MAX_RETRY - 1:
                    raise
                _sleep_backoff(attempt)
                continue
            raise

    raise last_err

# -----------------------
# Payload models
# -----------------------
class TrailItem(BaseModel):
    label: str = ""
    nodeId: Optional[str] = None

class AnswerStreamPayload(BaseModel):
    promptKey: str = Field(..., min_length=1)
    title: str = ""
    trail: List[TrailItem] = Field(default_factory=list)
    slotValues: Dict[str, Any] = Field(default_factory=dict)
    userText: str = ""

class LlmStreamPayload(BaseModel):
    text: str = Field(..., min_length=1)

# -----------------------
# Prompt builder
# -----------------------

def _common_format_rules_block() -> str:
    lines: List[str] = []
    lines.append("[작성 규칙]")
    lines.append("• 불확실한 사실을 단정하지 말 것.")
    lines.append("• 법/세무/대출은 '일반 정보'로 안내하고, 필요 시 전문가/공식기관 확인을 권고.")
    lines.append("• 발견 가능한 정보가 부족하면, 확인 질문은 최대 2개만.")
    lines.append("• 장문 금지. 표/체크리스트 중심.")
    lines.append("• 마크다운 문법(**, ###, *, |) 쓰지 말고 일반 텍스트로 작성.")
    lines.append("• 불릿은 반드시 '•'만 사용. ('-' 사용 금지)")
    lines.append("")
    lines.append("[출력 포맷]")
    lines.append("1) 핵심 요약(2~3줄)")
    lines.append("2) 체크리스트(불릿 5~8개)")
    lines.append("3) 추가로 확인하면 정확해지는 정보(선택)")
    lines.append("")
    lines.append("중요: 최종 출력은 반드시 위 1)~3)만 작성하고, 그 외 설명/서론/제목/여백을 추가하지 말 것.")
    return "\n".join(lines)

def _build_structured_prompt(payload: AnswerStreamPayload) -> str:
    lines: List[str] = []
    lines.append("너는 한국의 자영업자/소상공인을 돕는 실무형 AI 상담사다.")
    lines.append("사용자의 버튼 선택 흐름(의도)을 바탕으로, 바로 실행 가능한 답변을 작성하라.")
    lines.append("")

    lines.append("[사용자 의도(버튼 선택 흐름)]")
    if payload.trail:
        for i, t in enumerate(payload.trail, 1):
            label = (t.label or "").strip()
            if label:
                lines.append(f"• {label}")
    else:
        lines.append("(없음)")
    lines.append("")

    lines.append("[요청 키]")
    lines.append(payload.promptKey)
    lines.append("")

    if payload.slotValues:
        lines.append("[추가 정보]")
        for k, v in payload.slotValues.items():
            lines.append(f"• {k}: {v}")
        lines.append("")

    user_text = (payload.userText or "").strip()
    if user_text:
        lines.append("[사용자 추가 입력]")
        lines.append(user_text)
        lines.append("")
        lines.append("위 '사용자 추가 입력'을 반영해서, 이전 답변을 이어서 보완하라.")
        lines.append("")

    lines.append(_common_format_rules_block())
    return "\n".join(lines)

    
def _parse_json_body_with_fallback(raw: bytes) -> Dict[str, Any]:
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        pass

    for enc in ("cp949", "euc-kr"):
        try:
            return json.loads(raw.decode(enc))
        except Exception:
            continue

    raise HTTPException(
        status_code=400,
        detail="Invalid JSON or encoding. Send UTF-8 JSON (recommended).",
    )

# -----------------------
# ANSWER stream endpoint (버튼 선택 기반)
# -----------------------
@router.post("/answer/stream")
async def answer_stream(request: Request):
    raw = await request.body()
    data = _parse_json_body_with_fallback(raw)
    payload = AnswerStreamPayload.model_validate(data)

    prompt = _build_structured_prompt(payload)

    def gen() -> Generator[bytes, None, None]:
        # 동시 스트림 제한: 바로 실패시키고 FE에 안내(무한 대기 방지)
        if not _SEM.acquire(blocking=False):
            yield _sse(
                {"error": {"code": 503, "message": "요청이 많아 잠시 후 다시 시도해주세요."}, "done": True},
                event="error",
            )
            return

        try:
            for delta in _stream_generate(prompt):
                yield _sse({"delta": delta})
            yield _sse({"done": True})
        except Exception as e:
            if _is_429(e):
                yield _sse(
                    {
                        "error": {"code": 429, "message": "요청이 많아 일시적으로 제한됐어요. 잠시 후 다시 시도해주세요."},
                        "done": True,
                    },
                    event="error",
                )
            else:
                yield _sse(
                    {"error": {"code": 500, "message": str(e) or "AI 응답 생성 중 오류가 발생했어요."}, "done": True},
                    event="error",
                )
        finally:
            _SEM.release()

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)

# -----------------------
# Free LLM stream endpoint (자유 질문)
# -----------------------
@router.post("/llm/stream")
async def llm_stream(request: Request):
    raw = await request.body()
    data = _parse_json_body_with_fallback(raw)
    payload = LlmStreamPayload.model_validate(data)

    prompt_lines: List[str] = []
    prompt_lines.append("너는 한국의 자영업자/소상공인을 돕는 실무형 AI 상담사다.")
    prompt_lines.append("아래 질문에 대해, 아래 규칙/포맷을 반드시 지켜 답하라.")
    prompt_lines.append("")
    prompt_lines.append("[사용자 질문]")
    prompt_lines.append(payload.text.strip())
    prompt_lines.append("")
    prompt_lines.append(_common_format_rules_block())
    prompt = "\n".join(prompt_lines)

    def gen() -> Generator[bytes, None, None]:
        if not _SEM.acquire(blocking=False):
            yield _sse(
                {"error": {"code": 503, "message": "요청이 많아 잠시 후 다시 시도해주세요."}, "done": True},
                event="error",
            )
            return

        try:
            for delta in _stream_generate(prompt):
                yield _sse({"delta": delta})
            yield _sse({"done": True})
        except Exception as e:
            if _is_429(e):
                yield _sse(
                    {
                        "error": {"code": 429, "message": "요청이 많아 일시적으로 제한됐어요. 잠시 후 다시 시도해주세요."},
                        "done": True,
                    },
                    event="error",
                )
            else:
                yield _sse(
                    {"error": {"code": 500, "message": str(e) or "AI 응답 생성 중 오류가 발생했어요."}, "done": True},
                    event="error",
                )
        finally:
            _SEM.release()

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)
