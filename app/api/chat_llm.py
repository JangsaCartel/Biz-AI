# Biz-AI/app/api/chat_llm.py
import json
import os
from typing import Any, Dict, Generator, List, Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from fastapi import Request, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/ai/chat", tags=["ai-chat-llm"])

# -----------------------
# Gemini config
# -----------------------
GEMINI_API_KEY = (os.getenv("GEMINI_API_KEY") or "").strip()
GEMINI_MODEL = (os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")


# -----------------------
# Gemini client (supports google-genai OR google-generativeai)
# -----------------------
_GENAI_MODE = None
_client = None
_model = None


def _init_gemini():
    global _GENAI_MODE, _client, _model

    # 1) Prefer google-genai (new)
    try:
        from google import genai as genai_new  # pip install google-genai

        _GENAI_MODE = "google-genai"
        _client = genai_new.Client(api_key=GEMINI_API_KEY)
        return
    except Exception:
        pass



_init_gemini()


def _extract_text(chunk: Any) -> str:
    """
    SDK별 chunk 형태가 조금씩 달라서 최대한 안전하게 텍스트만 뽑는다.
    """
    if chunk is None:
        return ""
    # most common
    t = getattr(chunk, "text", None)
    if isinstance(t, str) and t:
        return t

    # google-genai 쪽은 구조가 변동될 수 있어 방어적으로 처리
    # dict-like
    if isinstance(chunk, dict):
        if isinstance(chunk.get("text"), str):
            return chunk["text"]
        # candidates[0].content.parts[0].text 형태일 수도 있음
        try:
            c0 = (chunk.get("candidates") or [])[0]
            parts = (((c0.get("content") or {}).get("parts")) or [])
            if parts and isinstance(parts[0].get("text"), str):
                return parts[0]["text"]
        except Exception:
            return ""

    return ""


def _stream_generate(prompt: str) -> Generator[str, None, None]:
    """
    Gemini streaming generator: yields text deltas.
    """
    if _GENAI_MODE == "google-genai":
        # google-genai
        # generate_content_stream API는 버전별로 약간 다를 수 있어 try 2가지 방식으로 방어
        try:
            stream = _client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=prompt,
            )
        except Exception:
            # 일부 버전은 contents 대신 input 또는 content를 요구할 수 있어 폴백
            stream = _client.models.generate_content_stream(
                model=GEMINI_MODEL,
                contents=[prompt],
            )

        for chunk in stream:
            text = _extract_text(chunk)
            if text:
                yield text
        return

    # google-generativeai
    stream = _model.generate_content(prompt, stream=True)
    for chunk in stream:
        text = _extract_text(chunk)
        if text:
            yield text


def _sse(obj: Dict[str, Any]) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


# -----------------------
# Payload models
# -----------------------
class TrailItem(BaseModel):
    label: str = ""
    nodeId: Optional[str] = None


class AnswerStreamPayload(BaseModel):
    # 예: "POLICY_NEED", "LOAN_RECO", "TIPS_TREND" 등
    promptKey: str = Field(..., min_length=1)

    # 화면에 보여주고 싶은 타이틀(옵션)
    title: str = ""

    # 버튼 선택 흐름
    # 예: [{"label":"정책 관련","nodeId":"POLICY_ROOT"}, {"label":"자영업에 필요한 정책","nodeId":"POLICY_NEED"}]
    trail: List[TrailItem] = Field(default_factory=list)

    # 추후 확장(업종/지역 등)
    slotValues: Dict[str, Any] = Field(default_factory=dict)
    
    # 버튼 답변 이후 사용자가 입력한 후속 질문/답변
    userText: str = ""


class LlmStreamPayload(BaseModel):
    text: str = Field(..., min_length=1)


# -----------------------
# Prompt builder
# -----------------------
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
                lines.append(f"{i}. {label}")
    else:
        lines.append("(없음)")
    lines.append("")

    lines.append("[요청 키]")
    lines.append(payload.promptKey)
    lines.append("")

    if payload.slotValues:
        lines.append("[추가 정보]")
        for k, v in payload.slotValues.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        
    # FOLLOWUP 입력이 있으면 프롬프트에 포함
    user_text = (payload.userText or "").strip()
    if user_text:
        lines.append("[사용자 추가 입력]")
        lines.append(user_text)
        lines.append("")
        lines.append("위 '사용자 추가 입력'을 반영해서, 이전 답변을 이어서 보완하라.")
        lines.append("")

    # 출력 규칙(너가 원하는 톤/형식: 짧고 실무적)
    lines.append("[작성 규칙]")
    lines.append("- 불확실한 사실을 단정하지 말 것.")
    lines.append("- 법/세무/대출은 '일반 정보'로 안내하고, 필요 시 전문가/공식기관 확인을 권고.")
    lines.append("- 발견 가능한 정보가 부족하면, 확인 질문은 최대 2개만.")
    lines.append("- 장문 금지. 표/체크리스트 중심.")
    lines.append("- 마크다운 문법(**, ###, *, |) 쓰지 말고 일반 텍스트로 작성.")
    lines.append("- 마크다운 문법(**, ###, *, |) 쓰지 말고 일반 텍스트로 작성.")
    lines.append("- 불릿은 반드시 '•'만 사용.")
    lines.append("")
    lines.append("[출력 포맷]")
    lines.append("1) 핵심 요약(2~3줄)")
    lines.append("2) 체크리스트(불릿 5~8개)")
    lines.append("3) 추가로 확인하면 정확해지는 정보(선택)")
    lines.append("")
    lines.append("이제 위 포맷으로 답변을 작성하라.")
    return "\n".join(lines)


# -----------------------
# ✅ ANSWER stream endpoint (버튼 선택 기반)
# -----------------------
@router.post("/answer/stream")
async def answer_stream(request: Request):
    raw = await request.body()
    data = _parse_json_body_with_fallback(raw)

    # ✅ 여기서 기존 Pydantic 모델로 재검증
    payload = AnswerStreamPayload.model_validate(data)

    prompt = _build_structured_prompt(payload)

    def gen() -> Generator[bytes, None, None]:
        try:
            for delta in _stream_generate(prompt):
                yield _sse({"delta": delta})
            yield _sse({"done": True})
        except Exception as e:
            yield _sse({"done": True, "error": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


# -----------------------
# ✅ Free LLM stream endpoint (자유 질문)
# -----------------------
@router.post("/llm/stream")
async def llm_stream(request: Request):
    raw = await request.body()
    data = _parse_json_body_with_fallback(raw)

    payload = LlmStreamPayload.model_validate(data)
    prompt = (
        "규칙: 마크다운(**,###,* 등) 금지. 일반 텍스트로만, 목록은 '1) 2) 3)' 또는 '-' 로.\n"
        "질문: " + payload.text.strip()
    )

    def gen() -> Generator[bytes, None, None]:
        try:
            for delta in _stream_generate(prompt):
                yield _sse({"delta": delta})
            yield _sse({"done": True})
        except Exception as e:
            yield _sse({"done": True, "error": str(e)})

    return StreamingResponse(gen(), media_type="text/event-stream")


def _parse_json_body_with_fallback(raw: bytes) -> Dict[str, Any]:
    # 1) 정석: UTF-8
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        pass

    # 2) Windows 콘솔/구형 인코딩 폴백 (테스트 편의용)
    for enc in ("cp949", "euc-kr"):
        try:
            return json.loads(raw.decode(enc))
        except Exception:
            continue

    # 여기까지 오면 진짜 깨진 바디(유효 JSON 아님)
    raise HTTPException(
        status_code=400,
        detail="Invalid JSON or encoding. Send UTF-8 JSON (recommended).",
    )
