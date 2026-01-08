# Biz-AI/app/api/chat_nodes.py
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter

router = APIRouter(prefix="/api/ai/chat", tags=["ai-chat-nodes"])

# app/resources/chat_nodes.json
_NODES_PATH = Path(__file__).resolve().parents[1] / "resources" / "chat_nodes.json"
_nodes_cache: Optional[Dict[str, Any]] = None


def load_nodes() -> Dict[str, Any]:
    global _nodes_cache
    if _nodes_cache is None:
        _nodes_cache = json.loads(_NODES_PATH.read_text(encoding="utf-8"))
    return _nodes_cache


@router.get("/node")
def get_node(nodeId: str):
    nodes = load_nodes()
    node = nodes.get(nodeId)

    if not node:
        return {
            "type": "ANSWER_STATIC",
            "title": "안내",
            "content": f"노드를 찾을 수 없어요: {nodeId}",
            "afterOptions": [{"label": "메뉴로", "next": "ROOT"}],
            "nodeId": nodeId,
        }

    out = dict(node)
    out["nodeId"] = nodeId
    return out


@router.get("/tree")
def get_tree():
    return load_nodes()


@router.post("/nodes/refresh")
def refresh_nodes():
    global _nodes_cache
    _nodes_cache = None
    return {"ok": True, "refreshed": True}
