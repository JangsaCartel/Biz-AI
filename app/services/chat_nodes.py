import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict


def _resource_path() -> Path:
    # app/services/chat_nodes.py 기준으로 app/resources/chat_nodes.json 찾기
    return Path(__file__).resolve().parents[1] / "resources" / "chat_nodes.json"


@lru_cache(maxsize=1)
def load_chat_nodes() -> Dict[str, Any]:
    """
    chat_nodes.json을 1회 로드해서 캐시.
    운영에서는 성능/안정성에 유리.
    개발 중 JSON을 자주 바꿀 거면 아래 refresh API로 캐시를 비울 수 있음.
    """
    path = _resource_path()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("chat_nodes.json root must be an object")

    if "ROOT" not in data:
        raise ValueError("chat_nodes.json must contain ROOT node")

    # 최소 검증: 각 노드는 dict, type 필수
    for node_id, node in data.items():
        if not isinstance(node, dict):
            raise ValueError(f"Node {node_id} must be an object")
        if "type" not in node:
            raise ValueError(f"Node {node_id} must have 'type'")

    return data


def get_node(node_id: str) -> Dict[str, Any]:
    nodes = load_chat_nodes()
    node = nodes.get(node_id)
    if node is None:
        raise KeyError(node_id)

    # FE가 쓰기 좋게 nodeId를 응답에 포함
    return {"nodeId": node_id, **node}


def refresh_chat_nodes_cache() -> None:
    # 개발 편의용: JSON 수정 후 캐시 비우기
    load_chat_nodes.cache_clear()
