from typing import Any, Dict


def extract_from_key(data: Dict[str, Any], key: str) -> Any:
    parts = key.split(".")
    current = data
    for part in parts:
        current = current[part]
    return current
