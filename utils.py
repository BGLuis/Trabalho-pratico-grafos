from datetime import datetime
from typing import Any, Dict
import json
import hashlib
from pathlib import Path
import shutil


class CacheStore:
    def __init__(self, namespace: str) -> None:
        self.__cache_dir = Path(__file__).parent / ".cache" / namespace
        self.__cache_dir.mkdir(parents=True, exist_ok=True)
        log(f"Cache directory: {self.__cache_dir}")

    def get_cache_key(self, query: Any, variables: Dict[str, Any] | None = None) -> str:
        query_str = str(query)
        variables_str = json.dumps(variables or {}, sort_keys=True)
        cache_string = f"{query_str}::{variables_str}"
        return hashlib.sha256(cache_string.encode()).hexdigest()

    def get(self, cache_key: str) -> Dict[str, Any] | None:
        cache_file = self.__cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    log(f"Cache hit: {cache_key}")
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                log(f"Cache read error for {cache_key}: {e}")
                return None
        return None

    def set(self, cache_key: str, data: Dict[str, Any]) -> None:
        cache_file = self.__cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(data, f)
            log(f"Cached response: {cache_key}")
        except (IOError, TypeError) as e:
            log(f"Cache write error for {cache_key}: {e}")

    def clear(self) -> None:
        if self.__cache_dir.exists():
            shutil.rmtree(self.__cache_dir)
            self.__cache_dir.mkdir(parents=True, exist_ok=True)
            log(f"Cleared cache directory: {self.__cache_dir}")


def get_path_relative(path: Path) -> Path:
    return Path(__file__).parent / path


def extract_from_key(data: Dict[str, Any], key: str) -> Any:
    parts = key.split(".")
    current = data
    for part in parts:
        current = current[part]
    return current


def log(message: str) -> None:
    print(f"[{datetime.now().isoformat()}] {message}")
