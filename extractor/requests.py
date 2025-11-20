from typing import Any, Dict, List
from extractor.service import GithubService
from concurrent.futures import Future, ProcessPoolExecutor


def fetch_all(service: GithubService) -> Dict[str, List[Any]]:
    data: Dict[str, Future[List[Any]]] = dict()
    with ProcessPoolExecutor() as executor:
        data["pullRequests"] = executor.submit(service.fetch_pull_requests)
        data["issues"] = executor.submit(service.fetch_issues)

    result: Dict[str, List[Any]] = dict()
    for key, future in data.items():
        result[key] = future.result()

    return result
