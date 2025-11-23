from typing import Any, Dict, List
from extractor.service import GithubService
from utils import submit_parallel_processes


def fetch_all(service: GithubService) -> Dict[str, List[Any]]:
    data = {
        "pullRequests": service.fetch_pull_requests,
        "issues": service.fetch_issues,
    }

    return submit_parallel_processes(data)
