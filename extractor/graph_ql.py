from extractor.service import GithubService


def fetch_all(service: GithubService):
    return {
        "pullRequests": service.fetch_pull_requests(),
        "issues": service.fetch_issues(),
    }
