from os import PathLike
from typing import Any, Callable, Dict, List

from requests import PreparedRequest
from requests.auth import AuthBase
from extractor.config import ExtractorConfig
from gql import Client, GraphQLRequest, gql
from gql.transport.requests import RequestsHTTPTransport
from pathlib import Path

from utils import extract_from_key


class Auth(AuthBase):
    def __init__(self, token: str):
        self.__token = token

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        r.headers["Authorization"] = f"Bearer {self.__token}"
        return r


class PageResolver:
    def __init__(self, filename: str, key: str) -> None:
        self.filename = filename
        self.key = key


class GithubService:
    def __init__(self, config: ExtractorConfig) -> None:
        self.__config = config
        self.__transport = RequestsHTTPTransport(url=f"{config.base_url()}/graphql")
        self.__transport.auth = Auth(self.__config.auth_token())
        self.__client = Client(transport=self.__transport)

    def __load_query_from_file(self, path: PathLike) -> GraphQLRequest:
        with open(path, "r") as file:
            return gql(file.read())

    def __fetch_paginated(
        self,
        filename: str,
        extract_data: Callable[[Dict[str, Any]], Dict[str, Any]],
        extra_variables: Dict[str, Any] | None = None,
        initial_list: List[Any] = [],
    ) -> List[Any]:
        query = self.__load_query_from_file(
            Path(__file__).parent / Path(f"queries/{filename}.graphql")
        )
        query.variable_values = {
            "owner": self.__config.repo_owner(),
            "name": self.__config.repo_name(),
        }
        if extra_variables is not None:
            query.variable_values |= extra_variables
        result = initial_list
        while True:
            request = self.__client.execute(query)
            data = extract_data(request)
            result.extend(data["nodes"])
            page_info = data["pageInfo"]
            if not page_info["hasNextPage"]:
                break
            query.variable_values["cursor"] = page_info["endCursor"]
            if "missing" in query.variable_values:
                query.variable_values["missing"] = data["totalCount"] - len(result)
        return result

    def __resolve_missing(
        self,
        data: List[Dict[str, Any]],
        resolvers: List[PageResolver],
        get_config: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        for each in data:
            for resolver in resolvers:
                last_key_part = resolver.key.split(".")[-1]
                current = each[last_key_part]
                page_info = current["pageInfo"]
                if not page_info["hasNextPage"]:
                    each[last_key_part] = each[last_key_part]["nodes"]
                    continue
                config = get_config(each)
                config["cursor"] = page_info["endCursor"]
                config["missing"] = current["totalCount"] - len(current["nodes"])

                def extract_data(data: Dict[str, Any]) -> Dict[str, Any]:
                    return extract_from_key(data, resolver.key)

                each[last_key_part] = self.__fetch_paginated(
                    resolver.filename, extract_data, config, current["nodes"]
                )
        return data

    def fetch_pull_requests(self):
        def extract_data(request: Dict[str, Any]) -> Dict[str, Any]:
            return request["repository"]["pullRequests"]

        data = self.__fetch_paginated("pull_requests", extract_data)
        return self.__resolve_missing(
            data,
            [
                PageResolver(
                    "pull_request_comments", "repository.pullRequest.comments"
                ),
                PageResolver("pull_request_reviews", "repository.pullRequest.reviews"),
            ],
            lambda pr: {"number": pr["number"]},
        )

    def fetch_issues(self):
        def extract_data(request: Dict[str, Any]) -> Dict[str, Any]:
            return request["repository"]["issues"]

        data = self.__fetch_paginated("issues", extract_data)
        return self.__resolve_missing(
            data,
            [
                PageResolver("issue_comments", "repository.issue.comments"),
                PageResolver("issue_timeline_items", "repository.issue.timelineItems"),
            ],
            lambda issue: {"number": issue["number"]},
        )
