from os import PathLike
from typing import Any, Callable, Dict, List

from requests import PreparedRequest
from requests.auth import AuthBase
from extractor.config import ExtractorConfig
from gql import Client, GraphQLRequest, gql
from gql.transport.requests import RequestsHTTPTransport
from pathlib import Path


class Auth(AuthBase):
    def __init__(self, token: str):
        self.__token = token

    def __call__(self, r: PreparedRequest) -> PreparedRequest:
        r.headers["Authorization"] = f"Bearer {self.__token}"
        return r


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
        self, filename: str, extract_data: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> List[Any]:
        query = self.__load_query_from_file(
            Path(__file__).parent / Path(f"queries/{filename}.graphql")
        )
        query.variable_values = {
            "owner": self.__config.repo_owner(),
            "repo": self.__config.repo_name(),
        }
        result = list()
        while True:
            request = self.__client.execute(query)
            data = extract_data(request)
            page_info = data["pageInfo"]
            result.extend(data["nodes"])
            if not page_info["hasNextPage"]:
                break
            query.variable_values["cursor"] = page_info["endCursor"]
        return result

    def fetch_pull_requests(self):
        def extract_data(request: Dict[str, Any]) -> Dict[str, Any]:
            return request["repository"]["pullRequests"]

        return self.__fetch_paginated("pull_requests", extract_data)

    def fetch_issues(self):
        def extract_data(request: Dict[str, Any]) -> Dict[str, Any]:
            return request["repository"]["issues"]

        return self.__fetch_paginated("issues", extract_data)
