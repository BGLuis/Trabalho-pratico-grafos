from os import PathLike
from typing import Any, Callable, Dict, List

from requests import PreparedRequest
from requests.auth import AuthBase
from extractor.config import ExtractorConfig, TokenManager
from gql import Client, GraphQLRequest, gql
from gql.transport.requests import RequestsHTTPTransport
from gql.transport.exceptions import (
    TransportConnectionFailed,
    TransportQueryError,
    TransportServerError,
)
from time import sleep
from pathlib import Path
from datetime import datetime, timezone, timedelta

from utils import extract_from_key, log, CacheStore


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
        self.__base_url = config.base_url()
        self.__token_manager = TokenManager(config.auth_tokens())
        self.__transport = RequestsHTTPTransport(url=f"{self.__base_url}/graphql")
        self.__transport.auth = Auth(self.__token_manager.get_current_token())
        self.__client = Client(transport=self.__transport)
        self.__cache = CacheStore(f"{config.repo_owner()}_{config.repo_name()}")

    def __switch_to_token(self, token: str) -> None:
        self.__transport.auth = Auth(token)
        self.__client = Client(transport=self.__transport)
        log("Switched client to new token")

    def __load_query_from_file(self, path: PathLike) -> GraphQLRequest:
        with open(path, "r") as file:
            return gql(file.read())

    def __handle_rate_limit(self, log_tag: str) -> bool:
        current_token = self.__token_manager.get_current_token()

        reset_at = None
        try:
            rate_limit_query = self.__load_query_from_file(
                Path(__file__).parent / Path("queries/rate_limit.graphql")
            )
            rate_limit_response = self.__client.execute(rate_limit_query)
            log(f"[{log_tag}] Rate limit response: {rate_limit_response}")
            reset_at_str = rate_limit_response.get("rateLimit", {}).get("resetAt")

            if reset_at_str:
                reset_at = datetime.fromisoformat(reset_at_str.replace("Z", "+00:00"))
        except Exception as e:
            log(f"[{log_tag}] Error fetching rate limit info: {e}")
            reset_at = datetime.now(timezone.utc) + timedelta(hours=1)

        self.__token_manager.mark_rate_limited(current_token, reset_at)
        next_token, next_reset_at = self.__token_manager.get_next_available_token()

        if next_reset_at is None:
            log(f"[{log_tag}] Switching to another available token")
            self.__switch_to_token(next_token)
            return True
        else:
            now = datetime.now(timezone.utc)
            wait_seconds = max(1, (next_reset_at - now).total_seconds())
            log(
                f"[{log_tag}] All tokens rate limited. Waiting {wait_seconds:.0f} seconds until {next_reset_at}"
            )
            self.__switch_to_token(next_token)
            sleep(wait_seconds + 1)
            return False

    def __execute_query_with_retry(
        self, log_tag: str, query: GraphQLRequest, use_cache: bool = True
    ) -> Dict[str, Any]:
        cache_key = (
            self.__cache.get_cache_key(query, query.variable_values)
            if use_cache
            else None
        )

        if use_cache and cache_key:
            cached_result = self.__cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        while True:
            try:
                result = self.__client.execute(query)
                if use_cache and cache_key:
                    self.__cache.set(cache_key, result)
                return result
            except TransportServerError as e:
                log(f"[{log_tag}] Server error: {e}")
                sleep(0.5)
            except TransportConnectionFailed as e:
                log(f"[{log_tag}] Connection failed: {e}")
                sleep(0.5)
            except TransportQueryError as e:
                if e.errors is not None:
                    err = e.errors[0]
                    if "type" in err and err["type"] == "RATE_LIMIT":
                        switched = self.__handle_rate_limit(log_tag)
                        if switched:
                            log(f"[{log_tag}] Retrying with new token")
                        continue
                raise e

    def __calculate_missing(self, total_count: int, current_count: int) -> int:
        return min(100, total_count - current_count)

    def clear_cache(self) -> None:
        self.__cache.clear()

    def __fetch_paginated(
        self,
        filename: str,
        extract_data: Callable[[Dict[str, Any]], Dict[str, Any]],
        extra_variables: Dict[str, Any] | None = None,
        initial_list: List[Any] = [],
    ) -> List[Any]:
        log(f"Fetching paginated data using {filename}.graphql")
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
            if (
                "missing" in query.variable_values
                and query.variable_values["missing"] <= 0
            ):
                return result
            log(f"[{filename}] Executing query with variables: {query.variable_values}")
            request = self.__execute_query_with_retry(filename, query)
            data = extract_data(request)
            result.extend(data["nodes"])
            page_info = data["pageInfo"]
            if page_info["endCursor"] is None:
                break
            query.variable_values["cursor"] = page_info["endCursor"]
            if "missing" in query.variable_values:
                query.variable_values["missing"] = self.__calculate_missing(
                    data["totalCount"], len(result)
                )
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
                config["missing"] = self.__calculate_missing(
                    current["totalCount"], len(current["nodes"])
                )

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
        log(f"Fetched {len(data)} pull requests")
        data = self.__resolve_missing(
            data,
            [
                PageResolver(
                    "pull_request_comments", "repository.pullRequest.comments"
                ),
                PageResolver("pull_request_reviews", "repository.pullRequest.reviews"),
            ],
            lambda pr: {"number": pr["number"]},
        )
        log("Resolved missing data for pull requests")
        return data

    def fetch_issues(self):
        def extract_data(request: Dict[str, Any]) -> Dict[str, Any]:
            return request["repository"]["issues"]

        data = self.__fetch_paginated("issues", extract_data)
        log(f"Fetched {len(data)} issues")
        data = self.__resolve_missing(
            data,
            [
                PageResolver("issue_comments", "repository.issue.comments"),
                PageResolver("issue_timeline_items", "repository.issue.timelineItems"),
            ],
            lambda issue: {"number": issue["number"]},
        )
        log("Resolved missing data for issues")
        return data
