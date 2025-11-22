import json
from collections.abc import Callable
from os import PathLike
from typing import Dict, Set, List, Tuple, Any, Optional, Union

from lib.abstract_graph import AbstractGraph


LoginExtractor = Callable[[Any], Optional[str]]
InteractionExtractor = Callable[[Any], List[Tuple[str, str]]]


class ExtractorConfig:
    def __init__(
        self,
        key: str,
        logins: List[LoginExtractor],
        interactions: List[InteractionExtractor],
    ):
        self.__key = key
        self.__logins = logins
        self.__interactions = interactions

    @property
    def key(self) -> str:
        return self.__key

    @property
    def logins(self) -> List[LoginExtractor]:
        return self.__logins

    @property
    def interactions(self) -> List[InteractionExtractor]:
        return self.__interactions


class InteractionsDataFactory:
    @staticmethod
    def build_from_extractors(
        file_path: Union[str, PathLike],
        extractors: List[ExtractorConfig],
    ) -> "InteractionsData":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        interactions = InteractionsData()

        for extractor_config in extractors:
            items = data.get(extractor_config.key, [])
            for item in items:
                for extractor in extractor_config.logins:
                    login = extractor(item)
                    if login:
                        interactions.add_login(login)

                for extractor in extractor_config.interactions:
                    pairs = extractor(item)
                    for source, target in pairs:
                        interactions.add_interaction(source, target)

        return interactions

    @staticmethod
    def build_closed_issues_graph(file_path: str) -> "InteractionsData":
        """Build graph from issue closures."""

        def extract_author_login(issue: Dict) -> Optional[str]:
            author = issue.get("author")
            return author["login"] if author else None

        def extract_close_interactions(issue: Dict) -> List[Tuple[str, str]]:
            interactions = []
            author = issue.get("author")
            if not author:
                return interactions
            author_login = author["login"]

            closes = issue.get("timelineItems", [])
            for closed in closes:
                actor = closed.get("actor")
                if actor:
                    actor_login = actor["login"]
                    interactions.append((author_login, actor_login))

            return interactions

        extractors = [
            ExtractorConfig(
                key="issues",
                logins=[extract_author_login],
                interactions=[extract_close_interactions],
            )
        ]

        return InteractionsDataFactory.build_from_extractors(file_path, extractors)

    @staticmethod
    def build_approve_merge_revision_pull_requests_graph(
        file_path: str,
    ) -> "InteractionsData":
        def extract_author_login(pr: Dict) -> Optional[str]:
            author = pr.get("author")
            return author["login"] if author else None

        def extract_review_and_merge_interactions(pr: Dict) -> List[Tuple[str, str]]:
            interactions = []
            author = pr.get("author")
            if not author:
                return interactions
            author_login = author["login"]

            reviews = pr.get("reviews", [])
            for review in reviews:
                review_author = review.get("author")
                if review_author:
                    reviewer_login = review_author["login"]
                    interactions.append((author_login, reviewer_login))

            merged_by = pr.get("mergedBy")
            if merged_by and merged_by.get("login"):
                merger_login = merged_by["login"]
                interactions.append((author_login, merger_login))

            return interactions

        extractors = [
            ExtractorConfig(
                key="pullRequests",
                logins=[extract_author_login],
                interactions=[extract_review_and_merge_interactions],
            )
        ]

        return InteractionsDataFactory.build_from_extractors(file_path, extractors)

    @staticmethod
    def build_comments_pull_requests_issues_graph(file_path: str) -> "InteractionsData":
        def extract_pr_author_login(pr: Dict) -> Optional[str]:
            author = pr.get("author")
            return author["login"] if author else None

        def extract_issue_author_login(issue: Dict) -> Optional[str]:
            author = issue.get("author")
            return author["login"] if author else None

        def extract_pr_comment_interactions(pr: Dict) -> List[Tuple[str, str]]:
            interactions = []
            author = pr.get("author")
            if not author:
                return interactions
            author_login = author["login"]

            comments = pr.get("comments", [])
            for comment in comments:
                if comment.get("author"):
                    commenter_login = comment["author"]["login"]
                    interactions.append((commenter_login, author_login))

            return interactions

        def extract_issue_comment_interactions(issue: Dict) -> List[Tuple[str, str]]:
            interactions = []
            author = issue.get("author")
            if not author:
                return interactions
            author_login = author["login"]

            comments = issue.get("comments", [])
            for comment in comments:
                if comment.get("author"):
                    commenter_login = comment["author"]["login"]
                    interactions.append((commenter_login, author_login))

            return interactions

        extractors = [
            ExtractorConfig(
                key="pullRequests",
                logins=[extract_pr_author_login],
                interactions=[extract_pr_comment_interactions],
            ),
            ExtractorConfig(
                key="issues",
                logins=[extract_issue_author_login],
                interactions=[extract_issue_comment_interactions],
            ),
        ]

        return InteractionsDataFactory.build_from_extractors(file_path, extractors)


class InteractionsData:
    def __init__(self):
        self.__logins: Set[str] = set()
        self.__interactions: Dict[str, Set[str]] = {}

    def add_login(self, login: str):
        self.__logins.add(login)

    def add_interaction(self, source: str, target: str):
        self.__logins.add(source)
        self.__logins.add(target)
        if source not in self.__interactions:
            self.__interactions[source] = set()
        self.__interactions[source].add(target)

    @property
    def authors(self) -> Set[str]:
        return self.__logins

    @property
    def interactions(self) -> Dict[str, Set[str]]:
        return self.__interactions


class GraphParser:
    def __init__(self, graph_factory: Callable[[int], AbstractGraph]):
        self.__mapper: Dict[str, int] = {}
        self.__graph_factory = graph_factory
        self.__graph: AbstractGraph | None = None

    def __get_vertex_idx(self, login_name: str) -> int:
        if login_name not in self.__mapper:
            idx = self.__mapper[login_name] = len(self.__mapper)
            self.__graph.set_vertex_label(idx, login_name)
            self.__graph.set_vertex_weight(idx, 1.0)
        return self.__mapper[login_name]

    def __add_interaction(self, source_login: str, target_login: str):
        try:
            self.__graph.add_edge(
                self.__get_vertex_idx(source_login), self.__get_vertex_idx(target_login)
            )
        except IndexError:
            print(f"Erro ao adicionar aresta de {source_login} para {target_login}.")

    def get_graph(self, data: InteractionsData) -> AbstractGraph:
        authors = data.authors
        interactions = data.interactions
        self.__graph = self.__graph_factory(len(authors))
        for source, targets in interactions.items():
            for target in targets:
                self.__add_interaction(source, target)
        return self.__graph
