import json
from collections.abc import Callable
from typing import Dict, Set

from lib.abstract_graph import AbstractGraph


class InteractionsDataFactory:
    @staticmethod
    def create() -> "InteractionsData":
        return InteractionsData()

    @staticmethod
    def build_closed_issues_graph(file_path: str) -> InteractionsData:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        interactions = InteractionsData()
        issues = data.get("issues", [])
        for issue in issues:
            author = issue.get("author", None)
            if not author:
                continue
            author_login = author["login"]
            closes = issue.get("timelineItems", [])
            for closed in closes:
                actor = closed.get("actor", None)
                if not actor:
                    continue
                actor_login = actor["login"]
                interactions.add_interaction(author_login, actor_login)

        return interactions

    @staticmethod
    def build_approve_merge_revision_pull_requests_graph(
        file_path: str,
    ) -> InteractionsData:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        interactions = InteractionsData()
        pull_requests = data.get("pullRequests", [])
        for pull_request in pull_requests:
            pull_request_author = pull_request.get("author", None)
            if not pull_request_author:
                continue
            author_login = pull_request_author["login"]
            reviews = pull_request.get("reviews", [])
            for review in reviews:
                review_author = review.get("author", None)
                if not review_author:
                    continue
                reviewer_login = review["author"]["login"]
                interactions.add_interaction(author_login, reviewer_login)
            merged_by = pull_request.get("mergedBy")
            if merged_by and merged_by.get("login", None):
                merger_login = merged_by["login"]
                interactions.add_interaction(author_login, merger_login)

        return interactions

    @staticmethod
    def build_comments_pull_requests_issues_graph(file_path: str) -> InteractionsData:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        interactions = InteractionsData()

        pull_requests = data.get("pullRequests", [])
        for pull_request in pull_requests:
            if not pull_request.get("author"):
                continue
            author_login = pull_request["author"]["login"]
            comments = pull_request.get("comments", [])
            for comment in comments:
                if comment.get("author"):
                    commenter_login = comment["author"]["login"]
                    interactions.add_interaction(commenter_login, author_login)

        issues = data.get("issues", [])
        for issue in issues:
            if not issue.get("author"):
                continue
            author_login = issue["author"]["login"]
            interactions.add_login(author_login)
            comments = issue.get("comments", [])
            for comment in comments:
                if comment.get("author"):
                    commenter_login = comment["author"]["login"]
                    interactions.add_interaction(commenter_login, author_login)

        return interactions


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
