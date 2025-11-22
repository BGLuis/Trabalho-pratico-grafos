import json
from src.graph_impl import Graph
from src.vertex import Vertex


class GraphBuilder:
    def __init__(self):
        self.mapper = {}
        self.graph = Graph()

    def _get_vertex_idx(self, login_name: str) -> int:
        if login_name not in self.mapper:
            v = Vertex(name=login_name, weight=1.0, edges=[])
            self.graph.vertices.append(v)
            new_index = self.get_graph().get_vertex_count() - 1
            self.mapper[login_name] = new_index
        return self.mapper[login_name]

    def _add_interaction(self, source_login: str, target_idx: int):
        source_idx = self._get_vertex_idx(source_login)
        if source_idx == target_idx:
            return
        if not self.graph.has_edge(source_idx, target_idx):
            self.graph.add_edge(source_idx, target_idx)

    def build_comments_pull_requests_issues_graph(self, file_path: str) -> None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Erro: Arquivo {file_path} não encontrado.")
            return

        pull_requests = data.get("pullRequests", [])
        for pull_request in pull_requests:
            if not pull_request.get("author"):
                continue
            author_login = pull_request["author"]["login"]
            target_idx = self._get_vertex_idx(author_login)
            comments = pull_request.get("comments", [])
            for comment in comments:
                if comment.get("author"):
                    commenter_login = comment["author"]["login"]
                    self._add_interaction(commenter_login, target_idx)

        issues = data.get("issues", [])
        for issue in issues:
            if not issue.get("author"):
                continue
            author_login = issue["author"]["login"]
            target_idx = self._get_vertex_idx(author_login)
            comments = issue.get("comments", [])
            for comment in comments:
                if comment.get("author"):
                    commenter_login = comment["author"]["login"]
                    self._add_interaction(commenter_login, target_idx)

    def build_approve_merge_revision_pull_requests_graph(self, file_path: str) -> None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Erro: Arquivo {file_path} não encontrado.")
            return

        pull_requests = data.get("pullRequests", [])
        for pull_request in pull_requests:
            if not pull_request.get("author"):
                continue
            author_login = pull_request["author"]["login"]
            target_idx = self._get_vertex_idx(author_login)
            reviews = pull_request.get("reviews", [])
            for review in reviews:
                if review.get("author"):
                    self._add_interaction(review["author"]["login"], target_idx)
            merged_by = pull_request.get("mergedBy")
            if merged_by and merged_by.get("login"):
                self._add_interaction(merged_by["login"], target_idx)

    def build_closed_issues_graph(self, file_path: str) -> None:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"Erro: Arquivo {file_path} não encontrado.")
            return
        issues = data.get("issues", [])
        for issue in issues:
            author = issue.get("author")
            if not author:
                continue
            author_login = author["login"]
            target_idx = self._get_vertex_idx(author_login)
            closes = issue.get("timelineItems", [])
            for closed in closes:
                actor = closed.get("actor")
                if actor:
                    self._add_interaction(actor["login"], target_idx)

    def get_graph(self) -> Graph:
        return self.graph
