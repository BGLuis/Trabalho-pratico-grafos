from os import PathLike
from typing import List, Tuple
from pathlib import Path
import csv

from lib.abstract_graph import AbstractGraph
from lib.common import Vertex
from utils import log


class AdjacencyGraphList(AbstractGraph):
    def __init__(self, num_vertices):
        super().__init__()
        self.__vertices: List[Vertex] = [
            Vertex(label=f"V{i}", weight=1.0) for i in range(num_vertices)
        ]
        self.__adjacency_lists: List[List[Tuple[Vertex, float]]] = [
            [] for i in range(num_vertices)
        ]

    @classmethod
    def from_gephi(cls, edges_file: Path, vertices_file: Path) -> "AdjacencyGraphList":
        vertex_labels = {}
        vertex_weights = {}
        with open(vertices_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = int(row["Id"])
                vertex_labels[node_id] = row["Label"]
                vertex_weights[node_id] = float(row["Weight"])

        num_vertices = len(vertex_labels)
        graph = cls(num_vertices)

        for node_id in range(num_vertices):
            if node_id in vertex_labels:
                graph.set_vertex_label(node_id, vertex_labels[node_id])
                graph.set_vertex_weight(node_id, vertex_weights[node_id])

        with open(edges_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = int(row["Source"])
                target = int(row["Target"])
                weight = float(row["Weight"])
                graph.add_edge(source, target)
                graph.set_edge_weight(source, target, weight)

        return graph

    def get_vertex_count(self) -> int:
        return len(self.__adjacency_lists)

    def get_edge_count(self) -> int:
        total = 0
        for neighbors in self.__adjacency_lists:
            total += len(neighbors)
        return total

    def __check_vertex_index(self, u: int) -> None:
        if 0 > u or u >= len(self.__vertices):
            raise IndexError(f"Vertex index {u} out of bounds.")

    def __check_edge_index(self, u: int, v: int) -> None:
        self.__check_vertex_index(u)
        self.__check_vertex_index(v)

    def __get_non_weighted_neighbors(self, u: int) -> List[Vertex]:
        self.__check_vertex_index(u)
        return [neighbor for neighbor, _ in self.__adjacency_lists[u]]

    def __get_adjacency_list(self, u: int) -> List[Tuple[Vertex, float]]:
        self.__check_vertex_index(u)
        return self.__adjacency_lists[u]

    def __get_vertex(self, v: int) -> Vertex:
        self.__check_vertex_index(v)
        return self.__vertices[v]

    def has_edge(self, u: int, v: int) -> bool:
        neighbors = self.__get_non_weighted_neighbors(u)
        target_vertex = self.__get_vertex(v)
        return target_vertex in neighbors

    def add_edge(self, u: int, v: int):
        if u == v or self.has_edge(u, v):
            return

        self.__get_adjacency_list(u).append((self.__vertices[v], 1.0))

    def remove_edge(self, u: int, v: int):
        self.__check_edge_index(u, v)

        target_vertex = self.__vertices[v]
        adjacency_list = self.__get_adjacency_list(u)
        for i in range(len(adjacency_list)):
            neighbor, _ = adjacency_list[i]
            if neighbor == target_vertex:
                del adjacency_list[i]
                break

    def is_successor(self, u: int, v: int) -> bool:
        return self.has_edge(v, u)

    def is_predecessor(self, u: int, v: int) -> bool:
        return self.has_edge(u, v)

    def is_divergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        return (
            not (u1 != u2 or v1 == v2)
            and self.has_edge(u1, v1)
            and self.has_edge(u2, v2)
        )

    def is_convergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        return (
            not (u1 == u2 or v1 != v2)
            and self.has_edge(u1, v1)
            and self.has_edge(u2, v2)
        )

    def is_incident(self, u: int, v: int, x: int) -> bool:
        return self.has_edge(u, v) and ((x == u) or (x == v))

    def get_vertex_in_degree(self, u: int) -> int:
        self.__check_vertex_index(u)
        count = 0
        target_vertex = self.__vertices[u]

        for neighbors in self.__adjacency_lists:
            if target_vertex in neighbors:
                count += 1
        return count

    def get_vertex_out_degree(self, u: int) -> int:
        self.__check_vertex_index(u)
        return len(self.__adjacency_lists[u])

    def set_vertex_weight(self, v: int, w: float):
        self.__get_vertex(v).set_weight(w)

    def get_vertex_weight(self, v: int) -> float:
        return self.__get_vertex(v).get_weight()

    def set_edge_weight(self, u: int, v: int, w: float):
        adjacency_list = self.__get_adjacency_list(u)
        for i in range(len(adjacency_list)):
            neighbor, _ = adjacency_list[i]
            if neighbor == self.__get_vertex(v):
                adjacency_list[i] = (neighbor, w)
                break

    def set_vertex_label(self, v: int, label: str):
        self.__get_vertex(v).set_label(label)

    def get_vertex_label(self, v: int) -> str:
        return self.__get_vertex(v).get_label()

    def get_edge_weight(self, u: int, v: int) -> float:
        adjacency_list = self.__get_adjacency_list(u)
        for neighbor, weight in adjacency_list:
            if neighbor == self.__get_vertex(v):
                return weight

        assert False

    def is_connected(self) -> bool:
        if self.is_empty_graph():
            return False

        num_vertices = len(self.__adjacency_lists)
        visited = [False] * num_vertices
        queue = [0]
        visited[0] = True

        vertex_to_idx = {v: i for i, v in enumerate(self.__vertices)}

        while queue:
            u = queue.pop(0)
            for neighbor_vertex, _ in self.__get_adjacency_list(u):
                v = vertex_to_idx.get(neighbor_vertex)
                if v is not None and not visited[v]:
                    visited[v] = True
                    queue.append(v)

            for i in range(num_vertices):
                if not visited[i]:
                    current_vertex = self.__vertices[u]
                    neighbors = [neighbor for neighbor, _ in self.__adjacency_lists[i]]
                    if current_vertex in neighbors:
                        visited[i] = True
                        queue.append(i)

        return all(visited)

    def is_empty_graph(self) -> bool:
        return self.get_vertex_count() == 0

    def is_complete_graph(self) -> bool:
        n = self.get_vertex_count()
        if n == 0:
            return False
        expected_edges = n * (n - 1)
        return self.get_edge_count() == expected_edges

    def export_to_gephi(self, path: PathLike):
        base_path = Path(path)
        if not base_path.suffix:
            base_path = base_path / "graph.csv"

        file_vertexes = base_path.parent / f"{base_path.stem}_vertexes.csv"
        file_edges = base_path.parent / f"{base_path.stem}_edges.csv"

        if not base_path.parent.exists():
            base_path.parent.mkdir(parents=True, exist_ok=True)

        vertex_to_idx = {v: i for i, v in enumerate(self.__vertices)}

        try:
            with open(file_vertexes, "w", encoding="utf-8") as f:
                f.write("Id,Label,Weight\n")
                for i, vertex in enumerate(self.__vertices):
                    safe_label = vertex.get_label().replace(",", " ")
                    f.write(f"{i},{safe_label},{vertex.get_weight()}\n")

            with open(file_edges, "w", encoding="utf-8") as f:
                f.write("Source,Target,Weight,Type\n")
                for source_idx, neighbors in enumerate(self.__adjacency_lists):
                    for neighbor_vertex, weight in neighbors:
                        target_idx = vertex_to_idx.get(neighbor_vertex)
                        if target_idx is not None:
                            f.write(f"{source_idx},{target_idx},{weight},Directed\n")

            log(f"Graph exported to {file_vertexes} and {file_edges}")
        except IOError as e:
            log(f"Error saving CSV files: {e}")


class AdjacencyMatrixGraph(AbstractGraph):
    def __init__(self, num_vertices):
        super().__init__()
        self._num_vertices = num_vertices
        self._vertices: List[Vertex] = [
            Vertex(label=f"V{i}", weight=1.0) for i in range(num_vertices)
        ]
        self.matriz = [[0.0] * num_vertices for _ in range(num_vertices)]

    @classmethod
    def from_gephi(
        cls, edges_file: Path, vertices_file: Path
    ) -> "AdjacencyMatrixGraph":
        vertex_labels = {}
        vertex_weights = {}
        with open(vertices_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = int(row["Id"])
                vertex_labels[node_id] = row["Label"]
                vertex_weights[node_id] = float(row["Weight"])

        num_vertices = len(vertex_labels)
        graph = cls(num_vertices)

        for node_id in range(num_vertices):
            if node_id in vertex_labels:
                graph.set_vertex_label(node_id, vertex_labels[node_id])
                graph.set_vertex_weight(node_id, vertex_weights[node_id])

        with open(edges_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = int(row["Source"])
                target = int(row["Target"])
                weight = float(row["Weight"])
                graph.add_edge(source, target)
                graph.set_edge_weight(source, target, weight)

        return graph

    def __check_vertex_index(self, v: int) -> None:
        if 0 > v or v >= self._num_vertices:
            raise IndexError(f"Vertex index {v} out of bounds.")

    def __get_vertex(self, v: int) -> Vertex:
        self.__check_vertex_index(v)
        return self._vertices[v]

    def get_vertex_count(self) -> int:
        return len(self._vertices)

    def get_edge_count(self) -> int:
        count = 0
        for i in range(self._num_vertices):
            for j in range(self._num_vertices):
                if self.matriz[i][j] > 0:
                    count += 1
        return count

    def has_edge(self, u: int, v: int) -> bool:
        self.__check_vertex_index(u)
        self.__check_vertex_index(v)
        return self.matriz[u][v] > 0

    def add_edge(self, u: int, v: int):
        self.__check_vertex_index(u)
        self.__check_vertex_index(v)
        if u == v or self.has_edge(u, v):
            return
        self.matriz[u][v] = 1.0

    def remove_edge(self, u: int, v: int):
        self.__check_vertex_index(u)
        self.__check_vertex_index(v)
        self.matriz[u][v] = 0.0

    def is_successor(self, u: int, v: int) -> bool:
        return self.has_edge(v, u)

    def is_predecessor(self, u: int, v: int) -> bool:
        return self.has_edge(u, v)

    def is_divergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        return (
            not (u1 != u2 or v1 == v2)
            and self.has_edge(u1, v1)
            and self.has_edge(u2, v2)
        )

    def is_convergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        return (
            not (u1 == u2 or v1 != v2)
            and self.has_edge(u1, v1)
            and self.has_edge(u2, v2)
        )

    def is_incident(self, u: int, v: int, x: int) -> bool:
        return self.has_edge(u, v) and ((x == u) or (x == v))

    def get_vertex_in_degree(self, u: int) -> int:
        self.__check_vertex_index(u)
        count = 0
        for i in range(self._num_vertices):
            if self.matriz[i][u] > 0:
                count += 1
        return count

    def get_vertex_out_degree(self, u: int) -> int:
        self.__check_vertex_index(u)
        count = 0
        for j in range(self._num_vertices):
            if self.matriz[u][j] > 0:
                count += 1
        return count

    def set_vertex_weight(self, v: int, w: float):
        self.__get_vertex(v).set_weight(w)

    def get_vertex_weight(self, v: int) -> float:
        return self.__get_vertex(v).get_weight()

    def set_edge_weight(self, u: int, v: int, w: float):
        self.__check_vertex_index(u)
        self.__check_vertex_index(v)
        if self.has_edge(u, v):
            self.matriz[u][v] = w

    def get_edge_weight(self, u: int, v: int) -> float:
        self.__check_vertex_index(u)
        self.__check_vertex_index(v)
        return self.matriz[u][v]

    def is_connected(self) -> bool:
        if self.is_empty_graph():
            return False

        visited = [False] * self._num_vertices
        queue = [0]
        visited[0] = True
        count = 1

        while queue:
            u = queue.pop(0)
            for v in range(self._num_vertices):
                if (self.matriz[u][v] > 0 or self.matriz[v][u] > 0) and not visited[v]:
                    visited[v] = True
                    queue.append(v)
                    count += 1

        return count == self._num_vertices

    def is_empty_graph(self) -> bool:
        return self.get_vertex_count() == 0

    def is_complete_graph(self) -> bool:
        n = self.get_vertex_count()
        if n == 0:
            return False
        expected_edges = n * (n - 1)
        return self.get_edge_count() == expected_edges

    def get_vertex_label(self, v: int) -> str:
        return self.__get_vertex(v).get_label()

    def set_vertex_label(self, v: int, label: str):
        self.__get_vertex(v).set_label(label)

    def export_to_gephi(self, path: PathLike):
        base_path = Path(path)
        if not base_path.suffix:
            base_path = base_path / "graph.csv"

        file_vertexes = base_path.parent / f"{base_path.stem}_vertexes.csv"
        file_edges = base_path.parent / f"{base_path.stem}_edges.csv"

        if not base_path.parent.exists():
            base_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_vertexes, "w", encoding="utf-8") as f:
                f.write("Id,Label,Weight\n")
                for i, vertex in enumerate(self._vertices):
                    safe_label = vertex.get_label().replace(",", " ")
                    f.write(f"{i},{safe_label},{vertex.get_weight()}\n")

            with open(file_edges, "w", encoding="utf-8") as f:
                f.write("Source,Target,Weight,Type\n")
                for i in range(self._num_vertices):
                    for j in range(self._num_vertices):
                        if self.matriz[i][j] > 0:
                            f.write(f"{i},{j},{self.matriz[i][j]},Directed\n")

            log(f"Graph exported to {file_vertexes} and {file_edges}")
        except IOError as e:
            log(f"Error saving CSV files: {e}")
