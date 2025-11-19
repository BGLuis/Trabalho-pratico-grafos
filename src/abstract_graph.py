from abc import ABC, abstractmethod
from typing import List
from vertex import Vertex


class AbstractGraph(ABC):
    def __init__(self):
        self._vertices: List[Vertex] = []

    def _check_vertex_index(self, v: int):
        if not (0 <= v < len(self._vertices)):
            raise IndexError(
                f"Índice de vértice inválido: {v}. Deve estar entre 0 e {len(self._vertices) - 1}."
            )

    def set_vertex(self, v: Vertex):
        self._vertices.append(v)

    def get_vertex(self, i: int) -> Vertex:
        return self._vertices[i]

    @abstractmethod
    def get_vertex_count(self) -> int:
        pass

    @abstractmethod
    def get_edge_count(self) -> int:
        pass

    @abstractmethod
    def has_edge(self, u: int, v: int) -> bool:
        pass

    @abstractmethod
    def add_edge(self, u: int, v: int):
        pass

    @abstractmethod
    def remove_edge(self, u: int, v: int):
        pass

    @abstractmethod
    def is_sucessor(self, u: int, v: int) -> bool:
        pass

    @abstractmethod
    def is_predessor(self, u: int, v: int) -> bool:
        pass

    @abstractmethod
    def is_divergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        pass

    @abstractmethod
    def is_convergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        pass

    @abstractmethod
    def is_incident(self, u: int, v: int, x: int) -> bool:
        pass

    @abstractmethod
    def get_vertex_in_degree(self, u: int) -> int:
        pass

    @abstractmethod
    def get_vertex_out_degree(self, u: int) -> int:
        pass

    @abstractmethod
    def set_vertex_weight(self, v: int, w: float):
        pass

    @abstractmethod
    def get_vertex_weight(self, v: int) -> float:
        pass

    @abstractmethod
    def set_edge_weight(self, u: int, v: int, w: float):
        pass

    @abstractmethod
    def get_edge_weight(self, u: int, v: int) -> float:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass

    @abstractmethod
    def is_empty_graph(self) -> bool:
        pass

    @abstractmethod
    def is_complete_graph(self) -> bool:
        pass
