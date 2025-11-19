from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from src.edge import Edge


class Vertex:
    def __init__(self, name: str, weight: float, edges: List[Edge]):
        self._edges = edges
        self._name = name
        self._weight = weight
        self._edges = edges

    def get_edges(self) -> List[Edge]:
        return self._edges

    def get_vertex_label(self) -> str:
        return self._name

    def get_vertex_weight(self) -> float:
        return self._weight

    def get_edge_count(self) -> int:
        return len(self._edges)

    def add_edge(self, new_edge):
        self._edges.append(new_edge)

    def set_edges(self, new_edges: List[Edge]):
        self._edges = new_edges

    def set_vertex_label(self, new_name: str):
        self._name = new_name

    def set_vertex_weight(self, new_weight: float):
        if new_weight < 0:
            raise ValueError("O peso do vértice não pode ser negativo.")
        self._weight = new_weight
