from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path


class AbstractGraph(ABC):
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def from_gephi(cls, edges_file: Path, vertices_file: Path) -> "AbstractGraph":
        pass

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
    def is_successor(self, u: int, v: int) -> bool:
        pass

    @abstractmethod
    def is_predecessor(self, u: int, v: int) -> bool:
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
    def get_vertex_label(self, v: int) -> str:
        pass

    @abstractmethod
    def set_vertex_label(self, v: int, label: str):
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

    @abstractmethod
    def export_to_gephi(self, path: PathLike):
        pass
