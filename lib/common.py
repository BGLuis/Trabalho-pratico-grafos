from typing import List


class Edge:
    def __init__(self, source: Vertex, target: Vertex):
        self._weight = 1
        self._source: Vertex = source
        self._target: Vertex = target

    def get_weight(self) -> float:
        return self._weight

    def get_source(self) -> Vertex:
        return self._source

    def get_target(self) -> Vertex:
        return self._target

    def set_weight(self, new_weight: float):
        if new_weight < 0:
            raise ValueError("O peso da aresta não pode ser negativo neste modelo.")
        self._weight = new_weight

    def set_source(self, source: Vertex):
        if source.get_vertex_weight() < 0:
            raise ValueError("O índice da origem deve ser não-negativo.")
        self._source = source

    def set_target(self, target: Vertex):
        if target.get_vertex_weight() < 0:
            raise ValueError("O índice do destino deve ser não-negativo.")
        self._target = target


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

    def has_target(self, target: Vertex) -> bool:
        target_exist = False
        for edge in self.get_edges():
            if edge.get_target() == target:
                target_exist = True
                break
        return target_exist

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
