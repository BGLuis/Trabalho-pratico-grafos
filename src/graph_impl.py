from abc import ABC
import abstract_graph
from edge import Edge


class Graph(abstract_graph.AbstractGraph, ABC):
    def __init__(self):
        super().__init__()

    def get_vertex_count(self) -> int:
        return len(self._vertices)

    def get_edge_count(self) -> int:
        total = 0
        for v in self._vertices:
            count = v.get_edge_count()
            total += count

        return total

    def has_edge(self, u: int, v: int) -> bool:
        self._check_vertex_index(u)
        self._check_vertex_index(v)
        edge_exist = False
        for edge in self.get_vertex(u).get_edges():
            if edge.get_target() == self.get_vertex(v):
                edge_exist = True
                break
        return edge_exist

    def add_edge(self, u: int, v: int):
        self._check_vertex_index(u)
        self._check_vertex_index(v)
        if u == v:
            return
        if self.has_edge(u, v):
            return
        new_edge = Edge(self._vertices[u], self._vertices[v])
        self._vertices[u].add_edge(new_edge)

    def remove_edge(self, u: int, v: int):
        self._check_vertex_index(u)
        self._check_vertex_index(v)
        source_vertex = self._vertices[u]
        target_vertex = self._vertices[v]
        edges = source_vertex.get_edges()
        for edge in edges:
            if edge.get_target() == target_vertex:
                edges.remove(edge)
                break

    def is_sucessor(self, u: int, v: int) -> bool:
        return self.has_edge(v, u)

    def is_predessor(self, u: int, v: int) -> bool:
        return self.has_edge(u, v)

    def is_divergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        if u1 != u2 or v1 == v2:
            return False
        return self.has_edge(u1, v1) and self.has_edge(u2, v2)

    def is_convergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        if u1 == u2 or v1 != v2:
            return False
        return self.has_edge(u1, v1) and self.has_edge(u2, v2)

    def is_incident(self, u: int, v: int, x: int) -> bool:
        if not self.has_edge(u, v):
            return False
        return (x == u) or (x == v)

    def get_vertex_in_degree(self, u: int) -> int:
        self._check_vertex_index(u)
        count = 0
        target_vertex = self.get_vertex(u)

        for v in range(len(self._vertices)):
            vertex = self.get_vertex(v)
            for edge in vertex.get_edges():
                if edge.get_target() == target_vertex:
                    count += 1
        return count

    def get_vertex_out_degree(self, u: int) -> int:
        self._check_vertex_index(u)
        return self.get_vertex(u).get_edge_count()

    def set_vertex_weight(self, v: int, w: float):
        self._check_vertex_index(v)
        self.get_vertex(v).set_vertex_weight(w)

    def get_vertex_weight(self, v: int) -> float:
        self._check_vertex_index(v)
        return self.get_vertex(v).get_vertex_weight()

    def set_edge_weight(self, u: int, v: int, w: float):
        self._check_vertex_index(u)
        self._check_vertex_index(v)
        for edge in self.get_vertex(u).get_edges():
            if edge.get_target() == self.get_vertex(v):
                edge.set_weight(w)
                break

    def get_edge_weight(self, u: int, v: int) -> float:
        self._check_vertex_index(u)
        self._check_vertex_index(v)
        edge_weight = 0.0
        for edge in self.get_vertex(u).get_edges():
            if edge.get_target() == self.get_vertex(v):
                edge_weight = edge.get_weight()
                break
        return edge_weight

    def is_connected(self) -> bool:
        connected = True
        if self.is_empty_graph():
            return False

        for i in range(len(self._vertices)):
            if self.get_vertex_in_degree(i) == 0 and self.get_vertex_out_degree(i) == 0:
                connected = False
                break
        return connected

    def is_empty_graph(self) -> bool:
        return len(self._vertices) == 0

    def is_complete_graph(self) -> bool:
        return (
            len(self._vertices) * (len(self._vertices) - 1) / 2 == self.get_edge_count()
        )
