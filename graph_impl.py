from abc import ABC

import abstract_graph
from edge import Edge


class Graph(abstract_graph.AbstractGraph, ABC):
    def getVertexCount(self) -> int:
        return self._num_vertex

    def getEdgeCount(self) -> int:
        total = 0
        for v in self._vertex:
            count = v.get_edge_count()
            total += count

        return total

    def hasEdge(self, u: int, v: int) -> bool:
        return self.getEdgeCount() > 0

    def addEdge(self, u: int, v: int):
        self._check_vertex_index(u)
        self._check_vertex_index(v)

        source_vertex = self._vertex[u]
        target_vertex = self._vertex[v]

        for edge in source_vertex.get_edges():
            if edge.get_target() == target_vertex:
                return
        new_edge = Edge(source_vertex, target_vertex)
        source_vertex.add_edge(new_edge)

    def removeEdge(self, u: int, v: int):
        self._check_vertex_index(u)
        self._check_vertex_index(v)
        source_vertex = self._vertex[u]
        target_vertex = self._vertex[v]
        edges = source_vertex.get_edges()
        for edge in edges:
            if edge.get_target() == target_vertex:
                edges.remove(edge)
                break

    def isSucessor(self, u: int, v: int) -> bool:
        pass

    def isPredessor(self, u: int, v: int) -> bool:
        pass

    def isDivergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        pass

    def isConvergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        pass

    def isIncident(self, u: int, v: int, x: int) -> bool:
        pass

    def getVertexInDegree(self, u: int) -> int:
        pass

    def getVertexOutDegree(self, u: int) -> int:
        pass

    def setVertexWeight(self, v: int, w: float):
        pass

    def getVertexWeight(self, v: int) -> float:
        pass

    def setEdgeWeight(self, u: int, v: int, w: float):
        pass

    def getEdgeWeight(self, u: int, v: int) -> float:
        pass

    def isConnected(self) -> bool:
        pass

    def isEmptyGraph(self) -> bool:
        return self._num_vertex == 0

    def isCompleteGraph(self) -> bool:
        return self._num_vertex * 2 == self.getEdgeCount()
