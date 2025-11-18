from abc import ABC

import abstract_graph


class Graph(abstract_graph.AbstractGraph, ABC):
    def getVertexCount(self) -> int:
        pass

    def getEdgeCount(self) -> int:
        pass

    def hasEdge(self, u: int, v: int) -> bool:
        pass

    def addEdge(self, u: int, v: int):
        pass

    def removeEdge(self, u: int, v: int):
        pass

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
        pass

    def isCompleteGraph(self) -> bool:
        pass
