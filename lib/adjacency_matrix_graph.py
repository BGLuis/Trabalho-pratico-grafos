from typing import List

from lib.common import Vertex


class AdjacencyMatrixGraph:
    def __init__(self, num_vertices):
        self._num_vertices = num_vertices
        self.matriz = [[0.0] * num_vertices for _ in range(num_vertices)]

    def print_matrix(self):
        print("Matriz de Adjacência:")
        for row in self.matriz:
            print(row)

    def update_matriz_from_vertices(self, vertices: List[Vertex]):
        if len(vertices) != self._num_vertices:
            return
        for i in range(self._num_vertices):
            for j in range(self._num_vertices):
                source_vertex = vertices[i]
                target_vertex = vertices[j]
                if source_vertex.has_target(target_vertex):
                    weight = 1.0
                    edges = source_vertex.get_edges()
                    for edge in edges:
                        if edge.get_target() == target_vertex:
                            weight = edge.get_weight()
                            break
                    self.matriz[i][j] = weight
                else:
                    self.matriz[i][j] = 0.0
