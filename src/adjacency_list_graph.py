from typing import List

from src.vertex import Vertex


class AdjacencyGraphList:
    def __init__(self, num_vertices):
        self._adjacency_list = [[] for _ in range(num_vertices)]

    def set_adjacency_list(self, vertices: List[Vertex]):
        for i in range(len(self._adjacency_list)):
            adjacency_vertices = []
            vertex = vertices[i]
            edges = vertex.get_edges()
            for edge in edges:
                target = edge.get_target()
                adjacency_vertices.append(target)
            self._adjacency_list[i] = adjacency_vertices

    def print_list(self):
        print("=== Lista de Adjacência ===")
        for i, neighbors in enumerate(self._adjacency_list):
            neighbor_labels = [v.get_vertex_label() for v in neighbors]
            print(f"Vértice {i} -> {neighbor_labels}")
