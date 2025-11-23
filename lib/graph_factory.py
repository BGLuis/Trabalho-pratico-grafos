from pathlib import Path
from lib.abstract_graph import AbstractGraph


class GraphFactory:
    @staticmethod
    def from_gephi(
        edges_file: Path, vertices_file: Path, graph_type: str = "list"
    ) -> AbstractGraph:
        from lib.implementations import AdjacencyGraphList, AdjacencyMatrixGraph

        if graph_type == "matrix":
            return AdjacencyMatrixGraph.from_gephi(edges_file, vertices_file)
        else:
            return AdjacencyGraphList.from_gephi(edges_file, vertices_file)
