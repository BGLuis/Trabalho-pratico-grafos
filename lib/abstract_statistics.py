from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

from lib.abstract_graph import AbstractGraph


class AbstractGraphStatistics(ABC):
    def __init__(self, graph: AbstractGraph):
        self.graph = graph

    @abstractmethod
    def calculate_degree_centrality(self) -> Dict[int, float]:
        pass

    @abstractmethod
    def calculate_in_degree_centrality(self) -> Dict[int, float]:
        pass

    @abstractmethod
    def calculate_out_degree_centrality(self) -> Dict[int, float]:
        pass

    @abstractmethod
    def calculate_betweenness_centrality(
        self, parallel: bool = True
    ) -> Dict[int, float]:
        pass

    @abstractmethod
    def calculate_closeness_centrality(self, parallel: bool = True) -> Dict[int, float]:
        pass

    @abstractmethod
    def calculate_pagerank(
        self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6
    ) -> Dict[int, float]:
        pass

    @abstractmethod
    def calculate_eigenvector_centrality(
        self, max_iter: int = 100, tol: float = 1e-6
    ) -> Dict[int, float]:
        pass

    @abstractmethod
    def calculate_density(self) -> float:
        pass

    @abstractmethod
    def calculate_clustering_coefficient(self) -> Dict[int, float]:
        pass

    @abstractmethod
    def calculate_average_clustering(self) -> float:
        pass

    @abstractmethod
    def calculate_assortativity(self) -> float:
        pass

    @abstractmethod
    def detect_communities(self) -> Dict[int, int]:
        pass

    @abstractmethod
    def calculate_modularity(self) -> float:
        pass

    @abstractmethod
    def identify_bridging_nodes(self) -> Dict[int, bool]:
        pass

    @abstractmethod
    def get_or_calculate_metrics(
        self, parallel: bool = True
    ) -> Dict[str, Dict[int, float]]:
        pass

    @abstractmethod
    def calculate_all_metrics(
        self, parallel: bool = True
    ) -> Dict[str, Dict[int, float]]:
        pass

    @abstractmethod
    def export_metrics_to_csv(self, output_file: Path):
        pass
