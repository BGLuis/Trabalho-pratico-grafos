import csv
import json
from pathlib import Path
from typing import Dict
import networkx as nx

from lib.abstract_graph import AbstractGraph
from lib.abstract_statistics import AbstractGraphStatistics
from utils import log, submit_parallel_processes, CacheStore


class NetworkXGraphStatistics(AbstractGraphStatistics):
    def __init__(self, graph: AbstractGraph):
        super().__init__(graph)
        self._nx_graph = self._convert_to_networkx()
        self.__cache_store = CacheStore("networkx/statistics")

    def _convert_to_networkx(self) -> nx.DiGraph:
        g = nx.DiGraph()

        num_vertices = self.graph.get_vertex_count()

        for node_id in range(num_vertices):
            g.add_node(
                node_id,
                label=self.graph.get_vertex_label(node_id),
                weight=self.graph.get_vertex_weight(node_id),
            )

        for source in range(num_vertices):
            for target in range(num_vertices):
                if self.graph.has_edge(source, target):
                    weight = self.graph.get_edge_weight(source, target)
                    g.add_edge(source, target, weight=weight)

        return g

    def calculate_degree_centrality(self) -> Dict[int, float]:
        return nx.degree_centrality(self._nx_graph)

    def calculate_in_degree_centrality(self) -> Dict[int, float]:
        return nx.in_degree_centrality(self._nx_graph)

    def calculate_out_degree_centrality(self) -> Dict[int, float]:
        return nx.out_degree_centrality(self._nx_graph)

    def calculate_betweenness_centrality(self) -> Dict[int, float]:
        return nx.betweenness_centrality(self._nx_graph, weight="weight")

    def calculate_closeness_centrality(self) -> Dict[int, float]:
        return nx.closeness_centrality(self._nx_graph)

    def calculate_pagerank(
        self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6
    ) -> Dict[int, float]:
        try:
            return nx.pagerank(
                self._nx_graph, alpha=alpha, max_iter=max_iter, tol=tol, weight="weight"
            )
        except nx.PowerIterationFailedConvergence:
            log("Warning: PageRank did not converge, using default values")
            return {
                node: 1.0 / self._nx_graph.number_of_nodes()
                for node in self._nx_graph.nodes()
            }

    def calculate_eigenvector_centrality(
        self, max_iter: int = 100, tol: float = 1e-6
    ) -> Dict[int, float]:
        try:
            return nx.eigenvector_centrality(
                self._nx_graph, max_iter=max_iter, tol=tol, weight="weight"
            )
        except (nx.PowerIterationFailedConvergence, nx.NetworkXError):
            log(
                "Warning: Eigenvector centrality did not converge, using default values"
            )
            return {
                node: 1.0 / self._nx_graph.number_of_nodes()
                for node in self._nx_graph.nodes()
            }

    def calculate_density(self) -> float:
        return nx.density(self._nx_graph)

    def calculate_clustering_coefficient(self) -> Dict[int, float]:
        undirected = self._nx_graph.to_undirected()
        return nx.clustering(undirected, weight="weight")

    def calculate_average_clustering(self) -> float:
        undirected = self._nx_graph.to_undirected()
        return nx.average_clustering(undirected, weight="weight")

    def calculate_assortativity(self) -> float:
        try:
            return nx.degree_assortativity_coefficient(self._nx_graph)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def detect_communities(self) -> Dict[int, int]:
        cached_key = self.__cache_store.get_statistic_cache_key(
            graph_data=self.__get_deterministic_cache_key(),
            metric_name="detect_communities",
            params=None,
        )
        cached_result = self.__cache_store.get(cached_key)
        if cached_result is not None:
            log("Using cached communities.")
            return {int(k): v for k, v in cached_result.items()}

        undirected = self._nx_graph.to_undirected()
        communities = nx.community.greedy_modularity_communities(
            undirected, weight="weight"
        )

        community_map = {}
        for community_id, community_set in enumerate(communities):
            for node in community_set:
                community_map[node] = community_id

        self.__cache_store.set(cached_key, community_map)
        return community_map

    def __get_deterministic_cache_key(self) -> str:
        edges_tuple = tuple(
            sorted(
                (u, v, d.get("weight", 1.0))
                for u, v, d in self._nx_graph.edges(data=True)
            )
        )
        return str(edges_tuple)

    def calculate_modularity(self) -> float:
        cached_key = self.__cache_store.get_statistic_cache_key(
            graph_data=self.__get_deterministic_cache_key(),
            metric_name="modularity",
        )
        cached = self.__cache_store.get(cached_key)
        if cached is not None:
            log("Using cached modularity.")
            return cached["modularity"]

        communities = self.detect_communities()
        undirected = self._nx_graph.to_undirected()

        max_community = max(communities.values())
        community_sets = [set() for _ in range(max_community + 1)]
        for node, comm_id in communities.items():
            community_sets[comm_id].add(node)

        return nx.community.modularity(undirected, community_sets, weight="weight")

    def identify_bridging_nodes(self) -> Dict[int, bool]:
        communities = self.detect_communities()
        bridging = {}

        for node in self._nx_graph.nodes():
            node_community = communities.get(node)
            is_bridge = False

            for neighbor in self._nx_graph.successors(node):
                if communities.get(neighbor) != node_community:
                    is_bridge = True
                    break

            if not is_bridge:
                for neighbor in self._nx_graph.predecessors(node):
                    if communities.get(neighbor) != node_community:
                        is_bridge = True
                        break

            bridging[node] = is_bridge

        return bridging

    def calculate_all_metrics(self) -> Dict[str, Dict[int, float]]:
        log("Calculating all metrics using NetworkX...")

        metrics = {
            "degree_centrality": self.calculate_degree_centrality,
            "in_degree_centrality": self.calculate_in_degree_centrality,
            "out_degree_centrality": self.calculate_out_degree_centrality,
            "betweenness_centrality": self.calculate_betweenness_centrality,
            "closeness_centrality": self.calculate_closeness_centrality,
            "pagerank": self.calculate_pagerank,
            "eigenvector_centrality": self.calculate_eigenvector_centrality,
            "clustering_coefficient": self.calculate_clustering_coefficient,
            "community": self.detect_communities,
        }

        calculated_metrics = submit_parallel_processes(metrics)

        calculated_metrics["bridging_node"] = self.identify_bridging_nodes()
        log(f"  Density: {self.calculate_density()}")
        log(f"  Assortativity: {self.calculate_assortativity()}")
        log(f"  Modularity: {self.calculate_modularity()}")
        del calculated_metrics["community"]

        log("[OK] All metrics calculated!")
        return calculated_metrics

    def export_metrics_to_csv(self, nodes_output_file: Path, graph_output_file: Path):
        log(f"Exporting metrics to {nodes_output_file}...")
        nodes_output_file.parent.mkdir(parents=True, exist_ok=True)

        metrics = self.calculate_all_metrics()

        with open(nodes_output_file, "w", encoding="utf-8", newline="") as f:
            fields = list(metrics.keys())
            fields.sort()
            fields = ["Id", "Label"] + fields
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for node in self._nx_graph.nodes():
                node_data = self._nx_graph.nodes[node]
                row = {
                    "Id": node,
                    "Label": node_data.get("label", f"Node{node}"),
                }

                for metric_name, metric_values in metrics.items():
                    value = metric_values.get(node, 0.0)
                    if isinstance(value, bool):
                        value = 1 if value else 0
                    row[metric_name] = value

                writer.writerow(row)

        with open(graph_output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "density": self.calculate_density(),
                    "assortativity": self.calculate_assortativity(),
                    "modularity": self.calculate_modularity(),
                },
                f,
            )
