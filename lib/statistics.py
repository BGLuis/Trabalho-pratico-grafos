import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import deque, defaultdict
import math
from concurrent.futures import ThreadPoolExecutor, as_completed

from lib.abstract_graph import AbstractGraph
from lib.abstract_statistics import AbstractGraphStatistics
from utils import log, submit_parallel_processes, CacheStore


class ManualGraphStatistics(AbstractGraphStatistics):
    def __init__(self, graph: AbstractGraph):
        super().__init__(graph)
        self.nodes: Set[int] = set(range(graph.get_vertex_count()))
        self.edges: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.in_edges: Dict[int, Dict[int, float]] = defaultdict(dict)
        self.node_weights: Dict[int, float] = {}
        self.vertex_labels: Dict[int, str] = {}
        self.__cache_store = CacheStore("manual/statistics")
        self._load_from_graph()

    def _load_from_graph(self):
        num_vertices = self.graph.get_vertex_count()

        for node_id in range(num_vertices):
            self.vertex_labels[node_id] = self.graph.get_vertex_label(node_id)
            self.node_weights[node_id] = self.graph.get_vertex_weight(node_id)

        for source in range(num_vertices):
            for target in range(num_vertices):
                if self.graph.has_edge(source, target):
                    weight = self.graph.get_edge_weight(source, target)
                    self.edges[source][target] = weight
                    self.in_edges[target][source] = weight

    def _get_successors(self, node: int) -> List[int]:
        return list(self.edges.get(node, {}).keys())

    def _get_predecessors(self, node: int) -> List[int]:
        return list(self.in_edges.get(node, {}).keys())

    def _get_all_neighbors_undirected(self, node: int) -> Set[int]:
        neighbors = set(self._get_successors(node))
        neighbors.update(self._get_predecessors(node))
        return neighbors

    def _bfs_shortest_paths(
        self, source: int
    ) -> Tuple[Dict[int, float], Dict[int, int]]:
        distances = {node: float("inf") for node in self.nodes}
        distances[source] = 0
        num_paths = {node: 0 for node in self.nodes}
        num_paths[source] = 1

        queue = deque([source])

        while queue:
            current = queue.popleft()

            for neighbor in self._get_successors(current):
                if distances[neighbor] == float("inf"):
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

                if distances[neighbor] == distances[current] + 1:
                    num_paths[neighbor] += num_paths[current]

        return distances, num_paths

    def calculate_degree_centrality(self) -> Dict[int, float]:
        n = len(self.nodes)
        if n <= 1:
            return {node: 0.0 for node in self.nodes}

        centrality = {}
        for node in self.nodes:
            degree = len(self._get_successors(node)) + len(self._get_predecessors(node))
            centrality[node] = degree / (n - 1)

        return centrality

    def calculate_in_degree_centrality(self) -> Dict[int, float]:
        n = len(self.nodes)
        if n <= 1:
            return {node: 0.0 for node in self.nodes}

        centrality = {}
        for node in self.nodes:
            in_degree = len(self._get_predecessors(node))
            centrality[node] = in_degree / (n - 1)

        return centrality

    def calculate_out_degree_centrality(self) -> Dict[int, float]:
        n = len(self.nodes)
        if n <= 1:
            return {node: 0.0 for node in self.nodes}

        centrality = {}
        for node in self.nodes:
            out_degree = len(self._get_successors(node))
            centrality[node] = out_degree / (n - 1)

        return centrality

    def _calculate_betweenness_for_source(self, source: int) -> Dict[int, float]:
        local_betweenness = {node: 0.0 for node in self.nodes}

        stack = []
        predecessors = {node: [] for node in self.nodes}
        distances = {node: -1 for node in self.nodes}
        distances[source] = 0
        num_paths = {node: 0 for node in self.nodes}
        num_paths[source] = 1

        queue = deque([source])

        while queue:
            current = queue.popleft()
            stack.append(current)

            for neighbor in self._get_successors(current):
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)

                if distances[neighbor] == distances[current] + 1:
                    num_paths[neighbor] += num_paths[current]
                    predecessors[neighbor].append(current)

        dependency = {node: 0.0 for node in self.nodes}

        while stack:
            node = stack.pop()
            for pred in predecessors[node]:
                if num_paths[node] > 0:
                    dependency[pred] += (num_paths[pred] / num_paths[node]) * (
                        1 + dependency[node]
                    )

            if node != source:
                local_betweenness[node] = dependency[node]

        return local_betweenness

    def calculate_betweenness_centrality(self) -> Dict[int, float]:
        cache_key = self.__cache_store.get_statistic_cache_key(
            graph_data=self.__get_deterministic_cache_key(),
            metric_name="betweenness_centrality",
        )
        cached = self.__cache_store.get(cache_key)
        if cached is not None:
            log("Using cached betweenness centrality.")
            return {int(k): v for k, v in cached.items()}

        betweenness = {node: 0.0 for node in self.nodes}

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._calculate_betweenness_for_source, source): source
                for source in self.nodes
            }

            for future in as_completed(futures):
                local_betweenness = future.result()
                for node, value in local_betweenness.items():
                    betweenness[node] += value

        n = len(self.nodes)
        if n > 2:
            scale = 1.0 / ((n - 1) * (n - 2))
            for node in betweenness:
                betweenness[node] *= scale

        self.__cache_store.set(cache_key, betweenness)
        return betweenness

    def _calculate_closeness_for_node(self, node: int) -> Tuple[int, float]:
        distances, _ = self._bfs_shortest_paths(node)

        reachable_distances = [
            d for d in distances.values() if d != float("inf") and d > 0
        ]

        if not reachable_distances:
            return (node, 0.0)
        else:
            total_distance = sum(reachable_distances)
            n_reachable = len(reachable_distances)

            if total_distance > 0:
                return (node, n_reachable / total_distance)
            else:
                return (node, 0.0)

    def calculate_closeness_centrality(self) -> Dict[int, float]:
        closeness = {}

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._calculate_closeness_for_node, node)
                for node in self.nodes
            ]

            for future in as_completed(futures):
                node, value = future.result()
                closeness[node] = value

        return closeness

    def calculate_pagerank(
        self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6
    ) -> Dict[int, float]:
        n = len(self.nodes)
        if n == 0:
            return {}

        pagerank = {node: 1.0 / n for node in self.nodes}

        out_degree = {node: len(self._get_successors(node)) for node in self.nodes}

        for iteration in range(max_iter):
            new_pagerank = {}
            diff = 0.0

            for node in self.nodes:
                rank_sum = 0.0
                for predecessor in self._get_predecessors(node):
                    if out_degree[predecessor] > 0:
                        weight = self.in_edges[node][predecessor]
                        total_out_weight = sum(self.edges[predecessor].values())
                        rank_sum += (weight / total_out_weight) * pagerank[predecessor]

                new_pagerank[node] = (1 - alpha) / n + alpha * rank_sum
                diff += abs(new_pagerank[node] - pagerank[node])

            pagerank = new_pagerank

            if diff < tol:
                break

        return pagerank

    def calculate_eigenvector_centrality(
        self, max_iter: int = 100, tol: float = 1e-6
    ) -> Dict[int, float]:
        n = len(self.nodes)
        if n == 0:
            return {}

        centrality = {node: 1.0 / n for node in self.nodes}

        for iteration in range(max_iter):
            new_centrality = {node: 0.0 for node in self.nodes}

            for node in self.nodes:
                for predecessor in self._get_predecessors(node):
                    weight = self.in_edges[node][predecessor]
                    new_centrality[node] += weight * centrality[predecessor]

            norm = math.sqrt(sum(v * v for v in new_centrality.values()))
            if norm > 0:
                new_centrality = {
                    node: val / norm for node, val in new_centrality.items()
                }

            diff = sum(
                abs(new_centrality[node] - centrality[node]) for node in self.nodes
            )
            centrality = new_centrality

            if diff < tol:
                break

        return centrality

    def calculate_density(self) -> float:
        n = len(self.nodes)
        if n <= 1:
            return 0.0

        num_edges = sum(len(targets) for targets in self.edges.values())
        max_edges = n * (n - 1)

        return num_edges / max_edges if max_edges > 0 else 0.0

    def calculate_clustering_coefficient(self) -> Dict[int, float]:
        clustering = {}

        for node in self.nodes:
            neighbors = self._get_all_neighbors_undirected(node)
            k = len(neighbors)

            if k < 2:
                clustering[node] = 0.0
                continue

            edges_between = 0
            neighbors_list = list(neighbors)

            for i in range(len(neighbors_list)):
                for j in range(i + 1, len(neighbors_list)):
                    n1, n2 = neighbors_list[i], neighbors_list[j]
                    if n2 in self.edges.get(n1, {}) or n1 in self.edges.get(n2, {}):
                        edges_between += 1

            max_edges = k * (k - 1) / 2
            clustering[node] = edges_between / max_edges if max_edges > 0 else 0.0

        return clustering

    def calculate_average_clustering(self) -> float:
        clustering = self.calculate_clustering_coefficient()
        if not clustering:
            return 0.0
        return sum(clustering.values()) / len(clustering)

    def calculate_assortativity(self) -> float:
        if not self.edges:
            return 0.0

        degrees = {}
        for node in self.nodes:
            degrees[node] = len(self._get_all_neighbors_undirected(node))

        edge_list = []
        for source in self.edges:
            for target in self.edges[source]:
                edge_list.append((degrees[source], degrees[target]))
                edge_list.append((degrees[target], degrees[source]))

        if not edge_list:
            return 0.0

        m = len(edge_list)
        sum_jk = sum(j * k for j, k in edge_list)
        sum_j = sum(j for j, _ in edge_list)
        sum_k = sum(k for _, k in edge_list)
        sum_j2 = sum(j * j for j, _ in edge_list)
        sum_k2 = sum(k * k for _, k in edge_list)

        numerator = sum_jk / m - (sum_j / m) * (sum_k / m)
        denominator_j = sum_j2 / m - (sum_j / m) ** 2
        denominator_k = sum_k2 / m - (sum_k / m) ** 2

        if denominator_j * denominator_k > 0:
            return numerator / math.sqrt(denominator_j * denominator_k)
        return 0.0

    def detect_communities(self) -> Dict[int, int]:
        cached_key = self.__cache_store.get_statistic_cache_key(
            graph_data=self.__get_deterministic_cache_key(),
            metric_name="communities",
        )
        cached = self.__cache_store.get(cached_key)
        if cached is not None:
            log("Using cached communities.")
            return {int(k): v for k, v in cached.items()}

        community = {node: node for node in self.nodes}

        total_weight = sum(sum(weights.values()) for weights in self.edges.values())
        if total_weight == 0:
            return community

        node_degree = {}
        for node in self.nodes:
            degree = sum(self.edges.get(node, {}).values())
            degree += sum(self.in_edges.get(node, {}).values())
            node_degree[node] = degree

        neighbors_cache = {
            node: self._get_all_neighbors_undirected(node) for node in self.nodes
        }

        comm_degree = {node: node_degree[node] for node in self.nodes}

        improved = True
        iterations = 0
        max_iterations = 100
        min_improvement = 1e-6

        while improved and iterations < max_iterations:
            log(f"Community detection iteration {iterations + 1}...")
            improved = False
            iterations += 1
            moves_count = 0

            nodes_list = list(self.nodes)

            for node in nodes_list:
                current_comm = community[node]
                best_comm = current_comm
                best_delta = 0.0

                neighbor_comms = {
                    community[neighbor] for neighbor in neighbors_cache[node]
                }

                for target_comm in neighbor_comms:
                    if target_comm == current_comm:
                        continue

                    delta = self._modularity_gain_optimized(
                        node,
                        current_comm,
                        target_comm,
                        community,
                        node_degree,
                        comm_degree,
                        total_weight,
                        neighbors_cache,
                    )

                    if delta > best_delta:
                        best_delta = delta
                        best_comm = target_comm

                if best_comm != current_comm and best_delta > min_improvement:
                    comm_degree[current_comm] -= node_degree[node]
                    comm_degree[best_comm] = (
                        comm_degree.get(best_comm, 0) + node_degree[node]
                    )

                    community[node] = best_comm
                    improved = True
                    moves_count += 1

            if moves_count > 0:
                log(f"Iteration {iterations}: {moves_count} nodes moved")

        log(f"Community detection converged after {iterations} iterations")

        unique_comms = sorted(set(community.values()))
        comm_map = {old: new for new, old in enumerate(unique_comms)}
        community = {node: comm_map[comm] for node, comm in community.items()}
        self.__cache_store.set(cached_key, community)
        return community

    def _modularity_gain_optimized(
        self,
        node: int,
        current_comm: int,
        target_comm: int,
        community: Dict[int, int],
        node_degree: Dict[int, float],
        comm_degree: Dict[int, float],
        total_weight: float,
        neighbors_cache: Dict[int, Set[int]],
    ) -> float:
        if total_weight == 0:
            return 0.0

        k_i_in = 0.0
        for neighbor in neighbors_cache[node]:
            if community[neighbor] == target_comm:
                weight = self.edges.get(node, {}).get(neighbor, 0.0)
                weight += self.edges.get(neighbor, {}).get(node, 0.0)
                k_i_in += weight

        sigma_tot = comm_degree.get(target_comm, 0.0)
        k_i = node_degree[node]

        delta = (k_i_in / total_weight) - (
            sigma_tot * k_i / (2 * total_weight * total_weight)
        )
        return delta

    def __get_deterministic_cache_key(self) -> str:
        edges_tuple = tuple(
            sorted((u, v, w) for u in self.edges for v, w in self.edges[u].items())
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
            return cached.get("modularity", 0.0)

        communities = self.detect_communities()

        total_weight = 0.0
        for source in self.edges:
            for target, weight in self.edges[source].items():
                total_weight += weight

        if total_weight == 0:
            return 0.0

        modularity = 0.0

        for node_i in self.nodes:
            for node_j in self.nodes:
                if communities[node_i] != communities[node_j]:
                    continue

                a_ij = self.edges.get(node_i, {}).get(node_j, 0.0)
                a_ij += self.edges.get(node_j, {}).get(node_i, 0.0)

                k_i = sum(self.edges.get(node_i, {}).values()) + sum(
                    self.in_edges.get(node_i, {}).values()
                )
                k_j = sum(self.edges.get(node_j, {}).values()) + sum(
                    self.in_edges.get(node_j, {}).values()
                )

                modularity += a_ij - (k_i * k_j) / (2 * total_weight)

        result = modularity / (2 * total_weight) if total_weight > 0 else 0.0
        self.__cache_store.set(cached_key, {"modularity": result})
        return result

    def identify_bridging_nodes(self) -> Dict[int, bool]:
        communities = self.detect_communities()
        bridging = {}

        for node in self.nodes:
            node_community = communities[node]
            is_bridge = False

            for neighbor in self._get_all_neighbors_undirected(node):
                if communities[neighbor] != node_community:
                    is_bridge = True
                    break

            bridging[node] = is_bridge

        return bridging

    def calculate_all_metrics(self) -> Dict[str, Dict[int, float]]:
        log("Calculating all metrics in parallel...")

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

        log("All metrics calculated!")
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

            for node in self.nodes:
                row = {
                    "Id": node,
                    "Label": self.vertex_labels.get(node, f"Node{node}"),
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
