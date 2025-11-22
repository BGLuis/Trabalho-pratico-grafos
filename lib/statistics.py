import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import deque, defaultdict
import math

from lib.abstract_graph import AbstractGraph


class GraphStatistics:
    def __init__(self, graph: AbstractGraph):
        self.graph = graph
        self.nodes: Set[int] = set(range(graph.get_vertex_count()))
        self.edges: Dict[int, Dict[int, float]] = defaultdict(
            dict
        )  # source -> {target: weight}
        self.in_edges: Dict[int, Dict[int, float]] = defaultdict(
            dict
        )  # target -> {source: weight}
        self.node_weights: Dict[int, float] = {}
        self.vertex_labels: Dict[int, str] = {}
        self._load_from_graph()

    @classmethod
    def from_csv(
        cls, edges_file: Path, vertices_file: Path, graph_type: str = "list"
    ) -> "GraphStatistics":
        from lib.implementations import AdjacencyGraphList, AdjacencyMatrixGraph

        vertex_labels = {}
        vertex_weights = {}
        with open(vertices_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = int(row["Id"])
                vertex_labels[node_id] = row["Label"]
                vertex_weights[node_id] = float(row["Weight"])

        num_vertices = len(vertex_labels)

        if graph_type == "matrix":
            graph = AdjacencyMatrixGraph(num_vertices)
        else:
            graph = AdjacencyGraphList(num_vertices)

        for node_id in range(num_vertices):
            if node_id in vertex_labels:
                graph.set_vertex_label(node_id, vertex_labels[node_id])
                graph.set_vertex_weight(node_id, vertex_weights[node_id])

        with open(edges_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = int(row["Source"])
                target = int(row["Target"])
                weight = float(row["Weight"])
                graph.add_edge(source, target)
                graph.set_edge_weight(source, target, weight)

        return cls(graph)

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

    def _get_neighbors(self, node: int) -> List[int]:
        return list(self.edges.get(node, {}).keys())

    def _get_predecessors(self, node: int) -> List[int]:
        return list(self.in_edges.get(node, {}).keys())

    def _get_all_neighbors_undirected(self, node: int) -> Set[int]:
        neighbors = set(self._get_neighbors(node))
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

            for neighbor in self._get_neighbors(current):
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
            degree = len(self._get_neighbors(node)) + len(self._get_predecessors(node))
            centrality[node] = degree / (2 * (n - 1))

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
            out_degree = len(self._get_neighbors(node))
            centrality[node] = out_degree / (n - 1)

        return centrality

    def calculate_betweenness_centrality(self) -> Dict[int, float]:
        betweenness = {node: 0.0 for node in self.nodes}

        for source in self.nodes:
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

                for neighbor in self._get_neighbors(current):
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
                    betweenness[node] += dependency[node]

        n = len(self.nodes)
        if n > 2:
            scale = 1.0 / ((n - 1) * (n - 2))
            for node in betweenness:
                betweenness[node] *= scale

        return betweenness

    def calculate_closeness_centrality(self) -> Dict[int, float]:
        closeness = {}

        for node in self.nodes:
            distances, _ = self._bfs_shortest_paths(node)

            reachable_distances = [
                d for d in distances.values() if d != float("inf") and d > 0
            ]

            if not reachable_distances:
                closeness[node] = 0.0
            else:
                total_distance = sum(reachable_distances)
                n_reachable = len(reachable_distances)

                if total_distance > 0:
                    closeness[node] = n_reachable / total_distance
                else:
                    closeness[node] = 0.0

        return closeness

    def calculate_pagerank(
        self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6
    ) -> Dict[int, float]:
        n = len(self.nodes)
        if n == 0:
            return {}

        pagerank = {node: 1.0 / n for node in self.nodes}

        out_degree = {node: len(self._get_neighbors(node)) for node in self.nodes}

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
                # Add reverse for undirected
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
        community = {node: node for node in self.nodes}

        total_weight = sum(sum(weights.values()) for weights in self.edges.values())
        if total_weight == 0:
            return community

        node_degree = {}
        for node in self.nodes:
            degree = 0.0
            for target, weight in self.edges.get(node, {}).items():
                degree += weight
            for source, weight in self.in_edges.get(node, {}).items():
                degree += weight
            node_degree[node] = degree

        improved = True
        iterations = 0
        max_iterations = 100

        while improved and iterations < max_iterations:
            improved = False
            iterations += 1

            for node in self.nodes:
                current_comm = community[node]
                best_comm = current_comm
                best_delta = 0.0

                neighbor_comms = set()
                for neighbor in self._get_all_neighbors_undirected(node):
                    neighbor_comms.add(community[neighbor])

                for target_comm in neighbor_comms:
                    if target_comm == current_comm:
                        continue

                    delta = self._modularity_gain(
                        node, target_comm, community, node_degree, total_weight
                    )

                    if delta > best_delta:
                        best_delta = delta
                        best_comm = target_comm

                if best_comm != current_comm and best_delta > 0:
                    community[node] = best_comm
                    improved = True

        unique_comms = sorted(set(community.values()))
        comm_map = {old: new for new, old in enumerate(unique_comms)}
        return {node: comm_map[comm] for node, comm in community.items()}

    def _modularity_gain(
        self,
        node: int,
        target_comm: int,
        community: Dict[int, int],
        node_degree: Dict[int, float],
        total_weight: float,
    ) -> float:
        k_i_in = 0.0
        for neighbor in self._get_all_neighbors_undirected(node):
            if community[neighbor] == target_comm:
                weight = self.edges.get(node, {}).get(neighbor, 0.0)
                weight += self.edges.get(neighbor, {}).get(node, 0.0)
                k_i_in += weight

        sigma_tot = sum(
            node_degree[n] for n in self.nodes if community[n] == target_comm
        )

        k_i = node_degree[node]

        if total_weight > 0:
            delta = (k_i_in / total_weight) - (
                sigma_tot * k_i / (2 * total_weight * total_weight)
            )
            return delta
        return 0.0

    def calculate_modularity(self) -> float:
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

        return modularity / (2 * total_weight) if total_weight > 0 else 0.0

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
        print("Calculating centrality metrics...")
        metrics = {
            "degree_centrality": self.calculate_degree_centrality(),
            "in_degree_centrality": self.calculate_in_degree_centrality(),
            "out_degree_centrality": self.calculate_out_degree_centrality(),
            "betweenness_centrality": self.calculate_betweenness_centrality(),
            "closeness_centrality": self.calculate_closeness_centrality(),
            "pagerank": self.calculate_pagerank(),
            "eigenvector_centrality": self.calculate_eigenvector_centrality(),
            "clustering_coefficient": self.calculate_clustering_coefficient(),
        }

        print("Calculating community metrics...")
        metrics["community"] = self.detect_communities()
        metrics["bridging_node"] = self.identify_bridging_nodes()

        return metrics

    def export_metrics_to_csv(self, output_file: Path):
        print(f"Exporting metrics to {output_file}...")

        metrics = self.calculate_all_metrics()

        with open(output_file, "w", encoding="utf-8", newline="") as f:
            fieldnames = ["Id", "Label"] + list(metrics.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for node in self.nodes:
                row = {
                    "Id": node,
                    "Label": self.vertex_labels.get(node, f"Node{node}"),
                }

                for metric_name, metric_values in metrics.items():
                    value = metric_values.get(node, 0.0)
                    # Convert boolean to int for bridging nodes
                    if isinstance(value, bool):
                        value = 1 if value else 0
                    row[metric_name] = value

                writer.writerow(row)

    def print_summary_statistics(self):
        print("\n" + "=" * 80)
        print("NETWORK STATISTICS SUMMARY")
        print("=" * 80)

        print("\n--- Basic Network Info ---")
        num_nodes = len(self.nodes)
        num_edges = sum(len(targets) for targets in self.edges.values())
        print(f"Number of nodes: {num_nodes}")
        print(f"Number of edges: {num_edges}")

        print("\n--- Structure and Cohesion Metrics ---")
        density = self.calculate_density()
        avg_clustering = self.calculate_average_clustering()
        assortativity = self.calculate_assortativity()

        print(f"Density: {density:.4f}")
        print(f"Average clustering coefficient: {avg_clustering:.4f}")
        print(f"Assortativity: {assortativity:.4f}")

        if density > 0.5:
            print("  Dense network: High overall collaboration")
        else:
            print("  Sparse network: Punctual connections")

        if assortativity > 0:
            print("  Assortative: Influential users connect with each other")
        elif assortativity < 0:
            print("  Disassortative: Influential users lead groups of newcomers")

        print("\n--- Community Metrics ---")
        communities = self.detect_communities()
        num_communities = len(set(communities.values()))
        modularity = self.calculate_modularity()

        print(f"Number of communities detected: {num_communities}")
        print(f"Modularity: {modularity:.4f}")

        bridging = self.identify_bridging_nodes()
        num_bridges = sum(1 for is_bridge in bridging.values() if is_bridge)
        print(f"Number of bridging nodes: {num_bridges}")

        print("\n--- Top 10 Contributors by Different Metrics ---")

        degree = self.calculate_degree_centrality()
        top_degree = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop by Degree Centrality (most active):")
        for i, (node, value) in enumerate(top_degree, 1):
            label = self.vertex_labels.get(node, f"Node{node}")
            print(f"  {i}. {label}: {value:.4f}")

        betweenness = self.calculate_betweenness_centrality()
        top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]
        print("\nTop by Betweenness Centrality (bridges):")
        for i, (node, value) in enumerate(top_betweenness, 1):
            label = self.vertex_labels.get(node, f"Node{node}")
            print(f"  {i}. {label}: {value:.4f}")

        pagerank = self.calculate_pagerank()
        top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\nTop by PageRank (most influential):")
        for i, (node, value) in enumerate(top_pagerank, 1):
            label = self.vertex_labels.get(node, f"Node{node}")
            print(f"  {i}. {label}: {value:.4f}")

        print("\n" + "=" * 80)
