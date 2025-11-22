import csv
from pathlib import Path
from typing import Dict
import networkx as nx


class GraphStatistics:
    def __init__(self, edges_file: Path, vertices_file: Path):
        self.edges_file = edges_file
        self.vertices_file = vertices_file
        self.graph = self._load_graph()
        self.vertex_labels = self._load_vertex_labels()

    def _load_vertex_labels(self) -> Dict[int, str]:
        labels = {}
        with open(self.vertices_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels[int(row["Id"])] = row["Label"]
        return labels

    def _load_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()

        with open(self.vertices_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                node_id = int(row["Id"])
                g.add_node(node_id, label=row["Label"], weight=float(row["Weight"]))

        with open(self.edges_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                source = int(row["Source"])
                target = int(row["Target"])
                weight = float(row["Weight"])
                g.add_edge(source, target, weight=weight)

        return g

    def calculate_degree_centrality(self) -> Dict[int, float]:
        return nx.degree_centrality(self.graph)

    def calculate_in_degree_centrality(self) -> Dict[int, float]:
        return nx.in_degree_centrality(self.graph)

    def calculate_out_degree_centrality(self) -> Dict[int, float]:
        return nx.out_degree_centrality(self.graph)

    def calculate_betweenness_centrality(self) -> Dict[int, float]:
        return nx.betweenness_centrality(self.graph, weight="weight")

    def calculate_closeness_centrality(self) -> Dict[int, float]:
        return nx.closeness_centrality(self.graph)

    def calculate_pagerank(self) -> Dict[int, float]:
        try:
            return nx.pagerank(self.graph, weight="weight", alpha=0.85)
        except nx.PowerIterationFailedConvergence:
            print("Warning: PageRank did not converge, using default values")
            return {
                node: 1.0 / self.graph.number_of_nodes() for node in self.graph.nodes()
            }

    def calculate_eigenvector_centrality(self) -> Dict[int, float]:
        try:
            return nx.eigenvector_centrality(self.graph, weight="weight", max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            print(
                "Warning: Eigenvector centrality did not converge, using default values"
            )
            return {
                node: 1.0 / self.graph.number_of_nodes() for node in self.graph.nodes()
            }

    def calculate_density(self) -> float:
        return nx.density(self.graph)

    def calculate_clustering_coefficient(self) -> Dict[int, float]:
        undirected = self.graph.to_undirected()
        return nx.clustering(undirected, weight="weight")

    def calculate_average_clustering(self) -> float:
        undirected = self.graph.to_undirected()
        return nx.average_clustering(undirected, weight="weight")

    def calculate_assortativity(self) -> float:
        try:
            return nx.degree_assortativity_coefficient(self.graph)
        except (ValueError, ZeroDivisionError):
            return 0.0

    def detect_communities(self) -> Dict[int, int]:
        undirected = self.graph.to_undirected()
        communities = nx.community.greedy_modularity_communities(
            undirected, weight="weight"
        )

        community_map = {}
        for community_id, community_set in enumerate(communities):
            for node in community_set:
                community_map[node] = community_id

        return community_map

    def calculate_modularity(self) -> float:
        communities = self.detect_communities()
        undirected = self.graph.to_undirected()

        max_community = max(communities.values())
        community_sets = [set() for _ in range(max_community + 1)]
        for node, comm_id in communities.items():
            community_sets[comm_id].add(node)

        return nx.community.modularity(undirected, community_sets, weight="weight")

    def identify_bridging_nodes(self) -> Dict[int, bool]:
        communities = self.detect_communities()
        bridging = {}

        for node in self.graph.nodes():
            node_community = communities.get(node)
            is_bridge = False

            for neighbor in self.graph.successors(node):
                if communities.get(neighbor) != node_community:
                    is_bridge = True
                    break

            if not is_bridge:
                for neighbor in self.graph.predecessors(node):
                    if communities.get(neighbor) != node_community:
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

            for node in self.graph.nodes():
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
        print(f"Number of nodes: {self.graph.number_of_nodes()}")
        print(f"Number of edges: {self.graph.number_of_edges()}")

        print("\n--- Structure and Cohesion Metrics ---")
        density = self.calculate_density()
        avg_clustering = self.calculate_average_clustering()
        assortativity = self.calculate_assortativity()

        print(f"Density: {density:.4f}")
        print(f"Average clustering coefficient: {avg_clustering:.4f}")
        print(f"Assortativity: {assortativity:.4f}")

        if density > 0.5:
            print("  → Dense network: High overall collaboration")
        else:
            print("  → Sparse network: Punctual connections")

        if assortativity > 0:
            print("  → Assortative: Influential users connect with each other")
        elif assortativity < 0:
            print("  → Disassortative: Influential users lead groups of newcomers")

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
