from abc import ABC
from typing import Dict

import abstract_graph
from edge import Edge
import networkx as nx


class Graph(abstract_graph.AbstractGraph, ABC):
    def __init__(self):
        super().__init__()

    def get_vertex_count(self) -> int:
        return len(self._vertices)

    def get_edge_count(self) -> int:
        total = 0
        for v in self._vertices:
            count = v.get_edge_count()
            total += count

        return total

    def has_edge(self, u: int, v: int) -> bool:
        self._check_vertex_index(u)
        self._check_vertex_index(v)
        return self.get_vertex(u).has_target(self.get_vertex(v))

    def add_edge(self, u: int, v: int):
        if u == v or self.has_edge(u, v):
            return
        new_edge = Edge(self._vertices[u], self._vertices[v])
        self._vertices[u].add_edge(new_edge)

    def remove_edge(self, u: int, v: int):
        self._check_vertex_index(u)
        self._check_vertex_index(v)
        source_vertex = self._vertices[u]
        target_vertex = self._vertices[v]
        edges = source_vertex.get_edges()
        for edge in edges:
            if edge.get_target() == target_vertex:
                edges.remove(edge)
                break

    def is_successor(self, u: int, v: int) -> bool:
        return self.has_edge(v, u)

    def is_predecessor(self, u: int, v: int) -> bool:
        return self.has_edge(u, v)

    def is_divergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        return (
            not (u1 != u2 or v1 == v2)
            and self.has_edge(u1, v1)
            and self.has_edge(u2, v2)
        )

    def is_convergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        return (
            not (u1 == u2 or v1 != v2)
            and self.has_edge(u1, v1)
            and self.has_edge(u2, v2)
        )

    def is_incident(self, u: int, v: int, x: int) -> bool:
        return self.has_edge(u, v) and ((x == u) or (x == v))

    def get_vertex_in_degree(self, u: int) -> int:
        self._check_vertex_index(u)
        count = 0
        target_vertex = self.get_vertex(u)

        for v in range(self.get_vertex_count()):
            vertex = self.get_vertex(v)
            for edge in vertex.get_edges():
                if edge.get_target() == target_vertex:
                    count += 1
        return count

    def get_vertex_out_degree(self, u: int) -> int:
        self._check_vertex_index(u)
        return self.get_vertex(u).get_edge_count()

    def set_vertex_weight(self, v: int, w: float):
        self._check_vertex_index(v)
        self.get_vertex(v).set_vertex_weight(w)

    def get_vertex_weight(self, v: int) -> float:
        self._check_vertex_index(v)
        return self.get_vertex(v).get_vertex_weight()

    def set_edge_weight(self, u: int, v: int, w: float):
        self._check_vertex_index(u)
        self._check_vertex_index(v)
        for edge in self.get_vertex(u).get_edges():
            if edge.get_target() == self.get_vertex(v):
                edge.set_weight(w)
                break

    def get_edge_weight(self, u: int, v: int) -> float:
        self._check_vertex_index(u)
        self._check_vertex_index(v)
        edge_weight = 0.0
        for edge in self.get_vertex(u).get_edges():
            if edge.get_target() == self.get_vertex(v):
                edge_weight = edge.get_weight()
                break
        return edge_weight

    def is_connected(self) -> bool:
        if self.is_empty_graph():
            return False

        vertex_map = {v: i for i, v in enumerate(self._vertices)}
        num_vertices = len(self._vertices)

        adj_list = [set() for _ in range(num_vertices)]

        for i, vertex in enumerate(self._vertices):
            for edge in vertex.get_edges():
                target = edge.get_target()
                if target in vertex_map:
                    target_idx = vertex_map[target]
                    adj_list[i].add(target_idx)
                    adj_list[target_idx].add(i)

        visited = set()
        queue = [0]
        visited.add(0)

        while queue:
            u = queue.pop(0)
            for v in adj_list[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        return len(visited) == num_vertices

    def is_empty_graph(self) -> bool:
        return self.get_vertex_count() == 0

    def is_complete_graph(self) -> bool:
        return (
            self.get_vertex_count() * (self.get_vertex_count() - 1)
            == self.get_edge_count()
        )

    def export_to_gephi(self, path: str):
        base_path = path.replace(".csv", "")
        file_nodes = f"{base_path}_nodes.csv"
        file_edges = f"{base_path}_edges.csv"

        try:
            with open(file_nodes, "w", encoding="utf-8") as f:
                f.write("Id,Label,Weight\n")
                for i, vertex in enumerate(self._vertices):
                    safe_label = vertex.get_vertex_label().replace(",", " ")
                    f.write(f"{i},{safe_label},{vertex.get_vertex_weight()}\n")
            vertex_map = {v: i for i, v in enumerate(self._vertices)}
            with open(file_edges, "w", encoding="utf-8") as f:
                f.write("Source,Target,Weight,Type\n")
                for source_idx, vertex in enumerate(self._vertices):
                    for edge in vertex.get_edges():
                        target_vertex = edge.get_target()
                        target_idx = vertex_map.get(target_vertex)

                        if target_idx is not None:
                            w = edge.get_weight()
                            f.write(f"{source_idx},{target_idx},{w},Directed\n")
        except IOError as e:
            print(f"Erro ao salvar os arquivos CSV: {e}")

    def _to_networkx(self) -> nx.DiGraph:
        nx_graph = nx.DiGraph()
        vertex_map = {v: i for i, v in enumerate(self._vertices)}
        for i in range(len(self._vertices)):
            nx_graph.add_node(i, label=self._vertices[i].get_vertex_label())
        for source_vertex in self._vertices:
            source_idx = vertex_map[source_vertex]
            for edge in source_vertex.get_edges():
                target_vertex = edge.get_target()
                if target_vertex in vertex_map:
                    target_idx = vertex_map[target_vertex]
                    nx_graph.add_edge(source_idx, target_idx, weight=edge.get_weight())

        return nx_graph

    def calculate_degree_centrality(self) -> Dict[int, float]:
        nx_graph = self._to_networkx()
        return nx.degree_centrality(nx_graph)

    def calculate_betweenness_centrality(self) -> Dict[int, float]:
        nx_graph = self._to_networkx()
        return nx.betweenness_centrality(nx_graph, weight=None)

    def calculate_closeness_centrality(self) -> Dict[int, float]:
        nx_graph = self._to_networkx()
        return nx.closeness_centrality(nx_graph)

    def calculate_pagerank(self) -> Dict[int, float]:
        nx_graph = self._to_networkx()
        try:
            return nx.pagerank(nx_graph, alpha=0.85, weight="weight")
        except nx.PowerIterationFailedConvergence:
            return {i: 0.0 for i in range(self.get_vertex_count())}

    def print_centrality_metrics(self):
        degree = self.calculate_degree_centrality()
        betweenness = self.calculate_betweenness_centrality()
        closeness = self.calculate_closeness_centrality()
        pagerank = self.calculate_pagerank()

        for i in range(self.get_vertex_count()):
            label = self.get_vertex(i).get_vertex_label()
            d_val = round(degree.get(i, 0), 4)
            b_val = round(betweenness.get(i, 0), 4)
            c_val = round(closeness.get(i, 0), 4)
            p_val = round(pagerank.get(i, 0), 4)

            print(
                f"{label:<15} | {d_val:<10} | {b_val:<10} | {c_val:<10} | {p_val:<10}"
            )

    def calculate_density(self) -> float:
        nx_graph = self._to_networkx()
        return nx.density(nx_graph)

    def calculate_average_clustering(self) -> float:
        nx_graph = self._to_networkx()
        return nx.average_clustering(nx_graph)

    def calculate_assortativity(self) -> float:
        nx_graph = self._to_networkx()
        try:
            return nx.degree_assortativity_coefficient(nx_graph)
        except ValueError:
            return 0.0

    def print_structure_metrics(self):
        density = self.calculate_density()
        clustering = self.calculate_average_clustering()
        assortativity = self.calculate_assortativity()

        print("--- Métricas de Estrutura e Coesão ---")
        print(f"{'Métrica':<30} | {'Valor':<10}")
        print("-" * 45)
        print(f"{'Densidade da Rede':<30} | {density:.4f}")
        print(f"{'Coef. de Aglomeração Médio':<30} | {clustering:.4f}")
        print(f"{'Assortatividade':<30} | {assortativity:.4f}")
        print("-" * 45)

        print("Interpretação rápida:")
        if density > 0.5:
            print("- Rede densa: Alta colaboração geral.")
        else:
            print("- Rede esparsa: Conexões pontuais.")

        if assortativity > 0:
            print("- Assortativa: Os influentes falam mais entre si.")
        elif assortativity < 0:
            print("- Disassortativa: Os influentes lideram grupos de 'novatos'.")

    def calculate_communities(self) -> Dict[int, int]:
        nx_graph = self._to_networkx()
        undirected_graph = nx_graph.to_undirected()
        communities_list = nx.community.greedy_modularity_communities(undirected_graph)
        community_map = {}
        for community_id, community_set in enumerate(communities_list):
            for node_idx in community_set:
                community_map[node_idx] = community_id

        return community_map

    def identify_bridging_nodes(self) -> Dict[int, bool]:
        nx_graph = self._to_networkx()
        community_map = self.calculate_communities()
        bridging_map = {}

        for node in nx_graph.nodes():
            my_community = community_map.get(node)
            is_bridge = False
            for neighbor in nx_graph.successors(node):
                neighbor_community = community_map.get(neighbor)
                if (
                    neighbor_community is not None
                    and neighbor_community != my_community
                ):
                    is_bridge = True
                    break

            bridging_map[node] = is_bridge

        return bridging_map

    def print_community_metrics(self):
        comm_map = self.calculate_communities()
        bridge_map = self.identify_bridging_nodes()

        print("--- Métricas de Comunidade ---")
        print(f"{'Label':<15} | {'Comunidade (ID)':<15} | {'É Ponte? (Bridge)':<15}")
        print("-" * 55)
        sorted_nodes = sorted(comm_map.keys(), key=lambda k: comm_map[k])
        current_comm = -1
        for i in sorted_nodes:
            label = self.get_vertex(i).get_vertex_label()
            comm_id = comm_map.get(i)
            is_bridge = "SIM" if bridge_map.get(i) else ""

            # Linha separadora entre grupos
            if comm_id != current_comm:
                print(f"{'--- Grupo ' + str(comm_id) + ' ---':^55}")
                current_comm = comm_id

            print(f"{label:<15} | {comm_id:<15} | {is_bridge:<15}")
