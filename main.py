from extractor.config import ExtractorConfig
from extractor.requests import fetch_all
from extractor.service import GithubService
from lib.implementations import AdjacencyMatrixGraph
from lib.parser import GraphParser, InteractionsDataFactory
from lib.statistics import ManualGraphStatistics
from lib.networkx_statistics import NetworkXGraphStatistics
from lib.graph_factory import GraphFactory

import argparse
from json import dump
from pathlib import Path

from utils import get_path_relative, log


def fetch_data():
    config = ExtractorConfig()
    service = GithubService(config)

    log("Fetching data from GitHub API...")
    data = fetch_all(service)

    output_path = get_path_relative(Path(f"data/{config.repo_name()}.json"))
    with open(output_path, "w") as file:
        dump(data, file, indent=2)

    log(f"Data saved to {output_path}")


def build_graph(input_file: str, output_dir: str, graph_type: str = "integrated"):
    log(f"Building graph from {input_file} using '{graph_type}' method...")

    graph_methods = {
        "integrated": InteractionsDataFactory.build_integrated_weighted_graph,
        "comments": InteractionsDataFactory.build_comments_pull_requests_issues_graph,
        "reviews": InteractionsDataFactory.build_approve_merge_revision_pull_requests_graph,
        "closed": InteractionsDataFactory.build_closed_issues_graph,
    }

    if graph_type not in graph_methods:
        available = ", ".join(graph_methods.keys())
        raise ValueError(
            f"Invalid graph type '{graph_type}'. Available types: {available}"
        )

    builder = GraphParser(AdjacencyMatrixGraph)
    data = graph_methods[graph_type](input_file)
    g = builder.get_graph(data)

    count = g.get_vertex_count()
    log(f"Graph built with {count} vertices and {g.get_edge_count()} edges")

    output_file = Path(f"{output_dir}/{Path(input_file).stem}/{graph_type}")
    log(f"Exporting to {output_file}...")
    g.export_to_gephi(output_file)
    log("Export complete!")


def analyze_graph(
    input_dir: str, output_dir: str = "statistics", strategy: str = "manual"
):
    input_path = Path(input_dir)

    edges_file = None
    vertices_file = None

    if input_path.is_dir():
        for file in input_path.glob("*_edges.csv"):
            edges_file = file
            base_name = file.name.replace("_edges.csv", "")
            vertices_file = input_path / f"{base_name}_vertexes.csv"
            break
    else:
        edges_file = Path(f"{input_path}_edges.csv")
        vertices_file = Path(f"{input_path}_vertexes.csv")

    if not edges_file or not edges_file.exists():
        raise FileNotFoundError(f"Edges file not found in {input_dir}")
    if not vertices_file or not vertices_file.exists():
        raise FileNotFoundError(f"Vertices file not found: {vertices_file}")

    log("Analyzing graph from:")
    log(f"  Edges: {edges_file}")
    log(f"  Vertices: {vertices_file}")
    log("  Using: AdjacencyListGraph representation")
    log(f"  Strategy: {strategy}")

    graph = GraphFactory.from_gephi(edges_file, vertices_file, graph_type="list")

    if strategy == "networkx":
        stats = NetworkXGraphStatistics(graph)
    else:
        stats = ManualGraphStatistics(graph)

    log("\nPre-calculating all metrics in parallel...")
    stats.get_or_calculate_metrics(parallel=True)
    log("[OK] Metrics calculation complete!\n")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    base_name = edges_file.stem.replace("_edges", "")
    metrics_file = output_path / f"{strategy}/{base_name}_metrics.csv"

    stats.export_metrics_to_csv(metrics_file)

    log(f"\nMetrics exported to: {metrics_file}")
    log("You can import this file into Gephi as node attributes!")


def main():
    parser = argparse.ArgumentParser(
        description="GitHub data extractor and graph builder"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Fetch command
    subparsers.add_parser(
        "fetch", help="Fetch data from GitHub API and save to JSON file"
    )

    # Build command
    build_parser = subparsers.add_parser(
        "build", help="Build graph from JSON file and export to Gephi format"
    )
    build_parser.add_argument(
        "-i",
        "--input",
        default="data/node.json",
        help="Input JSON file path (default: data/node.json)",
    )
    build_parser.add_argument(
        "-o",
        "--output",
        default="tables/",
        help="Output directory for Gephi files (default: tables/)",
    )
    build_parser.add_argument(
        "-t",
        "--type",
        dest="graph_type",
        choices=["integrated", "comments", "reviews", "closed"],
        default="integrated",
        help="Graph type to build (default: integrated). "
        "Options: integrated (weighted graph combining all interactions), "
        "comments (PR/issue comments), reviews (PR reviews and merges), "
        "closed (issue closures)",
    )

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze graph and calculate centrality, structure, and community metrics",
    )
    analyze_parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input graph directory or base path (e.g., tables/node/integrated)",
    )
    analyze_parser.add_argument(
        "-o",
        "--output",
        default="statistics",
        help="Output directory for statistics (default: statistics)",
    )
    analyze_parser.add_argument(
        "-s",
        "--strategy",
        choices=["manual", "networkx"],
        default="manual",
        help="Statistics calculation strategy: manual (custom algorithms) or networkx (library wrapper)",
    )

    args = parser.parse_args()

    if args.command == "fetch":
        fetch_data()
    elif args.command == "build":
        build_graph(args.input, args.output, args.graph_type)
    elif args.command == "analyze":
        analyze_graph(args.input, args.output, args.strategy)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
