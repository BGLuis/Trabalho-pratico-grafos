from extractor.config import ExtractorConfig
from extractor.requests import fetch_all
from extractor.service import GithubService
from lib.implementations import AdjacencyMatrixGraph
from lib.parser import GraphParser, InteractionsDataFactory

import argparse
from json import dump
from pathlib import Path

from utils import get_path_relative


def fetch_data():
    """Fetch data from GitHub API and save to JSON file."""
    config = ExtractorConfig()
    service = GithubService(config)

    print("Fetching data from GitHub API...")
    data = fetch_all(service)

    output_path = get_path_relative(Path(f"data/{config.repo_name()}.json"))
    with open(output_path, "w") as file:
        dump(data, file, indent=2)

    print(f"Data saved to {output_path}")


def build_graph(input_file: str, output_dir: str, graph_type: str = "integrated"):
    """Build graph from JSON file and export to Gephi format."""
    print(f"Building graph from {input_file} using '{graph_type}' method...")

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
    print(f"Graph built with {count} vertices and {g.get_edge_count()} edges")

    output_file = Path(f"{output_dir}/{Path(input_file).stem}/{graph_type}")
    print(f"Exporting to {output_file}...")
    g.export_to_gephi(output_file)
    print("Export complete!")


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

    args = parser.parse_args()

    if args.command == "fetch":
        fetch_data()
    elif args.command == "build":
        build_graph(args.input, args.output, args.graph_type)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
