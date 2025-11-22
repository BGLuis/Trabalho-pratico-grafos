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


def build_graph(input_file: str, output_dir: str):
    """Build graph from JSON file and export to Gephi format."""
    print(f"Building graph from {input_file}...")
    builder = GraphParser(AdjacencyMatrixGraph)
    data = InteractionsDataFactory.build_comments_pull_requests_issues_graph(input_file)
    g = builder.get_graph(data)

    count = g.get_vertex_count()
    print(f"Graph built with {count} vertices")

    output_file = Path(f"{output_dir}/{Path(input_file).stem}")
    print(f"Exporting to {output_dir}...")
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

    args = parser.parse_args()

    if args.command == "fetch":
        fetch_data()
    elif args.command == "build":
        build_graph(args.input, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
