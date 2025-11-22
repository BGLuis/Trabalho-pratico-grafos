from src.graph_build import GraphBuilder


def main():
    builder = GraphBuilder()
    builder.build_closed_issues_graph("data/node.json")
    g = builder.get_graph()
    count = g.get_vertex_count()
    print(count)
    g.export_to_gephi("tables/")


if __name__ == "__main__":
    main()
