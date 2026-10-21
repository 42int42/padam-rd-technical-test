import time

from input import parse_cmd_line, parse_file
from graph import Graph
from cpp_solver import solve_cpp


def main():
    in_file, plot_graph, closed, method = parse_cmd_line()
    vertices, edges = parse_file(in_file)
    print(f"#E={len(edges)}, #V={len(vertices)}")
    graph = Graph(vertices, edges)

    start = time.perf_counter()
    solution = solve_cpp(graph, closed=closed, method=method)
    elapsed = time.perf_counter() - start

    for i, walk in enumerate(solution.walks):
        print(
            f"Component {i}: {len(walk)} edges traversed, value {sum(edge[2] for edge in walk)}, "
            f"from vertex {walk[0][0]} to vertex {walk[-1][1]}"
        )
    print("Length of path found:", sum(len(walk) + 1 for walk in solution.walks))
    print("Sum of edge weights (lower bound):", solution.base_cost)
    print("Deadheading weight (duplicated edges):", solution.deadhead_cost)
    print("Value of path found:", solution.cost)
    print(f"Solved in {elapsed:.2f}s")

    if plot_graph:
        graph.plot()
        graph.display_paths(paths=solution.walks)


if __name__ == "__main__":
    main()
