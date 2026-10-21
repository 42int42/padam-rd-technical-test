from __future__ import annotations

import itertools
import random
from collections import Counter
from functools import lru_cache

import networkx as nx
import pytest

from cpp_solver import PostmanSolution, minimum_t_join, solve_cpp
from graph import Graph
from input import parse_file

METHODS = ["dense", "sparse"]


def make_graph(edges: list[tuple[int, int, int]]) -> Graph:
    n = 1 + max(max(u, v) for u, v, _ in edges)
    vertices = [(float(i), 0.0) for i in range(n)]
    return Graph(vertices, [(u, v, w, vertices[u], vertices[v]) for u, v, w in edges])


def assert_valid_cover(graph: Graph, solution: PostmanSolution, closed: bool) -> None:
    traversed = Counter()
    for walk in solution.walks:
        assert walk, "empty walk"
        for previous, current in itertools.pairwise(walk):
            assert previous[1] == current[0], "walk is not contiguous"
        if closed:
            assert walk[0][0] == walk[-1][1], "walk is not closed"
        traversed.update((frozenset(edge[:2]), edge[2]) for edge in walk)
    required = Counter((frozenset(edge[:2]), edge[2]) for edge in graph.edges)
    assert not required - traversed, "some edge is never traversed"
    assert solution.cost == sum(traversed[k] * k[1] for k in traversed)
    assert solution.cost == solution.base_cost + solution.deadhead_cost


def brute_force_cost(graph: Graph, closed: bool) -> int:
    """Optimum by enumerating every pairing of the odd vertices, inside each component."""
    multigraph = nx.MultiGraph()
    multigraph.add_weighted_edges_from(edge[:3] for edge in graph.edges)
    total = sum(edge[2] for edge in graph.edges)
    for nodes in nx.connected_components(multigraph):
        component = multigraph.subgraph(nodes)
        dist = dict(nx.all_pairs_dijkstra_path_length(component))
        odd = frozenset(v for v, d in component.degree if d % 2)

        @lru_cache(maxsize=None)
        def pairing(rest: frozenset) -> float:
            if not rest:
                return 0
            first, *others = sorted(rest)
            return min(dist[first][o] + pairing(rest - {first, o}) for o in others)

        best = pairing(odd)
        if not closed:
            best = min([best] + [pairing(odd - {u, v}) for u, v in itertools.combinations(odd, 2)])
        total += best
    return total


def random_graph(seed: int) -> Graph:
    rng = random.Random(seed)
    n = rng.randint(2, 9)
    order = list(range(n))
    rng.shuffle(order)
    edges = [(order[i], order[rng.randrange(i)], rng.randint(0, 9)) for i in range(1, n)]
    edges += [(rng.randrange(n), rng.randrange(n), rng.randint(0, 9)) for _ in range(rng.randint(0, 2 * n))]
    if seed % 5 == 0:
        edges += [(u + n, v + n, w) for u, v, w in edges]
    return make_graph(edges)


TRIANGLE = [(0, 1, 1), (1, 2, 2), (2, 0, 3)]
PATH = [(0, 1, 1), (1, 2, 2), (2, 3, 3)]
STAR = [(0, 1, 1), (0, 2, 2), (0, 3, 3)]
KONIGSBERG = [(0, 1, 1), (0, 1, 1), (0, 2, 1), (0, 2, 1), (0, 3, 1), (1, 3, 1), (2, 3, 1)]
K5 = [(u, v, u + v) for u, v in itertools.combinations(range(5), 2)]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "edges, closed, expected",
    [
        (TRIANGLE, True, 6),
        (K5, True, sum(w for *_, w in K5)),
        (PATH, True, 12),
        (PATH, False, 6),
        (STAR, True, 12),
        (STAR, False, 7),
        (KONIGSBERG, True, 9),
        (KONIGSBERG, False, 8),
    ],
    ids=["triangle", "K5", "path-closed", "path-open", "star-closed", "star-open", "konigsberg-closed", "konigsberg-open"],
)
def test_known_optima(edges, closed, expected, method):
    graph = make_graph(edges)
    solution = solve_cpp(graph, closed=closed, method=method)
    assert_valid_cover(graph, solution, closed)
    assert solution.cost == expected


@pytest.mark.parametrize("edges", [TRIANGLE, K5])
def test_eulerian_graph_needs_no_duplication(edges):
    solution = solve_cpp(make_graph(edges))
    assert solution.deadheads == []


def test_semi_eulerian_open_walk_is_eulerian_path():
    graph = make_graph(PATH)
    solution = solve_cpp(graph, closed=False)
    assert solution.deadheads == []
    assert {solution.walks[0][0][0], solution.walks[0][-1][1]} == {0, 3}


def test_parallel_edges_duplicate_the_cheapest_copy():
    graph = make_graph([(0, 1, 5), (0, 1, 2), (0, 1, 7)])
    solution = solve_cpp(graph)
    assert [edge[2] for edge in solution.deadheads] == [2]
    assert solution.cost == 16


def test_self_loops_are_traversed():
    graph = make_graph([(0, 0, 4), (0, 1, 1)])
    solution = solve_cpp(graph)
    assert_valid_cover(graph, solution, closed=True)
    assert solution.cost == 6


def test_one_walk_per_component_and_isolated_vertices_ignored():
    edges = TRIANGLE + [(u + 4, v + 4, w) for u, v, w in PATH]
    graph = make_graph(edges)
    solution = solve_cpp(graph)
    assert len(solution.walks) == 2
    assert_valid_cover(graph, solution, closed=True)
    assert solution.cost == 6 + 12


def test_negative_weight_is_rejected():
    with pytest.raises(ValueError):
        solve_cpp(make_graph([(0, 1, -1)]))


def test_t_join_rejects_odd_terminal_set():
    with pytest.raises(ValueError):
        minimum_t_join(nx.path_graph(3), {0})


@pytest.mark.parametrize("closed", [True, False], ids=["closed", "open"])
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("seed", range(60))
def test_matches_brute_force_on_random_multigraphs(seed, method, closed):
    graph = random_graph(seed)
    solution = solve_cpp(graph, closed=closed, method=method)
    assert_valid_cover(graph, solution, closed)
    assert solution.cost == brute_force_cost(graph, closed)


@pytest.mark.parametrize(
    "instance, closed, expected, methods",
    [
        ("hard_to_choose", True, 1200, METHODS),
        ("islands", True, 1916, METHODS),
        ("islands", False, 1914, METHODS),
        ("paris_map", True, 27927, ["sparse"]),
        ("paris_map", False, 27918, ["sparse"]),
    ],
)
def test_instances(instance, closed, expected, methods):
    graph = Graph(*parse_file(f"instances/{instance}.txt"))
    for method in methods:
        solution = solve_cpp(graph, closed=closed, method=method)
        assert_valid_cover(graph, solution, closed)
        assert solution.cost == expected
