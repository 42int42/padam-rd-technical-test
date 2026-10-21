"""Exact solver for the undirected Chinese Postman Problem (route inspection).

For each connected component C = (V, E) of the input graph:

1. T = {v in V : deg(v) odd}. By the handshaking lemma, sum deg(v) = 2|E|, so |T| is even.
2. Compute a minimum-weight T-join J: an edge multiset whose odd-degree vertices are
   exactly T. For non-negative weights, c(J) is the weight of a minimum perfect matching
   of T under shortest-path distances, with J being the union of the matched paths
   (Edmonds & Johnson, 1973).
3. In E + J every degree is even, so Hierholzer's algorithm extracts an Eulerian circuit
   of weight c(E) + c(J). Every covering closed walk duplicates the edges of some T-join,
   so this walk is optimal.

Two exact T-join backends are available:

- "dense": Dijkstra from every terminal, then Edmonds' blossom algorithm on the metric
  closure K_T (networkx). Cost is O(|T| (|E| + |V| log |V|) + |T|^3).
- "sparse": sparse blossom (PyMatching) run directly on C, which never builds the
  closure. This handles thousands of terminals in well under a second.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Literal

import networkx as nx

from graph import Edge, Graph

Node = Hashable
Method = Literal["auto", "dense", "sparse"]

DENSE_MAX_TERMINALS = 300
_SOURCE, _SINK = -1, -2


@dataclass(frozen=True)
class PostmanSolution:
    """One covering walk per connected component, with edges oriented in travel order."""

    walks: list[list[Edge]]
    deadheads: list[Edge]
    base_cost: int

    @property
    def cost(self) -> int:
        return sum(edge[2] for walk in self.walks for edge in walk)

    @property
    def deadhead_cost(self) -> int:
        return sum(edge[2] for edge in self.deadheads)


def solve_cpp(graph: Graph, *, closed: bool = True, method: Method = "auto") -> PostmanSolution:
    """Compute a minimum-weight walk covering every edge in each connected component.

    With ``closed=False``, a walk may end at a vertex other than its start. The optimum
    then leaves one pair of odd vertices unmatched (see `_open_terminals`).
    """
    if any(edge[2] < 0 for edge in graph.edges):
        raise ValueError("Chinese Postman requires non-negative edge weights.")

    walks, deadheads = [], []
    for edge_ids in _component_edge_ids(graph):
        walk, duplicated = _solve_component(graph, edge_ids, closed, method)
        walks.append(walk)
        deadheads.extend(graph.edges[i] for i in duplicated)
    return PostmanSolution(walks, deadheads, sum(edge[2] for edge in graph.edges))


def minimum_t_join(graph: nx.Graph, terminals: set[Node], method: Method = "auto") -> list[tuple[Node, Node]]:
    """Return the edges of a minimum-weight T-join of a simple graph with non-negative weights.

    With non-negative weights some minimum T-join is a forest, so every edge is used
    at most once.
    """
    if len(terminals) % 2:
        raise ValueError("A T-join exists only for |T| even.")
    if not terminals:
        return []
    if method == "auto":
        method = "dense" if len(terminals) <= DENSE_MAX_TERMINALS else "sparse"
    if method == "dense":
        return _t_join_dense(graph, terminals)
    if method == "sparse":
        return _t_join_sparse(graph, terminals)
    raise ValueError(f"Unknown method {method!r}.")


def _component_edge_ids(graph: Graph) -> list[list[int]]:
    """Group edge indices by connected component. Isolated vertices need no visit."""
    support = nx.Graph((edge[0], edge[1]) for edge in graph.edges)
    component_of = {v: k for k, nodes in enumerate(nx.connected_components(support)) for v in nodes}
    groups: dict[int, list[int]] = {}
    for i, edge in enumerate(graph.edges):
        groups.setdefault(component_of[edge[0]], []).append(i)
    return list(groups.values())


def _solve_component(
    graph: Graph, edge_ids: list[int], closed: bool, method: Method
) -> tuple[list[Edge], list[int]]:
    degree = Counter()
    for i in edge_ids:
        degree[graph.edges[i][0]] += 1
        degree[graph.edges[i][1]] += 1
    terminals = {v for v, d in degree.items() if d % 2}

    support = _cheapest_edge_graph(graph, edge_ids)
    if not closed and terminals:
        terminals = _open_terminals(support, terminals)

    join = minimum_t_join(support, terminals, method)
    duplicated = [support.edges[e]["eid"] for e in join if support.edges[e]["eid"] is not None]
    return _eulerian_walk(graph, edge_ids + duplicated), duplicated


def _cheapest_edge_graph(graph: Graph, edge_ids: list[int]) -> nx.Graph:
    """Simple graph keeping the cheapest copy of each parallel class. Loops never help a T-join."""
    support = nx.Graph()
    for i in edge_ids:
        u, v, w = graph.edges[i][:3]
        if u != v and (not support.has_edge(u, v) or w < support[u][v]["weight"]):
            support.add_edge(u, v, weight=w, eid=i)
    return support


def _open_terminals(support: nx.Graph, terminals: set[Node]) -> set[Node]:
    """Reduce the open-walk problem to a single T'-join in place (mutates `support`).

    Add virtual vertices s, t, each joined to every terminal by an edge of weight M,
    and set T' = T + {s, t}. A T'-join with exactly two virtual edges s-a, t-b is a
    (T - {a, b})-join plus 2M, i.e. an open walk from a to b. One with k >= 4 virtual
    edges frees a set F of terminals with |F| <= k. Dropping all but two virtual
    edges and re-pairing F with real paths saves at least (k - 2) M - (|F| - 2) diam / 2,
    which is positive when M > diam / 2. Taking M = ecc(r) + 1 >= diam / 2 + 1 is safe,
    so the optimum never uses more than two virtual edges.
    """
    root = next(iter(support))
    penalty = max(nx.single_source_dijkstra_path_length(support, root).values()) + 1
    for virtual in (_SOURCE, _SINK):
        support.add_edges_from(((virtual, t) for t in terminals), weight=penalty, eid=None)
    return terminals | {_SOURCE, _SINK}


def _t_join_dense(graph: nx.Graph, terminals: set[Node]) -> list[tuple[Node, Node]]:
    """Minimum perfect matching on the metric closure K_T, then the symmetric difference of matched paths."""
    ordered = sorted(terminals)
    closure = nx.Graph()
    predecessors = {}
    for k, s in enumerate(ordered):
        pred, dist = nx.dijkstra_predecessor_and_distance(graph, s)
        predecessors[s] = pred
        closure.add_weighted_edges_from((s, t, dist[t]) for t in ordered[k + 1 :] if t in dist)

    matching = nx.min_weight_matching(closure)
    if 2 * len(matching) != len(terminals):
        raise ValueError("Terminals are not pairwise connected; no T-join exists.")

    join: set[tuple[Node, Node]] = set()
    for s, t in matching:
        s, t = (s, t) if s in predecessors else (t, s)
        while t != s:
            p = predecessors[s][t][0]
            join ^= {(min(p, t), max(p, t))}
            t = p
    return list(join)


def _t_join_sparse(graph: nx.Graph, terminals: set[Node]) -> list[tuple[Node, Node]]:
    """Sparse blossom (Higgott & Gidney, 2023): exact minimum-weight T-join on the original sparse graph.

    PyMatching rounds weights to integers scaled by about 2^24 / max_weight. For
    integer inputs the total rounding error stays far below 1/2, so the optimum is exact.
    """
    try:
        import numpy as np
        import pymatching
    except ImportError as exc:
        raise ImportError("method='sparse' requires the `pymatching` package.") from exc

    index = {v: k for k, v in enumerate(graph)}
    edges = list(graph.edges(data="weight"))
    matching = pymatching.Matching()
    for k, (u, v, w) in enumerate(edges):
        matching.add_edge(index[u], index[v], fault_ids=k, weight=w)
    syndrome = np.zeros(len(index), dtype=np.uint8)
    syndrome[[index[t] for t in terminals]] = 1
    correction = matching.decode(syndrome)
    return [edges[k][:2] for k in np.flatnonzero(correction)]


def _eulerian_walk(graph: Graph, edge_ids: list[int]) -> list[Edge]:
    """Hierholzer's algorithm on the augmented multigraph, which has 0 or 2 odd vertices. O(|E|)."""
    multigraph = nx.MultiGraph()
    for i in edge_ids:
        multigraph.add_edge(graph.edges[i][0], graph.edges[i][1], eid=i)
    odd = [v for v, d in multigraph.degree if d % 2]
    source = odd[0] if odd else graph.edges[edge_ids[0]][0]

    walk = []
    for u, v, k in nx.eulerian_path(multigraph, source=source, keys=True):
        weight = graph.edges[multigraph.edges[u, v, k]["eid"]][2]
        walk.append((u, v, weight, graph.vertices[u], graph.vertices[v]))
    return walk
