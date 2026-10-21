from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

Coordinates = tuple[float, float]
Edge = tuple[int, int, int, Coordinates, Coordinates]


class Graph:
    def __init__(
        self,
        vertices: list[Coordinates],
        edges: list[Edge],
    ):
        """Basic constructor of a `Graph` instance.

        Parameters
        ----------
        vertices : list[Coordinates]
            List of vertices coordinates.

        edges : list[Edge]
            List of edges as tuple (id 1, id 2, weight, coordinates 1, coordinates 2).
        """
        self.vertices = vertices
        self.edges = edges

    def plot(self):
        """
        Plot the graph.
        """
        weights = list(set(edge[2] for edge in self.edges))
        colors = plt.get_cmap("viridis", len(weights))
        _, ax = plt.subplots()
        for i, weight in enumerate(weights):
            lines = [
                [edge[-2][::-1], edge[-1][::-1]]
                for edge in self.edges
                if edge[2] == weight
            ]
            ax.add_collection(
                LineCollection(
                    lines, colors=colors(i), alpha=0.7, label=f"weight {weight}"
                )
            )
        ax.plot()
        ax.legend()
        plt.title(f"#E={len(self.edges)}, #V={len(self.vertices)}")
        plt.show()

    @classmethod
    def display_path(cls, *, path: list[Edge]):
        return cls.display_paths(paths=[path])

    @staticmethod
    def display_paths(*, paths: list[list[Edge]]):
        colors = plt.get_cmap("viridis", len(paths))
        figure = plt.figure()
        ax = figure.add_subplot()
        for path_index, path in enumerate(paths):
            for edge_index, edge in enumerate(path):
                ax.annotate(
                    str(edge_index),
                    xytext=edge[3],
                    xy=edge[4],
                    arrowprops=dict(arrowstyle="->", color=colors(path_index)),
                )
        plt.show()
