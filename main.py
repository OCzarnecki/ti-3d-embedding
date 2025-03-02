from dataclasses import dataclass
from typing import Generic, TypeVar, Literal, Any

from tqdm.auto import tqdm
from matplotlib.axes import Axes
from functools import cache

import numpy as np
import numpy.typing as npt
import networkx as nx
import torch


NodeMeta = TypeVar("NodeMeta")
EdgeMeta = TypeVar("EdgeMeta")

EPS = 1e-10

EdgeType = tuple[EdgeMeta, int, int]


@dataclass(frozen=True)
class Edge(Generic[EdgeMeta]):
    i: int
    j: int
    meta: EdgeMeta
    weight: float = 1.0


class Graph(Generic[NodeMeta, EdgeMeta]):
    def __init__(
        self,
        node_meta: NodeMeta,
        edges: list[Edge],
    ):
        self.node_meta = node_meta
        self.edges = edges
        self._validate()

    def to_nx(self, directed: bool):
        if directed:
            G = nx.Digraph()
        else:
            G = nx.Graph()

        for node in self.node_meta:
            G.add_node(node)

        # Add edges with attributes
        for edge in self.edges:
            if edge.i > edge.j:
                G.add_edge(self.node_meta[edge.i], self.node_meta[edge.j], meta=edge.meta)

        return G

    @property
    def num_nodes(self):
        return len(self.node_meta)

    def _validate(self):
        for edge in self.edges:
            if not isinstance(edge.i, int) or not isinstance(edge.j, int):
                raise ValueError("Edge indexes must be integers")
            if not 0 <= edge.i < self.num_nodes:
                raise IndexError(f"Edge {edge} out of range")

    def adj_matrix(self) -> npt.NDArray:
        matrix = np.zeros((self.num_nodes, self.num_nodes))
        for edge in self.edges:
            matrix[edge.i, edge.j] = edge.weight
        return matrix

    @cache
    def pairwise_distances(self) -> npt.NDArray:
        distances = np.ndarray((self.num_nodes, self.num_nodes))
        distances[:] = np.inf
        for u in range(self.num_nodes):
            distances[u, u] = 0

        for edge in self.edges:
            for u in range(self.num_nodes):
                for v in range(self.num_nodes):
                    via_edge = distances[u, edge.i] + edge.weight + distances[edge.j, v]
                    if distances[u, v] > via_edge:
                        distances[u, v] = via_edge
                    via_edge_rev = distances[u, edge.j] + edge.weight + distances[edge.i, v]
                    if distances[u, v] > via_edge_rev:
                        distances[u, v] = via_edge_rev
        return distances

    def plot_2d(
        self,
        ax: Axes,
        positions: npt.NDArray[np.float64],
        edge_kwargs: dict[str, Any] | None = None,
    ):
        if edge_kwargs is None:
            edge_kwargs = {}

        artists = []
        for edge in self.edges:
            x1, y1 = positions[:, edge.i]
            x2, y2 = positions[:, edge.j]
            artists += (
                ax.plot(
                    [x1, x2], [y1, y2],
                    **edge_kwargs,
                )
            )
        return artists


@dataclass(frozen=True)
class ChessNode:
    name: str
    color: Literal["black", "white"]


def chessboard(n: int):
    def idx(x, y):
        return n * x + y
    nodes = []
    edges = []
    for i in range(n):
        for j in range(n):
            name = chr(ord("a") + i) + str(j + 1)
            color = "black" if (i + j) % 2 == 0 else "white"
            nodes.append(ChessNode(name, color))
            if i + 1 < n:
                edges.append(Edge(idx(i, j), idx(i + 1, j), None))
                edges.append(Edge(idx(i + 1, j), idx(i, j), None))
            if j + 1 < n:
                edges.append(Edge(idx(i, j), idx(i, j + 1), None))
                edges.append(Edge(idx(i, j + 1), idx(i, j), None))
    return Graph(nodes, edges)


@dataclass(frozen=True)
class GameboardNode:
    idx: int


def ti_board(n: int):
    if n % 2 != 1:
        raise ValueError("Hex grid dimensions must be odd.")

    rows = []
    edges = []
    nodes = []
    flat_idx = 0

    first_row_odd = ((n - 1) // 2) % 2 == 0

    num_rows = 2 * n - 1

    for row_idx in range(num_rows):
        row = []
        if row_idx % 2 == (0 if first_row_odd else 1):
            num_cols = (n - 1) // 2 + 1
            for col in range(num_cols):
                node = GameboardNode(flat_idx)
                row.append((flat_idx, node))
                nodes.append(node)
                if row_idx >= 1:
                    if col >= 1:
                        neibor_idx, _ = rows[row_idx - 1][col - 1]
                        edges.append(Edge(flat_idx, neibor_idx, None))
                    if col <= num_cols - 2:
                        neibor_idx, _ = rows[row_idx - 1][col]
                        edges.append(Edge(flat_idx, neibor_idx, None))
                if row_idx >= 2:
                    neibor_idx, _ = rows[row_idx - 2][col]
                    edges.append(Edge(flat_idx, neibor_idx, None))
                flat_idx += 1
        else:
            num_cols = (n - 1) // 2
            for col in range(num_cols):
                node = GameboardNode(flat_idx)
                row.append((flat_idx, node))
                nodes.append(node)
                if row_idx >= 1:
                    neibor_idx, _ = rows[row_idx - 1][col]
                    edges.append(Edge(flat_idx, neibor_idx, None))
                    if col <= num_cols - 1:
                        neibor_idx, _ = rows[row_idx - 1][col + 1]
                        edges.append(Edge(flat_idx, neibor_idx, None))
                if row_idx >= 2:
                    neibor_idx, _ = rows[row_idx - 2][col]
                    edges.append(Edge(flat_idx, neibor_idx, None))
                flat_idx += 1
        rows.append(row)
    return Graph(nodes, edges)


@dataclass(frozen=True)
class Planariser:
    graph: Graph
    num_dims: int
    loss_order: float = 2.0
    seed: int | None = None

    def _init_params(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        points = torch.nn.parameter.Parameter(
            torch.Tensor(self.num_dims, self.graph.num_nodes)
        )
        torch.nn.init.normal_(points)
        return points

    def _step(
        self,
        points,
        graph_distances,
        optimizer,
    ) -> float:
        deltas = points.unsqueeze(1) - points.unsqueeze(2)
        euclidian_distances = (
            (deltas ** self.loss_order).abs().sum(axis=0) + EPS
        ) ** (1 / self.loss_order)

        c = 1  # (euclidian_distances ** 2).sum()

        loss = (
            ((euclidian_distances - graph_distances).abs() ** 2)
            / (c + EPS)
            # / euclidian_distances
        ).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def _get_optimizer(
        self,
        points,
    ):
        return torch.optim.SGD([points])

    def iter_embedddings(
        self,
        num_iterations: int = 1000,
        iterations_per_yield: int = None,
    ) -> npt.NDArray:
        points = self._init_params()
        graph_distances = torch.Tensor(self.graph.pairwise_distances())
        optimizer = self._get_optimizer(points)

        pbar = tqdm(list(range(num_iterations)))
        for iteration in pbar:
            if iteration % iterations_per_yield == 0:
                yield points.detach()

            loss = self._step(
                points,
                graph_distances,
                optimizer,
            )
            pbar.set_description(f"Loss: {loss:.2e}")

        return points.detach()

    def fit_embeddding(
        self,
        num_iterations: int = 1000,
    ) -> npt.NDArray:
        points = self._init_params()
        graph_distances = torch.Tensor(self.graph.pairwise_distances())
        optimizer = self._get_optimizer(points)

        pbar = tqdm(list(range(num_iterations)))
        for iteration in pbar:
            loss = self._step(
                points,
                graph_distances,
                optimizer,
            )
            pbar.set_description(f"Loss: {loss:.2e}")

        return points.detach()
