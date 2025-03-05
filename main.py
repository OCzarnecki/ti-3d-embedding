from dataclasses import dataclass, field
from typing import Generic, TypeVar, Literal, Any
from itertools import repeat
from collections import defaultdict

import math

from tqdm.auto import tqdm
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D
from functools import cache
from PIL import Image

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

    @cache
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

    @cache
    def to_index_mapping(self) -> dict[NodeMeta, int]:
        return {
            node: idx
            for idx, node in enumerate(self.node_meta)
        }

    @cache
    def adj_list(self) -> dict[NodeMeta, list[NodeMeta]]:
        adj_list = defaultdict(list)
        for edge in self.edges:
            adj_list[self.node_meta[edge.i]].append(self.node_meta[edge.j])
            adj_list[self.node_meta[edge.j]].append(self.node_meta[edge.i])
        return adj_list

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

    def plot_3d(
        self,
        ax: Axes3D,
        positions: npt.NDArray[np.float64],
        edge_kwargs: dict[str, Any] | None = None,
    ):
        if edge_kwargs is None:
            edge_kwargs = {}

        artists = []

        for idx, node in enumerate(self.node_meta):
            x1, y1, z1 = positions[:, idx]
            x2, y2, z2 = [x1, y1, 0.0]
            artists += (
                ax.plot(
                    [x1, x2], [y1, y2], [z1, z2],
                    **edge_kwargs,
                    color="grey",
                )
            )

        for edge in self.edges:
            x1, y1, z1 = positions[:, edge.i]
            x2, y2, z2 = positions[:, edge.j]
            artists += (
                ax.plot(
                    [x1, x2], [y1, y2], [z1, z2],
                    **edge_kwargs,
                )
            )

        return artists

    @cache
    def get_triangle_indices(self):
        def find_cycles(depth: int, origin: NodeMeta, current: NodeMeta):
            if depth == 0:
                if current is origin:
                    return [[]]
                else:
                    return []

            cycles = []
            for neighbor in self.adj_list()[current]:
                neigh_cycles = find_cycles(depth - 1, origin, neighbor)
                cycles += [
                    [neighbor] + cycle
                    for cycle in neigh_cycles
                ]

            return cycles

        triangles = []
        for node in self.node_meta:
            triangles += find_cycles(3, node, node)

        indices = {
            tuple(sorted([self.to_index_mapping()[idx] for idx in tri]))
            for tri in triangles
        }

        return list(indices)


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


@dataclass(frozen=True, eq=True)
class TileSpec:
    idx: int
    side: Literal["A", "B"] | None = None
    clockwise_rotations: int = 0


@dataclass(frozen=True)
class GameboardNode:
    idx: int
    tile_spec: TileSpec
    wormhole_labels: tuple[str, ...] = field(default_factory=tuple)


def ti_board(
    n: int,
    tiles: list[list[TileSpec | int]] | None = None,
    extra_connections: list[tuple[int, int]] | None = None,  # tuple of tile indices
    wormholes: dict[str, list[int]] | None = None,
):
    if n % 2 != 1:
        raise ValueError("Hex grid dimensions must be odd.")

    idx_to_wormhole = defaultdict(list)
    if wormholes is not None:
        for label, group in wormholes.items():
            for hole in group:
                idx_to_wormhole[hole].append(label)

    idx_to_wormhole = defaultdict(
        tuple,
        {
            idx: tuple(wormhole_labels)
            for idx, wormhole_labels in idx_to_wormhole.items()
        }
    )

    if tiles is None:
        tiles = repeat([TileSpec(1)] * ((n - 1) // 2 + 1))
    else:
        tiles = [
            [
                TileSpec(tile) if isinstance(tile, int) else tile
                for tile in tile_row
            ]
            for tile_row in tiles
        ]

    rows = []
    edges = []
    nodes = []
    flat_idx = 0

    first_row_odd = ((n - 1) // 2) % 2 == 0

    num_rows = 2 * n - 1

    for row_idx, tile_row in zip(range(num_rows), tiles):
        row = []
        if row_idx % 2 == (0 if first_row_odd else 1):
            num_cols = (n - 1) // 2 + 1
            padding = (num_cols - len(tile_row)) // 2
            row_tiles = [None] * padding + tile_row + [None] * padding
            # assert len(row_tiles) == num_cols, f"{num_cols=} {padding=} {len(row_tiles)=}"
            for col, tile in zip(range(num_cols), row_tiles):
                node = GameboardNode(flat_idx, tile)
                if tile is not None:
                    node = GameboardNode(flat_idx, tile, wormhole_labels=idx_to_wormhole[tile.idx])
                    row.append((flat_idx, node))
                    nodes.append(node)
                else:
                    node = None
                    row.append((-1, node))
                if node is not None:
                    if row_idx >= 1:
                        if col >= 1:
                            neibor_idx, neighbor = rows[row_idx - 1][col - 1]
                            if neighbor is not None:
                                edges.append(Edge(flat_idx, neibor_idx, None))
                        if col <= num_cols - 2:
                            neibor_idx, neighbor = rows[row_idx - 1][col]
                            if neighbor is not None:
                                edges.append(Edge(flat_idx, neibor_idx, None))
                    if row_idx >= 2:
                        neibor_idx, neighbor = rows[row_idx - 2][col]
                        if neighbor is not None:
                            edges.append(Edge(flat_idx, neibor_idx, None))
                    flat_idx += 1
        else:
            num_cols = (n - 1) // 2
            padding = (num_cols - len(tile_row)) // 2
            row_tiles = [None] * padding + tile_row + [None] * padding
            # assert len(row_tiles) == num_cols
            for col, tile in zip(range(num_cols), row_tiles):
                if tile is not None:
                    node = GameboardNode(flat_idx, tile, wormhole_labels=idx_to_wormhole[tile.idx])
                    row.append((flat_idx, node))
                    nodes.append(node)
                else:
                    node = None
                    row.append((-1, node))
                if node is not None:
                    if row_idx >= 1:
                        neibor_idx, neighbor = rows[row_idx - 1][col]
                        if neighbor is not None:
                            edges.append(Edge(flat_idx, neibor_idx, None))
                        if col <= num_cols - 1:
                            neibor_idx, neighbor = rows[row_idx - 1][col + 1]
                            if neighbor is not None:
                                edges.append(Edge(flat_idx, neibor_idx, None))
                    if row_idx >= 2:
                        neibor_idx, neighbor = rows[row_idx - 2][col]
                        if neighbor is not None:
                            edges.append(Edge(flat_idx, neibor_idx, None))
                    flat_idx += 1
        rows.append(row)

    if extra_connections is not None:
        ordered_tiles = [node.tile_spec.idx for node in nodes]
        for from_, to in extra_connections:
            edges.append(Edge(
                ordered_tiles.index(from_),
                ordered_tiles.index(to),
                None,
            ))

    return Graph(nodes, edges)


@dataclass(frozen=True)
class Planariser:
    graph: Graph
    num_dims: int
    loss_order: float = 2.0
    seed: int | None = None
    starting_positions: torch.Tensor | None = None
    optimization_speed: float = 1.0

    def _init_params(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
        if self.starting_positions is None:
            points = torch.nn.parameter.Parameter(
                torch.Tensor(self.num_dims, self.graph.num_nodes)
            )
            torch.nn.init.normal_(points)
        else:
            points = torch.nn.parameter.Parameter(
                self.starting_positions,
            )
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
        return torch.optim.SGD([points], lr=0.01 * self.optimization_speed)

    def iter_embedddings(
        self,
        num_iterations: int = 1000,
        iterations_per_yield: int = 1,
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


def load_tile(
    tile_spec: TileSpec,
):
    if tile_spec.side is None:
        suffix = ""
    else:
        suffix = tile_spec.side

    path = f"tiles/ST_{tile_spec.idx}{suffix}.png"
    img = Image.open(path)
    return img.rotate(
        -60 * tile_spec.clockwise_rotations
    )


def plot_tile(
    ax,
    tile_height,
    xy,
    tile_spec: TileSpec,
):
    tile_width = tile_height / math.cos(30 / 360 * 2 * math.pi)

    x, y = xy
    left = x - tile_width / 2
    right = x + tile_width / 2
    bottom = y - tile_height / 2
    top = y + tile_height / 2

    tile_im = load_tile(tile_spec)

    ax.imshow(
        tile_im,
        extent=(left, right, bottom, top),
    )
