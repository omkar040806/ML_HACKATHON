"""Directed road graph for one trip (owner: Person 1).

* Nodes and arcs come straight from the ``.nodes`` / ``.arcs`` files.
* Arcs are straight segments between their two nodes (the files contain no
  shape points); lengths are measured in the trip's local projection.
* Some trips contain *duplicate arcs* (several arc ids with the same
  ``(from, to)`` pair, i.e. identical geometry).  They are indistinguishable
  from GPS, so each arc is mapped to a **canonical arc id** (the smallest id
  in its duplicate group).  Candidates, labels and metrics all use canonical
  ids.  In train + student_test only 3 of 34 813 reference route entries use a
  non-canonical duplicate, so the information loss is negligible.
* Shortest paths use a bounded Dijkstra with an LRU cache because the HMM
  issues many short queries from the same source nodes.
"""
from __future__ import annotations

import heapq
import math
from collections import OrderedDict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .geo import LocalProjection, bearing_deg


class RoadGraph:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, projection: LocalProjection,
                 cache_size: int = 20000):
        self.proj = projection
        self.n_nodes = int(len(nodes_lonlat))
        self.n_arcs = int(len(arcs))
        self.node_lon = nodes_lonlat[:, 0]
        self.node_lat = nodes_lonlat[:, 1]
        self.node_x, self.node_y = projection.forward(self.node_lon, self.node_lat)
        self.arc_from = arcs[:, 0].astype(np.int64)
        self.arc_to = arcs[:, 1].astype(np.int64)
        self.ax, self.ay = self.node_x[self.arc_from], self.node_y[self.arc_from]
        self.bx, self.by = self.node_x[self.arc_to], self.node_y[self.arc_to]
        self.arc_len = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.arc_bearing = bearing_deg(self.bx - self.ax, self.by - self.ay)

        # canonical ids for duplicate (from, to) pairs
        key = self.arc_from * (self.n_nodes + 1) + self.arc_to
        uniq, first_idx, inv = np.unique(key, return_index=True, return_inverse=True)
        self.canonical = first_idx[inv].astype(np.int64)
        self.is_canonical = self.canonical == np.arange(self.n_arcs)

        # reverse arc (to -> from), canonical, or -1
        rkey = self.arc_to * (self.n_nodes + 1) + self.arc_from
        pos = np.searchsorted(uniq, rkey)
        pos_c = np.clip(pos, 0, len(uniq) - 1)
        found = uniq[pos_c] == rkey
        self.reverse_arc = np.where(found, first_idx[pos_c], -1).astype(np.int64)

        can = self.is_canonical
        self.out_degree = np.bincount(self.arc_from[can], minlength=self.n_nodes)
        self.in_degree = np.bincount(self.arc_to[can], minlength=self.n_nodes)

        self._adj: list[list[tuple[int, float, int]]] | None = None
        self._cache: OrderedDict[int, tuple[float, dict, dict]] = OrderedDict()
        self._cache_size = cache_size

    # ------------------------------------------------------------------ build
    @classmethod
    def from_trip(cls, trip, projection: LocalProjection | None = None) -> "RoadGraph":
        if projection is None:
            projection = LocalProjection.from_points(trip.track[:, 0], trip.track[:, 1])
        return cls(trip.nodes, trip.arcs, projection)

    def _adjacency(self):
        if self._adj is None:
            adj: list[list[tuple[int, float, int]]] = [[] for _ in range(self.n_nodes)]
            idx = np.nonzero(self.is_canonical)[0]
            for a, u, v, ln in zip(idx.tolist(), self.arc_from[idx].tolist(),
                                   self.arc_to[idx].tolist(), self.arc_len[idx].tolist()):
                adj[u].append((v, ln, a))
            self._adj = adj
        return self._adj

    # --------------------------------------------------------------- queries
    def connected(self, a: int, b: int) -> bool:
        """True if arc ``b`` can directly follow arc ``a`` (or a == b)."""
        return a == b or self.arc_to[a] == self.arc_from[b]

    def bounded_dijkstra(self, source: int, limit_m: float):
        """Distances (m) and predecessor arcs of all nodes within ``limit_m`` of ``source``."""
        hit = self._cache.get(source)
        if hit is not None and hit[0] >= limit_m:
            self._cache.move_to_end(source)
            return hit[1], hit[2]
        adj = self._adjacency()
        dist = {source: 0.0}
        pred: dict[int, int] = {}
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, math.inf):
                continue
            for v, ln, a in adj[u]:
                nd = d + ln
                if nd <= limit_m and nd < dist.get(v, math.inf):
                    dist[v] = nd
                    pred[v] = a
                    heapq.heappush(heap, (nd, v))
        self._cache[source] = (limit_m, dist, pred)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return dist, pred

    def network_distance(self, src_node: int, dst_node: int, limit_m: float) -> float:
        if src_node == dst_node:
            return 0.0
        dist, _ = self.bounded_dijkstra(src_node, limit_m)
        d = dist.get(dst_node, math.inf)
        return d if d <= limit_m else math.inf

    def shortest_path_arcs(self, src_node: int, dst_node: int, limit_m: float) -> list[int] | None:
        """Arc ids of the shortest path ``src_node -> dst_node`` or ``None``."""
        if src_node == dst_node:
            return []
        dist, pred = self.bounded_dijkstra(src_node, limit_m)
        if dst_node not in dist or dist[dst_node] > limit_m:
            return None
        path = []
        node = dst_node
        guard = 0
        while node != src_node:
            a = pred[node]
            path.append(a)
            node = int(self.arc_from[a])
            guard += 1
            if guard > self.n_arcs:  # defensive: should never happen
                return None
        return path[::-1]

    # ------------------------------------------------------------ diagnostics
    def component_stats(self) -> dict:
        """Weak/strong connectivity statistics used by the EDA."""
        can = self.is_canonical
        m = coo_matrix((np.ones(int(can.sum())), (self.arc_from[can], self.arc_to[can])),
                       shape=(self.n_nodes, self.n_nodes)).tocsr()
        n_weak, lab_w = connected_components(m, directed=True, connection="weak")
        n_strong, lab_s = connected_components(m, directed=True, connection="strong")
        deg = self.out_degree + self.in_degree
        return {
            "n_nodes": self.n_nodes,
            "n_arcs": self.n_arcs,
            "n_duplicate_arcs": int((~can).sum()),
            "n_self_loops": int((self.arc_from == self.arc_to).sum()),
            "n_isolated_nodes": int((deg == 0).sum()),
            "n_weak_components": int(n_weak),
            "largest_weak_frac": float(np.bincount(lab_w).max() / self.n_nodes),
            "n_strong_components": int(n_strong),
            "largest_strong_frac": float(np.bincount(lab_s).max() / self.n_nodes),
            "two_way_arc_frac": float((self.reverse_arc[can] >= 0).mean()),
            "arc_len_median_m": float(np.median(self.arc_len)),
        }
