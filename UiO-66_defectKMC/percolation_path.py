"""
percolation_path.py -- 1-D pathway displacement over the node-defect graph
==========================================================================

A MONITOR for the mono-bearing-node transport network (see cage_graph.py). Once
missing linkers connect mono-bearing nodes into a cell-spanning cluster, this
measures the geometry of that 1-D pathway through the Zr6 node centres.

Feature returned (same keys the scripts use):
  percolates, rank,
  displacement_vector / displacement_magnitude / displacement_over_cell /
  displacement_over_cell_xyz   (net cell-spanning displacement; |D| ~ one
  cell-edge for a direct 1-D channel),
  path_length / tortuosity,
  and (when not spanning) extent_over_cell: the largest unwrapped node-centre
  reach of the biggest mono cluster, as a continuous approach-to-percolation
  signal.

`min_missing` is accepted for API compatibility but unused: the node-defect
criterion is binary (a node bears a mono or not), with no window threshold.
"""
from __future__ import annotations

import heapq
import numpy as np

from uio66_sites import periodic_components


def _bfs_offsets(start, nb, off, active):
    """Accumulated image offset of every node in one cluster (BFS)."""
    from collections import deque
    seen = {start: np.zeros(3, int)}
    q = deque([start])
    while q:
        u = q.popleft()
        for v, o in zip(nb[u], off[u]):
            v = int(v)
            if v in active and v not in seen:
                seen[v] = seen[u] + o
                q.append(v)
    return seen


def _shortest_spanning_path(nb, off, node_frac, cell, seed, Wtarget, bound=3):
    """Dijkstra in the covering graph (node, image-offset) from (seed,0) to
    (seed, Wtarget), weighted by node-centre step distance."""
    start = (seed, (0, 0, 0))
    target = (seed, tuple(int(x) for x in Wtarget))
    dist = {start: 0.0}; prev = {start: None}
    pq = [(0.0, start)]
    while pq:
        d, (c, o) = heapq.heappop(pq)
        if (c, o) == target:
            path = []; node = (c, o)
            while node is not None:
                path.append(node); node = prev[node]
            return path[::-1], d
        if d > dist.get((c, o), np.inf):
            continue
        for v, ov in zip(nb[c], off[c]):
            v = int(v); no = tuple(int(o[k] + ov[k]) for k in range(3))
            if max(abs(x) for x in no) > bound:
                continue
            step = np.linalg.norm((node_frac[v] + ov - node_frac[c]) @ cell)
            nd = d + step
            if nd < dist.get((v, no), np.inf):
                dist[(v, no)] = nd; prev[(v, no)] = (c, o)
                heapq.heappush(pq, (nd, (v, no)))
    return None, np.inf


def pathway_metrics(cg, occ, min_missing=1):
    """Geometry of the 1-D mono-node transport pathway for one occupancy."""
    occ = np.asarray(occ)
    active = set(int(x) for x in np.flatnonzero(cg.mono_mask(occ)))
    nb, off = cg.mono_adjacency(occ)
    edge = np.linalg.norm(cg.cell, axis=1)
    out = {"percolates": False, "rank": 0,
           "n_mono_nodes": len(active), "n_clusters": 0,
           "largest_mono_cluster": 0}
    if not active:
        out["extent_over_cell"] = 0.0
        return out

    comps = periodic_components(active, nb, off)
    out["n_clusters"] = len(comps)
    out["largest_mono_cluster"] = max((c["size"] for c in comps), default=0)
    top = max(comps, key=lambda c: (c["percolation_rank"], c["size"]))
    rank = top["percolation_rank"]
    out["rank"] = rank

    if rank == 0:
        seen = _bfs_offsets(top["members"][0], nb, off, active)
        pos = np.array([(cg.node_frac[c] + seen[c]) @ cg.cell for c in top["members"]])
        span = pos.max(0) - pos.min(0)
        out["extent_over_cell"] = float((span / edge).max())
        return out

    W = top["wrap_vectors"]
    Wp = W[int(np.argmin(np.linalg.norm(W @ cg.cell, axis=1)))]
    D = Wp @ cg.cell
    out["percolates"] = True
    out["wrap_vector"] = tuple(int(x) for x in Wp)
    out["displacement_vector"] = D
    out["displacement_magnitude"] = float(np.linalg.norm(D))
    out["displacement_over_cell"] = float(np.linalg.norm(D) / edge.min())
    out["displacement_over_cell_xyz"] = D / edge
    path, length = _shortest_spanning_path(nb, off, cg.node_frac, cg.cell,
                                           top["members"][0], Wp)
    if path is not None:
        out["path_length"] = float(length)
        out["path_n_nodes"] = len(path)
        out["tortuosity"] = float(length / out["displacement_magnitude"])
    return out


def describe(m: dict) -> str:
    if m["percolates"]:
        s = (f"PERCOLATES rank {m['rank']}  wrap {m['wrap_vector']}  "
             f"|displacement| {m['displacement_magnitude']:.1f} A "
             f"(= {m['displacement_over_cell']:.2f} cell-edges)")
        if "tortuosity" in m:
            s += f"  path {m['path_length']:.1f} A over {m['path_n_nodes']} nodes  tortuosity {m['tortuosity']:.2f}"
        return s
    return (f"no percolation (rank 0)  largest mono cluster {m['largest_mono_cluster']} "
            f"nodes  extent {m.get('extent_over_cell', 0):.2f} cell-edges")
