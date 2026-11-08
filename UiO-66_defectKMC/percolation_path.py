"""
percolation_path.py -- displacement metrics for a percolating guest pathway
============================================================================

A MONITOR (never a driver) for a kMC trajectory. Binary percolation is decided
by the rank (CageGraph.percolation); this module characterises the geometry of
the pathway once it exists, and gives a continuous "how close to percolating"
signal before it does -- both measured through cage-centre positions, in units
comparable to the cell.

Definitions
-----------
Open-window cage graph: cages are nodes, a window is an edge when it is open
(>= `min_missing` linkers absent, or supply your own open mask). A guest step
goes cage -> cage through an open window.

For a PERCOLATING pathway (rank >= 1):
  - displacement vector  D = W . cell, where W is the integer lattice vector the
    pathway wraps around (the net cage-centre displacement accumulated walking
    the spanning path from a cage to its own periodic image; route-independent
    by telescoping). |D| is the headline metric.
  - |D| / cell-edge tells directness: 1.0 = one straight crossing along a cell
    axis; ~1.41 = a face-diagonal span; ~1.73 = a body-diagonal span.
  - path length L = sum of cage-centre step distances along the shortest
    spanning path (route-dependent, >= |D|).
  - tortuosity  = L / |D|  (1.0 = straight channel; higher = meandering).

For a NON-percolating structure (rank 0), for monitoring the approach to
percolation:
  - extent = largest cage-centre separation within the biggest open cluster,
    unwrapped through the connectivity (max over the 3 cell axes of the span).
  - extent / cell-edge rises from ~0 toward 1 as a cluster grows to span the
    cell; it crosses ~1 at percolation. This is the continuous order parameter
    to plot along a kMC trajectory.

Depends on cage_graph.CageGraph and numpy.
"""

from __future__ import annotations

import heapq
from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
def open_windows_by_count(cg, occ, min_missing: int = 1) -> np.ndarray:
    """Boolean open-window mask: window open if >= min_missing of its 3 linkers
    are absent. min_missing=1 corresponds to the calibrated 5 A guest aperture
    (one missing linker opens a window from ~3.8 A to ~6.3 A)."""
    return (occ[cg.win_sites] == 0).sum(axis=1) >= min_missing


def _open_adjacency(cg, open_mask):
    """cage -> list of (neighbour cage, image offset, step distance)."""
    nb = [[] for _ in range(cg.n_cages)]
    for w in np.flatnonzero(open_mask):
        (c1, s1), (c2, s2) = cg.win_cages[w]
        off = np.asarray(s2) - np.asarray(s1)
        step = np.linalg.norm((cg.cage_frac[c2] + off - cg.cage_frac[c1]) @ cg.cell)
        nb[c1].append((c2, off, step))
        nb[c2].append((c1, -off, step))
    return nb


def _components(cg, adj):
    """Connected open-cage clusters (ignoring images); returns list of cage lists."""
    seen, comps = set(), []
    for s in range(cg.n_cages):
        if s in seen or not adj[s]:
            continue
        q, comp = deque([s]), []
        seen.add(s)
        while q:
            c = q.popleft(); comp.append(c)
            for nb, _, _ in adj[c]:
                if nb not in seen:
                    seen.add(nb); q.append(nb)
        comps.append(comp)
    # include isolated open cages too? only cages with >=1 open window are nodes here
    return comps


def _wrap_vectors(cg, adj, comp):
    """Wrap vectors of one cluster (image-aware BFS); rank = matrix rank."""
    seen = {comp[0]: np.zeros(3, int)}
    q, wraps = deque([comp[0]]), []
    while q:
        c = q.popleft()
        for nb, off, _ in adj[c]:
            sh = seen[c] + off
            if nb not in seen:
                seen[nb] = sh; q.append(nb)
            else:
                w = sh - seen[nb]
                if np.any(w != 0):
                    wraps.append(tuple(int(x) for x in w))
    W = np.array(sorted(set(wraps)), int).reshape(-1, 3)
    rank = int(np.linalg.matrix_rank(W)) if len(W) else 0
    return W, rank, seen


def _shortest_spanning_path(cg, adj, seed, Wtarget, bound=3):
    """Dijkstra in the covering graph (cage, image-offset) from (seed,0) to
    (seed, Wtarget); returns (path_nodes, length) or (None, inf)."""
    start = (seed, (0, 0, 0))
    target = (seed, tuple(int(x) for x in Wtarget))
    dist = {start: 0.0}
    prev = {start: None}
    pq = [(0.0, start)]
    while pq:
        d, (c, o) = heapq.heappop(pq)
        if (c, o) == target:
            path = []
            node = (c, o)
            while node is not None:
                path.append(node); node = prev[node]
            return path[::-1], d
        if d > dist.get((c, o), np.inf):
            continue
        for nb, off, step in adj[c]:
            no = tuple(int(o[k] + off[k]) for k in range(3))
            if max(abs(x) for x in no) > bound:
                continue
            nd = d + step
            if nd < dist.get((nb, no), np.inf):
                dist[(nb, no)] = nd
                prev[(nb, no)] = (c, o)
                heapq.heappush(pq, (nd, (nb, no)))
    return None, np.inf


# ---------------------------------------------------------------------------
def pathway_metrics(cg, occ=None, open_mask=None, min_missing: int = 1) -> dict:
    """Displacement metrics for the guest pathway of one structure.

    Supply either `occ` (and min_missing) or a precomputed `open_mask`.
    """
    if open_mask is None:
        if occ is None:
            raise ValueError("give occ or open_mask")
        open_mask = open_windows_by_count(cg, occ, min_missing)
    adj = _open_adjacency(cg, open_mask)
    comps = _components(cg, adj)
    cell_edges = np.linalg.norm(cg.cell, axis=1)

    out = {
        "percolates": False,
        "rank": 0,
        "n_open_windows": int(open_mask.sum()),
        "largest_open_cluster": max((len(c) for c in comps), default=0),
        "n_open_clusters": len(comps),
    }
    if not comps:
        out["extent_over_cell"] = 0.0
        return out

    # pick the cluster with the highest wrap rank (then largest)
    ranked = []
    for comp in comps:
        W, rank, seen = _wrap_vectors(cg, adj, comp)
        ranked.append((rank, len(comp), comp, W, seen))
    ranked.sort(key=lambda r: (r[0], r[1]), reverse=True)
    rank, size, comp, W, seen = ranked[0]
    out["rank"] = rank

    if rank == 0:
        # continuous approach-to-percolation: unwrapped spatial extent / cell
        pos = np.array([(cg.cage_frac[c] + seen[c]) @ cg.cell for c in comp])
        span = pos.max(0) - pos.min(0)                     # per-axis span (A)
        out["extent_vector"] = span
        out["extent_over_cell"] = float((span / cell_edges).max())
        out["extent_over_cell_xyz"] = (span / cell_edges)
        return out

    # percolating: primary (minimal-magnitude) wrap vector
    mags = np.linalg.norm(W @ cg.cell, axis=1)
    Wp = W[int(np.argmin(mags))]
    D = Wp @ cg.cell
    out["percolates"] = True
    out["wrap_vector"] = tuple(int(x) for x in Wp)
    out["all_wrap_vectors"] = [tuple(int(x) for x in w) for w in W]
    out["displacement_vector"] = D
    out["displacement_magnitude"] = float(np.linalg.norm(D))
    out["displacement_over_cell"] = float(np.linalg.norm(D) / cell_edges.min())
    out["displacement_over_cell_xyz"] = (D / cell_edges)

    # a representative shortest spanning path and its length (tortuosity)
    seed = comp[0]
    path, length = _shortest_spanning_path(cg, adj, seed, Wp)
    if path is not None:
        centers = np.array([(cg.cage_frac[c] + np.array(o)) @ cg.cell for c, o in path])
        out["path_cages"] = [int(c) for c, _ in path]
        out["path_centers"] = centers
        out["path_length"] = float(length)
        out["path_n_cages"] = len(path)
        out["tortuosity"] = float(length / out["displacement_magnitude"])
    return out


def describe(m: dict) -> str:
    """One-line human summary of pathway_metrics output."""
    if m["percolates"]:
        s = (f"PERCOLATES rank {m['rank']}  wrap {m['wrap_vector']}  "
             f"|displacement| {m['displacement_magnitude']:.1f} A "
             f"(= {m['displacement_over_cell']:.2f} cell-edges)")
        if "tortuosity" in m:
            s += (f"  path {m['path_length']:.1f} A over {m['path_n_cages']} cages"
                  f"  tortuosity {m['tortuosity']:.2f}")
        return s
    return (f"no percolation (rank 0)  largest open cluster {m['largest_open_cluster']} "
            f"cages  extent {m.get('extent_over_cell', 0):.2f} cell-edges")


if __name__ == "__main__":
    print(__doc__)
