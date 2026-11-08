"""
cage_graph.py -- node-defect percolation for UiO-66 (monocarboxylic-acid based)
===============================================================================

Percolation criterion:

    A metal-oxide node (Zr6 SBU) is DEFECTIVE when it carries a monocarboxylic
    acid, i.e. at least one of its 12 linker positions is a missing linker
    (mono) rather than an intact di-carboxylic linker.

A guest migrates through the network of mono-bearing nodes, hopping across the
missing linkers that connect them. Percolation = a cell-spanning cluster of such
mono-connected nodes.

This is binary and per-node (the node has a mono group or it does not), with no
threshold on how "open" a triangular window is -- that window concept was loose.
Because one missing linker places a mono on BOTH nodes it used to bridge, the
transport graph is exactly: nodes = Zr6 SBUs, edges = missing linkers.

Interface kept compatible with the rest of the pipeline: build_defect_graph(sg)
(alias build_cage_graph) returns a DefectGraph with .percolation(occ) and the
node geometry that percolation_path uses for the 1-D pathway displacement.

Depends on numpy and uio66_sites.periodic_components (image-aware spanning).
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from uio66_sites import periodic_components


@dataclass
class DefectGraph:
    cell: np.ndarray
    node_frac: np.ndarray                  # (N,3) Zr6 node centroids
    edges: list                            # one per linker site: (a, b, offset(3,))

    @property
    def n_nodes(self) -> int:
        return len(self.node_frac)

    @property
    def n_sites(self) -> int:
        return len(self.edges)

    # --- node coordination state ---------------------------------------------
    def mono_mask(self, occ) -> np.ndarray:
        """Boolean per node: True if the node bears >=1 mono (missing linker)."""
        occ = np.asarray(occ)
        m = np.zeros(self.n_nodes, bool)
        for s, (a, b, _) in enumerate(self.edges):
            if occ[s] == 0:
                m[a] = True; m[b] = True
        return m

    def mono_adjacency(self, occ):
        """Per-node neighbours/offsets using ONLY missing-linker (mono) edges."""
        occ = np.asarray(occ)
        nb = [[] for _ in range(self.n_nodes)]
        off = [[] for _ in range(self.n_nodes)]
        for s, (a, b, o) in enumerate(self.edges):
            if occ[s] == 0:
                o = np.asarray(o, int)
                nb[a].append(b); off[a].append(o)
                nb[b].append(a); off[b].append(-o)
        return ([np.array(x, int) for x in nb],
                [np.array(x, int).reshape(-1, 3) for x in off])

    # --- percolation ----------------------------------------------------------
    def percolation(self, occ) -> dict:
        """Spanning of the mono-bearing node network (image-aware)."""
        active = np.flatnonzero(self.mono_mask(occ))
        nb, off = self.mono_adjacency(occ)
        comps = periodic_components(active, nb, off)
        biggest = max((c["size"] for c in comps), default=0)
        return {
            "n_mono_nodes": int(len(active)),
            "mono_node_fraction": float(len(active) / max(self.n_nodes, 1)),
            "n_clusters": len(comps),
            "largest_mono_cluster": biggest,
            "percolation_rank": max((c["percolation_rank"] for c in comps), default=0),
        }

    def summary(self) -> dict:
        return {"n_nodes": self.n_nodes, "n_sites": self.n_sites}


# ---------------------------------------------------------------------------
def build_defect_graph(sg, verbose: bool = True) -> DefectGraph:
    """Build the node-defect graph from a SiteGraph (uio66_sites.build_site_graph).

    Each linker site becomes an edge (nodeA, nodeB, image-offset) between its two
    Zr6 nodes; the offset is recovered from the site midpoint geometry.
    """
    cell = np.asarray(sg.cell, float)
    nf = np.asarray(sg.node_frac, float)
    edges = []
    for s in range(sg.n_sites):
        a, b = (int(x) for x in sg.site_nodes[s])
        b_img = 2.0 * sg.site_frac[s] - nf[a]          # midpoint relation
        off = np.round(b_img - nf[b]).astype(int)
        edges.append((a, b, tuple(int(t) for t in off)))
    g = DefectGraph(cell=cell, node_frac=nf, edges=edges)
    if verbose:
        print(f"  defect graph: {g.n_nodes} Zr6 nodes, {g.n_sites} linker sites "
              "(node is defective if it bears a mono acid)")
    return g


# backward-compatible alias (old call signature build_cage_graph(nodes, cell, site_frac))
def build_cage_graph(sg_or_nodes, cell=None, site_frac=None, verbose: bool = True):
    """Accepts either a SiteGraph, or the legacy (node_frac, cell, site_frac)."""
    if cell is None:                                   # a SiteGraph was passed
        return build_defect_graph(sg_or_nodes, verbose=verbose)
    # legacy path: reconstruct endpoint nodes from geometry
    nf = np.asarray(sg_or_nodes, float); cell = np.asarray(cell, float)
    site_frac = np.asarray(site_frac, float); inv = np.linalg.inv(cell)
    edges = []
    for s in range(len(site_frac)):
        d = site_frac[s] - nf; d -= np.round(d)
        dist = np.linalg.norm(d @ cell, axis=1)
        order = np.argsort(dist)
        a, b = int(order[0]), int(order[1])
        b_img = 2.0 * site_frac[s] - nf[a]
        off = np.round(b_img - nf[b]).astype(int)
        edges.append((a, b, tuple(int(t) for t in off)))
    g = DefectGraph(cell=cell, node_frac=nf, edges=edges)
    if verbose:
        print(f"  defect graph: {g.n_nodes} Zr6 nodes, {g.n_sites} linker sites")
    return g


def validate(sg, verbose: bool = True) -> bool:
    """Node-defect percolation sanity checks: all-mono -> 3-D span, none -> 0."""
    g = build_defect_graph(sg, verbose=verbose)
    ok = True
    r_all = g.percolation(np.zeros(g.n_sites, np.int8))["percolation_rank"]   # all mono
    r_none = g.percolation(np.ones(g.n_sites, np.int8))["percolation_rank"]   # all di
    if verbose:
        print(f"  [{'OK' if r_all == 3 else 'FAIL'}] all linkers mono -> rank {r_all} (expect 3)")
        print(f"  [{'OK' if r_none == 0 else 'FAIL'}] all linkers intact -> rank {r_none} (expect 0)")
    ok = (r_all == 3 and r_none == 0)
    if verbose:
        print("PASS" if ok else "FAIL")
    return ok