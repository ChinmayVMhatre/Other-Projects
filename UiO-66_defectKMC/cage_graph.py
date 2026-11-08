"""
cage_graph.py -- the guest-transport cage graph of (defective) UiO-66
=====================================================================

This is the "cage graph approach": the object a guest actually percolates
through. Cages (octahedral + tetrahedral cavities) are nodes; windows
(triangular apertures, each bounded by 3 linkers) are edges. A window is OPEN
when enough of its 3 linkers are missing; a guest percolates when open windows
connect cages across the periodic cell.

This is the correct connectivity for the JPCC 2024 percolating-defective-pathway
criterion -- cage to cage through windows -- and it replaces the earlier
"windows sharing a linker" shortcut, which was wrong (a fully open framework did
not come out 3D-percolating). Here the all-open framework is rank 3 and the
empty framework is rank 0, as it must be.

Construction (robust by design, no fragile boundary bookkeeping):
  - Cage centres  = Voronoi vertices of the Zr6 node lattice, clustered with a
    tolerance (the Voronoi computation puts ~3 near-coincident vertices at each
    6-coordinate octahedral hole; clustering merges them).
  - Windows       = triples of mutually-nearest-neighbour nodes (with images).
  - window -> cages = the two cage images nearest the window centroid (one
    tetrahedral, one octahedral).
  - window -> linkers = the 3 node-pair midpoints, matched to linker-site
    fractional positions so occupancy drives openness.

Validated on a pristine UiO-66 conventional cell (a = 20.70 A, 4 Zr6 nodes):
  24 linker sites, 32 windows, 12 cages (4 octahedral + 8 tetrahedral),
  every window borders one tet + one oct, tet cages have 4 windows and oct
  cages 8, all-open -> percolation rank 3, no-open -> rank 0. Every window edge
  matches a linker site.

Dependencies: numpy, scipy (Voronoi). Reuses periodic_components from
uio66_sites for the image-aware spanning test. Does NOT require pymatgen or
MOFUN; node positions can come from any source (uio66_sites.find_nodes, a parsed
CIF, or a MofunSites graph).
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import Voronoi

from uio66_sites import periodic_components

NN_DEFAULT = 14.637      # Zr6 node-node nearest-neighbour distance (A), UiO-66
NN_TOL = 1.0             # tolerance for "is this a node-node contact" (A)
CAGE_MERGE_TOL = 1.5     # A; merge Voronoi vertices closer than this (degeneracy)


# ---------------------------------------------------------------------------
@dataclass
class CageGraph:
    cell: np.ndarray
    cage_frac: np.ndarray             # (C,3) cage centres, fractional, in [0,1)
    cage_coord: np.ndarray            # (C,) nodes touching each cage: 6=oct, 4=tet
    windows: list                     # per window: list of (node_idx, image(3,))
    win_cages: list                   # per window: [(cage_i, img(3,)), (cage_j, img(3,))]
    win_sites: np.ndarray             # (W,3) linker-site indices bounding each window
    node_frac: np.ndarray

    @property
    def n_cages(self) -> int:
        return len(self.cage_frac)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    def open_mask(self, occ: np.ndarray, threshold: int = 1) -> np.ndarray:
        """A window is open when >= threshold of its 3 linkers are missing."""
        missing = (occ[self.win_sites] == 0).sum(axis=1)
        return missing >= threshold

    def percolation(self, occ: np.ndarray, threshold: int = 1) -> dict:
        """Guest percolation through open windows, cage to cage.

        threshold is the number of missing linkers that makes a window passable
        (1 reproduces 'a defective window has at least one missing linker';
        raise it for a stricter aperture criterion).
        """
        open_w = np.flatnonzero(self.open_mask(occ, threshold))
        nb = [[] for _ in range(self.n_cages)]
        off = [[] for _ in range(self.n_cages)]
        for w in open_w:
            (c1, s1), (c2, s2) = self.win_cages[w]
            nb[c1].append(c2); off[c1].append(s2 - s1)
            nb[c2].append(c1); off[c2].append(s1 - s2)
        nb = [np.array(x, int) for x in nb]
        off = [np.array(x, int).reshape(-1, 3) for x in off]
        comps = periodic_components(range(self.n_cages), nb, off)
        biggest = max(comps, key=lambda c: c["size"]) if comps else None
        return {
            "n_open_windows": int(len(open_w)),
            "open_fraction": float(len(open_w) / max(self.n_windows, 1)),
            "n_cage_clusters": len(comps),
            "largest_cage_cluster": biggest["size"] if biggest else 0,
            "percolation_rank": max((c["percolation_rank"] for c in comps),
                                    default=0),
        }

    def percolation_pathways(self, occ: np.ndarray, threshold: int = 1,
                             open_windows=None, max_img: int = 2) -> dict:
        """Geometric length of the shortest spanning pathway, per lattice axis.

        For each axis a where the open-window cage network percolates, finds the
        shortest repeating channel -- the shortest path through open windows that
        advances by exactly one lattice vector e_a and returns to the same cage
        (a path that, tiled, forms an infinite channel). Edge weights are the
        Euclidean distances between the centres of adjacent cages, so the path
        length is a real physical length in Angstrom.

        Reported per spanning axis:
          pathway_length  -- channel length over one cell (A)
          cell_dimension  -- |lattice vector a| (A)
          tortuosity      -- pathway_length / cell_dimension (1 = straight,
                             >1 = winding)
          n_hops          -- number of cage-to-cage steps
          path_cart       -- (k,3) cage-centre coordinates along the channel

        Pass `open_windows` (bool array) to use a custom open/closed criterion
        (e.g. the aperture test); otherwise windows with >= `threshold` missing
        linkers are open.
        """
        import heapq
        from collections import defaultdict

        if open_windows is None:
            open_windows = self.open_mask(occ, threshold)
        open_windows = np.asarray(open_windows, bool)
        cf = self.cage_frac

        adj = defaultdict(list)                     # c1 -> [(c2, offset, weight)]
        for w in np.flatnonzero(open_windows):
            (c1, s1), (c2, s2) = self.win_cages[w]
            off = np.asarray(s2) - np.asarray(s1)
            wt = float(np.linalg.norm((cf[c2] + off - cf[c1]) @ self.cell))
            adj[c1].append((c2, tuple(int(x) for x in off), wt))
            adj[c2].append((c1, tuple(int(-x) for x in off), wt))

        out = {}
        for axis in range(3):
            tgt = [0, 0, 0]; tgt[axis] = 1; tgt = tuple(tgt)
            cell_dim = float(np.linalg.norm(self.cell[axis]))
            best, best_path = np.inf, None
            for c0 in range(self.n_cages):
                if c0 not in adj:
                    continue
                dist = {(c0, (0, 0, 0)): 0.0}
                prev = {}
                pq = [(0.0, c0, (0, 0, 0))]
                found = None
                while pq:
                    d, c, img = heapq.heappop(pq)
                    if d > dist.get((c, img), np.inf):
                        continue
                    if c == c0 and img == tgt:
                        found = d
                        break
                    for (c2, off, wt) in adj[c]:
                        nimg = (img[0] + off[0], img[1] + off[1], img[2] + off[2])
                        if max(abs(nimg[0]), abs(nimg[1]), abs(nimg[2])) > max_img:
                            continue
                        nd = d + wt
                        if nd < dist.get((c2, nimg), np.inf):
                            dist[(c2, nimg)] = nd
                            prev[(c2, nimg)] = (c, img)
                            heapq.heappush(pq, (nd, c2, nimg))
                if found is not None and found < best:
                    best = found
                    node = (c0, tgt); chain = [node]
                    while node in prev:
                        node = prev[node]; chain.append(node)
                    chain = chain[::-1]
                    best_path = np.array([(cf[c] + np.array(im)) @ self.cell
                                          for c, im in chain])
            if np.isfinite(best):
                out[axis] = {"spans": True, "pathway_length": best,
                             "cell_dimension": cell_dim,
                             "tortuosity": best / cell_dim,
                             "n_hops": len(best_path) - 1,
                             "path_cart": best_path}
            else:
                out[axis] = {"spans": False, "cell_dimension": cell_dim}
        return out

    def summary(self) -> dict:
        from collections import Counter
        wins_per_cage = Counter()
        for (c1, _), (c2, _) in self.win_cages:
            wins_per_cage[c1] += 1
            wins_per_cage[c2] += 1
        return {
            "n_cages": self.n_cages,
            "n_octahedral": int(np.sum(self.cage_coord == 6)),
            "n_tetrahedral": int(np.sum(self.cage_coord == 4)),
            "n_windows": self.n_windows,
            "windows_per_cage": dict(Counter(wins_per_cage.values())),
        }


# ---------------------------------------------------------------------------
def _wrap(x):
    return x - np.floor(x + 1e-9)


def _min_image(a, b, cell):
    d = a - b
    d -= np.round(d)
    return np.linalg.norm(d @ cell)


def build_cage_graph(node_frac: np.ndarray, cell: np.ndarray,
                     site_frac: np.ndarray, nn: float = NN_DEFAULT,
                     nn_tol: float = NN_TOL, merge_tol: float = CAGE_MERGE_TOL,
                     verbose: bool = True) -> CageGraph:
    """Build the cage graph from Zr6 node positions and linker-site positions.

    node_frac : (N,3) Zr6 node centroids, fractional. From uio66_sites.find_nodes
                or a SiteGraph's node_frac.
    cell      : (3,3) lattice, rows = vectors.
    site_frac : (M,3) linker-site midpoints, fractional. A SiteGraph's site_frac;
                each window's 3 edges are matched to these so occupancy drives
                window openness. (occ[s]==0 means site s is a missing linker.)
    """
    node_frac = np.mod(np.asarray(node_frac, float), 1.0)
    cell = np.asarray(cell, float)
    inv = np.linalg.inv(cell)
    N = len(node_frac)
    s3 = np.array(list(itertools.product((-1, 0, 1), repeat=3)))

    # --- cage centres: clustered Voronoi vertices of the node lattice ---
    S = np.array(list(itertools.product((-2, -1, 0, 1, 2), repeat=3)))
    big = (node_frac[None] + S[:, None]).reshape(-1, 3) @ cell
    vf = np.mod(Voronoi(big).vertices @ inv, 1.0)
    vf = vf[np.all(np.isfinite(vf), axis=1)]
    cages = []
    for v in vf:
        if not any(_min_image(v, c, cell) < merge_tol for c in cages):
            cages.append(v)
    cage_frac = np.array(cages)

    node_imgs = (node_frac[None] + s3[:, None]).reshape(-1, 3)
    coord = []
    for c in cage_frac:
        d = np.linalg.norm((node_imgs - c) @ cell, axis=1)
        coord.append(int(np.sum(np.abs(d - d.min()) < 0.8)))
    cage_coord = np.array(coord)

    # --- windows: mutually-nearest-neighbour node triples (with images) ---
    nbr = defaultdict(list)
    for a in range(N):
        for b in range(N):
            for s in s3:
                if a == b and np.all(s == 0):
                    continue
                if abs(np.linalg.norm((node_frac[b] + s - node_frac[a]) @ cell)
                       - nn) < nn_tol:
                    nbr[a].append((b, s))
    seen, windows = set(), []
    for a in range(N):
        for (b, ob) in nbr[a]:
            for (c, oc) in nbr[a]:
                if abs(np.linalg.norm((node_frac[c] + oc - (node_frac[b] + ob))
                                      @ cell) - nn) < nn_tol:
                    verts = sorted([(a, (0, 0, 0)), (b, tuple(ob)), (c, tuple(oc))])
                    base = np.array(verts[0][1])
                    key = tuple((nn_, tuple(np.array(im) - base)) for nn_, im in verts)
                    if key not in seen:
                        seen.add(key)
                        windows.append([(nn_, np.array(im)) for nn_, im in verts])

    # --- window -> two nearest cages (image-resolved) ---
    cimg = [(ci, s, (cage_frac[ci] + s) @ cell)
            for ci in range(len(cage_frac)) for s in s3]
    win_cages = []
    for verts in windows:
        cen = np.mean([node_frac[n] + im for n, im in verts], 0) @ cell
        order = sorted((np.linalg.norm(cen - cc), ci, tuple(s)) for ci, s, cc in cimg)
        chosen = []
        for _, ci, s in order:
            if not any(ci == cj for cj, _ in chosen):
                chosen.append((ci, np.array(s)))
            if len(chosen) == 2:
                break
        win_cages.append(chosen)

    # --- window -> 3 linker sites (match node-pair midpoints to site_frac) ---
    site_frac = np.mod(np.asarray(site_frac, float), 1.0)
    skey = {tuple(np.round(_wrap(s), 3)): i for i, s in enumerate(site_frac)}
    win_sites, missing_edges = [], 0
    for verts in windows:
        trio = []
        for (n1, i1), (n2, i2) in itertools.combinations(verts, 2):
            mid = _wrap(node_frac[n1] + i1 + 0.5 * ((node_frac[n2] + i2)
                                                    - (node_frac[n1] + i1)))
            idx = skey.get(tuple(np.round(mid, 3)))
            if idx is None:
                missing_edges += 1
                idx = -1
            trio.append(idx)
        win_sites.append(trio)
    win_sites = np.array(win_sites, int)
    if missing_edges and verbose:
        print(f"  ! {missing_edges} window edges did not match a linker site; "
              "site_frac may be from a different cell than node_frac.")

    g = CageGraph(cell=cell, cage_frac=cage_frac, cage_coord=cage_coord,
                  windows=windows, win_cages=win_cages, win_sites=win_sites,
                  node_frac=node_frac)
    if verbose:
        s = g.summary()
        print(f"  cages {s['n_cages']} ({s['n_octahedral']} oct + "
              f"{s['n_tetrahedral']} tet), windows {s['n_windows']}, "
              f"windows/cage {s['windows_per_cage']}")
    return g


# ---------------------------------------------------------------------------
def validate(node_frac, cell, site_frac, per_cell_nodes: int = 4,
             verbose: bool = True) -> bool:
    """Check the construction against known UiO-66 fcu topology.

    Scales expectations by how many conventional cells this structure spans
    (inferred from node count / per_cell_nodes = 4 nodes per conventional cell).
    """
    g = build_cage_graph(node_frac, cell, site_frac, verbose=verbose)
    cells = max(round(len(g.node_frac) / per_cell_nodes), 1)
    s = g.summary()
    ok = True

    def check(name, got, exp):
        nonlocal ok
        flag = "OK" if got == exp else "FAIL"
        if got != exp:
            ok = False
        if verbose:
            print(f"  [{flag}] {name}: {got} (expect {exp})")

    check("cages", s["n_cages"], 12 * cells)
    check("octahedral", s["n_octahedral"], 4 * cells)
    check("tetrahedral", s["n_tetrahedral"], 8 * cells)
    check("windows", s["n_windows"], 32 * cells)
    check("windows/cage profile", s["windows_per_cage"], {4: 8 * cells, 8: 4 * cells})
    check("no unmatched window edges", int((g.win_sites < 0).any()), 0)

    occ_all = np.zeros(int(g.win_sites.max()) + 1, np.int8)   # all defective
    occ_none = np.ones_like(occ_all)                           # pristine
    r_all = g.percolation(occ_all, threshold=1)["percolation_rank"]
    r_none = g.percolation(occ_none, threshold=1)["percolation_rank"]
    check("all-open percolation rank", r_all, 3)
    check("pristine percolation rank", r_none, 0)

    if verbose:
        print("PASS" if ok else "FAIL -- do not trust the cage metric yet")
    return ok


if __name__ == "__main__":
    import sys
    from uio66_sites import find_nodes, build_site_graph, Frame

    if len(sys.argv) == 2:
        # parse a simple P1 CIF with explicit fractional coords
        path = sys.argv[1]
        cell_len, species, frac = {}, [], []
        cols = []
        for ln in open(path):
            t = ln.split()
            for k in ("_cell_length_a", "_cell_length_b", "_cell_length_c"):
                if ln.strip().startswith(k):
                    cell_len[k] = float(t[1])
            if ln.strip().startswith("_atom_site_"):
                cols.append(ln.strip())
            elif len(t) >= 7 and len(t[0]) <= 2 and t[0][0].isalpha() and t[0][0].isupper():
                try:
                    frac.append([float(t[3]), float(t[4]), float(t[5])])
                    species.append(t[0])
                except ValueError:
                    pass
        cell = np.diag([cell_len["_cell_length_a"], cell_len["_cell_length_b"],
                        cell_len["_cell_length_c"]])
        frame = Frame(species, np.array(frac), cell)
        nodes, _ = find_nodes(frame)
        sg = build_site_graph(nodes, cell, hop_shell="shares_sbu", verbose=False)
        print(f"[{path}] {len(nodes)} nodes, {sg.n_sites} linker sites")
        validate(nodes, cell, sg.site_frac)
    else:
        print(__doc__)
        print("usage: python cage_graph.py PRISTINE.cif")
