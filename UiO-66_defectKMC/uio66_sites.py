"""
uio66_sites.py -- start the kMC from a DEFECTIVE, formate-capped UiO-66 CIF
===========================================================================

Companion to `uio66_defect_kmc.py`. This module replaces two things in that
script: the random initial occupancy, and the 6-connected simple-cubic proxy
net. Both come from a real structure instead.

Definition used here (matching the JPCC 2024 paper)
---------------------------------------------------
A linker site is an EDGE of the Zr6 node net. Each edge is either

    occupied : a BDC linker, C6H4(COO)2   -> occ = 1
    defective: NO linker; instead the two Zr6 nodes at the ends of that edge
               each carry a formate cap, HCOO   -> occ = 0

So one missing linker == two formate groups. That 2:1 ratio is the consistency
check the reader uses to confirm it parsed the structure correctly.

Pipeline
--------
    pristine.cif  --> node net --> edge list  == the SITE GRAPH (topology)
                  --> BDC atom groups         == the GEOMETRY LIBRARY
    defective.cif --> BDC centroids matched into the site graph
                                              == the INITIAL OCCUPANCY

The site graph and the geometry library come from the pristine cell; the
defective cell is read only for the occupancy PATTERN. Both CIFs must
therefore describe the same cell / supercell.

Everything except read_frame/write_frame is pure numpy, so the graph logic can
be tested without pymatgen installed.

Verified on an ideal fcu net (a = 20.7 A, 2x2x2 of the conventional cell):
  32 nodes, coordination 12, node-node 14.64 A
  192 linker sites (= 6 per node, = 24 per conventional cell)
  nearest site shell: 8 sites at 7.32 A
  shares-an-SBU shell: 22 sites (8 at 7.32, 4 at 10.35, 8 at 12.68, 2 at 14.64)
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field

import numpy as np

BOND_CUTOFF = 1.9      # A, covalent bonds among C/H/O (excludes Zr-O at ~2.2)
ZR_CUTOFF = 4.2        # A, Zr-Zr within one Zr6 node (~3.5); next node is ~10 A
C_H_BOND = 1.09        # A, C-H bond length used when generating formate caps


# ---------------------------------------------------------------------------
# 0. Minimal structure container + optional pymatgen I/O
# ---------------------------------------------------------------------------
@dataclass
class Frame:
    """A periodic structure: element symbols, fractional coords, cell (rows)."""
    species: list[str]
    frac: np.ndarray
    cell: np.ndarray

    def __post_init__(self):
        self.frac = np.asarray(self.frac, dtype=float).reshape(-1, 3)
        self.cell = np.asarray(self.cell, dtype=float).reshape(3, 3)
        if len(self.species) != len(self.frac):
            raise ValueError("species / frac length mismatch")

    def __len__(self):
        return len(self.species)

    @property
    def cart(self) -> np.ndarray:
        return self.frac @ self.cell

    def indices_of(self, symbol: str) -> np.ndarray:
        return np.array([i for i, s in enumerate(self.species) if s == symbol],
                        dtype=int)


def read_frame(path: str) -> Frame:
    from pymatgen.core import Structure
    s = Structure.from_file(path)
    return Frame([str(sp.symbol) for sp in s.species],
                 np.array(s.frac_coords), np.array(s.lattice.matrix))


def write_frame(frame: Frame, path: str) -> None:
    from pymatgen.core import Structure
    Structure(frame.cell, frame.species, frame.frac).to(filename=path)


# ---------------------------------------------------------------------------
# 1. Periodic neighbour search (brute force, vectorised; cells here are small)
# ---------------------------------------------------------------------------
def pbc_pairs(fracA: np.ndarray, fracB: np.ndarray, cell: np.ndarray,
              rmax: float, exclude_self: bool = True):
    """All pairs (a in A, b in B) with |r_b + offset - r_a| < rmax.

    Returns (ia, ib, offsets (n,3) int, dists (n,)). Offsets are the integer
    lattice translations applied to B, so the neighbour of a sits at
    fracB[ib] + offsets.
    """
    fracA = np.atleast_2d(fracA)
    fracB = np.atleast_2d(fracB)
    inv_len = np.linalg.norm(np.linalg.inv(cell), axis=0)
    nmax = np.maximum(np.ceil(rmax * inv_len).astype(int), 1)
    shifts = np.array(list(itertools.product(*[range(-n, n + 1) for n in nmax])))

    ia_out, ib_out, off_out, d_out = [], [], [], []
    same = fracA is fracB or (fracA.shape == fracB.shape and
                              np.allclose(fracA, fracB))
    for ia in range(len(fracA)):
        d = fracB[None, :, :] + shifts[:, None, :] - fracA[ia][None, None, :]
        dist = np.linalg.norm(d @ cell, axis=-1)          # (n_shift, n_B)
        si, ib = np.nonzero((dist < rmax) & (dist > 1e-6))
        if exclude_self and same:
            keep = ~((ib == ia) & np.all(shifts[si] == 0, axis=1))
            si, ib = si[keep], ib[keep]
        ia_out.append(np.full(len(ib), ia, dtype=int))
        ib_out.append(ib)
        off_out.append(shifts[si])
        d_out.append(dist[si, ib])
    if not ia_out:
        return (np.zeros(0, int), np.zeros(0, int),
                np.zeros((0, 3), int), np.zeros(0))
    return (np.concatenate(ia_out), np.concatenate(ib_out),
            np.concatenate(off_out).reshape(-1, 3), np.concatenate(d_out))


def snap_frac(f: np.ndarray, decimals: int = 6) -> np.ndarray:
    """Round away float noise BEFORE wrapping, so 0.9999997 becomes 0.0 rather
    than 1.0. Both site-graph constructors must normalise identically or their
    site labels will not line up."""
    return np.round(np.asarray(f, float), decimals)


def pairs_for_shell(frac: np.ndarray, cell: np.ndarray, k: int,
                    label: str = "", start: float = 0.35):
    """Neighbour pairs with a radius grown until every item has > k neighbours."""
    scale = float(np.min(np.linalg.norm(cell, axis=1)))
    r = start * scale
    for _ in range(6):
        ia, ib, off, d = pbc_pairs(frac, frac, cell, r)
        counts = np.bincount(ia, minlength=len(frac))
        if counts.min() > k:
            return ia, ib, off, d
        r *= 1.5
    raise ValueError(f"{label}: could not find more than {k} neighbours per site "
                     f"within {r:.1f} A. Check the cell and the coordination.")


def _shell_cutoff(dists_per_item: list[np.ndarray], k: int,
                  label: str, gap: float = 1.15) -> float:
    """Cutoff that isolates the k nearest neighbours, checking for a real gap."""
    d_k, d_k1 = [], []
    for d in dists_per_item:
        d = np.sort(d)
        if len(d) <= k:
            raise ValueError(f"{label}: only {len(d)} neighbours found, need > {k}. "
                             "Increase the search radius or check the cell.")
        d_k.append(d[k - 1])
        d_k1.append(d[k])
    d_k, d_k1 = float(np.max(d_k)), float(np.min(d_k1))
    if d_k1 < gap * d_k:
        print(f"  ! {label}: weak gap between shell {k} ({d_k:.2f} A) and "
              f"{k + 1} ({d_k1:.2f} A). Inspect before trusting the topology.")
    return 0.5 * (d_k + d_k1)


# ---------------------------------------------------------------------------
# 2. Zr6 nodes
# ---------------------------------------------------------------------------
def _components(n: int, adj: dict[int, list[int]]) -> list[list[int]]:
    seen, comps = set(), []
    for s in range(n):
        if s in seen:
            continue
        q, comp = deque([s]), []
        seen.add(s)
        while q:
            c = q.popleft()
            comp.append(c)
            for nb in adj.get(c, ()):
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        comps.append(comp)
    return comps


def find_nodes(frame: Frame, zr_symbol: str = "Zr",
               zr_cutoff: float = ZR_CUTOFF, expect_size: int = 6):
    """Group Zr atoms into Zr6 nodes; return (node_frac (n,3), clusters)."""
    zr = frame.indices_of(zr_symbol)
    if len(zr) == 0:
        raise ValueError(f"no {zr_symbol} atoms in structure")
    ia, ib, off, _ = pbc_pairs(frame.frac[zr], frame.frac[zr], frame.cell, zr_cutoff)

    adj: dict[int, list[int]] = {i: [] for i in range(len(zr))}
    shift: dict[tuple[int, int], np.ndarray] = {}
    for a, b, o in zip(ia, ib, off):
        adj[int(a)].append(int(b))
        shift[(int(a), int(b))] = o

    node_frac, clusters = [], []
    for comp in _components(len(zr), adj):
        if len(comp) != expect_size:
            print(f"  ! found a metal cluster of {len(comp)} {zr_symbol} atoms "
                  f"(expected {expect_size}); check zr_cutoff={zr_cutoff}")
        # unwrap the cluster by BFS so the centroid is not corrupted by PBC
        pos = {comp[0]: frame.frac[zr[comp[0]]]}
        q = deque([comp[0]])
        members = set(comp)
        while q:
            c = q.popleft()
            for nb in adj[c]:
                if nb in members and nb not in pos:
                    pos[nb] = pos[c] + (frame.frac[zr[nb]] + shift[(c, nb)]
                                        - frame.frac[zr[c]])
                    q.append(nb)
        node_frac.append(np.mean(np.array([pos[c] for c in comp]), axis=0))
        clusters.append([int(zr[c]) for c in comp])
    node_frac = np.mod(snap_frac(node_frac), 1.0)
    order = np.lexsort(node_frac.T[::-1])
    return node_frac[order], [clusters[i] for i in order]


# ---------------------------------------------------------------------------
# 3. The site graph: linker sites = edges of the node net
# ---------------------------------------------------------------------------
@dataclass
class SiteGraph:
    cell: np.ndarray
    node_frac: np.ndarray
    site_frac: np.ndarray                      # (M,3) edge midpoints, in [0,1)
    site_nodes: np.ndarray                     # (M,2) endpoint node indices
    site_node_shift: list[np.ndarray]          # per site: [(shift to node a),
                                               #            (shift to node b)]
    neighbours: list[np.ndarray] = field(default_factory=list)
    nbr_offsets: list[np.ndarray] = field(default_factory=list)
    node_sites: dict[int, list[tuple[int, np.ndarray]]] = field(default_factory=dict)

    @property
    def n_sites(self) -> int:
        return len(self.site_frac)

    def coordination(self) -> set[int]:
        return {len(n) for n in self.neighbours}


def build_site_graph(node_frac: np.ndarray, cell: np.ndarray,
                     node_coordination: int = 12,
                     hop_shell: str = "nearest",
                     verbose: bool = True) -> SiteGraph:
    """Build the linker-site graph from the Zr6 node positions.

    hop_shell:
      "nearest"     -- 8 nearest linker sites (7.32 A in ideal UiO-66). The
                       physically natural elementary relocation.
      "shares_sbu"  -- all 22 sites sharing an SBU with this one. Built from
                       node incidence, NOT a distance cutoff: a 15 A geometric
                       cutoff picks up 42 sites, many of which share no node.
    """
    ia, ib, off, d = pairs_for_shell(node_frac, cell, node_coordination,
                                     "node net")
    per_node = [d[ia == i] for i in range(len(node_frac))]
    cutoff = _shell_cutoff(per_node, node_coordination, "node net")
    keep = d < cutoff
    ia, ib, off = ia[keep], ib[keep], off[keep]
    if verbose:
        print(f"  nodes: {len(node_frac)}, coordination "
              f"{sorted({int((ia == i).sum()) for i in range(len(node_frac))})}, "
              f"node-node {d[keep].min():.2f}-{d[keep].max():.2f} A")

    # each edge -> one site, deduplicated by its wrapped midpoint
    sites: dict[tuple, tuple] = {}
    for a, b, o in zip(ia, ib, off):
        mid_raw = snap_frac(node_frac[a] + 0.5 * (node_frac[b] + o - node_frac[a]))
        w = -np.floor(mid_raw).astype(int)          # wrap into the home cell
        key = tuple(np.round(mid_raw + w, 6))
        if key not in sites:
            sites[key] = (int(a), int(b), np.asarray(o, int), w)

    site_frac = np.array([list(k) for k in sites])
    recs = list(sites.values())
    site_nodes = np.array([[r[0], r[1]] for r in recs], dtype=int)
    # shift that brings this site onto the home image of each endpoint node
    site_node_shift = [[-r[3], -(r[2] + r[3])] for r in recs]

    node_sites: dict[int, list[tuple[int, np.ndarray]]] = {
        n: [] for n in range(len(node_frac))}
    for s, (a, b, o, w) in enumerate(recs):
        node_sites[a].append((s, -w))
        node_sites[b].append((s, -(o + w)))

    g = SiteGraph(cell=np.asarray(cell, float), node_frac=node_frac,
                  site_frac=site_frac, site_nodes=site_nodes,
                  site_node_shift=site_node_shift, node_sites=node_sites)

    expected = len(node_frac) * node_coordination // 2
    if verbose:
        print(f"  linker sites: {g.n_sites} (expected {expected}); "
              f"sites per node "
              f"{sorted({len(v) for v in node_sites.values()})}")
    if g.n_sites != expected:
        print(f"  ! site count {g.n_sites} != {expected}. Topology is wrong; "
              "do not run production kMC on this graph.")

    _attach_hops(g, hop_shell, verbose)
    return g


def _attach_hops(g: SiteGraph, hop_shell: str, verbose: bool = True) -> None:
    if hop_shell == "nearest":
        ia, ib, off, d = pairs_for_shell(g.site_frac, g.cell, 8,
                                         "linker-site first shell")
        per = [d[ia == i] for i in range(g.n_sites)]
        cutoff = _shell_cutoff(per, 8, "linker-site first shell")
        keep = d < cutoff
        ia, ib, off = ia[keep], ib[keep], off[keep]
        g.neighbours = [ib[ia == s] for s in range(g.n_sites)]
        g.nbr_offsets = [off[ia == s] for s in range(g.n_sites)]
    elif hop_shell == "shares_sbu":
        adj: list[set] = [set() for _ in range(g.n_sites)]
        for lst in g.node_sites.values():
            for s, u in lst:
                for t, v in lst:
                    if t == s and np.array_equal(u, v):
                        continue
                    adj[s].add((t, tuple(np.asarray(v) - np.asarray(u))))
        g.neighbours = [np.array([t for t, _ in sorted(a)], dtype=int) for a in adj]
        g.nbr_offsets = [np.array([o for _, o in sorted(a)], dtype=int).reshape(-1, 3)
                         for a in adj]
    else:
        raise ValueError("hop_shell must be 'nearest' or 'shares_sbu'")

    if verbose:
        dists = []
        for s in range(g.n_sites):
            v = (g.site_frac[g.neighbours[s]] + g.nbr_offsets[s]
                 - g.site_frac[s]) @ g.cell
            dists.append(np.linalg.norm(v, axis=1))
        allmin = min(x.min() for x in dists)
        allmax = max(x.max() for x in dists)
        print(f"  hop shell '{hop_shell}': coordination {sorted(g.coordination())}, "
              f"hop distance {allmin:.2f}-{allmax:.2f} A")


def build_site_graph_from_matches(match_positions, node_frac: np.ndarray,
                                  cell: np.ndarray, hop_shell: str = "nearest",
                                  verbose: bool = True) -> SiteGraph:
    """Build the linker-site graph from MOFUN pattern matches.

    Use this instead of build_site_graph() when the sites come from
    find_pattern_in_structure(): MOFUN has already identified every linker, so
    there is no need to enumerate node-net edges or assume ideal fcu geometry.

    Args:
        match_positions: (M, n_pattern_atoms, 3) UNWRAPPED cartesian positions,
            exactly what find_pattern_in_structure(..., return_positions_and_quats
            =True) returns as its second value.
        node_frac: Zr6 node centroids in fractional coords (find_nodes()).
        cell: 3x3 cell, rows = lattice vectors.

    Each match is split along its own long axis; the extreme atom at each end is
    assigned to the nearest Zr6 node image. That gives the two endpoint nodes of
    the site and, with them, the image bookkeeping the periodic percolation test
    needs.
    """
    mp = np.asarray(match_positions, dtype=float)
    if mp.ndim != 3 or mp.shape[2] != 3:
        raise ValueError("match_positions must have shape (n_matches, n_atoms, 3)")
    if mp.shape[1] < 2:
        raise ValueError("the search pattern needs at least 2 atoms to define "
                         "the two ends of a linker")
    cell = np.asarray(cell, float)
    inv = np.linalg.inv(cell)
    M = len(mp)

    center_frac_raw = snap_frac(mp.mean(axis=1) @ inv)
    wrap = -np.floor(center_frac_raw).astype(int)
    site_frac = center_frac_raw + wrap

    site_nodes = np.zeros((M, 2), dtype=int)
    site_node_shift: list[list[np.ndarray]] = []
    node_sites: dict[int, list[tuple[int, np.ndarray]]] = {
        n: [] for n in range(len(node_frac))}
    end_dists = []

    for s in range(M):
        d = mp[s] - mp[s].mean(axis=0)
        axis = np.linalg.svd(d, full_matrices=False)[2][0]
        proj = d @ axis
        ends_frac = [(mp[s][int(np.argmax(proj))] @ inv) + wrap[s],
                     (mp[s][int(np.argmin(proj))] @ inv) + wrap[s]]
        shifts = []
        for e, ef in enumerate(ends_frac):
            g = np.round(ef - node_frac).astype(int)            # (n_nodes, 3)
            delta = ef - (node_frac + g)
            dist = np.linalg.norm(delta @ cell, axis=1)
            n = int(np.argmin(dist))
            end_dists.append(dist[n])
            site_nodes[s, e] = n
            shifts.append(-g[n])
        site_node_shift.append(shifts)
        node_sites[site_nodes[s, 0]].append((s, shifts[0]))
        node_sites[site_nodes[s, 1]].append((s, shifts[1]))
        if site_nodes[s, 0] == site_nodes[s, 1] and \
                np.array_equal(shifts[0], shifts[1]):
            print(f"  ! site {s}: both ends map to the same node image. The "
                  "search pattern is probably not a full bridging linker.")

    g = SiteGraph(cell=cell, node_frac=np.asarray(node_frac, float),
                  site_frac=site_frac, site_nodes=site_nodes,
                  site_node_shift=site_node_shift, node_sites=node_sites)

    if verbose:
        per_node = sorted({len(v) for v in node_sites.values()})
        print(f"  sites from MOFUN matches: {M}; linkers per node {per_node}; "
              f"end-to-node distance {min(end_dists):.2f}-{max(end_dists):.2f} A")
    if len(set(len(v) for v in node_sites.values())) != 1:
        print("  ! nodes carry different numbers of linkers. In a pristine "
              "structure every node should coordinate 12. Check the pattern "
              "and the match tolerance before trusting the topology.")

    _attach_hops(g, hop_shell, verbose)
    return g


# ---------------------------------------------------------------------------
# 4. Molecular fragments: which atoms are BDC, which are formate
# ---------------------------------------------------------------------------
def fragments(frame: Frame, bond_cutoff: float = BOND_CUTOFF,
              skip: tuple[str, ...] = ("Zr",)) -> list[list[int]]:
    """Connected components of the covalent bond graph over non-metal atoms."""
    sel = np.array([i for i, s in enumerate(frame.species) if s not in skip],
                   dtype=int)
    ia, ib, _, _ = pbc_pairs(frame.frac[sel], frame.frac[sel], frame.cell,
                             bond_cutoff)
    adj: dict[int, list[int]] = {i: [] for i in range(len(sel))}
    for a, b in zip(ia, ib):
        adj[int(a)].append(int(b))
    return [[int(sel[c]) for c in comp] for comp in _components(len(sel), adj)]


def classify(frame: Frame, frag: list[int]) -> str:
    """'bdc' (8 C), 'formate' (1 C), 'inorganic' (no C, e.g. mu3-O / OH)."""
    n_c = sum(1 for i in frag if frame.species[i] == "C")
    n_o = sum(1 for i in frag if frame.species[i] == "O")
    if n_c == 8 and n_o == 4:
        return "bdc"
    if n_c == 1 and n_o == 2:
        return "formate"
    if n_c == 0:
        return "inorganic"
    return f"unknown(C{n_c}O{n_o}n{len(frag)})"


def fragment_centroid(frame: Frame, frag: list[int]) -> np.ndarray:
    """PBC-safe geometric centroid, in fractional coords (unwrapped to frag[0])."""
    seed = frame.frac[frag[0]]
    d = frame.frac[frag] - seed
    d -= np.round(d)
    span = np.linalg.norm(d @ frame.cell, axis=1).max()
    if span > 0.4 * float(np.min(np.linalg.norm(frame.cell, axis=1))):
        print(f"  ! fragment spans {span:.1f} A; minimum-image unwrap may be wrong")
    return seed + d.mean(axis=0)


def inventory(frame: Frame, bond_cutoff: float = BOND_CUTOFF,
              verbose: bool = True) -> dict[str, list[list[int]]]:
    groups: dict[str, list[list[int]]] = {}
    for frag in fragments(frame, bond_cutoff):
        groups.setdefault(classify(frame, frag), []).append(frag)
    if verbose:
        for k, v in sorted(groups.items()):
            print(f"  {k:<28} {len(v)}")
    return groups


# ---------------------------------------------------------------------------
# 5. Occupancy: read the defect pattern out of a defective CIF
# ---------------------------------------------------------------------------
def read_occupancy(frame: Frame, g: SiteGraph, match_tol: float = 2.0,
                   verbose: bool = True):
    """Match BDC linkers in `frame` onto the sites of `g`.

    Returns (occ int8 (M,), report dict). occ = 1 where a BDC was found,
    0 where the site is empty (expected to be formate-capped).
    """
    groups = inventory(frame, verbose=verbose)
    bdc = groups.get("bdc", [])
    formate = groups.get("formate", [])

    occ = np.zeros(g.n_sites, dtype=np.int8)
    unmatched, doubles = [], []
    for frag in bdc:
        c = fragment_centroid(frame, frag)
        d = g.site_frac - c
        d -= np.round(d)
        dist = np.linalg.norm(d @ g.cell, axis=1)
        s = int(np.argmin(dist))
        if dist[s] > match_tol:
            unmatched.append((frag[0], float(dist[s])))
            continue
        if occ[s] == 1:
            doubles.append(s)
        occ[s] = 1

    # formate caps -> which node do they sit on
    caps_per_node: dict[int, int] = {n: 0 for n in range(len(g.node_frac))}
    for frag in formate:
        c = fragment_centroid(frame, frag)
        d = g.node_frac - c
        d -= np.round(d)
        caps_per_node[int(np.argmin(np.linalg.norm(d @ g.cell, axis=1)))] += 1

    defects = np.flatnonzero(occ == 0)
    expected_caps: dict[int, int] = {n: 0 for n in caps_per_node}
    for s in defects:
        for n in g.site_nodes[s]:
            expected_caps[int(n)] += 1

    report = {
        "n_sites": g.n_sites,
        "n_bdc": len(bdc),
        "n_formate": len(formate),
        "n_defects": int(len(defects)),
        "defect_fraction": float(len(defects) / g.n_sites),
        "formate_per_defect": (len(formate) / len(defects)) if len(defects) else None,
        "unmatched_bdc": unmatched,
        "double_matched_sites": doubles,
        "cap_mismatch": {n: (caps_per_node[n], expected_caps[n])
                         for n in caps_per_node
                         if caps_per_node[n] != expected_caps[n]},
    }

    if verbose:
        print(f"  defects: {report['n_defects']}/{g.n_sites} "
              f"({100 * report['defect_fraction']:.1f}%)")
        print(f"  formate groups: {report['n_formate']} "
              f"(expected {2 * report['n_defects']} = 2 per missing linker)")
        if unmatched:
            print(f"  ! {len(unmatched)} BDC linkers matched no site "
                  f"(nearest {unmatched[0][1]:.2f} A). Cells probably differ.")
        if doubles:
            print(f"  ! {len(doubles)} sites matched more than one BDC.")
        if report["cap_mismatch"]:
            print(f"  ! {len(report['cap_mismatch'])} nodes carry the wrong "
                  "number of caps for their missing linkers.")
        if not (unmatched or doubles or report["cap_mismatch"]) and \
                report["n_formate"] == 2 * report["n_defects"]:
            print("  OK: linkers, sites and formate caps are all consistent.")
    return occ, report


# ---------------------------------------------------------------------------
# 6. Geometry library: any occupancy -> a chemically complete capped structure
# ---------------------------------------------------------------------------
@dataclass
class GeometryLibrary:
    cell: np.ndarray
    base_species: list[str]        # Zr, mu3-O, OH -- present in every snapshot
    base_frac: np.ndarray
    bdc: dict[int, tuple[list[str], np.ndarray]]   # site -> intact linker atoms
    caps: dict[int, tuple[list[str], np.ndarray]]  # site -> its two formates


def _formate_from_bdc(frame: Frame, frag: list[int], add_h: bool):
    """Turn one BDC into the two HCOO caps that replace it.

    Keeps both carboxylate groups where they are (so the Zr-O coordination
    geometry is inherited from the pristine structure) and puts an H where the
    phenyl ring used to attach.
    """
    idx = np.array(frag)
    seed = frame.frac[frag[0]]
    d = frame.frac[idx] - seed
    d -= np.round(d)
    frac = seed + d                                  # unwrapped fragment
    cart = frac @ frame.cell
    sym = [frame.species[i] for i in idx]

    nb = {i: [] for i in range(len(idx))}
    for a in range(len(idx)):
        for b in range(len(idx)):
            if a != b and np.linalg.norm(cart[a] - cart[b]) < BOND_CUTOFF:
                nb[a].append(b)

    species_out, frac_out = [], []
    for a in range(len(idx)):
        if sym[a] != "C":
            continue
        oxy = [b for b in nb[a] if sym[b] == "O"]
        ring = [b for b in nb[a] if sym[b] == "C"]
        if len(oxy) != 2 or len(ring) != 1:
            continue                                  # not a carboxylate C
        species_out.append("C")
        frac_out.append(frac[a])
        for b in oxy:
            species_out.append("O")
            frac_out.append(frac[b])
        if add_h:
            u = cart[ring[0]] - cart[a]
            u /= np.linalg.norm(u)
            h_cart = cart[a] + C_H_BOND * u
            species_out.append("H")
            frac_out.append(h_cart @ np.linalg.inv(frame.cell))

    n_c = species_out.count("C")
    if n_c != 2:
        raise ValueError(f"expected 2 carboxylate carbons in a BDC, found {n_c}")
    return species_out, np.array(frac_out)


def build_library(pristine: Frame, g: SiteGraph, match_tol: float = 2.0,
                  verbose: bool = True) -> GeometryLibrary:
    """Split a pristine CIF into base framework + per-site linker + per-site caps."""
    groups = inventory(pristine, verbose=verbose)
    if groups.get("formate"):
        print("  ! the 'pristine' CIF already contains formate groups. "
              "Use a genuinely pristine structure for the geometry library.")
    add_h = any(s == "H" for s in pristine.species)
    if not add_h and verbose:
        print("  no H atoms in the pristine CIF; caps will be written as CO2 "
              "stubs to stay consistent with it.")

    bdc, caps, used = {}, {}, set()
    for frag in groups.get("bdc", []):
        c = fragment_centroid(pristine, frag)
        d = g.site_frac - c
        d -= np.round(d)
        dist = np.linalg.norm(d @ g.cell, axis=1)
        s = int(np.argmin(dist))
        if dist[s] > match_tol:
            raise ValueError(f"a BDC linker is {dist[s]:.2f} A from the nearest "
                             "site; the node net and the linkers disagree")
        seed = pristine.frac[frag[0]]
        dd = pristine.frac[frag] - seed
        dd -= np.round(dd)
        bdc[s] = ([pristine.species[i] for i in frag], seed + dd)
        caps[s] = _formate_from_bdc(pristine, frag, add_h)
        used.update(frag)

    missing = sorted(set(range(g.n_sites)) - set(bdc))
    if missing:
        raise ValueError(f"{len(missing)} sites have no linker in the pristine "
                         "CIF; it is not pristine, or the topology is wrong")

    base = sorted(set(range(len(pristine))) - used)
    lib = GeometryLibrary(cell=g.cell,
                          base_species=[pristine.species[i] for i in base],
                          base_frac=pristine.frac[base],
                          bdc=bdc, caps=caps)
    if verbose:
        print(f"  library: {len(base)} framework atoms + "
              f"{len(bdc)} linker templates + {len(caps)} cap templates")
    return lib


def compose(occ: np.ndarray, lib: GeometryLibrary) -> Frame:
    """Assemble the capped structure for a given occupancy array."""
    species = list(lib.base_species)
    frac = [lib.base_frac]
    for s in range(len(occ)):
        sym, f = lib.bdc[s] if occ[s] == 1 else lib.caps[s]
        species.extend(sym)
        frac.append(f)
    return Frame(species, np.mod(np.vstack(frac), 1.0), lib.cell)


# ---------------------------------------------------------------------------
# 7. Percolation: rigorous periodic test (replaces the face-touching proxy)
# ---------------------------------------------------------------------------
def periodic_components(active, neighbours, nbr_offsets):
    """Image-aware connected components on any periodic graph.

    active:      iterable of active node indices
    neighbours:  neighbours[i] -> array of neighbour indices
    nbr_offsets: nbr_offsets[i] -> (k,3) int lattice offsets, aligned with
                 neighbours[i]

    A component percolates when two BFS paths reach the same node with
    different accumulated image offsets: the difference is a nonzero lattice
    vector, so the component touches its own image. The rank of the set of such
    wrap vectors is the percolation dimensionality (0 none, 1 wire, 2 sheet,
    3 network).
    """
    active = set(int(x) for x in active)
    seen: dict[int, np.ndarray] = {}
    out = []
    for start in sorted(active):
        if start in seen:
            continue
        seen[start] = np.zeros(3, int)
        q, comp, wraps = deque([start]), [start], []
        while q:
            c = q.popleft()
            for t, off in zip(neighbours[c], nbr_offsets[c]):
                t = int(t)
                if t not in active:
                    continue
                shifted = seen[c] + np.asarray(off, int)
                if t not in seen:
                    seen[t] = shifted
                    comp.append(t)
                    q.append(t)
                else:
                    w = shifted - seen[t]
                    if np.any(w != 0):
                        wraps.append(tuple(int(x) for x in w))
        wraps = np.array(sorted(set(wraps)), dtype=int).reshape(-1, 3)
        rank = int(np.linalg.matrix_rank(wraps)) if len(wraps) else 0
        out.append({"members": comp, "size": len(comp),
                    "wrap_vectors": wraps, "percolation_rank": rank})
    return out


def defect_clusters(occ: np.ndarray, g: SiteGraph):
    """Clusters of connected defect sites, with their periodic wrap vectors."""
    defect = np.flatnonzero(occ == 0)
    comps = periodic_components(defect, g.neighbours, g.nbr_offsets)
    for c in comps:                       # keep the old key name for callers
        c["sites"] = c["members"]
    return comps


def observables(occ: np.ndarray, g: SiteGraph) -> dict:
    clusters = defect_clusters(occ, g)
    defects = np.flatnonzero(occ == 0)
    if len(defects):
        nbrs = [int(np.sum(occ[g.neighbours[int(s)]] == 0)) for s in defects]
        mean_nbrs = float(np.mean(nbrs))
    else:
        mean_nbrs = 0.0
    biggest = max(clusters, key=lambda c: c["size"]) if clusters else None
    return {
        "n_defects": int(len(defects)),
        "n_clusters": len(clusters),
        "largest_cluster": biggest["size"] if biggest else 0,
        "mean_defect_neighbours": mean_nbrs,
        "percolation_rank": max((c["percolation_rank"] for c in clusters),
                                default=0),
    }


# ---------------------------------------------------------------------------
# 7b. Triangular windows: the guest-diffusion percolation metric
# ---------------------------------------------------------------------------
@dataclass
class WindowGraph:
    """Triangular windows of the framework (guest-diffusion apertures).

    Each window is a triangular face of the Zr6 node net, bounded by 3 linker
    edges. A guest passes a window when enough of its 3 linkers are missing.
    The window-window graph (windows sharing a linker) is what a guest actually
    percolates through -- this is the criterion the JPCC 2024 paper's diffusion
    percolation is about, NOT bare defect-site adjacency.
    """
    cell: np.ndarray
    windows: np.ndarray                 # (W, 3) linker-edge indices per window
    win_frac: np.ndarray                # (W, 3) window centroid, fractional
    neighbours: list = field(default_factory=list)   # windows sharing a linker
    nbr_offsets: list = field(default_factory=list)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    def open_mask(self, occ: np.ndarray, threshold: int = 2) -> np.ndarray:
        """A window is open when >= `threshold` of its 3 linkers are missing."""
        missing = (occ[self.windows] == 0).sum(axis=1)
        return missing >= threshold

    def openness(self, occ: np.ndarray, threshold: int = 2) -> dict:
        """Open-window order parameter (defect correlation).

        A window is open when >= `threshold` of its 3 linkers are missing.
        Reports how many windows are open and how they cluster (via
        shared-linker adjacency). As defects correlate, open windows appear and
        merge into larger open clusters.

        NOTE: this is NOT the rigorous guest-diffusion percolation criterion.
        That requires the CAGE graph (cages as nodes, windows as the openings
        between them); a guest exits a cage through any of its windows, which is
        not the same as two windows sharing a linker. Largest-open-cluster is a
        defensible correlation proxy; a spanning/percolation rank on this graph
        is not, so it is deliberately not reported. See README.
        """
        open_w = np.flatnonzero(self.open_mask(occ, threshold))
        comps = periodic_components(open_w, self.neighbours, self.nbr_offsets)
        biggest = max(comps, key=lambda c: c["size"]) if comps else None
        return {
            "n_open_windows": int(len(open_w)),
            "open_fraction": float(len(open_w) / max(self.n_windows, 1)),
            "n_open_clusters": len(comps),
            "largest_open_cluster": biggest["size"] if biggest else 0,
        }


def build_window_graph(g: SiteGraph, verbose: bool = True) -> WindowGraph:
    """Enumerate triangular windows and their shared-linker adjacency.

    A window is a 3-cycle of nodes (a,b,c) each pair bridged by a linker. Found
    from node adjacency with image offsets; deduplicated by the frozenset of
    the 3 linker indices. Window-window offsets use the minimum image of the
    centroid difference (adjacent windows share a linker, so they are one
    linker apart -- minimum image is unambiguous), the same construction used
    for the site hop graph.
    """
    cell = g.cell
    node_frac = g.node_frac

    # node adjacency with offsets, reconstructed from edge midpoints
    node_adj: dict[int, list] = {n: [] for n in range(len(node_frac))}
    for s in range(g.n_sites):
        a, b = (int(x) for x in g.site_nodes[s])
        # b sits at 2*midpoint - a; offset is the integer lattice part
        b_img = 2 * g.site_frac[s] - node_frac[a]
        off = np.round(b_img - node_frac[b]).astype(int)
        node_adj[a].append((b, off, s))
        node_adj[b].append((a, -off, s))

    # find 3-cycles a-b-c with consistent offsets
    seen = {}
    windows, win_edges = [], []
    for a in range(len(node_frac)):
        for (b, o_ab, s_ab) in node_adj[a]:
            for (c, o_ac, s_ac) in node_adj[a]:
                if c <= b:
                    continue
                need = o_ac - o_ab                      # offset b_img -> c_img
                for (c2, o_bc, s_bc) in node_adj[b]:
                    if c2 == c and np.array_equal(o_bc, need):
                        key = frozenset((s_ab, s_ac, s_bc))
                        if len(key) == 3 and key not in seen:
                            seen[key] = True
                            win_edges.append((s_ab, s_ac, s_bc))
                            # centroid: a(home), b at a+o_ab, c at a+o_ac
                            cen = (node_frac[a]
                                   + (node_frac[b] + o_ab)
                                   + (node_frac[c] + o_ac)) / 3.0
                            windows.append(cen)
    win_frac = np.mod(np.array(windows), 1.0) if windows else np.zeros((0, 3))
    win_edges = np.array(win_edges, dtype=int).reshape(-1, 3)

    wg = WindowGraph(cell=np.asarray(cell, float), windows=win_edges,
                     win_frac=win_frac)
    _attach_window_hops(wg, g, verbose)
    if verbose:
        print(f"  windows: {wg.n_windows} triangular faces; "
              f"window adjacency {sorted(set(len(n) for n in wg.neighbours))}")
    return wg


def _attach_window_hops(wg: WindowGraph, g: SiteGraph, verbose: bool) -> None:
    """Windows are adjacent when they share a linker; offset by min-image."""
    edge_to_windows: dict[int, list[int]] = {}
    for w, tri in enumerate(wg.windows):
        for e in tri:
            edge_to_windows.setdefault(int(e), []).append(w)

    neighbours = [[] for _ in range(wg.n_windows)]
    offsets = [[] for _ in range(wg.n_windows)]
    for e, ws in edge_to_windows.items():
        for i in range(len(ws)):
            for j in range(len(ws)):
                if i == j:
                    continue
                w1, w2 = ws[i], ws[j]
                d = wg.win_frac[w2] - wg.win_frac[w1]
                off = -np.round(d).astype(int)          # min image w1 -> w2
                neighbours[w1].append(w2)
                offsets[w1].append(off)
    wg.neighbours = [np.array(n, dtype=int) for n in neighbours]
    wg.nbr_offsets = [np.array(o, dtype=int).reshape(-1, 3) for o in offsets]


# ---------------------------------------------------------------------------
# 8. Drop-in replacement for the Lattice class in uio66_defect_kmc.py
# ---------------------------------------------------------------------------
class StructureLattice:
    """Same interface the kMC loop uses (occ, neighbours, defect_sites,
    n_defect_neighbours) but backed by the real UiO-66 linker graph."""

    def __init__(self, p, graph: SiteGraph, occ: np.ndarray,
                 library: GeometryLibrary | None = None):
        self.p = p
        self.graph = graph
        self.library = library
        self.occ = np.asarray(occ, dtype=np.int8).copy()
        self.neighbours = graph.neighbours
        self.shape = (graph.n_sites, 1, 1)      # kept only for API compatibility

    def defect_sites(self) -> np.ndarray:
        return np.flatnonzero(self.occ == 0)

    def n_defect_neighbours(self, s: int) -> int:
        return int(np.sum(self.occ[self.neighbours[s]] == 0))

    def write_cif(self, path: str) -> None:
        if self.library is None:
            raise ValueError("no geometry library attached; build one from a "
                             "pristine CIF to write real structures")
        write_frame(compose(self.occ, self.library), path)

    def observables(self) -> dict:
        return observables(self.occ, self.graph)


def load_initial_state(defective_cif: str, pristine_cif: str,
                       hop_shell: str = "nearest", verbose: bool = True):
    """The whole setup in one call.

    Returns (graph, occ, library, report). Read the report before running
    anything: if formate_per_defect is not 2.0, the parse is wrong.
    """
    if verbose:
        print(f"[pristine] {pristine_cif}")
    pristine = read_frame(pristine_cif)
    nodes, _ = find_nodes(pristine)
    graph = build_site_graph(nodes, pristine.cell, hop_shell=hop_shell,
                             verbose=verbose)
    library = build_library(pristine, graph, verbose=verbose)

    if verbose:
        print(f"[defective] {defective_cif}")
    defective = read_frame(defective_cif)
    scale = np.linalg.norm(defective.cell, axis=1) / np.linalg.norm(pristine.cell, axis=1)
    if np.any(np.abs(scale - 1) > 0.05):
        print(f"  ! cell lengths differ from the pristine CIF by {scale}. "
              "Same supercell required for site matching.")
    occ, report = read_occupancy(defective, graph, verbose=verbose)
    if verbose:
        print(f"[state] {observables(occ, graph)}")
    return graph, occ, library, report


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print(__doc__)
        print("usage: python uio66_sites.py PRISTINE.cif DEFECTIVE.cif")
        sys.exit(1)
    load_initial_state(sys.argv[2], sys.argv[1])
