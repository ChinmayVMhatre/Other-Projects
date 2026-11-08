"""Build an idealised UiO-66-like structure in pure numpy and run the whole
uio66_sites pipeline on it. No pymatgen, no real CIF needed."""
import numpy as np
from uio66_sites import (Frame, find_nodes, build_site_graph, inventory,
                         read_occupancy, build_library, compose, observables,
                         defect_clusters, fragment_centroid, fragments, classify)

A0, REP = 20.7, 2
BASIS = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])


def perp(u):
    t = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(u, t)) > 0.9:
        t = np.array([1.0, 0.0, 0.0])
    v = t - np.dot(t, u) * u
    return v / np.linalg.norm(v)


def bdc_atoms(mid, u):
    """8 C, 4 O, 4 H laid out as a real BDC: ring + two carboxylates."""
    v = perp(u)
    sp, xyz = [], []
    for sgn in (+1, -1):
        sp.append("C"); xyz.append(mid + sgn * 1.40 * u)               # ipso
        sp.append("C"); xyz.append(mid + sgn * 2.90 * u)               # COO carbon
        for pv in (+1, -1):
            sp.append("O"); xyz.append(mid + sgn * 3.55 * u + pv * 1.10 * v)
    for sgn in (+1, -1):
        for pv in (+1, -1):
            sp.append("C"); xyz.append(mid + sgn * 0.70 * u + pv * 1.21 * v)
            sp.append("H"); xyz.append(mid + sgn * 0.70 * u + pv * 2.30 * v)
    return sp, np.array(xyz)


def formate_atoms(mid, u):
    """What replaces a BDC: keep the two COO groups, H where the ring was."""
    v = perp(u)
    sp, xyz = [], []
    for sgn in (+1, -1):
        c = mid + sgn * 2.90 * u
        sp.append("C"); xyz.append(c)
        for pv in (+1, -1):
            sp.append("O"); xyz.append(mid + sgn * 3.55 * u + pv * 1.10 * v)
        sp.append("H"); xyz.append(c + 1.09 * sgn * u)                 # toward ring
    return sp, np.array(xyz)


def build(defect_sites=()):
    cell = np.eye(3) * A0 * REP
    inv = np.linalg.inv(cell)
    nodes = np.vstack([(BASIS + [i, j, k]) / REP
                       for i in range(REP) for j in range(REP) for k in range(REP)])
    sp, xyz = [], []
    for n in nodes @ cell:
        for ax in np.eye(3):
            for s in (+1, -1):
                sp.append("Zr"); xyz.append(n + s * 2.47 * ax)
        for c in np.array(list(np.ndindex(2, 2, 2))) * 2 - 1:
            sp.append("O"); xyz.append(n + 1.5 * c / np.sqrt(3))

    # edges of the fcc node net -> linker sites
    seen = {}
    for a in range(len(nodes)):
        for b in range(len(nodes)):
            for sh in np.array(list(np.ndindex(3, 3, 3))) - 1:
                d = (nodes[b] + sh - nodes[a]) @ cell
                r = np.linalg.norm(d)
                if 1e-6 < r < 15.0:
                    mid_raw = nodes[a] + 0.5 * (nodes[b] + sh - nodes[a])
                    key = tuple(np.round(np.mod(mid_raw, 1.0), 4))
                    if key not in seen:
                        seen[key] = (np.mod(mid_raw, 1.0) @ cell, d / r)
    keys = sorted(seen)
    for i, k in enumerate(keys):
        mid, u = seen[k]
        s2, x2 = (formate_atoms(mid, u) if i in defect_sites
                  else bdc_atoms(mid, u))
        sp.extend(s2)
        xyz.extend(list(x2))
    return Frame(sp, np.mod(np.array(xyz) @ inv, 1.0), cell), len(keys)


if __name__ == "__main__":
    print("=== pristine fixture ===")
    pristine, n_sites_expected = build()
    print(f"  atoms: {len(pristine)}  (sites: {n_sites_expected})")
    nodes, clusters = find_nodes(pristine)
    g = build_site_graph(nodes, pristine.cell, hop_shell="nearest")
    print("  fragment inventory:")
    inventory(pristine)
    lib = build_library(pristine, g)
    
    print("\n=== 25% defective fixture ===")
    rng = np.random.default_rng(7)
    defects = set(int(x) for x in rng.choice(n_sites_expected,
                                             size=int(0.25 * n_sites_expected),
                                             replace=False))
    defective, _ = build(defects)
    print(f"  atoms: {len(defective)}  (planted defects: {len(defects)})")
    occ, report = read_occupancy(defective, g)
    print("  observables:", observables(occ, g))
    
    print("\n=== round trip: compose(occ) -> re-read -> same occupancy? ===")
    rebuilt = compose(occ, lib)
    occ2, rep2 = read_occupancy(rebuilt, g, verbose=False)
    print(f"  rebuilt atoms: {len(rebuilt)}   formate groups: {rep2['n_formate']} "
          f"(expect {2 * rep2['n_defects']})")
    print(f"  occupancy identical: {np.array_equal(occ, occ2)}")
    
    print("\n=== percolation test ===")
    for name, o in [("25% random", occ), ("all defects", np.zeros(g.n_sites, np.int8))]:
        cl = defect_clusters(o, g)
        big = max(cl, key=lambda c: c["size"])
        print(f"  {name:<12} clusters={len(cl):<4} largest={big['size']:<4} "
              f"rank={max(c['percolation_rank'] for c in cl)}")
    print("  isolated single defect  rank=", end="")
    o = np.ones(g.n_sites, np.int8); o[0] = 0
    print(max(c["percolation_rank"] for c in defect_clusters(o, g)))
    
    print("\n=== shares_sbu shell ===")
    g22 = build_site_graph(nodes, pristine.cell, hop_shell="shares_sbu")
    