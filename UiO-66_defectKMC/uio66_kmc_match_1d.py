"""
Rearrange a random defect structure to a 1-D percolating one (kMC)
==================================================================

Fixed Zr6 nodes, fixed missing-linker fraction. Defect and linker sites swap
locations; because the binding and desorption energies are equal (Part 1), each
swap is energy-neutral, so the kMC is an unbiased random walk over arrangements
at fixed fraction.

Percolation here = a 1-D UNINTERRUPTED PATHWAY for a guest molecule: it travels
between metal-oxide nodes that bear a monocarboxylic acid (missing-linker
defect), hopping across the missing linkers, and crosses the periodic cell in
one direction. We compare GEOMETRIC FEATURES against a known percolating
reference: the feature is the transport pathway's displacement vector, reduced
to sorted |components| per cell-edge --

    no pathway -> [0,0,0]    1-D -> [0,0,1]    2-D -> [0,1,1]    3-D -> [1,1,1]

The run stops when the rearranged structure's feature matches the reference's
1-D feature [0,0,1] (a single uninterrupted channel), i.e. the molecule gains
one and only one uninterrupted transport direction.

Reuses the validated cage_graph.py (guest-transport graph) and
percolation_path.py (displacement metric). Structure I/O via ASE.

Usage:
    python uio66_kmc_match_1d.py PRISTINE.cif [KNOWN.cif|KNOWN.lmpdat]
        [--ratio N] [--temp K] [--seeds S] [--rep R] [--steps M]
    # KNOWN is optional; omit it to match an ideal direct 1-D channel.
"""
import numpy as np
import networkx as nx

from uio66_sites import build_site_graph
from cage_graph import build_defect_graph
from percolation_path import pathway_metrics
from uio66_graph_kmc import random_valid_occupancy

RNG = np.random.default_rng(0)


def nodes_from_cif(cif, rep=2, node_cut=4.2):
    """Zr6 node centroids from a pristine CIF, tiled rep^3."""
    from ase.io import read
    a = read(cif); cell = a.cell.array; inv = np.linalg.inv(cell)
    zr = a.positions[a.numbers == 40]
    mic = lambda d: (d @ inv - np.round(d @ inv)) @ cell
    g = nx.Graph(); g.add_nodes_from(range(len(zr)))
    for i in range(len(zr)):
        for j in range(i + 1, len(zr)):
            if np.linalg.norm(mic(zr[i] - zr[j])) < node_cut:
                g.add_edge(i, j)
    nf = [np.mod((zr[list(c)[0]] + np.mean([mic(zr[k] - zr[list(c)[0]]) for k in c], 0)) @ inv, 1)
          for c in nx.connected_components(g)]
    shift = np.array([(x, y, z) for x in range(rep) for y in range(rep) for z in range(rep)])
    return np.vstack([(np.array(nf) + s) / rep for s in shift]), cell * rep


def occupancy_from_file(path, sg, cell, radius=2.0):
    """Occupancy (1 intact / 0 defect) by atoms near each site midpoint.
    Reads a CIF (element symbols) or LAMMPS data (positions only)."""
    if path.lower().endswith(".cif"):
        from ase.io import read
        a = read(path); P = a.positions
    else:
        lines = [l.rstrip() for l in open(path)]
        ai = [k for k, l in enumerate(lines) if l.strip().startswith("Atoms")][0]
        P = np.array([[float(x) for x in l.split()[4:7]] for l in lines[ai + 1:]
                      if len(l.split()) >= 7 and l.split()[0].isdigit()])
    inv = np.linalg.inv(cell); Pf = P @ inv
    occ = np.ones(sg.n_sites, np.int8)
    for s in range(sg.n_sites):
        d = Pf - sg.site_frac[s]; d -= np.round(d)
        occ[s] = 1 if (np.linalg.norm(d @ cell, axis=1) < radius).sum() >= 4 else 0
    return occ


def feature(cg, occ):
    """Geometric feature of the guest transport pathway: the spanning
    displacement magnitude in cell-edge units. 0 = no uninterrupted pathway;
    ~1.0 = a direct 1-D channel along a cell axis (like the known structure);
    ~1.41 = a diagonal 1-D channel."""
    m = pathway_metrics(cg, occ, min_missing=1)          # open window = >=1 missing (5 A guest)
    return float(m.get("displacement_magnitude", 0.0)) / float(np.linalg.norm(cg.cell[0]))


def kmc_match(sg, cg, occ0, ref_feat, steps=40000, tol=0.25, log_every=0):
    """Energy-neutral swaps until the feature matches the reference's.

    Match = the structure gains a 1-D uninterrupted transport pathway whose
    displacement magnitude equals the known structure's (a direct 1-D channel).
    `log_every` > 0 prints the feature every that many swaps.
    """
    occ = occ0.copy()
    for t in range(steps):
        f = feature(cg, occ)
        if log_every and t % log_every == 0:
            print(f"    step {t:>6}: feature {f:.2f}")
        if abs(f - ref_feat) < tol:
            return t, occ, True
        d = RNG.choice(np.flatnonzero(occ == 0))         # a defect site
        n = RNG.choice(np.flatnonzero(occ == 1))         # an intact-linker site
        occ[d], occ[n] = 1, 0                            # swap (fraction fixed)
    return steps, occ, abs(feature(cg, occ) - ref_feat) < tol


KB = 8.617333e-5          # eV/K
NU, E_DES = 1.0e13, 1.2    # 1/s attempt frequency, eV desorption energy (FIXED)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Rearrange a random dilute defect structure into a 1-D "
                    "percolating one; match a known structure by its transport-"
                    "pathway displacement feature.")
    ap.add_argument("pristine_cif", help="pristine UiO-66 cell (CIF)")
    ap.add_argument("known", nargs="?", default=None,
                    help="known percolating structure (CIF or LAMMPS) whose 1-D "
                         "transport-pathway feature is the match target; omit to "
                         "match an ideal direct 1-D channel")
    ap.add_argument("--ratio", type=float, default=9.0,
                    help="bulk di:mono acid ratio n (:1); defect fraction = 1/(n+1)")
    ap.add_argument("--temp", type=float, default=500.0,
                    help="temperature (K); sets the physical timescale (not the outcome)")
    ap.add_argument("--seeds", type=int, default=8, help="number of random starts to try")
    ap.add_argument("--rep", type=int, default=2, help="supercell replication")
    ap.add_argument("--steps", type=int, default=40000, help="max swaps per start (budget)")
    ap.add_argument("--log-every", type=int, default=0,
                    help="print the feature every N swaps during a run (0 = off)")
    args = ap.parse_args()

    frac = 1.0 / (args.ratio + 1.0)                          # acid ratio -> defect fraction
    dt = 1.0 / (NU * np.exp(-E_DES / (KB * args.temp)))      # seconds per swap (desorption-limited)

    nodes, cell = nodes_from_cif(args.pristine_cif, rep=args.rep)
    sg = build_site_graph(nodes, cell, hop_shell="shares_sbu", verbose=False)
    cg = build_defect_graph(sg, verbose=False)
    print(f"{len(nodes)} Zr6 nodes, {sg.n_sites} linker sites (node defective if it bears a mono)")
    print(f"acid ratio di:mono = {args.ratio:g}:1  ->  defect fraction {frac:.1%}")
    print(f"temperature {args.temp:g} K  ->  {dt:.2e} s per swap "
          f"(E_des = {E_DES} eV, fixed; energy-neutral swaps, so T sets time not outcome)")

    # reference feature: the known percolating structure's 1-D channel (|disp|~1)
    if args.known:
        ref_feat = feature(cg, occupancy_from_file(args.known, sg, cell))
        print(f"known structure {args.known}: feature |disp|/cell = {ref_feat:.2f}")
    else:
        ref_feat = 1.0
        print("target feature |disp|/cell = 1.00 (ideal direct 1-D channel)")

    print(f"\nrearrange random starts at {frac:.1%} defects (budget {args.steps} swaps):")
    swaps = []                                   # swaps-to-match per seed (None = failed)
    for seed in range(args.seeds):
        occ0 = random_valid_occupancy(sg, frac, min_node_degree=2, seed=seed)
        f0 = feature(cg, occ0)
        if f0 >= 0.5:                            # already has a 1-D pathway
            print(f"  seed {seed}: start {f0:.2f} -> already a 1-D pathway (0 swaps)")
            swaps.append(0); continue
        t, occ, ok = kmc_match(sg, cg, occ0, ref_feat,
                               steps=args.steps, log_every=args.log_every)
        if ok:
            print(f"  seed {seed}: 0.00 -> {feature(cg, occ):.2f}  "
                  f"matched at {t} swaps = {t*dt:.2e} s")
            swaps.append(t)
        else:
            print(f"  seed {seed}: 0.00 -> {feature(cg, occ):.2f}  "
                  f"no match in {args.steps} swaps")
            swaps.append(None)
    done = [s for s in swaps if s is not None]
    print(f"\nsummary: {len(done)}/{len(swaps)} starts reached a 1-D channel; "
          f"swaps-to-match min/median/max = "
          f"{min(done) if done else '-'}/"
          f"{int(np.median(done)) if done else '-'}/"
          f"{max(done) if done else '-'}"
          + (f" ; median time {int(np.median(done))*dt:.2e} s at {args.temp:g} K"
             if done else ""))
