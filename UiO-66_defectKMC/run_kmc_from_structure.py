"""
run_kmc_from_structure.py -- kMC seeded from a REAL defect structure
====================================================================

Like run_kmc_percolation.py, but the starting configuration is read from an
actual defective structure file (CIF or LAMMPS data) instead of being generated
from the pristine cell. The node lattice, site graph, and cage graph are all
built from that file (missing-linker defects leave the Zr6 nodes in place), the
occupancy is read geometrically, and the kMC evolves it at fixed fraction while
the passive percolation monitor records each snapshot.

Usage:
    python run_kmc_from_structure.py random_1.cif \
        [--events N] [--snap M] [--temp K] [--dE eV] [--seed S]

CSV columns and monitor are identical to run_kmc_percolation.py.

Needs: numpy, scipy, uio66_sites.py, uio66_graph_kmc.py, cage_graph.py,
percolation_path.py in the same folder.
"""
import sys
import argparse
from collections import Counter

import numpy as np

from uio66_sites import find_nodes, build_site_graph, Frame
from uio66_graph_kmc import GraphKMC, GraphKMCParams
from cage_graph import build_cage_graph
from percolation_path import pathway_metrics


# ---------------------------------------------------------------------------
# readers (CIF with element symbols, or LAMMPS 'full' data -- positions only)
# ---------------------------------------------------------------------------
def read_structure(path):
    """Return (species-or-None, positions_cartesian, cell). For LAMMPS files the
    element labels are unreliable, so species is None and nodes are found from
    the densest 6-fold clusters (Zr) via charge if present."""
    if path.lower().endswith((".cif",)):
        cl, sp, fr = {}, [], []
        for ln in open(path):
            t = ln.split()
            for k in ("_cell_length_a", "_cell_length_b", "_cell_length_c"):
                if ln.strip().startswith(k):
                    cl[k] = float(t[1])
            if len(t) >= 7 and len(t[0]) <= 2 and t[0][:1].isalpha() and t[0][:1].isupper():
                try:
                    fr.append([float(t[3]), float(t[4]), float(t[5])]); sp.append(t[0])
                except ValueError:
                    pass
        cell = np.diag([cl["_cell_length_a"], cl["_cell_length_b"], cl["_cell_length_c"]])
        P = np.array(fr) @ cell
        return sp, P, cell
    # LAMMPS data
    lines = [l.rstrip() for l in open(path)]
    box = {}
    import re
    for l in lines:
        m = re.match(r"\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+(xlo xhi|ylo yhi|zlo zhi)", l)
        if m:
            box[m.group(3)] = (float(m.group(1)), float(m.group(2)))
    L = np.array([box["xlo xhi"][1] - box["xlo xhi"][0],
                  box["ylo yhi"][1] - box["ylo yhi"][0],
                  box["zlo zhi"][1] - box["zlo zhi"][0]])
    cell = np.diag(L)
    ai = [k for k, l in enumerate(lines) if l.strip().startswith("Atoms")][0]
    P, q = [], []
    for l in lines[ai + 1:]:
        t = l.split()
        if len(t) >= 7 and t[0].isdigit():
            q.append(float(t[3])); P.append([float(t[4]), float(t[5]), float(t[6])])
        elif t and not t[0].isdigit() and P:
            break
    return (np.array(q) > 2.0), np.array(P), cell   # species slot carries a Zr mask


def nodes_from_structure(species, P, cell):
    """Find Zr6 node centroids. CIF: use Zr symbols. LAMMPS: use the Zr mask
    (high charge) passed in the species slot."""
    if isinstance(species, list):              # CIF element symbols
        zr = P[[i for i, s in enumerate(species) if s == "Zr"]]
    else:                                      # LAMMPS Zr mask (boolean array)
        zr = P[np.asarray(species, bool)]
    frac = np.mod(zr @ np.linalg.inv(cell), 1.0)
    N = len(frac); adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            d = frac[i] - frac[j]; d -= np.round(d)
            if np.linalg.norm(d @ cell) < 4.2:
                adj[i].append(j); adj[j].append(i)
    seen, nodes = set(), []
    for i in range(N):
        if i in seen:
            continue
        st, c = [i], [i]; seen.add(i)
        while st:
            x = st.pop()
            for y in adj[x]:
                if y not in seen:
                    seen.add(y); st.append(y); c.append(y)
        seed = frac[c[0]]; d = frac[c] - seed; d -= np.round(d)
        nodes.append(np.mod(seed + d.mean(0), 1.0))
    return np.array(nodes)


def occupancy_from_positions(sg, cell, P, radius=2.0, min_atoms=4):
    inv = np.linalg.inv(cell); Pf = P @ inv
    occ = np.ones(sg.n_sites, dtype=np.int8)
    for s in range(sg.n_sites):
        d = Pf - sg.site_frac[s]; d -= np.round(d)
        occ[s] = 1 if (np.linalg.norm(d @ cell, axis=1) < radius).sum() >= min_atoms else 0
    return occ


# ---------------------------------------------------------------------------
def monitor(cg, occ):
    spec = Counter((occ[cg.win_sites] == 0).sum(axis=1).tolist())
    m1 = pathway_metrics(cg, occ, min_missing=1)
    m2 = pathway_metrics(cg, occ, min_missing=2)
    d = {"open1": m1["n_open_windows"], "rank1": m1["rank"],
         "disp1_mag": m1.get("displacement_magnitude", 0.0),
         "disp1_over_cell": m1.get("displacement_over_cell", 0.0),
         "dx": 0.0, "dy": 0.0, "dz": 0.0, "tortuosity1": m1.get("tortuosity", 0.0),
         "open2": m2["n_open_windows"], "rank2": m2["rank"],
         "extent2_over_cell": m2.get("extent_over_cell", m2.get("displacement_over_cell", 0.0)),
         "w0": spec.get(0, 0), "w1": spec.get(1, 0), "w2": spec.get(2, 0), "w3": spec.get(3, 0)}
    if m1["percolates"]:
        x = m1["displacement_over_cell_xyz"]
        d["dx"], d["dy"], d["dz"] = float(x[0]), float(x[1]), float(x[2])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("structure", help="starting defective structure (.cif or .lmpdat)")
    ap.add_argument("--events", type=int, default=40000)
    ap.add_argument("--snap", type=int, default=4000)
    ap.add_argument("--temp", type=float, default=500.0)
    ap.add_argument("--dE", type=float, default=-0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="kmc_from_structure_trajectory.csv")
    ap.add_argument("--save-structures", metavar="DIR", default=None,
                    help="write a CIF per snapshot + occupancies.npz into DIR")
    ap.add_argument("--template", default=None,
                    help="pristine CIF (any cell; tiled to match) to build the "
                         "geometry library for writing snapshot CIFs. Required "
                         "with --save-structures to get atomic structures; "
                         "without it, only occupancies.npz is saved.")
    args = ap.parse_args()

    import itertools
    from uio66_graph_kmc import random_valid_occupancy

    import io, contextlib
    def silent():                       # swallow setup warnings / validate / build chatter
        return contextlib.redirect_stdout(io.StringIO())

    species, P, cell = read_structure(args.structure)
    nodes = nodes_from_structure(species, P, cell)
    sg0 = build_site_graph(nodes, cell, hop_shell="shares_sbu", verbose=False)
    occ_in = occupancy_from_positions(sg0, cell, P)
    frac_in = float((occ_in == 0).mean())
    substituted = len(nodes) <= 4       # 1x1x1 input: run a fresh random on a 2x2x2
    with silent():
        if substituted:
            reps = np.array(list(itertools.product((0, 1), repeat=3)))
            nodes = np.vstack([(nodes + r) / 2 for r in reps]); cell = cell * 2
            sg = build_site_graph(nodes, cell, hop_shell="shares_sbu", verbose=False)
            occ0 = random_valid_occupancy(sg, frac_in, min_node_degree=2, seed=args.seed)
        else:
            sg, occ0 = sg0, occ_in
        cg = build_cage_graph(nodes, cell, sg.site_frac, verbose=False)
        p = GraphKMCParams(temperature=args.temp, rate_model="environment",
                           dE_per_defect_neighbour=args.dE, enable_diffusion=False,
                           swap_shell="shares_sbu", n_events=args.events,
                           snapshot_every=args.snap, seed=args.seed)
        kmc = GraphKMC(sg, occ0, p, validate=True)
    ndef = int((occ0 == 0).sum())

    cols = ["event", "time_s", "mean_defect_neighbours",
            "open1", "rank1", "disp1_mag", "disp1_over_cell", "dx", "dy", "dz",
            "tortuosity1", "open2", "rank2", "extent2_over_cell",
            "w0", "w1", "w2", "w3"]
    log = open(args.out, "w"); log.write(",".join(cols) + "\n")

    # optional snapshot structure saving
    snap_writer, occ_log, snap_dir = None, None, None
    if args.save_structures:
        import os
        from snapshot_io import OccupancyLog, build_geometry_writer
        snap_dir = args.save_structures
        os.makedirs(snap_dir, exist_ok=True)
        occ_log = OccupancyLog()
        if args.template:
            with silent():
                snap_writer = build_geometry_writer(args.template, sg, cell, verbose=False)

    def record(step, t):
        mdn = kmc.observables()["mean_defect_neighbours"]
        row = {"event": step, "time_s": t, "mean_defect_neighbours": round(mdn, 3),
               **monitor(cg, kmc.state)}
        log.write(",".join(str(row[c]) for c in cols) + "\n"); log.flush()
        if occ_log is not None:
            occ_log.add(step, kmc.state)
            if snap_writer is not None:
                snap_writer(kmc.state,
                            f"{snap_dir}/snapshot_{step:06d}.cif", f"step_{step}")
        return row

    print(f"start: {args.structure}  ({ndef} defects, {ndef/sg.n_sites:.0%}), "
          f"T={args.temp:g}K, dE={args.dE} eV"
          + ("  [1x1x1 input -> fresh random 2x2x2]" if substituted else ""))
    print(f"{'event':>7} {'mean_nbr':>8} | thr1(5A): {'rank':>4} {'|disp|/cell':>11} "
          f"{'(dx,dy,dz)':>16} | thr2: {'rank':>4} {'extent/cell':>11} | windows[0,1,2,3]")

    def show(r):
        print(f"{r['event']:>7} {r['mean_defect_neighbours']:>8.2f} | "
              f"{'':>10}{r['rank1']:>4} {r['disp1_over_cell']:>11.2f} "
              f"({r['dx']:+.1f},{r['dy']:+.1f},{r['dz']:+.1f})".rjust(16) +
              f" | {'':>6}{r['rank2']:>4} {r['extent2_over_cell']:>11.2f} | "
              f"[{r['w0']},{r['w1']},{r['w2']},{r['w3']}]")

    t = 0.0; show(record(0, t))
    for step in range(1, args.events + 1):
        dt = kmc.step()
        if dt == 0.0:
            print("[stop] no allowed events"); break
        t += dt
        if step % args.snap == 0:
            show(record(step, t))
    log.close()
    if occ_log is not None:
        occ_log.save(f"{snap_dir}/occupancies.npz", sg.site_frac, cell)
        print(f"occupancies -> {snap_dir}/occupancies.npz "
              f"({len(occ_log.steps)} snapshots)")
    print(f"trajectory -> {args.out}")


if __name__ == "__main__":
    main()
