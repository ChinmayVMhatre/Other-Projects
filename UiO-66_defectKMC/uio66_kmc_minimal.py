"""
Minimal two-part kMC for UiO-66 missing-linker defects  (ASE + networkx)
========================================================================

Part 1 - attach / detach (3-state: empty / mono / di).
Mono- and di-carboxylic acids compete for the linker sites of the Zr6-node
graph. Each species has its OWN binding energy E_bind (attach) and desorption
energy E_diss (detach). A di acid = an intact linker; a mono acid (or an empty
site) = a missing-linker defect. Per species, the adsorption strength is

        S = E_diss - E_bind          (larger S = sticks better)

and the equilibrium coverages follow K = c * exp(S / kT) with c_di:c_mono = n:1:

        defect fraction = 1 - P_di = 1 - K_di / (1 + K_di + K_mono)

  * EQUAL energies for mono & di  ->  the binding terms cancel and
        defect fraction = 1 / (n + 1)   (set by the ratio, independent of T)
  * UNEQUAL energies (e.g. di binds more strongly, larger S_di) -> di is
        favored, so there are FEWER defects, and the fraction now depends on
        temperature (the S-difference is weighted by 1/kT).

Part 2 - rearrangement to a 1-D transport pathway (trajectory driver).
At the fixed defect fraction from Part 1, defects and linkers swap sites
(energy-neutral once the fraction is fixed -> unbiased random walk). Percolation
= a 1-D UNINTERRUPTED pathway for a guest molecule hopping cage -> cage through
open (defective) windows. run() logs the whole rearrangement to trajectory.csv
and prints a per-snapshot trace.

Structure I/O: ASE.  Graph + percolation: cage_graph / percolation_path.

Usage:
    python uio66_kmc_minimal.py pristine-uio66_unit_cell.cif \
        [--ratio N] [--temp K] \
        [--eb-di eV] [--ed-di eV] [--eb-mono eV] [--ed-mono eV] \
        [--rep R] [--events M] [--snap S] [--stop-rank K] [--outdir DIR]
"""
import os
import argparse
import numpy as np
import networkx as nx
from ase.io import read

from uio66_sites import build_site_graph
from cage_graph import build_cage_graph
from percolation_path import pathway_metrics
from uio66_graph_kmc import random_valid_occupancy

kB = 8.617333e-5                       # eV/K


def nodes_from_cif(cif, rep=2, node_cut=4.2):
    """Zr6 node centroids from a pristine CIF, tiled rep^3."""
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
    s = np.array([(x, y, z) for x in range(rep) for y in range(rep) for z in range(rep)])
    return np.vstack([(np.array(nf) + r) / rep for r in s]), cell * rep


# --- Part 1: 3-state attach/detach ------------------------------------------
def equilibrium_defect_fraction(ratio, temp, eb_di, ed_di, eb_mono, ed_mono):
    """ Analytical steady state (uses BOTH binding and desorption energies)."""
    kT = kB * temp
    K_di = ratio * np.exp((ed_di - eb_di) / kT)      # di adsorption constant
    K_mono = 1.0 * np.exp((ed_mono - eb_mono) / kT)  # mono adsorption constant
    P_di = K_di / (1.0 + K_di + K_mono)
    return 1.0 - P_di                                # defects = mono + empty


def part1_kmc(n_sites, ratio, temp, eb_di, ed_di, eb_mono, ed_mono,
              steps=60000, seed=0):
    """Gillespie 3-state kMC (empty=0, mono=1, di=2) -> defect fraction, time."""
    kT = kB * temp
    a_di = ratio * np.exp(-eb_di / kT)               # attach di  (uses E_bind)
    a_mono = 1.0 * np.exp(-eb_mono / kT)             # attach mono (uses E_bind)
    d_di = np.exp(-ed_di / kT)                       # detach di  (uses E_diss)
    d_mono = np.exp(-ed_mono / kT)                   # detach mono (uses E_diss)
    rng = np.random.default_rng(seed)
    state = np.zeros(n_sites, int)                   # start empty
    series, t = [], 0.0
    for _ in range(steps):
        ne = np.flatnonzero(state == 0); nm = np.flatnonzero(state == 1); nd = np.flatnonzero(state == 2)
        R = [len(ne) * a_di, len(ne) * a_mono, len(nm) * d_mono, len(nd) * d_di]
        tot = sum(R)
        if tot <= 0:
            break
        x = rng.random() * tot
        if x < R[0]:                       state[rng.choice(ne)] = 2       # attach di
        elif x < R[0] + R[1]:              state[rng.choice(ne)] = 1       # attach mono
        elif x < R[0] + R[1] + R[2]:       state[rng.choice(nm)] = 0       # detach mono
        else:                              state[rng.choice(nd)] = 0       # detach di
        t += rng.exponential(1.0 / tot)
        series.append(float(np.mean(state != 2)))    # defect = not a di-linker
    return float(np.mean(series[steps // 2:])), t


# --- Part 2: rearrangement trajectory ---------------------------------------
def observables(cg, occ):
    m = pathway_metrics(cg, occ, min_missing=1)       # window open if >=1 linker missing (5 A guest)
    nd = int((occ == 0).sum())
    return {"n_defects": nd, "defect_fraction": nd / len(occ),
            "open_fraction": m["n_open_windows"] / cg.n_windows,
            "largest_open_cluster": m["largest_open_cluster"],
            "percolation_rank": m["rank"],
            "displacement_over_cell": float(m.get("displacement_magnitude", 0.0))
            / float(np.linalg.norm(cg.cell[0]))}


def run(sg, cg, fraction, detach_rate, n_events=40000, snapshot_every=2000,
        stop_rank=0, outdir="kmc_out", seed=0):
    """ Rearrangement-kMC driver: log the trajectory and print a per-snapshot trace."""
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "trajectory.csv")
    rng = np.random.default_rng(seed)
    occ = random_valid_occupancy(sg, fraction, min_node_degree=2, seed=seed)

    with open(log_path, "w") as log:
        log.write("event,time_s,n_defects,defect_fraction,open_fraction,"
                  "largest_open_cluster,percolation_rank,displacement_over_cell\n")

        def record(step, t):
            o = observables(cg, occ)
            log.write(f"{step},{t:.6e},{o['n_defects']},{o['defect_fraction']:.4f},"
                      f"{o['open_fraction']:.4f},{o['largest_open_cluster']},"
                      f"{o['percolation_rank']},{o['displacement_over_cell']:.4f}\n")
            log.flush(); return o

        def reached_target(o):
            return stop_rank > 0 and o["percolation_rank"] >= stop_rank

        t = 0.0
        o = record(0, t)
        print(f"[init] defects={o['n_defects']}  rank={o['percolation_rank']}  "
              f"open_frac={o['open_fraction']:.3f}  "
              f"|disp|/cell={o['displacement_over_cell']:.2f}")

        step = 0
        for step in range(1, n_events + 1):
            d = rng.choice(np.flatnonzero(occ == 0))   # energy-neutral swap at fixed fraction
            n = rng.choice(np.flatnonzero(occ == 1))
            occ[d], occ[n] = 1, 0
            t += rng.exponential(1.0 / (len(occ) * detach_rate))
            if step % snapshot_every == 0:
                o = record(step, t)
                print(f"[{step:>7}] t={t:.2e}s  rank={o['percolation_rank']}  "
                      f"|disp|/cell={o['displacement_over_cell']:.2f}  "
                      f"open_frac={o['open_fraction']:.3f}  "
                      f"largest_open={o['largest_open_cluster']}/{cg.n_cages}")
                if reached_target(o):
                    print(f"[reached] percolation_rank >= {stop_rank} at step {step}; stopping.")
                    break

    print(f"\nDone. Trajectory -> {log_path}  ({t:.2e} s over {step} events)")
    return occ


# --- main -------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pristine_cif")
    ap.add_argument("--ratio", type=float, default=0.5, help="bulk di:mono ratio n (n:1)")
    ap.add_argument("--temp", type=float, default=400.0, help="temperature in K")
    ap.add_argument("--eb-di",  type=float, default=0.7, help="di  binding (attach) energy, eV")
    ap.add_argument("--ed-di",  type=float, default=1.0, help="di  desorption (detach) energy, eV")
    ap.add_argument("--eb-mono", type=float, default=0.7, help="mono binding (attach) energy, eV")
    ap.add_argument("--ed-mono", type=float, default=1.0, help="mono desorption (detach) energy, eV")
    ap.add_argument("--rep", type=int, default=2,help="Repitions of the system in xyz periodicity")
    ap.add_argument("--events", type=int, default=40000)
    ap.add_argument("--snap", type=int, default=2000)
    ap.add_argument("--stop-rank", type=int, default=0)
    ap.add_argument("--outdir", default="kmc_out")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    nodes, cell = nodes_from_cif(args.pristine_cif, rep=args.rep)
    sg = build_site_graph(nodes, cell, hop_shell="shares_sbu", verbose=False)
    cg = build_cage_graph(nodes, cell, sg.site_frac, verbose=False)
    print(f"{len(nodes)} Zr6 nodes, {sg.n_sites} linker sites, {cg.n_cages} cages\n")

    eq = dict(ratio=args.ratio, eb_di=args.eb_di, ed_di=args.ed_di,
              eb_mono=args.eb_mono, ed_mono=args.ed_mono)
    same = (args.eb_di == args.eb_mono and args.ed_di == args.ed_mono)
    print(f"Part 1 -- attach/detach, di:mono = {args.ratio:.0f}:1, T = {args.temp:.0f} K")
    print(f"  di:   E_bind={args.eb_di:.2f}  E_diss={args.ed_di:.2f}  (S_di = {args.ed_di-args.eb_di:+.2f} eV)")
    print(f"  mono: E_bind={args.eb_mono:.2f}  E_diss={args.ed_mono:.2f}  (S_mono = {args.ed_mono-args.eb_mono:+.2f} eV)")
    kfrac, t1 = part1_kmc(sg.n_sites, temp=args.temp, **eq, seed=args.seed)
    afrac = equilibrium_defect_fraction(temp=args.temp, **eq)
    print(f"  defect fraction: kMC {kfrac:.3f}   analytical {afrac:.3f}   "
          f"(1/(n+1) = {1/(args.ratio+1):.3f})")
    print(f"  {'EQUAL energies -> set by ratio, ~T-independent' if same else 'UNEQUAL energies -> depends on T and the S-difference'}")

    # temperature sweep: show equal (flat) vs the given/unequal case
    print("\n  defect fraction vs temperature:")
    print(f"    {'T(K)':>6} {'equal E':>9} {'this case':>11}")
    for T in (300, 400, 500, 600, 700):
        fe = equilibrium_defect_fraction(args.ratio, T, 0.9, 1.2, 0.9, 1.2)
        fc = equilibrium_defect_fraction(temp=T, **eq)
        print(f"    {T:>6} {fe:>9.3f} {fc:>11.3f}")

    kT = kB * args.temp
    detach = np.exp(-min(args.ed_di, args.ed_mono) / kT)   # sets Part-2 timescale
    print(f"\nPart 2 -- rearrange a random {afrac:.0%} structure (trajectory):")
    run(sg, cg, afrac, detach, n_events=args.events, snapshot_every=args.snap,
        stop_rank=args.stop_rank, outdir=args.outdir, seed=args.seed)
