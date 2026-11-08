"""
uio66_graph_kmc.py -- kMC in the node/edge latent space of defective UiO-66
===========================================================================

The model, exactly as specified
--------------------------------
- The Zr6O4(OH)4 SBUs are the graph NODES. They never move and never change
  state -- they are fixed scaffolding.
- The linker slots are the graph EDGES. Each edge carries a DISCRETE state:
      INTACT (1)     a BDC linker bridges the two nodes           weight w_intact
      DEFECTIVE (0)  no bridging linker; both nodes formate-capped weight w_defect
  Only edges change during the run. w_defect < w_intact ("weight reduced at a
  defective site"); both are integers you choose.
- kMC runs entirely in this latent space (integer arrays + fixed node
  coordinates). Atoms are only touched when a snapshot is written back to a
  structure -- see `to_structure` / the write_snapshot callback in `run`.

Event
-----
A defect relocates from a defective edge i to an INTACT edge j:
    state[i]: 0 -> 1   (linker heals, formate desorbs)
    state[j]: 1 -> 0   (linker removed, formate caps appear)
Defect count is conserved (fixed defect fraction, as in the JPCC 2024 paper).

Two constraints, both enforced at ENUMERATION time so BKL stays rejection-free:

1. COORDINATION SPHERE. i and j must share a Zr6 node -- the relocation is a
   single-SBU rearrangement. No swaps between linkers in different corners of
   the cell. `swap_shell` selects how wide the sphere is:
       "shares_sbu" : every edge on either of i's two nodes (the full first
                      coordination sphere; 22 edges in bulk UiO-66)
       "nearest"    : the geometrically closest subset (8 edges at ~7.3 A)

2. MINIMUM CONNECTIVITY. No node may fall below `min_node_degree` intact edges
   (default 2), so no SBU is ever left isolated or dangling by a single link.
   A move that would push either endpoint of j below the floor is never
   offered.

Rate
----
    k = nu * exp(-Ea / kB T),   Ea = Ea_gate + dE_per_defect_neighbour * n_nb
`n_nb` = number of defective edges sharing a node with the DESTINATION j
(excluding the mover). Negative dE => clustering is kinetically favoured.
Computed in O(1) per candidate from node bookkeeping, not by re-scanning edges.

Requires: numpy, and uio66_sites (for the SiteGraph). MOFUN only enters when
writing structures back out, via mofun_sites.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from uio66_sites import SiteGraph, observables, defect_clusters

KB_EV = 8.617333262e-5  # eV/K

INTACT = 1
DEFECTIVE = 0


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
@dataclass
class GraphKMCParams:
    # kinetics -- REPLACE barriers with your own DFT/CI-NEB values
    Ea_gate: float = 1.20            # eV, formate desorption / rearrangement gate
    nu: float = 1.0e13               # 1/s attempt frequency
    temperature: float = 500.0       # K
    Ea_floor: float = 0.05           # eV, minimum physical barrier

    # rate model
    #   "environment"     : Ea = Ea_gate + dE_per_defect_neighbour * n_nb(dest).
    #                       Kinetic bias only; does NOT sample an equilibrium.
    #   "detailed_balance": Ising-like. E = -interaction_J * (defect-defect
    #                       adjacencies); Ea = Ea_gate + dE/2 so forward/back
    #                       rates satisfy k_ij/k_ji = exp(-dE/kT). Anneal this to
    #                       drive a random config toward its clustered
    #                       equilibrium (use with anneal()).
    rate_model: str = "environment"
    dE_per_defect_neighbour: float = -0.02  # eV, "environment" model only
    interaction_J: float = 0.0       # eV, "detailed_balance" model; >0 favours
                                     # defect clustering. Ideally from DFT
                                     # defect-pair energies.

    # discrete edge weights (nodes carry no state)
    w_intact: int = 2
    w_defect: int = 1

    # --- acid diffusion / mass transfer (series resistance on each relocation) ---
    # A relocation is reaction (cap desorption/adsorption) in SERIES with the
    # diffusion of the freed acid molecule through the pore to the new site:
    #   1/k_reloc = 1/k_reaction + 1/k_diffusion
    # The reaction rate is the Ea above (with the chosen rate_model). The
    # diffusion barrier drops as the traversed pore opens up (more missing
    # linkers around the shared node), so acid moves easily through already-
    # open channels and is throttled in closed pores. This feedback favours
    # extended open channels over compact blobs.
    enable_diffusion: bool = False
    nu_diff: float = 1.0e13          # 1/s, acid diffusion attempt frequency
    Ea_diff_closed: float = 1.50     # eV, acid diffusion barrier in a CLOSED pore
                                     # (only the mover missing). Set >= Ea_gate so
                                     # mass transfer can be rate-limiting early.
    beta_open: float = 0.15          # eV lowered per extra missing linker around
                                     # the traversed (shared) node
    Ea_diff_floor: float = 0.30      # eV, open-pore diffusion still has a barrier

    # constraints
    min_node_degree: int = 2         # every node keeps >= this many intact edges
    swap_shell: str = "shares_sbu"   # or "nearest"

    # percolation / target-structure detection
    open_window_threshold: int = 2   # a window is "open" at >= this many missing
                                     # of its 3 linkers
    stop_on_percolation: bool = False
    stop_metric: str = "defect_rank"    # "defect_rank" | "largest_open_fraction"
    stop_value: float = 3.0             # defect_rank: stop when rank >= value
                                        # largest_open_fraction: stop when the
                                        # largest open-window cluster covers >=
                                        # this fraction of all windows

    # run control
    n_events: int = 200_000
    snapshot_every: int = 20_000
    seed: int = 0
    outdir: str = "graph_kmc_out"

    def __post_init__(self):
        if self.w_defect >= self.w_intact:
            raise ValueError("w_defect must be less than w_intact (weight is "
                             "reduced at a defective site)")
        if self.rate_model not in ("environment", "detailed_balance"):
            raise ValueError("rate_model must be 'environment' or "
                             "'detailed_balance'")
        if self.rate_model == "environment":
            span = abs(self.dE_per_defect_neighbour)
            if span * 12 > 0.5 * self.Ea_gate:
                print(f"  ! dE_per_defect_neighbour={self.dE_per_defect_neighbour} eV "
                      f"is strong: at high coordination it can rival Ea_gate="
                      f"{self.Ea_gate} eV and drive barriers into the floor. "
                      "Consider a smaller magnitude.")
        if self.rate_model == "detailed_balance" and self.enable_diffusion:
            print("  ! diffusion is ON with rate_model='detailed_balance'. The "
                  "mass-transfer resistance is a non-equilibrium kinetic effect "
                  "and breaks detailed balance, so anneal() no longer samples a "
                  "clean equilibrium. Turn diffusion off for equilibrium "
                  "sampling, or use rate_model='environment' with diffusion for "
                  "a kinetic (non-equilibrium) trajectory.")
        if self.enable_diffusion and self.Ea_diff_closed < self.Ea_gate:
            print(f"  note: Ea_diff_closed={self.Ea_diff_closed} < Ea_gate="
                  f"{self.Ea_gate}, so diffusion is never rate-limiting even in "
                  "a closed pore; the mass-transfer term will have little effect. "
                  "Raise Ea_diff_closed above Ea_gate to make closed pores throttle "
                  "relocation.")


# ---------------------------------------------------------------------------
# Coordination-sphere candidate lists (static: topology never changes)
# ---------------------------------------------------------------------------
def build_swap_candidates(graph: SiteGraph, shell: str = "shares_sbu",
                          verbose: bool = True) -> list[np.ndarray]:
    """For each edge, the edges it may swap a defect with.

    "shares_sbu": every edge incident to either endpoint node (built from
                  node_sites, so independent of how graph.neighbours was made).
    "nearest"   : the geometrically closest subset, using graph.neighbours if
                  the graph was built with hop_shell="nearest", else the 8
                  nearest by midpoint distance.
    """
    M = graph.n_sites
    if shell == "shares_sbu":
        cand = []
        for s in range(M):
            a, b = graph.site_nodes[s]
            edges = {e for e, _ in graph.node_sites[int(a)]}
            edges |= {e for e, _ in graph.node_sites[int(b)]}
            edges.discard(s)
            cand.append(np.array(sorted(edges), dtype=np.int64))
    elif shell == "nearest":
        # trust graph.neighbours only if it looks like a nearest shell
        looks_nearest = len(graph.neighbours) == M and all(
            len(n) <= 8 for n in graph.neighbours[:min(M, 50)])
        if looks_nearest:
            cand = [np.asarray(n, dtype=np.int64) for n in graph.neighbours]
        else:
            cand = []
            for s in range(M):
                d = graph.site_frac - graph.site_frac[s]
                d -= np.round(d)
                dist = np.linalg.norm(d @ graph.cell, axis=1)
                dist[s] = np.inf
                cand.append(np.argsort(dist)[:8].astype(np.int64))
    else:
        raise ValueError("swap_shell must be 'shares_sbu' or 'nearest'")

    # every candidate must genuinely share a node (defends the specification)
    for s in range(M):
        ai, bi = set(int(x) for x in graph.site_nodes[s]), None
        for j in cand[s]:
            aj = set(int(x) for x in graph.site_nodes[int(j)])
            if ai.isdisjoint(aj):
                raise ValueError(f"candidate {j} for edge {s} shares no node; "
                                 "swap would cross the cell")
    if verbose:
        sizes = sorted({len(c) for c in cand})
        print(f"  swap shell '{shell}': candidates per edge {sizes}")
    return cand


# ---------------------------------------------------------------------------
# The kMC engine
# ---------------------------------------------------------------------------
class GraphKMC:
    def __init__(self, graph: SiteGraph, occ: np.ndarray, p: GraphKMCParams,
                 validate: bool = True):
        self.g = graph
        self.p = p
        self.state = np.asarray(occ, dtype=np.int8).copy()
        self.rng = np.random.default_rng(p.seed)
        self._wg = None                  # window graph, built lazily

        # total incident edges per node (fixed) and current intact degree
        self.total_degree = np.array(
            [len(graph.node_sites[n]) for n in range(len(graph.node_frac))],
            dtype=np.int64)
        self.intact_degree = np.zeros(len(graph.node_frac), dtype=np.int64)
        for s in range(graph.n_sites):
            if self.state[s] == INTACT:
                for n in graph.site_nodes[s]:
                    self.intact_degree[int(n)] += 1

        self._assert_simple_graph()
        self.candidates = build_swap_candidates(graph, p.swap_shell, verbose=validate)

        if validate:
            self.validate_min_degree(raise_on_fail=True)
            n_def = int(np.sum(self.state == DEFECTIVE))
            print(f"  initial: {n_def} defective edges "
                  f"({100 * n_def / graph.n_sites:.1f}%), "
                  f"min node degree {int(self.intact_degree.min())}")

    # -- invariants ----------------------------------------------------------
    def _assert_simple_graph(self):
        seen = set()
        for s in range(self.g.n_sites):
            key = tuple(sorted(int(x) for x in self.g.site_nodes[s]))
            if key in seen:
                raise ValueError(
                    "graph has two edges between the same pair of nodes; the "
                    "degree constraint assumes a simple graph (UiO-66 fcu is). "
                    "Handle multi-edges before using this constraint.")
            seen.add(key)

    def validate_min_degree(self, raise_on_fail: bool = False) -> bool:
        bad = np.flatnonzero(self.intact_degree < self.p.min_node_degree)
        if len(bad):
            msg = (f"{len(bad)} node(s) below the connectivity floor "
                   f"(min {self.p.min_node_degree}); e.g. node {int(bad[0])} "
                   f"has degree {int(self.intact_degree[bad[0]])}. The initial "
                   "structure already violates the constraint.")
            if raise_on_fail:
                raise ValueError(msg)
            print("  ! " + msg)
            return False
        return True

    # -- event enumeration (constraints applied here) ------------------------
    def enumerate_events(self):
        """All allowed (i -> j) relocations and their rates."""
        p = self.p
        kT = KB_EV * p.temperature
        floor = p.min_node_degree
        events_i, events_j, rates = [], [], []

        defects = np.flatnonzero(self.state == DEFECTIVE)
        for i in defects:
            ni = self.g.site_nodes[i]
            cand = self.candidates[i]
            # destinations must currently be intact
            cand = cand[self.state[cand] == INTACT]
            if len(cand) == 0:
                continue
            nodes_j = self.g.site_nodes[cand]            # (K, 2)

            # losing node = the node of j that is NOT shared with i;
            # shared node = the pore the acid traverses on an i->j relocation
            shared0 = np.isin(nodes_j[:, 0], ni)
            losing = np.where(shared0, nodes_j[:, 1], nodes_j[:, 0])
            shared = np.where(shared0, nodes_j[:, 0], nodes_j[:, 1])
            # constraint: losing node must stay >= floor after -1
            allowed = self.intact_degree[losing] > floor
            cand, nodes_j = cand[allowed], nodes_j[allowed]
            shared = shared[allowed]
            if len(cand) == 0:
                continue

            # f(node) = number of defective linkers incident to that node
            fa_j = self.total_degree[nodes_j[:, 0]] - self.intact_degree[nodes_j[:, 0]]
            fb_j = self.total_degree[nodes_j[:, 1]] - self.intact_degree[nodes_j[:, 1]]
            # defective neighbours of destination j, excluding the mover i
            n_j = np.maximum(fa_j + fb_j - 1, 0)

            if p.rate_model == "environment":
                Ea = np.maximum(p.Ea_gate + p.dE_per_defect_neighbour * n_j,
                                p.Ea_floor)
            else:  # detailed_balance
                # defective neighbours of source i (i defective, counted at both
                # its nodes -> subtract 2); j intact so not counted
                fa_i = self.total_degree[ni[0]] - self.intact_degree[ni[0]]
                fb_i = self.total_degree[ni[1]] - self.intact_degree[ni[1]]
                n_i = max(int(fa_i + fb_i) - 2, 0)
                # E = -J * (defect-defect adjacencies); dE for moving i->j
                dE = -p.interaction_J * (n_j - n_i)
                Ea = np.maximum(p.Ea_gate + 0.5 * dE, p.Ea_floor)
            k_reaction = p.nu * np.exp(-Ea / kT)

            if p.enable_diffusion:
                # openness of the traversed pore: missing linkers on the shared
                # node (>=1 because the mover i is one of them). More open ->
                # lower diffusion barrier.
                n_open = self.total_degree[shared] - self.intact_degree[shared]
                Ea_d = np.maximum(p.Ea_diff_closed - p.beta_open * n_open,
                                  p.Ea_diff_floor)
                k_diff = p.nu_diff * np.exp(-Ea_d / kT)
                # series: 1/k = 1/k_reaction + 1/k_diff
                k = 1.0 / (1.0 / k_reaction + 1.0 / k_diff)
            else:
                k = k_reaction

            events_i.append(np.full(len(cand), i, dtype=np.int64))
            events_j.append(cand)
            rates.append(k)

        if not events_i:
            return (np.zeros((0, 2), np.int64), np.zeros(0),
                    self.p.Ea_floor)  # no moves
        ev = np.column_stack([np.concatenate(events_i),
                              np.concatenate(events_j)])
        return ev, np.concatenate(rates), None

    def step(self) -> float:
        ev, rates, _ = self.enumerate_events()
        if len(ev) == 0:
            return 0.0
        R = rates.sum()
        cum = np.cumsum(rates)
        e = int(np.searchsorted(cum, self.rng.random() * R))
        i, j = int(ev[e, 0]), int(ev[e, 1])

        # execute: defect moves i -> j
        self.state[i] = INTACT
        self.state[j] = DEFECTIVE
        for n in self.g.site_nodes[i]:
            self.intact_degree[int(n)] += 1
        for n in self.g.site_nodes[j]:
            self.intact_degree[int(n)] -= 1

        u = max(self.rng.random(), 1e-300)
        return -np.log(u) / R

    # -- discrete edge weights + observables ---------------------------------
    def edge_weights(self) -> np.ndarray:
        return np.where(self.state == INTACT, self.p.w_intact,
                        self.p.w_defect).astype(np.int64)

    def node_weight(self) -> np.ndarray:
        """Sum of incident edge weights per node (a discrete connection strength)."""
        w = self.edge_weights()
        out = np.zeros(len(self.g.node_frac), dtype=np.int64)
        for s in range(self.g.n_sites):
            for n in self.g.site_nodes[s]:
                out[int(n)] += w[s]
        return out

    @property
    def windows(self):
        if self._wg is None:
            from uio66_sites import build_window_graph
            self._wg = build_window_graph(self.g, verbose=False)
        return self._wg

    def observables(self) -> dict:
        obs = observables(self.state, self.g)
        obs["min_node_degree"] = int(self.intact_degree.min())
        obs["mean_node_degree"] = float(self.intact_degree.mean())
        obs["min_node_weight"] = int(self.node_weight().min())
        w = self.windows.openness(self.state, self.p.open_window_threshold)
        obs["open_fraction"] = w["open_fraction"]
        obs["largest_open_cluster"] = w["largest_open_cluster"]
        obs["largest_open_fraction"] = (w["largest_open_cluster"]
                                        / max(self.windows.n_windows, 1))
        return obs

    # -- back to geometry ----------------------------------------------------
    def to_structure(self, write_snapshot: Callable[[np.ndarray, str], None],
                     path: str) -> None:
        """Convert the current latent state to a real structure.

        `write_snapshot(occ, path)` does the geometry: pass
        `mofun_sites.MofunSites.write` for chemically capped CIFs, or any
        occupancy->file writer. `occ` uses the same 1=intact / 0=defective
        convention as `state`.
        """
        write_snapshot(self.state.copy(), path)


# ---------------------------------------------------------------------------
# Constraint-respecting random initial state (for synthetic runs / tests)
# ---------------------------------------------------------------------------
def random_valid_occupancy(graph: SiteGraph, defect_fraction: float,
                           min_node_degree: int = 2, seed: int = 0,
                           max_tries: int = 100) -> np.ndarray:
    """Place defects at random WITHOUT dropping any node below the floor.

    Real defective UiO-66 already satisfies the floor; use this only for
    synthetic starts or when you need a valid random configuration to test.
    """
    rng = np.random.default_rng(seed)
    n_target = int(round(defect_fraction * graph.n_sites))
    total = np.array([len(graph.node_sites[n]) for n in range(len(graph.node_frac))])

    occ = np.ones(graph.n_sites, dtype=np.int8)
    degree = total.copy()
    placed, tries = 0, 0
    order = list(range(graph.n_sites))
    while placed < n_target and tries < max_tries * n_target:
        rng.shuffle(order)
        progressed = False
        for s in order:
            if placed >= n_target:
                break
            if occ[s] == DEFECTIVE:
                continue
            a, b = (int(x) for x in graph.site_nodes[s])
            if degree[a] - 1 >= min_node_degree and degree[b] - 1 >= min_node_degree:
                occ[s] = DEFECTIVE
                degree[a] -= 1
                degree[b] -= 1
                placed += 1
                progressed = True
        tries += 1
        if not progressed:
            break
    if placed < n_target:
        print(f"  ! only placed {placed}/{n_target} defects without violating "
              f"the degree floor at fraction {defect_fraction}.")
    return occ


# ---------------------------------------------------------------------------
# Annealing driver: random -> clustered/percolating via detailed balance
# ---------------------------------------------------------------------------
def anneal(kmc: GraphKMC, schedule, window_graph=None, thresholds=(1, 2, 3),
           write_snapshot: Callable[[np.ndarray, str], None] | None = None,
           label: str = "anneal"):
    """Run a temperature schedule and log window percolation at each stage.

    schedule: list of (temperature_K, n_events) stages, hottest first. Cooling
    a detailed_balance run (interaction_J > 0) through the ordering transition
    drives a random defect configuration toward its clustered equilibrium; the
    window network is what is tested for percolation.

    window_graph: a uio66_windows.WindowGraph. If given, each stage logs the
    window-percolation rank at every threshold in `thresholds`. This is the
    order parameter that matches guest-transport percolation.
    """
    if kmc.p.rate_model != "detailed_balance":
        print("  ! anneal() is meant for rate_model='detailed_balance'. With "
              "'environment' the run has no equilibrium to anneal toward.")
    if window_graph is None:
        print("  ! no window_graph given; logging defect-adjacency percolation "
              "only. Pass one from uio66_windows for transport percolation.")

    os.makedirs(kmc.p.outdir, exist_ok=True)
    log_path = os.path.join(kmc.p.outdir, f"{label}.csv")
    from uio66_windows import percolation as _wperc

    def wobs():
        if window_graph is None:
            return {}
        out = {}
        for t in thresholds:
            r = _wperc(kmc.state, window_graph, t)
            out[f"open_frac_t{t}"] = round(r["open_fraction"], 4)
            out[f"win_rank_t{t}"] = r["window_percolation_rank"]
            out[f"win_largest_t{t}"] = r["largest_open_cluster"]
        return out

    wkeys = [f"{k}_t{t}" for t in thresholds for k in
             ("open_frac", "win_rank", "win_largest")] if window_graph else []
    with open(log_path, "w") as log:
        base = ("stage,temperature_K,event,time_s,n_defects,"
                "defect_adj_rank,mean_defect_neighbours,min_node_degree")
        log.write(base + ("," + ",".join(wkeys) if wkeys else "") + "\n")

        def record(stage, T, step, t):
            o = kmc.observables()
            w = wobs()
            row = (f"{stage},{T},{step},{t:.6e},{o['n_defects']},"
                   f"{o['percolation_rank']},{o['mean_defect_neighbours']:.4f},"
                   f"{o['min_node_degree']}")
            if wkeys:
                row += "," + ",".join(str(w[k]) for k in
                    [f"open_frac_t{t}" for t in thresholds]
                    + [f"win_rank_t{t}" for t in thresholds]
                    + [f"win_largest_t{t}" for t in thresholds])
                # reorder to match header grouping
            log.write(row + "\n")
            log.flush()
            return o, w

        t = 0.0
        o, w = record(0, schedule[0][0], 0, t)
        wtxt = (" ".join(f"t{k}:rank{w[f'win_rank_t{k}']}" for k in thresholds)
                if window_graph else f"defect_adj_rank={o['percolation_rank']}")
        print(f"[start] T={schedule[0][0]}K  defects={o['n_defects']}  {wtxt}")

        step = 0
        for stage, (T, n) in enumerate(schedule):
            kmc.p.temperature = float(T)
            for _ in range(n):
                dt = kmc.step()
                if dt == 0.0:
                    break
                t += dt
                step += 1
            o, w = record(stage, T, step, t)
            if write_snapshot:
                kmc.to_structure(write_snapshot,
                                 os.path.join(kmc.p.outdir,
                                              f"{label}_stage{stage:02d}_T{int(T)}.cif"))
            if window_graph:
                wtxt = "  ".join(f"t{k}: rank {w[f'win_rank_t{k}']}, "
                                 f"open {w[f'open_frac_t{k}']}" for k in thresholds)
            else:
                wtxt = f"defect_adj_rank={o['percolation_rank']}"
            print(f"[stage {stage}] T={T}K  mean_nbr={o['mean_defect_neighbours']:.2f}  {wtxt}")

    print(f"\nDone. Anneal log -> {log_path}")
    return log_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(kmc: GraphKMC, write_snapshot: Callable[[np.ndarray, str], None] | None = None):
    p = kmc.p
    os.makedirs(p.outdir, exist_ok=True)
    log_path = os.path.join(p.outdir, "trajectory.csv")

    with open(log_path, "w") as log:
        log.write("event,time_s,n_defects,n_clusters,largest_cluster,"
                  "percolation_rank,min_node_degree,mean_node_degree,"
                  "mean_defect_neighbours,open_fraction,largest_open_cluster,"
                  "largest_open_fraction\n")

        def record(step, t):
            o = kmc.observables()
            log.write(f"{step},{t:.6e},{o['n_defects']},{o['n_clusters']},"
                      f"{o['largest_cluster']},{o['percolation_rank']},"
                      f"{o['min_node_degree']},{o['mean_node_degree']:.4f},"
                      f"{o['mean_defect_neighbours']:.4f},{o['open_fraction']:.4f},"
                      f"{o['largest_open_cluster']},{o['largest_open_fraction']:.4f}\n")
            log.flush()
            return o

        def reached_target(o) -> bool:
            if not p.stop_on_percolation:
                return False
            if p.stop_metric == "defect_rank":
                return o["percolation_rank"] >= p.stop_value
            if p.stop_metric == "largest_open_fraction":
                return o["largest_open_fraction"] >= p.stop_value
            return False

        t = 0.0
        o = record(0, t)
        if write_snapshot:
            kmc.to_structure(write_snapshot,
                             os.path.join(p.outdir, "snapshot_000000.cif"))
        print(f"[init] T={p.temperature}K  defects={o['n_defects']}  "
              f"min_degree={o['min_node_degree']}  defect_rank={o['percolation_rank']}  "
              f"open_frac={o['open_fraction']:.3f}  "
              f"largest_open={o['largest_open_fraction']:.3f}")

        step = 0
        for step in range(1, p.n_events + 1):
            dt = kmc.step()
            if dt == 0.0:
                print("[stop] no allowed events (constraints block all moves)")
                break
            t += dt
            if step % p.snapshot_every == 0:
                o = record(step, t)
                if write_snapshot:
                    kmc.to_structure(
                        write_snapshot,
                        os.path.join(p.outdir, f"snapshot_{step:06d}.cif"))
                print(f"[{step:>7}] t={t:.3e}s  defect_rank={o['percolation_rank']}  "
                      f"largest_open={o['largest_open_fraction']:.3f}  "
                      f"min_deg={o['min_node_degree']}  "
                      f"mean_nbr={o['mean_defect_neighbours']:.3f}")
                if reached_target(o):
                    print(f"[reached] {p.stop_metric} >= {p.stop_value} at step "
                          f"{step}; stopping.")
                    break

    print(f"\nDone. Trajectory -> {log_path}  ({t:.3e} s over {step} events)")
    return t


# ---------------------------------------------------------------------------
# Convenience: assemble everything from MOFUN
# ---------------------------------------------------------------------------
def from_mofun(defective_cif: str, pristine_cif: str,
               linker_cml: str = "uio66-linker.cml",
               defective_cml: str = "uio66-linker-defective.cml",
               p: GraphKMCParams | None = None, atol: float = 0.05,
               cache: str | None = None):
    """Build a ready-to-run GraphKMC and a MOFUN write_snapshot callback."""
    import mofun_sites
    p = p or GraphKMCParams()
    hop = "nearest" if p.swap_shell == "nearest" else "shares_sbu"
    sites, occ, report = mofun_sites.load_initial_state(
        defective_cif, pristine_cif, linker_cml, defective_cml,
        hop_shell=hop, atol=atol, cache=cache)
    kmc = GraphKMC(sites.graph, occ, p)
    return kmc, (lambda o, path: sites.write(o, path)), report


if __name__ == "__main__":
    print(__doc__)
    print("Build a graph with uio66_sites or mofun_sites, then:")
    print("  kmc = GraphKMC(graph, occ, GraphKMCParams(temperature=600))")
    print("  run(kmc, write_snapshot=...)")
