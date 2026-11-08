"""Exercise uio66_graph_kmc on the ideal fixture. No MOFUN needed: geometry
round-trip uses uio66_sites.build_library / compose."""
import numpy as np
from uio66_sites import (find_nodes, build_site_graph, build_library, compose,
                         read_occupancy, observables, write_frame)
from uio66_graph_kmc import (GraphKMC, GraphKMCParams, run, build_swap_candidates,
                             random_valid_occupancy, INTACT, DEFECTIVE)
from test_uio66_sites import build

pristine, n_sites = build()
nodes, _ = find_nodes(pristine)
graph = build_site_graph(nodes, pristine.cell, hop_shell="shares_sbu", verbose=False)
lib = build_library(pristine, graph, verbose=False)
print(f"fixture: {len(nodes)} nodes, {graph.n_sites} edges")

for shell in ("shares_sbu", "nearest"):
    cand = build_swap_candidates(graph, shell)
    print(f"  {shell}: per-edge {sorted({len(c) for c in cand})}")

occ0 = random_valid_occupancy(graph, 0.25, min_node_degree=2, seed=1)
print(f"initial defects: {int(np.sum(occ0==0))}")


def run_and_check(dE, seed, label, n=8000):
    p = GraphKMCParams(temperature=600.0, dE_per_defect_neighbour=dE,
                       n_events=n, snapshot_every=n, seed=seed,
                       swap_shell="shares_sbu", w_intact=2, w_defect=1,
                       outdir=f"gk_{label}")
    kmc = GraphKMC(graph, occ0, p, validate=True)
    n_def0 = int(np.sum(kmc.state == DEFECTIVE))
    checked = 0
    for _ in range(n):
        before = kmc.state.copy()
        dt = kmc.step()
        if dt == 0.0:
            break
        changed = np.flatnonzero(before != kmc.state)
        assert len(changed) == 2
        i, j = (changed if before[changed[0]] == DEFECTIVE else changed[::-1])
        assert before[i] == DEFECTIVE and kmc.state[i] == INTACT
        assert before[j] == INTACT and kmc.state[j] == DEFECTIVE
        ni = set(int(x) for x in graph.site_nodes[i])
        nj = set(int(x) for x in graph.site_nodes[j])
        assert not ni.isdisjoint(nj), "swap crossed the cell"
        assert kmc.intact_degree.min() >= p.min_node_degree
        checked += 1
    o = kmc.observables()
    assert int(np.sum(kmc.state == DEFECTIVE)) == n_def0
    assert set(np.unique(kmc.edge_weights())).issubset({p.w_defect, p.w_intact})
    print(f"  [{label}] {checked} moves | defects={n_def0} | "
          f"min_deg={o['min_node_degree']} | largest={o['largest_cluster']} | "
          f"rank={o['percolation_rank']} | mean_nbr={o['mean_defect_neighbours']:.3f}")
    return kmc, o


print("\n--- clustering (dE=-0.02) ---")
kmc_c, obs_c = run_and_check(-0.02, 10, "cluster")
print("--- control   (dE= 0.0 ) ---")
kmc_0, obs_0 = run_and_check(0.0, 10, "control")
print(f"\nclustering raises defect adjacency: "
      f"{obs_c['mean_defect_neighbours']:.3f} vs {obs_0['mean_defect_neighbours']:.3f} "
      f"-> {obs_c['mean_defect_neighbours'] > obs_0['mean_defect_neighbours']}")

print("\n--- round trip latent -> geometry -> latent ---")
frame = compose(kmc_c.state, lib)
occ_back, rep = read_occupancy(frame, graph, verbose=False)
print(f"  {len(frame)} atoms; formate/defect={rep['formate_per_defect']}; "
      f"identical={np.array_equal(kmc_c.state.astype(int), occ_back.astype(int))}")

print("\n--- run() with snapshots ---")
writer = lambda o, path: np.save(path.replace(".cif", ".npy"), o)
p = GraphKMCParams(temperature=600.0, n_events=4000, snapshot_every=2000,
                   dE_per_defect_neighbour=-0.02, seed=5, outdir="gk_run")
run(GraphKMC(graph, occ0, p), write_snapshot=writer)
import os
print("  snapshots:", sorted(f for f in os.listdir("gk_run") if f.endswith(".npy")))

print("\n--- min_node_degree=6 must bite ---")
occ_tight = random_valid_occupancy(graph, 0.25, min_node_degree=6, seed=2)
p = GraphKMCParams(temperature=600.0, n_events=3000, snapshot_every=3000,
                   min_node_degree=6, dE_per_defect_neighbour=-0.02, seed=2,
                   outdir="gk_tight")
kmc = GraphKMC(graph, occ_tight, p)
mn = int(kmc.intact_degree.min())
for _ in range(3000):
    if kmc.step() == 0.0:
        break
    mn = min(mn, int(kmc.intact_degree.min()))
print(f"  min node degree observed: {mn} (floor 6)")
assert mn >= 6
print("\nALL INVARIANTS HOLD")
