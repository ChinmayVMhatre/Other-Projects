"""Check build_site_graph_from_matches() against build_site_graph().

MOFUN isn't installed here, so we fake its output: take the BDC fragments of the
ideal fixture and hand over their unwrapped cartesian positions, which is exactly
what find_pattern_in_structure(..., return_positions_and_quats=True) returns.
"""
import numpy as np
from uio66_sites import (find_nodes, build_site_graph, build_site_graph_from_matches,
                         inventory, fragments, classify, defect_clusters, observables)
from test_uio66_sites import build

pristine, n_sites = build()
nodes, _ = find_nodes(pristine)

# --- fake MOFUN match_positions: (M, n_atoms, 3) unwrapped cartesian ---
groups = inventory(pristine, verbose=False)
match_positions = []
for frag in groups["bdc"]:
    seed = pristine.frac[frag[0]]
    d = pristine.frac[frag] - seed
    d -= np.round(d)
    match_positions.append((seed + d) @ pristine.cell)
match_positions = np.array(match_positions)
print(f"fake match_positions: {match_positions.shape}")

print("\n--- graph from node-net edges ---")
g_edge = build_site_graph(nodes, pristine.cell, hop_shell="nearest")
print("\n--- graph from MOFUN matches ---")
g_mof = build_site_graph_from_matches(match_positions, nodes, pristine.cell,
                                      hop_shell="nearest")

# --- do the two graphs describe the same lattice? ---
def canon(g):
    """Sorted site midpoints + sorted neighbour-vector multiset per site."""
    order = np.lexsort(np.round(g.site_frac, 4).T[::-1])
    key = np.round(g.site_frac[order], 4)
    vecs = []
    for s in order:
        v = (g.site_frac[g.neighbours[s]] + g.nbr_offsets[s] - g.site_frac[s]) @ g.cell
        vecs.append(np.sort(np.round(np.linalg.norm(v, axis=1), 3)))
    return key, np.array(vecs), order

k1, v1, o1 = canon(g_edge)
k2, v2, o2 = canon(g_mof)
print(f"\nsame site positions : {np.allclose(k1, k2, atol=1e-3)}")
print(f"same hop geometry   : {np.allclose(v1, v2, atol=1e-3)}")

# --- topology equality: same neighbour sets after relabelling ---
lab1 = {tuple(k): i for i, k in enumerate(k1)}
remap_e = {int(s): lab1[tuple(np.round(g_edge.site_frac[s], 4))] for s in range(g_edge.n_sites)}
remap_m = {int(s): lab1[tuple(np.round(g_mof.site_frac[s], 4))] for s in range(g_mof.n_sites)}
adj_e = {remap_e[s]: sorted(remap_e[int(t)] for t in g_edge.neighbours[s]) for s in range(g_edge.n_sites)}
adj_m = {remap_m[s]: sorted(remap_m[int(t)] for t in g_mof.neighbours[s]) for s in range(g_mof.n_sites)}
print(f"identical adjacency : {adj_e == adj_m}")

# --- percolation must agree too (this is what uses the image offsets) ---
rng = np.random.default_rng(3)
for frac in (0.15, 0.25, 0.40):
    occ_e = np.ones(g_edge.n_sites, np.int8)
    pick = rng.choice(g_edge.n_sites, int(frac * g_edge.n_sites), replace=False)
    occ_e[pick] = 0
    occ_m = np.ones(g_mof.n_sites, np.int8)
    for s in pick:
        occ_m[[k for k, v in remap_m.items() if v == remap_e[int(s)]][0]] = 0
    oe, om = observables(occ_e, g_edge), observables(occ_m, g_mof)
    same = all(oe[k] == om[k] for k in oe)
    print(f"  defects {frac:.0%}: edge {oe} | matches agree: {same}")

print("\n--- shares_sbu shell from matches ---")
g22 = build_site_graph_from_matches(match_positions, nodes, pristine.cell,
                                    hop_shell="shares_sbu")
