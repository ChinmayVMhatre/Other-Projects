"""Validate cage_graph.py against a pristine UiO-66 P1 CIF.

Usage: python test_cage_graph.py pristine-uio66_unit_cell.cif

Checks (scaled by number of conventional cells): cage counts (4 oct + 8 tet per
cell), window count (32 per cell), windows-per-cage profile, every window edge
matching a linker site, all-open -> percolation rank 3, pristine -> rank 0.
Also runs a random-fraction sweep to show percolation onset.
"""
import sys
import numpy as np
from uio66_sites import find_nodes, build_site_graph, Frame
from cage_graph import build_cage_graph, validate


def read_p1_cif(path):
    cell_len, species, frac = {}, [], []
    for ln in open(path):
        t = ln.split()
        for k in ("_cell_length_a", "_cell_length_b", "_cell_length_c"):
            if ln.strip().startswith(k):
                cell_len[k] = float(t[1])
        if len(t) >= 7 and len(t[0]) <= 2 and t[0][:1].isalpha() and t[0][:1].isupper():
            try:
                frac.append([float(t[3]), float(t[4]), float(t[5])])
                species.append(t[0])
            except ValueError:
                pass
    cell = np.diag([cell_len["_cell_length_a"], cell_len["_cell_length_b"],
                    cell_len["_cell_length_c"]])
    return Frame(species, np.array(frac), cell)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    frame = read_p1_cif(sys.argv[1])
    nodes, _ = find_nodes(frame)
    sg = build_site_graph(nodes, frame.cell, hop_shell="shares_sbu", verbose=False)
    print(f"{len(nodes)} Zr6 nodes, {sg.n_sites} linker sites\n")

    ok = validate(nodes, frame.cell, sg.site_frac)
    assert ok, "cage-graph validation failed"

    print("\nrandom open-window sweep (threshold = 1 missing linker):")
    g = build_cage_graph(nodes, frame.cell, sg.site_frac, verbose=False)
    n_sites = sg.n_sites
    rng = np.random.default_rng(0)
    for defect_frac in (0.1, 0.25, 0.4, 0.6):
        occ = np.ones(n_sites, np.int8)
        occ[rng.choice(n_sites, int(defect_frac * n_sites), replace=False)] = 0
        p = g.percolation(occ, threshold=1)
        print(f"  defects {defect_frac:.0%}: open windows "
              f"{p['open_fraction']:.2f}, largest cage cluster "
              f"{p['largest_cage_cluster']}/{g.n_cages}, rank {p['percolation_rank']}")
    print("\nPASS")
