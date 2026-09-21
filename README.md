# UiO-66 defect kMC

Kinetic Monte Carlo of missing-linker defect rearrangement in defective
UiO-66. The framework is represented as a graph: Zr6O4(OH)4 SBUs are nodes
(fixed), linkers are edges (movable, discrete state). A defect relocates
between edges that share a node, under a minimum-connectivity constraint. The
run happens entirely on integer arrays; structures are written out only at
snapshots.

Physical basis: Mhatre et al., J. Phys. Chem. C 2024, 128(32), 13577-13587
(DOI: 10.1021/acs.jpcc.4c02982). The paper provides the structural facts
(sites, capping, defect fraction). It contains no kinetics, so barriers must
come from your own DFT/CI-NEB or literature.

## Files

Modules (bottom to top):

- `uio66_sites.py` — builds the Zr6-node / linker-edge graph from a CIF, reads
  defect occupancy, runs the periodic percolation test, writes structures.
  Uses numpy + pymatgen. No MOFUN.
- `mofun_sites.py` — same job done with MOFUN (the paper's tooling): site
  finding via `find_pattern_in_structure`, formate capping via the defect
  replacement pattern, plus `replace_at_matches()` for the site-selective
  replacement MOFUN does not expose. Depends on `uio66_sites`.
- `uio66_graph_kmc.py` — the kMC engine: fixed nodes, discrete-state edges,
  coordination-sphere swaps, connectivity floor. Depends on `uio66_sites`;
  uses `mofun_sites` only when writing capped CIFs.

Tests (synthetic fixtures, numpy only):

- `test_uio66_sites.py` — graph build, occupancy read, geometry round trip.
- `test_site_graph_equivalence.py` — confirms the MOFUN-match graph and the
  edge-enumerated graph describe the same lattice.
- `test_graph_kmc.py` — kMC invariants (conservation, connectivity floor,
  coordination-sphere-only swaps, clustering vs control, round trip).

## Dependencies

- Tests: `numpy`
- `uio66_sites` structure I/O: `numpy`, `pymatgen`
- `mofun_sites` and real-structure workflow: `numpy`, `pymatgen`, `mofun`
  (`pip install mofun`, requires Python > 3.8)

## Running the tests

No CIF or MOFUN required. Run from inside this folder (the files import each
other by module name):

```
pip install numpy
python test_uio66_sites.py
python test_site_graph_equivalence.py
python test_graph_kmc.py
```

Order is not strict; each test is standalone. `test_graph_kmc.py` should end
with `ALL INVARIANTS HOLD`.

## Running on real structures

Input files:

- `uio66_pristine.cif` — pristine supercell, P1
- `uio66_random25.cif` — defective (formate-capped) supercell, same cell as
  the pristine one, P1
- `uio66-linker.cml` — search pattern: one intact BDC linker
- `uio66-linker-defective.cml` — replace pattern: same coordinate frame, ring
  removed, formate caps in place

The two `.cml` files are from MOFUN's own UiO-66 defect example
(`docs/examples/` in the WilmerLab/mofun repo). Both CIFs must be P1 (convert
first if needed) and describe the same supercell.

Step 1 — validate parsing. Do this before running any kMC:

```
pip install numpy pymatgen mofun
python mofun_sites.py uio66_pristine.cif uio66_random25.cif \
       uio66-linker.cml uio66-linker-defective.cml 0.05
```

This runs `self_check()`: site count (6 per Zr6 node), intact/capped match
partition, defect fraction, and a full geometry round trip. It must print
`PASS`. If it prints `FAIL`, the most common causes are the wrong `atol` (last
argument; see below) or the two CIFs not being the same supercell. Do not
proceed to step 2 until this passes.

Step 2 — run the kMC:

```
python -c "
from uio66_graph_kmc import from_mofun, run, GraphKMCParams
p = GraphKMCParams(temperature=600.0,
                   dE_per_defect_neighbour=-0.02,
                   swap_shell='shares_sbu',
                   n_events=200000,
                   snapshot_every=20000,
                   outdir='graph_kmc_out')
kmc, writer, report = from_mofun('uio66_random25.cif', 'uio66_pristine.cif',
                                 'uio66-linker.cml', 'uio66-linker-defective.cml',
                                 p=p, cache='sites.npz')
print(report)  # confirm formate_per_defect is 2.0
run(kmc, write_snapshot=writer)
"
```

Outputs go to `outdir`: `trajectory.csv` (one row per snapshot) and
`snapshot_*.cif` (capped structures). The `cache='sites.npz'` argument stores
the MOFUN pattern search so it runs once, not per snapshot.

Temperature sweep: repeat step 2 with different `temperature` and `outdir`
values, then read the rearrangement timescale from each `trajectory.csv`.

Control run (no clustering bias, pure constrained random walk): set
`dE_per_defect_neighbour=0.0`. Run this first to confirm any observed
clustering comes from the environment term.

## Key parameters (GraphKMCParams)

- `Ea_gate` — gate barrier in eV (formate desorption / rearrangement).
  Placeholder default 1.20. Replace with your own value.
- `nu` — attempt frequency, 1/s. Default 1e13.
- `temperature` — K.
- `dE_per_defect_neighbour` — eV; barrier change per defective neighbour at the
  destination. Negative favours clustering. Default -0.02. Keep the magnitude
  small: with `swap_shell='shares_sbu'` there are up to 22 neighbours, so a
  large value drives barriers into the floor. The code warns if the value is
  large relative to `Ea_gate`.
- `Ea_floor` — minimum barrier, eV. Default 0.05.
- `w_intact`, `w_defect` — discrete edge weights (integers). Must satisfy
  `w_defect < w_intact`. Weight is for analysis; it does not affect the
  dynamics or the connectivity count.
- `min_node_degree` — minimum intact edges per node. Default 2.
- `swap_shell` — `'shares_sbu'` (all edges sharing a node, 22 in bulk) or
  `'nearest'` (8 geometrically closest). Both keep swaps within one
  coordination sphere.
- `n_events`, `snapshot_every`, `seed`, `outdir` — run control.

## trajectory.csv columns

`event, time_s, n_defects, n_clusters, largest_cluster, percolation_rank,
min_node_degree, mean_node_degree, mean_defect_neighbours`

`percolation_rank` is the dimensionality of defect-network percolation
(0 = none, 1/2/3 = wire/sheet/network), from the periodic image test.

## Known limitations

- Barriers (`Ea_gate`, `dE_per_defect_neighbour`) are placeholders. Replace
  before any quantitative claim.
- `percolation_rank` is computed on defect-edge adjacency. This is not the same
  criterion as the paper's guest-diffusion percolation (connected low-barrier
  windows). If you report a percolation transition, reconcile the two
  definitions or define the order parameter on windows (linker triples).
- `mofun_sites.py` is written against the published MOFUN 1.0 API and source
  but was not executed during development (MOFUN was not installed). Run
  `self_check()` on your own files before trusting it. `uio66_sites.py` and
  `uio66_graph_kmc.py` are tested on synthetic fixtures.
- The MOFUN replacement wraps inserted atoms assuming an orthorhombic cell.
  UiO-66's conventional cell is cubic, so this is fine; `replace_at_matches`
  raises on a triclinic cell rather than producing wrong coordinates.
- `atol` (MOFUN position tolerance) defaults to 0.05 A, which only matches an
  idealized structure. On DFT/MD-relaxed structures, raise it until the match
  count is correct (24 linkers per conventional cell), then stop.

## Scope

This models the kinetic feasibility and structural signature of
capping-mediated defect rearrangement: whether reorganization can occur, on
what timescale, with what structural change. It is not a claim about how
experimentally percolating UiO-66 is synthesized.
