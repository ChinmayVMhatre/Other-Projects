# UiO-66 defect kMC — project overview

## Objective

Take a defective UiO-66 structure (missing BDC linkers, formate-capped at the
defect sites), represent it as a graph, and run kinetic Monte Carlo (kMC) that
lets defects rearrange. The target end state is a *correlated / percolating*
defect arrangement of the kind discussed in the JPCC 2024 paper, reached from a
random starting arrangement.

Physical basis: Mhatre et al., *J. Phys. Chem. C* 2024, 128(32), 13577–13587
(DOI: 10.1021/acs.jpcc.4c02982). The paper supplies the structural facts
(linker sites, formate capping, defect fractions). It contains no kinetics, so
all barriers in the kMC are placeholders to be replaced with DFT/CI-NEB or
literature values.

## The graph model in one paragraph

The Zr6O4(OH)4 secondary building units (SBUs) are graph **nodes** and never
change. The linker slots are graph **edges**, each carrying a discrete state:
INTACT (a BDC linker bridges two nodes, occupancy 1) or DEFECTIVE (no bridging
linker; both nodes carry a formate cap, occupancy 0). One missing linker means
two formate groups — the 2:1 ratio is the consistency check used when reading a
structure. kMC runs entirely on integer arrays in this "latent" graph space;
atomic structures are written back only at snapshots.

## What was built

Three Python modules (bottom to top):

| Module | Role | Deps |
|---|---|---|
| `uio66_sites.py` | Node/edge graph from a CIF, occupancy reading, percolation test, triangular-window enumeration, structure I/O | numpy, pymatgen |
| `mofun_sites.py` | Same site finding and capping via MOFUN (the paper's tooling); robust to MOFUN version drift | numpy, pymatgen, mofun |
| `uio66_graph_kmc.py` | The kMC engine: fixed nodes, discrete-state edges, coordination-sphere swaps, connectivity floor, acid diffusion, percolation stopping | numpy, (mofun/pymatgen for writing CIFs) |

Three tests (synthetic fixtures, numpy only):
`test_uio66_sites.py`, `test_site_graph_equivalence.py`, `test_graph_kmc.py`.

Plus `README.md` at the top level.

## Status

Working and tested:
- Graph construction and site enumeration (validated on an ideal fcu fixture).
- Occupancy reading from a defective structure, with a covalent-fragmentation
  cross-check (formate-per-defect = 2).
- MOFUN integration, including a build that returns `(indices, positions)` with
  no quaternions and uses `rel_tol` (the user's installed build).
- Graph-space kMC: defect conservation, connectivity floor, coordination-sphere
  swaps, discrete edge weights, BKL stepping.
- Acid diffusion as a series mass-transfer resistance.
- Defect-adjacency percolation (rigorous, image-aware) and open-window
  correlation order parameter; percolation-based stopping.

Not implemented (documented, recommended next step):
- Rigorous guest-diffusion percolation. That needs the cage graph (cages as
  nodes, windows as the openings between them). The shipped percolation signals
  are a defect-adjacency spanning rank (a proxy) and an open-window correlation
  order parameter.

## Document map

- `00_overview.md` — this file.
- `01_physical_model.md` — definitions, validated numbers, events, rates,
  constraints, percolation metrics.
- `02_code_architecture.md` — module-by-module walkthrough and data flow.
- `03_usage.md` — dependencies, test order, real-structure workflow, parameter
  reference.
- `04_mofun_integration.md` — MOFUN version-drift handling and the
  quaternion-free write path.
- `05_decisions_caveats_next_steps.md` — design decisions, honest caveats, a log
  of the issues fixed during development, and recommended next steps.
