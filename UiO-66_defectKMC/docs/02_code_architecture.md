# Code architecture

Three modules, bottom to top. Higher modules import lower ones.

```
uio66_sites.py   (foundation: graph, occupancy, percolation, windows, I/O)
      ^
mofun_sites.py   (MOFUN-backed site finding + capping)
      ^
uio66_graph_kmc.py (kMC engine)
```

## uio66_sites.py

Pure numpy for all graph logic; pymatgen only for CIF read/write, so the graph
code can be tested without pymatgen.

Key pieces:

- `Frame` — a minimal periodic structure container (species, fractional coords,
  cell). `read_frame` / `write_frame` wrap pymatgen `Structure` I/O.
- `pbc_pairs`, `pairs_for_shell` — periodic neighbour search; the second grows
  the radius until every item has more than `k` neighbours (used for shell
  detection).
- `snap_frac` — rounds fractional coordinates before wrapping, so values like
  0.9999997 wrap to 0.0 rather than 1.0. Both site-graph constructors must
  normalise identically or their site labels will not line up.
- `find_nodes` — clusters Zr atoms into Zr6 nodes, PBC-safe centroid.
- `SiteGraph` — the linker-site graph: `node_frac`, `site_frac`, `site_nodes`
  (endpoint node indices per edge), `site_node_shift`, `neighbours` /
  `nbr_offsets` (hop shell), `node_sites` (node → incident edges).
- `build_site_graph` — from node positions: enumerate node-net edges → linker
  sites; attach a hop shell (`"nearest"` or `"shares_sbu"`).
- `build_site_graph_from_matches` — same graph but from MOFUN match positions
  (splits each match along its long axis, assigns ends to nearest node images).
  Proven equivalent to `build_site_graph` on the fixture.
- Fragment tools: `fragments`, `classify` (bdc / formate / inorganic),
  `fragment_centroid`, `inventory`.
- `read_occupancy` — matches BDC linkers onto sites → occupancy; counts formate
  per defect.
- Geometry library: `GeometryLibrary`, `build_library`, `_formate_from_bdc`
  (turns a BDC into its two formate caps by keeping the carboxylate groups and
  adding an H where the ring attached), `compose` (assembles the capped
  structure for any occupancy).
- Percolation: `periodic_components` (reusable image-aware BFS with wrap-vector
  rank), `defect_clusters`, `observables`.
- Windows: `WindowGraph`, `build_window_graph` (enumerate triangular faces),
  `WindowGraph.openness` (open-window fraction and clusters). No window spanning
  rank is reported.
- `StructureLattice` — a lattice adapter exposing the interface the original kMC
  loop expected (`occ`, `neighbours`, `defect_sites`, `n_defect_neighbours`,
  `write_cif`, `observables`).

## mofun_sites.py

Uses MOFUN for the chemistry; keeps `uio66_sites` for the topology MOFUN does
not provide. Robust to MOFUN version drift (see `04_mofun_integration.md`).

Key pieces:

- `_mofun` — quarantined imports so a missing install fails legibly.
- `load_atoms` — `Atoms.load` with the P1 requirement spelled out.
- `mofun_signature` — reports the installed `find_pattern_in_structure`
  signature: parameter list, which argument is the tolerance and whether it is
  relative, which is the positions/quaternions flag.
- `find_sites` — calls `find_pattern_in_structure` passing only the arguments
  the installed signature declares; classifies the return by type (indices are a
  2-D integer array, positions are (M, k, 3) floats, quats are scipy `Rotation`
  objects), regardless of order or count. Derives positions from indices if
  needed; fills quaternions with identity if absent.
- `replace_at_matches` — MOFUN's replacement logic with the random draw replaced
  by an explicit site list (kMC needs to replace *chosen* sites, not a random
  subset). Refuses to run on identity quaternions or missing indices.
- `MofunSites` — pristine structure + its linker sites + topology, built once.
  `from_pristine` / `from_cache` / `save_cache`. `has_real_quats` decides the
  write path. `occupancy_from` reads a defective structure (intact-linker match,
  then a covalent-fragmentation cross-check). `build_geometry_library` +
  `write` cap snapshots via `uio66_sites` when quaternions are unavailable.
- `MofunLattice` — `StructureLattice` whose snapshots are written by MOFUN.
- `load_initial_state` — one-call setup returning `(sites, occ, report)`.
- `self_check` — end-to-end validation to run before any production kMC.

## uio66_graph_kmc.py

The kMC engine. Imports `SiteGraph`, `observables`, `build_window_graph` etc.
from `uio66_sites`.

Key pieces:

- `GraphKMCParams` — all parameters (kinetics, rate model, discrete weights,
  diffusion, constraints, window threshold, stopping, run control). Validates on
  construction and warns about strong biases and model/diffusion conflicts.
- `build_swap_candidates` — per-edge coordination-sphere candidate lists from
  `node_sites`; asserts every candidate shares a node.
- `GraphKMC` — the engine:
  - maintains `state`, `intact_degree` (per node), `total_degree`.
  - `enumerate_events` — all allowed `(i → j)` with rates (constraints applied
    here; diffusion series resistance applied here).
  - `step` — BKL selection and execution, incremental degree update, time
    advance.
  - `edge_weights`, `node_weight` — discrete weight outputs.
  - `windows` (lazy `WindowGraph`), `observables` — defect and window metrics.
  - `to_structure` — write the current state via a supplied writer callback.
- `random_valid_occupancy` — a constraint-respecting random start (for tests /
  synthetic runs).
- `run` — the driver: logs `trajectory.csv`, writes snapshot CIFs, honours
  `stop_on_percolation`.
- `anneal` — temperature schedule driver (used with `detailed_balance`).
- `from_mofun` — assemble a ready-to-run `GraphKMC` plus a MOFUN write callback.

## Data flow (real workflow)

```
pristine.cif  --MOFUN find_pattern--> match indices/positions
              --> node net (uio66_sites.find_nodes)
              --> SiteGraph (build_site_graph_from_matches)
              --> GeometryLibrary (per-site linker + formate templates)

defective.cif --MOFUN intact match--> occupancy (occ)
              --covalent fragmentation cross-check--> formate/defect = 2

occ + SiteGraph + GraphKMCParams --> GraphKMC
      --> run() --> trajectory.csv + snapshot_*.cif (capped via geometry library)
```
