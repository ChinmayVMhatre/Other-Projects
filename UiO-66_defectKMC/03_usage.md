# Usage guide

## Dependencies

- Tests: `numpy`.
- `uio66_sites` structure I/O: `numpy`, `pymatgen`.
- `mofun_sites` and the real-structure workflow: `numpy`, `pymatgen`, `mofun`
  (`pip install mofun`, Python > 3.8).
- The graph-space kMC dynamics need only `numpy`; writing capped CIFs needs
  `pymatgen` (and `mofun` if you use the MOFUN path for site finding).

## Running the tests

No CIF or MOFUN required. Run from inside the folder that holds the modules (the
files import each other by module name, so run from there or add it to
`PYTHONPATH`):

```
python test_uio66_sites.py
python test_site_graph_equivalence.py
python test_graph_kmc.py
```

`test_graph_kmc.py` ends with `ALL INVARIANTS HOLD`. It checks defect
conservation, the connectivity floor, coordination-sphere-only swaps, clustering
vs control, and the latent → geometry → latent round trip.

## Real-structure workflow

Inputs:

- `pristine.cif` — pristine supercell, P1.
- defective CIF (e.g. `random_25.cif`) — formate-capped, same supercell, P1.
- `linker.cif` / `uio66-linker.cml` — search pattern: one intact BDC linker.
- `uio66-linker-defective.cml` — replace pattern: same coordinate frame, ring
  removed, formate caps in place (MOFUN's UiO-66 defect example patterns).

### Step 0 — check your MOFUN signature (once)

```python
import mofun_sites
print(mofun_sites.mofun_signature())
```

This tells you the installed parameter names, whether the tolerance is relative,
and whether quaternions are returned. See `04_mofun_integration.md`.

### Step 1 — validate parsing (gate)

```
python mofun_sites.py pristine.cif random_25.cif \
       linker.cif uio66-linker-defective.cml 0.05
```

`self_check()` runs: site count (6 per Zr6 node), agreement between the
intact-pattern read and covalent fragmentation (with ~2 formate groups per
missing linker), defect fraction, and a full geometry round trip. It must print
`PASS`. Common `FAIL` causes: wrong tolerance (last argument), or the two CIFs
not being the same supercell. Do not run kMC until this passes.

### Step 2 — run the kMC

```python
from uio66_graph_kmc import from_mofun, run, GraphKMCParams

p = GraphKMCParams(
    temperature=700, rate_model="environment",
    dE_per_defect_neighbour=-0.04,
    enable_diffusion=True, Ea_diff_closed=1.6, beta_open=0.20,
    open_window_threshold=2,
    stop_on_percolation=True, stop_metric="largest_open_fraction",
    stop_value=0.5,
    n_events=400000, snapshot_every=20000, outdir="graph_kmc_out")

kmc, writer, report = from_mofun("random_25.cif", "pristine.cif",
                                 "linker.cif", "uio66-linker-defective.cml",
                                 p=p, cache="sites.npz")
print(report)                 # confirm formate_per_defect == 2.0
run(kmc, write_snapshot=writer)
```

Outputs land in `outdir`: `trajectory.csv` (one row per snapshot) and
`snapshot_*.cif` (capped structures). `cache="sites.npz"` stores the MOFUN
pattern search so it runs once, not per snapshot.

Temperature sweep: repeat step 2 with different `temperature` and `outdir`,
then read the rearrangement timescale from each `trajectory.csv`.

Control run: set `dE_per_defect_neighbour=0.0` and/or `enable_diffusion=False`
to confirm any observed correlation comes from the intended term.

## trajectory.csv columns

`event, time_s, n_defects, n_clusters, largest_cluster, percolation_rank,
min_node_degree, mean_node_degree, mean_defect_neighbours, open_fraction,
largest_open_cluster, largest_open_fraction`

- `percolation_rank` — defect-adjacency spanning dimensionality (0/1/2/3),
  a proxy.
- `open_fraction` — fraction of triangular windows with ≥ `open_window_threshold`
  missing linkers.
- `largest_open_fraction` — fraction of all windows in the largest connected
  open-window cluster (the correlation order parameter to watch).

## Parameter reference (GraphKMCParams)

| Parameter | Default | Meaning |
|---|---|---|
| `Ea_gate` | 1.20 eV | Reaction/desorption gate barrier. Placeholder. |
| `nu` | 1e13 /s | Reaction attempt frequency. |
| `temperature` | 500 K | Temperature. |
| `Ea_floor` | 0.05 eV | Minimum reaction barrier. |
| `rate_model` | "environment" | "environment" (kinetic bias) or "detailed_balance" (Ising + `anneal()`). |
| `dE_per_defect_neighbour` | -0.02 eV | environment model: barrier change per defective neighbour at destination. Keep magnitude small. |
| `interaction_J` | 0.0 eV | detailed_balance model: >0 favours clustering. |
| `w_intact`, `w_defect` | 2, 1 | Discrete edge weights (`w_defect < w_intact`). Analysis only. |
| `enable_diffusion` | True | Add acid-diffusion series resistance. |
| `nu_diff` | 1e13 /s | Diffusion attempt frequency. |
| `Ea_diff_closed` | 1.50 eV | Diffusion barrier in a closed pore. Set ≥ `Ea_gate`. |
| `beta_open` | 0.15 eV | Barrier lowered per extra missing linker at the shared node. |
| `Ea_diff_floor` | 0.30 eV | Open-pore diffusion barrier floor. |
| `min_node_degree` | 2 | Minimum intact edges per node. |
| `swap_shell` | "shares_sbu" | Coordination sphere: "shares_sbu" (22) or "nearest" (8). |
| `open_window_threshold` | 2 | Missing linkers for a window to count as open. |
| `stop_on_percolation` | False | Halt when the target metric is reached. |
| `stop_metric` | "defect_rank" | "defect_rank" or "largest_open_fraction". |
| `stop_value` | 3.0 | Target: rank ≥ value, or largest_open_fraction ≥ value. |
| `n_events` | 200000 | Max events. |
| `snapshot_every` | 20000 | Snapshot / log interval. |
| `seed` | 0 | RNG seed. |
| `outdir` | "graph_kmc_out" | Output directory. |
