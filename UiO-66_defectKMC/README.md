# uio66-defect-kmc

Kinetic Monte Carlo of missing-linker defects in UiO-66: from the acid
composition that sets the defect fraction, to the rearrangement of a random
defect structure into a 1-D percolating guest-transport pathway.

The framework is represented as a graph built from a pristine UiO-66 cell: the
Zr6O4(OH)4 nodes are fixed, and the 24 linker sites per conventional cell are
edges that can hold a di-carboxylic acid (intact linker) or a mono-carboxylic
acid (missing-linker defect).

## The two tasks

**Task 1 - attach / detach (defect fraction).**
Mono- and di-carboxylic acids compete for the linker sites. Each species has its
own binding energy `E_bind` (attach) and desorption energy `E_diss` (detach).
With adsorption strength `S = E_diss - E_bind` and bulk ratio di:mono = n:1, the
steady-state defect fraction is

    defect fraction = 1 - K_di/(1 + K_di + K_mono),   K = c * exp(S / kT)

- Equal energies for mono & di -> `1/(n+1)`, set by the ratio, temperature-independent.
- di binds more strongly -> fewer defects, and the fraction becomes temperature-dependent.

**Task 2 - rearrangement to a 1-D transport pathway.**
At a fixed defect fraction, defects and linkers swap sites (energy-neutral, an
unbiased random walk). "Percolation" means a 1-D uninterrupted pathway for a
guest molecule hopping cage -> cage through open (defective) windows. The run
logs the trajectory and reports whether / how the structure reaches a
cell-spanning pathway, measured by the pathway's displacement vector.

## Install

```bash
pip install -r requirements.txt     # numpy, scipy, networkx, ase
```

`pymatgen` and `mofun` are optional (alternative I/O and MOFUN-based capping).

## Quick start

```bash
# Task 1 + Task 2 together (equal energies: fraction set by ratio)
python uio66_kmc_minimal.py data/pristine-uio66_unit_cell.cif --ratio 10 --temp 500

# Task 1 with UNEQUAL energies: di binds more strongly -> fewer, T-dependent defects
python uio66_kmc_minimal.py data/pristine-uio66_unit_cell.cif --ratio 10 --temp 700 \
    --eb-di 0.9 --ed-di 1.35 --eb-mono 0.9 --ed-mono 1.20

# Rearrange a random structure to MATCH a known 1-D percolating structure
python uio66_kmc_match_1d.py data/pristine-uio66_unit_cell.cif data/6_miss_link.lmpdat

# Run a kMC seeded from a provided defective structure, saving snapshot CIFs
python run_kmc_from_structure.py data/random_1.cif \
    --save-structures snaps --template data/pristine-uio66_unit_cell.cif
```

Task-2 runs write `trajectory.csv` (per-snapshot observables) to the output dir.

## Layout

Library modules:

| file | role |
|---|---|
| `uio66_sites.py` | node/linker graph from a CIF, occupancy, percolation core, windows, geometry library |
| `cage_graph.py` | guest-transport cage graph (cages = nodes, windows = edges); the percolation object |
| `percolation_path.py` | 1-D pathway displacement metric (direction, magnitude, tortuosity) |
| `uio66_graph_kmc.py` | graph-space kMC engine, constraints, `random_valid_occupancy` |
| `snapshot_io.py` | compose occupancy -> atomic structure, write CIF (no pymatgen) |
| `mofun_sites.py` | optional MOFUN-based site finding & formate capping |

Command-line scripts:

| file | task |
|---|---|
| `uio66_kmc_minimal.py` | Task 1 (attach/detach) + Task 2 (rearrangement trajectory) |
| `uio66_kmc_match_1d.py` | rearrange a random structure to match a known 1-D percolating one |
| `run_kmc_from_structure.py` | kMC seeded from a provided defective CIF/LAMMPS file, with snapshots |

Tests (numpy/scipy only; no ASE needed):

```bash
python test_uio66_sites.py                               # synthetic graph build + round trip
python test_site_graph_equivalence.py                    # MOFUN-match graph == edge graph
python test_graph_kmc.py                                 # kMC invariants
python test_cage_graph.py data/pristine-uio66_unit_cell.cif   # cage graph counts + percolation extremes
```

`docs/` holds the deeper reference (physical model, architecture, usage,
MOFUN notes, design decisions and caveats).

## Notes and caveats

- The 1-D transport percolation uses a window-open criterion equivalent to a
  ~5 A guest (a window opens when >= 1 of its 3 linkers is missing). The cage
  graph is validated against known UiO-66 pore topology (4 octahedral + 8
  tetrahedral cages and 32 windows per conventional cell; all-open -> 3-D
  percolation, pristine -> none).
- At fixed defect fraction all arrangements are iso-energetic, so Task 2 is an
  unbiased random walk: near the percolation threshold it fluctuates in and out
  of a 1-D pathway rather than converging. Use `--stop-rank 1` to halt at the
  first spanning pathway.
- Binding/desorption energies are inputs; replace the defaults with DFT/CI-NEB
  values for quantitative work.
- `mofun_sites.py` is written against the published MOFUN API but exercised here
  only against a simulated build; validate with its `self_check()` on your files.

## Physical basis

Structural facts (linker sites, formate capping, defect fractions) follow
Mhatre et al., *J. Phys. Chem. C* 2023/2024. The lattice-kMC / graph approach is
in the spirit of the graphanol lattice Monte Carlo of Achar et al.,
*ACS Appl. Mater. Interfaces* 2023, 15, 25873.
