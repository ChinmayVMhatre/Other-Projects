# Physical and mathematical model

## Graph representation

- **Nodes**: Zr6O4(OH)4 SBUs. Fixed positions, no state. Found by clustering Zr
  atoms within a Zr–Zr cutoff (~4.2 Å; Zr–Zr inside one cluster is ~3.5 Å, the
  next cluster is ~10 Å away).
- **Edges (linker sites)**: one per edge of the node net. Each edge has a
  discrete state:
  - `INTACT` (1): a BDC linker C6H4(COO)2 bridges the two end nodes.
  - `DEFECTIVE` (0): no bridging linker; each of the two end nodes carries a
    formate cap HCOO.
- **Discrete edge weight**: `w_intact` and `w_defect` (integers, `w_defect <
  w_intact`). Weight is a label for downstream analysis (e.g. a weighted graph);
  it does not drive the dynamics and is separate from the connectivity count.

Definition of "defective" used throughout: the presence of HCOO (formate) caps
instead of a C6H4(COO)2 (BDC) linker. One missing linker corresponds to two
formate groups (one on each end node). The **formate-per-defect = 2** ratio is
the primary consistency check when reading a structure.

## Numbers validated on an ideal fcu fixture

Built as a 2×2×2 conventional cell, a = 20.7 Å:

| Quantity | Value |
|---|---|
| Zr6 nodes | 32 |
| Node coordination | 12 |
| Node–node distance | 14.64 Å |
| Linker sites | 192 (= 6 per node = 24 per conventional cell) |
| Nearest hop shell | 8 sites at 7.32 Å |
| Shares-an-SBU shell | 22 sites (8 @ 7.32, 4 @ 10.35, 8 @ 12.68, 2 @ 14.64 Å) |
| Triangular windows | 192 |

Trap worth noting: a purely geometric 15 Å cutoff picks up **42** neighbours,
not 22. Many edge midpoints fall inside that radius without sharing a node. The
coordination sphere must be built from node incidence, not a distance cutoff.

## State and event

- **State**: an integer array `occ` (or `state`) of length = number of edges,
  entries in {0, 1}.
- **Event**: a defect relocates from a defective edge `i` to an intact edge `j`:
  - `state[i]`: 0 → 1 (linker heals, formate desorbs)
  - `state[j]`: 1 → 0 (linker removed, formate caps appear)
- **Conservation**: the number of defects is constant (fixed defect fraction).

## Constraints (enforced at event enumeration, so BKL stays rejection-free)

1. **Coordination sphere**: `i` and `j` must share a Zr6 node — the relocation
   is a single-SBU rearrangement, never a swap between distant linkers.
   - `swap_shell = "shares_sbu"`: every edge on either of `i`'s two nodes (22 in
     bulk).
   - `swap_shell = "nearest"`: the 8 geometrically closest of those.
2. **Minimum connectivity**: no node may fall below `min_node_degree` intact
   edges (default 2). For an `i → j` move, `i` heals (its nodes gain) and `j`
   defects (its non-shared node loses one); only that non-shared node of `j` can
   violate the floor, so it must have degree > floor before the move. The check
   uses an incrementally maintained per-node intact-degree array.

The graph is assumed simple (no two edges between the same node pair), which
UiO-66 fcu satisfies; the code asserts this because the degree bookkeeping
relies on it.

## Rate

Base Arrhenius form: `k = nu * exp(-Ea / (kB T))`.

Two reaction-rate models for the barrier `Ea`:

- **`environment`** (kinetic bias, not an equilibrium sampler):
  `Ea = Ea_gate + dE_per_defect_neighbour * n_nb`, where `n_nb` is the number of
  defective edges around the destination `j` (excluding the mover). Negative
  `dE_per_defect_neighbour` lowers the barrier where defects already cluster.
- **`detailed_balance`** (Ising-like): energy `E = -interaction_J *
  (defect–defect adjacencies)`, and `Ea = Ea_gate + dE/2` so forward/back rates
  satisfy `k_ij / k_ji = exp(-dE / kT)`. Use with `anneal()` to relax a random
  configuration toward its clustered equilibrium.

Caution with `environment`: at the shares_sbu shell there are up to 22
neighbours, so a large `|dE_per_defect_neighbour|` can rival `Ea_gate` and drive
barriers into the floor. Keep the magnitude small; the code warns otherwise.

## Acid diffusion (mass transfer)

Each relocation is modelled as two steps in series: the cap
desorption/adsorption (reaction) and the diffusion of the freed acid molecule
through the pore to the new site.

    1 / k_relocation = 1 / k_reaction + 1 / k_diffusion

- `k_reaction` is the Arrhenius rate above (with the chosen `rate_model`). This
  is the adsorption/desorption term; it was previously the only barrier and is
  now made explicit.
- `k_diffusion = nu_diff * exp(-Ea_diff / kT)`, with
  `Ea_diff = max(Ea_diff_closed - beta_open * n_open, Ea_diff_floor)`.
  `n_open` is the number of missing linkers around the shared node the acid
  crosses (≥ 1, since the mover itself is one).

Behaviour: in a closed pore `Ea_diff` is high (set `Ea_diff_closed ≥ Ea_gate`),
so diffusion is rate-limiting and the relocation is suppressed. As the pore
opens (more local defects), `Ea_diff` drops below `Ea_gate` and the relocation
becomes reaction-limited. The net effect is that acid moves easily through
already-open channels and is throttled in closed pores, biasing the system
toward extended open channels rather than compact blobs.

Isolated-effect test (no clustering bias, `dE = 0`): the open-window fraction
rose from 0.125 (diffusion off, pure random walk) to 0.188 (diffusion on), and
the largest open-window cluster from 0.042 to 0.089 of all windows. With a
clustering bias added, both rise further (0.214). The diffusion feedback is a
genuine, if modest, driver toward correlation.

Note: turning diffusion on with `rate_model = "detailed_balance"` breaks
detailed balance (mass transfer is a non-equilibrium kinetic effect), so
`anneal()` no longer samples a clean equilibrium. The code warns. Use diffusion
with the `environment` model for a kinetic trajectory, or turn it off for
equilibrium sampling.

## Percolation and order parameters

- **Defect-adjacency percolation rank** (rigorous, image-aware). BFS the defect
  clusters carrying accumulated periodic image offsets; when two paths reach the
  same site with different offsets, the difference is a lattice vector and the
  cluster touches its own image. The rank of the set of such wrap vectors is the
  dimensionality (0 none, 1 wire, 2 sheet, 3 network). Verified: single defect →
  0, all defects → 3. This is a **proxy**: it saturates near 25% defects (random
  25% is already rank ≥ 1), so it is a weak discriminator of correlation.

- **Open-window order parameter** (correlation measure). Triangular windows are
  the faces of the node net (3 nodes mutually bridged by 3 linkers). A window is
  "open" when ≥ `open_window_threshold` (default 2) of its 3 linkers are missing.
  Reported: `open_fraction` (fraction of windows open) and `largest_open_cluster`
  / `largest_open_fraction` (largest connected set of open windows, adjacency =
  sharing a linker). This separates correlated from random arrangements well:
  random 25% is usually non-percolating on the window metric even though it is
  rank ≥ 1 on defect adjacency.

- **Not implemented — rigorous guest percolation.** Guest diffusion percolates
  through **cages** (a guest exits a cage via any of its windows). "Windows
  sharing a linker" is not the cage graph, so no window spanning rank is
  reported. Building the cage graph (cages as nodes, windows as edges between
  them) is the recommended next step for a true guest-percolation criterion.

## Important distinction: clustering is not spanning

A strong clustering bias drives all defects into a single compact blob (the
defect-adjacency `largest_cluster` becomes the full defect count). A compact blob
at moderate defect fraction need not span the periodic cell. The diffusion
feedback favours extended channels, which helps, but whether the final structure
percolates depends on defect fraction, temperature, and bias strength — these
must be swept, not assumed.
