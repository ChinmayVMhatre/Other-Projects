# Design decisions, caveats, and next steps

## Design decisions

- **Nodes fixed, edges movable.** SBUs never change; only linker states move.
  This matches the physics (SBUs are rigid scaffolding) and keeps the state a
  single integer array.
- **Latent-space kMC.** The run touches only integer arrays and fixed node
  coordinates. Atomic structures are written only at snapshots, so the expensive
  MOFUN pattern search runs once (cached) and each snapshot is a cheap
  replacement.
- **Constraints at enumeration, not rejection.** Forbidden moves (crossing the
  cell, or dropping a node below the connectivity floor) are never offered, so
  BKL stays rejection-free and exact.
- **Coordination-sphere swaps.** A relocation is a single-SBU rearrangement, so
  source and destination must share a node. Two shells are available:
  `shares_sbu` (22) and `nearest` (8).
- **Diffusion as a series resistance.** Rather than adding a new event type,
  each relocation combines a reaction rate and a diffusion rate in series. This
  keeps the event structure unchanged while making mass transfer able to limit
  the rate. The diffusion barrier depends on local pore openness, giving a
  feedback that favours channels.
- **MOFUN for chemistry, `uio66_sites` for topology.** The paper's tooling
  defines sites and caps; the topology (which linkers share an SBU, the
  percolation test, the windows) is computed independently and was shown to
  agree with the edge-enumerated graph.

## Honest caveats

- **Barriers are placeholders.** `Ea_gate`, `dE_per_defect_neighbour`,
  `interaction_J`, `Ea_diff_closed`, `beta_open` are not fitted to data. Replace
  before any quantitative claim.
- **The percolation metric is a proxy.** Defect-adjacency spanning rank is
  rigorous but saturates near 25% defects, so "random → percolating" is nearly
  trivial on it. The open-window order parameter discriminates correlation much
  better, but it is not the guest-diffusion percolation criterion either.
- **Rigorous guest percolation is not implemented.** Guest diffusion percolates
  through cages, not through "windows sharing a linker". Attempts to compute a
  window spanning rank did not validate (a fully open framework did not return
  the expected 3-D rank), so no window spanning rank is reported. The correct
  object is the cage graph.
- **Clustering is not spanning.** A strong clustering bias yields a compact
  defect blob, which need not span the cell at moderate defect fraction. Whether
  percolation is reached depends on defect fraction, temperature, and bias
  strength.
- **`mofun_sites` is exercised against a simulated build.** The MOFUN-dependent
  paths were tested against a mock with the installed signature, not a full live
  MOFUN run in the development environment. `self_check()` on real files is the
  gate.

## Development log (issues found and fixed)

1. **Stale test import.** A test imported the fixture as `test_fixture`, but in
   the delivered layout it is `test_uio66_sites`. Fixed the import; both test
   files now import `build` from `test_uio66_sites`. Reminder: run tests from
   the folder that holds the modules.
2. **`atol` TypeError.** The installed `find_pattern_in_structure` did not accept
   `atol`. Replaced hard-wired keywords with signature introspection that passes
   only declared arguments; added `mofun_signature()`.
3. **Return-shape unpacking.** The installed build returned 2 values, not 3, so a
   fixed `a, b, c = out` failed. Replaced with a type-based classifier that
   handles any order and count and derives missing pieces.
4. **Relative tolerance.** The tolerance argument was `rel_tol` (relative), which
   the first alias list missed. Added relative aliases and a note that the value
   is relative, not absolute Angstroms.
5. **No quaternions.** The build returns `(indices, positions)` with no
   quaternions, so MOFUN replacement cannot orient caps. Added a geometry-library
   write path that caps snapshots from the returned indices via `uio66_sites`
   (needs pymatgen); `replace_at_matches` refuses identity quaternions / missing
   indices.
6. **False `FAIL` on the cross-check.** The capped two-formate pattern (~11 Å
   span, void between groups) matched almost nothing at a tight relative
   tolerance, causing a false FAIL although the occupancy was correct. Replaced
   the capped-pattern cross-check with covalent fragmentation; hardened the
   empty-match path.
7. **Fractional-coordinate wrapping.** Values like 0.9999997 rounded to 1.0
   instead of wrapping to 0.0, so the two site-graph constructors disagreed on
   labels. Fixed by `snap_frac` (round before wrap) in both.
8. **Invalid window spanning rank.** A fully open framework returned rank 1, not
   3, revealing that "windows sharing a linker" is not the guest-diffusion graph.
   Dropped the window spanning rank; kept open-window fraction and cluster sizes,
   which are correct and meaningful.

## Validation summary

- Graph on ideal fcu: 32 nodes (coordination 12), 192 sites (6/node), nearest
  shell 8 @ 7.32 Å, shares_sbu shell 22, 192 triangular windows.
- MOFUN-match graph equals the edge-enumerated graph: same site positions, same
  adjacency, same percolation results at 15/25/40% defects.
- Occupancy round trip exact; formate/defect = 2.0.
- kMC invariants hold over thousands of moves: defect conservation, connectivity
  floor, coordination-sphere-only swaps, discrete weights.
- Diffusion feedback isolated: diffusion-only raises open-window fraction and
  clustering above a pure random walk; compounds with a clustering bias.
- Percolation-based stopping fires at the target; detailed_balance + diffusion
  warning prints.

## Recommended next steps

1. **Cage graph for guest percolation.** Enumerate cages (octahedral and
   tetrahedral) as nodes and windows as the openings between them. Mark windows
   open by missing-linker count; test percolation of the cage graph through open
   windows. This gives the paper-faithful order parameter and would let you claim
   percolation rigorously rather than via the correlation proxy.
2. **Real barriers.** Replace the placeholder energetics with DFT/CI-NEB values
   for formate desorption/adsorption, defect-pair interaction, and acid
   diffusion in open vs closed pores.
3. **Parameter sweeps.** Vary defect fraction, temperature, and bias strength to
   map where a spanning/percolating structure actually forms, and extract the
   rearrangement timescale from `trajectory.csv`.
4. **Optional: explicit mobile-acid species.** Instead of the series-resistance
   approximation, track acid molecules as walkers on the pore network with
   explicit desorption → diffusion → adsorption events. More faithful, more code.
