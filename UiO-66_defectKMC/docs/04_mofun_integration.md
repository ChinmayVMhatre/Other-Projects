# MOFUN integration and version drift

MOFUN (WilmerLab/mofun) is the paper's own tooling, so using it for site finding
and capping keeps the site definitions identical to how the structures were
made. Its PyPI release and its documentation drift, so the code adapts to
whatever signature is installed rather than assuming one.

## What MOFUN does and does not provide

- MOFUN provides: what a linker site is (`find_pattern_in_structure` on the BDC
  pattern) and what replaces it (the formate caps, via the defect replacement
  pattern).
- MOFUN does not provide: which linkers share an SBU (the topology). That comes
  from `uio66_sites` (`find_nodes` + `build_site_graph_from_matches`), which was
  proven to reproduce the edge-enumerated graph exactly on the fixture.
- MOFUN's `replace_pattern_in_structure` can only replace a *random* subset.
  kMC needs to replace *chosen* sites, so `replace_at_matches` reuses MOFUN's
  replacement logic with the random draw swapped for an explicit list.

## The installed build in this project

`mofun_signature()` returned:

```
{'params': ['structure', 'pattern', 'return_positions', 'rel_tol', 'verbose'],
 'tolerance_arg': 'rel_tol', 'tolerance_is_relative': True,
 'posquat_arg': 'return_positions'}
```

Two consequences:

1. The tolerance argument is `rel_tol`, a **relative** tolerance, not absolute
   Angstroms. The value passed as `atol` is routed to `rel_tol`; the code prints
   a note. Start around 0.01–0.05 and raise it if a relaxed structure returns too
   few matches.
2. `return_positions` returns `(indices, positions)` with **no quaternions**.
   MOFUN's own replacement cannot orient the caps without quaternions, so
   snapshots are written via the `uio66_sites` geometry library instead (it uses
   the returned indices to pull each linker's atoms from the pristine structure
   and derive the formate caps). This path needs pymatgen.

## How `find_sites` is robust to drift

- Reads the installed signature with `inspect`.
- Passes only the arguments that exist. The tolerance is matched against
  `atol`, `abs_tol`, `position_tolerance` (absolute) and `rel_tol`, `rtol`,
  `tol` (relative); the positions/quaternions flag against
  `return_positions_and_quats`, `return_positions`, `return_quats`,
  `return_positions_and_quaternions`.
- Classifies the return by **type**, not position: indices are a 2-D integer
  array, positions are an (M, k, 3) float array, quaternions are scipy
  `Rotation` objects. This handles 2- or 3-value returns in any order.
- Fills in what is missing: positions derived from indices; quaternions set to
  identity if absent (with a printed note); indices left `None` if absent.

`replace_at_matches` refuses to run on identity quaternions or missing indices,
so it never places caps at wrong orientations silently.

## Reading occupancy and the cross-check

`occupancy_from` reads the defective structure with the **intact-linker**
pattern: matched sites are occupied, the rest are defective. This is
authoritative.

The independent cross-check is **covalent fragmentation** (the same method the
pure `uio66_sites` path uses): the structure is broken into fragments, BDC
linkers and formate caps are counted, and the result is compared to the
intact-pattern occupancy. Agreement plus ~2 formate groups per defect confirms
the parse.

This replaced an earlier cross-check that searched for the two-formate composite
pattern. That pattern spans ~11 Å with a void between the two formate groups, so
at a tight relative tolerance it matched almost nothing and produced a false
`FAIL` even when the occupancy was correct. The fragmentation cross-check is
robust and does not depend on long-range pattern matching.

## Requirements and gotchas

- Both CIFs must be P1 and the same supercell; sites are matched by position.
- The pristine CIF must be genuinely pristine; the geometry library build raises
  if any site lacks a linker.
- Use `Atoms.save(path)` for snapshots, not `to_ase().write(path)` — the latter
  drops charges.
- The MOFUN replacement wraps inserted atoms assuming an orthorhombic cell.
  UiO-66's conventional cell is cubic, so this is fine; `replace_at_matches`
  raises on a triclinic cell rather than producing wrong coordinates.
- `find_pattern_in_structure` chooses randomly among symmetry-equivalent atom
  orderings, so `random` is seeded for reproducible snapshots. A stray debug
  print in the released function is silenced.
