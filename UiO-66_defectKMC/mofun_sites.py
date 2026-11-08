"""
mofun_sites.py -- MOFUN-backed sites and capping for the UiO-66 defect kMC
==========================================================================

Uses the paper's own tooling (WilmerLab/mofun) for everything chemical, and
keeps `uio66_sites` only for the topology MOFUN doesn't provide.

    MOFUN owns:  what a linker site IS (find_pattern_in_structure on the BDC
                 pattern) and what replaces it (replace pattern = the two
                 formate caps, i.e. MOFUN examples' uio66-linker-defective.cml)
    we own:      which sites are adjacent, and periodic percolation

Why a wrapper is needed at all
------------------------------
`replace_pattern_in_structure` can only replace a RANDOM subset
(`replace_fraction` -> `random.sample`). kMC needs to replace a SPECIFIC set of
sites, the ones that are currently defective. `replace_at_matches()` below is
that function: the body is MOFUN's own replacement logic (quaternion placement,
unchanged-atom mapping, deferred deletion) with the random draw swapped for an
explicit site list. Everything chemical still comes from MOFUN.

Cost model
----------
`find_pattern_in_structure` is a distance-geometry search and is slow. It runs
ONCE on the pristine structure; the matches, positions and quaternions are
cached (and can be written to an .npz). Each kMC snapshot is then just a
replacement pass over the current defect list -- no re-searching.

Files needed
------------
    uio66_pristine.cif          P1 CIF of the pristine supercell
    uio66-linker.cml            search pattern: one BDC linker
    uio66-linker-defective.cml  replace pattern: same frame, ring removed,
                                formate caps in place
The two .cml patterns are the ones from MOFUN's own UiO-66 defect example
(docs/examples/ in the repo). The replace pattern MUST be in the same
coordinate system as the search pattern -- shared atoms need identical
coordinates or the inserted caps come out offset.

STATUS: written against the published MOFUN 1.0 API and source, but NOT
executed here (mofun was not installed in the environment where this was
written). Run `self_check()` on your own files before trusting it.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import random
from dataclasses import dataclass, field

import numpy as np

from uio66_sites import (SiteGraph, StructureLattice, build_site_graph_from_matches,
                         find_nodes, observables, defect_clusters, Frame, snap_frac)

__all__ = ["MofunSites", "MofunLattice", "replace_at_matches", "find_sites",
           "load_initial_state"]

# one-time diagnostic notes, keyed by topic
_NOTED: dict = {}


# ---------------------------------------------------------------------------
# 0. MOFUN entry points, quarantined so import errors are legible
# ---------------------------------------------------------------------------
def _mofun():
    try:
        from mofun import Atoms, replace_pattern_in_structure  # noqa: F401
        from mofun.mofun import find_pattern_in_structure
        from mofun.atoms import find_unchanged_atom_pairs
    except ImportError as exc:                      # pragma: no cover
        raise ImportError("MOFUN is required: pip install mofun") from exc
    return Atoms, find_pattern_in_structure, find_unchanged_atom_pairs


def load_atoms(path: str):
    """Atoms.load, with the P1 requirement spelled out.

    MOFUN's CIF reader ignores symmetry and rejects anything that declares a
    space group other than P1. Convert first if needed.
    """
    Atoms, _, _ = _mofun()
    try:
        return Atoms.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"MOFUN could not load {path}: {exc}\n"
            "CIFs must be P1 (Atoms.load_p1_cif ignores symmetry and raises "
            "otherwise); .cml and .lmpdat are also accepted."
        ) from exc


# parameter names MOFUN has used across versions for the position tolerance
_ABS_TOL_ALIASES = ("atol", "abs_tol", "position_tolerance")
_REL_TOL_ALIASES = ("rel_tol", "rtol", "tol")
_TOL_ALIASES = _ABS_TOL_ALIASES + _REL_TOL_ALIASES
# names for the flag that also returns positions (and sometimes quaternions)
_POSQUAT_ALIASES = ("return_positions_and_quats", "return_positions",
                    "return_quats", "return_positions_and_quaternions")


def mofun_signature(func=None) -> dict:
    """Report the installed find_pattern_in_structure signature.

    Run this once if a TypeError about an unexpected keyword shows up: the
    PyPI release and the docs drift, so the real parameter names vary by
    version. Returns {'params': [...], 'tolerance_arg': name-or-None,
    'tolerance_is_relative': bool, 'posquat_arg': name-or-None}.
    """
    if func is None:
        _, func, _ = _mofun()
    params = list(inspect.signature(func).parameters)
    tol = next((a for a in _TOL_ALIASES if a in params), None)
    posquat = next((a for a in _POSQUAT_ALIASES if a in params), None)
    return {"params": params, "tolerance_arg": tol,
            "tolerance_is_relative": tol in _REL_TOL_ALIASES,
            "posquat_arg": posquat}


def _build_kwargs(func, atol, verbose):
    """Keep only the kwargs the installed function actually declares."""
    params = inspect.signature(func).parameters
    kw = {}
    tol = next((a for a in _TOL_ALIASES if a in params), None)
    if tol is not None:
        kw[tol] = atol
        if tol in _REL_TOL_ALIASES and not _NOTED.get("tol"):
            _NOTED["tol"] = True
            print(f"  note: this MOFUN build's tolerance is '{tol}' (RELATIVE, "
                  f"not absolute Angstroms). The value you pass as atol={atol} "
                  "is used as a relative tolerance; start small (~0.01-0.05) "
                  "and raise it if a relaxed structure returns too few matches.")
    else:
        print("  ! this MOFUN build's find_pattern_in_structure takes no "
              f"tolerance argument (looked for {_TOL_ALIASES}). Matching will "
              "use its built-in default; if a relaxed structure returns no "
              "matches, that default is the reason. Run "
              "mofun_sites.mofun_signature() to see the real signature.")
    posquat = next((a for a in _POSQUAT_ALIASES if a in params), None)
    if posquat is not None:
        kw[posquat] = True
    if "verbose" in params:
        kw["verbose"] = verbose
    return kw, posquat


def find_sites(structure, search_pattern, atol: float = 0.05,
               seed: int = 0, quiet: bool = True, verbose: bool = False):
    """find_pattern_in_structure, returning (indices, positions, quats).

    Robust to MOFUN version drift: the tolerance and return-flag arguments are
    matched against the INSTALLED signature (see mofun_signature()), so a build
    that renamed or dropped `atol` no longer raises a TypeError -- it just
    falls back to that build's default tolerance, with a warning.

    `atol` is the absolute position tolerance in Angstroms. The 0.05 A default
    only matches an idealised structure; on anything relaxed by DFT or MD the
    linker geometry drifts and you get zero matches, so raise it until the
    match count is right, then stop. Too loose and the search starts matching
    non-linkers.

    MOFUN picks randomly among symmetry-equivalent atom orderings, so `random`
    is seeded for reproducible snapshots. The released function also has an
    unguarded debug print, silenced by `quiet`.
    """
    _, find_pattern_in_structure, _ = _mofun()
    kw, posquat = _build_kwargs(find_pattern_in_structure, atol, verbose)

    random.seed(seed)
    sink = io.StringIO() if quiet else None
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        out = find_pattern_in_structure(structure, search_pattern, **kw)

    return _normalize_find_output(out, structure, search_pattern)


def _is_positions(e) -> bool:
    try:
        a = np.asarray(e, dtype=float)
    except (ValueError, TypeError):
        return False
    return a.ndim == 3 and a.shape[-1] == 3


def _is_quats(e) -> bool:
    try:
        first = e[0]
    except (TypeError, IndexError, KeyError):
        return False
    from scipy.spatial.transform import Rotation
    return hasattr(first, "as_quat") or isinstance(first, Rotation)


def _is_indices(e) -> bool:
    try:
        a = np.asarray(e)
    except (ValueError, TypeError):
        return False
    return a.ndim == 2 and np.issubdtype(a.dtype, np.integer)


def _normalize_find_output(out, structure, search_pattern):
    """Turn whatever find_pattern_in_structure returned into (indices,
    positions, quats), regardless of how many items it returned or in what
    order. Missing pieces are derived (positions from indices) or filled with
    a sentinel (quats = identity, indices = None) that downstream code guards.
    """
    is_bundle = (isinstance(out, (tuple, list)) and 2 <= len(out) <= 3
                 and any(_is_positions(e) or _is_quats(e) for e in out))

    indices = positions = quats = None
    if not is_bundle:
        indices = list(out)                     # bare match-index list
    else:
        for e in out:
            if quats is None and _is_quats(e):
                quats = list(e)
            elif positions is None and _is_positions(e):
                positions = np.asarray(e, dtype=float)
            elif indices is None and _is_indices(e):
                indices = list(e)
        # a leftover unclassified element is the index list (ragged / tuples)
        if indices is None:
            for e in out:
                if not _is_quats(e) and not _is_positions(e):
                    indices = list(e)
                    break

    if positions is None:
        if indices is None:
            raise RuntimeError(
                "find_pattern_in_structure returned neither positions nor "
                "usable indices. Run mofun_sites.mofun_signature() and share "
                "the output.")
        positions, _ = _positions_and_quats_from_indices(
            structure, search_pattern, indices)
    if quats is None:
        from scipy.spatial.transform import Rotation
        quats = [Rotation.identity() for _ in range(len(positions))]

    n = len(positions)
    _find_output_note(indices, positions, quats)
    return indices, np.asarray(positions, dtype=float), quats


def _find_output_note(indices, positions, quats):
    if _NOTED.get("shape"):
        return
    _NOTED["shape"] = True
    from scipy.spatial.transform import Rotation
    have_idx = indices is not None
    have_q = quats and not all(
        isinstance(q, Rotation) and np.allclose(q.as_quat(), [0, 0, 0, 1])
        for q in quats)
    print(f"  find_pattern return: {len(positions)} matches, "
          f"indices={'yes' if have_idx else 'DERIVED'}, "
          f"quaternions={'yes' if have_q else 'IDENTITY (replacement disabled)'}")


def _positions_and_quats_from_indices(structure, search_pattern, match_indices):
    """Fallback for builds whose find_pattern_in_structure returns indices only.

    Positions come straight from the structure; the graph builder only needs
    positions, so the quaternion is set to identity. This is enough to read
    occupancy and build the topology. It is NOT enough for replace_at_matches,
    which needs the true rotation -- if you land here, upgrade MOFUN to a build
    whose find_pattern_in_structure supports returning quaternions.
    """
    from scipy.spatial.transform import Rotation
    pos = np.asarray(structure.positions, dtype=float)
    match_positions = np.array([pos[list(m)] for m in match_indices], dtype=float)
    quats = [Rotation.identity() for _ in match_indices]
    return match_positions, quats


# ---------------------------------------------------------------------------
# 1. The function MOFUN doesn't expose: replace at CHOSEN sites
# ---------------------------------------------------------------------------
def replace_at_matches(structure, search_pattern, replace_pattern,
                       match_indices, match_positions, quats,
                       sites, replace_all: bool = False,
                       ignore_double_delete: bool = False):
    """Replace `search_pattern` with `replace_pattern` at the chosen matches only.

    This is `replace_pattern_in_structure` with the `random.sample` draw
    replaced by an explicit `sites` list, and with the pattern search hoisted
    out so it can be done once and reused.

    Args:
        structure: the Atoms object to modify (a copy is returned).
        search_pattern, replace_pattern: Atoms, in a shared coordinate frame.
        match_indices, match_positions, quats: output of find_sites() run on
            THIS structure with THIS search pattern.
        sites: indices into match_indices -- the sites to replace.
        replace_all: replace even atoms whose position and element are unchanged.
        ignore_double_delete: allow two matches to delete the same atom. Leave
            False; it firing means two sites share atoms, which for a linker
            pattern means the pattern or the tolerance is wrong.

    Returns:
        Atoms: the modified structure.
    """
    _, _, find_unchanged_atom_pairs = _mofun()

    search_pattern = search_pattern.copy()
    replace_pattern = replace_pattern.copy()
    # both patterns get shifted so search atom 0 is at the origin; the match
    # positions are then absolute anchors for atom 0 of each match
    replace_pattern.translate(-search_pattern.positions[0])
    search_pattern.translate(-search_pattern.positions[0])

    sites = [int(s) for s in sites]
    if match_indices is None:
        raise RuntimeError(
            "this MOFUN build's find_pattern_in_structure did not return match "
            "indices, so replace_at_matches cannot know which atoms to remove. "
            "Write structures with the uio66_sites geometry library instead, "
            "or upgrade MOFUN (see mofun_signature()).")
    chosen_indices = [match_indices[s] for s in sites]
    chosen_positions = np.asarray(match_positions)[sites]
    chosen_quats = [quats[s] for s in sites]

    # the find_sites fallback path fills quats with identity when the installed
    # MOFUN cannot return them; placing caps with the wrong rotation would
    # corrupt the geometry silently, so refuse instead
    if chosen_quats and all(
            np.allclose(q.as_quat(), [0, 0, 0, 1]) for q in chosen_quats):
        raise RuntimeError(
            "all match quaternions are identity, which means this MOFUN build's "
            "find_pattern_in_structure did not return real rotations (see "
            "mofun_signature()). replace_at_matches needs them to orient the "
            "caps. Upgrade MOFUN to a build that supports "
            "return_positions_and_quats, or write structures with the "
            "uio66_sites geometry library instead of MOFUN.")

    new_structure = structure.copy()
    to_delete: set[int] = set()

    if len(replace_pattern) == 0:
        to_delete |= {int(i) for m in chosen_indices for i in m}
    else:
        replace2search = {k: v for (k, v) in
                          find_unchanged_atom_pairs(replace_pattern, search_pattern)}
        offsets = new_structure.extend_types(replace_pattern)

        if not _cell_is_orthorhombic(new_structure.cell):
            raise NotImplementedError(
                "MOFUN's replacement wraps inserted atoms with "
                "positions %= np.diag(cell), which is only correct for an "
                "orthorhombic cell. The UiO-66 conventional cell is cubic; if "
                "yours is triclinic, wrap in fractional coordinates instead.")

        for k, atom_positions in enumerate(chosen_positions):
            new_atoms = replace_pattern.copy()
            new_atoms.positions = chosen_quats[k].apply(new_atoms.positions)
            new_atoms.translate(atom_positions[0])
            new_atoms.positions %= np.diag(new_structure.cell)

            structure_index_map = {}
            if not replace_all:
                # atoms common to both patterns (the carboxylate oxygens) keep
                # their existing indices, charges and force-field terms
                structure_index_map = {key: chosen_indices[k][val]
                                       for key, val in replace2search.items()}
                new_structure.extend(new_atoms, offsets=offsets,
                                     structure_index_map=structure_index_map)
            else:
                new_structure.extend(new_atoms, offsets=offsets)

            doomed = set(int(i) for i in chosen_indices[k]) - \
                set(int(v) for v in structure_index_map.values())
            if to_delete.isdisjoint(doomed) or ignore_double_delete:
                to_delete |= doomed
            else:
                raise RuntimeError(
                    f"site {sites[k]} would delete atoms already claimed by "
                    "another site. Two linker matches overlap -- check the "
                    "search pattern and lower atol.")

    del new_structure[sorted(to_delete)]
    return new_structure


def _cell_is_orthorhombic(cell, tol: float = 1e-8) -> bool:
    c = np.asarray(cell, float)
    return bool(np.all(np.abs(c - np.diag(np.diag(c))) < tol))


# ---------------------------------------------------------------------------
# 2. The cached site set
# ---------------------------------------------------------------------------
@dataclass
class MofunSites:
    """Pristine structure + its linker sites + the topology, built once."""
    pristine: object                       # mofun Atoms
    search_pattern: object                 # mofun Atoms: intact BDC
    replace_pattern: object                # mofun Atoms: the two formate caps
    match_indices: list
    match_positions: np.ndarray
    quats: list
    graph: SiteGraph = field(default=None)
    has_real_quats: bool = field(default=False)
    _geomlib: object = field(default=None, repr=False)

    @property
    def n_sites(self) -> int:
        return len(self.match_indices)

    @property
    def cell(self) -> np.ndarray:
        return np.asarray(self.pristine.cell, float)

    @staticmethod
    def _quats_are_real(quats) -> bool:
        from scipy.spatial.transform import Rotation
        if not quats:
            return False
        return not all(isinstance(q, Rotation)
                       and np.allclose(q.as_quat(), [0, 0, 0, 1]) for q in quats)

    # -- construction --------------------------------------------------------
    @classmethod
    def from_pristine(cls, pristine_cif: str, linker_cml: str,
                      defective_cml: str, hop_shell: str = "nearest",
                      atol: float = 0.05, expected_sites: int | None = None,
                      verbose: bool = True) -> "MofunSites":
        pristine = load_atoms(pristine_cif)
        search = load_atoms(linker_cml)
        replace = load_atoms(defective_cml)
        if verbose:
            print(f"[pristine] {pristine_cif}: {len(pristine)} atoms")
            print(f"  search pattern  {linker_cml}: {len(search)} atoms")
            print(f"  replace pattern {defective_cml}: {len(replace)} atoms")

        mi, mp, q = find_sites(pristine, search, atol=atol)
        if verbose:
            print(f"  linker matches: {len(mi)}")
        if len(mi) == 0:
            raise RuntimeError(
                "no linkers matched. atol is probably too tight for this "
                "structure, or the pattern does not correspond to this CIF.")
        if expected_sites is not None and len(mi) != expected_sites:
            print(f"  ! expected {expected_sites} sites, matched {len(mi)}. "
                  "Fix this before running: a partial match silently drops "
                  "linker sites from the lattice.")

        obj = cls(pristine=pristine, search_pattern=search,
                  replace_pattern=replace, match_indices=mi,
                  match_positions=mp, quats=q)
        obj.has_real_quats = cls._quats_are_real(q)
        if verbose and not obj.has_real_quats:
            print("  this build returns no quaternions; snapshots will be "
                  "capped via the uio66_sites geometry library (uses the "
                  "returned match indices, needs pymatgen).")
        obj.build_graph(hop_shell=hop_shell, verbose=verbose)
        return obj

    def build_graph(self, hop_shell: str = "nearest", verbose: bool = True):
        nodes, clusters = find_nodes(
            Frame([str(e) for e in self.pristine.elements],
                  np.asarray(self.pristine.positions) @ np.linalg.inv(self.cell),
                  self.cell))
        self.graph = build_site_graph_from_matches(
            self.match_positions, nodes, self.cell,
            hop_shell=hop_shell, verbose=verbose)
        if verbose:
            n_per_node = self.n_sites / max(len(nodes), 1)
            print(f"  {len(nodes)} Zr6 nodes, {self.n_sites} sites "
                  f"({n_per_node:.1f} per node; UiO-66 fcu expects 6.0)")
        return self.graph

    # -- reading an existing defective structure -----------------------------
    def occupancy_from(self, defective_cif: str, atol: float = 0.05,
                       match_tol: float = 1.5, cross_check: bool = True,
                       verbose: bool = True):
        """Read the defect pattern out of an already-capped defective CIF.

        Primary read: the INTACT-linker pattern is matched in the defective
        structure; matched sites are occupied, the rest are defective. This is
        authoritative.

        Cross-check (independent method): the defective structure is broken
        into covalent fragments and BDC linkers / formate caps are counted and
        placed. Agreement with the primary read, plus ~2 formate groups per
        defect, confirms the parse. The old capped-pattern match was dropped:
        the two-formate pattern spans ~11 A with a void between the groups, so
        its relative-distance signature is too weak to match reliably.
        """
        defective = load_atoms(defective_cif)
        if verbose:
            print(f"[defective] {defective_cif}: {len(defective)} atoms")
        scale = (np.linalg.norm(np.asarray(defective.cell, float), axis=1) /
                 np.linalg.norm(self.cell, axis=1))
        if np.any(np.abs(scale - 1) > 0.05):
            print(f"  ! cell differs from the pristine one by {scale}. Sites "
                  "are matched by position, so the same supercell is required.")

        occ = np.zeros(self.n_sites, dtype=np.int8)
        intact = self._match_into_sites(defective, self.search_pattern, atol,
                                        match_tol, "intact linker", verbose)
        occ[intact] = 1
        defects = np.flatnonzero(occ == 0)

        report = {
            "n_sites": self.n_sites,
            "n_intact": int(len(intact)),
            "n_defects": int(len(defects)),
            "defect_fraction": float(len(defects) / self.n_sites),
        }

        if cross_check:
            report.update(self._fragment_cross_check(defective, occ, match_tol,
                                                     verbose))

        if verbose:
            print(f"  defects: {report['n_defects']}/{self.n_sites} "
                  f"({100 * report['defect_fraction']:.1f}%)")
            print(f"  state: {observables(occ, self.graph)}")
        return occ, report

    def _frame(self, atoms):
        """Convert a MOFUN Atoms object to a uio66_sites.Frame."""
        return Frame([str(e) for e in atoms.elements],
                     np.asarray(atoms.positions) @ np.linalg.inv(
                         np.asarray(atoms.cell, float)),
                     np.asarray(atoms.cell, float))

    def _fragment_cross_check(self, defective, occ, match_tol, verbose):
        """Independent check by covalent fragmentation of the defective cell.

        Uses the same routine as the pure uio66_sites path, on the same site
        graph. Reports whether the two methods agree on which sites are
        defective, and the formate-per-defect ratio.
        """
        from uio66_sites import read_occupancy
        frame = self._frame(defective)
        try:
            occ_frag, frag = read_occupancy(frame, self.graph,
                                            match_tol=max(match_tol, 2.0),
                                            verbose=False)
        except Exception as exc:
            if verbose:
                print(f"  ! fragmentation cross-check could not run: {exc}")
            return {"crosscheck": "failed", "crosscheck_error": str(exc)}

        disagree = int(np.sum(occ.astype(int) != occ_frag.astype(int)))
        out = {
            "crosscheck": "fragmentation",
            "crosscheck_agrees": bool(disagree == 0),
            "n_disagreements": disagree,
            "frag_n_bdc": frag["n_bdc"],
            "frag_n_formate": frag["n_formate"],
            "formate_per_defect": frag["formate_per_defect"],
        }
        if verbose:
            print(f"  cross-check (fragmentation): {frag['n_bdc']} BDC, "
                  f"{frag['n_formate']} formate; formate/defect="
                  f"{frag['formate_per_defect']}")
            if disagree == 0:
                print("  OK: pattern match and fragmentation agree on every site.")
            else:
                print(f"  ! methods disagree on {disagree} site(s). Inspect "
                      "before running: raise atol if the intact pattern missed "
                      "linkers, or check the caps if fragmentation missed them.")
        return out

    def _match_into_sites(self, structure, pattern, atol, match_tol,
                          label, verbose):
        mi, mp, _ = find_sites(structure, pattern, atol=atol)
        inv = np.linalg.inv(self.cell)
        hits, worst = [], 0.0
        unplaced = 0
        for positions in mp:
            positions = np.asarray(positions, dtype=float)
            if positions.size == 0 or positions.ndim != 2:
                unplaced += 1                 # degenerate / empty match, skip
                continue
            c = snap_frac(positions.mean(axis=0) @ inv)
            d = self.graph.site_frac - c
            d -= np.round(d)
            dist = np.linalg.norm(d @ self.cell, axis=1)
            s = int(np.argmin(dist))
            if dist[s] > match_tol:
                unplaced += 1
                continue
            worst = max(worst, float(dist[s]))
            hits.append(s)
        if verbose:
            extra = f", {unplaced} unplaced" if unplaced else ""
            print(f"  {label}: {len(mi)} matches -> {len(set(hits))} sites "
                  f"(worst offset {worst:.2f} A{extra})")
        if len(hits) != len(set(hits)):
            print(f"  ! {label}: two matches mapped onto the same site.")
        return sorted(set(hits))

    # -- writing a snapshot --------------------------------------------------
    def snapshot(self, occ: np.ndarray):
        """The capped structure for this occupancy, as a MOFUN Atoms object.

        Only available when the build returned real quaternions. Otherwise use
        write(), which routes through the uio66_sites geometry library.
        """
        occ = np.asarray(occ)
        if len(occ) != self.n_sites:
            raise ValueError(f"occ has {len(occ)} entries, {self.n_sites} sites")
        if not self.has_real_quats:
            raise RuntimeError(
                "snapshot() needs real quaternions, which this MOFUN build did "
                "not return. Use write(), which caps via the geometry library.")
        defects = np.flatnonzero(occ == 0)
        if len(defects) == 0:
            return self.pristine.copy()
        return replace_at_matches(self.pristine, self.search_pattern,
                                  self.replace_pattern, self.match_indices,
                                  self.match_positions, self.quats,
                                  sites=defects)

    def build_geometry_library(self):
        """A uio66_sites.GeometryLibrary keyed by THIS object's site order.

        Uses the returned match indices to pull each linker's atoms from the
        pristine structure and derive its two formate caps the uio66_sites way.
        This is the write path when MOFUN does not return quaternions: it needs
        only indices, and inherits the linker geometry from the pristine CIF.
        Requires pymatgen (for CIF output via write_frame).
        """
        from uio66_sites import Frame, GeometryLibrary, _formate_from_bdc, classify
        if self.match_indices is None:
            raise RuntimeError("no match indices available; cannot build a "
                               "geometry library (see mofun_signature()).")
        cell = self.cell
        frame = Frame([str(e) for e in self.pristine.elements],
                      np.asarray(self.pristine.positions) @ np.linalg.inv(cell),
                      cell)
        add_h = any(s == "H" for s in frame.species)

        bdc, caps, used, bad = {}, {}, set(), 0
        for s in range(self.n_sites):
            frag = [int(i) for i in self.match_indices[s]]
            kind = classify(frame, frag)
            if kind != "bdc":
                bad += 1
            seed = frame.frac[frag[0]]
            d = frame.frac[frag] - seed
            d -= np.round(d)
            bdc[s] = ([frame.species[i] for i in frag], seed + d)
            caps[s] = _formate_from_bdc(frame, frag, add_h)
            used.update(frag)
        if bad:
            print(f"  ! {bad}/{self.n_sites} matched fragments are not a clean "
                  "BDC (C8O4). The search pattern likely includes framework "
                  "atoms; caps may be malformed.")
        base = sorted(set(range(len(frame))) - used)
        return GeometryLibrary(cell=cell,
                               base_species=[frame.species[i] for i in base],
                               base_frac=frame.frac[base], bdc=bdc, caps=caps)

    def write(self, occ: np.ndarray, path: str) -> None:
        """Save a capped snapshot for this occupancy.

        Uses MOFUN replacement when quaternions are available, otherwise the
        geometry library. `occ` is 1=intact / 0=defective, in this object's
        site order (the order the kMC uses).
        """
        occ = np.asarray(occ)
        if len(occ) != self.n_sites:
            raise ValueError(f"occ has {len(occ)} entries, {self.n_sites} sites")
        if self.has_real_quats:
            self.snapshot(occ).save(path)
            return
        from uio66_sites import compose, write_frame
        if self._geomlib is None:
            self._geomlib = self.build_geometry_library()
        write_frame(compose(occ.astype(np.int8), self._geomlib), path)

    # -- cache ---------------------------------------------------------------
    def save_cache(self, path: str) -> None:
        """Store the pattern search so it never has to run twice."""
        np.savez(path,
                 match_indices=np.asarray(self.match_indices, dtype=int),
                 match_positions=self.match_positions,
                 quats=np.array([q.as_quat() for q in self.quats], dtype=float),
                 cell=self.cell)

    @classmethod
    def from_cache(cls, path: str, pristine_cif: str, linker_cml: str,
                   defective_cml: str, hop_shell: str = "nearest",
                   verbose: bool = True) -> "MofunSites":
        from scipy.spatial.transform import Rotation
        z = np.load(path)
        pristine = load_atoms(pristine_cif)
        if not np.allclose(z["cell"], np.asarray(pristine.cell, float), atol=1e-6):
            raise ValueError("cached cell does not match this CIF's cell")
        obj = cls(pristine=pristine,
                  search_pattern=load_atoms(linker_cml),
                  replace_pattern=load_atoms(defective_cml),
                  match_indices=[tuple(r) for r in z["match_indices"]],
                  match_positions=z["match_positions"],
                  quats=[Rotation.from_quat(q) for q in z["quats"]])
        obj.has_real_quats = cls._quats_are_real(obj.quats)
        if verbose:
            print(f"[cache] {path}: {obj.n_sites} sites")
        obj.build_graph(hop_shell=hop_shell, verbose=verbose)
        return obj


# ---------------------------------------------------------------------------
# 3. kMC adapter
# ---------------------------------------------------------------------------
class MofunLattice(StructureLattice):
    """StructureLattice whose snapshots are written by MOFUN."""

    def __init__(self, p, sites: MofunSites, occ: np.ndarray):
        super().__init__(p, sites.graph, occ, library=None)
        self.sites = sites

    def write_cif(self, path: str) -> None:
        self.sites.write(self.occ, path)


def load_initial_state(defective_cif: str, pristine_cif: str,
                       linker_cml: str = "uio66-linker.cml",
                       defective_cml: str = "uio66-linker-defective.cml",
                       hop_shell: str = "nearest", atol: float = 0.05,
                       cache: str | None = None, verbose: bool = True):
    """Drop-in replacement for uio66_sites.load_initial_state, via MOFUN.

    Returns (sites, occ, report). Pass `cache` to reuse the pattern search.
    """
    if cache and __import__("os").path.exists(cache):
        sites = MofunSites.from_cache(cache, pristine_cif, linker_cml,
                                      defective_cml, hop_shell, verbose)
    else:
        sites = MofunSites.from_pristine(pristine_cif, linker_cml,
                                         defective_cml, hop_shell=hop_shell,
                                         atol=atol, verbose=verbose)
        if cache:
            sites.save_cache(cache)
    occ, report = sites.occupancy_from(defective_cif, atol=atol, verbose=verbose)
    return sites, occ, report


# ---------------------------------------------------------------------------
# 4. Run this before trusting anything
# ---------------------------------------------------------------------------
def self_check(pristine_cif: str, defective_cif: str,
               linker_cml: str = "uio66-linker.cml",
               defective_cml: str = "uio66-linker-defective.cml",
               atol: float = 0.05, expected_sites: int | None = None,
               expected_defect_fraction: float | None = None) -> bool:
    """End-to-end validation on your own files.

    1. pristine search finds the expected number of sites, 6 per Zr6 node
    2. intact-pattern read and covalent fragmentation agree on which sites are
       defective, with ~2 formate groups per missing linker
    3. the defect fraction is what you think it is
    4. round trip: rebuild the defective structure from the pristine one at the
       same sites, re-read it, and get the same occupancy back
    """
    ok = True
    print("=" * 70)
    sites = MofunSites.from_pristine(pristine_cif, linker_cml, defective_cml,
                                     atol=atol, expected_sites=expected_sites)
    n_nodes = len(sites.graph.node_frac)
    if sites.n_sites != 6 * n_nodes:
        print(f"  FAIL: {sites.n_sites} sites for {n_nodes} nodes "
              f"(expected {6 * n_nodes})")
        ok = False
    if len(sites.graph.coordination()) != 1:
        print(f"  FAIL: non-uniform hop coordination "
              f"{sorted(sites.graph.coordination())}")
        ok = False

    print("-" * 70)
    occ, report = sites.occupancy_from(defective_cif, atol=atol)
    if report.get("crosscheck") == "fragmentation":
        if not report.get("crosscheck_agrees"):
            print(f"  FAIL: intact-pattern read and fragmentation disagree on "
                  f"{report.get('n_disagreements')} site(s)")
            ok = False
        fpd = report.get("formate_per_defect")
        if fpd is not None and abs(fpd - 2.0) > 0.05:
            print(f"  FAIL: {fpd} formate groups per defect (expected 2.0). "
                  "The caps in the defective CIF are not 2-per-missing-linker, "
                  "or fragmentation mis-identified them.")
            ok = False
    elif report.get("crosscheck") == "failed":
        print("  WARN: fragmentation cross-check could not run; occupancy is "
              "from the intact-pattern read only. Verify the caps manually.")
    if expected_defect_fraction is not None and \
            abs(report["defect_fraction"] - expected_defect_fraction) > 0.01:
        print(f"  FAIL: defect fraction {report['defect_fraction']:.3f} != "
              f"{expected_defect_fraction:.3f}")
        ok = False

    print("-" * 70)
    print("[round trip] rebuilding the defective structure from pristine")
    rebuilt_path = "_selfcheck_rebuilt.cif"
    sites.write(occ, rebuilt_path)
    occ2, _ = sites.occupancy_from(rebuilt_path, atol=atol, cross_check=False,
                                   verbose=False)
    same = bool(np.array_equal(occ, occ2))
    print(f"  occupancy round trip: {same}")
    if not same:
        diff = np.flatnonzero(occ != occ2)
        print(f"  FAIL: {len(diff)} sites differ, e.g. {diff[:10]}")
        ok = False

    print("=" * 70)
    print("PASS" if ok else "FAIL -- do not run production kMC yet")
    return ok


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print(__doc__)
        print("usage: python mofun_sites.py PRISTINE.cif DEFECTIVE.cif "
              "[linker.cml] [linker-defective.cml] [atol]")
        sys.exit(1)
    args = sys.argv[1:]
    self_check(args[0], args[1],
               linker_cml=args[2] if len(args) > 2 else "uio66-linker.cml",
               defective_cml=args[3] if len(args) > 3 else "uio66-linker-defective.cml",
               atol=float(args[4]) if len(args) > 4 else 0.05)
