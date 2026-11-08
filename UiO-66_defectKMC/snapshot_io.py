"""
snapshot_io.py -- save kMC snapshots as real structures (no pymatgen)
=====================================================================

Turns a kMC occupancy array into a full atomic structure (intact BDC linkers +
formate caps at defects, via uio66_sites.compose) and writes a standard P1 CIF.
Also stacks occupancies into a single .npz for lightweight storage/reanalysis.

A geometry writer needs a PRISTINE template (atoms for every linker site) to
build the per-site BDC/formate library. The template is tiled to match the
kMC supercell automatically.
"""
from __future__ import annotations

import itertools
import numpy as np

from uio66_sites import Frame, build_library, compose


def read_p1_cif(path):
    cl, sp, fr = {}, [], []
    for ln in open(path):
        t = ln.split()
        for k in ("_cell_length_a", "_cell_length_b", "_cell_length_c"):
            if ln.strip().startswith(k):
                cl[k] = float(t[1])
        if len(t) >= 6 and len(t[0]) <= 2 and t[0][:1].isalpha() and t[0][:1].isupper():
            # standard column order: symbol label occupancy fract_x fract_y fract_z
            try:
                fr.append([float(t[3]), float(t[4]), float(t[5])]); sp.append(t[0])
            except ValueError:
                pass
    cell = np.diag([cl["_cell_length_a"], cl["_cell_length_b"], cl["_cell_length_c"]])
    return Frame(sp, np.array(fr), cell)


def write_cif(frame: Frame, path: str, title: str = "snapshot") -> None:
    """Minimal standard P1 CIF (ASE-style columns: symbol label occ x y z)."""
    a, b, c = np.linalg.norm(frame.cell, axis=1)
    frac = np.mod(frame.frac, 1.0)
    cnt: dict = {}
    with open(path, "w") as o:
        o.write(f"data_{title}\n")
        o.write("_symmetry_space_group_name_H-M   'P 1'\n")
        o.write("_symmetry_Int_Tables_number      1\n")
        o.write(f"_cell_length_a   {a:.5f}\n_cell_length_b   {b:.5f}\n_cell_length_c   {c:.5f}\n")
        o.write("_cell_angle_alpha   90.0\n_cell_angle_beta    90.0\n_cell_angle_gamma   90.0\n")
        o.write("loop_\n _atom_site_type_symbol\n _atom_site_label\n _atom_site_occupancy\n")
        o.write(" _atom_site_fract_x\n _atom_site_fract_y\n _atom_site_fract_z\n")
        for s, f in zip(frame.species, frac):
            cnt[s] = cnt.get(s, 0) + 1
            o.write(f"{s}  {s}{cnt[s]}  1.0000  {f[0]:.6f}  {f[1]:.6f}  {f[2]:.6f}\n")


def build_geometry_writer(template_cif: str, sg, cell, verbose: bool = True):
    """Return write(occ, path) that composes occupancy -> CIF.

    The pristine template is tiled to match `cell`/`sg` (the kMC supercell).
    """
    tmpl = read_p1_cif(template_cif)
    R = int(round(np.linalg.norm(cell[0]) / np.linalg.norm(tmpl.cell[0])))
    reps = np.array(list(itertools.product(range(R), repeat=3)))
    frac = np.vstack([(tmpl.frac + r) / R for r in reps])
    species = list(tmpl.species) * (R ** 3)
    tiled = Frame(species, frac, cell)
    lib = build_library(tiled, sg, verbose=verbose)

    def write(occ, path, title="snapshot"):
        write_cif(compose(np.asarray(occ, np.int8), lib), path, title)
    return write


class OccupancyLog:
    """Accumulate occupancies + step indices; save as one .npz."""
    def __init__(self):
        self.steps, self.occs = [], []

    def add(self, step, occ):
        self.steps.append(int(step)); self.occs.append(np.asarray(occ, np.int8).copy())

    def save(self, path, site_frac=None, cell=None):
        kw = dict(steps=np.array(self.steps), occ=np.array(self.occs))
        if site_frac is not None:
            kw["site_frac"] = site_frac
        if cell is not None:
            kw["cell"] = cell
        np.savez_compressed(path, **kw)
