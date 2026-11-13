"""Time phonopy.harmonic.force_constants.symmetrize_force_constants_by_space_group.

Workload: 2x2x2 supercell of example/NaCl/POSCAR-unitcell (64 atoms) and the
space-group operations of that cell. This is the kernel behind
``--fc-spg-symmetry``. Each sample symmetrizes a fresh copy so the in-place
update does not include the copy in the timed section.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(_REPO)]

import common  # noqa: E402

from phonopy.harmonic.force_constants import (  # noqa: E402
    symmetrize_force_constants_by_space_group,
)
from phonopy.structure.symmetry import Symmetry  # noqa: E402


def main() -> None:
    cell = common.nacl_supercell((2, 2, 2))
    symmetry = Symmetry(cell)
    force_constants = common.deterministic_force_constants(len(cell))
    n_operations = len(symmetry.symmetry_operations["rotations"])
    copies = [
        force_constants.copy() for _ in range(common.WARMUP + common.REPEATS)
    ]
    cursor = {"i": 0}

    def call() -> float:
        work = copies[cursor["i"]]
        cursor["i"] += 1
        symmetrize_force_constants_by_space_group(work, cell.cell, symmetry)
        return float(work.ravel()[0])

    # One untimed check that the kernel runs on this cell, then time the rest
    # of the preallocated copies. time_calls itself warms up once.
    probe = force_constants.copy()
    symmetrize_force_constants_by_space_group(probe, cell.cell, symmetry)
    assert probe.shape == force_constants.shape
    assert np_isfinite(probe)

    samples = common.time_calls(call)
    common.emit(
        "phonopy.harmonic.force_constants.symmetrize_force_constants_by_space_group",
        samples,
        natom=len(cell),
        n_operations=n_operations,
        structure="example/NaCl/POSCAR-unitcell supercell 2x2x2",
    )


def np_isfinite(array) -> bool:
    import numpy as np

    return bool(np.isfinite(array).all())


if __name__ == "__main__":
    main()
