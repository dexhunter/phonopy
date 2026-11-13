"""Time phonopy.file_IO.get_FORCE_CONSTANTS_lines.

Workload: full force constants of a 3x3x3 supercell of example/NaCl
(POSCAR-unitcell, 8 atoms -> 216 atoms). The text writer formats every
3-vector in Python.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(_REPO)]

import common  # noqa: E402

from phonopy.file_IO import get_FORCE_CONSTANTS_lines  # noqa: E402


def main() -> None:
    cell = common.nacl_supercell((3, 3, 3))
    force_constants = common.deterministic_force_constants(len(cell))
    natom = len(cell)
    expected_lines = 1 + natom * natom * 4

    def call() -> int:
        lines = get_FORCE_CONSTANTS_lines(force_constants)
        return len(lines)

    assert call() == expected_lines
    samples = common.time_calls(call)
    common.emit(
        "phonopy.file_IO.get_FORCE_CONSTANTS_lines",
        samples,
        natom=natom,
        expected_lines=expected_lines,
        structure="example/NaCl/POSCAR-unitcell supercell 3x3x3",
    )


if __name__ == "__main__":
    main()
