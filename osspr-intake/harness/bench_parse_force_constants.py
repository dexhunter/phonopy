"""Time phonopy.file_IO.parse_FORCE_CONSTANTS.

Workload: the text FORCE_CONSTANTS file for a 3x3x3 supercell of
example/NaCl (216 atoms, full FC). The file is written once; each sample
only parses it.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(_REPO)]

import common  # noqa: E402

from phonopy.file_IO import (  # noqa: E402
    parse_FORCE_CONSTANTS,
    write_FORCE_CONSTANTS,
)


def main() -> None:
    cell = common.nacl_supercell((3, 3, 3))
    force_constants = common.deterministic_force_constants(len(cell))
    shm = Path("/dev/shm")
    directory = shm if shm.is_dir() else None
    with tempfile.TemporaryDirectory(prefix="osspr-fc-", dir=directory) as tmp:
        filename = Path(tmp) / "FORCE_CONSTANTS"
        write_FORCE_CONSTANTS(force_constants, filename)
        file_bytes = filename.stat().st_size

        def call() -> tuple[int, int, int, int]:
            parsed = parse_FORCE_CONSTANTS(filename)
            return parsed.shape

        assert call() == force_constants.shape
        samples = common.time_calls(call)
        common.emit(
            "phonopy.file_IO.parse_FORCE_CONSTANTS",
            samples,
            natom=len(cell),
            file_bytes=file_bytes,
            structure="example/NaCl/POSCAR-unitcell supercell 3x3x3",
        )


if __name__ == "__main__":
    main()
