"""Time phonopy.phonon.mesh.Mesh.write_yaml.

Workload: example/Al2O3/POSCAR-unitcell (30 atoms) on a Gamma-centered
4x4x4 mesh with eigenvectors. Frequencies and eigenvectors are deterministic
stand-ins with the shapes a real mesh run stores; write_yaml only formats
those arrays. The file is written to a temporary directory on each call.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(_REPO)]

import common  # noqa: E402

from phonopy.phonon.mesh import Mesh  # noqa: E402


def gamma_mesh(mesh: tuple[int, int, int]) -> np.ndarray:
    """Return the full Gamma-centered q-point grid in fractional coordinates."""
    axes = [np.arange(n, dtype=np.float64) / n for n in mesh]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return np.ascontiguousarray(grid.reshape(-1, 3))


def mesh_ready_to_write(cell, mesh_numbers: tuple[int, int, int]) -> Mesh:
    """Build a Mesh whose write_yaml inputs match a full-grid eigenvector run."""
    qpoints = gamma_mesh(mesh_numbers)
    nqpoint = len(qpoints)
    nbands = len(cell) * 3
    band = np.arange(nbands, dtype=np.float64)
    frequencies = band[None, :] * (12.0 / max(nbands - 1, 1))
    frequencies = frequencies + (np.arange(nqpoint, dtype=np.float64)[:, None] * 1.0e-3)
    component = np.arange(nbands, dtype=np.float64)[None, :, None]
    mode = np.arange(nbands, dtype=np.float64)[None, None, :]
    q_index = np.arange(nqpoint, dtype=np.float64)[:, None, None]
    phase = q_index * 0.01 + component * 0.002 + mode * 0.003
    eigenvectors = (np.cos(phase) + 1j * np.sin(phase * 0.5)) / np.sqrt(nbands)

    mesh = Mesh.__new__(Mesh)
    mesh._cell = cell
    mesh._mesh = np.array(mesh_numbers, dtype=np.int64)
    mesh._qpoints = qpoints
    mesh._weights = np.ones(nqpoint, dtype=np.int64)
    mesh._frequencies = np.ascontiguousarray(frequencies, dtype=np.float64)
    mesh._group_velocities = None
    mesh._with_eigenvectors = True
    mesh._eigenvectors = np.ascontiguousarray(eigenvectors)
    return mesh


def main() -> None:
    cell = common.al2o3_unitcell()
    mesh_numbers = (4, 4, 4)
    mesh = mesh_ready_to_write(cell, mesh_numbers)
    shm = Path("/dev/shm")
    directory = shm if shm.is_dir() else None
    with tempfile.TemporaryDirectory(prefix="osspr-mesh-", dir=directory) as tmp:
        filename = Path(tmp) / "mesh.yaml"

        def call() -> int:
            mesh.write_yaml(filename)
            return filename.stat().st_size

        output_bytes = call()
        assert output_bytes > 1_000_000
        samples = common.time_calls(call)
        common.emit(
            "phonopy.phonon.mesh.Mesh.write_yaml",
            samples,
            natom=len(cell),
            nqpoint=int(mesh._qpoints.shape[0]),
            mesh=[4, 4, 4],
            with_eigenvectors=True,
            output_bytes=output_bytes,
            structure="example/Al2O3/POSCAR-unitcell",
        )


if __name__ == "__main__":
    main()
