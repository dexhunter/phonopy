"""Shared timing and example-structure helpers for intake benchmarks.

Workloads are built from files under example/. Timed sections call one phonopy
function; setup (structure expansion, file creation, symmetry) is outside the
timer. Noise is the full spread of the samples: (max - min) / median * 100.
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
REPEATS = 7
WARMUP = 1


def repo_path(*parts: str) -> Path:
    """Return an absolute path inside the phonopy checkout."""
    return REPO_ROOT.joinpath(*parts)


def time_calls(fn, repeats: int = REPEATS, warmup: int = WARMUP) -> list[float]:
    """Return per-call wall times in milliseconds."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def summarize(samples: list[float]) -> dict[str, float]:
    """Return median milliseconds and the full-spread noise percent."""
    median = statistics.median(samples)
    noise = (max(samples) - min(samples)) / median * 100.0
    return {"median_ms": median, "noise_pct": noise}


def emit(function: str, samples: list[float], **extra) -> None:
    """Print one JSON object describing this run."""
    stats = summarize(samples)
    payload = {
        "function": function,
        "repeats": len(samples),
        "warmup": WARMUP,
        "samples_ms": samples,
        "median_ms": stats["median_ms"],
        "noise_pct": stats["noise_pct"],
        "noise_definition": "(max-min)/median*100",
    }
    payload.update(extra)
    print(json.dumps(payload))


def deterministic_force_constants(natom: int) -> np.ndarray:
    """Return a dense, deterministic full force-constant array.

    Values are not physical. They only give the writer, parser, and
    symmetrizer a stable nonzero tensor of the shape a real supercell
    calculation produces.
    """
    index_i = np.arange(natom, dtype=np.float64)[:, None, None, None]
    index_j = np.arange(natom, dtype=np.float64)[None, :, None, None]
    axis_a = np.arange(3, dtype=np.float64)[None, None, :, None]
    axis_b = np.arange(3, dtype=np.float64)[None, None, None, :]
    values = np.sin(
        (index_i + 1.0) * 0.17
        + (index_j + 1.0) * 0.13
        + (axis_a + 1.0) * 0.07
        + (axis_b + 1.0) * 0.05
    )
    return np.ascontiguousarray(values * 0.05, dtype=np.float64)


def nacl_supercell(diagonal: tuple[int, int, int]):
    """Expand example/NaCl/POSCAR-unitcell by a diagonal supercell matrix."""
    from phonopy.interface.vasp import read_vasp
    from phonopy.structure.cells import get_supercell

    cell = read_vasp(repo_path("example", "NaCl", "POSCAR-unitcell"))
    return get_supercell(cell, np.diag(diagonal))


def al2o3_unitcell():
    """Return the bundled Al2O3 conventional cell."""
    from phonopy.interface.vasp import read_vasp

    return read_vasp(repo_path("example", "Al2O3", "POSCAR-unitcell"))
