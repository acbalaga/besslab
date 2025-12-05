"""Grid search utilities for evaluating candidate BESS sizes.

Deprecated in favor of :mod:`utils.sweeps` but kept for backward compatibility.
"""
from __future__ import annotations

from utils.sweeps import run_candidate_simulation, sweep_bess_sizes

__all__ = [
    "run_candidate_simulation",
    "sweep_bess_sizes",
]
