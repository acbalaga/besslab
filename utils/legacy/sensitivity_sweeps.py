"""Sensitivity sweep utilities for PV/SOC/RTE what-if grids.

Deprecated in favor of :mod:`utils.sweeps` but kept for backward compatibility.
"""
from __future__ import annotations

from utils.sweeps import build_soc_windows, generate_values, run_sensitivity_grid

__all__ = [
    "build_soc_windows",
    "generate_values",
    "run_sensitivity_grid",
]
