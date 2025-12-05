"""Backwards-compatible shim for legacy imports.

Some environments still import :mod:`bess_size_sweeps` directly. This module
re-exports the current sweep utilities from :mod:`utils.sweeps` so legacy
entry points and Streamlit reruns continue to work without requiring callers
to update their import paths.
"""

from utils.sweeps import run_candidate_simulation, sweep_bess_sizes

__all__ = ["run_candidate_simulation", "sweep_bess_sizes"]
