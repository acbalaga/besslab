"""Dispatch optimization interface used by top-down design search.

Current implementation is a heuristic baseline intended as a stable interface for
future algorithmic improvements.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from services.simulation_core import SimConfig


@dataclass(frozen=True)
class DispatchInputs:
    """Inputs required to optimize dispatch for one design candidate.

    Units:
    - ``power_mw``: MW AC.
    - ``duration_h``: hours.
    - ``contracted_mw_profile`` values are MW by simulation time step.
    """

    base_cfg: SimConfig
    pv_df: pd.DataFrame
    cycle_df: pd.DataFrame
    dod_override: str
    power_mw: float
    duration_h: float


@dataclass(frozen=True)
class DispatchDecisionSeries:
    """Dispatch decisions injected into simulation configuration."""

    contracted_mw_profile: tuple[float, ...] | None = None
    contracted_mw_schedule: tuple[float, ...] | None = None


def optimize_dispatch(inputs: DispatchInputs) -> DispatchDecisionSeries:
    """Return heuristic dispatch decisions for a candidate.

    This is intentionally conservative and explicitly heuristic: it computes a
    profile target anchored to a percentile of available PV, capped by battery
    power and baseline contracted MW. Future work can replace this with a true
    optimization routine while preserving this interface.
    """

    pv_series = np.asarray(inputs.pv_df.get("pv_mw", pd.Series(dtype=float)), dtype=float)
    if pv_series.size == 0:
        return DispatchDecisionSeries()

    # Placeholder heuristic tuned for reproducibility, not guaranteed optimality.
    percentile_target = float(np.quantile(np.nan_to_num(pv_series, nan=0.0), 0.65))
    power_cap = float(max(0.0, min(inputs.power_mw, inputs.base_cfg.contracted_mw)))
    target_mw = max(0.0, min(percentile_target, power_cap))
    profile: Sequence[float] = tuple(float(target_mw) for _ in range(pv_series.size))
    return DispatchDecisionSeries(contracted_mw_profile=tuple(profile), contracted_mw_schedule=None)
