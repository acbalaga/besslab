"""Sensitivity sweep utilities for PV/SOC/RTE what-if grids.

This module isolates the heavier grid computations from the main Streamlit app
so they can be triggered on demand (e.g., inside an expander). The functions
here are intentionally light on Streamlit dependencies; callers should pass in
functions for simulation and summarization to keep coupling minimal.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, List, Sequence, Tuple

import pandas as pd


def generate_values(min_value: float, max_value: float, steps: int) -> List[float]:
    """Return an inclusive list of evenly spaced values.

    The caller is responsible for ensuring ``steps`` is positive. When
    ``steps`` is ``1``, the midpoint is returned to keep the sweep centered.
    """

    steps = max(1, steps)
    if steps == 1:
        return [float((min_value + max_value) / 2.0)]

    span = max_value - min_value
    if span <= 0:
        return [float(min_value)]

    step = span / float(steps - 1)
    return [float(min_value + i * step) for i in range(steps)]


def build_soc_windows(
    floor_range: Tuple[float, float],
    ceiling_range: Tuple[float, float],
    floor_steps: int,
    ceiling_steps: int,
    min_gap: float = 0.05,
) -> List[Tuple[float, float]]:
    """Generate SOC floor/ceiling pairs while respecting a minimum gap."""

    floors = generate_values(floor_range[0], floor_range[1], floor_steps)
    ceilings = generate_values(ceiling_range[0], ceiling_range[1], ceiling_steps)

    windows: List[Tuple[float, float]] = []
    for floor in floors:
        for ceiling in ceilings:
            if floor + min_gap >= ceiling:
                continue
            windows.append((floor, ceiling))

    return windows


def run_sensitivity_grid(
    base_cfg: Any,
    pv_df: pd.DataFrame,
    cycle_df: pd.DataFrame,
    dod_override: str,
    pv_oversize_factors: Sequence[float],
    soc_windows: Sequence[Tuple[float, float]],
    rte_values: Sequence[float],
    simulate_project_fn: Callable[[Any, pd.DataFrame, pd.DataFrame, str, bool], Any],
    summarize_fn: Callable[[Any], Any],
) -> pd.DataFrame:
    """Run a grid of sensitivity scenarios and return a tidy summary table.

    Each combination scales the PV profile, adjusts the SOC window and round-trip
    efficiency, and then reuses the supplied simulator to avoid duplicating logic.
    Results include a few headline KPIs for downstream charting.
    """

    rows: List[dict[str, Any]] = []

    for pv_factor in pv_oversize_factors:
        scaled_pv = pv_df.copy()
        scaled_pv["pv_mw"] = scaled_pv["pv_mw"] * pv_factor

        for floor, ceiling in soc_windows:
            for rte in rte_values:
                cfg_for_run = replace(
                    base_cfg,
                    rte_roundtrip=float(rte),
                    soc_floor=float(floor),
                    soc_ceiling=float(ceiling),
                )

                sim_output = simulate_project_fn(
                    cfg_for_run, scaled_pv, cycle_df, dod_override, False
                )
                summary = summarize_fn(sim_output)

                rows.append(
                    {
                        "pv_oversize_factor": float(pv_factor),
                        "soc_floor": float(floor),
                        "soc_ceiling": float(ceiling),
                        "rte_roundtrip": float(rte),
                        "compliance_pct": summary.compliance,
                        "bess_share_pct": summary.bess_share_of_firm,
                        "shortfall_mwh": summary.total_shortfall_mwh,
                    }
                )

    return pd.DataFrame(rows)


__all__ = [
    "build_soc_windows",
    "generate_values",
    "run_sensitivity_grid",
]
