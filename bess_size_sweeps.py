"""Grid search utilities for evaluating candidate BESS sizes.

The functions here keep the grid-search mechanics isolated from the main
Streamlit app. A lightweight adapter wraps the existing simulation engine so
callers can plug in either the real simulator or a stub during tests.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import numpy as np
import pandas as pd

from app import SimulationOutput, SimulationSummary, SimConfig, simulate_project, summarize_simulation


def run_candidate_simulation(
    base_cfg: SimConfig,
    pv_df: pd.DataFrame,
    cycle_df: pd.DataFrame,
    dod_override: str,
    power_mw: float,
    duration_h: float,
    simulate_fn: Callable[[SimConfig, pd.DataFrame, pd.DataFrame, str, bool], SimulationOutput] = simulate_project,
    summarize_fn: Callable[[SimulationOutput], SimulationSummary] = summarize_simulation,
) -> Tuple[SimulationOutput, SimulationSummary]:
    """Call the core engine for a single BESS size.

    The adapter lives in one place so the streamlit app can swap in the real
    engine while tests can supply a stub without touching the grid-search
    logic.
    """

    candidate_energy_mwh = float(power_mw * duration_h)
    cfg_for_run = replace(
        base_cfg,
        initial_power_mw=float(power_mw),
        initial_usable_mwh=candidate_energy_mwh,
    )

    sim_output = simulate_fn(cfg_for_run, pv_df, cycle_df, dod_override, False)
    summary = summarize_fn(sim_output)
    return sim_output, summary


def _evaluate_feasibility(
    sim_output: SimulationOutput,
    summary: SimulationSummary,
    cfg: SimConfig,
    min_soh: float,
) -> Tuple[float, float, bool, bool, bool]:
    """Return simple feasibility markers derived from simulation results."""

    max_eq_cycles = max((r.eq_cycles for r in sim_output.results), default=float("nan"))
    min_soh_total = min((r.soh_total for r in sim_output.results), default=float("nan"))

    cycles_allowed_per_year = cfg.max_cycles_per_day_cap * 365.0
    cycle_limit_hit = bool(
        not np.isnan(max_eq_cycles)
        and cycles_allowed_per_year > 0
        and max_eq_cycles > cycles_allowed_per_year
    )
    soh_below_min = bool(not np.isnan(min_soh_total) and min_soh_total < min_soh)
    feasible = (not cycle_limit_hit) and (not soh_below_min)
    return max_eq_cycles, min_soh_total, cycle_limit_hit, soh_below_min, feasible


def _resolve_ranking_column(use_case: str, ranking_kpi: str | None) -> Tuple[str, bool]:
    """Map a use case/KPI to a column and sort order.

    The returned tuple is ``(column_name, ascending)`` so callers can hand it
    directly to ``DataFrame.sort_values``.
    """

    defaults = {
        "reliability": ("compliance_pct", False),
        "energy": ("total_project_generation_mwh", False),
        "shortfall": ("total_shortfall_mwh", True),
    }

    if ranking_kpi:
        return ranking_kpi, False
    return defaults.get(use_case, ("compliance_pct", False))


def sweep_bess_sizes(
    base_cfg: SimConfig,
    pv_df: pd.DataFrame,
    cycle_df: pd.DataFrame,
    dod_override: str,
    power_mw_values: Iterable[float],
    duration_h_values: Iterable[float],
    use_case: str = "reliability",
    ranking_kpi: str | None = None,
    min_soh: float = 0.6,
    simulate_fn: Callable[[SimConfig, pd.DataFrame, pd.DataFrame, str, bool], SimulationOutput] = simulate_project,
    summarize_fn: Callable[[SimulationOutput], SimulationSummary] = summarize_simulation,
) -> pd.DataFrame:
    """Run a simple grid search over power/duration candidates.

    The grid is fully enumerated (no heuristics) and returns one row per
    candidate with core KPIs, feasibility markers, and the top-ranked option
    flagged.
    """

    rows: List[dict[str, float | bool | str]] = []

    for power_mw in power_mw_values:
        for duration_h in duration_h_values:
            candidate_energy_mwh = float(power_mw * duration_h)
            sim_output, summary = run_candidate_simulation(
                base_cfg,
                pv_df,
                cycle_df,
                dod_override,
                power_mw,
                duration_h,
                simulate_fn=simulate_fn,
                summarize_fn=summarize_fn,
            )

            (
                max_eq_cycles,
                min_soh_total,
                cycle_limit_hit,
                soh_below_min,
                feasible,
            ) = _evaluate_feasibility(sim_output, summary, sim_output.cfg, min_soh)

            rows.append(
                {
                    "power_mw": float(power_mw),
                    "duration_h": float(duration_h),
                    "energy_mwh": candidate_energy_mwh,
                    "compliance_pct": summary.compliance,
                    "total_project_generation_mwh": summary.total_project_generation_mwh,
                    "bess_generation_mwh": summary.bess_generation_mwh,
                    "pv_generation_mwh": summary.pv_generation_mwh,
                    "pv_excess_mwh": summary.pv_excess_mwh,
                    "bess_losses_mwh": summary.bess_losses_mwh,
                    "total_shortfall_mwh": summary.total_shortfall_mwh,
                    "avg_eq_cycles_per_year": summary.avg_eq_cycles_per_year,
                    "max_eq_cycles_per_year": max_eq_cycles,
                    "min_soh_total": min_soh_total,
                    "cycle_limit_hit": cycle_limit_hit,
                    "soh_below_min": soh_below_min,
                    "feasible": feasible,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    ranking_column, ascending = _resolve_ranking_column(use_case, ranking_kpi)
    if ranking_column not in df.columns:
        df[ranking_column] = np.nan

    df["is_best"] = False
    feasible_df = df[df["feasible"]]
    if not feasible_df.empty:
        best_idx = feasible_df[ranking_column].sort_values(ascending=ascending).index[0]
        df.loc[best_idx, "is_best"] = True

    return df


__all__ = [
    "run_candidate_simulation",
    "sweep_bess_sizes",
]


def _load_sample_inputs() -> tuple[SimConfig, pd.DataFrame, pd.DataFrame]:
    """Load the packaged sample PV and cycle-model inputs for quick demos."""

    repo_root = Path(__file__).resolve().parent
    pv_df = pd.read_csv(repo_root / "data" / "PV_8760_MW.csv")
    cycle_df = pd.read_excel(repo_root / "data" / "cycle_model.xlsx")

    # One-year run keeps the demo fast while still exercising the simulator.
    base_cfg = SimConfig(years=1)
    return base_cfg, pv_df, cycle_df


def _main_example() -> None:
    """Execute a small sweep using bundled sample data and print the table."""

    base_cfg, pv_df, cycle_df = _load_sample_inputs()

    df = sweep_bess_sizes(
        base_cfg,
        pv_df,
        cycle_df,
        "Auto (infer)",
        power_mw_values=[10.0, 15.0, 20.0],
        duration_h_values=[2.0, 4.0],
    )

    # Show a compact summary so callers know where the best candidate landed.
    display_cols = [
        "power_mw",
        "duration_h",
        "energy_mwh",
        "compliance_pct",
        "total_shortfall_mwh",
        "avg_eq_cycles_per_year",
        "min_soh_total",
        "feasible",
        "is_best",
    ]
    print("\nGrid-search results (sample data):")
    print(df[display_cols].to_string(index=False))


if __name__ == "__main__":
    _main_example()
