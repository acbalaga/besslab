"""Sweep utilities used by both the Streamlit app and tests."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from utils.economics import EconomicInputs, compute_lcoe_lcos_with_augmentation_fallback

if TYPE_CHECKING:
    from app import SimConfig, SimulationOutput, SimulationSummary


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
                        "charge_discharge_ratio": summary.charge_discharge_ratio,
                        "pv_capture_ratio": summary.pv_capture_ratio,
                    }
                )

    return pd.DataFrame(rows)


def _load_default_simulation_hooks() -> tuple[
    Callable[["SimConfig", pd.DataFrame, pd.DataFrame, str, bool], "SimulationOutput"],
    Callable[["SimulationOutput"], "SimulationSummary"],
]:
    """Import default simulation and summary functions lazily to avoid UI side effects."""

    from app import SimConfig, SimulationOutput, SimulationSummary, simulate_project, summarize_simulation

    return simulate_project, summarize_simulation


def run_candidate_simulation(
    base_cfg: "SimConfig",
    pv_df: pd.DataFrame,
    cycle_df: pd.DataFrame,
    dod_override: str,
    power_mw: float,
    duration_h: float,
    simulate_fn: Callable[["SimConfig", pd.DataFrame, pd.DataFrame, str, bool], "SimulationOutput"] | None = None,
    summarize_fn: Callable[["SimulationOutput"], "SimulationSummary"] | None = None,
) -> Tuple["SimulationOutput", "SimulationSummary"]:
    """Call the core engine for a single BESS size.

    The adapter lives in one place so the streamlit app can swap in the real
    engine while tests can supply a stub without touching the grid-search
    logic.
    """

    if simulate_fn is None or summarize_fn is None:
        default_simulate, default_summarize = _load_default_simulation_hooks()
        simulate_fn = simulate_fn or default_simulate
        summarize_fn = summarize_fn or default_summarize

    candidate_energy_mwh = float(power_mw * duration_h)
    cfg_for_run = replace(
        base_cfg,
        initial_power_mw=float(power_mw),
        initial_usable_mwh=candidate_energy_mwh,
    )

    sim_output = simulate_fn(cfg_for_run, pv_df, cycle_df, dod_override, False)
    summary = summarize_fn(sim_output)
    return sim_output, summary


def _compute_candidate_economics(
    sim_output: "SimulationOutput",
    economics_inputs: EconomicInputs,
) -> tuple[float, float]:
    """Return LCOE and discounted-cost NPV for a simulation.

    The helper mirrors the standalone economics module by computing an implied
    augmentation unit rate from the initial usable energy. Augmentation spend
    is converted to USD and discounted inside the LCOE calculation to keep
    sensitivity runs aligned with the main app.
    """

    results = sim_output.results
    if not results:
        return float("nan"), float("nan")

    base_cfg = sim_output.cfg
    augmentation_unit_rate_usd_per_kwh = 0.0
    if base_cfg.initial_usable_mwh > 0 and economics_inputs.capex_musd > 0:
        augmentation_unit_rate_usd_per_kwh = (
            economics_inputs.capex_musd * 1_000_000.0
        ) / (base_cfg.initial_usable_mwh * 1_000.0)

    augmentation_energy_added = list(getattr(sim_output, "augmentation_energy_added_mwh", []))
    if len(augmentation_energy_added) < len(results):
        augmentation_energy_added.extend([0.0] * (len(results) - len(augmentation_energy_added)))
    elif len(augmentation_energy_added) > len(results):
        augmentation_energy_added = augmentation_energy_added[: len(results)]

    augmentation_costs_usd = [
        add_e * 1_000.0 * augmentation_unit_rate_usd_per_kwh for add_e in augmentation_energy_added
    ]

    economics_outputs = compute_lcoe_lcos_with_augmentation_fallback(
        [r.delivered_firm_mwh for r in results],
        [r.bess_to_contract_mwh for r in results],
        economics_inputs,
        augmentation_costs_usd=augmentation_costs_usd,
    )

    return economics_outputs.lcoe_usd_per_mwh, economics_outputs.discounted_costs_usd


def _evaluate_feasibility(
    sim_output: "SimulationOutput",
    summary: "SimulationSummary",
    cfg: "SimConfig",
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
        low_is_better = {
            "total_shortfall_mwh",
            "lcoe_usd_per_mwh",
            "npv_costs_usd",
        }
        return ranking_kpi, ranking_kpi in low_is_better
    return defaults.get(use_case, ("compliance_pct", False))


def sweep_bess_sizes(
    base_cfg: "SimConfig",
    pv_df: pd.DataFrame,
    cycle_df: pd.DataFrame,
    dod_override: str,
    power_mw_values: Iterable[float] | None = None,
    duration_h_values: Iterable[float] | None = None,
    *,
    energy_mwh_values: Iterable[float] | None = None,
    fixed_power_mw: float | None = None,
    economics_inputs: EconomicInputs | None = None,
    use_case: str = "reliability",
    ranking_kpi: str | None = None,
    min_soh: float = 0.6,
    simulate_fn: Callable[["SimConfig", pd.DataFrame, pd.DataFrame, str, bool], "SimulationOutput"] | None = None,
    summarize_fn: Callable[["SimulationOutput"], "SimulationSummary"] | None = None,
) -> pd.DataFrame:
    """Run a simple grid search over BESS sizes.

    The grid is fully enumerated (no heuristics) and returns one row per
    candidate with core KPIs, feasibility markers, economics (optional), and
    the top-ranked option flagged. When ``energy_mwh_values`` is provided the
    sweep fixes power at ``fixed_power_mw`` (defaulting to ``base_cfg``) and
    derives the duration from each energy candidate.
    """

    rows: List[dict[str, float | bool | str]] = []

    # When an explicit energy sweep is requested, collapse the grid to one power value
    # and derive the matching duration for each energy point.
    if energy_mwh_values is not None:
        power_mw_values = [
            float(fixed_power_mw)
            if fixed_power_mw is not None
            else float(base_cfg.initial_power_mw)
        ]
        if power_mw_values[0] <= 0:
            return pd.DataFrame(rows)
        duration_h_values = [float(energy / power_mw_values[0]) for energy in energy_mwh_values]

    power_mw_values = list(power_mw_values or [])
    duration_h_values = list(duration_h_values or [])

    if energy_mwh_values is not None:
        energy_candidates = list(energy_mwh_values)
        if not power_mw_values or not duration_h_values:
            return pd.DataFrame(rows)
        power_mw_values = [power_mw_values[0] for _ in energy_candidates]
        duration_h_values = [float(energy / power_mw_values[0]) for energy in energy_candidates]
    else:
        energy_candidates = [float(p * d) for p in power_mw_values for d in duration_h_values]
        power_mw_values = [p for p in power_mw_values for _ in duration_h_values]
        duration_h_values = duration_h_values * (len(power_mw_values) // len(duration_h_values) if duration_h_values else 0)

    for power_mw, duration_h, candidate_energy_mwh in zip(
        power_mw_values, duration_h_values, energy_candidates
    ):
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

        lcoe = float("nan")
        discounted_costs = float("nan")
        if economics_inputs is not None:
            lcoe, discounted_costs = _compute_candidate_economics(sim_output, economics_inputs)

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
                "lcoe_usd_per_mwh": lcoe,
                "npv_costs_usd": discounted_costs,
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


def _load_sample_inputs() -> tuple["SimConfig", pd.DataFrame, pd.DataFrame]:
    """Load the packaged sample PV and cycle-model inputs for quick demos."""

    from app import SimConfig

    repo_root = Path(__file__).resolve().parent.parent
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


__all__ = [
    "build_soc_windows",
    "generate_values",
    "run_sensitivity_grid",
    "run_candidate_simulation",
    "sweep_bess_sizes",
]


if __name__ == "__main__":
    _main_example()
