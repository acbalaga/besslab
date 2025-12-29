"""Sweep utilities used by both the Streamlit app and tests."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, Tuple, TYPE_CHECKING

import math

import numpy as np
import pandas as pd

from utils.economics import (
    EconomicInputs,
    PriceInputs,
    _compute_npv,
    compute_cash_flows_and_irr,
    compute_lcoe_lcos_with_augmentation_fallback,
)

if TYPE_CHECKING:
    from app import SimConfig, SimulationOutput, SimulationSummary


@dataclass(frozen=True)
class BessEconomicCandidate:
    """Static BESS sizing economics used when simulations are pre-computed.

    The structure mirrors the minimal fields needed to calculate cash flows
    when generation outcomes (e.g., compliance, deficit, surplus) are already
    known. Monetary values are expected in USD. ``deficit_mwh`` may be
    negative to indicate energy that must be procured at the WESM price and is
    treated as a cost in the cash-flow stream.
    """

    energy_mwh: float
    capex_musd: float
    fixed_opex_musd: float
    compliance_mwh: float
    deficit_mwh: float
    surplus_mwh: float


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


def _resolve_energy_prices(price_inputs: PriceInputs) -> tuple[float, float]:
    """Return effective contract and PV prices, honoring a blended override."""

    if price_inputs.blended_price_usd_per_mwh is not None:
        blended_price = float(price_inputs.blended_price_usd_per_mwh)
        return blended_price, blended_price
    return (
        float(price_inputs.contract_price_usd_per_mwh),
        float(price_inputs.pv_market_price_usd_per_mwh),
    )


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


def compute_static_bess_sweep_economics(
    candidates: Sequence[BessEconomicCandidate],
    economics_template: EconomicInputs,
    price_inputs: PriceInputs,
    wesm_price_usd_per_mwh: float,
    *,
    years: int = 1,
) -> pd.DataFrame:
    """Compute NPV/IRR for pre-computed BESS sizing candidates.

    This helper mirrors the economics applied during full simulations but
    operates on pre-aggregated generation outcomes. It assumes the provided
    generation volumes repeat each year; callers can pre-scale the inputs when
    modeling degradation or growth. Deficits are treated as an annual cost
    using the supplied WESM price so negative compliance impacts project value
    explicitly.
    """

    rows: list[dict[str, float]] = []
    inflation_rate = float(economics_template.inflation_rate)
    discount_rate = float(economics_template.discount_rate)
    contract_price_usd_per_mwh, pv_market_price_usd_per_mwh = _resolve_energy_prices(price_inputs)

    for candidate in candidates:
        cash_flows: list[float] = [-max(candidate.capex_musd, 0.0) * 1_000_000.0]
        discounted_npv = cash_flows[0]

        for year_idx in range(1, years + 1):
            inflation_multiplier = (1.0 + inflation_rate) ** (year_idx - 1)

            # Positive compliance/surplus yield revenue; deficits represent market purchases
            # at the assumed WESM price. Negative values are treated as a cost to avoid
            # overstating project value when the profile misses contract energy.
            contract_revenue = max(candidate.compliance_mwh, 0.0) * contract_price_usd_per_mwh
            surplus_revenue = max(candidate.surplus_mwh, 0.0) * pv_market_price_usd_per_mwh
            deficit_penalty = abs(candidate.deficit_mwh) * wesm_price_usd_per_mwh

            annual_revenue = contract_revenue + surplus_revenue - deficit_penalty
            if price_inputs.escalate_with_inflation:
                annual_revenue *= inflation_multiplier

            annual_opex_usd = max(candidate.fixed_opex_musd, 0.0) * 1_000_000.0 * inflation_multiplier
            net_cash_flow = annual_revenue - annual_opex_usd

            cash_flows.append(net_cash_flow)
            discounted_npv += net_cash_flow / ((1.0 + discount_rate) ** year_idx)

        irr_pct = _solve_irr_pct(cash_flows)
        rows.append(
            {
                "energy_mwh": float(candidate.energy_mwh),
                "npv_usd": float(discounted_npv),
                "irr_pct": float(irr_pct),
                "capex_musd": float(candidate.capex_musd),
                "fixed_opex_musd": float(candidate.fixed_opex_musd),
                "compliance_mwh": float(candidate.compliance_mwh),
                "deficit_mwh": float(candidate.deficit_mwh),
                "surplus_mwh": float(candidate.surplus_mwh),
            }
        )

    return pd.DataFrame(rows)


def _compute_candidate_economics(
    sim_output: "SimulationOutput",
    economics_inputs: EconomicInputs,
    price_inputs: PriceInputs | None = None,
    base_initial_energy_mwh: float | None = None,
) -> tuple[float, float, float, float]:
    """Return LCOE, discounted-cost NPV, implied IRR, and net-project NPV.

    The helper mirrors the standalone economics module by computing an implied
    augmentation unit rate from the initial usable energy. Augmentation spend
    is converted to USD and discounted inside the LCOE calculation to keep
    sensitivity runs aligned with the main app. An IRR is then derived from the
    same cash-flow stream: when price assumptions are supplied, IRR and NPV use
    the revenue-based cash flows; otherwise, they fall back to LCOE-based flows
    that normalize economics against delivered energy.
    """

    results = sim_output.results
    if not results:
        return float("nan"), float("nan"), float("nan"), float("nan")

    base_cfg = sim_output.cfg
    size_scale = 1.0
    if base_initial_energy_mwh and base_initial_energy_mwh > 0:
        size_scale = max(sim_output.cfg.initial_usable_mwh / base_initial_energy_mwh, 0.0)

    scaled_economics = EconomicInputs(
        capex_musd=economics_inputs.capex_musd * size_scale,
        fixed_opex_pct_of_capex=economics_inputs.fixed_opex_pct_of_capex,
        fixed_opex_musd=economics_inputs.fixed_opex_musd * size_scale,
        inflation_rate=economics_inputs.inflation_rate,
        discount_rate=economics_inputs.discount_rate,
        variable_opex_usd_per_mwh=economics_inputs.variable_opex_usd_per_mwh,
        variable_opex_schedule_usd=
            tuple(v * size_scale for v in economics_inputs.variable_opex_schedule_usd)
            if economics_inputs.variable_opex_schedule_usd is not None
            else None,
        periodic_variable_opex_usd=
            economics_inputs.periodic_variable_opex_usd * size_scale
            if economics_inputs.periodic_variable_opex_usd is not None
            else None,
        periodic_variable_opex_interval_years=economics_inputs.periodic_variable_opex_interval_years,
    )

    augmentation_unit_rate_usd_per_kwh = 0.0
    if base_cfg.initial_usable_mwh > 0 and scaled_economics.capex_musd > 0:
        augmentation_unit_rate_usd_per_kwh = (
            scaled_economics.capex_musd * 1_000_000.0
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
        scaled_economics,
        augmentation_costs_usd=augmentation_costs_usd,
    )

    irr_pct = float("nan")
    npv_usd = -economics_outputs.discounted_costs_usd
    if price_inputs is not None:
        cash_outputs = compute_cash_flows_and_irr(
            [r.delivered_firm_mwh for r in results],
            [r.bess_to_contract_mwh for r in results],
            [r.pv_curtailed_mwh for r in results],
            scaled_economics,
            price_inputs,
            annual_shortfall_mwh=[r.shortfall_mwh for r in results],
            augmentation_costs_usd=augmentation_costs_usd,
        )
        irr_pct = cash_outputs.irr_pct
        npv_usd = cash_outputs.npv_usd
    else:
        # Use LCOE as an implied tariff to create a revenue stream that balances costs.
        # This keeps the IRR interpretable without requiring a separate price input.
        capex_usd = scaled_economics.capex_musd * 1_000_000.0
        inflation_rate = scaled_economics.inflation_rate
        fixed_opex_from_capex = scaled_economics.capex_musd * (
            scaled_economics.fixed_opex_pct_of_capex / 100.0
        )

        if capex_usd > 0 and math.isfinite(economics_outputs.lcoe_usd_per_mwh):
            cash_flows: List[float] = [-capex_usd]
            for year_idx, annual_result in enumerate(results, start=1):
                inflation_multiplier = (1.0 + inflation_rate) ** (year_idx - 1)
                annual_fixed_opex_usd = (fixed_opex_from_capex + scaled_economics.fixed_opex_musd) * 1_000_000
                annual_fixed_opex_usd *= inflation_multiplier
                augmentation_cost = (
                    float(augmentation_costs_usd[year_idx - 1])
                    if year_idx - 1 < len(augmentation_costs_usd)
                    else 0.0
                )
                # Escalate the breakeven tariff with inflation so larger (or more productive)
                # designs reflect higher nominal cash inflows when computing IRR.
                revenue = (
                    economics_outputs.lcoe_usd_per_mwh
                    * float(annual_result.delivered_firm_mwh)
                    * inflation_multiplier
                )
                cash_flows.append(revenue - annual_fixed_opex_usd - augmentation_cost)
            irr_pct = _solve_irr_pct(cash_flows)
            npv_usd = _compute_npv(cash_flows, scaled_economics.discount_rate)

    return (
        economics_outputs.lcoe_usd_per_mwh,
        economics_outputs.discounted_costs_usd,
        irr_pct,
        npv_usd,
    )


def _solve_irr_pct(cash_flows: Sequence[float], max_iterations: int = 100) -> float:
    """Compute IRR (%) using a robust bisection search.

    ``numpy.irr`` was removed in NumPy 2.0, and numpy_financial may not be
    available in all environments. This helper performs a simple bisection
    search for a rate that drives NPV to zero. It returns NaN when cash flows do
    not change sign or when a root cannot be located within the search bounds.
    """

    if not any(cf < 0 for cf in cash_flows) or not any(cf > 0 for cf in cash_flows):
        return float("nan")

    def npv(rate: float) -> float:
        return sum(cf / ((1.0 + rate) ** idx) for idx, cf in enumerate(cash_flows))

    low = -0.99
    high = 1.0
    npv_low = npv(low)
    npv_high = npv(high)

    # Expand the upper bound until the NPV changes sign or the range becomes unreasonable.
    while npv_low * npv_high > 0 and high < 1000:
        high *= 2.0
        npv_high = npv(high)

    if npv_low * npv_high > 0:
        return float("nan")

    for _ in range(max_iterations):
        mid = (low + high) / 2.0
        npv_mid = npv(mid)
        if abs(npv_mid) < 1e-6:
            return mid * 100.0
        if npv_low * npv_mid < 0:
            high = mid
            npv_high = npv_mid
        else:
            low = mid
            npv_low = npv_mid

    return mid * 100.0 if math.isfinite(mid) else float("nan")


def _evaluate_feasibility(
    sim_output: "SimulationOutput",
    summary: "SimulationSummary",
    cfg: "SimConfig",
    min_soh: float,
) -> Tuple[float, float, bool, bool, bool, float, float]:
    """Return simple feasibility markers derived from simulation results.

    The helper also computes numeric margins so downstream consumers can see
    how far a candidate is from the cycle ceiling or minimum SOH threshold.
    Positive ``cycles_over_cap`` means the candidate exceeded the annual
    cycle limit, while negative values indicate remaining headroom. A
    positive ``soh_margin`` reflects remaining SOH above the minimum.
    """

    max_eq_cycles = max((r.eq_cycles for r in sim_output.results), default=float("nan"))
    min_soh_total = min((r.soh_total for r in sim_output.results), default=float("nan"))

    cycles_allowed_per_year = cfg.max_cycles_per_day_cap * 365.0
    cycles_over_cap = float("nan")
    if math.isfinite(max_eq_cycles) and cycles_allowed_per_year > 0:
        cycles_over_cap = max_eq_cycles - cycles_allowed_per_year

    soh_margin = float("nan")
    if math.isfinite(min_soh_total):
        soh_margin = min_soh_total - min_soh

    cycle_limit_hit = bool(math.isfinite(cycles_over_cap) and cycles_over_cap > 0)
    soh_below_min = bool(math.isfinite(soh_margin) and soh_margin < 0)
    feasible = (not cycle_limit_hit) and (not soh_below_min)
    return (
        max_eq_cycles,
        min_soh_total,
        cycle_limit_hit,
        soh_below_min,
        feasible,
        cycles_over_cap,
        soh_margin,
    )


def _validate_positive_values(values: Iterable[float], label: str) -> list[float]:
    """Return validated float values, ensuring they are positive and finite."""

    parsed = [float(v) for v in values]
    if not parsed:
        raise ValueError(f"{label} must include at least one value.")

    invalid_values = [v for v in parsed if not math.isfinite(v) or v <= 0]
    if invalid_values:
        raise ValueError(
            f"{label} must be positive and finite; received {invalid_values}."
        )

    return parsed


def _prepare_sweep_candidates(
    base_cfg: "SimConfig",
    power_mw_values: Iterable[float] | None,
    duration_h_values: Iterable[float] | None,
    energy_mwh_values: Iterable[float] | None,
    fixed_power_mw: float | None,
) -> list[tuple[float, float, float]]:
    """Validate sweep inputs and return candidate tuples.

    Each tuple contains ``(power_mw, duration_h, energy_mwh)``. Validation is
    performed up front so calling code can fail fast with clear messaging
    before any simulation work begins.
    """

    if energy_mwh_values is not None:
        energy_candidates = _validate_positive_values(energy_mwh_values, "energy_mwh_values")
        resolved_power = float(
            fixed_power_mw if fixed_power_mw is not None else base_cfg.initial_power_mw
        )
        if not math.isfinite(resolved_power) or resolved_power <= 0:
            raise ValueError(
                "A positive fixed_power_mw or base_cfg.initial_power_mw is required when sweeping energy_mwh_values."
            )

        candidates: list[tuple[float, float, float]] = []
        for energy in energy_candidates:
            duration = energy / resolved_power
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(
                    "Derived duration must be positive and finite for all energy_mwh_values."
                )
            candidates.append((resolved_power, duration, energy))
        return candidates

    power_candidates = _validate_positive_values(power_mw_values or [], "power_mw_values")
    duration_candidates = _validate_positive_values(duration_h_values or [], "duration_h_values")

    candidates = []
    for power_mw in power_candidates:
        for duration_h in duration_candidates:
            energy_mwh = power_mw * duration_h
            if not math.isfinite(energy_mwh) or energy_mwh <= 0:
                raise ValueError("Computed energy_mwh must be positive and finite for all power/duration combinations.")
            candidates.append((power_mw, duration_h, energy_mwh))
    return candidates


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
    price_inputs: PriceInputs | None = None,
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
    base_initial_energy_mwh = float(base_cfg.initial_usable_mwh)
    candidates = _prepare_sweep_candidates(
        base_cfg,
        power_mw_values,
        duration_h_values,
        energy_mwh_values,
        fixed_power_mw,
    )

    for power_mw, duration_h, candidate_energy_mwh in candidates:
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
            cycles_over_cap,
            soh_margin,
        ) = _evaluate_feasibility(sim_output, summary, sim_output.cfg, min_soh)

        lcoe = float("nan")
        discounted_costs = float("nan")
        irr_pct = float("nan")
        npv_usd = float("nan")
        if economics_inputs is not None:
            lcoe, discounted_costs, irr_pct, npv_usd = _compute_candidate_economics(
                sim_output,
                economics_inputs,
                price_inputs,
                base_initial_energy_mwh=base_initial_energy_mwh,
            )

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
                "cycles_over_cap": cycles_over_cap,
                "soh_margin": soh_margin,
                "lcoe_usd_per_mwh": lcoe,
                "npv_costs_usd": discounted_costs,
                "irr_pct": irr_pct,
                "npv_usd": npv_usd,
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
    "BessEconomicCandidate",
    "build_soc_windows",
    "generate_values",
    "run_sensitivity_grid",
    "run_candidate_simulation",
    "compute_static_bess_sweep_economics",
    "sweep_bess_sizes",
]


if __name__ == "__main__":
    _main_example()
