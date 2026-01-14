"""Sweep utilities used by both the Streamlit app and tests."""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import math

import numpy as np
import pandas as pd

from utils.economics import (
    EconomicInputs,
    PriceInputs,
    _compute_npv,
    aggregate_wesm_profile_to_annual,
    compute_cash_flows_and_irr,
    compute_lcoe_lcos_with_augmentation_fallback,
)

if TYPE_CHECKING:
    from services.simulation_core import SimConfig, SimulationOutput, SimulationSummary


@dataclass(frozen=True)
class BessEconomicCandidate:
    """Static BESS sizing economics used when simulations are pre-computed.

    The structure mirrors the minimal fields needed to calculate cash flows
    when generation outcomes (e.g., compliance, deficit, surplus) are already
    known. Monetary values are expected in USD. ``deficit_mwh`` may be
    negative to indicate energy that must be procured at the WESM deficit price
    and is treated as a cost in the cash-flow stream.
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


def _normalize_price_scenarios(
    price_inputs: PriceInputs | Sequence[PriceInputs] | None,
    price_scenario_names: Sequence[str] | None = None,
) -> list[tuple[str | None, PriceInputs | None]]:
    """Return a list of (scenario_name, PriceInputs) tuples for downstream use.

    When a single ``PriceInputs`` is supplied the scenario name is omitted to
    preserve the legacy output shape. Multiple scenarios attach a ``str`` name
    (either user-provided or auto-generated) so the caller can pivot results
    without re-running the physical simulation.
    """

    if price_inputs is None:
        if price_scenario_names:
            raise ValueError("price_scenario_names cannot be used when price_inputs is None.")
        return [(None, None)]

    if isinstance(price_inputs, PriceInputs):
        if price_scenario_names and len(price_scenario_names) != 1:
            raise ValueError("price_scenario_names must have length 1 when a single PriceInputs is provided.")
        scenario_name = price_scenario_names[0] if price_scenario_names else None
        return [(scenario_name, price_inputs)]

    inputs_list = list(price_inputs)
    if not inputs_list:
        raise ValueError("price_inputs must include at least one PriceInputs when provided as a sequence.")

    if price_scenario_names is not None and len(price_scenario_names) != len(inputs_list):
        raise ValueError("price_scenario_names must align with the number of PriceInputs provided.")

    scenario_names = (
        list(price_scenario_names)
        if price_scenario_names is not None
        else [f"scenario_{idx + 1}" for idx in range(len(inputs_list))]
    )
    return list(zip(scenario_names, inputs_list))


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
    Callable[..., "SimulationOutput"],
    Callable[["SimulationOutput"], "SimulationSummary"],
]:
    """Import default simulation and summary functions lazily to avoid UI side effects."""

    from services.simulation_core import SimulationOutput, SimulationSummary, simulate_project, summarize_simulation

    return simulate_project, summarize_simulation


def _collect_optional_simulation_kwargs(
    simulate_fn: Callable[..., Any],
    seed: int | None,
    deterministic: bool | None,
) -> dict[str, Any]:
    """Return optional simulation kwargs that are accepted by ``simulate_fn``.

    The helper inspects the function signature to avoid passing unexpected keyword
    arguments to simulators that do not support seeding or deterministic behavior.
    """

    try:
        parameters = inspect.signature(simulate_fn).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_var_kw = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())

    def _supports_kwarg(name: str) -> bool:
        param = parameters.get(name)
        return accepts_var_kw or (
            param is not None
            and param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        )

    kwargs: dict[str, Any] = {}
    if seed is not None and _supports_kwarg("seed"):
        kwargs["seed"] = seed
    if deterministic is not None and _supports_kwarg("deterministic"):
        kwargs["deterministic"] = deterministic
    return kwargs


def run_candidate_simulation(
    base_cfg: "SimConfig",
    pv_df: pd.DataFrame,
    cycle_df: pd.DataFrame,
    dod_override: str,
    power_mw: float,
    duration_h: float,
    simulate_fn: Callable[..., "SimulationOutput"] | None = None,
    summarize_fn: Callable[["SimulationOutput"], "SimulationSummary"] | None = None,
    need_logs: bool = False,
    seed: int | None = None,
    deterministic: bool | None = None,
) -> Tuple["SimulationOutput", "SimulationSummary"]:
    """Call the core engine for a single BESS size.

    The adapter lives in one place so the streamlit app can swap in the real
    engine while tests can supply a stub without touching the grid-search
    logic. Optional ``seed`` and ``deterministic`` flags are forwarded to
    simulators that accept them so callers can enforce reproducible runs.
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

    simulate_kwargs = _collect_optional_simulation_kwargs(simulate_fn, seed, deterministic)
    sim_output = simulate_fn(
        cfg_for_run, pv_df, cycle_df, dod_override, need_logs, **simulate_kwargs
    )
    summary = summarize_fn(sim_output)
    return sim_output, summary


def _build_hourly_summary_df(logs: Any) -> pd.DataFrame:
    """Build an hourly summary table for WESM profile aggregation."""

    hour_index = np.arange(len(logs.hod), dtype=int)
    data: dict[str, Any] = {
        "hour_index": hour_index,
        "hod": logs.hod,
        "pv_mw": logs.pv_mw,
        "pv_to_contract_mw": logs.pv_to_contract_mw,
        "bess_to_contract_mw": logs.bess_to_contract_mw,
        "delivered_mw": logs.delivered_mw,
        "shortfall_mw": logs.shortfall_mw,
        "charge_mw": logs.charge_mw,
        "discharge_mw": logs.discharge_mw,
        "soc_mwh": logs.soc_mwh,
    }
    if logs.timestamp is not None:
        data["timestamp"] = pd.to_datetime(logs.timestamp)
    df = pd.DataFrame(data)
    df["pv_surplus_mw"] = np.maximum(
        df["pv_mw"] - df["pv_to_contract_mw"] - df["charge_mw"],
        0.0,
    )
    return df


def compute_static_bess_sweep_economics(
    candidates: Sequence[BessEconomicCandidate],
    economics_template: EconomicInputs,
    price_inputs: PriceInputs,
    *,
    years: int = 1,
) -> pd.DataFrame:
    """Compute NPV/IRR for pre-computed BESS sizing candidates.

    This helper mirrors the economics applied during full simulations by
    delegating to :func:`utils.economics.compute_cash_flows_and_irr`. The
    provided generation volumes are assumed to repeat each year; callers can
    pre-scale the inputs when modeling degradation or growth. WESM pricing is
    not supported here because it requires hourly profiles.
    """

    if price_inputs.apply_wesm_to_shortfall or price_inputs.sell_to_wesm:
        raise ValueError(
            "Static sweep economics require a WESM profile; disable WESM pricing for static candidates."
        )

    rows: list[dict[str, float]] = []

    for candidate in candidates:
        economics_inputs = replace(
            economics_template,
            capex_musd=float(candidate.capex_musd),
            fixed_opex_musd=float(candidate.fixed_opex_musd),
        )

        annual_compliance = max(candidate.compliance_mwh, 0.0)
        annual_surplus = max(candidate.surplus_mwh, 0.0)
        annual_shortfall = max(-candidate.deficit_mwh, 0.0)

        cash_outputs = compute_cash_flows_and_irr(
            [annual_compliance for _ in range(years)],
            [annual_compliance for _ in range(years)],
            [annual_surplus for _ in range(years)],
            economics_inputs,
            price_inputs,
            annual_pv_delivered_mwh=[0.0 for _ in range(years)],
            annual_shortfall_mwh=[annual_shortfall for _ in range(years)],
        )
        rows.append(
            {
                "energy_mwh": float(candidate.energy_mwh),
                "npv_usd": float(cash_outputs.npv_usd),
                "irr_pct": float(cash_outputs.irr_pct),
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
    *,
    augmentation_energy_added_mwh: Sequence[float] | None = None,
    delivered_firm_mwh: Sequence[float] | None = None,
    bess_to_contract_mwh: Sequence[float] | None = None,
    pv_curtailed_mwh: Sequence[float] | None = None,
    shortfall_mwh: Sequence[float] | None = None,
    annual_wesm_shortfall_cost_usd: Sequence[float] | None = None,
    annual_wesm_surplus_revenue_usd: Sequence[float] | None = None,
) -> tuple[float, float, float, float]:
    """Return LCOE, discounted-cost NPV, implied IRR, and net-project NPV.

    The helper mirrors the standalone economics module by computing an implied
    augmentation unit rate from the initial usable energy. Augmentation spend
    is converted to USD and discounted inside the LCOE calculation to keep
    sensitivity runs aligned with the main app. An IRR is then derived from the
    same cash-flow stream: when price assumptions are supplied, IRR and NPV use
    the revenue-based cash flows; otherwise, they fall back to LCOE-based flows
    that normalize economics against delivered energy.

    Parameters
    ----------
    augmentation_energy_added_mwh
        Optional per-year augmentation energy additions (AC-side, MWh). When
        provided, the sequence must align with project years and replaces the
        schedule embedded in ``sim_output``. Values are scaled to kWh before
        applying the derived augmentation unit rate (USD/kWh).
    delivered_firm_mwh
        Optional per-year delivered firm energy (AC-side, MWh). Overrides the
        series from ``sim_output.results`` to support candidate-specific
        degradation assumptions.
    bess_to_contract_mwh
        Optional per-year BESS-to-contract energy (AC-side, MWh). Overrides
        the series from ``sim_output.results`` for cash-flow calculations.
    pv_curtailed_mwh
        Optional per-year curtailed PV (AC-side, MWh). Overrides the series
        from ``sim_output.results`` for cash-flow calculations.
    shortfall_mwh
        Optional per-year contract shortfall (AC-side, MWh). Overrides the
        series from ``sim_output.results`` for cash-flow calculations.
    """

    results = sim_output.results
    if not results:
        return float("nan"), float("nan"), float("nan"), float("nan")

    base_cfg = sim_output.cfg
    size_scale = 1.0
    if base_initial_energy_mwh and base_initial_energy_mwh > 0:
        size_scale = max(sim_output.cfg.initial_usable_mwh / base_initial_energy_mwh, 0.0)

    scaled_variable_opex_schedule_usd = (
        tuple(v * size_scale for v in economics_inputs.variable_opex_schedule_usd)
        if economics_inputs.variable_opex_schedule_usd is not None
        else None
    )
    scaled_periodic_variable_opex_usd = (
        economics_inputs.periodic_variable_opex_usd * size_scale
        if economics_inputs.periodic_variable_opex_usd is not None
        else None
    )
    scaled_economics = replace(
        economics_inputs,
        capex_musd=economics_inputs.capex_musd * size_scale,
        fixed_opex_musd=economics_inputs.fixed_opex_musd * size_scale,
        variable_opex_schedule_usd=scaled_variable_opex_schedule_usd,
        periodic_variable_opex_usd=scaled_periodic_variable_opex_usd,
    )

    def _override_or_results(
        override: Sequence[float] | None, accessor: Callable[[Any], float], label: str
    ) -> list[float]:
        values = [float(accessor(r)) for r in results] if override is None else [float(v) for v in override]
        if len(values) != len(results):
            raise ValueError(f"{label} must align with the number of project years ({len(results)}).")
        return values

    augmentation_unit_rate_usd_per_kwh = 0.0
    if base_cfg.initial_usable_mwh > 0 and scaled_economics.capex_musd > 0:
        augmentation_unit_rate_usd_per_kwh = (
            scaled_economics.capex_musd * 1_000_000.0
        ) / (base_cfg.initial_usable_mwh * 1_000.0)

    if augmentation_energy_added_mwh is not None:
        augmentation_energy_added = [float(v) for v in augmentation_energy_added_mwh]
        if len(augmentation_energy_added) != len(results):
            raise ValueError("augmentation_energy_added_mwh must align with the number of project years.")
    else:
        augmentation_energy_added = list(getattr(sim_output, "augmentation_energy_added_mwh", []))
        if len(augmentation_energy_added) < len(results):
            augmentation_energy_added.extend([0.0] * (len(results) - len(augmentation_energy_added)))
        elif len(augmentation_energy_added) > len(results):
            augmentation_energy_added = augmentation_energy_added[: len(results)]

    augmentation_costs_usd = [
        add_e * 1_000.0 * augmentation_unit_rate_usd_per_kwh for add_e in augmentation_energy_added
    ]

    delivered_firm_series = _override_or_results(
        delivered_firm_mwh, lambda r: r.delivered_firm_mwh, "delivered_firm_mwh"
    )
    bess_to_contract_series = _override_or_results(
        bess_to_contract_mwh, lambda r: r.bess_to_contract_mwh, "bess_to_contract_mwh"
    )
    pv_curtailed_series = _override_or_results(
        pv_curtailed_mwh, lambda r: r.pv_curtailed_mwh, "pv_curtailed_mwh"
    )
    shortfall_series = _override_or_results(
        shortfall_mwh, lambda r: r.shortfall_mwh, "shortfall_mwh"
    )
    pv_delivered_series = [
        float(delivered) - float(bess)
        for delivered, bess in zip(delivered_firm_series, bess_to_contract_series)
    ]

    economics_outputs = compute_lcoe_lcos_with_augmentation_fallback(
        delivered_firm_series,
        bess_to_contract_series,
        scaled_economics,
        augmentation_costs_usd=augmentation_costs_usd,
    )

    irr_pct = float("nan")
    npv_usd = -economics_outputs.discounted_costs_usd
    if price_inputs is not None:
        cash_outputs = compute_cash_flows_and_irr(
            delivered_firm_series,
            bess_to_contract_series,
            pv_curtailed_series,
            scaled_economics,
            price_inputs,
            annual_pv_delivered_mwh=pv_delivered_series,
            annual_shortfall_mwh=shortfall_series,
            annual_wesm_shortfall_cost_usd=annual_wesm_shortfall_cost_usd,
            annual_wesm_surplus_revenue_usd=annual_wesm_surplus_revenue_usd,
            augmentation_costs_usd=augmentation_costs_usd,
        )
        irr_pct = cash_outputs.irr_pct
        npv_usd = cash_outputs.npv_usd
    else:
        # Use LCOE as an implied tariff to create a revenue stream that balances costs.
        # This keeps the IRR interpretable without requiring a separate price input.
        total_capex_musd = scaled_economics.capex_musd + scaled_economics.pv_capex_musd
        capex_usd = total_capex_musd * 1_000_000.0
        inflation_rate = scaled_economics.inflation_rate
        fixed_opex_from_capex = total_capex_musd * (
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


def _chunked(
    sequence: Sequence[Tuple[float, float, float]],
    chunk_size: int,
) -> list[list[tuple[float, float, float]]]:
    """Return a list of evenly sized chunks while preserving order."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    return [
        list(sequence[idx: idx + chunk_size]) for idx in range(0, len(sequence), chunk_size)
    ]


def _emit_progress_rows(
    new_rows: Sequence[dict[str, float | bool | str]],
    progress_callback: Callable[[pd.DataFrame], None] | None,
) -> None:
    """Send incremental sweep rows to a callback while keeping failures non-blocking."""

    if progress_callback is None or not new_rows:
        return

    try:
        progress_callback(pd.DataFrame(list(new_rows)))
    except Exception:
        logging.getLogger(__name__).exception("Progress callback failed; continuing sweep.")


def _resolve_ranking_column(use_case: str, ranking_kpi: str | None) -> Tuple[str, bool, str, bool]:
    """Map a use case/KPI to a column and sort order with a fallback.

    The returned tuple is ``(column_name, ascending, fallback_column, fallback_ascending)``
    so callers can hand it directly to ``DataFrame.sort_values`` while retaining a
    sensible default if the requested KPI is missing or empty.
    """

    defaults = {
        "reliability": ("compliance_pct", False),
        "energy": ("total_project_generation_mwh", False),
        "shortfall": ("total_shortfall_mwh", True),
    }
    fallback_column, fallback_ascending = defaults.get(use_case, ("compliance_pct", False))

    if ranking_kpi:
        low_is_better = {
            "total_shortfall_mwh",
            "lcoe_usd_per_mwh",
            "npv_costs_usd",
            "capex_per_kw_usd",
        }
        return ranking_kpi, ranking_kpi in low_is_better, fallback_column, fallback_ascending
    return fallback_column, fallback_ascending, fallback_column, fallback_ascending


def _evaluate_candidate_row(
    base_cfg: "SimConfig",
    pv_df: pd.DataFrame,
    cycle_df: pd.DataFrame,
    dod_override: str,
    power_mw: float,
    duration_h: float,
    candidate_energy_mwh: float,
    min_soh: float,
    economics_inputs: EconomicInputs | None,
    price_scenarios: Sequence[tuple[str | None, PriceInputs | None]],
    base_initial_energy_mwh: float,
    wesm_profile_df: pd.DataFrame | None,
    wesm_step_hours: float | None,
    simulate_fn: Callable[..., "SimulationOutput"] | None,
    summarize_fn: Callable[["SimulationOutput"], "SimulationSummary"] | None,
    min_compliance_pct: float | None,
    max_shortfall_mwh: float | None,
) -> list[dict[str, float | bool | str]]:
    """Run a single candidate simulation and compute derived KPIs for all price scenarios."""

    wesm_requested = any(
        scenario_price_inputs is not None
        and (scenario_price_inputs.apply_wesm_to_shortfall or scenario_price_inputs.sell_to_wesm)
        for _, scenario_price_inputs in price_scenarios
    )
    if wesm_requested and wesm_profile_df is None:
        raise ValueError("WESM pricing requires an hourly WESM profile.")

    needs_wesm_profile = wesm_profile_df is not None and wesm_requested

    sim_output, summary = run_candidate_simulation(
        base_cfg,
        pv_df,
        cycle_df,
        dod_override,
        power_mw,
        duration_h,
        simulate_fn=simulate_fn,
        summarize_fn=summarize_fn,
        need_logs=needs_wesm_profile,
    )

    annual_wesm_shortfall_cost_usd: list[float] | None = None
    annual_wesm_surplus_revenue_usd: list[float] | None = None
    if needs_wesm_profile and sim_output.hourly_logs_by_year:
        hourly_summary_by_year = {
            year_index: _build_hourly_summary_df(logs)
            for year_index, logs in sim_output.hourly_logs_by_year.items()
        }
        try:
            (
                annual_wesm_shortfall_cost_usd,
                annual_wesm_surplus_revenue_usd,
            ) = aggregate_wesm_profile_to_annual(
                hourly_summary_by_year,
                wesm_profile_df,
                step_hours=wesm_step_hours or getattr(sim_output.cfg, "step_hours", 1.0),
                apply_inflation=False,
                inflation_rate=economics_inputs.inflation_rate if economics_inputs else 0.0,
            )
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "WESM profile aggregation failed; WESM economics will error if enabled. (%s)",
                exc,
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

    # Apply optional pruning thresholds to avoid spending economics cycles on unpromising designs.
    meets_compliance = True
    if min_compliance_pct is not None:
        meets_compliance = math.isfinite(summary.compliance) and summary.compliance >= min_compliance_pct

    within_shortfall = True
    if max_shortfall_mwh is not None:
        within_shortfall = math.isfinite(summary.total_shortfall_mwh) and summary.total_shortfall_mwh <= max_shortfall_mwh

    status = "evaluated"
    if cycle_limit_hit:
        status = "cycle_limit"
    elif soh_below_min:
        status = "soh_below_min"
    elif not meets_compliance:
        status = "below_min_compliance"
    elif not within_shortfall:
        status = "exceeds_shortfall"

    include_scenario_column = len(price_scenarios) > 1 or any(name is not None for name, _ in price_scenarios)
    rows: list[dict[str, float | bool | str]] = []
    for scenario_name, scenario_price_inputs in price_scenarios:
        lcoe = float("nan")
        discounted_costs = float("nan")
        irr_pct = float("nan")
        npv_usd = float("nan")
        npv_per_mwh_usd = float("nan")
        capex_per_kw_usd = float("nan")
        if economics_inputs is not None and status == "evaluated":
            use_wesm_profile = (
                scenario_price_inputs is not None
                and (
                    scenario_price_inputs.apply_wesm_to_shortfall
                    or scenario_price_inputs.sell_to_wesm
                )
            )
            lcoe, discounted_costs, irr_pct, npv_usd = _compute_candidate_economics(
                sim_output,
                economics_inputs,
                scenario_price_inputs,
                base_initial_energy_mwh=base_initial_energy_mwh,
                annual_wesm_shortfall_cost_usd=(
                    annual_wesm_shortfall_cost_usd if use_wesm_profile else None
                ),
                annual_wesm_surplus_revenue_usd=(
                    annual_wesm_surplus_revenue_usd if use_wesm_profile else None
                ),
            )
            size_scale = (
                candidate_energy_mwh / base_initial_energy_mwh
                if base_initial_energy_mwh > 0
                else 1.0
            )
            # CAPEX is normalized to installed power (USD/kW) to match common quoting units.
            total_capex_musd = economics_inputs.capex_musd * size_scale + economics_inputs.pv_capex_musd
            capex_usd = total_capex_musd * 1_000_000.0
            if power_mw > 0:
                capex_per_kw_usd = capex_usd / (power_mw * 1_000.0)
            # NPV is normalized to usable BESS energy (USD/MWh) to make rankings resilient to duration changes.
            if math.isfinite(npv_usd) and candidate_energy_mwh > 0:
                npv_per_mwh_usd = npv_usd / candidate_energy_mwh

        row: dict[str, float | bool | str] = {
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
            "soc_floor": sim_output.cfg.soc_floor,
            "soc_ceiling": sim_output.cfg.soc_ceiling,
            "rte_roundtrip": sim_output.cfg.rte_roundtrip,
            "dod_override": dod_override,
            "cycle_limit_hit": cycle_limit_hit,
            "soh_below_min": soh_below_min,
            "feasible": feasible,
            "cycles_over_cap": cycles_over_cap,
            "soh_margin": soh_margin,
            "lcoe_usd_per_mwh": lcoe,
            "npv_costs_usd": discounted_costs,
            "irr_pct": irr_pct,
            "npv_usd": npv_usd,
            "npv_per_mwh_usd": npv_per_mwh_usd,
            "capex_per_kw_usd": capex_per_kw_usd,
            "status": status,
        }
        if include_scenario_column:
            row["price_scenario"] = scenario_name
        rows.append(row)

    return rows


def _evaluate_candidate_tuple(
    candidate: tuple[float, float, float],
    base_cfg: "SimConfig",
    pv_df: pd.DataFrame,
    cycle_df: pd.DataFrame,
    dod_override: str,
    min_soh: float,
    economics_inputs: EconomicInputs | None,
    price_scenarios: Sequence[tuple[str | None, PriceInputs | None]],
    base_initial_energy_mwh: float,
    wesm_profile_df: pd.DataFrame | None,
    wesm_step_hours: float | None,
    simulate_fn: Callable[..., "SimulationOutput"] | None,
    summarize_fn: Callable[["SimulationOutput"], "SimulationSummary"] | None,
    min_compliance_pct: float | None,
    max_shortfall_mwh: float | None,
) -> list[dict[str, float | bool | str]]:
    """Adapter for executor.map that unwraps the candidate tuple."""

    power_mw, duration_h, candidate_energy_mwh = candidate
    return _evaluate_candidate_row(
        base_cfg,
        pv_df,
        cycle_df,
        dod_override,
        power_mw,
        duration_h,
        candidate_energy_mwh,
        min_soh,
        economics_inputs,
        price_scenarios,
        base_initial_energy_mwh,
        wesm_profile_df,
        wesm_step_hours,
        simulate_fn,
        summarize_fn,
        min_compliance_pct,
        max_shortfall_mwh,
    )


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
    price_inputs: PriceInputs | Sequence[PriceInputs] | None = None,
    price_scenario_names: Sequence[str] | None = None,
    wesm_profile_df: pd.DataFrame | None = None,
    wesm_step_hours: float | None = None,
    use_case: str = "reliability",
    ranking_kpi: str | None = None,
    min_soh: float = 0.6,
    min_compliance_pct: float | None = None,
    max_shortfall_mwh: float | None = None,
    simulate_fn: Callable[..., "SimulationOutput"] | None = None,
    summarize_fn: Callable[["SimulationOutput"], "SimulationSummary"] | None = None,
    max_candidates: int | None = None,
    on_exceed: str = "raise",
    batch_size: int | None = None,
    concurrency: str | None = None,
    max_workers: int | None = None,
    progress_callback: Callable[[pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    """Run a simple grid search over BESS sizes.

    The grid is fully enumerated (no heuristics) and returns one row per
    candidate with core KPIs, feasibility markers, economics (optional), and
    the top-ranked option flagged. When ``energy_mwh_values`` is provided the
    sweep fixes power at ``fixed_power_mw`` (defaulting to ``base_cfg``) and
    derives the duration from each energy candidate.

    Parameters
    ----------
    price_inputs
        Optional pricing assumptions. Supply a single ``PriceInputs`` for legacy
        behavior or a sequence to compute economics across multiple price
        scenarios without re-running the physical simulation.
    price_scenario_names
        Optional names for each price scenario. When provided, the length must
        match ``price_inputs``. Names are added to the ``price_scenario`` column
        in the returned DataFrame to support pivoting or grouping.
    wesm_profile_df
        Optional hourly WESM profile data. When supplied and WESM pricing is
        enabled in ``price_inputs``, hourly shortfall and surplus pricing is
        aggregated into annual cost/revenue streams.
    wesm_step_hours
        Time resolution (hours) for the hourly logs used in WESM aggregation.
    max_candidates
        Optional limit on the total candidate count. When provided the function will
        either raise, return an empty DataFrame, or process the sweep in batches
        depending on ``on_exceed``. The pre-check runs before any simulation work to
        avoid firing long jobs inadvertently.
    on_exceed
        Controls behavior when ``max_candidates`` is exceeded. Accepted values are
        ``"raise"`` (default), ``"return"`` (empty DataFrame), and ``"batch"``
        (process in batches to cap concurrent work).
    batch_size
        Number of candidates to process per batch. When not set, batching uses
        ``max_candidates`` as the batch size when ``on_exceed`` is ``"batch"``.
        A positive value can also be supplied to process smaller batches even when
        ``max_candidates`` is not provided.
    concurrency
        Optional executor type. Set to ``"thread"`` or ``"process"`` to parallelize
        candidate evaluation. Results preserve the input ordering regardless of
        completion order. ``None`` (default) runs sequentially.
    max_workers
        Maximum workers for the executor when ``concurrency`` is enabled. Defaults to
        the library default for the chosen executor.
    min_compliance_pct
        Optional compliance floor. Candidates below the threshold are marked as
        pruned and skip economics to avoid unnecessary work.
    max_shortfall_mwh
        Optional annual shortfall ceiling. Candidates exceeding the cap are marked
        as pruned and skip economics for the same reason.
    progress_callback
        Optional callable that receives incremental sweep results as a ``DataFrame``.
        The callback is invoked once per candidate (or batch when using a process/thread
        executor) to support UI progress indicators or streaming displays.
    """

    rows: List[dict[str, float | bool | str]] = []
    base_initial_energy_mwh = float(base_cfg.initial_usable_mwh)
    price_scenarios = _normalize_price_scenarios(price_inputs, price_scenario_names)
    candidates = _prepare_sweep_candidates(
        base_cfg,
        power_mw_values,
        duration_h_values,
        energy_mwh_values,
        fixed_power_mw,
    )
    candidate_count = len(candidates)
    batches: list[list[tuple[float, float, float]]] = [candidates]

    if max_candidates is not None and candidate_count > max_candidates:
        if on_exceed == "return":
            return pd.DataFrame()
        if on_exceed == "batch":
            effective_batch_size = batch_size or max_candidates
            batches = _chunked(candidates, effective_batch_size)
        else:
            raise ValueError(
                f"Candidate count {candidate_count} exceeds max_candidates={max_candidates}. "
                "Use on_exceed='batch' or 'return' to change the behavior."
            )
    elif batch_size is not None:
        batches = _chunked(candidates, batch_size)

    executor_cls = None
    if concurrency is not None:
        if concurrency not in {"thread", "process"}:
            raise ValueError("concurrency must be 'thread', 'process', or None.")
        # Process pools require picklable inputs; callers should avoid local simulate/summary
        # functions when selecting that mode.
        executor_cls = ThreadPoolExecutor if concurrency == "thread" else ProcessPoolExecutor

    for batch_candidates in batches:
        if executor_cls is None:
            for power_mw, duration_h, candidate_energy_mwh in batch_candidates:
                row_group = _evaluate_candidate_row(
                    base_cfg,
                    pv_df,
                    cycle_df,
                    dod_override,
                    power_mw,
                    duration_h,
                    candidate_energy_mwh,
                    min_soh,
                    economics_inputs,
                    price_scenarios,
                    base_initial_energy_mwh,
                    wesm_profile_df,
                    wesm_step_hours,
                    simulate_fn,
                    summarize_fn,
                    min_compliance_pct,
                    max_shortfall_mwh,
                )
                rows.extend(row_group)
                _emit_progress_rows(row_group, progress_callback)
        else:
            evaluate_fn = partial(
                _evaluate_candidate_tuple,
                base_cfg=base_cfg,
                pv_df=pv_df,
                cycle_df=cycle_df,
                dod_override=dod_override,
                min_soh=min_soh,
                economics_inputs=economics_inputs,
                price_scenarios=price_scenarios,
                base_initial_energy_mwh=base_initial_energy_mwh,
                wesm_profile_df=wesm_profile_df,
                wesm_step_hours=wesm_step_hours,
                simulate_fn=simulate_fn,
                summarize_fn=summarize_fn,
                min_compliance_pct=min_compliance_pct,
                max_shortfall_mwh=max_shortfall_mwh,
            )
            with executor_cls(max_workers=max_workers) as executor:
                for row_group in executor.map(evaluate_fn, batch_candidates):
                    rows.extend(row_group)
                    _emit_progress_rows(row_group, progress_callback)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    ranking_column, ascending, fallback_column, fallback_ascending = _resolve_ranking_column(use_case, ranking_kpi)
    ranking_missing = ranking_column not in df.columns or not df[ranking_column].notna().any()
    if ranking_missing:
        if ranking_kpi:
            logging.getLogger(__name__).warning(
                "Ranking KPI '%s' is missing or empty; falling back to '%s'.",
                ranking_column,
                fallback_column,
            )
        ranking_column = fallback_column
        ascending = fallback_ascending
    if ranking_column not in df.columns:
        df[ranking_column] = np.nan

    df["is_best"] = False
    eligible_df = df[df["status"] == "evaluated"]
    if not eligible_df.empty:
        if "price_scenario" in df.columns:
            for _, scenario_df in eligible_df.groupby("price_scenario", dropna=False):
                if scenario_df.empty:
                    continue
                best_idx = scenario_df[ranking_column].sort_values(ascending=ascending).index[0]
                df.loc[best_idx, "is_best"] = True
        else:
            best_idx = eligible_df[ranking_column].sort_values(ascending=ascending).index[0]
            df.loc[best_idx, "is_best"] = True

    return df


def _load_sample_inputs() -> tuple["SimConfig", pd.DataFrame, pd.DataFrame]:
    """Load the packaged sample PV and cycle-model inputs for quick demos."""

    from services.simulation_core import SimConfig

    repo_root = Path(__file__).resolve().parent.parent
    # PV profile is stored in MW and the cycle model rows are dimensionless;
    # downstream sweeps interpret power in MW, energy in MWh, and costs in USD.
    pv_df = pd.read_csv(repo_root / "data" / "PV_8760_MW.csv")
    cycle_df = pd.read_excel(repo_root / "data" / "cycle_model.xlsx")

    # One-year run keeps the demo fast while still exercising the simulator.
    base_cfg = SimConfig(years=1)
    return base_cfg, pv_df, cycle_df


def _main_example() -> None:
    """Execute a small sweep using bundled sample data and print the table."""

    # Streamlit emits cache warnings when the runtime is absent; quiet them for CLI demos.
    logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)

    base_cfg, pv_df, cycle_df = _load_sample_inputs()
    # Allow a bit more cycling headroom so at least a few candidates are feasible while
    # still showing margin columns in the output.
    base_cfg = replace(base_cfg, max_cycles_per_day_cap=2.0)

    # Economics use USD throughout; capex and fixed opex are expressed in millions of USD
    # while prices are per delivered MWh.
    economics_inputs = EconomicInputs(
        capex_musd=0.8,
        fixed_opex_pct_of_capex=0.015,
        fixed_opex_musd=0.0,
        inflation_rate=0.02,
        discount_rate=0.07,
    )
    price_inputs = PriceInputs(
        contract_price_usd_per_mwh=120.0,
        apply_wesm_to_shortfall=False,
    )

    df = sweep_bess_sizes(
        base_cfg,
        pv_df,
        cycle_df,
        "Auto (infer)",
        power_mw_values=[10.0, 15.0, 20.0],
        duration_h_values=[2.0, 4.0],
        min_soh=0.9,
        use_case="reliability",
        ranking_kpi="npv_per_mwh_usd",
        economics_inputs=economics_inputs,
        price_inputs=price_inputs,
    )

    feasibility_cols = [
        "power_mw",
        "duration_h",
        "energy_mwh",
        "compliance_pct",
        "cycles_over_cap",
        "soh_margin",
        "status",
        "is_best",
    ]
    print("\nFeasibility snapshot (power MW, energy MWh, cycle margins are annualized):")
    print(df[feasibility_cols].to_string(index=False))

    evaluated = df[df["status"] == "evaluated"]
    if not evaluated.empty:
        ranked = evaluated.sort_values("npv_per_mwh_usd", ascending=False)
        ranking_cols = [
            "power_mw",
            "duration_h",
            "npv_per_mwh_usd",
            "npv_usd",
            "lcoe_usd_per_mwh",
            "capex_per_kw_usd",
            "irr_pct",
            "is_best",
        ]
        print("\nRanking KPI (npv_per_mwh_usd): higher is better and is_best marks the top pick.")
        print(ranked[ranking_cols].to_string(index=False))
    else:
        print("\nNo candidates passed feasibility; economics were skipped.")


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
