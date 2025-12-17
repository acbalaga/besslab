"""Economic helpers shared across app and CLI entrypoints."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence


DEFAULT_FOREX_RATE_PHP_PER_USD = 58.0
# DevEx is modeled as a PHP-denominated amount to align with other UI inputs.
# A default USD conversion is retained for callers that do not override
# ``devex_cost_usd`` explicitly (e.g., during tests or CLI usage).
DEVEX_COST_PHP = 100_000_000.0
DEVEX_COST_USD = DEVEX_COST_PHP / DEFAULT_FOREX_RATE_PHP_PER_USD


@dataclass
class EconomicInputs:
    """High-level project economics.

    All monetary values are expressed in USD to avoid mixing units. CAPEX and
    fixed OPEX can be entered in millions to keep UI inputs compact. The
    inflation_rate is applied as an annual escalator to fixed OPEX before
    discounting.
    """

    capex_musd: float
    fixed_opex_pct_of_capex: float
    fixed_opex_musd: float
    inflation_rate: float
    discount_rate: float
    variable_opex_usd_per_mwh: float | None = None
    variable_opex_schedule_usd: tuple[float, ...] | None = None
    periodic_variable_opex_usd: float | None = None
    periodic_variable_opex_interval_years: int | None = None
    devex_cost_usd: float = DEVEX_COST_USD
    include_devex_year0: bool = False


@dataclass
class EconomicOutputs:
    """Discounted cost and energy aggregates plus derived LCOE/LCOS."""

    discounted_costs_usd: float
    discounted_augmentation_costs_usd: float
    discounted_energy_mwh: float
    discounted_bess_energy_mwh: float
    lcoe_usd_per_mwh: float
    lcos_usd_per_mwh: float


@dataclass
class PriceInputs:
    """Energy price assumptions used for cash-flow based metrics."""

    contract_price_usd_per_mwh: float
    pv_market_price_usd_per_mwh: float
    escalate_with_inflation: bool = False


@dataclass
class CashFlowOutputs:
    """Cash-flow oriented metrics such as discounted revenue and IRR."""

    discounted_revenues_usd: float
    discounted_pv_excess_revenue_usd: float
    npv_usd: float
    irr_pct: float


def _discount_factor(discount_rate: float, year_index: int) -> float:
    """Return the discount factor for a given year index (1-indexed)."""

    return 1.0 / ((1.0 + discount_rate) ** year_index)


def _ensure_non_negative_finite(value: float, name: str) -> None:
    """Raise ValueError when a numeric value is negative or non-finite."""

    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_inputs(
    annual_delivered_mwh: Sequence[float],
    annual_bess_mwh: Sequence[float],
    inputs: EconomicInputs,
) -> None:
    """Validate energy series lengths and economic inputs before computing outputs."""

    if len(annual_delivered_mwh) != len(annual_bess_mwh):
        raise ValueError(
            "annual_delivered_mwh and annual_bess_mwh must have the same number of years"
        )

    for idx, value in enumerate(annual_delivered_mwh, start=1):
        _ensure_non_negative_finite(float(value), f"annual_delivered_mwh[{idx}]")
    for idx, value in enumerate(annual_bess_mwh, start=1):
        _ensure_non_negative_finite(float(value), f"annual_bess_mwh[{idx}]")

    _ensure_non_negative_finite(inputs.capex_musd, "capex_musd")
    _ensure_non_negative_finite(inputs.fixed_opex_pct_of_capex, "fixed_opex_pct_of_capex")
    _ensure_non_negative_finite(inputs.fixed_opex_musd, "fixed_opex_musd")
    _ensure_non_negative_finite(inputs.inflation_rate, "inflation_rate")
    _ensure_non_negative_finite(inputs.discount_rate, "discount_rate")

    if inputs.variable_opex_usd_per_mwh is not None:
        _ensure_non_negative_finite(inputs.variable_opex_usd_per_mwh, "variable_opex_usd_per_mwh")
    if inputs.periodic_variable_opex_usd is not None:
        _ensure_non_negative_finite(inputs.periodic_variable_opex_usd, "periodic_variable_opex_usd")
    if inputs.periodic_variable_opex_interval_years is not None and inputs.periodic_variable_opex_interval_years <= 0:
        raise ValueError("periodic_variable_opex_interval_years must be positive when provided")

    if inputs.variable_opex_schedule_usd is not None:
        for idx, value in enumerate(inputs.variable_opex_schedule_usd, start=1):
            _ensure_non_negative_finite(float(value), f"variable_opex_schedule_usd[{idx}]")


def _validate_price_inputs(price_inputs: PriceInputs) -> None:
    """Raise ValueError when provided price assumptions are invalid."""

    _ensure_non_negative_finite(
        price_inputs.contract_price_usd_per_mwh, "contract_price_usd_per_mwh"
    )
    _ensure_non_negative_finite(
        price_inputs.pv_market_price_usd_per_mwh, "pv_market_price_usd_per_mwh"
    )


def _resolve_variable_opex_schedule(years: int, inputs: EconomicInputs) -> list[float] | None:
    """Return a per-year variable OPEX schedule honoring user overrides.

    Precedence is applied as follows:
    1) ``variable_opex_schedule_usd`` when explicitly provided.
    2) ``periodic_variable_opex_usd`` on the specified cadence.
    3) ``None`` to signal that fixed OPEX or per-MWh costs should be used instead.
    """

    if inputs.variable_opex_schedule_usd is not None:
        schedule = list(inputs.variable_opex_schedule_usd)
        if schedule and len(schedule) != years:
            raise ValueError("variable_opex_schedule_usd must align with the number of years provided")
        schedule.extend([0.0] * (years - len(schedule)))
        return schedule

    if (
        inputs.periodic_variable_opex_usd is not None
        and inputs.periodic_variable_opex_interval_years is not None
    ):
        cadence = max(int(inputs.periodic_variable_opex_interval_years), 1)
        amount = float(inputs.periodic_variable_opex_usd)
        schedule = [0.0 for _ in range(years)]
        for year_idx in range(1, years + 1):
            if (year_idx - 1) % cadence == 0:
                schedule[year_idx - 1] = amount
        return schedule

    return None


def _initial_project_spend(inputs: EconomicInputs) -> float:
    """Return the upfront spend applied at year 0 including optional DevEx."""

    base_capex_usd = inputs.capex_musd * 1_000_000.0
    if inputs.include_devex_year0:
        return base_capex_usd + inputs.devex_cost_usd
    return base_capex_usd


def compute_lcoe_lcos(
    annual_delivered_mwh: Sequence[float],
    annual_bess_mwh: Sequence[float],
    inputs: EconomicInputs,
    augmentation_costs_usd: Sequence[float] | None = None,
) -> EconomicOutputs:
    """Compute LCOE and LCOS using discounted cash-flow style math.

    Parameters
    ----------
    annual_delivered_mwh
        Total firm energy delivered each project year (AC-side).
    annual_bess_mwh
        Portion of firm energy that came from the BESS each year (AC-side).
    inputs
        Economic assumptions such as CAPEX, OPEX, and discount rate.
    augmentation_costs_usd
        Optional per-year augmentation CAPEX (USD, undiscounted) to include in the cash flows.

    Notes
    -----
    Explicit variable OPEX schedules take precedence over per-MWh operating
    costs, which in turn override fixed OPEX derived from CAPEX percentages and
    adders. Custom schedules are assumed to already reflect nominal year-by-
    year spending and are not escalated further.
    """

    _validate_inputs(annual_delivered_mwh, annual_bess_mwh, inputs)

    if augmentation_costs_usd is not None and len(augmentation_costs_usd) != len(
        annual_delivered_mwh
    ):
        raise ValueError("augmentation_costs_usd must match number of years")
    if augmentation_costs_usd is not None:
        for idx, value in enumerate(augmentation_costs_usd, start=1):
            _ensure_non_negative_finite(float(value), f"augmentation_costs_usd[{idx}]")

    years = len(annual_delivered_mwh)
    if years == 0:
        return EconomicOutputs(
            float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
        )

    discounted_costs = _initial_project_spend(inputs)
    discounted_augmentation_costs = 0.0
    discounted_energy = 0.0
    discounted_bess_energy = 0.0
    # fixed_opex_pct_of_capex is expressed as a percent (e.g., 2.5 = 2.5%)
    fixed_opex_from_capex = inputs.capex_musd * (inputs.fixed_opex_pct_of_capex / 100.0)
    variable_opex_schedule = _resolve_variable_opex_schedule(years, inputs)

    for year_idx in range(1, years + 1):
        firm_mwh = float(annual_delivered_mwh[year_idx - 1])
        bess_mwh = float(annual_bess_mwh[year_idx - 1])
        factor = _discount_factor(inputs.discount_rate, year_idx)

        # Escalate OPEX annually by the assumed inflation rate when using fixed inputs
        # or per-MWh costs. Explicit schedules are assumed to already reflect the intended
        # nominal spend for the corresponding year.
        inflation_multiplier = (1.0 + inputs.inflation_rate) ** (year_idx - 1)
        annual_opex = 0.0
        if variable_opex_schedule is not None:
            if year_idx - 1 >= len(variable_opex_schedule):
                raise ValueError("variable_opex_schedule_usd must match the number of project years")
            annual_opex = float(variable_opex_schedule[year_idx - 1])
        elif inputs.variable_opex_usd_per_mwh is not None:
            annual_opex = firm_mwh * float(inputs.variable_opex_usd_per_mwh) * inflation_multiplier
        else:
            annual_fixed_opex = (fixed_opex_from_capex + inputs.fixed_opex_musd) * 1_000_000
            annual_opex = annual_fixed_opex * inflation_multiplier
        augmentation_cost = 0.0
        if augmentation_costs_usd is not None:
            augmentation_cost = float(augmentation_costs_usd[year_idx - 1])

        discounted_augmentation_costs += augmentation_cost * factor
        discounted_costs += (annual_opex + augmentation_cost) * factor
        discounted_energy += firm_mwh * factor
        discounted_bess_energy += bess_mwh * factor

    lcoe = discounted_costs / discounted_energy if discounted_energy > 0 else float("nan")
    lcos = discounted_costs / discounted_bess_energy if discounted_bess_energy > 0 else float("nan")

    return EconomicOutputs(
        discounted_costs_usd=discounted_costs,
        discounted_augmentation_costs_usd=discounted_augmentation_costs,
        discounted_energy_mwh=discounted_energy,
        discounted_bess_energy_mwh=discounted_bess_energy,
        lcoe_usd_per_mwh=lcoe,
        lcos_usd_per_mwh=lcos,
    )


def compute_cash_flows_and_irr(
    annual_delivered_mwh: Sequence[float],
    annual_bess_mwh: Sequence[float],
    annual_pv_excess_mwh: Sequence[float],
    inputs: EconomicInputs,
    price_inputs: PriceInputs,
    augmentation_costs_usd: Sequence[float] | None = None,
    max_iterations: int = 200,
) -> CashFlowOutputs:
    """Compute discounted revenues, project NPV, and an implied IRR.

    Revenue is split into two streams:

    * Contract revenue from BESS-originated energy using a fixed contract price.
    * Market revenue from excess PV that would otherwise be curtailed.

    Contract and market prices can optionally escalate with the same inflation
    rate used for OPEX. Augmentation costs are treated as a year-specific cash
    outflow alongside fixed OPEX. The IRR calculation uses the undiscounted
    cash-flow list to avoid dependence on the chosen discount rate.

    When provided, variable OPEX schedules override per-MWh costs, which in turn
    override fixed OPEX derived from CAPEX-based percentages and adders.
    """

    _validate_inputs(annual_delivered_mwh, annual_bess_mwh, inputs)
    if len(annual_pv_excess_mwh) != len(annual_delivered_mwh):
        raise ValueError("annual_pv_excess_mwh must match number of years")
    for idx, value in enumerate(annual_pv_excess_mwh, start=1):
        _ensure_non_negative_finite(float(value), f"annual_pv_excess_mwh[{idx}]")
    _validate_price_inputs(price_inputs)

    if augmentation_costs_usd is not None and len(augmentation_costs_usd) != len(
        annual_delivered_mwh
    ):
        raise ValueError("augmentation_costs_usd must match number of years")
    if augmentation_costs_usd is not None:
        for idx, value in enumerate(augmentation_costs_usd, start=1):
            _ensure_non_negative_finite(float(value), f"augmentation_costs_usd[{idx}]")

    years = len(annual_delivered_mwh)
    if years == 0:
        return CashFlowOutputs(float("nan"), float("nan"), float("nan"), float("nan"))

    discounted_revenues = 0.0
    discounted_pv_revenue = 0.0
    cash_flows = [-_initial_project_spend(inputs)]

    fixed_opex_from_capex = inputs.capex_musd * (inputs.fixed_opex_pct_of_capex / 100.0)
    variable_opex_schedule = _resolve_variable_opex_schedule(years, inputs)

    for year_idx in range(1, years + 1):
        firm_mwh = float(annual_delivered_mwh[year_idx - 1])
        bess_mwh = float(annual_bess_mwh[year_idx - 1])
        pv_excess_mwh = float(annual_pv_excess_mwh[year_idx - 1])
        factor = _discount_factor(inputs.discount_rate, year_idx)
        inflation_multiplier = (1.0 + inputs.inflation_rate) ** (year_idx - 1)

        annual_opex = 0.0
        if variable_opex_schedule is not None:
            if year_idx - 1 >= len(variable_opex_schedule):
                raise ValueError("variable_opex_schedule_usd must match the number of project years")
            annual_opex = float(variable_opex_schedule[year_idx - 1])
        elif inputs.variable_opex_usd_per_mwh is not None:
            annual_opex = firm_mwh * float(inputs.variable_opex_usd_per_mwh) * inflation_multiplier
        else:
            annual_fixed_opex = (fixed_opex_from_capex + inputs.fixed_opex_musd) * 1_000_000
            annual_opex = annual_fixed_opex * inflation_multiplier
        augmentation_cost = 0.0
        if augmentation_costs_usd is not None:
            augmentation_cost = float(augmentation_costs_usd[year_idx - 1])

        bess_revenue = bess_mwh * price_inputs.contract_price_usd_per_mwh
        pv_revenue = pv_excess_mwh * price_inputs.pv_market_price_usd_per_mwh
        if price_inputs.escalate_with_inflation:
            bess_revenue *= inflation_multiplier
            pv_revenue *= inflation_multiplier

        total_revenue = bess_revenue + pv_revenue
        discounted_revenues += total_revenue * factor
        discounted_pv_revenue += pv_revenue * factor
        cash_flows.append(total_revenue - annual_opex - augmentation_cost)

    npv_usd = _compute_npv(cash_flows, inputs.discount_rate)
    irr_pct = _solve_irr_pct(cash_flows, max_iterations=max_iterations)

    return CashFlowOutputs(
        discounted_revenues_usd=discounted_revenues,
        discounted_pv_excess_revenue_usd=discounted_pv_revenue,
        npv_usd=npv_usd,
        irr_pct=irr_pct,
    )


def _discount_augmentation_costs(
    augmentation_costs_usd: Sequence[float] | None, discount_rate: float
) -> float:
    """Return discounted augmentation costs using the provided discount rate."""

    if augmentation_costs_usd is None:
        return 0.0

    discounted_total = 0.0
    for year_idx, cost in enumerate(augmentation_costs_usd, start=1):
        discounted_total += float(cost) / ((1.0 + discount_rate) ** year_idx)
    return discounted_total


def _compute_npv(cash_flows: Sequence[float], discount_rate: float) -> float:
    """Return the net present value of the provided cash flows."""

    return sum(cf / ((1.0 + discount_rate) ** idx) for idx, cf in enumerate(cash_flows))


def _solve_irr_pct(cash_flows: Sequence[float], max_iterations: int = 200) -> float:
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


def compute_lcoe_lcos_with_augmentation_fallback(
    annual_delivered_mwh: Sequence[float],
    annual_bess_mwh: Sequence[float],
    inputs: EconomicInputs,
    augmentation_costs_usd: Sequence[float] | None = None,
    compute_fn: Callable[..., EconomicOutputs] = compute_lcoe_lcos,
) -> EconomicOutputs:
    """Run ``compute_lcoe_lcos`` and add augmentation even if the function signature lags.

    ``compute_lcoe_lcos`` gained an ``augmentation_costs_usd`` parameter, but
    deployed environments can lag behind the codebase. This wrapper first tries
    the new API. If ``compute_fn`` raises ``TypeError`` due to the keyword, it
    falls back to calling without augmentation and then manually layers the
    discounted augmentation spend onto the outputs so LCOE/LCOS still reflect
    the added costs.
    """

    try:
        return compute_fn(
            annual_delivered_mwh,
            annual_bess_mwh,
            inputs,
            augmentation_costs_usd=augmentation_costs_usd,
        )
    except TypeError as exc:
        # Only fall back when the TypeError indicates the compute function rejected the
        # augmentation keyword argument. Re-raise other TypeErrors (e.g., input validation)
        # so calling code is not masked.
        if "augmentation_costs_usd" not in str(exc):
            raise

        discounted_augmentation_costs = _discount_augmentation_costs(
            augmentation_costs_usd, inputs.discount_rate
        )
        base_outputs = compute_fn(annual_delivered_mwh, annual_bess_mwh, inputs)
        updated_discounted_costs = base_outputs.discounted_costs_usd + discounted_augmentation_costs

        return EconomicOutputs(
            discounted_costs_usd=updated_discounted_costs,
            discounted_augmentation_costs_usd=discounted_augmentation_costs,
            discounted_energy_mwh=base_outputs.discounted_energy_mwh,
            discounted_bess_energy_mwh=base_outputs.discounted_bess_energy_mwh,
            lcoe_usd_per_mwh=
                updated_discounted_costs / base_outputs.discounted_energy_mwh
                if base_outputs.discounted_energy_mwh > 0
                else float("nan"),
            lcos_usd_per_mwh=
                updated_discounted_costs / base_outputs.discounted_bess_energy_mwh
                if base_outputs.discounted_bess_energy_mwh > 0
                else float("nan"),
        )


def estimate_augmentation_costs_by_year(
    augmentation_energy_added_mwh: Sequence[float],
    initial_usable_mwh: float,
    capex_musd: float,
) -> list[float]:
    """Estimate per-year augmentation spend based on the energy added.

    Costs scale linearly with the proportion of new BOL-equivalent energy
    relative to the initial usable capacity. This mirrors common augmentation
    pricing that pegs refresh spend to the share of the original system being
    added back. Returns an array of USD amounts aligned with the input series.
    """

    if initial_usable_mwh <= 0:
        # Avoid division by zero while keeping alignment with the input series.
        return [0.0 for _ in augmentation_energy_added_mwh]

    base_capex_usd = capex_musd * 1_000_000.0
    costs: list[float] = []
    for add_energy in augmentation_energy_added_mwh:
        add_energy_safe = max(0.0, float(add_energy))
        share_of_bol = add_energy_safe / initial_usable_mwh
        costs.append(base_capex_usd * share_of_bol)

    return costs


__all__ = [
    "DEFAULT_FOREX_RATE_PHP_PER_USD",
    "DEVEX_COST_PHP",
    "DEVEX_COST_USD",
    "EconomicInputs",
    "EconomicOutputs",
    "PriceInputs",
    "CashFlowOutputs",
    "compute_lcoe_lcos",
    "compute_cash_flows_and_irr",
    "compute_lcoe_lcos_with_augmentation_fallback",
    "_discount_factor",
    "_ensure_non_negative_finite",
    "_validate_inputs",
    "_validate_price_inputs",
    "_discount_augmentation_costs",
    "_compute_npv",
    "_solve_irr_pct",
    "estimate_augmentation_costs_by_year",
]
