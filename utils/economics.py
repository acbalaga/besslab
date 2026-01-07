"""Economic helpers shared across app and CLI entrypoints."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Literal, Sequence


DEFAULT_FOREX_RATE_PHP_PER_USD = 58.0
# DevEx is modeled as a PHP-denominated amount to align with other UI inputs.
# A default USD conversion is retained for callers that do not override
# ``devex_cost_usd`` explicitly (e.g., during tests or CLI usage).
DEVEX_COST_PHP = 100_000_000.0
DEVEX_COST_USD = DEVEX_COST_PHP / DEFAULT_FOREX_RATE_PHP_PER_USD


@dataclass
class EconomicInputs:
    """High-level project economics.

    Unit conventions:
    - CAPEX can be entered as USD/kWh (with a BOL size in kWh) or as a total USD override.
      Canonical CAPEX is stored in ``capex_musd`` (USD millions).
    - Fixed OPEX may be entered as % of CAPEX per year (percent value, e.g., 2.0 for 2%).
    - Variable OPEX may be entered in PHP/kWh on total generation and is converted to USD/MWh.
    - FX conversion uses ``forex_rate_php_per_usd`` (PHP per 1 USD).
    - BESS size inputs are in kWh for conversions and MWh for annual energy series.
    """

    capex_musd: float = 0.0
    capex_usd_per_kwh: float | None = None
    capex_total_usd: float | None = None
    bess_bol_kwh: float | None = None
    fixed_opex_pct_of_capex: float = 0.0
    fixed_opex_musd: float = 0.0
    opex_php_per_kwh: float | None = None
    inflation_rate: float = 0.0
    discount_rate: float = 0.0
    variable_opex_usd_per_mwh: float | None = None
    variable_opex_schedule_usd: tuple[float, ...] | None = None
    periodic_variable_opex_usd: float | None = None
    periodic_variable_opex_interval_years: int | None = None
    variable_opex_basis: Literal["delivered", "total_generation"] = "delivered"
    forex_rate_php_per_usd: float = DEFAULT_FOREX_RATE_PHP_PER_USD
    devex_cost_php: float = DEVEX_COST_PHP
    devex_cost_usd: float | None = None
    include_devex_year0: bool = False
    debt_ratio: float = 0.0
    cost_of_debt: float = 0.0
    tenor_years: int | None = None
    wacc: float = 0.0


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
    """Energy price assumptions used for cash-flow based metrics.

    When ``blended_price_usd_per_mwh`` is provided it overrides the individual
    contract and PV rates so all energy is monetized at the blended value. The
    optional WESM deficit price applies to contract shortfalls when enabled,
    treated as a purchase cost. When ``sell_to_wesm`` is enabled, PV surplus
    (excess energy) can be credited at a dedicated WESM sale price; otherwise,
    surplus revenue is excluded from the cash-flow stream.
    """

    contract_price_usd_per_mwh: float
    pv_market_price_usd_per_mwh: float
    escalate_with_inflation: bool = False
    blended_price_usd_per_mwh: float | None = None
    wesm_deficit_price_usd_per_mwh: float | None = None
    wesm_surplus_price_usd_per_mwh: float | None = None
    apply_wesm_to_shortfall: bool = False
    sell_to_wesm: bool = False


@dataclass
class CashFlowOutputs:
    """Cash-flow oriented metrics such as discounted revenue and IRR."""

    discounted_revenues_usd: float
    discounted_pv_excess_revenue_usd: float
    discounted_wesm_value_usd: float
    npv_usd: float
    irr_pct: float


@dataclass
class FinancingOutputs:
    """Financing-aware metrics such as EBITDA and project/equity IRR."""

    ebitda_usd: float
    ebitda_margin: float
    project_npv_usd: float
    project_irr_pct: float
    equity_irr_pct: float


def _discount_factor(discount_rate: float, year_index: int) -> float:
    """Return the discount factor for a given year index (1-indexed)."""

    return 1.0 / ((1.0 + discount_rate) ** year_index)


def _ensure_non_negative_finite(value: float, name: str) -> None:
    """Raise ValueError when a numeric value is negative or non-finite."""

    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def normalize_economic_inputs(inputs: EconomicInputs) -> EconomicInputs:
    """Normalize economic inputs to canonical USD-based values.

    This helper converts CAPEX and OPEX inputs into the USD terms consumed by
    the core cash-flow calculations. It also applies the PHP-based DevEx input
    (defaulting to 100M PHP) and sets variable OPEX to use total-generation
    energy when the PHP/kWh mode is supplied.
    """

    forex_rate = float(inputs.forex_rate_php_per_usd or DEFAULT_FOREX_RATE_PHP_PER_USD)
    _ensure_non_negative_finite(forex_rate, "forex_rate_php_per_usd")
    if forex_rate == 0:
        raise ValueError("forex_rate_php_per_usd must be greater than zero")

    capex_total_usd: float | None = None
    if inputs.capex_total_usd is not None:
        capex_total_usd = float(inputs.capex_total_usd)
        _ensure_non_negative_finite(capex_total_usd, "capex_total_usd")
    elif inputs.capex_usd_per_kwh is not None:
        capex_usd_per_kwh = float(inputs.capex_usd_per_kwh)
        _ensure_non_negative_finite(capex_usd_per_kwh, "capex_usd_per_kwh")
        if inputs.bess_bol_kwh is None:
            raise ValueError("bess_bol_kwh must be provided when capex_usd_per_kwh is set")
        bess_bol_kwh = float(inputs.bess_bol_kwh)
        _ensure_non_negative_finite(bess_bol_kwh, "bess_bol_kwh")
        capex_total_usd = capex_usd_per_kwh * bess_bol_kwh
    else:
        capex_total_usd = float(inputs.capex_musd) * 1_000_000.0
        _ensure_non_negative_finite(capex_total_usd, "capex_musd")

    devex_cost_usd = inputs.devex_cost_usd
    if devex_cost_usd is None:
        devex_cost_php = float(inputs.devex_cost_php)
        _ensure_non_negative_finite(devex_cost_php, "devex_cost_php")
        devex_cost_usd = devex_cost_php / forex_rate
    else:
        devex_cost_usd = float(devex_cost_usd)
        _ensure_non_negative_finite(devex_cost_usd, "devex_cost_usd")

    variable_opex_usd_per_mwh = inputs.variable_opex_usd_per_mwh
    variable_opex_basis = inputs.variable_opex_basis
    if inputs.opex_php_per_kwh is not None:
        opex_php_per_kwh = float(inputs.opex_php_per_kwh)
        _ensure_non_negative_finite(opex_php_per_kwh, "opex_php_per_kwh")
        variable_opex_usd_per_mwh = opex_php_per_kwh / forex_rate * 1000.0
        variable_opex_basis = "total_generation"

    if variable_opex_basis not in {"delivered", "total_generation"}:
        raise ValueError("variable_opex_basis must be 'delivered' or 'total_generation'")

    return replace(
        inputs,
        capex_musd=capex_total_usd / 1_000_000.0,
        variable_opex_usd_per_mwh=variable_opex_usd_per_mwh,
        variable_opex_basis=variable_opex_basis,
        forex_rate_php_per_usd=forex_rate,
        devex_cost_usd=devex_cost_usd,
    )


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
    _ensure_non_negative_finite(inputs.forex_rate_php_per_usd, "forex_rate_php_per_usd")
    _ensure_non_negative_finite(inputs.devex_cost_usd, "devex_cost_usd")
    _ensure_non_negative_finite(inputs.debt_ratio, "debt_ratio")
    _ensure_non_negative_finite(inputs.cost_of_debt, "cost_of_debt")
    _ensure_non_negative_finite(inputs.wacc, "wacc")
    if inputs.debt_ratio > 1:
        raise ValueError("debt_ratio must be between 0 and 1")
    if inputs.tenor_years is not None and inputs.tenor_years <= 0:
        raise ValueError("tenor_years must be positive when provided")
    if inputs.debt_ratio > 0 and (inputs.tenor_years is None or inputs.tenor_years <= 0):
        raise ValueError("tenor_years must be provided when debt_ratio is greater than zero")

    if inputs.variable_opex_usd_per_mwh is not None:
        _ensure_non_negative_finite(inputs.variable_opex_usd_per_mwh, "variable_opex_usd_per_mwh")
    if inputs.periodic_variable_opex_usd is not None:
        _ensure_non_negative_finite(inputs.periodic_variable_opex_usd, "periodic_variable_opex_usd")
    if inputs.periodic_variable_opex_interval_years is not None and inputs.periodic_variable_opex_interval_years <= 0:
        raise ValueError("periodic_variable_opex_interval_years must be positive when provided")

    if inputs.variable_opex_schedule_usd is not None:
        for idx, value in enumerate(inputs.variable_opex_schedule_usd, start=1):
            _ensure_non_negative_finite(float(value), f"variable_opex_schedule_usd[{idx}]")

    if inputs.variable_opex_basis not in {"delivered", "total_generation"}:
        raise ValueError("variable_opex_basis must be 'delivered' or 'total_generation'")


def _validate_price_inputs(price_inputs: PriceInputs) -> None:
    """Raise ValueError when provided price assumptions are invalid."""

    _ensure_non_negative_finite(
        price_inputs.contract_price_usd_per_mwh, "contract_price_usd_per_mwh"
    )
    _ensure_non_negative_finite(
        price_inputs.pv_market_price_usd_per_mwh, "pv_market_price_usd_per_mwh"
    )
    if price_inputs.blended_price_usd_per_mwh is not None:
        _ensure_non_negative_finite(
            price_inputs.blended_price_usd_per_mwh, "blended_price_usd_per_mwh"
        )
    if price_inputs.wesm_deficit_price_usd_per_mwh is not None:
        _ensure_non_negative_finite(
            price_inputs.wesm_deficit_price_usd_per_mwh, "wesm_deficit_price_usd_per_mwh"
        )
    if price_inputs.wesm_surplus_price_usd_per_mwh is not None:
        _ensure_non_negative_finite(
            price_inputs.wesm_surplus_price_usd_per_mwh, "wesm_surplus_price_usd_per_mwh"
        )
    if (
        price_inputs.apply_wesm_to_shortfall
        and price_inputs.wesm_deficit_price_usd_per_mwh is None
    ):
        raise ValueError(
            "wesm_deficit_price_usd_per_mwh must be provided when applying WESM to shortfalls"
        )
    if (
        price_inputs.sell_to_wesm
        and price_inputs.wesm_surplus_price_usd_per_mwh is None
        and price_inputs.wesm_deficit_price_usd_per_mwh is None
    ):
        raise ValueError(
            "wesm_surplus_price_usd_per_mwh or wesm_deficit_price_usd_per_mwh must be provided when "
            "sell_to_wesm is enabled"
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


def _build_debt_service_schedule(
    principal_usd: float,
    annual_rate: float,
    tenor_years: int,
    project_years: int,
) -> list[float]:
    """Return annual debt service (principal + interest) for the project horizon."""

    if principal_usd <= 0 or tenor_years <= 0 or project_years <= 0:
        return [0.0 for _ in range(project_years)]

    balance = principal_usd
    debt_service: list[float] = []
    if annual_rate == 0:
        annual_principal = principal_usd / tenor_years
        for year_idx in range(1, project_years + 1):
            if year_idx <= tenor_years:
                payment = min(annual_principal, balance)
                balance -= payment
                debt_service.append(payment)
            else:
                debt_service.append(0.0)
        return debt_service

    payment = (
        principal_usd
        * (annual_rate * (1.0 + annual_rate) ** tenor_years)
        / ((1.0 + annual_rate) ** tenor_years - 1.0)
    )
    for year_idx in range(1, project_years + 1):
        if year_idx <= tenor_years and balance > 0:
            interest = balance * annual_rate
            principal_payment = max(payment - interest, 0.0)
            principal_payment = min(principal_payment, balance)
            balance -= principal_payment
            debt_service.append(interest + principal_payment)
        else:
            debt_service.append(0.0)
    return debt_service


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
    annual_total_generation_mwh: Sequence[float] | None = None,
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
    annual_total_generation_mwh
        Optional per-year total generation (MWh). Used when variable OPEX is defined per kWh
        of total generation rather than delivered firm energy.

    Notes
    -----
    Explicit variable OPEX schedules take precedence over per-MWh operating
    costs, which in turn override fixed OPEX derived from CAPEX percentages and
    adders. Custom schedules are assumed to already reflect nominal year-by-
    year spending and are not escalated further.
    """

    inputs = normalize_economic_inputs(inputs)
    _validate_inputs(annual_delivered_mwh, annual_bess_mwh, inputs)

    if augmentation_costs_usd is not None and len(augmentation_costs_usd) != len(
        annual_delivered_mwh
    ):
        raise ValueError("augmentation_costs_usd must match number of years")
    if augmentation_costs_usd is not None:
        for idx, value in enumerate(augmentation_costs_usd, start=1):
            _ensure_non_negative_finite(float(value), f"augmentation_costs_usd[{idx}]")

    if annual_total_generation_mwh is not None and len(annual_total_generation_mwh) != len(
        annual_delivered_mwh
    ):
        raise ValueError("annual_total_generation_mwh must match number of years")
    if annual_total_generation_mwh is not None:
        for idx, value in enumerate(annual_total_generation_mwh, start=1):
            _ensure_non_negative_finite(float(value), f"annual_total_generation_mwh[{idx}]")

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
            opex_basis_mwh = firm_mwh
            if inputs.variable_opex_basis == "total_generation" and annual_total_generation_mwh is not None:
                opex_basis_mwh = float(annual_total_generation_mwh[year_idx - 1])
            elif inputs.variable_opex_basis == "total_generation":
                # Fall back to delivered energy when total generation is unavailable.
                opex_basis_mwh = firm_mwh
            annual_opex = opex_basis_mwh * float(inputs.variable_opex_usd_per_mwh) * inflation_multiplier
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
    *,
    annual_pv_delivered_mwh: Sequence[float] | None = None,
    annual_shortfall_mwh: Sequence[float] | None = None,
    augmentation_costs_usd: Sequence[float] | None = None,
    annual_total_generation_mwh: Sequence[float] | None = None,
    max_iterations: int = 200,
) -> CashFlowOutputs:
    """Compute discounted revenues, project NPV, and an implied IRR.

    Revenue is split into two streams (plus optional WESM adjustments):

    * Contract revenue from BESS-originated energy using a fixed contract price.
    * Market revenue from PV energy delivered to the firm contract.
    * Optional WESM sales or purchases tied to contract shortfalls.

    Contract and market prices can optionally escalate with the same inflation
    rate used for OPEX. When a blended energy price is provided, it overrides
    the individual contract and PV rates and is applied to all delivered and
    marketed energy streams (including PV that serves the firm contract).
    Augmentation costs are treated as a year-specific cash outflow alongside
    fixed OPEX. The IRR calculation uses the undiscounted cash-flow list to
    avoid dependence on the chosen discount rate.

    Variable OPEX can be applied to total generation (e.g., PV output) when
    ``annual_total_generation_mwh`` is supplied; otherwise it defaults to firm
    deliveries (or delivered + PV excess when using the built-in generation
    proxy).

    The PV-delivered series can be passed explicitly via
    ``annual_pv_delivered_mwh`` or derived as ``annual_delivered_mwh -
    annual_bess_mwh``. The delivered split is validated to ensure it sums to
    total firm deliveries each year.

    When provided, variable OPEX schedules override per-MWh costs, which in turn
    override fixed OPEX derived from CAPEX-based percentages and adders. When
    ``apply_wesm_to_shortfall`` is True, shortfall MWh are monetized using the
    WESM deficit price as a purchase (cost). Surplus PV (``annual_pv_excess_mwh``)
    is only credited at the WESM sale price (falling back to the deficit price
    when no sale-specific rate is provided) when ``sell_to_wesm`` is True;
    otherwise it is excluded from revenue when WESM pricing is enabled.
    """

    inputs = normalize_economic_inputs(inputs)
    _validate_inputs(annual_delivered_mwh, annual_bess_mwh, inputs)
    if len(annual_pv_excess_mwh) != len(annual_delivered_mwh):
        raise ValueError("annual_pv_excess_mwh must match number of years")
    for idx, value in enumerate(annual_pv_excess_mwh, start=1):
        _ensure_non_negative_finite(float(value), f"annual_pv_excess_mwh[{idx}]")
    if annual_pv_delivered_mwh is None:
        annual_pv_delivered_mwh = [
            float(delivered) - float(bess)
            for delivered, bess in zip(annual_delivered_mwh, annual_bess_mwh)
        ]
    if len(annual_pv_delivered_mwh) != len(annual_delivered_mwh):
        raise ValueError("annual_pv_delivered_mwh must match number of years")
    for idx, value in enumerate(annual_pv_delivered_mwh, start=1):
        _ensure_non_negative_finite(float(value), f"annual_pv_delivered_mwh[{idx}]")
    for idx, (delivered, bess, pv_delivered) in enumerate(
        zip(annual_delivered_mwh, annual_bess_mwh, annual_pv_delivered_mwh), start=1
    ):
        delivered_value = float(delivered)
        split_value = float(bess) + float(pv_delivered)
        if not math.isclose(delivered_value, split_value, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                "annual_bess_mwh and annual_pv_delivered_mwh must sum to annual_delivered_mwh "
                f"for year {idx}"
            )
    if annual_shortfall_mwh is None:
        annual_shortfall_mwh = [0.0 for _ in annual_delivered_mwh]
    if len(annual_shortfall_mwh) != len(annual_delivered_mwh):
        raise ValueError("annual_shortfall_mwh must match number of years")
    for idx, value in enumerate(annual_shortfall_mwh, start=1):
        _ensure_non_negative_finite(float(value), f"annual_shortfall_mwh[{idx}]")
    _validate_price_inputs(price_inputs)

    if augmentation_costs_usd is not None and len(augmentation_costs_usd) != len(
        annual_delivered_mwh
    ):
        raise ValueError("augmentation_costs_usd must match number of years")
    if augmentation_costs_usd is not None:
        for idx, value in enumerate(augmentation_costs_usd, start=1):
            _ensure_non_negative_finite(float(value), f"augmentation_costs_usd[{idx}]")

    if annual_total_generation_mwh is not None and len(annual_total_generation_mwh) != len(
        annual_delivered_mwh
    ):
        raise ValueError("annual_total_generation_mwh must match number of years")
    if annual_total_generation_mwh is not None:
        for idx, value in enumerate(annual_total_generation_mwh, start=1):
            _ensure_non_negative_finite(float(value), f"annual_total_generation_mwh[{idx}]")

    years = len(annual_delivered_mwh)
    if years == 0:
        return CashFlowOutputs(float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))

    discounted_revenues = 0.0
    discounted_pv_revenue = 0.0
    discounted_wesm_value = 0.0
    cash_flows = [-_initial_project_spend(inputs)]

    fixed_opex_from_capex = inputs.capex_musd * (inputs.fixed_opex_pct_of_capex / 100.0)
    variable_opex_schedule = _resolve_variable_opex_schedule(years, inputs)
    blended_price = price_inputs.blended_price_usd_per_mwh
    contract_price = price_inputs.contract_price_usd_per_mwh
    pv_market_price = price_inputs.pv_market_price_usd_per_mwh
    if blended_price is not None:
        contract_price = float(blended_price)
        pv_market_price = float(blended_price)
    wesm_deficit_price = (
        float(price_inputs.wesm_deficit_price_usd_per_mwh)
        if price_inputs.wesm_deficit_price_usd_per_mwh is not None
        else None
    )
    wesm_surplus_price = (
        float(price_inputs.wesm_surplus_price_usd_per_mwh)
        if price_inputs.wesm_surplus_price_usd_per_mwh is not None
        else None
    )

    for year_idx in range(1, years + 1):
        firm_mwh = float(annual_delivered_mwh[year_idx - 1])
        bess_mwh = float(annual_bess_mwh[year_idx - 1])
        pv_delivered_mwh = float(annual_pv_delivered_mwh[year_idx - 1])
        pv_excess_mwh = float(annual_pv_excess_mwh[year_idx - 1])
        shortfall_mwh = float(annual_shortfall_mwh[year_idx - 1])
        factor = _discount_factor(inputs.discount_rate, year_idx)
        inflation_multiplier = (1.0 + inputs.inflation_rate) ** (year_idx - 1)

        annual_opex = 0.0
        if variable_opex_schedule is not None:
            if year_idx - 1 >= len(variable_opex_schedule):
                raise ValueError("variable_opex_schedule_usd must match the number of project years")
            annual_opex = float(variable_opex_schedule[year_idx - 1])
        elif inputs.variable_opex_usd_per_mwh is not None:
            opex_basis_mwh = firm_mwh + pv_excess_mwh
            if inputs.variable_opex_basis == "delivered":
                opex_basis_mwh = firm_mwh
            elif annual_total_generation_mwh is not None:
                opex_basis_mwh = float(annual_total_generation_mwh[year_idx - 1])
            annual_opex = opex_basis_mwh * float(inputs.variable_opex_usd_per_mwh) * inflation_multiplier
        else:
            annual_fixed_opex = (fixed_opex_from_capex + inputs.fixed_opex_musd) * 1_000_000
            annual_opex = annual_fixed_opex * inflation_multiplier
        augmentation_cost = 0.0
        if augmentation_costs_usd is not None:
            augmentation_cost = float(augmentation_costs_usd[year_idx - 1])

        if blended_price is not None:
            bess_revenue = firm_mwh * contract_price
            pv_delivered_revenue = 0.0
            pv_excess_revenue = pv_excess_mwh * pv_market_price
        else:
            bess_revenue = bess_mwh * contract_price
            pv_delivered_revenue = pv_delivered_mwh * pv_market_price
            pv_excess_revenue = 0.0
        wesm_shortfall_cost = 0.0
        wesm_surplus_revenue = 0.0
        if price_inputs.apply_wesm_to_shortfall and wesm_deficit_price is not None:
            wesm_shortfall_cost = shortfall_mwh * wesm_deficit_price
            if not price_inputs.sell_to_wesm:
                pv_excess_revenue = 0.0
        if price_inputs.sell_to_wesm:
            surplus_price = (
                wesm_surplus_price if wesm_surplus_price is not None else wesm_deficit_price
            )
            if surplus_price is not None:
                pv_excess_revenue = pv_excess_mwh * surplus_price
                wesm_surplus_revenue = pv_excess_revenue

        if price_inputs.escalate_with_inflation:
            bess_revenue *= inflation_multiplier
            pv_delivered_revenue *= inflation_multiplier
            pv_excess_revenue *= inflation_multiplier
            wesm_shortfall_cost *= inflation_multiplier
            wesm_surplus_revenue *= inflation_multiplier

        total_revenue = bess_revenue + pv_delivered_revenue + pv_excess_revenue - wesm_shortfall_cost
        discounted_revenues += total_revenue * factor
        discounted_pv_revenue += pv_excess_revenue * factor
        discounted_wesm_value += (wesm_surplus_revenue - wesm_shortfall_cost) * factor
        cash_flows.append(total_revenue - annual_opex - augmentation_cost)

    npv_usd = _compute_npv(cash_flows, inputs.discount_rate)
    irr_pct = _solve_irr_pct(cash_flows, max_iterations=max_iterations)

    return CashFlowOutputs(
        discounted_revenues_usd=discounted_revenues,
        discounted_pv_excess_revenue_usd=discounted_pv_revenue,
        discounted_wesm_value_usd=discounted_wesm_value,
        npv_usd=npv_usd,
        irr_pct=irr_pct,
    )


def compute_financing_cash_flows(
    annual_delivered_mwh: Sequence[float],
    annual_bess_mwh: Sequence[float],
    annual_pv_excess_mwh: Sequence[float],
    inputs: EconomicInputs,
    price_inputs: PriceInputs,
    annual_shortfall_mwh: Sequence[float] | None = None,
    augmentation_costs_usd: Sequence[float] | None = None,
    annual_total_generation_mwh: Sequence[float] | None = None,
    max_iterations: int = 200,
) -> FinancingOutputs:
    """Compute financing-aware cash flows, EBITDA, and project/equity IRR.

    Debt service is modeled as a level-payment amortizing loan sized using the
    provided debt ratio. Augmentation costs are treated as equity-funded CAPEX
    outflows in the year they occur. WACC is used to discount project cash
    flows for NPV; equity metrics are reported via IRR in the absence of a
    separate equity discount rate input. Variable OPEX can be applied to total
    generation when ``annual_total_generation_mwh`` is supplied.
    """

    inputs = normalize_economic_inputs(inputs)
    _validate_inputs(annual_delivered_mwh, annual_bess_mwh, inputs)
    if len(annual_pv_excess_mwh) != len(annual_delivered_mwh):
        raise ValueError("annual_pv_excess_mwh must match number of years")
    for idx, value in enumerate(annual_pv_excess_mwh, start=1):
        _ensure_non_negative_finite(float(value), f"annual_pv_excess_mwh[{idx}]")
    if annual_shortfall_mwh is None:
        annual_shortfall_mwh = [0.0 for _ in annual_delivered_mwh]
    if len(annual_shortfall_mwh) != len(annual_delivered_mwh):
        raise ValueError("annual_shortfall_mwh must match number of years")
    for idx, value in enumerate(annual_shortfall_mwh, start=1):
        _ensure_non_negative_finite(float(value), f"annual_shortfall_mwh[{idx}]")
    _validate_price_inputs(price_inputs)

    if augmentation_costs_usd is not None and len(augmentation_costs_usd) != len(
        annual_delivered_mwh
    ):
        raise ValueError("augmentation_costs_usd must match number of years")
    if augmentation_costs_usd is not None:
        for idx, value in enumerate(augmentation_costs_usd, start=1):
            _ensure_non_negative_finite(float(value), f"augmentation_costs_usd[{idx}]")

    if annual_total_generation_mwh is not None and len(annual_total_generation_mwh) != len(
        annual_delivered_mwh
    ):
        raise ValueError("annual_total_generation_mwh must match number of years")
    if annual_total_generation_mwh is not None:
        for idx, value in enumerate(annual_total_generation_mwh, start=1):
            _ensure_non_negative_finite(float(value), f"annual_total_generation_mwh[{idx}]")

    years = len(annual_delivered_mwh)
    if years == 0:
        return FinancingOutputs(
            ebitda_usd=float("nan"),
            ebitda_margin=float("nan"),
            project_npv_usd=float("nan"),
            project_irr_pct=float("nan"),
            equity_irr_pct=float("nan"),
        )

    total_capex = _initial_project_spend(inputs)
    debt_principal = total_capex * inputs.debt_ratio
    equity_contribution = total_capex - debt_principal
    debt_schedule = (
        _build_debt_service_schedule(
            debt_principal, inputs.cost_of_debt, int(inputs.tenor_years or 0), years
        )
        if inputs.debt_ratio > 0
        else [0.0 for _ in range(years)]
    )

    fixed_opex_from_capex = inputs.capex_musd * (inputs.fixed_opex_pct_of_capex / 100.0)
    variable_opex_schedule = _resolve_variable_opex_schedule(years, inputs)
    blended_price = price_inputs.blended_price_usd_per_mwh
    contract_price = price_inputs.contract_price_usd_per_mwh
    pv_market_price = price_inputs.pv_market_price_usd_per_mwh
    if blended_price is not None:
        contract_price = float(blended_price)
        pv_market_price = float(blended_price)
    wesm_deficit_price = (
        float(price_inputs.wesm_deficit_price_usd_per_mwh)
        if price_inputs.wesm_deficit_price_usd_per_mwh is not None
        else None
    )
    wesm_surplus_price = (
        float(price_inputs.wesm_surplus_price_usd_per_mwh)
        if price_inputs.wesm_surplus_price_usd_per_mwh is not None
        else None
    )

    total_revenue = 0.0
    total_ebitda = 0.0
    project_cash_flows = [-total_capex]
    equity_cash_flows = [-equity_contribution]

    for year_idx in range(1, years + 1):
        firm_mwh = float(annual_delivered_mwh[year_idx - 1])
        bess_mwh = float(annual_bess_mwh[year_idx - 1])
        pv_excess_mwh = float(annual_pv_excess_mwh[year_idx - 1])
        shortfall_mwh = float(annual_shortfall_mwh[year_idx - 1])
        inflation_multiplier = (1.0 + inputs.inflation_rate) ** (year_idx - 1)

        annual_opex = 0.0
        if variable_opex_schedule is not None:
            if year_idx - 1 >= len(variable_opex_schedule):
                raise ValueError("variable_opex_schedule_usd must match the number of project years")
            annual_opex = float(variable_opex_schedule[year_idx - 1])
        elif inputs.variable_opex_usd_per_mwh is not None:
            opex_basis_mwh = firm_mwh + pv_excess_mwh
            if inputs.variable_opex_basis == "delivered":
                opex_basis_mwh = firm_mwh
            elif annual_total_generation_mwh is not None:
                opex_basis_mwh = float(annual_total_generation_mwh[year_idx - 1])
            annual_opex = opex_basis_mwh * float(inputs.variable_opex_usd_per_mwh) * inflation_multiplier
        else:
            annual_fixed_opex = (fixed_opex_from_capex + inputs.fixed_opex_musd) * 1_000_000
            annual_opex = annual_fixed_opex * inflation_multiplier

        augmentation_cost = 0.0
        if augmentation_costs_usd is not None:
            augmentation_cost = float(augmentation_costs_usd[year_idx - 1])

        bess_revenue = firm_mwh * contract_price if blended_price is not None else bess_mwh * contract_price
        pv_revenue = pv_excess_mwh * pv_market_price
        wesm_shortfall_cost = 0.0
        wesm_surplus_revenue = 0.0
        if price_inputs.apply_wesm_to_shortfall and wesm_deficit_price is not None:
            wesm_shortfall_cost = shortfall_mwh * wesm_deficit_price
            pv_revenue = 0.0
            if price_inputs.sell_to_wesm:
                surplus_price = (
                    wesm_surplus_price if wesm_surplus_price is not None else wesm_deficit_price
                )
                wesm_surplus_revenue = pv_excess_mwh * surplus_price

        if price_inputs.escalate_with_inflation:
            bess_revenue *= inflation_multiplier
            pv_revenue *= inflation_multiplier
            wesm_shortfall_cost *= inflation_multiplier
            wesm_surplus_revenue *= inflation_multiplier

        revenue = bess_revenue + pv_revenue + wesm_surplus_revenue - wesm_shortfall_cost
        ebitda = revenue - annual_opex
        total_revenue += revenue
        total_ebitda += ebitda

        project_cash_flow = ebitda - augmentation_cost
        debt_service = debt_schedule[year_idx - 1]
        equity_cash_flow = project_cash_flow - debt_service

        project_cash_flows.append(project_cash_flow)
        equity_cash_flows.append(equity_cash_flow)

    ebitda_margin = total_ebitda / total_revenue if total_revenue > 0 else float("nan")
    project_npv = _compute_npv(project_cash_flows, inputs.wacc)
    project_irr_pct = _solve_irr_pct(project_cash_flows, max_iterations=max_iterations)
    equity_irr_pct = _solve_irr_pct(equity_cash_flows, max_iterations=max_iterations)

    return FinancingOutputs(
        ebitda_usd=total_ebitda,
        ebitda_margin=ebitda_margin,
        project_npv_usd=project_npv,
        project_irr_pct=project_irr_pct,
        equity_irr_pct=equity_irr_pct,
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
    annual_total_generation_mwh: Sequence[float] | None = None,
    compute_fn: Callable[..., EconomicOutputs] = compute_lcoe_lcos,
) -> EconomicOutputs:
    """Run ``compute_lcoe_lcos`` and add augmentation even if the function signature lags.

    ``compute_lcoe_lcos`` gained ``augmentation_costs_usd`` and optional total
    generation inputs, but deployed environments can lag behind the codebase.
    This wrapper first tries the new API. If ``compute_fn`` raises ``TypeError``
    due to keyword mismatches, it falls back to calling without unsupported
    args and then manually layers the discounted augmentation spend onto the
    outputs so LCOE/LCOS still reflect the added costs.
    """

    kwargs: dict[str, object] = {"augmentation_costs_usd": augmentation_costs_usd}
    if annual_total_generation_mwh is not None:
        kwargs["annual_total_generation_mwh"] = annual_total_generation_mwh

    try:
        return compute_fn(annual_delivered_mwh, annual_bess_mwh, inputs, **kwargs)
    except TypeError as exc:
        message = str(exc)
        if "annual_total_generation_mwh" in message:
            kwargs.pop("annual_total_generation_mwh", None)
            try:
                return compute_fn(annual_delivered_mwh, annual_bess_mwh, inputs, **kwargs)
            except TypeError as nested_exc:
                message = str(nested_exc)
                if "augmentation_costs_usd" not in message:
                    raise
        elif "augmentation_costs_usd" not in message:
            raise

        discounted_augmentation_costs = _discount_augmentation_costs(
            augmentation_costs_usd, inputs.discount_rate
        )
        base_kwargs: dict[str, object] = {}
        if "annual_total_generation_mwh" in kwargs:
            base_kwargs["annual_total_generation_mwh"] = kwargs["annual_total_generation_mwh"]
        base_outputs = compute_fn(annual_delivered_mwh, annual_bess_mwh, inputs, **base_kwargs)
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
    "FinancingOutputs",
    "compute_lcoe_lcos",
    "compute_cash_flows_and_irr",
    "compute_financing_cash_flows",
    "compute_lcoe_lcos_with_augmentation_fallback",
    "normalize_economic_inputs",
    "_discount_factor",
    "_ensure_non_negative_finite",
    "_validate_inputs",
    "_validate_price_inputs",
    "_build_debt_service_schedule",
    "_discount_augmentation_costs",
    "_compute_npv",
    "_solve_irr_pct",
    "estimate_augmentation_costs_by_year",
]
