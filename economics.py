"""Economic metrics for BESSLab simulations.

This module stays free of Streamlit/UI dependencies so it can be reused
from notebooks or other entrypoints. Provide annual energy series and
high-level cost assumptions to calculate LCOE/LCOS values.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


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
    _ensure_non_negative_finite(inputs.variable_opex_usd_per_mwh, "variable_opex_usd_per_mwh")
    _ensure_non_negative_finite(inputs.discount_rate, "discount_rate")


@dataclass
class EconomicInputs:
    """High-level project economics.

    All monetary values are expressed in USD to avoid mixing units. CAPEX and
    fixed OPEX can be entered in millions to keep UI inputs compact.
    """

    capex_musd: float
    fixed_opex_pct_of_capex: float
    fixed_opex_musd: float
    variable_opex_usd_per_mwh: float
    discount_rate: float


@dataclass
class EconomicOutputs:
    """Discounted cost and energy aggregates plus derived LCOE/LCOS."""

    discounted_costs_usd: float
    discounted_augmentation_costs_usd: float
    discounted_energy_mwh: float
    discounted_bess_energy_mwh: float
    lcoe_usd_per_mwh: float
    lcos_usd_per_mwh: float


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

    discounted_costs = inputs.capex_musd * 1_000_000
    discounted_augmentation_costs = 0.0
    discounted_energy = 0.0
    discounted_bess_energy = 0.0
    # fixed_opex_pct_of_capex is expressed as a percent (e.g., 2.5 = 2.5%)
    fixed_opex_from_capex = inputs.capex_musd * (inputs.fixed_opex_pct_of_capex / 100.0)

    for year_idx in range(1, years + 1):
        firm_mwh = float(annual_delivered_mwh[year_idx - 1])
        bess_mwh = float(annual_bess_mwh[year_idx - 1])
        factor = _discount_factor(inputs.discount_rate, year_idx)

        annual_fixed_opex = (fixed_opex_from_capex + inputs.fixed_opex_musd) * 1_000_000
        variable_opex = inputs.variable_opex_usd_per_mwh * firm_mwh
        augmentation_cost = 0.0
        if augmentation_costs_usd is not None:
            augmentation_cost = float(augmentation_costs_usd[year_idx - 1])

        discounted_augmentation_costs += augmentation_cost * factor
        discounted_costs += (annual_fixed_opex + variable_opex + augmentation_cost) * factor
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
