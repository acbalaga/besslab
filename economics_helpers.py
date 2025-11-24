"""Helper utilities that sit on top of the core economics module.

These helpers keep the Streamlit app resilient to API mismatches (e.g.,
deployed code picking up an older economics module) while still applying
augmentation costs in LCOE/LCOS calculations.
"""
from __future__ import annotations

from typing import Callable, Sequence

from economics import EconomicInputs, EconomicOutputs, compute_lcoe_lcos


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
    except TypeError:
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

