"""Economic metrics for BESSLab simulations.

This module stays free of Streamlit/UI dependencies so it can be reused
from notebooks or other entrypoints. Provide annual energy series and
high-level cost assumptions to calculate LCOE/LCOS values.
"""
from __future__ import annotations

from utils.economics import (
    EconomicInputs,
    EconomicOutputs,
    _discount_factor,
    _discount_augmentation_costs,
    _ensure_non_negative_finite,
    _validate_inputs,
    compute_lcoe_lcos,
    compute_lcoe_lcos_with_augmentation_fallback,
)

__all__ = [
    "EconomicInputs",
    "EconomicOutputs",
    "compute_lcoe_lcos",
    "compute_lcoe_lcos_with_augmentation_fallback",
    "_discount_factor",
    "_discount_augmentation_costs",
    "_ensure_non_negative_finite",
    "_validate_inputs",
]
