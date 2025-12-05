"""Legacy wrapper modules maintained for backward compatibility.

These thin re-export modules live under :mod:`utils.legacy` to keep the
repository root focused on the Streamlit entrypoint (:mod:`app`). External
callers can import the wrappers from here without touching the main folder.
"""
from utils.legacy.bess_size_sweeps import run_candidate_simulation, sweep_bess_sizes
from utils.legacy.economics import (
    EconomicInputs,
    EconomicOutputs,
    _discount_factor,
    _discount_augmentation_costs,
    _ensure_non_negative_finite,
    _validate_inputs,
    compute_lcoe_lcos,
    compute_lcoe_lcos_with_augmentation_fallback,
)
from utils.legacy.economics_helpers import _discount_augmentation_costs as legacy_discount_augmentation_costs
from utils.legacy.economics_helpers import compute_lcoe_lcos_with_augmentation_fallback as legacy_compute_lcoe_lcos_with_augmentation_fallback
from utils.legacy.sensitivity_sweeps import (
    build_soc_windows,
    generate_values,
    run_sensitivity_grid,
)

__all__ = [
    "run_candidate_simulation",
    "sweep_bess_sizes",
    "EconomicInputs",
    "EconomicOutputs",
    "compute_lcoe_lcos",
    "compute_lcoe_lcos_with_augmentation_fallback",
    "_discount_factor",
    "_discount_augmentation_costs",
    "_ensure_non_negative_finite",
    "_validate_inputs",
    "legacy_discount_augmentation_costs",
    "legacy_compute_lcoe_lcos_with_augmentation_fallback",
    "build_soc_windows",
    "generate_values",
    "run_sensitivity_grid",
]
