"""Helper utilities that sit on top of the core economics module."""
from __future__ import annotations

from utils.economics import _discount_augmentation_costs, compute_lcoe_lcos_with_augmentation_fallback

__all__ = [
    "_discount_augmentation_costs",
    "compute_lcoe_lcos_with_augmentation_fallback",
]
