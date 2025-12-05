"""Tests for economics_helpers helper utilities."""
import unittest
from typing import Sequence

from utils.economics import EconomicInputs, EconomicOutputs
from utils.economics import _discount_augmentation_costs, compute_lcoe_lcos_with_augmentation_fallback


class EconomicsHelperTests(unittest.TestCase):
    def test_discount_augmentation_costs_handles_none_and_values(self) -> None:
        self.assertEqual(_discount_augmentation_costs(None, 0.05), 0.0)

        augmentation_costs = [1_000_000.0, 500_000.0, 250_000.0]
        discount_rate = 0.10

        expected_discounted = sum(
            cost / ((1.0 + discount_rate) ** year)
            for year, cost in enumerate(augmentation_costs, start=1)
        )

        self.assertAlmostEqual(
            _discount_augmentation_costs(augmentation_costs, discount_rate), expected_discounted
        )

    def test_fallback_passthrough_when_no_augmentation_costs(self) -> None:
        def legacy_compute(
            annual_delivered_mwh: Sequence[float],
            annual_bess_mwh: Sequence[float],
            inputs: EconomicInputs,
        ) -> EconomicOutputs:
            return EconomicOutputs(
                discounted_costs_usd=10_000.0,
                discounted_augmentation_costs_usd=0.0,
                discounted_energy_mwh=sum(annual_delivered_mwh),
                discounted_bess_energy_mwh=sum(annual_bess_mwh),
                lcoe_usd_per_mwh=1_000.0,
                lcos_usd_per_mwh=2_000.0,
            )

        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            inflation_rate=0.0,
            discount_rate=0.05,
        )

        outputs = compute_lcoe_lcos_with_augmentation_fallback(
            annual_delivered_mwh=[100.0, 100.0],
            annual_bess_mwh=[60.0, 40.0],
            inputs=inputs,
            augmentation_costs_usd=None,
            compute_fn=legacy_compute,
        )

        self.assertEqual(outputs.discounted_augmentation_costs_usd, 0.0)
        self.assertEqual(outputs.discounted_costs_usd, 10_000.0)
        self.assertAlmostEqual(outputs.lcoe_usd_per_mwh, 10_000.0 / 200.0)
        self.assertAlmostEqual(outputs.lcos_usd_per_mwh, 10_000.0 / 100.0)

    def test_fallback_adjusts_costs_and_lcoe_when_aug_included(self) -> None:
        def legacy_compute(
            annual_delivered_mwh: Sequence[float],
            annual_bess_mwh: Sequence[float],
            inputs: EconomicInputs,
        ) -> EconomicOutputs:
            return EconomicOutputs(
                discounted_costs_usd=50_000.0,
                discounted_augmentation_costs_usd=0.0,
                discounted_energy_mwh=sum(annual_delivered_mwh) / 2.0,
                discounted_bess_energy_mwh=sum(annual_bess_mwh) / 2.0,
                lcoe_usd_per_mwh=0.0,
                lcos_usd_per_mwh=0.0,
            )

        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            inflation_rate=0.0,
            discount_rate=0.10,
        )

        augmentation_costs = [1_000_000.0, 250_000.0]
        outputs = compute_lcoe_lcos_with_augmentation_fallback(
            annual_delivered_mwh=[100.0, 80.0],
            annual_bess_mwh=[70.0, 60.0],
            inputs=inputs,
            augmentation_costs_usd=augmentation_costs,
            compute_fn=legacy_compute,
        )

        expected_discounted_aug = _discount_augmentation_costs(augmentation_costs, inputs.discount_rate)
        expected_energy = (100.0 + 80.0) / 2.0
        expected_bess_energy = (70.0 + 60.0) / 2.0
        expected_costs = 50_000.0 + expected_discounted_aug

        self.assertAlmostEqual(outputs.discounted_augmentation_costs_usd, expected_discounted_aug)
        self.assertAlmostEqual(outputs.discounted_costs_usd, expected_costs)
        self.assertAlmostEqual(outputs.lcoe_usd_per_mwh, expected_costs / expected_energy)
        self.assertAlmostEqual(outputs.lcos_usd_per_mwh, expected_costs / expected_bess_energy)

    def test_typeerrors_from_validation_are_not_masked(self) -> None:
        def validating_compute(
            annual_delivered_mwh: Sequence[float],
            annual_bess_mwh: Sequence[float],
            inputs: EconomicInputs,
            augmentation_costs_usd: Sequence[float] | None = None,
        ) -> EconomicOutputs:
            if augmentation_costs_usd is None:
                raise TypeError("augmentation_costs_usd must be provided")

            return EconomicOutputs(
                discounted_costs_usd=0.0,
                discounted_augmentation_costs_usd=0.0,
                discounted_energy_mwh=0.0,
                discounted_bess_energy_mwh=0.0,
                lcoe_usd_per_mwh=float("nan"),
                lcos_usd_per_mwh=float("nan"),
            )

        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            inflation_rate=0.0,
            discount_rate=0.10,
        )

        with self.assertRaises(TypeError):
            compute_lcoe_lcos_with_augmentation_fallback(
                [100.0, 100.0],
                [50.0, 50.0],
                inputs,
                augmentation_costs_usd=None,
                compute_fn=validating_compute,
            )


if __name__ == "__main__":
    unittest.main()
