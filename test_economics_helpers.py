import math
import unittest
from typing import Sequence

from economics import EconomicInputs, EconomicOutputs
from economics_helpers import compute_lcoe_lcos_with_augmentation_fallback


class EconomicsHelperTests(unittest.TestCase):
    def test_falls_back_when_compute_lacks_augmentation_keyword(self) -> None:
        def legacy_compute(
            annual_delivered_mwh: Sequence[float],
            annual_bess_mwh: Sequence[float],
            inputs: EconomicInputs,
        ) -> EconomicOutputs:
            return EconomicOutputs(
                discounted_costs_usd=0.0,
                discounted_augmentation_costs_usd=0.0,
                discounted_energy_mwh=sum(annual_delivered_mwh),
                discounted_bess_energy_mwh=sum(annual_bess_mwh),
                lcoe_usd_per_mwh=float("nan"),
                lcos_usd_per_mwh=float("nan"),
            )

        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            variable_opex_usd_per_mwh=0.0,
            discount_rate=0.10,
        )

        outputs = compute_lcoe_lcos_with_augmentation_fallback(
            [100.0, 100.0],
            [50.0, 50.0],
            inputs,
            augmentation_costs_usd=[1_000_000.0, 0.0],
            compute_fn=legacy_compute,
        )

        discounted_aug = 1_000_000.0 / 1.1
        self.assertAlmostEqual(outputs.discounted_augmentation_costs_usd, discounted_aug)
        self.assertAlmostEqual(outputs.discounted_costs_usd, discounted_aug)
        expected_energy = 200.0  # legacy_compute uses undiscounted sums
        self.assertAlmostEqual(outputs.lcoe_usd_per_mwh, discounted_aug / expected_energy)
        self.assertFalse(math.isnan(outputs.lcoe_usd_per_mwh))

    def test_fallback_only_triggers_for_augmentation_keyword_errors(self) -> None:
        call_count = {"attempts": 0}

        def maybe_legacy_compute(
            annual_delivered_mwh: Sequence[float],
            annual_bess_mwh: Sequence[float],
            inputs: EconomicInputs,
            augmentation_costs_usd: Sequence[float] | None = None,
        ) -> EconomicOutputs:
            call_count["attempts"] += 1
            if augmentation_costs_usd is not None:
                raise TypeError("compute_lcoe_lcos() got an unexpected keyword argument 'augmentation_costs_usd'")

            return EconomicOutputs(
                discounted_costs_usd=5.0,
                discounted_augmentation_costs_usd=0.0,
                discounted_energy_mwh=sum(annual_delivered_mwh),
                discounted_bess_energy_mwh=sum(annual_bess_mwh),
                lcoe_usd_per_mwh=float("nan"),
                lcos_usd_per_mwh=float("nan"),
            )

        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            variable_opex_usd_per_mwh=0.0,
            discount_rate=0.10,
        )

        outputs = compute_lcoe_lcos_with_augmentation_fallback(
            [100.0, 100.0],
            [50.0, 50.0],
            inputs,
            augmentation_costs_usd=[1_000_000.0, 0.0],
            compute_fn=maybe_legacy_compute,
        )

        self.assertEqual(call_count["attempts"], 2)  # tried with augmentation then without
        discounted_aug = 1_000_000.0 / 1.1
        self.assertAlmostEqual(outputs.discounted_augmentation_costs_usd, discounted_aug)
        self.assertAlmostEqual(outputs.discounted_costs_usd, 5.0 + discounted_aug)

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
            variable_opex_usd_per_mwh=0.0,
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
