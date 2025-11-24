import math
import unittest

from economics import EconomicInputs, compute_lcoe_lcos


class EconomicModuleTests(unittest.TestCase):
    def test_known_discounted_case(self) -> None:
        inputs = EconomicInputs(
            capex_musd=100.0,
            fixed_opex_pct_of_capex=2.0,
            fixed_opex_musd=1.0,
            variable_opex_usd_per_mwh=10.0,
            discount_rate=0.05,
        )
        outputs = compute_lcoe_lcos(
            annual_delivered_mwh=[100_000, 100_000, 100_000],
            annual_bess_mwh=[50_000, 50_000, 50_000],
            inputs=inputs,
        )

        self.assertAlmostEqual(outputs.discounted_costs_usd, 110_892_992.1175, places=3)
        self.assertAlmostEqual(outputs.discounted_energy_mwh, 272_324.8029, places=3)
        self.assertAlmostEqual(outputs.discounted_bess_energy_mwh, 136_162.4015, places=3)
        self.assertAlmostEqual(outputs.lcoe_usd_per_mwh, 407.2086, places=3)
        self.assertAlmostEqual(outputs.lcos_usd_per_mwh, 814.4171, places=3)

    def test_empty_series_returns_nan(self) -> None:
        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            variable_opex_usd_per_mwh=0.0,
            discount_rate=0.0,
        )
        outputs = compute_lcoe_lcos([], [], inputs)
        self.assertTrue(math.isnan(outputs.discounted_costs_usd))
        self.assertTrue(math.isnan(outputs.discounted_energy_mwh))
        self.assertTrue(math.isnan(outputs.discounted_bess_energy_mwh))
        self.assertTrue(math.isnan(outputs.lcoe_usd_per_mwh))
        self.assertTrue(math.isnan(outputs.lcos_usd_per_mwh))

    def test_mismatched_lengths_raise(self) -> None:
        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            variable_opex_usd_per_mwh=0.0,
            discount_rate=0.0,
        )
        with self.assertRaises(ValueError):
            compute_lcoe_lcos([1.0, 2.0], [1.0], inputs)

    def test_invalid_energy_values_raise(self) -> None:
        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            variable_opex_usd_per_mwh=0.0,
            discount_rate=0.0,
        )
        with self.assertRaises(ValueError):
            compute_lcoe_lcos([1.0, -1.0], [1.0, 1.0], inputs)

    def test_invalid_financial_values_raise(self) -> None:
        inputs = EconomicInputs(
            capex_musd=-1.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            variable_opex_usd_per_mwh=0.0,
            discount_rate=0.0,
        )
        with self.assertRaises(ValueError):
            compute_lcoe_lcos([1.0], [1.0], inputs)


if __name__ == "__main__":
    unittest.main()
