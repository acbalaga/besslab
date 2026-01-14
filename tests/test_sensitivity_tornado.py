import unittest

import pandas as pd

from frontend.ui.sensitivity_tornado import (
    apply_capex_delta,
    apply_opex_delta,
    apply_tariff_delta,
    prepare_tornado_data,
)
from utils.economics import EconomicInputs, PriceInputs, compute_financing_cash_flows, normalize_economic_inputs


class SensitivityTornadoTests(unittest.TestCase):
    def test_prepare_tornado_data_sorts_and_schema(self) -> None:
        table = pd.DataFrame(
            [
                {
                    "Lever": "A",
                    "Low change": -1.0,
                    "High change": 1.0,
                    "Low impact (pp)": -0.5,
                    "High impact (pp)": 0.5,
                    "Notes": "Minor",
                },
                {
                    "Lever": "B",
                    "Low change": -1.0,
                    "High change": 1.0,
                    "Low impact (pp)": -3.0,
                    "High impact (pp)": 2.0,
                    "Notes": "Major",
                },
            ]
        )

        result = prepare_tornado_data(table)

        expected_columns = {"Lever", "Notes", "Scenario", "Impact (pp)", "sort_key"}
        self.assertTrue(expected_columns.issubset(result.columns))
        self.assertTrue(result["sort_key"].is_monotonic_increasing)
        self.assertIn("Low impact", result["Scenario"].unique())
        self.assertIn("High impact", result["Scenario"].unique())

    def test_financial_levers_shift_pirr(self) -> None:
        annual_delivered = [1200.0, 1200.0]
        annual_bess = [900.0, 900.0]
        annual_pv_excess = [100.0, 100.0]
        annual_shortfall = [0.0, 0.0]
        annual_total_generation = [1300.0, 1300.0]

        econ_inputs = EconomicInputs(
            capex_musd=1.0,
            fixed_opex_pct_of_capex=2.0,
            fixed_opex_musd=0.0,
            inflation_rate=0.0,
            discount_rate=0.05,
            wacc=0.08,
        )
        price_inputs = PriceInputs(
            contract_price_usd_per_mwh=110.0,
            escalate_with_inflation=False,
        )

        def compute_pirr(inputs: EconomicInputs, prices: PriceInputs) -> float:
            normalized_inputs = normalize_economic_inputs(inputs)
            outputs = compute_financing_cash_flows(
                annual_delivered,
                annual_bess,
                annual_pv_excess,
                normalized_inputs,
                prices,
                annual_shortfall_mwh=annual_shortfall,
                annual_total_generation_mwh=annual_total_generation,
            )
            return outputs.project_irr_pct

        baseline = compute_pirr(econ_inputs, price_inputs)
        capex_pirr = compute_pirr(apply_capex_delta(econ_inputs, 10.0), price_inputs)
        opex_pirr = compute_pirr(apply_opex_delta(econ_inputs, 10.0), price_inputs)
        tariff_pirr = compute_pirr(econ_inputs, apply_tariff_delta(price_inputs, 10.0))

        self.assertNotEqual(baseline, capex_pirr)
        self.assertNotEqual(baseline, opex_pirr)
        self.assertNotEqual(baseline, tariff_pirr)
        self.assertLess(capex_pirr, baseline)
        self.assertLess(opex_pirr, baseline)
        self.assertGreater(tariff_pirr, baseline)
