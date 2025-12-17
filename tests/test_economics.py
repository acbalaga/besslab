import math
import unittest

from utils.economics import (
    CashFlowOutputs,
    DEVEX_COST_USD,
    EconomicInputs,
    PriceInputs,
    compute_cash_flows_and_irr,
    compute_lcoe_lcos,
)


class EconomicModuleTests(unittest.TestCase):
    def test_known_discounted_case(self) -> None:
        inputs = EconomicInputs(
            capex_musd=100.0,
            fixed_opex_pct_of_capex=2.0,
            fixed_opex_musd=1.0,
            inflation_rate=0.03,
            discount_rate=0.05,
        )
        outputs = compute_lcoe_lcos(
            annual_delivered_mwh=[100_000, 100_000, 100_000],
            annual_bess_mwh=[50_000, 50_000, 50_000],
            inputs=inputs,
        )

        self.assertAlmostEqual(outputs.discounted_costs_usd, 108_409_199.8704, places=3)
        self.assertAlmostEqual(outputs.discounted_energy_mwh, 272_324.8029, places=3)
        self.assertAlmostEqual(outputs.discounted_bess_energy_mwh, 136_162.4015, places=3)
        self.assertAlmostEqual(outputs.lcoe_usd_per_mwh, 398.0879, places=3)
        self.assertAlmostEqual(outputs.lcos_usd_per_mwh, 796.1757, places=3)

    def test_empty_series_returns_nan(self) -> None:
        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            inflation_rate=0.0,
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
            inflation_rate=0.0,
            discount_rate=0.0,
        )
        with self.assertRaises(ValueError):
            compute_lcoe_lcos([1.0, 2.0], [1.0], inputs)

    def test_invalid_energy_values_raise(self) -> None:
        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            inflation_rate=0.0,
            discount_rate=0.0,
        )
        with self.assertRaises(ValueError):
            compute_lcoe_lcos([1.0, -1.0], [1.0, 1.0], inputs)

    def test_invalid_financial_values_raise(self) -> None:
        inputs = EconomicInputs(
            capex_musd=-1.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            inflation_rate=0.0,
            discount_rate=0.0,
        )
        with self.assertRaises(ValueError):
            compute_lcoe_lcos([1.0], [1.0], inputs)

    def test_augmentation_costs_are_discounted_and_added(self) -> None:
        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            inflation_rate=0.0,
            discount_rate=0.10,
        )
        outputs = compute_lcoe_lcos(
            annual_delivered_mwh=[100.0, 100.0],
            annual_bess_mwh=[100.0, 100.0],
            inputs=inputs,
            augmentation_costs_usd=[1_000_000.0, 0.0],
        )

        expected_discounted_aug_costs = 1_000_000.0 / 1.1
        self.assertAlmostEqual(outputs.discounted_augmentation_costs_usd, expected_discounted_aug_costs)
        self.assertAlmostEqual(outputs.discounted_costs_usd, expected_discounted_aug_costs)
        expected_discounted_energy = (100.0 / 1.1) + (100.0 / (1.1**2))
        self.assertAlmostEqual(outputs.lcoe_usd_per_mwh, expected_discounted_aug_costs / expected_discounted_energy)

    def test_devex_is_added_upfront(self) -> None:
        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            inflation_rate=0.0,
            discount_rate=0.0,
            include_devex_year0=True,
        )
        outputs = compute_lcoe_lcos(
            annual_delivered_mwh=[100.0],
            annual_bess_mwh=[100.0],
            inputs=inputs,
        )

        self.assertAlmostEqual(outputs.discounted_costs_usd, DEVEX_COST_USD)
        self.assertAlmostEqual(outputs.lcoe_usd_per_mwh, DEVEX_COST_USD / 100.0)

    def test_devex_respects_custom_usd_amount(self) -> None:
        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            inflation_rate=0.0,
            discount_rate=0.0,
            devex_cost_usd=2_000_000.0,
            include_devex_year0=True,
        )

        outputs = compute_lcoe_lcos(
            annual_delivered_mwh=[200.0],
            annual_bess_mwh=[200.0],
            inputs=inputs,
        )

        self.assertAlmostEqual(outputs.discounted_costs_usd, inputs.devex_cost_usd)
        self.assertAlmostEqual(outputs.lcoe_usd_per_mwh, inputs.devex_cost_usd / 200.0)

    def test_cash_flows_and_irr_handles_contract_and_pv_revenue(self) -> None:
        inputs = EconomicInputs(
            capex_musd=0.10,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            inflation_rate=0.0,
            discount_rate=0.05,
        )
        price_inputs = PriceInputs(
            contract_price_usd_per_mwh=110.0,
            pv_market_price_usd_per_mwh=35.0,
            escalate_with_inflation=False,
        )

        cashflow_outputs = compute_cash_flows_and_irr(
            annual_delivered_mwh=[1_000.0],
            annual_bess_mwh=[1_000.0],
            annual_pv_excess_mwh=[100.0],
            inputs=inputs,
            price_inputs=price_inputs,
        )

        expected_revenue = 1_000.0 * 110.0
        expected_pv_revenue = 100.0 * 35.0
        expected_discounted_revenue = (expected_revenue + expected_pv_revenue) / 1.05
        expected_npv = -100_000.0 + expected_discounted_revenue

        self.assertIsInstance(cashflow_outputs, CashFlowOutputs)
        self.assertAlmostEqual(cashflow_outputs.discounted_revenues_usd, expected_discounted_revenue)
        self.assertAlmostEqual(
            cashflow_outputs.discounted_pv_excess_revenue_usd, expected_pv_revenue / 1.05
        )
        self.assertAlmostEqual(cashflow_outputs.discounted_wesm_value_usd, 0.0)
        self.assertAlmostEqual(cashflow_outputs.npv_usd, expected_npv)
        self.assertAlmostEqual(cashflow_outputs.irr_pct, 13.5, places=3)

    def test_cash_flows_and_irr_respects_blended_price_override(self) -> None:
        inputs = EconomicInputs(
            capex_musd=0.10,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            inflation_rate=0.0,
            discount_rate=0.05,
        )
        price_inputs = PriceInputs(
            contract_price_usd_per_mwh=110.0,
            pv_market_price_usd_per_mwh=35.0,
            escalate_with_inflation=False,
            blended_price_usd_per_mwh=80.0,
        )

        cashflow_outputs = compute_cash_flows_and_irr(
            annual_delivered_mwh=[1_000.0],
            annual_bess_mwh=[1_000.0],
            annual_pv_excess_mwh=[100.0],
            inputs=inputs,
            price_inputs=price_inputs,
        )

        expected_revenue = (1_000.0 + 100.0) * 80.0
        expected_pv_revenue = 100.0 * 80.0
        expected_discounted_revenue = expected_revenue / 1.05
        expected_npv = -100_000.0 + expected_discounted_revenue

        self.assertAlmostEqual(cashflow_outputs.discounted_revenues_usd, expected_discounted_revenue)
        self.assertAlmostEqual(
            cashflow_outputs.discounted_pv_excess_revenue_usd, expected_pv_revenue / 1.05
        )
        self.assertAlmostEqual(cashflow_outputs.discounted_wesm_value_usd, 0.0)
        self.assertAlmostEqual(cashflow_outputs.npv_usd, expected_npv)
        self.assertAlmostEqual(cashflow_outputs.irr_pct, -12.0, places=1)

    def test_blended_price_monetizes_pv_contribution_to_contract(self) -> None:
        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            inflation_rate=0.0,
            discount_rate=0.0,
        )
        price_inputs = PriceInputs(
            contract_price_usd_per_mwh=0.0,
            pv_market_price_usd_per_mwh=0.0,
            blended_price_usd_per_mwh=50.0,
        )

        cashflow_outputs = compute_cash_flows_and_irr(
            annual_delivered_mwh=[1_200.0],  # PV + BESS meeting firm deliveries
            annual_bess_mwh=[1_000.0],
            annual_pv_excess_mwh=[50.0],
            inputs=inputs,
            price_inputs=price_inputs,
        )

        expected_revenue = (1_200.0 + 50.0) * 50.0
        self.assertAlmostEqual(cashflow_outputs.discounted_revenues_usd, expected_revenue)
        self.assertAlmostEqual(cashflow_outputs.discounted_pv_excess_revenue_usd, 50.0 * 50.0)
        self.assertAlmostEqual(cashflow_outputs.discounted_wesm_value_usd, 0.0)

    def test_wesm_shortfall_costs_reduce_revenue(self) -> None:
        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            inflation_rate=0.0,
            discount_rate=0.10,
        )
        price_inputs = PriceInputs(
            contract_price_usd_per_mwh=100.0,
            pv_market_price_usd_per_mwh=0.0,
            wesm_price_usd_per_mwh=120.0,
            apply_wesm_to_shortfall=True,
            sell_to_wesm=True,
        )

        outputs = compute_cash_flows_and_irr(
            annual_delivered_mwh=[950.0],
            annual_bess_mwh=[950.0],
            annual_pv_excess_mwh=[0.0],
            inputs=inputs,
            price_inputs=price_inputs,
            annual_shortfall_mwh=[50.0],
        )

        expected_wesm_cost = -50.0 * 120.0
        expected_revenue = (950.0 * 100.0) + expected_wesm_cost
        expected_discounted_revenue = expected_revenue / 1.10

        self.assertAlmostEqual(outputs.discounted_wesm_value_usd, expected_wesm_cost / 1.10)
        self.assertAlmostEqual(outputs.discounted_revenues_usd, expected_discounted_revenue)
        self.assertAlmostEqual(outputs.npv_usd, expected_discounted_revenue)

    def test_wesm_pv_surplus_counted_only_when_selling(self) -> None:
        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            inflation_rate=0.0,
            discount_rate=0.0,
        )

        price_inputs_sell = PriceInputs(
            contract_price_usd_per_mwh=0.0,
            pv_market_price_usd_per_mwh=0.0,
            wesm_price_usd_per_mwh=80.0,
            apply_wesm_to_shortfall=True,
            sell_to_wesm=True,
        )
        outputs_sell = compute_cash_flows_and_irr(
            annual_delivered_mwh=[0.0],
            annual_bess_mwh=[0.0],
            annual_pv_excess_mwh=[25.0],
            inputs=inputs,
            price_inputs=price_inputs_sell,
            annual_shortfall_mwh=[0.0],
        )

        expected_revenue = 25.0 * 80.0
        self.assertAlmostEqual(outputs_sell.discounted_wesm_value_usd, expected_revenue)
        self.assertAlmostEqual(outputs_sell.discounted_revenues_usd, expected_revenue)
        self.assertAlmostEqual(outputs_sell.discounted_pv_excess_revenue_usd, expected_revenue)

        price_inputs_hold = PriceInputs(
            contract_price_usd_per_mwh=0.0,
            pv_market_price_usd_per_mwh=0.0,
            wesm_price_usd_per_mwh=80.0,
            apply_wesm_to_shortfall=True,
            sell_to_wesm=False,
        )
        outputs_hold = compute_cash_flows_and_irr(
            annual_delivered_mwh=[0.0],
            annual_bess_mwh=[0.0],
            annual_pv_excess_mwh=[25.0],
            inputs=inputs,
            price_inputs=price_inputs_hold,
            annual_shortfall_mwh=[0.0],
        )

        self.assertAlmostEqual(outputs_hold.discounted_wesm_value_usd, 0.0)
        self.assertAlmostEqual(outputs_hold.discounted_revenues_usd, 0.0)
        self.assertAlmostEqual(outputs_hold.discounted_pv_excess_revenue_usd, 0.0)

    def test_devex_flows_into_initial_cash_flow_and_npv(self) -> None:
        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            inflation_rate=0.0,
            discount_rate=0.0,
            include_devex_year0=True,
        )
        price_inputs = PriceInputs(
            contract_price_usd_per_mwh=50.0,
            pv_market_price_usd_per_mwh=0.0,
            escalate_with_inflation=False,
        )

        cashflow_outputs = compute_cash_flows_and_irr(
            annual_delivered_mwh=[1_000.0],
            annual_bess_mwh=[1_000.0],
            annual_pv_excess_mwh=[0.0],
            inputs=inputs,
            price_inputs=price_inputs,
        )

        expected_revenue = 50.0 * 1_000.0
        self.assertAlmostEqual(cashflow_outputs.npv_usd, expected_revenue - DEVEX_COST_USD)
        self.assertTrue(math.isnan(cashflow_outputs.irr_pct) or cashflow_outputs.irr_pct < 0)

    def test_variable_opex_per_mwh_overrides_fixed_costs(self) -> None:
        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=50.0,  # large value to ensure override
            fixed_opex_musd=10.0,
            inflation_rate=0.0,
            discount_rate=0.0,
            variable_opex_usd_per_mwh=2.0,
        )

        outputs = compute_lcoe_lcos(
            annual_delivered_mwh=[100.0],
            annual_bess_mwh=[100.0],
            inputs=inputs,
        )

        self.assertAlmostEqual(outputs.discounted_costs_usd, 200.0)
        self.assertAlmostEqual(outputs.lcoe_usd_per_mwh, 2.0)

    def test_variable_schedule_takes_precedence(self) -> None:
        inputs = EconomicInputs(
            capex_musd=0.0,
            fixed_opex_pct_of_capex=0.0,
            fixed_opex_musd=0.0,
            inflation_rate=0.0,
            discount_rate=0.0,
            variable_opex_usd_per_mwh=10.0,
            variable_opex_schedule_usd=(50.0,),
        )

        outputs = compute_lcoe_lcos(
            annual_delivered_mwh=[100.0],
            annual_bess_mwh=[100.0],
            inputs=inputs,
        )

        self.assertAlmostEqual(outputs.discounted_costs_usd, 50.0)
        self.assertAlmostEqual(outputs.lcoe_usd_per_mwh, 0.5)


if __name__ == "__main__":
    unittest.main()
