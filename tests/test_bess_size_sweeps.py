import math
import numpy as np
import pandas as pd

from app import SimConfig, SimulationOutput, SimulationSummary, YearResult
from utils.economics import EconomicInputs, PriceInputs
from utils.sweeps import (
    BessEconomicCandidate,
    compute_static_bess_sweep_economics,
    sweep_bess_sizes,
)


def _fake_simulation_factory(
    cycles_by_energy: dict[float, float], soh_by_energy: dict[float, float]
):
    def _simulate(cfg: SimConfig, pv_df, cycle_df, dod_override, need_logs=False):
        eq_cycles = cycles_by_energy[cfg.initial_usable_mwh]
        soh_total = soh_by_energy[cfg.initial_usable_mwh]
        year = YearResult(
            year_index=1,
            expected_firm_mwh=0.0,
            delivered_firm_mwh=0.0,
            shortfall_mwh=0.0,
            breach_days=0,
            charge_mwh=0.0,
            discharge_mwh=0.0,
            available_pv_mwh=0.0,
            pv_to_contract_mwh=0.0,
            bess_to_contract_mwh=0.0,
            avg_rte=1.0,
            eq_cycles=eq_cycles,
            cum_cycles=eq_cycles,
            soh_cycle=1.0,
            soh_calendar=1.0,
            soh_total=soh_total,
            eoy_usable_mwh=cfg.initial_usable_mwh,
            eoy_power_mw=cfg.initial_power_mw,
            pv_curtailed_mwh=0.0,
            flags={},
        )
        return SimulationOutput(
            cfg=cfg,
            discharge_hours_per_day=4.0,
            results=[year],
            monthly_results=[],
            first_year_logs=None,
            final_year_logs=None,
            hod_count=np.zeros(24),
            hod_sum_pv=np.zeros(24),
            hod_sum_pv_resource=np.zeros(24),
            hod_sum_bess=np.zeros(24),
            hod_sum_charge=np.zeros(24),
            augmentation_energy_added_mwh=[0.0 for _ in range(cfg.years)],
            augmentation_retired_energy_mwh=[0.0 for _ in range(cfg.years)],
            augmentation_events=0,
        )

    return _simulate


def _fake_summary_factory(
    compliance_by_energy: dict[float, float], cycles_by_energy: dict[float, float]
):
    def _summarize(sim_output: SimulationOutput) -> SimulationSummary:
        energy = sim_output.cfg.initial_usable_mwh
        return SimulationSummary(
            compliance=compliance_by_energy[energy],
            bess_share_of_firm=0.0,
            charge_discharge_ratio=1.0,
            pv_capture_ratio=1.0,
            discharge_capacity_factor=1.0,
            total_project_generation_mwh=sim_output.cfg.initial_power_mw * energy,
            bess_generation_mwh=0.0,
            pv_generation_mwh=0.0,
            pv_excess_mwh=0.0,
            bess_losses_mwh=0.0,
            total_shortfall_mwh=max(0.0, 100.0 - energy),
            avg_eq_cycles_per_year=cycles_by_energy[energy],
            cap_ratio_final=1.0,
        )

    return _summarize


def test_sweep_flags_feasible_best_candidate():
    energy_candidates = [20.0, 40.0, 60.0]
    cycles_by_energy = {20.0: 100.0, 40.0: 300.0, 60.0: 500.0}
    soh_by_energy = {20.0: 0.6, 40.0: 0.75, 60.0: 0.8}
    compliance_by_energy = {20.0: 70.0, 40.0: 90.0, 60.0: 95.0}

    base_cfg = SimConfig(years=1, max_cycles_per_day_cap=1.0)
    pv_df = pd.DataFrame({"pv_mw": [0.0]})
    cycle_df = pd.DataFrame()

    df = sweep_bess_sizes(
        base_cfg,
        pv_df,
        cycle_df,
        "Auto (infer)",
        energy_mwh_values=energy_candidates,
        fixed_power_mw=10.0,
        use_case="reliability",
        ranking_kpi=None,
        min_soh=0.65,
        simulate_fn=_fake_simulation_factory(cycles_by_energy, soh_by_energy),
        summarize_fn=_fake_summary_factory(compliance_by_energy, cycles_by_energy),
    )

    assert len(df) == len(energy_candidates)
    assert df.loc[df["is_best"], "energy_mwh"].iloc[0] == 40.0
    assert df.loc[df["energy_mwh"] == 60.0, "cycle_limit_hit"].iloc[0]
    assert df.loc[df["energy_mwh"] == 20.0, "soh_below_min"].iloc[0]
    assert not df.loc[df["energy_mwh"] == 60.0, "is_best"].iloc[0]


def test_sweep_computes_economics_when_inputs_provided():
    pv_df = pd.DataFrame({"pv_mw": [0.0]})
    cycle_df = pd.DataFrame()
    base_cfg = SimConfig(years=1)

    def _simulate(cfg: SimConfig, pv_df, cycle_df, dod_override, need_logs=False):
        year = YearResult(
            year_index=1,
            expected_firm_mwh=0.0,
            delivered_firm_mwh=100.0,
            shortfall_mwh=0.0,
            breach_days=0,
            charge_mwh=0.0,
            discharge_mwh=0.0,
            available_pv_mwh=0.0,
            pv_to_contract_mwh=0.0,
            bess_to_contract_mwh=40.0,
            avg_rte=1.0,
            eq_cycles=100.0,
            cum_cycles=100.0,
            soh_cycle=1.0,
            soh_calendar=1.0,
            soh_total=1.0,
            eoy_usable_mwh=cfg.initial_usable_mwh,
            eoy_power_mw=cfg.initial_power_mw,
            pv_curtailed_mwh=0.0,
            flags={},
        )
        return SimulationOutput(
            cfg=cfg,
            discharge_hours_per_day=4.0,
            results=[year],
            monthly_results=[],
            first_year_logs=None,
            final_year_logs=None,
            hod_count=np.zeros(24),
            hod_sum_pv=np.zeros(24),
            hod_sum_pv_resource=np.zeros(24),
            hod_sum_bess=np.zeros(24),
            hod_sum_charge=np.zeros(24),
            augmentation_energy_added_mwh=[1.0],
            augmentation_retired_energy_mwh=[0.0],
            augmentation_events=0,
        )

    def _summarize(sim_output: SimulationOutput) -> SimulationSummary:
        return SimulationSummary(
            compliance=95.0,
            bess_share_of_firm=0.4,
            charge_discharge_ratio=1.0,
            pv_capture_ratio=1.0,
            discharge_capacity_factor=1.0,
            total_project_generation_mwh=sim_output.cfg.initial_power_mw * sim_output.cfg.initial_usable_mwh,
            bess_generation_mwh=0.0,
            pv_generation_mwh=0.0,
            pv_excess_mwh=0.0,
            bess_losses_mwh=0.0,
            total_shortfall_mwh=0.0,
            avg_eq_cycles_per_year=100.0,
            cap_ratio_final=1.0,
        )

    econ_inputs = EconomicInputs(
        capex_musd=1.0,
        fixed_opex_pct_of_capex=0.0,
        fixed_opex_musd=0.0,
        inflation_rate=0.0,
        discount_rate=0.0,
    )

    df = sweep_bess_sizes(
        base_cfg,
        pv_df,
        cycle_df,
        "Auto (infer)",
        energy_mwh_values=[100.0],
        fixed_power_mw=10.0,
        use_case="reliability",
        ranking_kpi=None,
        min_soh=0.5,
        simulate_fn=_simulate,
        summarize_fn=_summarize,
        economics_inputs=econ_inputs,
    )

    assert abs(df.loc[0, "lcoe_usd_per_mwh"] - 8416.6667) < 1e-4
    assert abs(df.loc[0, "npv_costs_usd"] - 841666.6667) < 1e-2
    assert abs(df.loc[0, "irr_pct"] - 0.0) < 1e-6


def test_sweep_scales_economics_with_energy_size():
    pv_df = pd.DataFrame({"pv_mw": [0.0]})
    cycle_df = pd.DataFrame()
    base_cfg = SimConfig(years=1, initial_power_mw=10.0, initial_usable_mwh=50.0)

    def _simulate(cfg: SimConfig, pv_df, cycle_df, dod_override, need_logs=False):
        energy = cfg.initial_usable_mwh
        year = YearResult(
            year_index=1,
            expected_firm_mwh=0.0,
            delivered_firm_mwh=energy,
            shortfall_mwh=0.0,
            breach_days=0,
            charge_mwh=0.0,
            discharge_mwh=0.0,
            available_pv_mwh=0.0,
            pv_to_contract_mwh=0.0,
            bess_to_contract_mwh=energy,
            avg_rte=1.0,
            eq_cycles=0.0,
            cum_cycles=0.0,
            soh_cycle=1.0,
            soh_calendar=1.0,
            soh_total=1.0,
            eoy_usable_mwh=energy,
            eoy_power_mw=cfg.initial_power_mw,
            pv_curtailed_mwh=0.0,
            flags={},
        )
        return SimulationOutput(
            cfg=cfg,
            discharge_hours_per_day=4.0,
            results=[year],
            monthly_results=[],
            first_year_logs=None,
            final_year_logs=None,
            hod_count=np.zeros(24),
            hod_sum_pv=np.zeros(24),
            hod_sum_pv_resource=np.zeros(24),
            hod_sum_bess=np.zeros(24),
            hod_sum_charge=np.zeros(24),
            augmentation_energy_added_mwh=[0.0],
            augmentation_retired_energy_mwh=[0.0],
            augmentation_events=0,
        )

    def _summarize(sim_output: SimulationOutput) -> SimulationSummary:
        energy = sim_output.cfg.initial_usable_mwh
        return SimulationSummary(
            compliance=100.0,
            bess_share_of_firm=1.0,
            charge_discharge_ratio=1.0,
            pv_capture_ratio=1.0,
            discharge_capacity_factor=1.0,
            total_project_generation_mwh=energy,
            bess_generation_mwh=energy,
            pv_generation_mwh=0.0,
            pv_excess_mwh=0.0,
            bess_losses_mwh=0.0,
            total_shortfall_mwh=0.0,
            avg_eq_cycles_per_year=0.0,
            cap_ratio_final=1.0,
        )

    econ_inputs = EconomicInputs(
        capex_musd=2.0,
        fixed_opex_pct_of_capex=0.0,
        fixed_opex_musd=0.0,
        inflation_rate=0.0,
        discount_rate=0.0,
    )

    df = sweep_bess_sizes(
        base_cfg,
        pv_df,
        cycle_df,
        "Auto (infer)",
        energy_mwh_values=[50.0, 100.0],
        fixed_power_mw=10.0,
        use_case="reliability",
        ranking_kpi=None,
        min_soh=0.5,
        simulate_fn=_simulate,
        summarize_fn=_summarize,
        economics_inputs=econ_inputs,
    )

    base_cost = df.loc[df["energy_mwh"] == 50.0, "npv_costs_usd"].iloc[0]
    double_cost = df.loc[df["energy_mwh"] == 100.0, "npv_costs_usd"].iloc[0]

    assert double_cost > base_cost
    assert abs(double_cost - 2 * base_cost) < 1e-6


def test_static_economic_sweep_penalizes_deficits():
    candidates = [
        BessEconomicCandidate(
            energy_mwh=50.0,
            capex_musd=1.0,
            fixed_opex_musd=0.05,
            compliance_mwh=2500.0,
            deficit_mwh=0.0,
            surplus_mwh=200.0,
        ),
        BessEconomicCandidate(
            energy_mwh=80.0,
            capex_musd=1.0,
            fixed_opex_musd=0.05,
            compliance_mwh=2500.0,
            deficit_mwh=-500.0,
            surplus_mwh=200.0,
        ),
    ]

    economics_template = EconomicInputs(
        capex_musd=0.0,
        fixed_opex_pct_of_capex=0.0,
        fixed_opex_musd=0.0,
        inflation_rate=0.0,
        discount_rate=0.08,
    )

    price_inputs = PriceInputs(
        contract_price_usd_per_mwh=120.0,
        pv_market_price_usd_per_mwh=50.0,
    )

    df = compute_static_bess_sweep_economics(
        candidates,
        economics_template,
        price_inputs,
        wesm_price_usd_per_mwh=90.0,
        years=5,
    )

    base_npv = df.loc[df["deficit_mwh"] == 0.0, "npv_usd"].iloc[0]
    deficit_npv = df.loc[df["deficit_mwh"] < 0.0, "npv_usd"].iloc[0]
    base_irr = df.loc[df["deficit_mwh"] == 0.0, "irr_pct"].iloc[0]
    deficit_irr = df.loc[df["deficit_mwh"] < 0.0, "irr_pct"].iloc[0]

    assert base_npv > 0
    assert deficit_npv < base_npv
    assert base_irr > 0
    assert deficit_irr < base_irr


def test_static_economic_sweep_uses_blended_price() -> None:
    candidates = [
        BessEconomicCandidate(
            energy_mwh=50.0,
            capex_musd=0.0,
            fixed_opex_musd=0.0,
            compliance_mwh=100.0,
            deficit_mwh=0.0,
            surplus_mwh=50.0,
        )
    ]

    economics_template = EconomicInputs(
        capex_musd=0.0,
        fixed_opex_pct_of_capex=0.0,
        fixed_opex_musd=0.0,
        inflation_rate=0.0,
        discount_rate=0.0,
    )

    price_inputs = PriceInputs(
        contract_price_usd_per_mwh=120.0,
        pv_market_price_usd_per_mwh=60.0,
        blended_price_usd_per_mwh=50.0,
    )

    df = compute_static_bess_sweep_economics(
        candidates,
        economics_template,
        price_inputs,
        wesm_price_usd_per_mwh=90.0,
        years=1,
    )

    expected_revenue = (100.0 + 50.0) * 50.0
    assert math.isclose(df.loc[0, "npv_usd"], expected_revenue)
