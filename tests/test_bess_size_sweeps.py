import logging
import math
import time
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import pytest

from services.simulation_core import SimConfig, SimulationOutput, SimulationSummary, YearResult
from utils.economics import EconomicInputs, PriceInputs
from utils.sweeps import (
    BessEconomicCandidate,
    _compute_candidate_economics,
    _main_example,
    compute_static_bess_sweep_economics,
    sweep_bess_sizes,
)


@dataclass
class _StubCfg:
    initial_usable_mwh: float
    initial_power_mw: float


@dataclass
class _StubResult:
    delivered_firm_mwh: float
    bess_to_contract_mwh: float
    pv_curtailed_mwh: float
    shortfall_mwh: float
    eq_cycles: float = 0.0
    soh_total: float = 1.0


@dataclass
class _StubSimOutput:
    cfg: _StubCfg
    results: list[_StubResult]
    augmentation_energy_added_mwh: list[float] | None = field(default_factory=list)


def test_sweep_validates_power_duration_inputs():
    base_cfg = SimConfig(years=1)
    pv_df = pd.DataFrame({"pv_mw": [0.0]})
    cycle_df = pd.DataFrame()

    with pytest.raises(ValueError):
        sweep_bess_sizes(
            base_cfg,
            pv_df,
            cycle_df,
            "Auto (infer)",
            power_mw_values=[-5.0],
            duration_h_values=[2.0],
        )

    with pytest.raises(ValueError):
        sweep_bess_sizes(
            base_cfg,
            pv_df,
            cycle_df,
            "Auto (infer)",
            energy_mwh_values=[50.0],
            fixed_power_mw=0.0,
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


def test_sweep_includes_margin_outputs():
    energy_candidates = [20.0, 40.0]
    cycles_by_energy = {20.0: 365.0, 40.0: 500.0}
    soh_by_energy = {20.0: 0.62, 40.0: 0.55}
    compliance_by_energy = {20.0: 90.0, 40.0: 95.0}

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
        min_soh=0.6,
        simulate_fn=_fake_simulation_factory(cycles_by_energy, soh_by_energy),
        summarize_fn=_fake_summary_factory(compliance_by_energy, cycles_by_energy),
    )

    cycles_over_cap = dict(zip(df["energy_mwh"], df["cycles_over_cap"]))
    soh_margin = dict(zip(df["energy_mwh"], df["soh_margin"]))

    assert cycles_over_cap[20.0] == pytest.approx(0.0)
    assert cycles_over_cap[40.0] == pytest.approx(135.0)
    assert soh_margin[20.0] == pytest.approx(0.02)
    assert soh_margin[40.0] == pytest.approx(-0.05)


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


def test_sweep_prunes_candidates_and_skips_economics_with_thresholds():
    pv_df = pd.DataFrame({"pv_mw": [0.0]})
    cycle_df = pd.DataFrame()
    base_cfg = SimConfig(years=1, initial_power_mw=10.0, initial_usable_mwh=10.0)

    compliance_by_energy = {20.0: 80.0, 40.0: 95.0, 60.0: 92.0}
    shortfall_by_energy = {20.0: 2.0, 40.0: 10.0, 60.0: 3.0}

    def _simulate(cfg: SimConfig, pv_df, cycle_df, dod_override, need_logs=False):
        energy = cfg.initial_usable_mwh
        year = YearResult(
            year_index=1,
            expected_firm_mwh=0.0,
            delivered_firm_mwh=energy,
            shortfall_mwh=shortfall_by_energy[energy],
            breach_days=0,
            charge_mwh=0.0,
            discharge_mwh=0.0,
            available_pv_mwh=0.0,
            pv_to_contract_mwh=0.0,
            bess_to_contract_mwh=energy,
            avg_rte=1.0,
            eq_cycles=100.0,
            cum_cycles=100.0,
            soh_cycle=1.0,
            soh_calendar=1.0,
            soh_total=0.9,
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
            compliance=compliance_by_energy[energy],
            bess_share_of_firm=1.0,
            charge_discharge_ratio=1.0,
            pv_capture_ratio=1.0,
            discharge_capacity_factor=1.0,
            total_project_generation_mwh=energy,
            bess_generation_mwh=energy,
            pv_generation_mwh=0.0,
            pv_excess_mwh=0.0,
            bess_losses_mwh=0.0,
            total_shortfall_mwh=shortfall_by_energy[energy],
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
        energy_mwh_values=[20.0, 40.0, 60.0],
        fixed_power_mw=10.0,
        min_soh=0.5,
        min_compliance_pct=90.0,
        max_shortfall_mwh=5.0,
        simulate_fn=_simulate,
        summarize_fn=_summarize,
        economics_inputs=econ_inputs,
    )

    status_by_energy = dict(zip(df["energy_mwh"], df["status"]))
    assert status_by_energy[20.0] == "below_min_compliance"
    assert status_by_energy[40.0] == "exceeds_shortfall"
    assert status_by_energy[60.0] == "evaluated"

    assert math.isnan(df.loc[df["energy_mwh"] == 20.0, "npv_costs_usd"].iloc[0])
    assert math.isnan(df.loc[df["energy_mwh"] == 40.0, "irr_pct"].iloc[0])
    assert math.isfinite(df.loc[df["energy_mwh"] == 60.0, "npv_costs_usd"].iloc[0])
    assert df.loc[df["is_best"], "energy_mwh"].iloc[0] == 60.0


def test_sweep_uses_price_inputs_for_cashflow_npv() -> None:
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
            bess_to_contract_mwh=100.0,
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
            augmentation_energy_added_mwh=[0.0],
            augmentation_retired_energy_mwh=[0.0],
            augmentation_events=0,
        )

    def _summarize(sim_output: SimulationOutput) -> SimulationSummary:
        return SimulationSummary(
            compliance=95.0,
            bess_share_of_firm=1.0,
            charge_discharge_ratio=1.0,
            pv_capture_ratio=1.0,
            discharge_capacity_factor=1.0,
            total_project_generation_mwh=sim_output.cfg.initial_power_mw * sim_output.cfg.initial_usable_mwh,
            bess_generation_mwh=sim_output.cfg.initial_usable_mwh,
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
    price_inputs = PriceInputs(
        contract_price_usd_per_mwh=120.0,
        pv_market_price_usd_per_mwh=0.0,
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
        price_inputs=price_inputs,
    )

    size_scale = df.loc[0, "energy_mwh"] / base_cfg.initial_usable_mwh
    expected_capex_usd = econ_inputs.capex_musd * size_scale * 1_000_000.0
    expected_npv = -expected_capex_usd + (df.loc[0, "energy_mwh"] * price_inputs.contract_price_usd_per_mwh)

    assert abs(df.loc[0, "npv_usd"] - expected_npv) < 1e-6
    assert df.loc[0, "irr_pct"] < -98.0


def test_sweep_streams_rows_via_callback() -> None:
    pv_df = pd.DataFrame({"pv_mw": [0.0]})
    cycle_df = pd.DataFrame()
    base_cfg = SimConfig(years=1, soc_floor=0.2, soc_ceiling=0.85, rte_roundtrip=0.9)
    received: list[pd.DataFrame] = []

    def _simulate(cfg: SimConfig, pv_df, cycle_df, dod_override, need_logs=False):
        year = YearResult(
            year_index=1,
            expected_firm_mwh=0.0,
            delivered_firm_mwh=cfg.initial_usable_mwh,
            shortfall_mwh=0.0,
            breach_days=0,
            charge_mwh=0.0,
            discharge_mwh=0.0,
            available_pv_mwh=0.0,
            pv_to_contract_mwh=0.0,
            bess_to_contract_mwh=cfg.initial_usable_mwh,
            avg_rte=cfg.rte_roundtrip,
            eq_cycles=10.0,
            cum_cycles=10.0,
            soh_cycle=1.0,
            soh_calendar=1.0,
            soh_total=0.9,
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
            augmentation_energy_added_mwh=[0.0],
            augmentation_retired_energy_mwh=[0.0],
            augmentation_events=0,
        )

    def _summarize(sim_output: SimulationOutput) -> SimulationSummary:
        return SimulationSummary(
            compliance=95.0,
            bess_share_of_firm=0.5,
            charge_discharge_ratio=1.0,
            pv_capture_ratio=1.0,
            discharge_capacity_factor=1.0,
            total_project_generation_mwh=sim_output.cfg.initial_power_mw * sim_output.cfg.initial_usable_mwh,
            bess_generation_mwh=sim_output.cfg.initial_usable_mwh,
            pv_generation_mwh=0.0,
            pv_excess_mwh=0.0,
            bess_losses_mwh=0.0,
            total_shortfall_mwh=0.0,
            avg_eq_cycles_per_year=10.0,
            cap_ratio_final=1.0,
        )

    def _capture_progress(df: pd.DataFrame) -> None:
        received.append(df)

    df = sweep_bess_sizes(
        base_cfg,
        pv_df,
        cycle_df,
        "Custom DoD",
        power_mw_values=[5.0, 10.0],
        duration_h_values=[2.0],
        min_soh=0.5,
        simulate_fn=_simulate,
        summarize_fn=_summarize,
        progress_callback=_capture_progress,
    )

    assert len(df) == 2
    assert len(received) == 2

    streamed = pd.concat(received, ignore_index=True)
    assert set(["soc_floor", "soc_ceiling", "rte_roundtrip", "dod_override"]).issubset(streamed.columns)
    assert streamed["dod_override"].unique().tolist() == ["Custom DoD"]
    assert streamed["soc_floor"].iloc[0] == pytest.approx(0.2)
    assert streamed["rte_roundtrip"].iloc[0] == pytest.approx(0.9)


def test_compute_candidate_economics_respects_overrides() -> None:
    base_cfg = SimConfig(years=2, initial_power_mw=5.0, initial_usable_mwh=10.0)
    results = [
        YearResult(
            year_index=1,
            expected_firm_mwh=0.0,
            delivered_firm_mwh=1.0,
            shortfall_mwh=0.0,
            breach_days=0,
            charge_mwh=0.0,
            discharge_mwh=0.0,
            available_pv_mwh=0.0,
            pv_to_contract_mwh=0.0,
            bess_to_contract_mwh=1.0,
            avg_rte=1.0,
            eq_cycles=0.0,
            cum_cycles=0.0,
            soh_cycle=1.0,
            soh_calendar=1.0,
            soh_total=1.0,
            eoy_usable_mwh=base_cfg.initial_usable_mwh,
            eoy_power_mw=base_cfg.initial_power_mw,
            pv_curtailed_mwh=0.0,
            flags={},
        ),
        YearResult(
            year_index=2,
            expected_firm_mwh=0.0,
            delivered_firm_mwh=1.0,
            shortfall_mwh=0.0,
            breach_days=0,
            charge_mwh=0.0,
            discharge_mwh=0.0,
            available_pv_mwh=0.0,
            pv_to_contract_mwh=0.0,
            bess_to_contract_mwh=1.0,
            avg_rte=1.0,
            eq_cycles=0.0,
            cum_cycles=0.0,
            soh_cycle=1.0,
            soh_calendar=1.0,
            soh_total=1.0,
            eoy_usable_mwh=base_cfg.initial_usable_mwh,
            eoy_power_mw=base_cfg.initial_power_mw,
            pv_curtailed_mwh=0.0,
            flags={},
        ),
    ]

    sim_output = SimulationOutput(
        cfg=base_cfg,
        discharge_hours_per_day=4.0,
        results=results,
        monthly_results=[],
        first_year_logs=None,
        final_year_logs=None,
        hod_count=np.zeros(24),
        hod_sum_pv=np.zeros(24),
        hod_sum_pv_resource=np.zeros(24),
        hod_sum_bess=np.zeros(24),
        hod_sum_charge=np.zeros(24),
        augmentation_energy_added_mwh=[0.0, 0.0],
        augmentation_retired_energy_mwh=[0.0, 0.0],
        augmentation_events=0,
    )

    econ_inputs = EconomicInputs(
        capex_musd=1.0,
        fixed_opex_pct_of_capex=0.0,
        fixed_opex_musd=0.0,
        inflation_rate=0.0,
        discount_rate=0.0,
    )
    price_inputs = PriceInputs(
        contract_price_usd_per_mwh=100.0,
        pv_market_price_usd_per_mwh=0.0,
    )

    lcoe, discounted_costs, irr_pct, npv_usd = _compute_candidate_economics(
        sim_output,
        econ_inputs,
        price_inputs,
        base_initial_energy_mwh=base_cfg.initial_usable_mwh,
        augmentation_energy_added_mwh=[1.0, 2.0],
        delivered_firm_mwh=[20.0, 20.0],
        bess_to_contract_mwh=[20.0, 20.0],
        pv_curtailed_mwh=[0.0, 0.0],
        shortfall_mwh=[0.0, 0.0],
    )

    assert math.isclose(discounted_costs, 1_300_000.0)
    assert math.isclose(lcoe, 32500.0)
    assert math.isclose(npv_usd, -1_296_000.0)
    assert math.isnan(irr_pct)


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


def test_sweep_warns_and_falls_back_when_ranking_missing(caplog: pytest.LogCaptureFixture):
    energy_candidates = [20.0, 40.0]
    cycles_by_energy = {20.0: 100.0, 40.0: 100.0}
    soh_by_energy = {20.0: 0.75, 40.0: 0.75}
    compliance_by_energy = {20.0: 80.0, 40.0: 90.0}

    base_cfg = SimConfig(years=1, max_cycles_per_day_cap=1.0)
    pv_df = pd.DataFrame({"pv_mw": [0.0]})
    cycle_df = pd.DataFrame()

    with caplog.at_level(logging.WARNING):
        df = sweep_bess_sizes(
            base_cfg,
            pv_df,
            cycle_df,
            "Auto (infer)",
            energy_mwh_values=energy_candidates,
            fixed_power_mw=10.0,
            use_case="reliability",
            ranking_kpi="npv_usd",
            min_soh=0.65,
            simulate_fn=_fake_simulation_factory(cycles_by_energy, soh_by_energy),
            summarize_fn=_fake_summary_factory(compliance_by_energy, cycles_by_energy),
        )

    assert any("Ranking KPI" in record.message for record in caplog.records)
    assert df.loc[df["is_best"], "energy_mwh"].iloc[0] == 40.0


def test_sweep_computes_normalized_kpis_and_ranks_new_metrics():
    pv_df = pd.DataFrame({"pv_mw": [0.0]})
    cycle_df = pd.DataFrame()
    base_cfg = SimConfig(years=1, initial_power_mw=10.0, initial_usable_mwh=50.0)

    def _simulate(cfg: SimConfig, pv_df, cycle_df, dod_override, need_logs=False):
        energy = cfg.initial_usable_mwh
        delivered = energy * (0.6 if energy <= 50.0 else 0.9)
        year = YearResult(
            year_index=1,
            expected_firm_mwh=0.0,
            delivered_firm_mwh=delivered,
            shortfall_mwh=0.0,
            breach_days=0,
            charge_mwh=0.0,
            discharge_mwh=0.0,
            available_pv_mwh=0.0,
            pv_to_contract_mwh=0.0,
            bess_to_contract_mwh=delivered,
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
        delivered = sim_output.results[0].delivered_firm_mwh
        return SimulationSummary(
            compliance=90.0,
            bess_share_of_firm=1.0,
            charge_discharge_ratio=1.0,
            pv_capture_ratio=1.0,
            discharge_capacity_factor=1.0,
            total_project_generation_mwh=delivered,
            bess_generation_mwh=delivered,
            pv_generation_mwh=0.0,
            pv_excess_mwh=0.0,
            bess_losses_mwh=0.0,
            total_shortfall_mwh=0.0,
            avg_eq_cycles_per_year=0.0,
            cap_ratio_final=1.0,
        )

    econ_inputs = EconomicInputs(
        capex_musd=0.01,
        fixed_opex_pct_of_capex=0.0,
        fixed_opex_musd=0.0,
        inflation_rate=0.0,
        discount_rate=0.0,
    )
    price_inputs = PriceInputs(
        contract_price_usd_per_mwh=200.0,
        pv_market_price_usd_per_mwh=0.0,
    )

    df = sweep_bess_sizes(
        base_cfg,
        pv_df,
        cycle_df,
        "Auto (infer)",
        energy_mwh_values=[50.0, 100.0],
        fixed_power_mw=10.0,
        use_case="reliability",
        ranking_kpi="npv_per_mwh_usd",
        min_soh=0.5,
        simulate_fn=_simulate,
        summarize_fn=_summarize,
        economics_inputs=econ_inputs,
        price_inputs=price_inputs,
    )

    capex_per_kw_small = df.loc[df["energy_mwh"] == 50.0, "capex_per_kw_usd"].iloc[0]
    assert math.isclose(capex_per_kw_small, 1.0)

    npv_per_mwh_large = df.loc[df["energy_mwh"] == 100.0, "npv_per_mwh_usd"].iloc[0]
    assert npv_per_mwh_large > df.loc[df["energy_mwh"] == 50.0, "npv_per_mwh_usd"].iloc[0]

    assert df.loc[df["is_best"], "energy_mwh"].iloc[0] == 100.0


def test_sweep_enforces_limit_and_batches() -> None:
    pv_df = pd.DataFrame({"pv_mw": [0.0]})
    cycle_df = pd.DataFrame()
    base_cfg = SimConfig(years=1)

    def _simulate(cfg: SimConfig, pv_df, cycle_df, dod_override, need_logs=False):
        year = YearResult(
            year_index=1,
            expected_firm_mwh=0.0,
            delivered_firm_mwh=cfg.initial_usable_mwh,
            shortfall_mwh=0.0,
            breach_days=0,
            charge_mwh=0.0,
            discharge_mwh=0.0,
            available_pv_mwh=0.0,
            pv_to_contract_mwh=0.0,
            bess_to_contract_mwh=cfg.initial_usable_mwh,
            avg_rte=1.0,
            eq_cycles=100.0,
            cum_cycles=100.0,
            soh_cycle=1.0,
            soh_calendar=1.0,
            soh_total=0.7,
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
            augmentation_energy_added_mwh=[0.0],
            augmentation_retired_energy_mwh=[0.0],
            augmentation_events=0,
        )

    def _summarize(sim_output: SimulationOutput) -> SimulationSummary:
        return SimulationSummary(
            compliance=90.0,
            bess_share_of_firm=1.0,
            charge_discharge_ratio=1.0,
            pv_capture_ratio=1.0,
            discharge_capacity_factor=1.0,
            total_project_generation_mwh=sim_output.cfg.initial_usable_mwh,
            bess_generation_mwh=sim_output.cfg.initial_usable_mwh,
            pv_generation_mwh=0.0,
            pv_excess_mwh=0.0,
            bess_losses_mwh=0.0,
            total_shortfall_mwh=0.0,
            avg_eq_cycles_per_year=100.0,
            cap_ratio_final=1.0,
        )

    with pytest.raises(ValueError):
        sweep_bess_sizes(
            base_cfg,
            pv_df,
            cycle_df,
            "Auto (infer)",
            power_mw_values=[5.0, 10.0],
            duration_h_values=[1.0, 2.0],
            max_candidates=2,
            simulate_fn=_simulate,
            summarize_fn=_summarize,
        )

    df = sweep_bess_sizes(
        base_cfg,
        pv_df,
        cycle_df,
        "Auto (infer)",
        power_mw_values=[5.0, 10.0],
        duration_h_values=[1.0, 2.0],
        max_candidates=2,
        on_exceed="batch",
        batch_size=2,
        simulate_fn=_simulate,
        summarize_fn=_summarize,
    )

    assert len(df) == 4
    assert df["energy_mwh"].tolist() == [5.0, 10.0, 10.0, 20.0]


def test_sweep_concurrency_preserves_order() -> None:
    pv_df = pd.DataFrame({"pv_mw": [0.0]})
    cycle_df = pd.DataFrame()
    base_cfg = SimConfig(years=1)
    energy_candidates = [30.0, 10.0, 20.0]

    def _simulate(cfg: SimConfig, pv_df, cycle_df, dod_override, need_logs=False):
        sleep_time = cfg.initial_usable_mwh / 1000.0
        time.sleep(sleep_time)
        year = YearResult(
            year_index=1,
            expected_firm_mwh=0.0,
            delivered_firm_mwh=cfg.initial_usable_mwh,
            shortfall_mwh=0.0,
            breach_days=0,
            charge_mwh=0.0,
            discharge_mwh=0.0,
            available_pv_mwh=0.0,
            pv_to_contract_mwh=0.0,
            bess_to_contract_mwh=cfg.initial_usable_mwh,
            avg_rte=1.0,
            eq_cycles=cfg.initial_usable_mwh,
            cum_cycles=cfg.initial_usable_mwh,
            soh_cycle=1.0,
            soh_calendar=1.0,
            soh_total=0.9,
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
            augmentation_energy_added_mwh=[0.0],
            augmentation_retired_energy_mwh=[0.0],
            augmentation_events=0,
        )

    def _summarize(sim_output: SimulationOutput) -> SimulationSummary:
        energy = sim_output.cfg.initial_usable_mwh
        return SimulationSummary(
            compliance=energy,
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
            avg_eq_cycles_per_year=energy,
            cap_ratio_final=1.0,
        )

    df = sweep_bess_sizes(
        base_cfg,
        pv_df,
        cycle_df,
        "Auto (infer)",
        energy_mwh_values=energy_candidates,
        fixed_power_mw=10.0,
        concurrency="thread",
        max_workers=3,
        simulate_fn=_simulate,
        summarize_fn=_summarize,
    )

    assert df["energy_mwh"].tolist() == energy_candidates
    assert df["compliance_pct"].tolist() == energy_candidates


def test_sweep_supports_multiple_price_scenarios_without_extra_sims() -> None:
    pv_df = pd.DataFrame({"pv_mw": [0.0]})
    cycle_df = pd.DataFrame()
    base_cfg = SimConfig(years=1, initial_power_mw=5.0, initial_usable_mwh=10.0)
    energy_candidates = [10.0, 20.0]
    sim_calls: list[float] = []

    def _simulate(cfg: SimConfig, pv_df, cycle_df, dod_override, need_logs=False):
        sim_calls.append(cfg.initial_usable_mwh)
        year = YearResult(
            year_index=1,
            expected_firm_mwh=0.0,
            delivered_firm_mwh=cfg.initial_usable_mwh,
            shortfall_mwh=0.0,
            breach_days=0,
            charge_mwh=0.0,
            discharge_mwh=0.0,
            available_pv_mwh=0.0,
            pv_to_contract_mwh=0.0,
            bess_to_contract_mwh=cfg.initial_usable_mwh,
            avg_rte=1.0,
            eq_cycles=cfg.initial_usable_mwh,
            cum_cycles=cfg.initial_usable_mwh,
            soh_cycle=1.0,
            soh_calendar=1.0,
            soh_total=0.9,
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
            augmentation_energy_added_mwh=[0.0],
            augmentation_retired_energy_mwh=[0.0],
            augmentation_events=0,
        )

    def _summarize(sim_output: SimulationOutput) -> SimulationSummary:
        energy = sim_output.cfg.initial_usable_mwh
        return SimulationSummary(
            compliance=80.0 if energy <= 10.0 else 95.0,
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
            avg_eq_cycles_per_year=energy,
            cap_ratio_final=1.0,
        )

    econ_inputs = EconomicInputs(
        capex_musd=0.5,
        fixed_opex_pct_of_capex=0.0,
        fixed_opex_musd=0.0,
        inflation_rate=0.0,
        discount_rate=0.0,
    )
    price_scenarios = [
        PriceInputs(contract_price_usd_per_mwh=100.0, pv_market_price_usd_per_mwh=0.0),
        PriceInputs(contract_price_usd_per_mwh=150.0, pv_market_price_usd_per_mwh=0.0),
    ]
    scenario_names = ["base", "upside"]

    df = sweep_bess_sizes(
        base_cfg,
        pv_df,
        cycle_df,
        "Auto (infer)",
        energy_mwh_values=energy_candidates,
        fixed_power_mw=5.0,
        simulate_fn=_simulate,
        summarize_fn=_summarize,
        economics_inputs=econ_inputs,
        price_inputs=price_scenarios,
        price_scenario_names=scenario_names,
    )

    assert len(sim_calls) == len(energy_candidates)
    assert len(df) == len(energy_candidates) * len(price_scenarios)
    assert set(df["price_scenario"]) == set(scenario_names)
    for _, group in df.groupby("energy_mwh"):
        assert group["compliance_pct"].nunique() == 1

    base_npv = df.loc[(df["price_scenario"] == "base") & (df["energy_mwh"] == 20.0), "npv_usd"].iloc[0]
    upside_npv = df.loc[(df["price_scenario"] == "upside") & (df["energy_mwh"] == 20.0), "npv_usd"].iloc[0]
    assert upside_npv > base_npv
    for name in scenario_names:
        assert df.loc[df["price_scenario"] == name, "is_best"].sum() == 1


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
        apply_wesm_to_shortfall=True,
        wesm_deficit_price_usd_per_mwh=90.0,
        sell_to_wesm=True,
    )

    df = compute_static_bess_sweep_economics(
        candidates,
        economics_template,
        price_inputs,
        wesm_deficit_price_usd_per_mwh=90.0,
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
        wesm_deficit_price_usd_per_mwh=90.0,
        years=1,
    )

    expected_revenue = (100.0 + 50.0) * 50.0
    assert math.isclose(df.loc[0, "npv_usd"], expected_revenue)


def test_static_economic_sweep_applies_wesm_penalty_when_enabled() -> None:
    candidates = [
        BessEconomicCandidate(
            energy_mwh=50.0,
            capex_musd=0.0,
            fixed_opex_musd=0.0,
            compliance_mwh=1000.0,
            deficit_mwh=-200.0,
            surplus_mwh=0.0,
        )
    ]

    economics_template = EconomicInputs(
        capex_musd=0.0,
        fixed_opex_pct_of_capex=0.0,
        fixed_opex_musd=0.0,
        inflation_rate=0.0,
        discount_rate=0.0,
    )

    price_inputs_disabled = PriceInputs(
        contract_price_usd_per_mwh=120.0,
        pv_market_price_usd_per_mwh=50.0,
        apply_wesm_to_shortfall=False,
        wesm_deficit_price_usd_per_mwh=90.0,
    )
    price_inputs_enabled = PriceInputs(
        contract_price_usd_per_mwh=120.0,
        pv_market_price_usd_per_mwh=50.0,
        apply_wesm_to_shortfall=True,
        wesm_deficit_price_usd_per_mwh=90.0,
    )

    df_disabled = compute_static_bess_sweep_economics(
        candidates,
        economics_template,
        price_inputs_disabled,
        wesm_deficit_price_usd_per_mwh=90.0,
        years=1,
    )
    df_enabled = compute_static_bess_sweep_economics(
        candidates,
        economics_template,
        price_inputs_enabled,
        wesm_deficit_price_usd_per_mwh=90.0,
        years=1,
    )

    assert df_enabled.loc[0, "npv_usd"] < df_disabled.loc[0, "npv_usd"]
    disabled_irr = df_disabled.loc[0, "irr_pct"]
    enabled_irr = df_enabled.loc[0, "irr_pct"]
    if math.isnan(disabled_irr):
        assert math.isnan(enabled_irr) or enabled_irr <= disabled_irr
    else:
        assert enabled_irr < disabled_irr


def test_candidate_economics_honors_blended_price_and_wesm_shortfalls() -> None:
    sim_output = _StubSimOutput(
        cfg=_StubCfg(initial_usable_mwh=50.0, initial_power_mw=10.0),
        results=[
            _StubResult(
                delivered_firm_mwh=100.0,
                bess_to_contract_mwh=80.0,
                pv_curtailed_mwh=20.0,
                shortfall_mwh=10.0,
            )
        ],
    )
    economics_inputs = EconomicInputs(
        capex_musd=0.0,
        fixed_opex_pct_of_capex=0.0,
        fixed_opex_musd=0.0,
        inflation_rate=0.0,
        discount_rate=0.0,
    )
    price_inputs_blended = PriceInputs(
        contract_price_usd_per_mwh=120.0,
        pv_market_price_usd_per_mwh=60.0,
        blended_price_usd_per_mwh=50.0,
        apply_wesm_to_shortfall=True,
        wesm_deficit_price_usd_per_mwh=90.0,
        sell_to_wesm=True,
    )

    revenue_expected = (100.0 * 50.0) + (20.0 * 90.0) - (10.0 * 90.0)
    _, _, _, npv_usd = _compute_candidate_economics(
        sim_output,
        economics_inputs,
        price_inputs_blended,
        base_initial_energy_mwh=50.0,
    )

    assert math.isclose(npv_usd, revenue_expected)


def test_main_example_outputs_sections(capsys: pytest.CaptureFixture[str]) -> None:
    _main_example()

    output = capsys.readouterr().out
    assert "Feasibility snapshot" in output
    assert "cycles_over_cap" in output
    assert "soh_margin" in output
    assert "Ranking KPI (npv_per_mwh_usd)" in output
    assert "is_best" in output
    assert "npv_per_mwh_usd" in output
