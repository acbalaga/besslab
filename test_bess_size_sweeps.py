import numpy as np
import pandas as pd

from app import SimConfig, SimulationOutput, SimulationSummary, YearResult
from bess_size_sweeps import sweep_bess_sizes


def _fake_simulation_factory(cycles_by_power: dict[float, float], soh_by_power: dict[float, float]):
    def _simulate(cfg: SimConfig, pv_df, cycle_df, dod_override, need_logs=False):
        eq_cycles = cycles_by_power[cfg.initial_power_mw]
        soh_total = soh_by_power[cfg.initial_power_mw]
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


def _fake_summary_factory(compliance_by_power: dict[float, float], cycles_by_power: dict[float, float]):
    def _summarize(sim_output: SimulationOutput) -> SimulationSummary:
        power = sim_output.cfg.initial_power_mw
        return SimulationSummary(
            compliance=compliance_by_power[power],
            bess_share_of_firm=0.0,
            charge_discharge_ratio=1.0,
            pv_capture_ratio=1.0,
            discharge_capacity_factor=1.0,
            total_project_generation_mwh=power * sim_output.cfg.initial_usable_mwh,
            bess_generation_mwh=0.0,
            pv_generation_mwh=0.0,
            pv_excess_mwh=0.0,
            bess_losses_mwh=0.0,
            total_shortfall_mwh=max(0.0, 100.0 - power),
            avg_eq_cycles_per_year=cycles_by_power[power],
            cap_ratio_final=1.0,
        )

    return _summarize


def test_sweep_flags_feasible_best_candidate():
    power_candidates = [5.0, 8.0, 12.0]
    duration_candidates = [2.0]
    cycles_by_power = {5.0: 100.0, 8.0: 300.0, 12.0: 500.0}
    soh_by_power = {5.0: 0.6, 8.0: 0.75, 12.0: 0.8}
    compliance_by_power = {5.0: 70.0, 8.0: 90.0, 12.0: 95.0}

    base_cfg = SimConfig(years=1, max_cycles_per_day_cap=1.0)
    pv_df = pd.DataFrame({"pv_mw": [0.0]})
    cycle_df = pd.DataFrame()

    df = sweep_bess_sizes(
        base_cfg,
        pv_df,
        cycle_df,
        "Auto (infer)",
        power_candidates,
        duration_candidates,
        use_case="reliability",
        ranking_kpi=None,
        min_soh=0.65,
        simulate_fn=_fake_simulation_factory(cycles_by_power, soh_by_power),
        summarize_fn=_fake_summary_factory(compliance_by_power, cycles_by_power),
    )

    assert len(df) == len(power_candidates)
    assert df.loc[df["is_best"], "power_mw"].iloc[0] == 8.0
    assert df.loc[df["power_mw"] == 12.0, "cycle_limit_hit"].iloc[0]
    assert df.loc[df["power_mw"] == 5.0, "soh_below_min"].iloc[0]
    assert not df.loc[df["power_mw"] == 12.0, "is_best"].iloc[0]
