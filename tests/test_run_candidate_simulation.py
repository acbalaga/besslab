import numpy as np
import pandas as pd
import pytest

from app import SimConfig, SimulationOutput, SimulationSummary, YearResult
from utils.sweeps import run_candidate_simulation


def _seeded_simulator(
    cfg: SimConfig,
    pv_df: pd.DataFrame,
    cycle_df: pd.DataFrame,
    dod_override: str,
    need_logs: bool = False,
    seed: int | None = None,
    deterministic: bool | None = None,
) -> SimulationOutput:
    rng_seed = seed
    if deterministic and rng_seed is None:
        rng_seed = 0

    rng = np.random.default_rng(rng_seed)
    draw = float(rng.random())
    if not deterministic:
        draw += 1.0

    year = YearResult(
        year_index=1,
        expected_firm_mwh=0.0,
        delivered_firm_mwh=0.0,
        shortfall_mwh=0.0,
        breach_days=0,
        charge_mwh=0.0,
        discharge_mwh=0.0,
        available_pv_mwh=float(pv_df["pv_mw"].sum()),
        pv_to_contract_mwh=0.0,
        bess_to_contract_mwh=0.0,
        avg_rte=1.0,
        eq_cycles=draw,
        cum_cycles=draw,
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
        augmentation_energy_added_mwh=[0.0 for _ in range(cfg.years)],
        augmentation_retired_energy_mwh=[0.0 for _ in range(cfg.years)],
        augmentation_events=0,
    )


def _seeded_summary(sim_output: SimulationOutput) -> SimulationSummary:
    draw = sim_output.results[0].eq_cycles
    return SimulationSummary(
        compliance=draw,
        bess_share_of_firm=0.0,
        charge_discharge_ratio=1.0,
        pv_capture_ratio=1.0,
        discharge_capacity_factor=1.0,
        total_project_generation_mwh=sim_output.cfg.initial_power_mw * sim_output.cfg.initial_usable_mwh,
        bess_generation_mwh=0.0,
        pv_generation_mwh=0.0,
        pv_excess_mwh=0.0,
        bess_losses_mwh=0.0,
        total_shortfall_mwh=0.0,
        avg_eq_cycles_per_year=draw,
        cap_ratio_final=1.0,
    )


def test_run_candidate_simulation_forwards_seed() -> None:
    base_cfg = SimConfig(years=1, initial_power_mw=2.0, initial_usable_mwh=4.0)
    pv_df = pd.DataFrame({"pv_mw": [1.0, 1.5]})
    cycle_df = pd.DataFrame()

    _, summary_one = run_candidate_simulation(
        base_cfg,
        pv_df,
        cycle_df,
        "Auto (infer)",
        power_mw=3.0,
        duration_h=2.0,
        simulate_fn=_seeded_simulator,
        summarize_fn=_seeded_summary,
        seed=123,
    )
    _, summary_two = run_candidate_simulation(
        base_cfg,
        pv_df,
        cycle_df,
        "Auto (infer)",
        power_mw=3.0,
        duration_h=2.0,
        simulate_fn=_seeded_simulator,
        summarize_fn=_seeded_summary,
        seed=123,
    )
    _, summary_three = run_candidate_simulation(
        base_cfg,
        pv_df,
        cycle_df,
        "Auto (infer)",
        power_mw=3.0,
        duration_h=2.0,
        simulate_fn=_seeded_simulator,
        summarize_fn=_seeded_summary,
        seed=456,
    )

    assert summary_one.compliance == pytest.approx(summary_two.compliance)
    assert summary_one.compliance != pytest.approx(summary_three.compliance)


def test_run_candidate_simulation_accepts_deterministic_flag() -> None:
    base_cfg = SimConfig(years=1, initial_power_mw=2.0, initial_usable_mwh=4.0)
    pv_df = pd.DataFrame({"pv_mw": [1.0]})
    cycle_df = pd.DataFrame()

    _, deterministic_one = run_candidate_simulation(
        base_cfg,
        pv_df,
        cycle_df,
        "Auto (infer)",
        power_mw=3.0,
        duration_h=2.0,
        simulate_fn=_seeded_simulator,
        summarize_fn=_seeded_summary,
        deterministic=True,
    )
    _, deterministic_two = run_candidate_simulation(
        base_cfg,
        pv_df,
        cycle_df,
        "Auto (infer)",
        power_mw=3.0,
        duration_h=2.0,
        simulate_fn=_seeded_simulator,
        summarize_fn=_seeded_summary,
        deterministic=True,
    )
    _, random_summary = run_candidate_simulation(
        base_cfg,
        pv_df,
        cycle_df,
        "Auto (infer)",
        power_mw=3.0,
        duration_h=2.0,
        simulate_fn=_seeded_simulator,
        summarize_fn=_seeded_summary,
    )

    assert deterministic_one.compliance == pytest.approx(deterministic_two.compliance)
    assert deterministic_one.compliance != pytest.approx(random_summary.compliance)
