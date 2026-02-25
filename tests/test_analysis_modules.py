from __future__ import annotations

import numpy as np
import pandas as pd

from services.analysis_bruteforce import BruteForceAnalysisRequest, run_bruteforce_analysis
from services.analysis_common import BusinessRuleAssumptions, SimulationExecutionContext
from services.analysis_de import DifferentialEvolutionRequest, run_differential_evolution_analysis
from services.analysis_sensitivity import SensitivityAnalysisRequest, run_sensitivity_analysis
from services.simulation_core import SimConfig, SimulationOutput, SimulationSummary, YearResult


EXPECTED_COLUMNS = [
    "analysis_mode",
    "scenario_id",
    "candidate_id",
    "candidate_rank",
    "power_mw",
    "duration_h",
    "energy_mwh",
    "feasible",
    "objective_metric",
    "objective_direction",
    "objective_value",
    "compliance_pct",
    "total_shortfall_mwh",
    "bess_share_of_firm_pct",
    "avg_eq_cycles_per_year",
    "cap_ratio_final",
    "tariff_escalation_rate_pct",
    "shortfall_penalty_usd_per_mwh",
    "seed",
    "deterministic",
]


def _deterministic_simulator(
    cfg: SimConfig,
    pv_df: pd.DataFrame,
    cycle_df: pd.DataFrame,
    dod_override: str,
    need_logs: bool = False,
    seed: int | None = None,
    deterministic: bool | None = None,
) -> SimulationOutput:
    rng_seed = seed if deterministic else None
    rng = np.random.default_rng(rng_seed)
    noise = float(rng.random()) * 1e-6
    compliance = max(0.0, 1.0 - 0.02 * cfg.initial_power_mw - 0.01 * cfg.initial_usable_mwh + noise)
    shortfall = max(0.0, 100.0 * (1.0 - compliance))

    year = YearResult(
        year_index=1,
        expected_firm_mwh=1000.0,
        delivered_firm_mwh=1000.0 * compliance,
        shortfall_mwh=shortfall,
        breach_days=0,
        charge_mwh=0.0,
        discharge_mwh=0.0,
        available_pv_mwh=float(pv_df["pv_mw"].sum()),
        pv_to_contract_mwh=0.0,
        bess_to_contract_mwh=0.0,
        avg_rte=1.0,
        eq_cycles=1.0,
        cum_cycles=1.0,
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


def _summary(sim_output: SimulationOutput) -> SimulationSummary:
    result = sim_output.results[0]
    compliance = result.delivered_firm_mwh / result.expected_firm_mwh
    return SimulationSummary(
        compliance=compliance,
        bess_share_of_firm=0.4,
        charge_discharge_ratio=1.0,
        pv_capture_ratio=0.9,
        discharge_capacity_factor=0.6,
        total_project_generation_mwh=1000.0,
        bess_generation_mwh=400.0,
        pv_generation_mwh=600.0,
        pv_excess_mwh=0.0,
        bess_losses_mwh=0.0,
        total_shortfall_mwh=result.shortfall_mwh,
        avg_eq_cycles_per_year=1.0,
        cap_ratio_final=0.95,
    )


def _context() -> SimulationExecutionContext:
    return SimulationExecutionContext(
        base_cfg=SimConfig(years=1, initial_power_mw=5.0, initial_usable_mwh=10.0),
        pv_df=pd.DataFrame({"pv_mw": [1.0, 2.0, 3.0]}),
        cycle_df=pd.DataFrame(),
        dod_override="Auto (infer)",
        simulate_fn=_deterministic_simulator,
        summarize_fn=_summary,
    )


def _assumptions() -> BusinessRuleAssumptions:
    return BusinessRuleAssumptions(
        tariff_escalation_rate_pct=2.0,
        shortfall_penalty_usd_per_mwh=150.0,
        min_compliance_pct=0.6,
        objective_metric="shortfall_mwh",
        objective_direction="min",
    )


def test_bruteforce_request_parsing_and_columns() -> None:
    request = BruteForceAnalysisRequest.from_dict(
        {
            "scenario_id": "base",
            "power_mw_values": [2, 4],
            "duration_h_values": [1, 2],
            "assumptions": _assumptions().__dict__,
            "deterministic": True,
            "seed": 7,
        }
    )

    response = run_bruteforce_analysis(request=request, context=_context())

    assert list(response.results_df.columns) == EXPECTED_COLUMNS
    assert len(response.records) == 4
    assert response.results_df["candidate_rank"].tolist() == [1, 2, 3, 4]


def test_sensitivity_schema_and_output_stability() -> None:
    request = SensitivityAnalysisRequest.from_dict(
        {
            "scenario_id": "sens",
            "base_power_mw": 3,
            "base_duration_h": 2,
            "axes": [
                {"parameter": "power_mw_multiplier", "multipliers": [0.8, 1.0, 1.2]},
                {"parameter": "duration_h_multiplier", "multipliers": [0.5, 1.0]},
            ],
            "assumptions": _assumptions().__dict__,
            "deterministic": True,
            "seed": 11,
        }
    )

    response = run_sensitivity_analysis(request=request, context=_context())

    assert list(response.results_df.columns) == EXPECTED_COLUMNS
    assert set(response.results_df["scenario_id"]) == {"sens"}
    assert response.results_df["deterministic"].all()


def test_de_fixed_seed_is_deterministic() -> None:
    request = DifferentialEvolutionRequest.from_dict(
        {
            "scenario_id": "de",
            "power_mw_bounds": [1.0, 4.0],
            "duration_h_bounds": [1.0, 3.0],
            "population_size": 5,
            "generations": 2,
            "differential_weight": 0.6,
            "crossover_rate": 0.8,
            "assumptions": _assumptions().__dict__,
            "deterministic": True,
            "seed": 99,
        }
    )

    first = run_differential_evolution_analysis(request=request, context=_context())
    second = run_differential_evolution_analysis(request=request, context=_context())

    assert first.records == second.records
    assert list(first.results_df.columns) == EXPECTED_COLUMNS
    assert first.best_record is not None
