from __future__ import annotations

import numpy as np
import pandas as pd

from services.analysis_bruteforce import BruteForceAnalysisRequest, run_bruteforce_analysis
from services.analysis_common import BusinessRuleAssumptions, SimulationExecutionContext
from services.analysis_de import DERequest, run_differential_evolution_analysis
from services.simulation_core import SimConfig, SimulationOutput, SimulationSummary, YearResult


def _benchmark_simulator(
    cfg: SimConfig,
    pv_df: pd.DataFrame,
    cycle_df: pd.DataFrame,
    dod_override: str,
    need_logs: bool = False,
    seed: int | None = None,
    deterministic: bool | None = None,
) -> SimulationOutput:
    del pv_df, cycle_df, dod_override, need_logs, seed, deterministic
    power = float(cfg.initial_power_mw)
    duration = float(cfg.initial_usable_mwh / max(power, 1e-9))

    shortfall = (power - 2.0) ** 2 + (duration - 3.0) ** 2
    compliance = max(0.0, 1.0 - 0.1 * shortfall)

    year = YearResult(
        year_index=1,
        expected_firm_mwh=1000.0,
        delivered_firm_mwh=1000.0 * compliance,
        shortfall_mwh=shortfall,
        breach_days=0,
        charge_mwh=0.0,
        discharge_mwh=0.0,
        available_pv_mwh=0.0,
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
    yr = sim_output.results[0]
    compliance = yr.delivered_firm_mwh / yr.expected_firm_mwh
    return SimulationSummary(
        compliance=compliance,
        bess_share_of_firm=0.5,
        charge_discharge_ratio=1.0,
        pv_capture_ratio=1.0,
        discharge_capacity_factor=0.5,
        total_project_generation_mwh=1000.0,
        bess_generation_mwh=500.0,
        pv_generation_mwh=500.0,
        pv_excess_mwh=0.0,
        bess_losses_mwh=0.0,
        total_shortfall_mwh=yr.shortfall_mwh,
        avg_eq_cycles_per_year=1.0,
        cap_ratio_final=1.0,
    )


def _context() -> SimulationExecutionContext:
    return SimulationExecutionContext(
        base_cfg=SimConfig(years=1, initial_power_mw=1.0, initial_usable_mwh=1.0),
        pv_df=pd.DataFrame({"pv_mw": [0.0, 0.0, 0.0]}),
        cycle_df=pd.DataFrame(),
        dod_override="Auto (infer)",
        simulate_fn=_benchmark_simulator,
        summarize_fn=_summary,
        dispatch_optimizer_fn=None,
    )


def _assumptions() -> BusinessRuleAssumptions:
    return BusinessRuleAssumptions(
        tariff_escalation_rate_pct=0.0,
        shortfall_penalty_usd_per_mwh=0.0,
        min_compliance_pct=0.0,
        objective_metric="shortfall_mwh",
        objective_direction="min",
    )


def _de_request(seed: int = 123) -> DERequest:
    return DERequest.from_dict(
        {
            "scenario_id": "de-integration",
            "variables": [
                {"name": "power_mw", "lower_bound": 1, "upper_bound": 4, "is_integer": True, "unit": "MW"},
                {"name": "duration_h", "lower_bound": 1, "upper_bound": 5, "is_integer": True, "unit": "h"},
            ],
            "objective": {"metric": "shortfall_mwh", "direction": "min"},
            "constraints": [],
            "assumptions": _assumptions().__dict__,
            "deterministic": True,
            "seed": seed,
            "hyperparameters": {
                "population_size": 12,
                "generations": 10,
                "differential_weight": 0.8,
                "crossover_rate": 0.9,
                "top_n_candidates": 5,
                "use_scipy_if_available": False,
            },
        }
    )


def test_de_deterministic_replay_with_same_seed() -> None:
    req = _de_request(seed=77)
    first = run_differential_evolution_analysis(request=req, context=_context())
    second = run_differential_evolution_analysis(request=req, context=_context())

    assert first.records == second.records
    assert first.convergence_history == second.convergence_history
    assert first.random_seed == 77


def test_de_bounded_domain_and_integer_enforcement() -> None:
    req = _de_request(seed=88)
    response = run_differential_evolution_analysis(request=req, context=_context())

    assert not response.results_df.empty
    assert response.results_df["input__power_mw"].between(1, 4).all()
    assert response.results_df["input__duration_h"].between(1, 5).all()
    assert (response.results_df["input__power_mw"] % 1 == 0).all()
    assert (response.results_df["input__duration_h"] % 1 == 0).all()


def test_de_fallback_convergence_is_non_increasing_for_min_objective() -> None:
    req = _de_request(seed=303)
    response = run_differential_evolution_analysis(request=req, context=_context())

    best_scores = [float(row["best_objective_score"]) for row in response.convergence_history]
    assert best_scores
    assert all(curr <= prev for prev, curr in zip(best_scores, best_scores[1:]))

def test_de_penalty_behavior_for_infeasible_candidates() -> None:
    req = DERequest.from_dict(
        {
            "scenario_id": "de-integration",
            "variables": [
                {"name": "power_mw", "lower_bound": 1, "upper_bound": 4, "is_integer": True, "unit": "MW"},
                {"name": "duration_h", "lower_bound": 1, "upper_bound": 5, "is_integer": True, "unit": "h"},
            ],
            "objective": {"metric": "shortfall_mwh", "direction": "min"},
            "constraints": [{"metric": "compliance_pct", "operator": ">=", "value": 0.95, "penalty": 25.0}],
            "assumptions": _assumptions().__dict__,
            "deterministic": True,
            "seed": 91,
            "hyperparameters": {
                "population_size": 12,
                "generations": 10,
                "differential_weight": 0.8,
                "crossover_rate": 0.9,
                "top_n_candidates": 5,
                "use_scipy_if_available": False,
            },
        }
    )
    response = run_differential_evolution_analysis(request=req, context=_context())

    infeasible = response.results_df.loc[~response.results_df["feasible"]]
    assert not infeasible.empty
    assert (infeasible["constraint_penalty"] >= 25.0).all()
    assert (infeasible["objective_score"] > infeasible["objective_value"]).all()


def test_de_vs_bruteforce_sanity_on_small_grid() -> None:
    req = _de_request(seed=101)
    de_response = run_differential_evolution_analysis(request=req, context=_context())
    de_best = de_response.best_record
    assert de_best is not None

    brute_request = BruteForceAnalysisRequest.from_dict(
        {
            "scenario_id": "grid-check",
            "variables": [
                {"name": "power_mw", "values": [1, 2, 3, 4], "unit": "MW"},
                {"name": "duration_h", "values": [1, 2, 3, 4, 5], "unit": "h"},
            ],
            "objective": {"metric": "shortfall_mwh", "direction": "min"},
            "constraints": [],
            "assumptions": _assumptions().__dict__,
            "deterministic": True,
            "seed": 101,
        }
    )
    brute_response = run_bruteforce_analysis(request=brute_request, context=_context())
    brute_best = brute_response.records[0]

    assert float(de_best["power_mw"]) == float(brute_best["power_mw"])
    assert float(de_best["duration_h"]) == float(brute_best["duration_h"])
    assert float(de_best["objective_score"]) == float(brute_best["objective_score"])
