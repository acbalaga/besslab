"""Differential Evolution (DE) integration using shared simulation evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from services.analysis_common import BusinessRuleAssumptions, SimulationExecutionContext
from services.analysis_bruteforce import ConstraintDefinition, ObjectiveDefinition
from utils.sweeps import run_candidate_simulation

try:  # SciPy is optional in this repository.
    from scipy.optimize import differential_evolution as scipy_differential_evolution
except Exception:  # pragma: no cover - optional dependency path
    scipy_differential_evolution = None


@dataclass(frozen=True)
class DEVariable:
    """Decision variable schema for DE.

    Units are domain-specific and tracked by ``unit`` for replay metadata.
    """

    name: str
    lower_bound: float
    upper_bound: float
    is_integer: bool = False
    unit: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DEVariable":
        return cls(
            name=str(payload["name"]),
            lower_bound=float(payload["lower_bound"]),
            upper_bound=float(payload["upper_bound"]),
            is_integer=bool(payload.get("is_integer", False)),
            unit=str(payload.get("unit", "")),
        )


@dataclass(frozen=True)
class DEHyperparameters:
    """Tunable DE configuration values."""

    population_size: int = 20
    generations: int = 20
    differential_weight: float = 0.8
    crossover_rate: float = 0.7
    top_n_candidates: int = 5
    use_scipy_if_available: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DEHyperparameters":
        return cls(
            population_size=int(payload.get("population_size", 20)),
            generations=int(payload.get("generations", 20)),
            differential_weight=float(payload.get("differential_weight", 0.8)),
            crossover_rate=float(payload.get("crossover_rate", 0.7)),
            top_n_candidates=int(payload.get("top_n_candidates", 5)),
            use_scipy_if_available=bool(payload.get("use_scipy_if_available", True)),
        )


@dataclass(frozen=True)
class DERequest:
    """Request payload for generalized DE search."""

    scenario_id: str
    variables: tuple[DEVariable, ...]
    objective: ObjectiveDefinition
    constraints: tuple[ConstraintDefinition, ...]
    assumptions: BusinessRuleAssumptions
    deterministic: bool
    seed: int | None
    hyperparameters: DEHyperparameters

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DERequest":
        assumptions = BusinessRuleAssumptions(**payload["assumptions"])

        if "variables" in payload:
            variables = tuple(DEVariable.from_dict(item) for item in payload["variables"])
        else:
            # Backward-compatible 2D schema.
            variables = (
                DEVariable(
                    name="power_mw",
                    lower_bound=float(payload["power_mw_bounds"][0]),
                    upper_bound=float(payload["power_mw_bounds"][1]),
                    is_integer=False,
                    unit="MW",
                ),
                DEVariable(
                    name="duration_h",
                    lower_bound=float(payload["duration_h_bounds"][0]),
                    upper_bound=float(payload["duration_h_bounds"][1]),
                    is_integer=False,
                    unit="h",
                ),
            )

        objective_payload = payload.get(
            "objective",
            {"metric": assumptions.objective_metric, "direction": assumptions.objective_direction},
        )
        constraints_payload = payload.get(
            "constraints",
            [{"metric": "compliance_pct", "operator": ">=", "value": assumptions.min_compliance_pct, "penalty": 0.0}],
        )

        return cls(
            scenario_id=str(payload["scenario_id"]),
            variables=variables,
            objective=ObjectiveDefinition.from_dict(objective_payload),
            constraints=tuple(ConstraintDefinition.from_dict(item) for item in constraints_payload),
            assumptions=assumptions,
            deterministic=bool(payload["deterministic"]),
            seed=(None if payload.get("seed") is None else int(payload["seed"])),
            hyperparameters=DEHyperparameters.from_dict(
                {
                    "population_size": payload.get("population_size", 20),
                    "generations": payload.get("generations", 20),
                    "differential_weight": payload.get("differential_weight", 0.8),
                    "crossover_rate": payload.get("crossover_rate", 0.7),
                    "top_n_candidates": payload.get("top_n_candidates", 5),
                    "use_scipy_if_available": payload.get("use_scipy_if_available", True),
                    **payload.get("hyperparameters", {}),
                }
            ),
        )


# Backward compatibility name.
DifferentialEvolutionRequest = DERequest


@dataclass(frozen=True)
class DifferentialEvolutionResponse:
    """DE response payload with ranking and replay metadata."""

    results_df: pd.DataFrame
    records: list[dict[str, Any]]
    best_record: dict[str, Any] | None
    convergence_history: list[dict[str, Any]]
    random_seed: int | None
    hyperparameters: dict[str, Any]
    replay_candidates: list[dict[str, Any]]


def _clip_and_cast_vector(vector: np.ndarray, variables: tuple[DEVariable, ...]) -> np.ndarray:
    candidate = np.asarray(vector, dtype=float).copy()
    for idx, variable in enumerate(variables):
        clipped = float(np.clip(candidate[idx], variable.lower_bound, variable.upper_bound))
        if variable.is_integer:
            clipped = float(int(round(clipped)))
            clipped = float(np.clip(clipped, variable.lower_bound, variable.upper_bound))
        candidate[idx] = clipped
    return candidate


def _constraint_satisfied(value: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return value >= threshold
    if operator == "<=":
        return value <= threshold
    if operator == ">":
        return value > threshold
    if operator == "<":
        return value < threshold
    if operator == "==":
        return value == threshold
    raise ValueError(f"Unsupported constraint operator: {operator}")


def _extract_metric(metric: str, row: dict[str, Any]) -> float:
    if metric == "compliance_pct":
        return float(row["compliance_pct"])
    if metric == "shortfall_mwh":
        return float(row["total_shortfall_mwh"])
    if metric == "penalized_shortfall_cost_usd":
        return float(row["total_shortfall_mwh"]) * float(row["shortfall_penalty_usd_per_mwh"])
    if metric == "bess_share_of_firm_pct":
        return float(row["bess_share_of_firm_pct"])
    raise ValueError(f"Unsupported metric: {metric}")


def _evaluate_candidate(
    *,
    request: DERequest,
    context: SimulationExecutionContext,
    candidate_id: str,
    vector: np.ndarray,
) -> dict[str, Any]:
    variables = request.variables
    bounded = _clip_and_cast_vector(vector, variables)
    values_by_name = {variables[idx].name: float(bounded[idx]) for idx in range(len(variables))}

    power_mw = float(values_by_name.get("power_mw", context.base_cfg.initial_power_mw))
    duration_h = float(
        values_by_name.get("duration_h", context.base_cfg.initial_usable_mwh / max(context.base_cfg.initial_power_mw, 1e-9))
    )

    _, summary = run_candidate_simulation(
        base_cfg=context.base_cfg,
        pv_df=context.pv_df,
        cycle_df=context.cycle_df,
        dod_override=context.dod_override,
        power_mw=power_mw,
        duration_h=duration_h,
        simulate_fn=context.simulate_fn,
        summarize_fn=context.summarize_fn,
        need_logs=context.need_logs,
        seed=request.seed,
        deterministic=request.deterministic,
        dispatch_optimizer_fn=context.dispatch_optimizer_fn,
    )

    row: dict[str, Any] = {
        "analysis_mode": "differential_evolution",
        "scenario_id": request.scenario_id,
        "candidate_id": candidate_id,
        "candidate_rank": -1,
        "power_mw": power_mw,
        "duration_h": duration_h,
        "energy_mwh": power_mw * duration_h,
        "objective_metric": request.objective.metric,
        "objective_direction": request.objective.direction,
        "compliance_pct": float(summary.compliance),
        "total_shortfall_mwh": float(summary.total_shortfall_mwh),
        "bess_share_of_firm_pct": float(summary.bess_share_of_firm),
        "avg_eq_cycles_per_year": float(summary.avg_eq_cycles_per_year),
        "cap_ratio_final": float(summary.cap_ratio_final),
        "tariff_escalation_rate_pct": float(request.assumptions.tariff_escalation_rate_pct),
        "shortfall_penalty_usd_per_mwh": float(request.assumptions.shortfall_penalty_usd_per_mwh),
        "seed": request.seed,
        "deterministic": bool(request.deterministic),
    }

    objective_value = _extract_metric(request.objective.metric, row)
    penalty_value = 0.0
    violation_count = 0
    for constraint in request.constraints:
        metric_value = _extract_metric(constraint.metric, row)
        if not _constraint_satisfied(metric_value, constraint.operator, constraint.value):
            violation_count += 1
            penalty_value += float(constraint.penalty)

    direction = request.objective.direction
    if direction not in {"min", "max"}:
        raise ValueError("objective.direction must be 'min' or 'max'.")

    row["feasible"] = violation_count == 0
    row["objective_value"] = float(objective_value)
    row["constraint_penalty"] = float(penalty_value)
    row["constraint_violations"] = int(violation_count)
    row["objective_score"] = (
        float(objective_value + penalty_value) if direction == "min" else float(objective_value - penalty_value)
    )
    for idx, variable in enumerate(variables):
        row[f"input__{variable.name}"] = float(bounded[idx])
        row[f"input_unit__{variable.name}"] = variable.unit
    return row


def _run_fallback_de(request: DERequest, context: SimulationExecutionContext) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(request.seed)
    variable_count = len(request.variables)
    hp = request.hyperparameters
    direction = request.objective.direction

    population = np.zeros((hp.population_size, variable_count), dtype=float)
    for idx, variable in enumerate(request.variables):
        population[:, idx] = rng.uniform(variable.lower_bound, variable.upper_bound, size=hp.population_size)

    evaluated_rows: list[dict[str, Any]] = []
    convergence_history: list[dict[str, Any]] = []

    def _is_better_score(candidate_score: float, incumbent_score: float) -> bool:
        """Return ``True`` when ``candidate_score`` improves the incumbent."""

        if direction == "min":
            return candidate_score <= incumbent_score
        return candidate_score >= incumbent_score

    current_rows: list[dict[str, Any]] = []
    for target_idx in range(hp.population_size):
        row = _evaluate_candidate(
            request=request,
            context=context,
            candidate_id=f"de_g0_i{target_idx}",
            vector=population[target_idx],
        )
        current_rows.append(row)

    evaluated_rows.extend(current_rows)
    initial_scores = np.array([row["objective_score"] for row in current_rows], dtype=float)
    initial_best_score = float(np.min(initial_scores)) if direction == "min" else float(np.max(initial_scores))
    convergence_history.append({"generation": 0, "best_objective_score": initial_best_score})

    for generation in range(1, hp.generations + 1):
        generation_rows: list[dict[str, Any]] = []
        for target_idx in range(hp.population_size):
            available = [i for i in range(hp.population_size) if i != target_idx]
            a_idx, b_idx, c_idx = rng.choice(available, size=3, replace=False)
            mutant = population[a_idx] + hp.differential_weight * (population[b_idx] - population[c_idx])
            mutant = _clip_and_cast_vector(mutant, request.variables)

            crossover_mask = rng.random(variable_count) < hp.crossover_rate
            if not crossover_mask.any():
                crossover_mask[rng.integers(0, variable_count)] = True
            trial = np.where(crossover_mask, mutant, population[target_idx])
            trial = _clip_and_cast_vector(trial, request.variables)
            trial_row = _evaluate_candidate(
                request=request,
                context=context,
                candidate_id=f"de_g{generation}_trial_{target_idx}",
                vector=trial,
            )

            incumbent_row = current_rows[target_idx]
            trial_score = float(trial_row["objective_score"])
            incumbent_score = float(incumbent_row["objective_score"])
            if _is_better_score(trial_score, incumbent_score):
                population[target_idx] = trial
                selected_row = trial_row
            else:
                selected_row = incumbent_row

            snapshot_row = dict(selected_row)
            snapshot_row["candidate_id"] = f"de_g{generation}_i{target_idx}"
            generation_rows.append(snapshot_row)

        evaluated_rows.extend(generation_rows)
        current_rows = generation_rows
        scores = np.array([row["objective_score"] for row in generation_rows], dtype=float)
        best_score = float(np.min(scores)) if direction == "min" else float(np.max(scores))
        convergence_history.append({"generation": generation, "best_objective_score": best_score})

    return evaluated_rows, convergence_history


def _run_scipy_de(request: DERequest, context: SimulationExecutionContext) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assert scipy_differential_evolution is not None
    hp = request.hyperparameters
    rows_by_vector: dict[tuple[float, ...], dict[str, Any]] = {}
    evaluations: list[dict[str, Any]] = []
    convergence_history: list[dict[str, Any]] = []

    bounds = [(variable.lower_bound, variable.upper_bound) for variable in request.variables]
    eval_counter = {"idx": 0}

    def _evaluate(vector: np.ndarray) -> float:
        bounded = _clip_and_cast_vector(vector, request.variables)
        key = tuple(float(v) for v in bounded)
        cached = rows_by_vector.get(key)
        if cached is None:
            row = _evaluate_candidate(
                request=request,
                context=context,
                candidate_id=f"de_eval_{eval_counter['idx']}",
                vector=bounded,
            )
            eval_counter["idx"] += 1
            rows_by_vector[key] = row
            evaluations.append(row)
            cached = row
        return float(cached["objective_score"])

    def _callback(xk: np.ndarray, convergence: float) -> bool:
        score = _evaluate(np.asarray(xk, dtype=float))
        convergence_history.append(
            {"generation": len(convergence_history), "best_objective_score": float(score), "scipy_convergence": float(convergence)}
        )
        return False

    scipy_differential_evolution(
        func=_evaluate,
        bounds=bounds,
        maxiter=hp.generations,
        popsize=max(2, hp.population_size // max(1, len(request.variables))),
        mutation=hp.differential_weight,
        recombination=hp.crossover_rate,
        seed=request.seed,
        polish=False,
        callback=_callback,
    )
    return evaluations, convergence_history


def run_differential_evolution_analysis(
    *,
    request: DERequest,
    context: SimulationExecutionContext,
) -> DifferentialEvolutionResponse:
    """Run DE analysis with optional SciPy solver and deterministic replay metadata."""

    if not request.variables:
        raise ValueError("DE requires at least one variable.")

    use_scipy = bool(request.hyperparameters.use_scipy_if_available and scipy_differential_evolution is not None)
    rows, convergence_history = (
        _run_scipy_de(request, context)
        if use_scipy
        else _run_fallback_de(request, context)
    )

    if not rows:
        return DifferentialEvolutionResponse(
            results_df=pd.DataFrame(),
            records=[],
            best_record=None,
            convergence_history=convergence_history,
            random_seed=request.seed,
            hyperparameters=request.hyperparameters.__dict__,
            replay_candidates=[],
        )

    results_df = pd.DataFrame(rows)
    ascending = request.objective.direction == "min"
    results_df = results_df.sort_values(
        by=["objective_score", "scenario_id", "candidate_id"],
        ascending=[ascending, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    results_df["candidate_rank"] = np.arange(1, len(results_df) + 1)

    records = results_df.to_dict(orient="records")
    replay_n = max(1, request.hyperparameters.top_n_candidates)
    replay_candidates = records[:replay_n]

    return DifferentialEvolutionResponse(
        results_df=results_df,
        records=records,
        best_record=(records[0] if records else None),
        convergence_history=convergence_history,
        random_seed=request.seed,
        hyperparameters=request.hyperparameters.__dict__,
        replay_candidates=replay_candidates,
    )
