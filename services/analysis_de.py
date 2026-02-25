"""Differential Evolution (DE) analysis using shared simulation evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from services.analysis_common import (
    BusinessRuleAssumptions,
    CandidateDefinition,
    SimulationExecutionContext,
    evaluate_candidates,
)


@dataclass(frozen=True)
class DifferentialEvolutionRequest:
    """Request payload for DE optimization.

    Units:
    - ``power_mw_bounds``: [min, max] bounds in MW.
    - ``duration_h_bounds``: [min, max] bounds in hours.
    - ``differential_weight`` and ``crossover_rate`` are unitless.
    """

    scenario_id: str
    power_mw_bounds: tuple[float, float]
    duration_h_bounds: tuple[float, float]
    population_size: int
    generations: int
    differential_weight: float
    crossover_rate: float
    assumptions: BusinessRuleAssumptions
    deterministic: bool
    seed: int | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DifferentialEvolutionRequest":
        """Parse dictionary payload into a validated DE request."""

        return cls(
            scenario_id=str(payload["scenario_id"]),
            power_mw_bounds=(float(payload["power_mw_bounds"][0]), float(payload["power_mw_bounds"][1])),
            duration_h_bounds=(float(payload["duration_h_bounds"][0]), float(payload["duration_h_bounds"][1])),
            population_size=int(payload["population_size"]),
            generations=int(payload["generations"]),
            differential_weight=float(payload["differential_weight"]),
            crossover_rate=float(payload["crossover_rate"]),
            assumptions=BusinessRuleAssumptions(**payload["assumptions"]),
            deterministic=bool(payload["deterministic"]),
            seed=(None if payload.get("seed") is None else int(payload["seed"])),
        )


@dataclass(frozen=True)
class DifferentialEvolutionResponse:
    """DE response payload with normalized outputs and best candidate."""

    results_df: pd.DataFrame
    records: list[dict[str, Any]]
    best_record: dict[str, Any] | None


def _clip_candidate(vector: np.ndarray, request: DifferentialEvolutionRequest) -> np.ndarray:
    """Enforce configured power/duration bounds for proposed vectors."""

    return np.array(
        [
            np.clip(vector[0], request.power_mw_bounds[0], request.power_mw_bounds[1]),
            np.clip(vector[1], request.duration_h_bounds[0], request.duration_h_bounds[1]),
        ],
        dtype=float,
    )


def run_differential_evolution_analysis(
    *,
    request: DifferentialEvolutionRequest,
    context: SimulationExecutionContext,
) -> DifferentialEvolutionResponse:
    """Run a compact DE/rand/1/bin loop and evaluate via shared helper."""

    rng = np.random.default_rng(request.seed)
    population = np.column_stack(
        [
            rng.uniform(request.power_mw_bounds[0], request.power_mw_bounds[1], size=request.population_size),
            rng.uniform(request.duration_h_bounds[0], request.duration_h_bounds[1], size=request.population_size),
        ]
    )

    history_candidates: list[CandidateDefinition] = []
    for i in range(request.population_size):
        history_candidates.append(
            CandidateDefinition(
                scenario_id=request.scenario_id,
                candidate_id=f"de_0_{i}",
                power_mw=float(population[i, 0]),
                duration_h=float(population[i, 1]),
            )
        )

    for generation in range(1, request.generations + 1):
        for target_idx in range(request.population_size):
            candidate_indices = [idx for idx in range(request.population_size) if idx != target_idx]
            a_idx, b_idx, c_idx = rng.choice(candidate_indices, size=3, replace=False)
            mutant = population[a_idx] + request.differential_weight * (population[b_idx] - population[c_idx])
            mutant = _clip_candidate(mutant, request)

            crossover_mask = rng.random(2) < request.crossover_rate
            if not crossover_mask.any():
                crossover_mask[rng.integers(0, 2)] = True
            trial = np.where(crossover_mask, mutant, population[target_idx])

            population[target_idx] = trial
            history_candidates.append(
                CandidateDefinition(
                    scenario_id=request.scenario_id,
                    candidate_id=f"de_{generation}_{target_idx}",
                    power_mw=float(trial[0]),
                    duration_h=float(trial[1]),
                )
            )

    results_df, records = evaluate_candidates(
        analysis_mode="differential_evolution",
        context=context,
        assumptions=request.assumptions,
        candidates=history_candidates,
        deterministic=request.deterministic,
        seed=request.seed,
    )
    best = records[0] if records else None
    return DifferentialEvolutionResponse(results_df=results_df, records=records, best_record=best)
