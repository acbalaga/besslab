"""Brute-force analysis over a deterministic power-duration grid."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from services.analysis_common import (
    BusinessRuleAssumptions,
    CandidateDefinition,
    SimulationExecutionContext,
    evaluate_candidates,
)


@dataclass(frozen=True)
class BruteForceAnalysisRequest:
    """Request payload for brute-force analysis.

    Units:
    - ``power_mw_values``: candidate power values in MW.
    - ``duration_h_values``: candidate duration values in hours.
    """

    scenario_id: str
    power_mw_values: list[float]
    duration_h_values: list[float]
    assumptions: BusinessRuleAssumptions
    deterministic: bool
    seed: int | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BruteForceAnalysisRequest":
        """Parse a dictionary payload into a validated request object."""

        return cls(
            scenario_id=str(payload["scenario_id"]),
            power_mw_values=[float(v) for v in payload["power_mw_values"]],
            duration_h_values=[float(v) for v in payload["duration_h_values"]],
            assumptions=BusinessRuleAssumptions(**payload["assumptions"]),
            deterministic=bool(payload["deterministic"]),
            seed=(None if payload.get("seed") is None else int(payload["seed"])),
        )


@dataclass(frozen=True)
class BruteForceAnalysisResponse:
    """Brute-force response payload with both tabular and record views."""

    results_df: pd.DataFrame
    records: list[dict[str, Any]]


def run_bruteforce_analysis(
    *,
    request: BruteForceAnalysisRequest,
    context: SimulationExecutionContext,
) -> BruteForceAnalysisResponse:
    """Evaluate every power-duration combination for one scenario."""

    candidates: list[CandidateDefinition] = []
    for i, power_mw in enumerate(request.power_mw_values):
        for j, duration_h in enumerate(request.duration_h_values):
            candidates.append(
                CandidateDefinition(
                    scenario_id=request.scenario_id,
                    candidate_id=f"bf_{i}_{j}",
                    power_mw=float(power_mw),
                    duration_h=float(duration_h),
                )
            )

    results_df, records = evaluate_candidates(
        analysis_mode="bruteforce",
        context=context,
        assumptions=request.assumptions,
        candidates=candidates,
        deterministic=request.deterministic,
        seed=request.seed,
    )
    return BruteForceAnalysisResponse(results_df=results_df, records=records)
