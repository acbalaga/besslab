"""Sensitivity analysis module using a shared deterministic simulation contract."""
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
class SensitivityAxis:
    """Perturbation axis for sensitivity runs.

    ``parameter`` supports ``power_mw_multiplier`` and ``duration_h_multiplier``.
    Multipliers are unitless factors applied to the base candidate values.
    """

    parameter: str
    multipliers: list[float]


@dataclass(frozen=True)
class SensitivityAnalysisRequest:
    """Request payload for sensitivity analysis.

    Units:
    - ``base_power_mw``: MW AC power.
    - ``base_duration_h``: hours at rated power.
    - ``multipliers`` in ``axes`` are unitless scaling factors.
    """

    scenario_id: str
    base_power_mw: float
    base_duration_h: float
    axes: list[SensitivityAxis]
    assumptions: BusinessRuleAssumptions
    deterministic: bool
    seed: int | None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SensitivityAnalysisRequest":
        """Parse and validate dictionary input for API/UI payloads."""

        axes = [
            SensitivityAxis(
                parameter=str(axis["parameter"]),
                multipliers=[float(v) for v in axis["multipliers"]],
            )
            for axis in payload["axes"]
        ]
        return cls(
            scenario_id=str(payload["scenario_id"]),
            base_power_mw=float(payload["base_power_mw"]),
            base_duration_h=float(payload["base_duration_h"]),
            axes=axes,
            assumptions=BusinessRuleAssumptions(**payload["assumptions"]),
            deterministic=bool(payload["deterministic"]),
            seed=(None if payload.get("seed") is None else int(payload["seed"])),
        )


@dataclass(frozen=True)
class SensitivityAnalysisResponse:
    """Sensitivity response with tabular and serializable result outputs."""

    results_df: pd.DataFrame
    records: list[dict[str, Any]]


def _build_candidate_for_axis(
    *,
    scenario_id: str,
    base_power_mw: float,
    base_duration_h: float,
    axis: SensitivityAxis,
    axis_index: int,
) -> list[CandidateDefinition]:
    """Expand one sensitivity axis into concrete candidate definitions."""

    candidates: list[CandidateDefinition] = []
    for value_index, multiplier in enumerate(axis.multipliers):
        power_mw = base_power_mw
        duration_h = base_duration_h

        if axis.parameter == "power_mw_multiplier":
            power_mw = base_power_mw * multiplier
        elif axis.parameter == "duration_h_multiplier":
            duration_h = base_duration_h * multiplier
        else:
            raise ValueError(f"Unsupported sensitivity parameter: {axis.parameter}")

        candidates.append(
            CandidateDefinition(
                scenario_id=scenario_id,
                candidate_id=f"sens_{axis_index}_{value_index}",
                power_mw=float(power_mw),
                duration_h=float(duration_h),
            )
        )
    return candidates


def run_sensitivity_analysis(
    *,
    request: SensitivityAnalysisRequest,
    context: SimulationExecutionContext,
) -> SensitivityAnalysisResponse:
    """Run one-at-a-time multiplier perturbations from a base candidate."""

    candidates: list[CandidateDefinition] = []
    for axis_index, axis in enumerate(request.axes):
        candidates.extend(
            _build_candidate_for_axis(
                scenario_id=request.scenario_id,
                base_power_mw=request.base_power_mw,
                base_duration_h=request.base_duration_h,
                axis=axis,
                axis_index=axis_index,
            )
        )

    results_df, records = evaluate_candidates(
        analysis_mode="sensitivity",
        context=context,
        assumptions=request.assumptions,
        candidates=candidates,
        deterministic=request.deterministic,
        seed=request.seed,
    )
    return SensitivityAnalysisResponse(results_df=results_df, records=records)
