"""Shared analysis helpers for deterministic BESS candidate evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import pandas as pd

from services.simulation_core import SimConfig, SimulationOutput, SimulationSummary
from utils.sweeps import run_candidate_simulation


NORMALIZED_RESULT_COLUMNS: tuple[str, ...] = (
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
)


@dataclass(frozen=True)
class BusinessRuleAssumptions:
    """Explicit business assumptions used by analysis routines.

    Units:
    - ``tariff_escalation_rate_pct``: annual percent increase (%/year).
    - ``shortfall_penalty_usd_per_mwh``: penalty applied for shortfall (USD/MWh).
    - ``min_compliance_pct``: minimum compliance threshold for feasibility (0..1).
    """

    tariff_escalation_rate_pct: float
    shortfall_penalty_usd_per_mwh: float
    min_compliance_pct: float
    objective_metric: str
    objective_direction: str


@dataclass(frozen=True)
class SimulationExecutionContext:
    """Inputs required to run candidate simulations against the shared core."""

    base_cfg: SimConfig
    pv_df: pd.DataFrame
    cycle_df: pd.DataFrame
    dod_override: str
    need_logs: bool = False
    simulate_fn: Callable[..., SimulationOutput] | None = None
    summarize_fn: Callable[[SimulationOutput], SimulationSummary] | None = None


@dataclass(frozen=True)
class CandidateDefinition:
    """BESS candidate definition.

    Units:
    - ``power_mw``: MW AC power rating.
    - ``duration_h``: discharge duration at rated power in hours.
    """

    scenario_id: str
    candidate_id: str
    power_mw: float
    duration_h: float


def _compute_objective_value(
    assumptions: BusinessRuleAssumptions,
    summary: SimulationSummary,
) -> float:
    """Compute the objective value using explicit assumptions only."""

    if assumptions.objective_metric == "compliance_pct":
        return float(summary.compliance)
    if assumptions.objective_metric == "shortfall_mwh":
        return float(summary.total_shortfall_mwh)
    if assumptions.objective_metric == "penalized_shortfall_cost_usd":
        return float(summary.total_shortfall_mwh) * float(assumptions.shortfall_penalty_usd_per_mwh)
    raise ValueError(f"Unsupported objective_metric: {assumptions.objective_metric}")


def evaluate_candidates(
    *,
    analysis_mode: str,
    context: SimulationExecutionContext,
    assumptions: BusinessRuleAssumptions,
    candidates: Sequence[CandidateDefinition],
    seed: int | None,
    deterministic: bool,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run candidate simulations through one contract and return normalized outputs."""

    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        _, summary = run_candidate_simulation(
            base_cfg=context.base_cfg,
            pv_df=context.pv_df,
            cycle_df=context.cycle_df,
            dod_override=context.dod_override,
            power_mw=float(candidate.power_mw),
            duration_h=float(candidate.duration_h),
            simulate_fn=context.simulate_fn,
            summarize_fn=context.summarize_fn,
            need_logs=context.need_logs,
            seed=seed,
            deterministic=deterministic,
        )

        objective_value = _compute_objective_value(assumptions, summary)
        rows.append(
            {
                "analysis_mode": analysis_mode,
                "scenario_id": candidate.scenario_id,
                "candidate_id": candidate.candidate_id,
                "candidate_rank": -1,
                "power_mw": float(candidate.power_mw),
                "duration_h": float(candidate.duration_h),
                "energy_mwh": float(candidate.power_mw * candidate.duration_h),
                "feasible": bool(float(summary.compliance) >= assumptions.min_compliance_pct),
                "objective_metric": assumptions.objective_metric,
                "objective_direction": assumptions.objective_direction,
                "objective_value": float(objective_value),
                "compliance_pct": float(summary.compliance),
                "total_shortfall_mwh": float(summary.total_shortfall_mwh),
                "bess_share_of_firm_pct": float(summary.bess_share_of_firm),
                "avg_eq_cycles_per_year": float(summary.avg_eq_cycles_per_year),
                "cap_ratio_final": float(summary.cap_ratio_final),
                "tariff_escalation_rate_pct": float(assumptions.tariff_escalation_rate_pct),
                "shortfall_penalty_usd_per_mwh": float(assumptions.shortfall_penalty_usd_per_mwh),
                "seed": seed,
                "deterministic": bool(deterministic),
            }
        )

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return pd.DataFrame(columns=NORMALIZED_RESULT_COLUMNS), []

    ascending = assumptions.objective_direction == "min"
    results_df = results_df.sort_values(
        by=["objective_value", "scenario_id", "candidate_id"],
        ascending=[ascending, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    results_df["candidate_rank"] = range(1, len(results_df) + 1)
    results_df = results_df.loc[:, list(NORMALIZED_RESULT_COLUMNS)]
    return results_df, results_df.to_dict(orient="records")
