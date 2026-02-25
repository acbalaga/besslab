"""Generalized brute-force search over arbitrary candidate variables."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import itertools
import logging
from typing import Any, Callable, Iterable, Sequence

import pandas as pd

from services.analysis_common import BusinessRuleAssumptions, SimulationExecutionContext
from services.simulation_core import SimulationSummary
from utils.sweeps import run_candidate_simulation


@dataclass(frozen=True)
class SearchVariable:
    """Named variable used in Cartesian brute-force search.

    Units are carried for traceability in result outputs and UI displays.
    """

    name: str
    values: tuple[float, ...]
    unit: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchVariable":
        return cls(
            name=str(payload["name"]),
            values=tuple(float(v) for v in payload["values"]),
            unit=str(payload.get("unit", "")),
        )


@dataclass(frozen=True)
class ObjectiveDefinition:
    """Objective score metadata for candidate ranking."""

    metric: str
    direction: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ObjectiveDefinition":
        return cls(metric=str(payload["metric"]), direction=str(payload["direction"]))


@dataclass(frozen=True)
class ConstraintDefinition:
    """Constraint rule with optional additive objective penalty when violated."""

    metric: str
    operator: str
    value: float
    penalty: float = 0.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConstraintDefinition":
        return cls(
            metric=str(payload["metric"]),
            operator=str(payload["operator"]),
            value=float(payload["value"]),
            penalty=float(payload.get("penalty", 0.0)),
        )


@dataclass(frozen=True)
class BruteForceRequest:
    """Request payload for generalized brute-force analysis."""

    scenario_id: str
    variables: tuple[SearchVariable, ...]
    objective: ObjectiveDefinition
    constraints: tuple[ConstraintDefinition, ...]
    assumptions: BusinessRuleAssumptions
    deterministic: bool
    seed: int | None
    max_candidates: int | None = None
    on_max_candidates: str = "raise"
    batch_size: int | None = None
    concurrency: str | None = None
    max_workers: int | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BruteForceRequest":
        """Parse dictionary payload into a validated request.

        Legacy payloads with ``power_mw_values`` and ``duration_h_values`` are
        still supported so existing UI callers keep working.
        """

        assumptions = BusinessRuleAssumptions(**payload["assumptions"])
        if "variables" in payload:
            variables = tuple(SearchVariable.from_dict(item) for item in payload["variables"])
        else:
            variables = (
                SearchVariable(name="power_mw", values=tuple(float(v) for v in payload["power_mw_values"]), unit="MW"),
                SearchVariable(name="duration_h", values=tuple(float(v) for v in payload["duration_h_values"]), unit="h"),
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
            max_candidates=(None if payload.get("max_candidates") is None else int(payload["max_candidates"])),
            on_max_candidates=str(payload.get("on_max_candidates", "raise")),
            batch_size=(None if payload.get("batch_size") is None else int(payload["batch_size"])),
            concurrency=payload.get("concurrency"),
            max_workers=(None if payload.get("max_workers") is None else int(payload["max_workers"])),
        )


BruteForceAnalysisRequest = BruteForceRequest


@dataclass(frozen=True)
class BruteForceAnalysisResponse:
    """Brute-force response payload with ranked records and counters."""

    results_df: pd.DataFrame
    records: list[dict[str, Any]]
    skipped_candidates: int
    clipped_candidates: int


def _chunked(sequence: Sequence[dict[str, float]], chunk_size: int) -> list[list[dict[str, float]]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    return [list(sequence[idx: idx + chunk_size]) for idx in range(0, len(sequence), chunk_size)]


def _build_candidate_grid(variables: Sequence[SearchVariable]) -> tuple[list[dict[str, float]], dict[str, str]]:
    if not variables:
        raise ValueError("At least one search variable is required.")

    value_vectors: list[tuple[float, ...]] = []
    units: dict[str, str] = {}
    for variable in variables:
        if not variable.values:
            raise ValueError(f"Variable '{variable.name}' must include at least one value.")
        value_vectors.append(tuple(float(v) for v in variable.values))
        units[variable.name] = variable.unit

    candidates: list[dict[str, float]] = []
    for combo in itertools.product(*value_vectors):
        candidate = {variables[idx].name: float(combo[idx]) for idx in range(len(variables))}
        candidates.append(candidate)
    return candidates, units


def _build_physical_simulation_inputs(candidate: dict[str, float]) -> tuple[float, float]:
    """Map candidate variables to simulation config inputs.

    The shared simulation wrapper currently requires power and duration. The
    adapter keeps this mapping explicit for future extension.
    """

    if "power_mw" not in candidate or "duration_h" not in candidate:
        raise ValueError("Candidates must include 'power_mw' and 'duration_h' for simulation.")
    return float(candidate["power_mw"]), float(candidate["duration_h"])


def _build_commercial_assumptions(
    candidate: dict[str, float],
    base_assumptions: BusinessRuleAssumptions,
) -> BusinessRuleAssumptions:
    """Map candidate variables to commercial assumptions.

    This adapter supports variable overrides while keeping base assumptions as
    defaults so objective and feasibility behavior stay explicit.
    """

    return BusinessRuleAssumptions(
        tariff_escalation_rate_pct=float(candidate.get("tariff_escalation_rate_pct", base_assumptions.tariff_escalation_rate_pct)),
        shortfall_penalty_usd_per_mwh=float(
            candidate.get("shortfall_penalty_usd_per_mwh", base_assumptions.shortfall_penalty_usd_per_mwh)
        ),
        min_compliance_pct=float(candidate.get("min_compliance_pct", base_assumptions.min_compliance_pct)),
        objective_metric=base_assumptions.objective_metric,
        objective_direction=base_assumptions.objective_direction,
    )


def _extract_metric(metric: str, summary: SimulationSummary, assumptions: BusinessRuleAssumptions) -> float:
    if metric == "compliance_pct":
        return float(summary.compliance)
    if metric == "shortfall_mwh":
        return float(summary.total_shortfall_mwh)
    if metric == "penalized_shortfall_cost_usd":
        return float(summary.total_shortfall_mwh) * float(assumptions.shortfall_penalty_usd_per_mwh)
    if metric == "bess_share_of_firm_pct":
        return float(summary.bess_share_of_firm)
    raise ValueError(f"Unsupported metric: {metric}")


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


def _evaluate_one_candidate(
    *,
    request: BruteForceRequest,
    context: SimulationExecutionContext,
    candidate_id: str,
    candidate: dict[str, float],
    candidate_units: dict[str, str],
) -> dict[str, Any]:
    power_mw, duration_h = _build_physical_simulation_inputs(candidate)
    assumptions = _build_commercial_assumptions(candidate, request.assumptions)

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

    objective_value = _extract_metric(request.objective.metric, summary, assumptions)
    penalties = 0.0
    violation_count = 0
    for constraint in request.constraints:
        constraint_value = _extract_metric(constraint.metric, summary, assumptions)
        if not _constraint_satisfied(constraint_value, constraint.operator, constraint.value):
            violation_count += 1
            penalties += float(constraint.penalty)

    direction = request.objective.direction
    if direction not in {"min", "max"}:
        raise ValueError("objective.direction must be 'min' or 'max'.")
    objective_score = objective_value + penalties if direction == "min" else objective_value - penalties

    row: dict[str, Any] = {
        "analysis_mode": "bruteforce",
        "scenario_id": request.scenario_id,
        "candidate_id": candidate_id,
        "candidate_rank": -1,
        "power_mw": power_mw,
        "duration_h": duration_h,
        "energy_mwh": power_mw * duration_h,
        "feasible": violation_count == 0,
        "objective_metric": request.objective.metric,
        "objective_direction": direction,
        "objective_value": float(objective_value),
        "objective_score": float(objective_score),
        "constraint_violations": int(violation_count),
        "constraint_penalty": float(penalties),
        "compliance_pct": float(summary.compliance),
        "total_shortfall_mwh": float(summary.total_shortfall_mwh),
        "bess_share_of_firm_pct": float(summary.bess_share_of_firm),
        "avg_eq_cycles_per_year": float(summary.avg_eq_cycles_per_year),
        "cap_ratio_final": float(summary.cap_ratio_final),
        "tariff_escalation_rate_pct": float(assumptions.tariff_escalation_rate_pct),
        "shortfall_penalty_usd_per_mwh": float(assumptions.shortfall_penalty_usd_per_mwh),
        "seed": request.seed,
        "deterministic": bool(request.deterministic),
    }
    for key, value in candidate.items():
        row[f"input__{key}"] = float(value)
        row[f"input_unit__{key}"] = candidate_units.get(key, "")
    return row


def _evaluate_candidates_in_batches(
    *,
    request: BruteForceRequest,
    context: SimulationExecutionContext,
    candidate_payloads: Sequence[tuple[str, dict[str, float]]],
    candidate_units: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    executor_cls: type[ThreadPoolExecutor] | None = None
    if request.concurrency is not None:
        if request.concurrency not in {"thread", "process"}:
            raise ValueError("concurrency must be one of: None, 'thread', 'process'.")
        if request.concurrency == "process":
            raise ValueError("Process concurrency is not supported for brute-force analysis; use thread or None.")
        executor_cls = ThreadPoolExecutor

    eval_fn: Callable[[tuple[str, dict[str, float]]], dict[str, Any]]

    def _evaluate(payload: tuple[str, dict[str, float]]) -> dict[str, Any]:
        candidate_id, candidate = payload
        return _evaluate_one_candidate(
            request=request,
            context=context,
            candidate_id=candidate_id,
            candidate=candidate,
            candidate_units=candidate_units,
        )

    eval_fn = _evaluate

    if executor_cls is None:
        for payload in candidate_payloads:
            rows.append(eval_fn(payload))
        return rows

    # Process pools require pickle-safe callables and simulation hooks; callers are
    # responsible for selecting an appropriate mode.
    with executor_cls(max_workers=request.max_workers) as executor:
        for row in executor.map(eval_fn, candidate_payloads):
            rows.append(row)
    return rows


def run_bruteforce_analysis(
    *,
    request: BruteForceRequest,
    context: SimulationExecutionContext,
) -> BruteForceAnalysisResponse:
    """Run generalized Cartesian brute-force search and return ranked candidates."""

    candidates, candidate_units = _build_candidate_grid(request.variables)
    candidate_count = len(candidates)

    if request.max_candidates is not None and candidate_count > request.max_candidates:
        if request.on_max_candidates == "return":
            return BruteForceAnalysisResponse(
                results_df=pd.DataFrame(),
                records=[],
                skipped_candidates=candidate_count,
                clipped_candidates=0,
            )
        if request.on_max_candidates != "batch":
            raise ValueError(
                f"Candidate count {candidate_count} exceeds max_candidates={request.max_candidates}. "
                "Use on_max_candidates='batch' or 'return' to change behavior."
            )
        logging.getLogger(__name__).warning(
            "Candidate count %s exceeds max_candidates=%s; processing in batches.",
            candidate_count,
            request.max_candidates,
        )

    batches: list[list[tuple[str, dict[str, float]]]]
    payloads = [(f"bf_{idx}", candidate) for idx, candidate in enumerate(candidates)]
    if request.batch_size is not None:
        batches = _chunked(payloads, request.batch_size)
    elif request.max_candidates is not None and candidate_count > request.max_candidates and request.on_max_candidates == "batch":
        batches = _chunked(payloads, request.max_candidates)
    else:
        batches = [payloads]

    rows: list[dict[str, Any]] = []
    for batch in batches:
        rows.extend(
            _evaluate_candidates_in_batches(
                request=request,
                context=context,
                candidate_payloads=batch,
                candidate_units=candidate_units,
            )
        )

    if not rows:
        return BruteForceAnalysisResponse(results_df=pd.DataFrame(), records=[], skipped_candidates=0, clipped_candidates=0)

    results_df = pd.DataFrame(rows)
    ascending = request.objective.direction == "min"
    results_df = results_df.sort_values(
        by=["objective_score", "scenario_id", "candidate_id"],
        ascending=[ascending, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    results_df["candidate_rank"] = range(1, len(results_df) + 1)
    records = results_df.to_dict(orient="records")

    return BruteForceAnalysisResponse(
        results_df=results_df,
        records=records,
        skipped_candidates=0,
        clipped_candidates=0,
    )
