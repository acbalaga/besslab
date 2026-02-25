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
    two_way_grid: dict[str, Any] | None = None

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
            two_way_grid=payload.get("two_way_grid"),
        )


@dataclass(frozen=True)
class SensitivityAnalysisResponse:
    """Sensitivity response with tabular and serializable result outputs."""

    results_df: pd.DataFrame
    records: list[dict[str, Any]]
    long_form_df: pd.DataFrame | None = None
    long_form_records: list[dict[str, Any]] | None = None
    matrix_df: pd.DataFrame | None = None
    matrix_records: list[dict[str, Any]] | None = None


KPI_COLUMNS: tuple[str, ...] = (
    "compliance_pct",
    "total_shortfall_mwh",
    "bess_share_of_firm_pct",
    "avg_eq_cycles_per_year",
    "cap_ratio_final",
)

KPI_UNITS: dict[str, str] = {
    "compliance_pct": "%",
    "total_shortfall_mwh": "MWh",
    "bess_share_of_firm_pct": "%",
    "avg_eq_cycles_per_year": "cycles/year",
    "cap_ratio_final": "ratio",
}


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


def _apply_parameter_multiplier(
    parameter: str,
    multiplier: float,
    base_power_mw: float,
    base_duration_h: float,
) -> tuple[float, float]:
    """Return power and duration values after applying one supported multiplier."""

    power_mw = float(base_power_mw)
    duration_h = float(base_duration_h)
    if parameter == "power_mw_multiplier":
        power_mw = float(base_power_mw * multiplier)
    elif parameter == "duration_h_multiplier":
        duration_h = float(base_duration_h * multiplier)
    else:
        raise ValueError(f"Unsupported sensitivity parameter: {parameter}")
    return power_mw, duration_h


def _build_two_way_grid_candidates(
    *,
    scenario_id: str,
    base_power_mw: float,
    base_duration_h: float,
    parameter_a: str,
    parameter_b: str,
    values_a: list[float],
    values_b: list[float],
) -> list[CandidateDefinition]:
    """Build cartesian candidates for two-way sensitivity over two multiplier axes."""

    candidates: list[CandidateDefinition] = [
        CandidateDefinition(
            scenario_id=scenario_id,
            candidate_id="baseline",
            power_mw=float(base_power_mw),
            duration_h=float(base_duration_h),
        )
    ]
    for index_a, value_a in enumerate(values_a):
        for index_b, value_b in enumerate(values_b):
            power_mw = float(base_power_mw)
            duration_h = float(base_duration_h)
            for parameter, multiplier in ((parameter_a, value_a), (parameter_b, value_b)):
                power_mw, duration_h = _apply_parameter_multiplier(
                    parameter,
                    multiplier,
                    power_mw,
                    duration_h,
                )

            candidates.append(
                CandidateDefinition(
                    scenario_id=scenario_id,
                    candidate_id=f"grid_{index_a}_{index_b}",
                    power_mw=power_mw,
                    duration_h=duration_h,
                )
            )
    return candidates


def _compute_delta(value: float, baseline: float, delta_mode: str) -> float | None:
    if delta_mode == "absolute":
        return float(value - baseline)
    if delta_mode == "percent":
        if baseline == 0.0:
            return None
        return float((value - baseline) / abs(baseline) * 100.0)
    raise ValueError(f"Unsupported delta_mode: {delta_mode}")


def _build_two_way_outputs(
    *,
    scenario_id: str,
    results_df: pd.DataFrame,
    parameter_a: str,
    parameter_b: str,
    values_a: list[float],
    values_b: list[float],
    selected_kpis: list[str],
    delta_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build long-form and matrix outputs for two-way sensitivity grids."""

    baseline = results_df.loc[results_df["candidate_id"] == "baseline"]
    if baseline.empty:
        raise ValueError("Two-way sensitivity requires a baseline candidate result.")
    baseline_row = baseline.iloc[0]

    grid_rows = []
    grid_only = results_df.loc[results_df["candidate_id"].str.startswith("grid_")].reset_index(drop=True)
    expected_points = len(values_a) * len(values_b)
    if len(grid_only) != expected_points:
        raise ValueError("Two-way sensitivity output size mismatch with requested grid.")

    for _, row in grid_only.iterrows():
        _, index_a_raw, index_b_raw = str(row["candidate_id"]).split("_")
        value_a = float(values_a[int(index_a_raw)])
        value_b = float(values_b[int(index_b_raw)])
        for kpi_name in selected_kpis:
            kpi_value = float(row[kpi_name])
            baseline_value = float(baseline_row[kpi_name])
            delta_value = _compute_delta(kpi_value, baseline_value, delta_mode)
            grid_rows.append(
                {
                    "scenario_id": scenario_id,
                    "candidate_id": row["candidate_id"],
                    "parameter_a_name": parameter_a,
                    "parameter_a_value": value_a,
                    "parameter_b_name": parameter_b,
                    "parameter_b_value": value_b,
                    "kpi_name": kpi_name,
                    "kpi_value": kpi_value,
                    "kpi_unit": KPI_UNITS.get(kpi_name, ""),
                    "baseline_kpi_value": baseline_value,
                    "delta_mode": delta_mode,
                    "delta_value": delta_value,
                }
            )

    long_form_df = pd.DataFrame(grid_rows)
    matrix_frames: list[pd.DataFrame] = []
    for kpi_name in selected_kpis:
        kpi_long = long_form_df.loc[long_form_df["kpi_name"] == kpi_name]
        pivot_df = kpi_long.pivot(
            index="parameter_a_value",
            columns="parameter_b_value",
            values="delta_value",
        ).sort_index(axis=0).sort_index(axis=1)
        pivot_df.columns = [str(col) for col in pivot_df.columns]
        matrix_row = pivot_df.reset_index()
        matrix_row.insert(0, "scenario_id", scenario_id)
        matrix_row.insert(1, "kpi_name", kpi_name)
        matrix_row.insert(2, "kpi_unit", KPI_UNITS.get(kpi_name, ""))
        matrix_row.insert(3, "parameter_a_name", parameter_a)
        matrix_row.insert(4, "parameter_b_name", parameter_b)
        matrix_row.insert(5, "delta_mode", delta_mode)
        matrix_frames.append(matrix_row)

    matrix_df = pd.concat(matrix_frames, ignore_index=True) if matrix_frames else pd.DataFrame()
    return long_form_df, matrix_df


def run_sensitivity_analysis(
    *,
    request: SensitivityAnalysisRequest,
    context: SimulationExecutionContext,
) -> SensitivityAnalysisResponse:
    """Run one-at-a-time multiplier perturbations from a base candidate."""

    if request.two_way_grid:
        grid = request.two_way_grid
        parameter_a = str(grid["parameter_a"])
        parameter_b = str(grid["parameter_b"])
        values_a = [float(v) for v in grid["values_a"]]
        values_b = [float(v) for v in grid["values_b"]]
        selected_kpis = [str(k) for k in grid.get("selected_kpis", list(KPI_COLUMNS))]
        invalid_kpis = [kpi for kpi in selected_kpis if kpi not in KPI_COLUMNS]
        if invalid_kpis:
            raise ValueError(f"Unsupported KPI columns requested: {invalid_kpis}")
        delta_mode = str(grid.get("delta_mode", "absolute"))

        candidates = _build_two_way_grid_candidates(
            scenario_id=request.scenario_id,
            base_power_mw=request.base_power_mw,
            base_duration_h=request.base_duration_h,
            parameter_a=parameter_a,
            parameter_b=parameter_b,
            values_a=values_a,
            values_b=values_b,
        )
        results_df, records = evaluate_candidates(
            analysis_mode="sensitivity",
            context=context,
            assumptions=request.assumptions,
            candidates=candidates,
            deterministic=request.deterministic,
            seed=request.seed,
        )
        long_form_df, matrix_df = _build_two_way_outputs(
            scenario_id=request.scenario_id,
            results_df=results_df,
            parameter_a=parameter_a,
            parameter_b=parameter_b,
            values_a=values_a,
            values_b=values_b,
            selected_kpis=selected_kpis,
            delta_mode=delta_mode,
        )
        return SensitivityAnalysisResponse(
            results_df=results_df,
            records=records,
            long_form_df=long_form_df,
            long_form_records=long_form_df.to_dict(orient="records"),
            matrix_df=matrix_df,
            matrix_records=matrix_df.to_dict(orient="records"),
        )

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
