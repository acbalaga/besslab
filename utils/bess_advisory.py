"""Technical advisory helpers for comprehensive BESS sizing analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from utils.sweeps import _resolve_ranking_column


@dataclass(frozen=True)
class PvSizingSignals:
    """Derived PV profile indicators that influence storage sizing decisions."""

    annual_energy_mwh: float
    peak_power_mw: float
    mean_power_mw: float
    implied_capacity_factor_pct: float
    active_hours_pct: float
    p95_hourly_ramp_mw: float


def summarize_pv_sizing_signals(pv_df: pd.DataFrame) -> PvSizingSignals:
    """Return concise PV variability metrics for sizing context.

    The helper uses the observed PV peak as a proxy for installed AC capacity,
    which is practical when a separate AC nameplate field is not available.
    """

    pv_series = pd.to_numeric(pv_df.get("pv_mw"), errors="coerce").fillna(0.0)
    if pv_series.empty:
        raise ValueError("PV profile must contain at least one pv_mw value.")

    annual_energy_mwh = float(pv_series.sum())
    peak_power_mw = float(pv_series.max())
    mean_power_mw = float(pv_series.mean())
    implied_capacity_factor_pct = (
        float((mean_power_mw / peak_power_mw) * 100.0) if peak_power_mw > 0 else 0.0
    )
    active_hours_pct = float((pv_series > 0.05).mean() * 100.0)

    ramps = pv_series.diff().abs().fillna(0.0)
    p95_hourly_ramp_mw = float(ramps.quantile(0.95)) if not ramps.empty else 0.0

    return PvSizingSignals(
        annual_energy_mwh=annual_energy_mwh,
        peak_power_mw=peak_power_mw,
        mean_power_mw=mean_power_mw,
        implied_capacity_factor_pct=implied_capacity_factor_pct,
        active_hours_pct=active_hours_pct,
        p95_hourly_ramp_mw=p95_hourly_ramp_mw,
    )


def build_sizing_benchmark_table(
    sweep_df: pd.DataFrame,
    *,
    duration_band_hours: tuple[float, float] = (2.0, 6.0),
    c_rate_band_per_h: tuple[float, float] = (0.17, 0.5),
    compliance_target_pct: float = 99.0,
) -> pd.DataFrame:
    """Annotate sweep candidates with benchmark-oriented sizing indicators."""

    if sweep_df.empty:
        return sweep_df.copy()

    annotated = sweep_df.copy()
    annotated["duration_h"] = annotated["energy_mwh"] / annotated["power_mw"]
    annotated["c_rate_per_h"] = annotated["power_mw"] / annotated["energy_mwh"]
    annotated["compliance_gap_pct"] = 99.0 - annotated.get("compliance_pct", np.nan)
    annotated["shortfall_intensity_mwh_per_mw"] = annotated.get("total_shortfall_mwh", np.nan) / annotated["power_mw"]

    # Common utility-scale heuristics used in pre-feasibility benchmarking.
    # These are explicit placeholders and should be adjusted per market/code.
    duration_floor, duration_ceiling = sorted(duration_band_hours)
    c_rate_floor, c_rate_ceiling = sorted(c_rate_band_per_h)
    annotated["benchmark_duration_ok"] = annotated["duration_h"].between(
        duration_floor,
        duration_ceiling,
        inclusive="both",
    )
    annotated["benchmark_c_rate_ok"] = annotated["c_rate_per_h"].between(
        c_rate_floor,
        c_rate_ceiling,
        inclusive="both",
    )
    annotated["benchmark_reliability_ok"] = (
        annotated.get("compliance_pct", np.nan) >= compliance_target_pct
    )

    return annotated


def rank_recommendation_candidates(
    sweep_df: pd.DataFrame,
    *,
    ranking_column: str,
    ascending: bool,
    enforce_benchmark_gate: bool = True,
    duration_band_hours: tuple[float, float] = (2.0, 6.0),
    c_rate_band_per_h: tuple[float, float] = (0.17, 0.5),
    compliance_target_pct: float = 99.0,
) -> pd.DataFrame:
    """Return ranked recommendation candidates using a shared sorting policy.

    The ranking policy is configurable so Streamlit displays can stay aligned with
    the backend recommendation logic for any selected KPI.
    """

    if sweep_df.empty:
        return pd.DataFrame()

    df = build_sizing_benchmark_table(
        sweep_df,
        duration_band_hours=duration_band_hours,
        c_rate_band_per_h=c_rate_band_per_h,
        compliance_target_pct=compliance_target_pct,
    )
    evaluated = df[df.get("status", "evaluated") == "evaluated"].copy()
    if evaluated.empty:
        return evaluated

    fully_qualified = evaluated[
        evaluated["benchmark_reliability_ok"]
        & evaluated["benchmark_duration_ok"]
        & evaluated["benchmark_c_rate_ok"]
    ]
    if enforce_benchmark_gate:
        candidate_pool = fully_qualified
    else:
        candidate_pool = fully_qualified if not fully_qualified.empty else evaluated
    if candidate_pool.empty:
        return candidate_pool

    resolved_column = ranking_column
    resolved_ascending = ascending
    if resolved_column not in candidate_pool.columns or not candidate_pool[resolved_column].notna().any():
        fallback_column, fallback_ascending, _, _ = _resolve_ranking_column("reliability", None)
        resolved_column = fallback_column
        resolved_ascending = fallback_ascending

    ranking_keys = [
        "benchmark_reliability_ok",
        resolved_column,
        "total_shortfall_mwh",
        "energy_mwh",
    ]
    ranking_ascending = [False, resolved_ascending, True, True]

    return candidate_pool.sort_values(ranking_keys, ascending=ranking_ascending)


def choose_recommended_candidate(
    sweep_df: pd.DataFrame,
    *,
    ranking_column: str = "compliance_pct",
    ascending: bool = False,
    enforce_benchmark_gate: bool = True,
    duration_band_hours: tuple[float, float] = (2.0, 6.0),
    c_rate_band_per_h: tuple[float, float] = (0.17, 0.5),
    compliance_target_pct: float = 99.0,
) -> Optional[pd.Series]:
    """Select a practical recommendation using the configured ranking policy."""

    ranked = rank_recommendation_candidates(
        sweep_df,
        ranking_column=ranking_column,
        ascending=ascending,
        enforce_benchmark_gate=enforce_benchmark_gate,
        duration_band_hours=duration_band_hours,
        c_rate_band_per_h=c_rate_band_per_h,
        compliance_target_pct=compliance_target_pct,
    )
    if ranked.empty:
        return None
    recommended = ranked.iloc[0]
    return recommended


def extract_pareto_frontier(
    sweep_df: pd.DataFrame,
    *,
    economic_objective: Optional[str] = None,
) -> pd.DataFrame:
    """Return non-dominated candidates for reliability and optional economics.

    Objectives are defined as:
    - maximize compliance_pct
    - minimize total_shortfall_mwh
    - optionally maximize npv_usd OR minimize lcoe_usd_per_mwh
    """

    if sweep_df.empty:
        return sweep_df.copy()

    evaluated = sweep_df[sweep_df.get("status", "evaluated") == "evaluated"].copy()
    if evaluated.empty:
        return evaluated

    objective_specs: list[tuple[str, bool]] = [
        ("compliance_pct", True),
        ("total_shortfall_mwh", False),
    ]
    if economic_objective in {"npv_usd", "lcoe_usd_per_mwh"} and economic_objective in evaluated.columns:
        objective_specs.append((economic_objective, economic_objective == "npv_usd"))

    objective_cols = [col for col, _ in objective_specs]
    valid = evaluated.dropna(subset=objective_cols).copy()
    if valid.empty:
        return valid

    transformed_columns: list[pd.Series] = []
    for col, maximize in objective_specs:
        series = valid[col].astype(float)
        transformed_columns.append(-series if maximize else series)
    objective_matrix = np.column_stack(transformed_columns)

    dominates_all = objective_matrix[:, None, :] <= objective_matrix[None, :, :]
    strictly_better_any = objective_matrix[:, None, :] < objective_matrix[None, :, :]
    dominates = dominates_all.all(axis=2) & strictly_better_any.any(axis=2)
    dominated = dominates.any(axis=0)

    frontier = valid.loc[~dominated].copy()
    return frontier.sort_values(["power_mw", "energy_mwh"]).reset_index(drop=True)


def build_power_block_marginals(
    frontier_df: pd.DataFrame,
    *,
    economic_objective: Optional[str] = None,
) -> pd.DataFrame:
    """Compute stepwise marginal improvements across energy increments per power block."""

    if frontier_df.empty:
        return frontier_df.copy()

    marginals = frontier_df.sort_values(["power_mw", "energy_mwh"]).copy()
    group = marginals.groupby("power_mw", group_keys=False)
    marginals["prev_energy_mwh"] = group["energy_mwh"].shift(1)
    marginals["prev_compliance_pct"] = group["compliance_pct"].shift(1)
    marginals["prev_shortfall_mwh"] = group["total_shortfall_mwh"].shift(1)
    marginals["delta_energy_mwh"] = marginals["energy_mwh"] - marginals["prev_energy_mwh"]
    marginals["delta_compliance_pct"] = marginals["compliance_pct"] - marginals["prev_compliance_pct"]
    marginals["delta_shortfall_reduction_mwh"] = (
        marginals["prev_shortfall_mwh"] - marginals["total_shortfall_mwh"]
    )

    valid_delta = marginals["delta_energy_mwh"] > 0
    marginals["compliance_gain_per_mwh"] = np.where(
        valid_delta,
        marginals["delta_compliance_pct"] / marginals["delta_energy_mwh"],
        np.nan,
    )
    marginals["shortfall_reduction_per_mwh"] = np.where(
        valid_delta,
        marginals["delta_shortfall_reduction_mwh"] / marginals["delta_energy_mwh"],
        np.nan,
    )

    if economic_objective in {"npv_usd", "lcoe_usd_per_mwh"} and economic_objective in marginals.columns:
        marginals["prev_economic"] = group[economic_objective].shift(1)
        if economic_objective == "npv_usd":
            marginals["delta_economic"] = marginals[economic_objective] - marginals["prev_economic"]
        else:
            marginals["delta_economic"] = marginals["prev_economic"] - marginals[economic_objective]
        marginals["economic_marginal_value_per_mwh"] = np.where(
            valid_delta,
            marginals["delta_economic"] / marginals["delta_energy_mwh"],
            np.nan,
        )

    score_columns = ["compliance_gain_per_mwh", "shortfall_reduction_per_mwh"]
    if "economic_marginal_value_per_mwh" in marginals.columns:
        score_columns.append("economic_marginal_value_per_mwh")

    rank_parts = []
    for column in score_columns:
        rank_parts.append(group[column].rank(method="average", pct=True))
    marginals["elbow_score"] = pd.concat(rank_parts, axis=1).mean(axis=1)
    valid_rows = marginals["delta_energy_mwh"].notna()
    elbow_idx = marginals[valid_rows].groupby("power_mw")["elbow_score"].idxmax()
    marginals["is_elbow"] = False
    if len(elbow_idx) > 0:
        marginals.loc[elbow_idx.values, "is_elbow"] = True

    return marginals
