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


def build_sizing_benchmark_table(sweep_df: pd.DataFrame) -> pd.DataFrame:
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
    annotated["benchmark_duration_ok"] = annotated["duration_h"].between(2.0, 6.0, inclusive="both")
    annotated["benchmark_c_rate_ok"] = annotated["c_rate_per_h"].between(0.17, 0.5, inclusive="both")
    annotated["benchmark_reliability_ok"] = annotated.get("compliance_pct", np.nan) >= 99.0

    return annotated


def rank_recommendation_candidates(
    sweep_df: pd.DataFrame,
    *,
    ranking_column: str,
    ascending: bool,
    enforce_benchmark_gate: bool = True,
) -> pd.DataFrame:
    """Return ranked recommendation candidates using a shared sorting policy.

    The ranking policy is configurable so Streamlit displays can stay aligned with
    the backend recommendation logic for any selected KPI.
    """

    if sweep_df.empty:
        return pd.DataFrame()

    df = build_sizing_benchmark_table(sweep_df)
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
) -> Optional[pd.Series]:
    """Select a practical recommendation using the configured ranking policy."""

    ranked = rank_recommendation_candidates(
        sweep_df,
        ranking_column=ranking_column,
        ascending=ascending,
        enforce_benchmark_gate=enforce_benchmark_gate,
    )
    if ranked.empty:
        return None
    recommended = ranked.iloc[0]
    return recommended
