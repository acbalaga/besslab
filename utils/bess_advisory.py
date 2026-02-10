"""Technical advisory helpers for comprehensive BESS sizing analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


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


def choose_recommended_candidate(sweep_df: pd.DataFrame) -> Optional[pd.Series]:
    """Select a practical recommendation using reliability-first screening."""

    if sweep_df.empty:
        return None

    df = build_sizing_benchmark_table(sweep_df)
    evaluated = df[df.get("status", "evaluated") == "evaluated"].copy()
    if evaluated.empty:
        return None

    fully_qualified = evaluated[
        evaluated["benchmark_reliability_ok"]
        & evaluated["benchmark_duration_ok"]
        & evaluated["benchmark_c_rate_ok"]
    ]
    candidate_pool = fully_qualified if not fully_qualified.empty else evaluated

    sort_cols = ["total_shortfall_mwh", "energy_mwh"]
    ascending = [True, True]
    if "npv_usd" in candidate_pool.columns and candidate_pool["npv_usd"].notna().any():
        sort_cols = ["benchmark_reliability_ok", "npv_usd", "energy_mwh"]
        ascending = [False, False, True]

    recommended = candidate_pool.sort_values(sort_cols, ascending=ascending).iloc[0]
    return recommended
