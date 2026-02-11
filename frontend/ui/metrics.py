"""Reusable KPI helpers for Streamlit pages."""

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import streamlit as st

from services.simulation_core import SimConfig, YearResult


@dataclass
class KPIResults:
    compliance: float
    min_yearly_coverage: float
    final_soh_pct: float
    eoy_capacity_margin_pct: float
    augmentation_events: int
    augmentation_energy_mwh: float
    bess_share_of_firm: float
    charge_discharge_ratio: float
    pv_capture_ratio: float
    discharge_capacity_factor: float
    total_project_generation_mwh: float
    bess_generation_mwh: float
    pv_generation_mwh: float
    pv_excess_mwh: float
    bess_losses_mwh: float
    total_shortfall_mwh: float
    avg_eq_cycles_per_year: float
    cap_ratio_final: float
    surplus_pct: float
    deficit_pct: float


def compute_kpis(
    cfg: SimConfig,
    results: List[YearResult],
    summary,
    augmentation_events: int,
    augmentation_energy_added_mwh: Sequence[float],
) -> KPIResults:
    """Derive headline KPIs from the simulation summary for reuse across pages."""
    coverage_by_year = [
        (r.delivered_firm_mwh / r.expected_firm_mwh) if r.expected_firm_mwh > 0 else float("nan") for r in results
    ]
    min_yearly_coverage = float(np.nanmin(coverage_by_year)) if coverage_by_year else float("nan")
    final = results[-1]
    expected_total_mwh = float(np.sum([r.expected_firm_mwh for r in results]))
    deficit_pct = (summary.total_shortfall_mwh / expected_total_mwh * 100.0) if expected_total_mwh > 0 else float("nan")
    augmentation_energy_mwh = float(np.sum(augmentation_energy_added_mwh)) if augmentation_energy_added_mwh else 0.0
    return KPIResults(
        compliance=summary.compliance,
        min_yearly_coverage=min_yearly_coverage,
        final_soh_pct=final.soh_total * 100.0,
        eoy_capacity_margin_pct=summary.cap_ratio_final * 100.0,
        augmentation_events=augmentation_events,
        augmentation_energy_mwh=augmentation_energy_mwh,
        bess_share_of_firm=summary.bess_share_of_firm,
        charge_discharge_ratio=summary.charge_discharge_ratio,
        pv_capture_ratio=summary.pv_capture_ratio,
        discharge_capacity_factor=summary.discharge_capacity_factor,
        total_project_generation_mwh=summary.total_project_generation_mwh,
        bess_generation_mwh=summary.bess_generation_mwh,
        pv_generation_mwh=summary.pv_generation_mwh,
        pv_excess_mwh=summary.pv_excess_mwh,
        bess_losses_mwh=summary.bess_losses_mwh,
        total_shortfall_mwh=summary.total_shortfall_mwh,
        avg_eq_cycles_per_year=summary.avg_eq_cycles_per_year,
        cap_ratio_final=summary.cap_ratio_final,
        surplus_pct=summary.surplus_pct,
        deficit_pct=deficit_pct,
    )


def _fmt_percent(value: float, as_fraction: bool = False) -> str:
    if np.isnan(value):
        return "—"
    pct_value = value * 100.0 if as_fraction else value
    return f"{pct_value:,.2f}%"


def render_primary_metrics(cfg: SimConfig, kpis: KPIResults) -> None:
    """Render the top-level KPI cards shown after a simulation run."""
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Compliance %",
        _fmt_percent(kpis.compliance),
        help="Total firm energy delivered vs expected firm energy across project life.",
    )
    c2.metric(
        "Deficit %",
        _fmt_percent(kpis.deficit_pct),
        help="Total shortfall MWh divided by expected firm MWh across the project life.",
    )
    c3.metric(
        "Surplus %",
        _fmt_percent(kpis.surplus_pct),
        help="PV curtailed MWh divided by expected firm MWh across the project life.",
    )
    c4.metric(
        "Final SOH total",
        _fmt_percent(kpis.final_soh_pct, as_fraction=False),
        help="End-of-life usable fraction after cycle + calendar fade.",
    )
    c5.metric(
        "Augmentation Events",
        f"{kpis.augmentation_events}",
        help=f"Energy added over life: {kpis.augmentation_energy_mwh:,.0f} MWh (BOL basis).",
    )
