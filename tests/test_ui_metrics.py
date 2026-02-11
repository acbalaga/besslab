from types import SimpleNamespace

import pytest

from frontend.ui.metrics import compute_kpis
from services.simulation_core import SimConfig, YearResult


def _year_result(*, expected_firm_mwh: float, shortfall_mwh: float, soh_total: float = 0.9) -> YearResult:
    delivered_firm_mwh = expected_firm_mwh - shortfall_mwh
    return YearResult(
        year_index=1,
        expected_firm_mwh=expected_firm_mwh,
        delivered_firm_mwh=delivered_firm_mwh,
        shortfall_mwh=shortfall_mwh,
        breach_days=0,
        charge_mwh=0.0,
        discharge_mwh=0.0,
        available_pv_mwh=0.0,
        pv_to_contract_mwh=0.0,
        bess_to_contract_mwh=0.0,
        avg_rte=0.0,
        eq_cycles=0.0,
        cum_cycles=0.0,
        soh_cycle=1.0,
        soh_calendar=1.0,
        soh_total=soh_total,
        eoy_usable_mwh=0.0,
        eoy_power_mw=0.0,
        pv_curtailed_mwh=0.0,
        flags={},
    )


def test_compute_kpis_includes_deficit_and_surplus_percent() -> None:
    cfg = SimConfig()
    results = [
        _year_result(expected_firm_mwh=100.0, shortfall_mwh=10.0, soh_total=0.92),
        _year_result(expected_firm_mwh=50.0, shortfall_mwh=5.0, soh_total=0.91),
    ]
    summary = SimpleNamespace(
        compliance=90.0,
        cap_ratio_final=0.8,
        bess_share_of_firm=0.0,
        charge_discharge_ratio=0.0,
        pv_capture_ratio=0.0,
        discharge_capacity_factor=0.0,
        total_project_generation_mwh=0.0,
        bess_generation_mwh=0.0,
        pv_generation_mwh=0.0,
        pv_excess_mwh=30.0,
        bess_losses_mwh=0.0,
        total_shortfall_mwh=15.0,
        avg_eq_cycles_per_year=0.0,
        surplus_pct=20.0,
    )

    kpis = compute_kpis(
        cfg,
        results,
        summary,
        augmentation_events=2,
        augmentation_energy_added_mwh=[3.0, 2.0],
    )

    assert kpis.deficit_pct == pytest.approx(10.0)
    assert kpis.surplus_pct == pytest.approx(20.0)


def test_compute_kpis_deficit_percent_nan_with_zero_expected() -> None:
    cfg = SimConfig()
    results = [_year_result(expected_firm_mwh=0.0, shortfall_mwh=0.0)]
    summary = SimpleNamespace(
        compliance=float("nan"),
        cap_ratio_final=0.0,
        bess_share_of_firm=0.0,
        charge_discharge_ratio=0.0,
        pv_capture_ratio=0.0,
        discharge_capacity_factor=0.0,
        total_project_generation_mwh=0.0,
        bess_generation_mwh=0.0,
        pv_generation_mwh=0.0,
        pv_excess_mwh=0.0,
        bess_losses_mwh=0.0,
        total_shortfall_mwh=0.0,
        avg_eq_cycles_per_year=0.0,
        surplus_pct=float("nan"),
    )

    kpis = compute_kpis(
        cfg,
        results,
        summary,
        augmentation_events=0,
        augmentation_energy_added_mwh=[],
    )

    assert kpis.deficit_pct != kpis.deficit_pct
