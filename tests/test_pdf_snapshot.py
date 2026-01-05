import numpy as np

from app import build_pdf_summary
from services.simulation_core import SimConfig, YearResult


def _sample_year_result() -> YearResult:
    return YearResult(
        year_index=1,
        expected_firm_mwh=100.0,
        delivered_firm_mwh=95.0,
        shortfall_mwh=5.0,
        breach_days=2,
        charge_mwh=60.0,
        discharge_mwh=55.0,
        available_pv_mwh=140.0,
        pv_to_contract_mwh=80.0,
        bess_to_contract_mwh=15.0,
        avg_rte=0.9,
        eq_cycles=120.0,
        cum_cycles=120.0,
        soh_cycle=0.98,
        soh_calendar=0.99,
        soh_total=0.97,
        eoy_usable_mwh=18.0,
        eoy_power_mw=9.5,
        pv_curtailed_mwh=20.0,
        flags={},
    )


def test_build_pdf_summary_returns_bytes():
    cfg = SimConfig(years=1, contracted_mw=10.0, initial_power_mw=10.0, initial_usable_mwh=20.0)
    results = [_sample_year_result()]

    hod_count = np.ones(24)
    hod_sum_pv_resource = np.zeros(24)
    hod_sum_pv = np.zeros(24)
    hod_sum_bess = np.zeros(24)
    hod_sum_charge = np.zeros(24)

    pdf_bytes = build_pdf_summary(
        cfg,
        results,
        compliance=95.0,
        bess_share=15.0,
        charge_discharge_ratio=1.05,
        pv_capture_ratio=0.75,
        discharge_capacity_factor=0.26,
        discharge_windows_text="10-14, 18-22",
        charge_windows_text="Any PV hour",
        hod_count=hod_count,
        hod_sum_pv_resource=hod_sum_pv_resource,
        hod_sum_pv=hod_sum_pv,
        hod_sum_bess=hod_sum_bess,
        hod_sum_charge=hod_sum_charge,
        total_shortfall_mwh=5.0,
        pv_excess_mwh=20.0,
        total_generation_mwh=95.0,
        bess_generation_mwh=15.0,
        pv_generation_mwh=80.0,
        bess_losses_mwh=5.0,
    )

    assert isinstance(pdf_bytes, (bytes, bytearray))
    assert len(pdf_bytes) > 0
