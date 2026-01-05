import numpy as np
import pandas as pd
import pytest

from services.simulation_core import SimConfig, Window, infer_dod_bucket, resolve_efficiencies, simulate_project


def test_infer_dod_bucket_scales_with_available_energy():
    daily_dis = np.array([50.0, 52.0, 48.0])

    # When the usable energy has faded to ~60 MWh, the median daily discharge
    # (~50 MWh) should map to an ~80% DoD bucket rather than dropping to 40%
    # because of a stale BOL reference.
    bucket_with_faded_energy = infer_dod_bucket(daily_dis, usable_mwh_available=60.0)
    bucket_with_bol_energy = infer_dod_bucket(daily_dis, usable_mwh_available=120.0)

    assert bucket_with_faded_energy == 80
    assert bucket_with_bol_energy == 40


def _flat_cycle_table() -> pd.DataFrame:
    """Return a simple cycle table that keeps SOH flat across DoD buckets."""

    data = {
        "DoD10_Cycles": [0, 100_000],
        "DoD10_Ret(%)": [100, 100],
        "DoD20_Cycles": [0, 100_000],
        "DoD20_Ret(%)": [100, 100],
        "DoD40_Cycles": [0, 100_000],
        "DoD40_Ret(%)": [100, 100],
        "DoD80_Cycles": [0, 100_000],
        "DoD80_Ret(%)": [100, 100],
        "DoD100_Cycles": [0, 100_000],
        "DoD100_Ret(%)": [100, 100],
    }
    return pd.DataFrame(data)


def test_resolve_efficiencies_respects_split_inputs() -> None:
    cfg = SimConfig(
        use_split_rte=True,
        charge_efficiency=0.78,
        discharge_efficiency=0.92,
        rte_roundtrip=0.85,  # ignored when split inputs are present
    )

    eta_ch, eta_dis, eta_rt = resolve_efficiencies(cfg)

    assert eta_ch == pytest.approx(0.78)
    assert eta_dis == pytest.approx(0.92)
    assert eta_rt == pytest.approx(0.78 * 0.92)


def test_equivalent_cycles_use_adjusted_usable_energy_across_years():
    """Year 1 cycles should not overcount relative to later years."""

    cycle_df = _flat_cycle_table()

    pv_profile = []
    for _ in range(365):
        for hod in range(24):
            pv_profile.append(0.0 if 6 <= hod < 18 else 40.0)
    pv_df = pd.DataFrame({"pv_mw": pv_profile})

    cfg = SimConfig(
        years=2,
        initial_power_mw=25.0,
        initial_usable_mwh=100.0,
        contracted_mw=20.0,
        discharge_windows=[Window(6, 18)],
        bess_availability=0.97,
        rte_roundtrip=0.89,
        pv_availability=1.0,
    )

    output = simulate_project(
        cfg, pv_df=pv_df, cycle_df=cycle_df, dod_override="Auto (infer)", need_logs=False
    )

    year1_cycles, year2_cycles = [r.eq_cycles for r in output.results]

    assert year1_cycles > 0
    assert year1_cycles == pytest.approx(year2_cycles, rel=0.1)
