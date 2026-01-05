import numpy as np
import pandas as pd
import pytest

from services.simulation_core import SimConfig, Window, infer_step_hours_from_pv, simulate_project


def _flat_cycle_table() -> pd.DataFrame:
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


def _base_config(step_hours: float, contracted_mw: float) -> SimConfig:
    return SimConfig(
        years=1,
        step_hours=step_hours,
        initial_power_mw=5.0,
        initial_usable_mwh=20.0,
        contracted_mw=contracted_mw,
        discharge_windows=[Window(0, 24)],
        pv_availability=1.0,
        bess_availability=1.0,
        rte_roundtrip=0.9,
    )


def test_hourly_profile_uses_inferred_step_and_month_boundaries() -> None:
    timestamps = pd.date_range("2021-01-01", periods=24 * 365, freq="h")
    pv_df = pd.DataFrame({"timestamp": timestamps, "pv_mw": np.ones(len(timestamps))})

    cfg = _base_config(step_hours=1.0, contracted_mw=1.0)
    cfg.step_hours = infer_step_hours_from_pv(pv_df) or cfg.step_hours

    output = simulate_project(cfg, pv_df=pv_df, cycle_df=_flat_cycle_table(), dod_override="Auto (infer)")

    feb_hours = int((pv_df["timestamp"].dt.month == 2).sum())
    feb_month = next(m for m in output.monthly_results if m.month_index == 2)

    assert len(output.final_year_logs.hod) == len(pv_df)
    assert len(output.monthly_results) == 12
    assert feb_month.expected_firm_mwh == pytest.approx(feb_hours * cfg.contracted_mw * cfg.step_hours)


def test_subhourly_profile_tracks_fractional_hours() -> None:
    timestamps = pd.date_range("2021-01-01", periods=96 * 365, freq="15min")
    pv_df = pd.DataFrame({"timestamp": timestamps, "pv_mw": np.zeros(len(timestamps))})

    cfg = _base_config(step_hours=1.0, contracted_mw=0.0)
    cfg.step_hours = infer_step_hours_from_pv(pv_df) or cfg.step_hours

    output = simulate_project(cfg, pv_df=pv_df, cycle_df=_flat_cycle_table(), dod_override="Auto (infer)")
    logs = output.final_year_logs

    assert len(logs.hod) == len(pv_df)
    np.testing.assert_allclose(logs.hod[:4], [0.0, 0.25, 0.5, 0.75])
    assert logs.hod[-1] == pytest.approx(23.75)


def test_leap_year_profile_keeps_extra_day() -> None:
    timestamps = pd.date_range("2020-01-01", periods=24 * 366, freq="h")
    pv_df = pd.DataFrame({"timestamp": timestamps, "pv_mw": np.full(len(timestamps), 2.0)})

    cfg = _base_config(step_hours=1.0, contracted_mw=1.0)
    cfg.step_hours = infer_step_hours_from_pv(pv_df) or cfg.step_hours

    output = simulate_project(cfg, pv_df=pv_df, cycle_df=_flat_cycle_table(), dod_override="Auto (infer)")
    logs = output.final_year_logs

    feb_hours = int((pv_df["timestamp"].dt.month == 2).sum())
    feb_month = next(m for m in output.monthly_results if m.month_index == 2)

    assert len(logs.hod) == 24 * 366
    assert feb_month.expected_firm_mwh == pytest.approx(feb_hours * cfg.contracted_mw * cfg.step_hours)
