from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.simulation_core import (  # noqa: E402
    SimConfig,
    SimulationOutput,
    Window,
    simulate_project,
    summarize_simulation,
)


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


def test_simulation_core_runs_with_parsed_windows() -> None:
    """Smoke-test the extracted core without Streamlit dependencies."""

    cfg = SimConfig(
        years=1,
        step_hours=1.0,
        contracted_mw=2.0,
        initial_power_mw=2.0,
        initial_usable_mwh=4.0,
        discharge_windows=[Window(0, 1)],
        charge_windows=[Window(1, 2)],
        pv_availability=1.0,
        bess_availability=1.0,
    )
    pv_df = pd.DataFrame({"pv_mw": [2.5, 2.5]})
    cycle_df = _flat_cycle_table()

    output: SimulationOutput = simulate_project(
        cfg, pv_df=pv_df, cycle_df=cycle_df, dod_override="Auto (infer)", need_logs=True
    )

    assert len(output.results) == 1
    first_year = output.results[0]
    assert first_year.shortfall_mwh == pytest.approx(0.0)
    assert first_year.delivered_firm_mwh == pytest.approx(cfg.contracted_mw * cfg.step_hours)
    assert output.cfg.charge_windows == cfg.charge_windows

    summary = summarize_simulation(output)
    assert summary.compliance == pytest.approx(100.0)
    assert summary.surplus_pct == pytest.approx((summary.pv_excess_mwh / first_year.expected_firm_mwh) * 100.0)


def test_simulation_core_uses_contracted_schedule() -> None:
    """Ensure the hourly schedule overrides discharge windows when provided."""

    schedule = [2.0] + [0.0] * 23
    cfg = SimConfig(
        years=1,
        step_hours=1.0,
        contracted_mw=5.0,
        contracted_mw_schedule=schedule,
        initial_power_mw=2.0,
        initial_usable_mwh=4.0,
        discharge_windows=[Window(10, 11)],
        charge_windows=[],
        pv_availability=1.0,
        bess_availability=1.0,
    )
    pv_df = pd.DataFrame({"pv_mw": [2.5, 0.0]})
    cycle_df = _flat_cycle_table()

    output = simulate_project(cfg, pv_df=pv_df, cycle_df=cycle_df, dod_override="Auto (infer)", need_logs=True)

    first_year = output.results[0]
    assert first_year.expected_firm_mwh == pytest.approx(2.0)
    assert first_year.delivered_firm_mwh == pytest.approx(2.0)


def test_simulation_core_allows_profile_without_windows() -> None:
    """Requirement profiles should not require discharge windows."""

    cfg = SimConfig(
        years=1,
        step_hours=1.0,
        contracted_mw=5.0,
        contracted_mw_profile=[1.0, 0.0],
        initial_power_mw=2.0,
        initial_usable_mwh=4.0,
        discharge_windows=[],
        charge_windows=[],
        pv_availability=1.0,
        bess_availability=1.0,
    )
    pv_df = pd.DataFrame({"pv_mw": [2.5, 0.0]})
    cycle_df = _flat_cycle_table()

    output = simulate_project(cfg, pv_df=pv_df, cycle_df=cycle_df, dod_override="Auto (infer)", need_logs=True)

    first_year = output.results[0]
    assert first_year.expected_firm_mwh == pytest.approx(1.0)


def test_simulation_core_repeats_profile_to_match_length() -> None:
    """Profiles should repeat when PV data spans multiple profile lengths."""

    cfg = SimConfig(
        years=1,
        step_hours=1.0,
        contracted_mw=5.0,
        contracted_mw_profile=[1.0, 0.0],
        initial_power_mw=2.0,
        initial_usable_mwh=4.0,
        discharge_windows=[],
        charge_windows=[],
        pv_availability=1.0,
        bess_availability=1.0,
    )
    pv_df = pd.DataFrame({"pv_mw": [2.5, 0.0, 2.5, 0.0]})
    cycle_df = _flat_cycle_table()

    output = simulate_project(cfg, pv_df=pv_df, cycle_df=cycle_df, dod_override="Auto (infer)", need_logs=True)

    first_year = output.results[0]
    assert first_year.expected_firm_mwh == pytest.approx(2.0)
