from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.simulation_core import SimConfig, Window, simulate_project  # noqa: E402


def _flat_cycle_table() -> pd.DataFrame:
    """Return a cycle table that keeps state-of-health constant for tests."""

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


def test_simulation_outputs_when_pv_covers_contract() -> None:
    """PV-only delivery should meet the target with zero shortfall."""

    cfg = SimConfig(
        years=1,
        step_hours=1.0,
        contracted_mw=5.0,
        initial_power_mw=0.0,
        initial_usable_mwh=0.0,
        discharge_windows=[Window(0, 1)],
        charge_windows=[],
        pv_availability=1.0,
        bess_availability=1.0,
    )
    pv_df = pd.DataFrame({"pv_mw": [5.0, 0.0]})
    cycle_df = _flat_cycle_table()

    output = simulate_project(cfg, pv_df=pv_df, cycle_df=cycle_df, dod_override="Auto (infer)", need_logs=True)

    year = output.results[0]
    assert year.expected_firm_mwh == pytest.approx(5.0)
    assert year.delivered_firm_mwh == pytest.approx(5.0)
    assert year.pv_to_contract_mwh == pytest.approx(5.0)
    assert year.bess_to_contract_mwh == pytest.approx(0.0)
    assert year.shortfall_mwh == pytest.approx(0.0)
    assert year.charge_mwh == pytest.approx(0.0)
    assert year.pv_curtailed_mwh == pytest.approx(0.0)

    logs = output.final_year_logs
    assert logs is not None
    np.testing.assert_allclose(
        logs.delivered_mw,
        logs.pv_to_contract_mw + logs.bess_to_contract_mw,
    )


def test_simulation_outputs_when_no_resources() -> None:
    """No PV and no storage should yield full shortfall."""

    cfg = SimConfig(
        years=1,
        step_hours=1.0,
        contracted_mw=5.0,
        initial_power_mw=0.0,
        initial_usable_mwh=0.0,
        discharge_windows=[Window(0, 1)],
        charge_windows=[],
        pv_availability=1.0,
        bess_availability=1.0,
    )
    pv_df = pd.DataFrame({"pv_mw": [0.0, 0.0]})
    cycle_df = _flat_cycle_table()

    output = simulate_project(cfg, pv_df=pv_df, cycle_df=cycle_df, dod_override="Auto (infer)", need_logs=True)

    year = output.results[0]
    assert year.expected_firm_mwh == pytest.approx(5.0)
    assert year.delivered_firm_mwh == pytest.approx(0.0)
    assert year.shortfall_mwh == pytest.approx(5.0)
    logs = output.final_year_logs
    assert logs is not None
    assert logs.shortfall_mw.max() == pytest.approx(5.0)
