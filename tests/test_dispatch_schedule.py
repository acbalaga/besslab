import numpy as np

from services.simulation_core import SimConfig, Window
from utils.dispatch_schedule import (
    build_contracted_mw_profile,
    build_hourly_schedule_from_period_table,
    normalize_hourly_schedule,
)


def test_build_contracted_mw_profile_falls_back_to_windows() -> None:
    cfg = SimConfig(
        contracted_mw=4.0,
        discharge_windows=[Window(6, 8)],
        charge_windows=[],
    )

    schedule = build_contracted_mw_profile(cfg)

    assert schedule[6] == 4.0
    assert schedule[7] == 4.0
    assert schedule[5] == 0.0
    assert schedule[8] == 0.0


def test_build_contracted_mw_profile_prefers_hourly_schedule() -> None:
    cfg = SimConfig(
        contracted_mw=4.0,
        contracted_mw_schedule=[1.0] * 24,
        discharge_windows=[Window(6, 8)],
        charge_windows=[],
    )

    schedule = build_contracted_mw_profile(cfg)

    assert schedule == [1.0] * 24


def test_period_table_to_hourly_schedule_wraps_midnight() -> None:
    period_table = [
        {"start_time": "22:00", "end_time": "02:00", "capacity_mw": 3.0},
    ]
    schedule = build_hourly_schedule_from_period_table(period_table)

    assert schedule is not None
    assert schedule[22] == 3.0
    assert schedule[23] == 3.0
    assert schedule[0] == 3.0
    assert schedule[1] == 3.0
    assert schedule[2] == 0.0


def test_normalize_hourly_schedule_rejects_invalid_length() -> None:
    assert normalize_hourly_schedule([1.0, 2.0]) is None
    assert normalize_hourly_schedule(None) is None
    assert np.isclose(sum(normalize_hourly_schedule([0.5] * 24) or []), 12.0)
