from frontend.ui.forms import _normalize_hourly_schedule_payload


def test_normalize_hourly_schedule_payload_handles_column_dicts() -> None:
    payload = {"Hour": list(range(24)), "Capacity (MW)": [1.0] * 24}

    df = _normalize_hourly_schedule_payload(payload)

    assert list(df.columns) == ["Hour", "Capacity (MW)"]
    assert len(df) == 24
    assert df["Capacity (MW)"].sum() == 24.0


def test_normalize_hourly_schedule_payload_handles_row_dicts() -> None:
    payload = {
        0: {"Hour": 0, "Capacity (MW)": 2.0},
        1: {"Hour": 1, "Capacity (MW)": 3.0},
    }

    df = _normalize_hourly_schedule_payload(payload)

    assert len(df) == 24
    assert df.loc[df["Hour"] == 0, "Capacity (MW)"].iloc[0] == 2.0
    assert df.loc[df["Hour"] == 2, "Capacity (MW)"].iloc[0] == 0.0
