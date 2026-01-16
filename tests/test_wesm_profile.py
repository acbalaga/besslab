from __future__ import annotations

import io

import pandas as pd

from utils.io import read_wesm_forecast_profile_average, read_wesm_profile


def _build_month_hour_csv(price_php_per_mwh: float = 5000.0) -> io.StringIO:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    rows = ['\ufeffmonth,hr,"Central, PHP/MWh"']
    for month in months:
        for hour in range(1, 25):
            rows.append(f"{month},{hour},{price_php_per_mwh}")
    return io.StringIO("\n".join(rows))


def test_read_wesm_profile_expands_month_hour_profile() -> None:
    buffer = _build_month_hour_csv(price_php_per_mwh=5000.0)
    df = read_wesm_profile([buffer], forex_rate_php_per_usd=50.0)

    assert len(df) == 8760
    assert {"timestamp", "hour_index", "wesm_deficit_price_usd_per_mwh"}.issubset(df.columns)
    assert df["wesm_deficit_price_usd_per_mwh"].iloc[0] == 100.0


def test_read_wesm_profile_maps_legacy_php_per_mwh_columns() -> None:
    csv_text = "\n".join(
        [
            'hour_index,"Central, PHP/MWh"',
            "0,4200",
            "1,4500",
        ]
    )
    df = read_wesm_profile([io.StringIO(csv_text)], forex_rate_php_per_usd=60.0)

    assert list(df["hour_index"]) == [0, 1]
    assert pd.Series([70.0, 75.0]).tolist() == df["wesm_deficit_price_usd_per_mwh"].tolist()


def test_read_wesm_forecast_profile_average() -> None:
    csv_text = "\n".join(
        [
            'hour_index,"Central, PHP/MWh"',
            "0,4",
            "1,8",
            "2,12",
            "3,16",
            "4,6",
            "5,10",
            "6,14",
            "7,18",
        ]
    )
    df = read_wesm_forecast_profile_average(
        [io.StringIO(csv_text)],
        forex_rate_php_per_usd=1.0,
        hours_in_year=4,
    )

    assert list(df["hour_index"]) == [0, 1, 2, 3]
    assert df["wesm_deficit_price_usd_per_mwh"].tolist() == [5.0, 9.0, 13.0, 17.0]
    assert df["wesm_surplus_price_usd_per_mwh"].tolist() == [5.0, 9.0, 13.0, 17.0]
