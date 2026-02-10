import pandas as pd

from utils.bess_advisory import (
    build_sizing_benchmark_table,
    choose_recommended_candidate,
    summarize_pv_sizing_signals,
)


def test_summarize_pv_sizing_signals_returns_expected_fields():
    pv_df = pd.DataFrame({"pv_mw": [0.0, 1.0, 2.0, 1.0, 0.0]})

    summary = summarize_pv_sizing_signals(pv_df)

    assert summary.annual_energy_mwh == 4.0
    assert summary.peak_power_mw == 2.0
    assert summary.implied_capacity_factor_pct == 40.0
    assert summary.active_hours_pct == 60.0


def test_choose_recommended_candidate_prefers_reliability_and_lower_energy():
    sweep_df = pd.DataFrame(
        {
            "power_mw": [20.0, 20.0, 20.0],
            "energy_mwh": [40.0, 60.0, 80.0],
            "compliance_pct": [98.5, 99.3, 99.3],
            "total_shortfall_mwh": [150.0, 30.0, 35.0],
            "status": ["evaluated", "evaluated", "evaluated"],
        }
    )

    recommendation = choose_recommended_candidate(sweep_df)

    assert recommendation is not None
    assert recommendation["energy_mwh"] == 60.0


def test_build_sizing_benchmark_table_adds_duration_and_c_rate():
    sweep_df = pd.DataFrame(
        {
            "power_mw": [10.0],
            "energy_mwh": [40.0],
            "compliance_pct": [99.2],
            "total_shortfall_mwh": [12.0],
        }
    )

    benchmark_df = build_sizing_benchmark_table(sweep_df)

    assert benchmark_df.loc[0, "duration_h"] == 4.0
    assert benchmark_df.loc[0, "c_rate_per_h"] == 0.25
    assert bool(benchmark_df.loc[0, "benchmark_reliability_ok"])
