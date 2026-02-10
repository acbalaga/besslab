import pandas as pd

from utils.bess_advisory import (
    build_sizing_benchmark_table,
    choose_recommended_candidate,
    rank_recommendation_candidates,
    summarize_pv_sizing_signals,
)
from utils.sweeps import _resolve_ranking_column


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


def test_recommendation_changes_when_ranking_kpi_changes():
    sweep_df = pd.DataFrame(
        {
            "power_mw": [20.0, 20.0, 20.0],
            "energy_mwh": [40.0, 80.0, 100.0],
            "compliance_pct": [99.1, 99.4, 99.2],
            "total_shortfall_mwh": [80.0, 20.0, 40.0],
            "npv_usd": [850_000.0, 900_000.0, 1_100_000.0],
            "status": ["evaluated", "evaluated", "evaluated"],
        }
    )

    compliance_column, compliance_ascending, _, _ = _resolve_ranking_column("reliability", "compliance_pct")
    npv_column, npv_ascending, _, _ = _resolve_ranking_column("reliability", "npv_usd")

    compliance_recommendation = choose_recommended_candidate(
        sweep_df,
        ranking_column=compliance_column,
        ascending=compliance_ascending,
    )
    npv_recommendation = choose_recommended_candidate(
        sweep_df,
        ranking_column=npv_column,
        ascending=npv_ascending,
    )

    assert compliance_recommendation is not None
    assert npv_recommendation is not None
    assert compliance_recommendation["energy_mwh"] == 80.0
    assert npv_recommendation["energy_mwh"] == 100.0


def test_top_candidates_table_top_row_matches_recommendation():
    sweep_df = pd.DataFrame(
        {
            "power_mw": [20.0, 20.0, 20.0],
            "energy_mwh": [40.0, 80.0, 100.0],
            "compliance_pct": [99.1, 99.4, 99.2],
            "total_shortfall_mwh": [80.0, 20.0, 40.0],
            "npv_usd": [850_000.0, 900_000.0, 1_100_000.0],
            "status": ["evaluated", "evaluated", "evaluated"],
        }
    )

    ranking_column, ascending, _, _ = _resolve_ranking_column("reliability", "npv_usd")
    recommendation = choose_recommended_candidate(
        sweep_df,
        ranking_column=ranking_column,
        ascending=ascending,
    )
    ranked_table = rank_recommendation_candidates(
        sweep_df,
        ranking_column=ranking_column,
        ascending=ascending,
    )

    assert recommendation is not None
    assert not ranked_table.empty
    assert ranked_table.iloc[0]["power_mw"] == recommendation["power_mw"]
    assert ranked_table.iloc[0]["energy_mwh"] == recommendation["energy_mwh"]


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
