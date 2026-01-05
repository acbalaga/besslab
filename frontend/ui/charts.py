"""Chart and data prep helpers for Streamlit visualizations."""

from dataclasses import dataclass
from typing import Dict, Optional

import altair as alt
import numpy as np
import pandas as pd

from services.simulation_core import HourlyLog, SimConfig


def prepare_soc_heatmap_data(logs: HourlyLog, initial_energy_mwh: float) -> pd.DataFrame:
    """Return a day-of-year × hour-of-day SOC fraction pivot for heatmap rendering."""
    hours_total = len(logs.hod)
    if hours_total == 0:
        return pd.DataFrame(
            index=pd.Index([], name="day_of_year"),
            columns=pd.Index(range(24), name="hour"),
        )

    hours_per_day = 24
    day_count = int(np.ceil(hours_total / hours_per_day))
    df = pd.DataFrame(
        {
            "hour_index": np.arange(hours_total, dtype=int),
            "hour": logs.hod.astype(int),
            "soc_frac": logs.soc_mwh / max(initial_energy_mwh, 1e-9),
        }
    )
    df["day_of_year"] = (df["hour_index"] // hours_per_day) + 1

    pivot = (
        df.pivot_table(index="day_of_year", columns="hour", values="soc_frac", aggfunc="mean")
        .reindex(index=pd.RangeIndex(1, day_count + 1, name="day_of_year"))
        .reindex(columns=pd.RangeIndex(0, 24, name="hour"), fill_value=np.nan)
    )
    return pivot


def prepare_charge_discharge_envelope(logs: HourlyLog) -> pd.DataFrame:
    """Aggregate charge/discharge distributions by hour for envelope plotting."""
    df = pd.DataFrame(
        {
            "hour": logs.hod.astype(int),
            "charge_mw": logs.charge_mw,
            "discharge_mw": logs.discharge_mw,
        }
    )
    grouped = df.groupby("hour").agg(
        charge_p05=("charge_mw", lambda s: s.quantile(0.05)),
        charge_p50=("charge_mw", "median"),
        charge_p95=("charge_mw", lambda s: s.quantile(0.95)),
        discharge_p05=("discharge_mw", lambda s: s.quantile(0.05)),
        discharge_p50=("discharge_mw", "median"),
        discharge_p95=("discharge_mw", lambda s: s.quantile(0.95)),
    )
    envelope = grouped.reindex(pd.RangeIndex(0, 24, name="hour")).fillna(0.0).reset_index()
    envelope["charge_low"] = -envelope["charge_p95"]
    envelope["charge_high"] = -envelope["charge_p05"]
    envelope["charge_median_neg"] = -envelope["charge_p50"]
    envelope["discharge_low"] = envelope["discharge_p05"]
    envelope["discharge_high"] = envelope["discharge_p95"]
    return envelope


def build_avg_profile_df(logs: HourlyLog, cfg: SimConfig) -> pd.DataFrame:
    """Compute the average daily power profile for a given simulation log."""
    contracted_series = np.array(
        [
            cfg.contracted_mw if any(w.contains(int(h)) for w in cfg.discharge_windows) else 0.0
            for h in logs.hod
        ],
        dtype=float,
    )
    df_hr = pd.DataFrame(
        {
            "hod": logs.hod.astype(int),
            "pv_resource_mw": logs.pv_mw,
            "pv_to_contract_mw": logs.pv_to_contract_mw,
            "bess_to_contract_mw": logs.bess_to_contract_mw,
            "charge_mw": logs.charge_mw,
            "contracted_mw": contracted_series,
        }
    )
    avg = df_hr.groupby("hod", as_index=False).mean().rename(columns={"hod": "hour"})
    avg["pv_surplus_mw"] = np.maximum(avg["pv_resource_mw"] - avg["pv_to_contract_mw"] - avg["charge_mw"], 0.0)
    avg["charge_mw_neg"] = -avg["charge_mw"]
    return avg[
        [
            "hour",
            "pv_resource_mw",
            "pv_to_contract_mw",
            "bess_to_contract_mw",
            "pv_surplus_mw",
            "charge_mw_neg",
            "contracted_mw",
        ]
    ]


@dataclass
class AvgProfileBundle:
    first_year: Optional[pd.DataFrame]
    final_year: Optional[pd.DataFrame]
    project: Optional[pd.DataFrame]


def build_avg_profile_bundle(
    cfg: SimConfig,
    first_year_logs: Optional[HourlyLog],
    final_year_logs: Optional[HourlyLog],
    hod_count: np.ndarray,
    hod_sum_pv_resource: np.ndarray,
    hod_sum_pv: np.ndarray,
    hod_sum_bess: np.ndarray,
    hod_sum_charge: np.ndarray,
) -> AvgProfileBundle:
    """Prepare average daily profile DataFrames for first, final, and full-project views."""
    contracted_by_hour = np.array(
        [cfg.contracted_mw if any(w.contains(h) for w in cfg.discharge_windows) else 0.0 for h in range(24)],
        dtype=float,
    )
    project_df: Optional[pd.DataFrame] = None
    with np.errstate(invalid="ignore", divide="ignore"):
        avg_charge = np.divide(hod_sum_charge, hod_count, out=np.zeros_like(hod_sum_charge), where=hod_count > 0)
        project_df = pd.DataFrame(
            {
                "hour": np.arange(24),
                "pv_resource_mw": np.divide(
                    hod_sum_pv_resource, hod_count, out=np.zeros_like(hod_sum_pv_resource), where=hod_count > 0
                ),
                "pv_to_contract_mw": np.divide(hod_sum_pv, hod_count, out=np.zeros_like(hod_sum_pv), where=hod_count > 0),
                "bess_to_contract_mw": np.divide(
                    hod_sum_bess, hod_count, out=np.zeros_like(hod_sum_bess), where=hod_count > 0
                ),
                "charge_mw": avg_charge,
                "charge_mw_neg": -avg_charge,
                "contracted_mw": contracted_by_hour,
            }
        )
        project_df["pv_surplus_mw"] = np.maximum(
            project_df["pv_resource_mw"] - project_df["pv_to_contract_mw"] - project_df["charge_mw"], 0.0
        )

    return AvgProfileBundle(
        first_year=build_avg_profile_df(first_year_logs, cfg) if first_year_logs is not None else None,
        final_year=build_avg_profile_df(final_year_logs, cfg) if final_year_logs is not None else None,
        project=project_df,
    )


def build_avg_profile_chart(avg_df: pd.DataFrame) -> alt.LayerChart:
    """Return the layered Altair chart for the average daily profile."""
    axis_x = alt.Axis(values=list(range(0, 24, 2)))
    x_hour = alt.X("hour:O", title="Hour of Day", axis=axis_x)
    base = alt.Chart(avg_df).encode(x=x_hour)

    contrib_long = avg_df.melt(
        id_vars=["hour"],
        value_vars=["pv_to_contract_mw", "bess_to_contract_mw"],
        var_name="Source",
        value_name="MW",
    )
    contrib_long["Source"] = contrib_long["Source"].replace(
        {"pv_to_contract_mw": "PV→Contract", "bess_to_contract_mw": "BESS→Contract"}
    )
    contrib_long["SourceOrder"] = contrib_long["Source"].map({"PV→Contract": 0, "BESS→Contract": 1})
    contrib_fill = (
        alt.Chart(contrib_long)
        .mark_bar(opacity=0.28, size=16)
        .encode(
            x=x_hour,
            y=alt.Y("MW:Q", stack="zero"),
            color=alt.Color(
                "Source:N",
                scale=alt.Scale(domain=["PV→Contract", "BESS→Contract"], range=["#86c5da", "#7fd18b"]),
                legend=None,
            ),
            order=alt.Order("SourceOrder:Q", sort="ascending"),
        )
    )
    contrib_chart = (
        alt.Chart(contrib_long)
        .mark_bar(opacity=0.9, size=16)
        .encode(
            x=x_hour,
            y=alt.Y("MW:Q", title="MW", stack="zero"),
            color=alt.Color(
                "Source:N",
                scale=alt.Scale(domain=["PV→Contract", "BESS→Contract"], range=["#86c5da", "#7fd18b"]),
            ),
            order=alt.Order("SourceOrder:Q", sort="ascending"),
        )
    )

    pv_resource_area = (
        base.mark_area(opacity=0.18, color="#f2d7a0", line=alt.LineConfig(color="#c78100", strokeDash=[6, 3], strokeWidth=2))
        .encode(
            x=x_hour,
            y=alt.Y("pv_resource_mw:Q", title="MW"),
            tooltip=[alt.Tooltip("pv_resource_mw:Q", title="PV resource (MW)", format=".2f")],
        )
    )

    pv_surplus_area = (
        base.mark_area(color="#f7c5c5", opacity=0.45)
        .encode(
            x=x_hour,
            y=alt.Y("pv_surplus_mw:Q", title="MW"),
            tooltip=[alt.Tooltip("pv_surplus_mw:Q", title="PV surplus (MW)", format=".2f")],
        )
    )

    area_chg = base.mark_area(opacity=0.5).encode(y="charge_mw_neg:Q", color=alt.value("#caa6ff"))

    contract_steps = avg_df[["hour", "contracted_mw"]].copy()
    contract_outline = pd.concat(
        [contract_steps, pd.DataFrame({"hour": [contract_steps["hour"].max() + 1], "contracted_mw": [0.0]})],
        ignore_index=True,
    )
    contract_box = (
        alt.Chart(contract_steps)
        .mark_bar(color="#f2a900", opacity=0.1, size=26)
        .encode(x=x_hour, y=alt.Y("contracted_mw:Q", title="MW"), y2=alt.value(0))
    )
    line_contract = alt.Chart(contract_outline).mark_line(color="#f2a900", strokeWidth=2, interpolate="step-after").encode(
        x=x_hour, y="contracted_mw:Q"
    )

    return alt.layer(contract_box, line_contract, pv_resource_area, pv_surplus_area, area_chg, contrib_fill, contrib_chart).properties(
        height=360
    )

