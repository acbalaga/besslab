from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from app import BASE_DIR
from services.simulation_core import summarize_simulation
from utils.ui_layout import init_page_layout
from utils.ui_state import bootstrap_session_state, get_simulation_results, get_simulation_snapshot


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    label: str
    description: str


def _metric_definitions() -> List[MetricDefinition]:
    return [
        MetricDefinition(
            key="compliance_pct",
            label="% Compliance",
            description="Total firm energy delivered vs contracted across the project life.",
        ),
        MetricDefinition(
            key="deficit_pct",
            label="% Deficit",
            description="Shortfall energy as a share of total expected firm energy.",
        ),
        MetricDefinition(
            key="surplus_pct",
            label="% Surplus",
            description="PV curtailment as a share of total PV generation.",
        ),
        MetricDefinition(
            key="soh_pct",
            label="% SOH at project year-end",
            description="Final-year total state-of-health (cycle × calendar) in percent.",
        ),
        MetricDefinition(
            key="pirr_pct",
            label="% PIRR",
            description="Project IRR from the finance model (enter manually if unavailable).",
        ),
    ]


def _default_lever_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Lever": "RTE ± / temperature derate",
                "Low impact (pp)": 0.0,
                "High impact (pp)": 0.0,
                "Notes": "Roundtrip efficiency or temperature-driven derate.",
            },
            {
                "Lever": "Degradation severity (high/low)",
                "Low impact (pp)": 0.0,
                "High impact (pp)": 0.0,
                "Notes": "Higher/lower fade assumptions for cycle + calendar aging.",
            },
            {
                "Lever": "Price volatility/spreads",
                "Low impact (pp)": 0.0,
                "High impact (pp)": 0.0,
                "Notes": "Stress price spreads that change revenue/penalty exposure.",
            },
            {
                "Lever": "Penalty rates/compliance threshold",
                "Low impact (pp)": 0.0,
                "High impact (pp)": 0.0,
                "Notes": "Vary penalties or target compliance requirements.",
            },
            {
                "Lever": "POI limit",
                "Low impact (pp)": 0.0,
                "High impact (pp)": 0.0,
                "Notes": "Grid interconnect cap limiting dispatch or charge.",
            },
            {
                "Lever": "Dispatch window changes",
                "Low impact (pp)": 0.0,
                "High impact (pp)": 0.0,
                "Notes": "Shift discharge/charge windows to test schedule risk.",
            },
            {
                "Lever": "Availability (e.g., 97% vs 93%)",
                "Low impact (pp)": 0.0,
                "High impact (pp)": 0.0,
                "Notes": "Availability impacts on achievable delivery and energy.",
            },
            {
                "Lever": "BESS Size Capacity (MW)",
                "Low impact (pp)": 0.0,
                "High impact (pp)": 0.0,
                "Notes": "Power rating in MW; keep units explicit in assumptions.",
            },
            {
                "Lever": "BESS Size Energy Capacity (MWh)",
                "Low impact (pp)": 0.0,
                "High impact (pp)": 0.0,
                "Notes": "Usable energy in MWh; align with BOL sizing inputs.",
            },
        ]
    )


def _baseline_from_simulation() -> Dict[str, Optional[float]]:
    baseline: Dict[str, Optional[float]] = {
        "compliance_pct": None,
        "deficit_pct": None,
        "surplus_pct": None,
        "soh_pct": None,
        "pirr_pct": None,
    }
    cached = get_simulation_results()
    if cached is None:
        snapshot = get_simulation_snapshot() or {}
        compliance_value = snapshot.get("Compliance (%)")
        soh_value = snapshot.get("Final SOH_total")
        baseline.update(
            {
                "compliance_pct": float(compliance_value) if compliance_value is not None else None,
                "soh_pct": float(soh_value) * 100.0 if soh_value is not None else None,
            }
        )
        return baseline

    summary = summarize_simulation(cached.sim_output)
    expected_total = sum(r.expected_firm_mwh for r in cached.sim_output.results)
    pv_total_generation = sum(r.available_pv_mwh for r in cached.sim_output.results)
    deficit_pct = (summary.total_shortfall_mwh / expected_total * 100.0) if expected_total > 0 else None
    surplus_pct = (
        summary.pv_excess_mwh / pv_total_generation * 100.0
        if pv_total_generation > 0
        else None
    )
    final_soh_pct = cached.sim_output.results[-1].soh_total * 100.0
    baseline.update(
        {
            "compliance_pct": summary.compliance,
            "deficit_pct": deficit_pct,
            "surplus_pct": surplus_pct,
            "soh_pct": final_soh_pct,
        }
    )
    return baseline


def _baseline_inputs(
    metrics: List[MetricDefinition],
    baseline_values: Dict[str, Optional[float]],
) -> Dict[str, Optional[float]]:
    st.markdown("#### Baseline values")
    st.caption(
        "Baseline values seed the tornado chart. Values are in percent; override any missing metric "
        "(especially PIRR) before exporting the sensitivity table."
    )
    baseline_inputs: Dict[str, Optional[float]] = {}
    cols = st.columns(2)
    for idx, metric in enumerate(metrics):
        default_value = baseline_values.get(metric.key)
        input_value = cols[idx % 2].number_input(
            metric.label,
            value=float(default_value) if default_value is not None else 0.0,
            step=0.1,
            help=metric.description,
        )
        baseline_inputs[metric.key] = input_value
    return baseline_inputs


def _get_metric_state_key(metric_key: str) -> str:
    return f"sensitivity_impacts_{metric_key}"


def _load_impact_table(metric_key: str, default_df: pd.DataFrame) -> pd.DataFrame:
    state_key = _get_metric_state_key(metric_key)
    if state_key not in st.session_state:
        st.session_state[state_key] = default_df.copy()
    table = st.session_state[state_key]
    if not isinstance(table, pd.DataFrame):
        st.session_state[state_key] = default_df.copy()
    return st.session_state[state_key]


def _save_impact_table(metric_key: str, table: pd.DataFrame) -> None:
    st.session_state[_get_metric_state_key(metric_key)] = table


def _prepare_tornado_data(table: pd.DataFrame) -> pd.DataFrame:
    numeric_table = table.copy()
    numeric_table["Low impact (pp)"] = pd.to_numeric(
        numeric_table["Low impact (pp)"], errors="coerce"
    ).fillna(0.0)
    numeric_table["High impact (pp)"] = pd.to_numeric(
        numeric_table["High impact (pp)"], errors="coerce"
    ).fillna(0.0)
    numeric_table["sort_key"] = numeric_table[["Low impact (pp)", "High impact (pp)"]].abs().max(axis=1)
    melted = numeric_table.melt(
        id_vars=["Lever", "Notes", "sort_key"],
        value_vars=["Low impact (pp)", "High impact (pp)"],
        var_name="Scenario",
        value_name="Impact (pp)",
    )
    return melted.sort_values("sort_key", ascending=True)


def _build_tornado_chart(source: pd.DataFrame) -> alt.Chart:
    if source.empty:
        return alt.Chart(pd.DataFrame({"Impact (pp)": [], "Lever": []})).mark_bar()

    impact_domain = (
        float(source["Impact (pp)"].min()),
        float(source["Impact (pp)"].max()),
    )
    extent = max(abs(impact_domain[0]), abs(impact_domain[1]), 1.0)
    zero_line = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color="#6b6b6b").encode(x="zero:Q")

    bars = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            x=alt.X(
                "Impact (pp):Q",
                title="Impact (percentage points)",
                scale=alt.Scale(domain=[-extent, extent]),
            ),
            y=alt.Y("Lever:N", sort=None, title="Sensitivity lever"),
            color=alt.Color(
                "Scenario:N",
                scale=alt.Scale(range=["#d95f02", "#1b9e77"]),
                title="Scenario",
            ),
            tooltip=[
                alt.Tooltip("Lever:N"),
                alt.Tooltip("Scenario:N"),
                alt.Tooltip("Impact (pp):Q", format=".2f"),
                alt.Tooltip("Notes:N"),
            ],
        )
    )
    return (bars + zero_line).properties(height=420)


render_layout = init_page_layout(
    page_title="Sensitivity & Stress Test",
    main_title="Sensitivity & Stress Test",
    description=(
        "Capture how key levers move compliance, deficit/surplus, SOH, and PIRR. "
        "Populate the impact ranges below to generate a tornado view."
    ),
    base_dir=BASE_DIR,
)

bootstrap_session_state()
render_layout()

metrics = _metric_definitions()
baseline_values = _baseline_from_simulation()

st.markdown("### Sensitivity levers")
st.caption(
    "Enter low/high impacts in percentage points relative to the baseline. "
    "Positive values improve the metric; negative values reduce it."
)

baseline_inputs = _baseline_inputs(metrics, baseline_values)

metric_lookup = {metric.label: metric for metric in metrics}
selected_metric_label = st.selectbox("Select metric", list(metric_lookup.keys()))
selected_metric = metric_lookup[selected_metric_label]

impact_table = _load_impact_table(selected_metric.key, _default_lever_table())

with st.form("sensitivity_table_form", clear_on_submit=False):
    edited_table = st.data_editor(
        impact_table,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Lever": st.column_config.TextColumn("Lever", disabled=True),
            "Low impact (pp)": st.column_config.NumberColumn(
                "Low impact (pp)",
                help="Negative for downside impact; enter in percentage points.",
                step=0.1,
                format="%.2f",
            ),
            "High impact (pp)": st.column_config.NumberColumn(
                "High impact (pp)",
                help="Positive for upside impact; enter in percentage points.",
                step=0.1,
                format="%.2f",
            ),
            "Notes": st.column_config.TextColumn("Notes", disabled=True),
        },
    )
    saved = st.form_submit_button("Save sensitivity inputs", use_container_width=True)

if saved:
    _save_impact_table(selected_metric.key, edited_table)
    st.success("Saved sensitivity inputs for this metric.")
else:
    _save_impact_table(selected_metric.key, edited_table)

st.markdown("### Tornado chart")
baseline_value = baseline_inputs.get(selected_metric.key)
if baseline_value is not None:
    st.metric("Baseline", f"{baseline_value:,.2f}%")

chart_source = _prepare_tornado_data(edited_table)
if chart_source["Impact (pp)"].abs().sum() == 0:
    st.info(
        "All impacts are currently zero. Enter low/high values above to populate the tornado chart.",
        icon="ℹ️",
    )

st.altair_chart(_build_tornado_chart(chart_source), use_container_width=True)

st.markdown("### Notes & export")
st.caption(
    "Example usage: if an RTE downside is expected to reduce compliance by 1.5pp, "
    "enter -1.5 in the Low impact column for that lever."
)

export_df = edited_table.copy()
export_df["Metric"] = selected_metric.label
export_df["Baseline (%)"] = baseline_value
export_csv = export_df.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download sensitivity table (CSV)",
    data=export_csv,
    file_name="sensitivity_inputs.csv",
    mime="text/csv",
    use_container_width=True,
)

st.caption(
    "Surplus % uses PV curtailment ÷ total PV generation. Deficit % uses total shortfall ÷ total expected firm energy."
)
