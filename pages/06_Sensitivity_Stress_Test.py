from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from services.simulation_core import HourlyLog, SimConfig, Window, simulate_project, summarize_simulation
from utils import read_wesm_profile
from utils.economics import (
    EconomicInputs,
    PriceInputs,
    aggregate_wesm_profile_to_annual,
    compute_financing_cash_flows,
    normalize_economic_inputs,
)
from utils.ui_layout import init_page_layout
from utils.ui_state import (
    bootstrap_session_state,
    get_cached_simulation_config,
    get_base_dir,
    get_latest_economics_payload,
    get_simulation_results,
    get_simulation_snapshot,
)


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    label: str
    description: str


@dataclass(frozen=True)
class SensitivityLever:
    label: str
    delta_unit: str
    apply_delta: Callable[[SimConfig, float], Optional[SimConfig]]


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
                "Low change": -2.0,
                "High change": 2.0,
                "Low impact (pp)": 0.0,
                "High impact (pp)": 0.0,
                "Notes": "Roundtrip efficiency or temperature-driven derate.",
            },
            {
                "Lever": "Degradation severity (high/low)",
                "Low change": -0.2,
                "High change": 0.2,
                "Low impact (pp)": 0.0,
                "High impact (pp)": 0.0,
                "Notes": "Calendar fade severity delta (pp). Placeholder mapping; see notes.",
            },
            {
                "Lever": "Price volatility/spreads",
                "Low change": -5.0,
                "High change": 5.0,
                "Low impact (pp)": 0.0,
                "High impact (pp)": 0.0,
                "Notes": "Requires economics payload; not simulated for energy KPIs.",
            },
            {
                "Lever": "Penalty rates/compliance threshold",
                "Low change": -5.0,
                "High change": 5.0,
                "Low impact (pp)": 0.0,
                "High impact (pp)": 0.0,
                "Notes": "Requires economics/contract logic; not simulated for energy KPIs.",
            },
            {
                "Lever": "Dispatch window duration (hours)",
                "Low change": -1.0,
                "High change": 1.0,
                "Low impact (pp)": 0.0,
                "High impact (pp)": 0.0,
                "Notes": "Adjusts duration of discharge windows; start times remain fixed.",
            },
            {
                "Lever": "Availability (e.g., 97% vs 93%)",
                "Low change": -2.0,
                "High change": 2.0,
                "Low impact (pp)": 0.0,
                "High impact (pp)": 0.0,
                "Notes": "Availability impacts on achievable delivery and energy.",
            },
            {
                "Lever": "Power rating (MW)",
                "Low change": -5.0,
                "High change": 5.0,
                "Low impact (pp)": 0.0,
                "High impact (pp)": 0.0,
                "Notes": "Maps to initial_power_mw (covers POI or inverter limit assumptions).",
            },
            {
                "Lever": "BESS Size Energy Capacity (MWh)",
                "Low change": -10.0,
                "High change": 10.0,
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
    deficit_pct = (summary.total_shortfall_mwh / expected_total * 100.0) if expected_total > 0 else None
    surplus_pct = (summary.pv_excess_mwh / expected_total * 100.0) if expected_total > 0 else None
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
    default_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, Optional[float]]:
    st.markdown("#### Baseline values")
    st.caption(
        "Baseline values seed the tornado chart. Values are in percent; override any missing metric "
        "(especially PIRR) before exporting the sensitivity table."
    )
    baseline_inputs: Dict[str, Optional[float]] = {}
    default_overrides = default_overrides or {}
    cols = st.columns(2)
    for idx, metric in enumerate(metrics):
        default_value = default_overrides.get(metric.key, baseline_values.get(metric.key))
        value_text = "" if default_value is None else str(default_value)
        input_text = cols[idx % 2].text_input(
            metric.label,
            value=value_text,
            help=metric.description,
            placeholder="Missing",
            key=f"baseline_{metric.key}",
        )
        parsed_value, error_message = _parse_optional_float(input_text)
        if error_message:
            cols[idx % 2].caption(f":red[{error_message}]")
        elif parsed_value is None:
            cols[idx % 2].caption(":orange[Missing baseline]")
        baseline_inputs[metric.key] = parsed_value
    return baseline_inputs


def _baseline_inputs_ready_for_refresh(
    baseline_inputs: Dict[str, Optional[float]],
    metrics: List[MetricDefinition],
) -> bool:
    """Return True when existing inputs appear to be placeholders (all zero/None)."""

    for metric in metrics:
        value = baseline_inputs.get(metric.key)
        if value not in (None, 0.0):
            return False
    return True


def _baseline_defaults_from_simulation(
    metrics: List[MetricDefinition],
    baseline_values: Dict[str, Optional[float]],
    current_inputs: Optional[Dict[str, Optional[float]]],
) -> Dict[str, Optional[float]]:
    """Seed baseline defaults from simulation once real results are available."""

    current_inputs = current_inputs or {}
    has_baseline = any(baseline_values.get(metric.key) is not None for metric in metrics)
    if has_baseline and _baseline_inputs_ready_for_refresh(current_inputs, metrics):
        return {metric.key: baseline_values.get(metric.key) for metric in metrics}
    return current_inputs


def _parse_optional_float(raw_value: str) -> Tuple[Optional[float], Optional[str]]:
    cleaned = raw_value.strip()
    if not cleaned:
        return None, None
    try:
        return float(cleaned), None
    except ValueError:
        return None, "Enter a numeric value or leave blank."


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


def _normalize_baseline_inputs(
    payload: Dict[str, Any],
    metrics: List[MetricDefinition],
    defaults: Dict[str, Optional[float]],
) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for metric in metrics:
        raw_value = payload.get(metric.key, defaults.get(metric.key))
        try:
            normalized[metric.key] = float(raw_value) if raw_value is not None else 0.0
        except (TypeError, ValueError):
            normalized[metric.key] = float(defaults.get(metric.key) or 0.0)
    return normalized


def _normalize_impact_table(table: pd.DataFrame, default_df: pd.DataFrame) -> pd.DataFrame:
    """Align imported tables to the fixed lever schema used by the UI."""

    if "Lever" not in table.columns:
        return default_df.copy()
    merged = default_df[["Lever"]].merge(table, on="Lever", how="left")
    for column in default_df.columns:
        if column not in merged.columns:
            merged[column] = default_df[column]
    merged = merged[default_df.columns]
    for column in ["Low change", "High change", "Low impact (pp)", "High impact (pp)"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(default_df[column])
    merged["Notes"] = merged["Notes"].fillna(default_df["Notes"])
    return merged


def _collect_impact_tables(metrics: List[MetricDefinition]) -> Dict[str, List[Dict[str, Any]]]:
    tables: Dict[str, List[Dict[str, Any]]] = {}
    for metric in metrics:
        table = st.session_state.get(_get_metric_state_key(metric.key))
        if isinstance(table, pd.DataFrame):
            tables[metric.key] = table.to_dict(orient="records")
    return tables


def _prepare_tornado_data(table: pd.DataFrame) -> pd.DataFrame:
    scenario_label_map = {
        "Low impact (pp)": "Low impact",
        "High impact (pp)": "High impact",
    }
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
    melted["Scenario"] = melted["Scenario"].map(scenario_label_map).fillna(melted["Scenario"])
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


def _build_hourly_summary_df(logs: HourlyLog) -> pd.DataFrame:
    hour_index = range(len(logs.hod))
    data = {
        "hour_index": list(hour_index),
        "hod": logs.hod,
        "pv_mw": logs.pv_mw,
        "pv_to_contract_mw": logs.pv_to_contract_mw,
        "bess_to_contract_mw": logs.bess_to_contract_mw,
        "delivered_mw": logs.delivered_mw,
        "shortfall_mw": logs.shortfall_mw,
        "charge_mw": logs.charge_mw,
        "discharge_mw": logs.discharge_mw,
        "soc_mwh": logs.soc_mwh,
    }
    if logs.timestamp is not None:
        data["timestamp"] = pd.to_datetime(logs.timestamp)
    df = pd.DataFrame(data)
    df["pv_surplus_mw"] = pd.Series(
        (df["pv_mw"] - df["pv_to_contract_mw"] - df["charge_mw"]).clip(lower=0.0)
    )
    return df


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(min(value, maximum), minimum)


def _window_duration(window: Window) -> float:
    if window.end >= window.start:
        return window.end - window.start
    return (24.0 - window.start) + window.end


def _adjust_window_duration(windows: List[Window], delta_hours: float) -> List[Window]:
    adjusted: List[Window] = []
    for window in windows:
        duration = _window_duration(window)
        new_duration = _clamp(duration + delta_hours, 0.25, 24.0)
        new_end = (window.start + new_duration) % 24.0
        adjusted.append(Window(start=window.start, end=new_end))
    return adjusted


def _clone_config(cfg: SimConfig) -> SimConfig:
    return deepcopy(cfg)


def _apply_rte_delta(cfg: SimConfig, delta_pp: float) -> SimConfig:
    updated = _clone_config(cfg)
    updated.rte_roundtrip = _clamp(updated.rte_roundtrip + delta_pp / 100.0, 0.5, 0.99)
    return updated


def _apply_calendar_fade_delta(cfg: SimConfig, delta_pp: float) -> SimConfig:
    updated = _clone_config(cfg)
    updated.calendar_fade_rate = max(updated.calendar_fade_rate + delta_pp / 100.0, 0.0)
    return updated


def _apply_availability_delta(cfg: SimConfig, delta_pp: float) -> SimConfig:
    updated = _clone_config(cfg)
    updated.bess_availability = _clamp(updated.bess_availability + delta_pp / 100.0, 0.0, 1.0)
    return updated


def _apply_power_delta(cfg: SimConfig, delta_mw: float) -> SimConfig:
    updated = _clone_config(cfg)
    updated.initial_power_mw = max(updated.initial_power_mw + delta_mw, 0.1)
    return updated


def _apply_energy_delta(cfg: SimConfig, delta_mwh: float) -> SimConfig:
    updated = _clone_config(cfg)
    updated.initial_usable_mwh = max(updated.initial_usable_mwh + delta_mwh, 0.1)
    return updated


def _apply_window_delta(cfg: SimConfig, delta_hours: float) -> SimConfig:
    updated = _clone_config(cfg)
    updated.discharge_windows = _adjust_window_duration(updated.discharge_windows, delta_hours)
    return updated


def _unsupported_lever(_: SimConfig, __: float) -> Optional[SimConfig]:
    return None


def _sensitivity_levers() -> Dict[str, SensitivityLever]:
    return {
        "RTE ± / temperature derate": SensitivityLever(
            label="RTE ± / temperature derate",
            delta_unit="pp",
            apply_delta=_apply_rte_delta,
        ),
        "Degradation severity (high/low)": SensitivityLever(
            label="Degradation severity (high/low)",
            delta_unit="pp",
            apply_delta=_apply_calendar_fade_delta,
        ),
        "Price volatility/spreads": SensitivityLever(
            label="Price volatility/spreads",
            delta_unit="pp",
            apply_delta=_unsupported_lever,
        ),
        "Penalty rates/compliance threshold": SensitivityLever(
            label="Penalty rates/compliance threshold",
            delta_unit="pp",
            apply_delta=_unsupported_lever,
        ),
        "Dispatch window duration (hours)": SensitivityLever(
            label="Dispatch window duration (hours)",
            delta_unit="hours",
            apply_delta=_apply_window_delta,
        ),
        "Availability (e.g., 97% vs 93%)": SensitivityLever(
            label="Availability (e.g., 97% vs 93%)",
            delta_unit="pp",
            apply_delta=_apply_availability_delta,
        ),
        "Power rating (MW)": SensitivityLever(
            label="Power rating (MW)",
            delta_unit="MW",
            apply_delta=_apply_power_delta,
        ),
        "BESS Size Energy Capacity (MWh)": SensitivityLever(
            label="BESS Size Energy Capacity (MWh)",
            delta_unit="MWh",
            apply_delta=_apply_energy_delta,
        ),
    }


def _is_supported_lever(lever_label: str, levers: Dict[str, SensitivityLever]) -> bool:
    lever = levers.get(lever_label)
    return lever is not None and lever.apply_delta is not _unsupported_lever


def _split_lever_table(lever_table: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    levers = _sensitivity_levers()
    supported_mask = lever_table["Lever"].map(lambda label: _is_supported_lever(str(label), levers))
    supported = lever_table[supported_mask].reset_index(drop=True)
    unsupported = lever_table[~supported_mask].reset_index(drop=True)
    return supported, unsupported


def _merge_supported_edits(full_table: pd.DataFrame, supported_table: pd.DataFrame) -> pd.DataFrame:
    updated_table = full_table.set_index("Lever")
    supported_updates = supported_table.set_index("Lever")
    updated_table.update(supported_updates)
    return updated_table.reset_index()


def _compute_metric_value(metric_key: str, sim_output) -> Optional[float]:
    summary = summarize_simulation(sim_output)
    expected_total = sum(r.expected_firm_mwh for r in sim_output.results)
    if metric_key == "compliance_pct":
        return summary.compliance
    if metric_key == "deficit_pct":
        return (summary.total_shortfall_mwh / expected_total * 100.0) if expected_total > 0 else None
    if metric_key == "surplus_pct":
        return (summary.pv_excess_mwh / expected_total * 100.0) if expected_total > 0 else None
    if metric_key == "soh_pct":
        return sim_output.results[-1].soh_total * 100.0
    return None


def _compute_pirr(
    sim_output,
    econ_inputs: EconomicInputs,
    price_inputs: PriceInputs,
    wesm_profile_source: Optional[str],
) -> Optional[float]:
    if econ_inputs.capex_usd_per_kwh is not None and econ_inputs.bess_bol_kwh is None:
        econ_inputs = econ_inputs.__class__(
            **{
                **econ_inputs.__dict__,
                "bess_bol_kwh": sim_output.cfg.initial_usable_mwh * 1000.0,
            }
        )
    normalized_inputs = normalize_economic_inputs(econ_inputs)
    annual_delivered = [r.delivered_firm_mwh for r in sim_output.results]
    annual_bess = [r.bess_to_contract_mwh for r in sim_output.results]
    annual_pv_excess = [r.pv_curtailed_mwh for r in sim_output.results]
    annual_shortfall = [r.shortfall_mwh for r in sim_output.results]
    annual_total_generation = [r.available_pv_mwh for r in sim_output.results]
    annual_wesm_shortfall_cost_usd: Optional[List[float]] = None
    annual_wesm_surplus_revenue_usd: Optional[List[float]] = None

    if wesm_profile_source and sim_output.hourly_logs_by_year:
        wesm_profile_df = read_wesm_profile(
            [wesm_profile_source],
            forex_rate_php_per_usd=normalized_inputs.forex_rate_php_per_usd,
        )
        hourly_summary_by_year = {
            year_index: _build_hourly_summary_df(logs)
            for year_index, logs in sim_output.hourly_logs_by_year.items()
        }
        (
            annual_wesm_shortfall_cost_usd,
            annual_wesm_surplus_revenue_usd,
        ) = aggregate_wesm_profile_to_annual(
            hourly_summary_by_year,
            wesm_profile_df,
            step_hours=sim_output.cfg.step_hours,
            apply_inflation=False,
            inflation_rate=normalized_inputs.inflation_rate,
        )

    financing_outputs = compute_financing_cash_flows(
        annual_delivered,
        annual_bess,
        annual_pv_excess,
        normalized_inputs,
        price_inputs,
        annual_shortfall_mwh=annual_shortfall,
        annual_wesm_shortfall_cost_usd=annual_wesm_shortfall_cost_usd,
        annual_wesm_surplus_revenue_usd=annual_wesm_surplus_revenue_usd,
        augmentation_costs_usd=None,
        annual_total_generation_mwh=annual_total_generation,
    )
    return financing_outputs.project_irr_pct


def _run_sensitivity(
    base_cfg: SimConfig,
    pv_df: pd.DataFrame,
    cycle_df: pd.DataFrame,
    metric_key: str,
    lever_table: pd.DataFrame,
    baseline_sim_output,
    dod_override: str,
    econ_inputs: Optional[EconomicInputs],
    price_inputs: Optional[PriceInputs],
    wesm_profile_source: Optional[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Run low/high deltas for mapped levers and return the updated impact table."""
    updated_table = lever_table.copy()
    warnings: List[str] = []
    levers = _sensitivity_levers()
    supported_mask = updated_table["Lever"].map(lambda label: _is_supported_lever(str(label), levers))

    def _metric_from_output(sim_output) -> Optional[float]:
        if metric_key == "pirr_pct":
            if econ_inputs is None or price_inputs is None:
                return None
            return _compute_pirr(sim_output, econ_inputs, price_inputs, wesm_profile_source)
        return _compute_metric_value(metric_key, sim_output)

    if metric_key == "pirr_pct" and (econ_inputs is None or price_inputs is None):
        return updated_table, [
            "PIRR sensitivity needs economics inputs. Run a simulation with economics or load them first."
        ]

    baseline_value = _metric_from_output(baseline_sim_output)
    if baseline_value is None:
        warnings.append("Baseline metric could not be computed; impacts will remain blank.")

    for idx, row in updated_table[supported_mask].iterrows():
        lever_label = str(row.get("Lever"))
        lever = levers.get(lever_label)
        if lever is None:
            warnings.append(f"Unknown lever '{lever_label}' skipped.")
            continue

        low_delta = float(row.get("Low change", 0.0) or 0.0)
        high_delta = float(row.get("High change", 0.0) or 0.0)

        for column, delta in (("Low impact (pp)", low_delta), ("High impact (pp)", high_delta)):
            adjusted_cfg = lever.apply_delta(base_cfg, delta)
            if adjusted_cfg is None:
                continue
            need_logs = bool(
                metric_key == "pirr_pct"
                and price_inputs is not None
                and (price_inputs.apply_wesm_to_shortfall or price_inputs.sell_to_wesm)
                and wesm_profile_source
            )
            sim_output = simulate_project(
                adjusted_cfg,
                pv_df=pv_df,
                cycle_df=cycle_df,
                dod_override=dod_override,
                need_logs=need_logs,
            )
            metric_value = _metric_from_output(sim_output)
            if metric_value is None or baseline_value is None:
                updated_table.at[idx, column] = None
            else:
                updated_table.at[idx, column] = metric_value - baseline_value

    return updated_table, sorted(set(warnings))


render_layout = init_page_layout(
    page_title="Sensitivity & Stress Test",
    main_title="Sensitivity & Stress Test",
    description=(
        "Capture how key levers move compliance, deficit/surplus, SOH, and PIRR. "
        "Populate the change ranges below and run sensitivity to generate impacts."
    ),
    base_dir=get_base_dir(),
)

bootstrap_session_state()

pv_df, cycle_df = render_layout()

metrics = _metric_definitions()
baseline_values = _baseline_from_simulation()

with st.expander("Load/save sensitivity inputs (JSON)", expanded=False):
    st.caption(
        "Use JSON to persist baseline values and the lever table across sessions. "
        "Example keys: selected_metric_label, baseline_inputs, impact_tables."
    )
    uploaded_inputs = st.file_uploader(
        "Upload sensitivity inputs JSON",
        type=["json"],
        accept_multiple_files=False,
        key="sensitivity_inputs_upload",
    )
    pasted_inputs = st.text_area(
        "Or paste sensitivity inputs JSON",
        placeholder='{"baseline_inputs": {"compliance_pct": 98.5}}',
        height=120,
        key="sensitivity_inputs_paste",
    )
    if st.button("Apply JSON inputs", use_container_width=True):
        payload_text = ""
        if uploaded_inputs is not None:
            payload_text = uploaded_inputs.read().decode("utf-8")
        elif pasted_inputs.strip():
            payload_text = pasted_inputs
        if payload_text:
            try:
                payload = json.loads(payload_text)
            except json.JSONDecodeError as exc:
                st.error(f"Invalid JSON: {exc}")
            else:
                if isinstance(payload, dict):
                    baseline_defaults = {metric.key: baseline_values.get(metric.key) for metric in metrics}
                    baseline_inputs = _normalize_baseline_inputs(
                        payload.get("baseline_inputs", {}),
                        metrics,
                        baseline_defaults,
                    )
                    st.session_state["sensitivity_baseline_inputs"] = baseline_inputs
                    impact_tables = payload.get("impact_tables", {})
                    if isinstance(impact_tables, dict):
                        default_df = _default_lever_table()
                        for metric in metrics:
                            records = impact_tables.get(metric.key)
                            if isinstance(records, list):
                                table_df = pd.DataFrame(records)
                                st.session_state[_get_metric_state_key(metric.key)] = _normalize_impact_table(
                                    table_df,
                                    default_df,
                                )
                    selected_metric_label = payload.get("selected_metric_label")
                    if (
                        isinstance(selected_metric_label, str)
                        and selected_metric_label in [metric.label for metric in metrics]
                    ):
                        st.session_state["sensitivity_metric_select"] = selected_metric_label
                    st.success("Sensitivity inputs loaded. Re-rendering with the new values.")
                    st.rerun()
                else:
                    st.error("Expected a JSON object with sensitivity inputs.")
        else:
            st.info("Provide JSON content to load.", icon="ℹ️")

st.markdown("### Sensitivity levers")
st.caption(
    "Enter low/high changes for each lever, then click Run Sensitivity to compute impacts. "
    "Impacts are reported in percentage points relative to the baseline."
)
st.caption(
    "Units: pp for RTE, degradation, and availability; MW for POI/power; MWh for energy; hours for dispatch window duration."
)

baseline_inputs = _baseline_inputs(
    metrics,
    baseline_values,
    default_overrides=st.session_state.get("sensitivity_baseline_inputs"),
)
st.session_state["sensitivity_baseline_inputs"] = baseline_inputs

metric_lookup = {metric.label: metric for metric in metrics}
metric_labels = list(metric_lookup.keys())
selected_metric_label = st.selectbox(
    "Select metric",
    metric_labels,
    index=metric_labels.index(
        st.session_state.get("sensitivity_metric_select")
    )
    if st.session_state.get("sensitivity_metric_select") in metric_labels
    else 0,
    key="sensitivity_metric_select",
)
selected_metric = metric_lookup[selected_metric_label]

impact_table = _load_impact_table(selected_metric.key, _default_lever_table())
supported_table, unsupported_table = _split_lever_table(impact_table)

with st.form("sensitivity_table_form", clear_on_submit=False):
    if supported_table.empty:
        st.info("No simulated levers are available for sensitivity runs.", icon="ℹ️")
        edited_supported_table = supported_table
    else:
        edited_supported_table = st.data_editor(
            supported_table,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Lever": st.column_config.TextColumn("Lever", disabled=True),
                "Low change": st.column_config.NumberColumn(
                    "Low change",
                    help="Magnitude of the downside change (see units in notes).",
                    step=0.1,
                    format="%.2f",
                ),
                "High change": st.column_config.NumberColumn(
                    "High change",
                    help="Magnitude of the upside change (see units in notes).",
                    step=0.1,
                    format="%.2f",
                ),
                "Low impact (pp)": st.column_config.NumberColumn(
                    "Low impact (pp)",
                    help="Computed change in the selected metric after running sensitivity.",
                    step=0.1,
                    format="%.2f",
                ),
                "High impact (pp)": st.column_config.NumberColumn(
                    "High impact (pp)",
                    help="Computed change in the selected metric after running sensitivity.",
                    step=0.1,
                    format="%.2f",
                ),
                "Notes": st.column_config.TextColumn("Notes", disabled=True),
            },
        )
    run_sensitivity = st.form_submit_button("Run Sensitivity", use_container_width=True)

edited_table = _merge_supported_edits(impact_table, edited_supported_table)

if not unsupported_table.empty:
    st.markdown("#### Not simulated (economic-only)")
    st.caption("These levers are tracked for documentation but are not simulated in the model.")
    unsupported_display = unsupported_table.copy()
    unsupported_display.insert(1, "Status", "Not simulated (economic-only)")
    st.data_editor(
        unsupported_display,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        disabled=True,
        column_config={
            "Lever": st.column_config.TextColumn("Lever", disabled=True),
            "Status": st.column_config.TextColumn("Status", disabled=True),
            "Low change": st.column_config.NumberColumn("Low change", format="%.2f", disabled=True),
            "High change": st.column_config.NumberColumn("High change", format="%.2f", disabled=True),
            "Low impact (pp)": st.column_config.NumberColumn(
                "Low impact (pp)",
                format="%.2f",
                disabled=True,
            ),
            "High impact (pp)": st.column_config.NumberColumn(
                "High impact (pp)",
                format="%.2f",
                disabled=True,
            ),
            "Notes": st.column_config.TextColumn("Notes", disabled=True),
        },
    )

_save_impact_table(selected_metric.key, edited_table)

if run_sensitivity:
    if baseline_inputs.get(selected_metric.key) is None:
        st.warning(
            f"Baseline is required for {selected_metric.label}. "
            "Enter a baseline value (especially PIRR) before running sensitivity."
        )
    else:
        cached_results = get_simulation_results()
        econ_payload = get_latest_economics_payload() or {}
        econ_inputs = econ_payload.get("economic_inputs")
        price_inputs = econ_payload.get("price_inputs")
        wesm_profile_source = econ_payload.get("wesm_profile_source")

        cached_cfg, cached_dod_override = get_cached_simulation_config()
        base_cfg = cached_cfg or SimConfig()
        dod_override = cached_dod_override or "Auto (infer)"

        baseline_output = cached_results.sim_output if cached_results else simulate_project(
            base_cfg,
            pv_df=pv_df,
            cycle_df=cycle_df,
            dod_override=dod_override,
            need_logs=bool(
                selected_metric.key == "pirr_pct"
                and price_inputs is not None
                and (price_inputs.apply_wesm_to_shortfall or price_inputs.sell_to_wesm)
                and wesm_profile_source
            ),
        )

        updated_table, warnings = _run_sensitivity(
            base_cfg=base_cfg,
            pv_df=pv_df,
            cycle_df=cycle_df,
            metric_key=selected_metric.key,
            lever_table=edited_table,
            baseline_sim_output=baseline_output,
            dod_override=dod_override,
            econ_inputs=econ_inputs,
            price_inputs=price_inputs,
            wesm_profile_source=wesm_profile_source,
        )
        _save_impact_table(selected_metric.key, updated_table)
        edited_table = updated_table

        if warnings:
            st.warning("\n".join(f"• {warning}" for warning in warnings))
        else:
            st.success("Sensitivity impacts updated.")

st.markdown("### Tornado chart")
baseline_value = baseline_inputs.get(selected_metric.key)
if baseline_value is None:
    st.metric("Baseline", "Missing")
else:
    st.metric("Baseline", f"{baseline_value:,.2f}%")

chart_source = _prepare_tornado_data(edited_table)
if chart_source["Impact (pp)"].abs().sum() == 0:
    st.info(
        "All impacts are currently zero. Enter low/high changes and run sensitivity to populate the chart.",
        icon="ℹ️",
    )

st.altair_chart(_build_tornado_chart(chart_source), use_container_width=True)

st.markdown("### Notes & export")
st.caption(
    "Example usage: if an RTE downside is expected to reduce compliance by 1.5pp, "
    "enter -1.5 in Low change for that lever and click Run Sensitivity."
)

export_df = edited_table.copy()
export_df["Metric"] = selected_metric.label
export_df["Baseline (%)"] = baseline_value
export_csv = export_df.to_csv(index=False).encode("utf-8")
export_json = json.dumps(
    {
        "schema_version": 1,
        "selected_metric_label": selected_metric.label,
        "baseline_inputs": baseline_inputs,
        "impact_tables": _collect_impact_tables(metrics),
    },
    indent=2,
).encode("utf-8")

st.download_button(
    "Download sensitivity table (CSV)",
    data=export_csv,
    file_name="sensitivity_inputs.csv",
    mime="text/csv",
    use_container_width=True,
    disabled=export_disabled,
)
st.download_button(
    "Download sensitivity inputs (JSON)",
    data=export_json,
    file_name="sensitivity_inputs.json",
    mime="application/json",
    use_container_width=True,
    disabled=export_disabled,
)
st.download_button(
    "Download sensitivity inputs (JSON)",
    data=export_json,
    file_name="sensitivity_inputs.json",
    mime="application/json",
    use_container_width=True,
)
st.download_button(
    "Download sensitivity inputs (JSON)",
    data=export_json,
    file_name="sensitivity_inputs.json",
    mime="application/json",
    use_container_width=True,
)

st.caption(
    "Surplus % uses PV curtailment ÷ total expected firm energy. Deficit % uses total shortfall ÷ total expected firm energy."
)
