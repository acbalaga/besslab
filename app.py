import io
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import utils.economics as economics
from services.simulation_core import (
    HourlyLog,
    SimConfig,
    Window,
    YearResult,
    resolve_efficiencies,
    simulate_project,
    summarize_simulation,
)
from frontend.ui.charts import (
    AvgProfileBundle,
    build_avg_profile_bundle,
    build_avg_profile_chart,
    prepare_charge_discharge_envelope,
    prepare_soc_heatmap_data,
)
from frontend.ui.forms import (
    SimulationFormResult,
    render_rate_limit_section,
    render_simulation_form,
)
from frontend.ui.metrics import KPIResults, compute_kpis, render_primary_metrics
from frontend.ui.rendering import MetricSpec, render_formatted_dataframe, render_metrics
from frontend.ui.pdf import build_pdf_summary
from frontend.ui.sensitivity_tornado import (
    apply_capex_delta,
    apply_opex_delta,
    apply_tariff_delta,
    build_simple_lever_table,
    build_tornado_chart,
    prepare_tornado_data,
)
from utils import (
    FLAG_DEFINITIONS,
    build_flag_insights,
    enforce_rate_limit,
    read_wesm_profile,
)
from utils.economics import (
    CashFlowOutputs,
    EconomicInputs,
    EconomicOutputs,
    FinancingOutputs,
    PriceInputs,
    aggregate_wesm_profile_to_annual,
    build_financing_cash_flow_table,
    build_operating_cash_flow_table,
    compute_cash_flows_and_irr,
    compute_financing_cash_flows,
    compute_lcoe_lcos_with_augmentation_fallback,
    estimate_augmentation_costs_by_year,
    normalize_economic_inputs,
)
from utils.ui_layout import init_page_layout
from utils.ui_state import (
    bootstrap_session_state,
    cache_latest_economics_payload,
    build_inputs_fingerprint,
    get_base_dir,
    get_cached_simulation_config,
    get_simulation_results,
    load_shared_data,
    save_last_run_inputs_fingerprint,
    save_manual_aug_schedule_rows,
    save_simulation_config,
    save_simulation_results,
    save_simulation_snapshot,
)


BASE_DIR = get_base_dir()

INPUTS_JSON_SCHEMA_VERSION = 1
INPUTS_FORM_KEYS = {
    "years": "inputs_years",
    "pv_deg_pct": "inputs_pv_deg_pct",
    "pv_avail": "inputs_pv_avail",
    "bess_avail": "inputs_bess_avail",
    "use_split_rte": "inputs_use_split_rte",
    "charge_eff": "inputs_charge_eff",
    "discharge_eff": "inputs_discharge_eff",
    "rte": "inputs_rte",
    "run_economics": "inputs_run_economics",
    "augmentation_mode": "augmentation_strategy_mode",
    "aug_trigger_type": "inputs_aug_trigger_type",
    "aug_threshold_margin_pct": "inputs_aug_threshold_margin_pct",
    "aug_topup_margin_pct": "inputs_aug_topup_margin_pct",
    "aug_soh_trigger_pct": "inputs_aug_soh_trigger_pct",
    "aug_soh_add_pct": "inputs_aug_soh_add_pct",
    "aug_periodic_every_years": "inputs_aug_periodic_every_years",
    "aug_periodic_add_pct": "inputs_aug_periodic_add_pct",
    "aug_size_mode": "inputs_aug_size_mode",
    "aug_fixed_energy_mwh": "inputs_aug_fixed_energy_mwh",
    "aug_retire_enabled": "inputs_aug_retire_enabled",
    "aug_retire_soh_pct": "inputs_aug_retire_soh_pct",
    "aug_retire_replace_mode": "inputs_aug_retire_replace_mode",
    "aug_retire_replace_pct": "inputs_aug_retire_replace_pct",
    "aug_retire_replace_fixed_mwh": "inputs_aug_retire_replace_fixed_mwh",
    "initial_power_mw": "inputs_initial_power_mw",
    "initial_usable_mwh": "inputs_initial_usable_mwh",
    "soc_floor_pct": "inputs_soc_floor_pct",
    "soc_ceiling_pct": "inputs_soc_ceiling_pct",
    "contracted_mw": "inputs_contracted_mw",
    "discharge_windows": "inputs_discharge_windows",
    "charge_windows": "inputs_charge_windows",
    "calendar_fade_pct": "inputs_calendar_fade_pct",
    "dod_override": "inputs_dod_override",
    "wacc_pct": "inputs_wacc_pct",
    "inflation_pct": "inputs_inflation_pct",
    "forex_rate_php_per_usd": "inputs_forex_rate_php_per_usd",
    "capex_mode": "inputs_capex_mode",
    "capex_usd_per_kwh": "inputs_capex_usd_per_kwh",
    "capex_musd": "inputs_capex_musd",
    "pv_capex_musd": "inputs_pv_capex_musd",
    "opex_mode": "inputs_opex_mode",
    "fixed_opex_pct": "inputs_fixed_opex_pct",
    "opex_php_per_kwh": "inputs_opex_php_per_kwh",
    "fixed_opex_musd": "inputs_fixed_opex_musd",
    "include_devex_year0": "inputs_include_devex_year0",
    "devex_cost_php": "inputs_devex_cost_php",
    "contract_price_php_per_kwh": "inputs_contract_price_php_per_kwh",
    "escalate_prices": "inputs_escalate_prices",
    "wesm_pricing_enabled": "inputs_wesm_pricing_enabled",
    "sell_to_wesm": "inputs_sell_to_wesm",
    "debt_equity_ratio": "inputs_debt_equity_ratio",
    "cost_of_debt_pct": "inputs_cost_of_debt_pct",
    "tenor_years": "inputs_tenor_years",
    "variable_schedule_choice": "inputs_variable_schedule_choice",
    "periodic_variable_opex_usd": "inputs_periodic_variable_opex_usd",
    "periodic_variable_opex_interval_years": "inputs_periodic_variable_opex_interval_years",
    "variable_opex_custom_text": "inputs_variable_opex_custom_text",
}


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_choice(value: Any, options: List[Any], default: Any) -> Any:
    """Return a valid choice for selectboxes/radios or fall back to default."""

    return value if value in options else default


def _coerce_year_choice(value: Any, options: List[int], fallback: int) -> int:
    """Return a valid project-life option or fall back to the closest default."""

    if value in options:
        return int(value)
    if fallback in options:
        return int(fallback)
    return options[0]


def _format_hhmm(hour_value: float) -> str:
    hours = int(hour_value)
    minutes = int(round((hour_value - hours) * 60))
    if minutes == 60:
        hours = (hours + 1) % 24
        minutes = 0
    return f"{hours:02d}:{minutes:02d}"


def _windows_to_text(windows: List[Window]) -> str:
    return ", ".join(f"{_format_hhmm(w.start)}-{_format_hhmm(w.end)}" for w in windows)


def _coerce_windows(payload: Any, fallback: List[Window]) -> List[Window]:
    if isinstance(payload, list):
        windows: List[Window] = []
        for item in payload:
            if isinstance(item, Window):
                windows.append(item)
            elif isinstance(item, dict):
                start = _coerce_optional_float(item.get("start"))
                end = _coerce_optional_float(item.get("end"))
                if start is not None and end is not None:
                    windows.append(Window(start=start, end=end))
        if windows:
            return windows
    return fallback


def _coerce_manual_schedule(payload: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(payload, list):
        return rows
    for item in payload:
        if not isinstance(item, dict):
            continue
        year = _coerce_optional_int(item.get("year") or item.get("Year"))
        basis = item.get("basis") or item.get("Basis")
        amount = item.get("value") if "value" in item else item.get("Amount")
        amount_value = _coerce_optional_float(amount)
        if year is None or basis is None or amount_value is None:
            continue
        rows.append({"Year": year, "Basis": str(basis), "Amount": amount_value})
    return rows


def _build_inputs_payload(
    cfg: SimConfig,
    dod_override: Optional[str],
    run_economics: bool,
    econ_inputs: Optional[EconomicInputs],
    price_inputs: Optional[PriceInputs],
    discharge_windows_text: str,
    charge_windows_text: str,
) -> Dict[str, Any]:
    manual_schedule_rows: List[Dict[str, Any]] = []
    for entry in cfg.augmentation_schedule or []:
        if isinstance(entry, dict):
            year_value = entry.get("year") or entry.get("Year")
            basis_value = entry.get("basis") or entry.get("Basis")
            amount_value = entry.get("value") if "value" in entry else entry.get("Amount")
        else:
            year_value = getattr(entry, "year", None)
            basis_value = getattr(entry, "basis", None)
            amount_value = getattr(entry, "value", None)
        if year_value is None or basis_value is None or amount_value is None:
            continue
        manual_schedule_rows.append(
            {"Year": int(year_value), "Basis": str(basis_value), "Amount": float(amount_value)}
        )

    payload = {
        "schema_version": INPUTS_JSON_SCHEMA_VERSION,
        "config": asdict(cfg),
        "discharge_windows_text": discharge_windows_text,
        "charge_windows_text": charge_windows_text,
        "dod_override": dod_override,
        "run_economics": run_economics,
        "economic_inputs": asdict(econ_inputs) if econ_inputs is not None else None,
        "price_inputs": asdict(price_inputs) if price_inputs is not None else None,
        "manual_augmentation_schedule": manual_schedule_rows,
    }
    return payload


def _apply_inputs_payload(payload: Dict[str, Any], fallback_cfg: SimConfig) -> None:
    """Populate session state defaults for input widgets before they render."""

    config_payload = payload.get("config")
    config_payload = config_payload if isinstance(config_payload, dict) else {}
    econ_payload = payload.get("economic_inputs")
    econ_payload = econ_payload if isinstance(econ_payload, dict) else {}
    price_payload = payload.get("price_inputs")
    price_payload = price_payload if isinstance(price_payload, dict) else {}

    year_options = list(range(10, 36, 5))
    years = _coerce_year_choice(
        _coerce_int(config_payload.get("years"), fallback_cfg.years),
        year_options,
        fallback_cfg.years,
    )
    discharge_windows = _coerce_windows(config_payload.get("discharge_windows"), fallback_cfg.discharge_windows)
    discharge_windows_text = config_payload.get("discharge_windows_text") or payload.get("discharge_windows_text")
    if not discharge_windows_text:
        discharge_windows_text = _windows_to_text(discharge_windows)
    charge_windows_text = config_payload.get("charge_windows_text")
    if charge_windows_text is None:
        charge_windows_text = fallback_cfg.charge_windows_text or ""

    forex_rate_php_per_usd = _coerce_float(
        econ_payload.get("forex_rate_php_per_usd"),
        economics.DEFAULT_FOREX_RATE_PHP_PER_USD,
    )
    run_economics = _coerce_bool(
        payload.get("run_economics"),
        bool(econ_payload or price_payload),
    )

    st.session_state[INPUTS_FORM_KEYS["years"]] = years
    st.session_state[INPUTS_FORM_KEYS["pv_deg_pct"]] = _coerce_float(
        config_payload.get("pv_deg_rate"),
        fallback_cfg.pv_deg_rate,
    ) * 100.0
    st.session_state[INPUTS_FORM_KEYS["pv_avail"]] = _coerce_float(
        config_payload.get("pv_availability"),
        fallback_cfg.pv_availability,
    )
    st.session_state[INPUTS_FORM_KEYS["bess_avail"]] = _coerce_float(
        config_payload.get("bess_availability"),
        fallback_cfg.bess_availability,
    )
    use_split_rte = _coerce_bool(
        config_payload.get("use_split_rte"),
        fallback_cfg.use_split_rte,
    )
    st.session_state[INPUTS_FORM_KEYS["use_split_rte"]] = use_split_rte
    st.session_state[INPUTS_FORM_KEYS["charge_eff"]] = _coerce_float(
        config_payload.get("charge_efficiency"),
        fallback_cfg.charge_efficiency or 0.94,
    )
    st.session_state[INPUTS_FORM_KEYS["discharge_eff"]] = _coerce_float(
        config_payload.get("discharge_efficiency"),
        fallback_cfg.discharge_efficiency or 0.94,
    )
    st.session_state[INPUTS_FORM_KEYS["rte"]] = _coerce_float(
        config_payload.get("rte_roundtrip"),
        fallback_cfg.rte_roundtrip,
    )
    st.session_state[INPUTS_FORM_KEYS["run_economics"]] = run_economics
    st.session_state[INPUTS_FORM_KEYS["augmentation_mode"]] = _coerce_choice(
        config_payload.get("augmentation"),
        ["None", "Threshold", "Periodic", "Manual"],
        fallback_cfg.augmentation,
    )
    st.session_state[INPUTS_FORM_KEYS["aug_trigger_type"]] = _coerce_choice(
        config_payload.get("aug_trigger_type"),
        ["Capability", "SOH"],
        fallback_cfg.aug_trigger_type,
    )
    st.session_state[INPUTS_FORM_KEYS["aug_threshold_margin_pct"]] = _coerce_float(
        config_payload.get("aug_threshold_margin"),
        fallback_cfg.aug_threshold_margin,
    ) * 100.0
    st.session_state[INPUTS_FORM_KEYS["aug_topup_margin_pct"]] = _coerce_float(
        config_payload.get("aug_topup_margin"),
        fallback_cfg.aug_topup_margin,
    ) * 100.0
    st.session_state[INPUTS_FORM_KEYS["aug_soh_trigger_pct"]] = _coerce_float(
        config_payload.get("aug_soh_trigger_pct"),
        fallback_cfg.aug_soh_trigger_pct,
    ) * 100.0
    st.session_state[INPUTS_FORM_KEYS["aug_soh_add_pct"]] = _coerce_float(
        config_payload.get("aug_soh_add_frac_initial"),
        fallback_cfg.aug_soh_add_frac_initial,
    ) * 100.0
    st.session_state[INPUTS_FORM_KEYS["aug_periodic_every_years"]] = _coerce_int(
        config_payload.get("aug_periodic_every_years"),
        fallback_cfg.aug_periodic_every_years,
    )
    st.session_state[INPUTS_FORM_KEYS["aug_periodic_add_pct"]] = _coerce_float(
        config_payload.get("aug_periodic_add_frac_of_bol"),
        fallback_cfg.aug_periodic_add_frac_of_bol,
    ) * 100.0
    st.session_state[INPUTS_FORM_KEYS["aug_size_mode"]] = _coerce_choice(
        config_payload.get("aug_add_mode"),
        ["Percent", "Fixed"],
        fallback_cfg.aug_add_mode,
    )
    st.session_state[INPUTS_FORM_KEYS["aug_fixed_energy_mwh"]] = _coerce_float(
        config_payload.get("aug_fixed_energy_mwh"),
        fallback_cfg.aug_fixed_energy_mwh,
    )
    st.session_state[INPUTS_FORM_KEYS["aug_retire_enabled"]] = _coerce_bool(
        config_payload.get("aug_retire_old_cohort"),
        fallback_cfg.aug_retire_old_cohort,
    )
    st.session_state[INPUTS_FORM_KEYS["aug_retire_soh_pct"]] = _coerce_float(
        config_payload.get("aug_retire_soh_pct"),
        fallback_cfg.aug_retire_soh_pct,
    ) * 100.0
    st.session_state[INPUTS_FORM_KEYS["aug_retire_replace_mode"]] = _coerce_choice(
        config_payload.get("aug_retire_replacement_mode"),
        ["None", "Percent", "Fixed"],
        fallback_cfg.aug_retire_replacement_mode,
    )
    st.session_state[INPUTS_FORM_KEYS["aug_retire_replace_pct"]] = _coerce_float(
        config_payload.get("aug_retire_replacement_pct_bol"),
        fallback_cfg.aug_retire_replacement_pct_bol,
    ) * 100.0
    st.session_state[INPUTS_FORM_KEYS["aug_retire_replace_fixed_mwh"]] = _coerce_float(
        config_payload.get("aug_retire_replacement_fixed_mwh"),
        fallback_cfg.aug_retire_replacement_fixed_mwh,
    )
    st.session_state[INPUTS_FORM_KEYS["initial_power_mw"]] = _coerce_float(
        config_payload.get("initial_power_mw"),
        fallback_cfg.initial_power_mw,
    )
    st.session_state[INPUTS_FORM_KEYS["initial_usable_mwh"]] = _coerce_float(
        config_payload.get("initial_usable_mwh"),
        fallback_cfg.initial_usable_mwh,
    )
    st.session_state[INPUTS_FORM_KEYS["soc_floor_pct"]] = _coerce_float(
        config_payload.get("soc_floor"),
        fallback_cfg.soc_floor,
    ) * 100.0
    st.session_state[INPUTS_FORM_KEYS["soc_ceiling_pct"]] = _coerce_float(
        config_payload.get("soc_ceiling"),
        fallback_cfg.soc_ceiling,
    ) * 100.0
    st.session_state[INPUTS_FORM_KEYS["contracted_mw"]] = _coerce_float(
        config_payload.get("contracted_mw"),
        fallback_cfg.contracted_mw,
    )
    st.session_state[INPUTS_FORM_KEYS["discharge_windows"]] = discharge_windows_text
    st.session_state[INPUTS_FORM_KEYS["charge_windows"]] = charge_windows_text
    st.session_state[INPUTS_FORM_KEYS["calendar_fade_pct"]] = _coerce_float(
        config_payload.get("calendar_fade_rate"),
        fallback_cfg.calendar_fade_rate,
    ) * 100.0
    dod_options = ["Auto (infer)", "10%", "20%", "40%", "80%", "100%"]
    st.session_state[INPUTS_FORM_KEYS["dod_override"]] = _coerce_choice(
        payload.get("dod_override"),
        dod_options,
        st.session_state.get(INPUTS_FORM_KEYS["dod_override"], "Auto (infer)"),
    )

    manual_schedule_rows = _coerce_manual_schedule(
        payload.get("manual_augmentation_schedule") or config_payload.get("augmentation_schedule")
    )
    if manual_schedule_rows:
        save_manual_aug_schedule_rows(manual_schedule_rows, years)

    st.session_state[INPUTS_FORM_KEYS["wacc_pct"]] = _coerce_float(
        econ_payload.get("wacc"),
        0.08,
    ) * 100.0
    st.session_state[INPUTS_FORM_KEYS["inflation_pct"]] = _coerce_float(
        econ_payload.get("inflation_rate"),
        0.03,
    ) * 100.0
    st.session_state[INPUTS_FORM_KEYS["forex_rate_php_per_usd"]] = forex_rate_php_per_usd

    capex_mode = "Total CAPEX (USD million)"
    if econ_payload.get("capex_usd_per_kwh") is not None:
        capex_mode = "USD/kWh (BOL)"
    st.session_state[INPUTS_FORM_KEYS["capex_mode"]] = _coerce_choice(
        econ_payload.get("capex_mode"),
        ["USD/kWh (BOL)", "Total CAPEX (USD million)"],
        capex_mode,
    )
    st.session_state[INPUTS_FORM_KEYS["capex_usd_per_kwh"]] = _coerce_float(
        econ_payload.get("capex_usd_per_kwh"),
        0.0,
    )
    capex_musd = econ_payload.get("capex_musd")
    capex_total_usd = econ_payload.get("capex_total_usd")
    if capex_musd is None and capex_total_usd is not None:
        capex_musd = _coerce_float(capex_total_usd, 0.0) / 1_000_000.0
    st.session_state[INPUTS_FORM_KEYS["capex_musd"]] = _coerce_float(capex_musd, 40.0)
    st.session_state[INPUTS_FORM_KEYS["pv_capex_musd"]] = _coerce_float(
        econ_payload.get("pv_capex_musd"),
        0.0,
    )

    opex_mode = "% of CAPEX per year"
    if econ_payload.get("opex_php_per_kwh") is not None:
        opex_mode = "PHP/kWh on total generation"
    st.session_state[INPUTS_FORM_KEYS["opex_mode"]] = _coerce_choice(
        econ_payload.get("opex_mode"),
        ["% of CAPEX per year", "PHP/kWh on total generation"],
        opex_mode,
    )
    st.session_state[INPUTS_FORM_KEYS["fixed_opex_pct"]] = _coerce_float(
        econ_payload.get("fixed_opex_pct_of_capex"),
        2.0,
    )
    st.session_state[INPUTS_FORM_KEYS["opex_php_per_kwh"]] = _coerce_float(
        econ_payload.get("opex_php_per_kwh"),
        0.0,
    )
    st.session_state[INPUTS_FORM_KEYS["fixed_opex_musd"]] = _coerce_float(
        econ_payload.get("fixed_opex_musd"),
        0.0,
    )
    st.session_state[INPUTS_FORM_KEYS["include_devex_year0"]] = _coerce_bool(
        econ_payload.get("include_devex_year0"),
        False,
    )
    st.session_state[INPUTS_FORM_KEYS["devex_cost_php"]] = _coerce_float(
        econ_payload.get("devex_cost_php"),
        economics.DEVEX_COST_PHP,
    )

    st.session_state[INPUTS_FORM_KEYS["contract_price_php_per_kwh"]] = _coerce_float(
        price_payload.get("contract_price_usd_per_mwh"),
        0.0,
    ) * forex_rate_php_per_usd / 1000.0
    st.session_state[INPUTS_FORM_KEYS["escalate_prices"]] = _coerce_bool(
        price_payload.get("escalate_with_inflation"),
        False,
    )
    st.session_state[INPUTS_FORM_KEYS["wesm_pricing_enabled"]] = _coerce_bool(
        price_payload.get("apply_wesm_to_shortfall"),
        False,
    )
    st.session_state[INPUTS_FORM_KEYS["sell_to_wesm"]] = _coerce_bool(
        price_payload.get("sell_to_wesm"),
        False,
    )

    default_debt_ratio = economics.DEFAULT_DEBT_EQUITY_RATIO / (1.0 + economics.DEFAULT_DEBT_EQUITY_RATIO)
    debt_ratio = _coerce_float(econ_payload.get("debt_ratio"), default_debt_ratio)
    debt_equity_ratio = debt_ratio / (1.0 - debt_ratio) if 0 < debt_ratio < 1.0 else 0.0
    st.session_state[INPUTS_FORM_KEYS["debt_equity_ratio"]] = debt_equity_ratio
    st.session_state[INPUTS_FORM_KEYS["cost_of_debt_pct"]] = _coerce_float(
        econ_payload.get("cost_of_debt"),
        economics.DEFAULT_COST_OF_DEBT_PCT / 100.0,
    ) * 100.0
    st.session_state[INPUTS_FORM_KEYS["tenor_years"]] = _coerce_int(
        econ_payload.get("tenor_years"),
        economics.DEFAULT_TENOR_YEARS,
    )

    variable_schedule_choice = "None"
    variable_schedule = econ_payload.get("variable_opex_schedule_usd")
    periodic_variable_opex_usd = econ_payload.get("periodic_variable_opex_usd")
    periodic_variable_opex_interval_years = econ_payload.get("periodic_variable_opex_interval_years")
    custom_variable_text = ""
    if isinstance(variable_schedule, list):
        variable_schedule_choice = "Custom"
        custom_variable_text = "\n".join(str(val) for val in variable_schedule)
    elif periodic_variable_opex_usd is not None:
        variable_schedule_choice = "Periodic"
    st.session_state[INPUTS_FORM_KEYS["variable_schedule_choice"]] = _coerce_choice(
        econ_payload.get("variable_schedule_choice"),
        ["None", "Periodic", "Custom"],
        variable_schedule_choice,
    )
    st.session_state[INPUTS_FORM_KEYS["periodic_variable_opex_usd"]] = _coerce_float(
        periodic_variable_opex_usd,
        0.0,
    )
    st.session_state[INPUTS_FORM_KEYS["periodic_variable_opex_interval_years"]] = _coerce_int(
        periodic_variable_opex_interval_years,
        5,
    )
    st.session_state[INPUTS_FORM_KEYS["variable_opex_custom_text"]] = custom_variable_text


def _build_hourly_summary_df(logs: HourlyLog) -> pd.DataFrame:
    """Return a tidy hourly summary with MW/MWh metrics for download."""
    hour_index = np.arange(len(logs.hod), dtype=int)
    data = {
        "hour_index": hour_index,
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
    df["pv_surplus_mw"] = np.maximum(
        df["pv_mw"] - df["pv_to_contract_mw"] - df["charge_mw"],
        0.0,
    )
    column_order = [
        "timestamp",
        "hour_index",
        "hod",
        "pv_mw",
        "pv_to_contract_mw",
        "bess_to_contract_mw",
        "delivered_mw",
        "shortfall_mw",
        "charge_mw",
        "discharge_mw",
        "soc_mwh",
        "pv_surplus_mw",
    ]
    existing_columns = [col for col in column_order if col in df.columns]
    return df[existing_columns]


@dataclass(frozen=True)
class TornadoLever:
    label: str
    delta_unit: str
    category: str
    apply_config: Optional[Callable[[SimConfig, float], SimConfig]] = None
    apply_econ: Optional[Callable[[EconomicInputs, float], EconomicInputs]] = None
    apply_price: Optional[Callable[[PriceInputs, float], PriceInputs]] = None


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
    return SimConfig(**cfg.__dict__)


def _apply_rte_delta(cfg: SimConfig, delta_pp: float) -> SimConfig:
    updated = _clone_config(cfg)
    updated.rte_roundtrip = _clamp(updated.rte_roundtrip + delta_pp / 100.0, 0.5, 0.99)
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


def _simple_tornado_levers() -> Dict[str, TornadoLever]:
    return {
        "Availability": TornadoLever(
            label="Availability",
            delta_unit="pp",
            category="operational",
            apply_config=_apply_availability_delta,
        ),
        "RTE": TornadoLever(
            label="RTE",
            delta_unit="pp",
            category="operational",
            apply_config=_apply_rte_delta,
        ),
        "BESS size (MWh)": TornadoLever(
            label="BESS size (MWh)",
            delta_unit="MWh",
            category="operational",
            apply_config=_apply_energy_delta,
        ),
        "BESS capacity (MW)": TornadoLever(
            label="BESS capacity (MW)",
            delta_unit="MW",
            category="operational",
            apply_config=_apply_power_delta,
        ),
        "Dispatch window": TornadoLever(
            label="Dispatch window",
            delta_unit="hours",
            category="operational",
            apply_config=_apply_window_delta,
        ),
        "CAPEX": TornadoLever(
            label="CAPEX",
            delta_unit="%",
            category="financial",
            apply_econ=apply_capex_delta,
        ),
        "OPEX": TornadoLever(
            label="OPEX",
            delta_unit="%",
            category="financial",
            apply_econ=apply_opex_delta,
        ),
        "Tariff": TornadoLever(
            label="Tariff",
            delta_unit="%",
            category="financial",
            apply_price=apply_tariff_delta,
        ),
    }


def _extract_annual_series(sim_output) -> Dict[str, List[float]]:
    results = sim_output.results
    annual_delivered = [r.delivered_firm_mwh for r in results]
    annual_bess = [r.bess_to_contract_mwh for r in results]
    annual_pv_excess = [r.pv_curtailed_mwh for r in results]
    annual_shortfall = [r.shortfall_mwh for r in results]
    annual_total_generation = [r.available_pv_mwh for r in results]
    annual_pv_delivered = [
        float(delivered) - float(bess) for delivered, bess in zip(annual_delivered, annual_bess)
    ]
    return {
        "annual_delivered": annual_delivered,
        "annual_bess": annual_bess,
        "annual_pv_excess": annual_pv_excess,
        "annual_shortfall": annual_shortfall,
        "annual_total_generation": annual_total_generation,
        "annual_pv_delivered": annual_pv_delivered,
    }


def _compute_wesm_costs(
    sim_output,
    normalized_econ_inputs: EconomicInputs,
    price_inputs: PriceInputs,
    wesm_profile_source: Optional[str],
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    if not (price_inputs.apply_wesm_to_shortfall or price_inputs.sell_to_wesm):
        return None, None
    if wesm_profile_source is None or not sim_output.hourly_logs_by_year:
        return None, None

    wesm_profile_df = read_wesm_profile(
        [wesm_profile_source],
        forex_rate_php_per_usd=normalized_econ_inputs.forex_rate_php_per_usd,
    )
    hourly_summary_by_year = {
        year_index: _build_hourly_summary_df(logs)
        for year_index, logs in sim_output.hourly_logs_by_year.items()
    }
    return aggregate_wesm_profile_to_annual(
        hourly_summary_by_year,
        wesm_profile_df,
        step_hours=sim_output.cfg.step_hours,
        apply_inflation=False,
        inflation_rate=normalized_econ_inputs.inflation_rate,
    )


def _compute_pirr(
    sim_output,
    econ_inputs: EconomicInputs,
    price_inputs: PriceInputs,
    wesm_profile_source: Optional[str],
) -> float:
    if econ_inputs.capex_usd_per_kwh is not None and econ_inputs.bess_bol_kwh is None:
        econ_inputs = econ_inputs.__class__(
            **{
                **econ_inputs.__dict__,
                "bess_bol_kwh": sim_output.cfg.initial_usable_mwh * 1000.0,
            }
        )
    normalized_inputs = normalize_economic_inputs(econ_inputs)
    series = _extract_annual_series(sim_output)
    annual_wesm_shortfall_cost_usd, annual_wesm_surplus_revenue_usd = _compute_wesm_costs(
        sim_output,
        normalized_inputs,
        price_inputs,
        wesm_profile_source,
    )
    augmentation_costs_usd = estimate_augmentation_costs_by_year(
        sim_output.augmentation_energy_added_mwh,
        sim_output.cfg.initial_usable_mwh,
        normalized_inputs.capex_musd,
    )
    financing_outputs = compute_financing_cash_flows(
        series["annual_delivered"],
        series["annual_bess"],
        series["annual_pv_excess"],
        normalized_inputs,
        price_inputs,
        annual_shortfall_mwh=series["annual_shortfall"],
        annual_wesm_shortfall_cost_usd=annual_wesm_shortfall_cost_usd,
        annual_wesm_surplus_revenue_usd=annual_wesm_surplus_revenue_usd,
        augmentation_costs_usd=augmentation_costs_usd if any(augmentation_costs_usd) else None,
        annual_total_generation_mwh=series["annual_total_generation"],
    )
    return financing_outputs.project_irr_pct


def _compute_operational_metric(sim_output, metric_key: str) -> Optional[float]:
    summary = summarize_simulation(sim_output)
    expected_total = sum(r.expected_firm_mwh for r in sim_output.results)
    if metric_key == "compliance_pct":
        return summary.compliance
    if metric_key == "surplus_pct":
        return (summary.pv_excess_mwh / expected_total * 100.0) if expected_total > 0 else None
    return None


def _run_simple_tornado(
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
    updated_table = lever_table.copy()
    warnings: List[str] = []
    levers = _simple_tornado_levers()

    if metric_key == "pirr_pct" and (econ_inputs is None or price_inputs is None):
        return updated_table, ["PIRR sensitivity requires economics inputs. Run the model with economics first."]

    baseline_value = (
        _compute_pirr(baseline_sim_output, econ_inputs, price_inputs, wesm_profile_source)
        if metric_key == "pirr_pct" and econ_inputs is not None and price_inputs is not None
        else _compute_operational_metric(baseline_sim_output, metric_key)
    )

    if baseline_value is None:
        warnings.append("Baseline metric could not be computed; impacts will remain blank.")

    need_logs = bool(
        metric_key == "pirr_pct"
        and price_inputs is not None
        and (price_inputs.apply_wesm_to_shortfall or price_inputs.sell_to_wesm)
        and wesm_profile_source
    )

    for idx, row in updated_table.iterrows():
        lever_label = str(row.get("Lever"))
        lever = levers.get(lever_label)
        if lever is None:
            continue

        if metric_key in {"compliance_pct", "surplus_pct"} and lever.category == "financial":
            # Finance-only levers do not affect operational metrics; they only shift PIRR.
            updated_table.at[idx, "Low impact (pp)"] = None
            updated_table.at[idx, "High impact (pp)"] = None
            continue

        low_delta = float(row.get("Low change", 0.0) or 0.0)
        high_delta = float(row.get("High change", 0.0) or 0.0)

        for column, delta in (("Low impact (pp)", low_delta), ("High impact (pp)", high_delta)):
            adjusted_cfg = base_cfg
            adjusted_econ = econ_inputs
            adjusted_price = price_inputs

            if lever.apply_config is not None:
                adjusted_cfg = lever.apply_config(base_cfg, delta)
            if lever.apply_econ is not None and adjusted_econ is not None:
                adjusted_econ = lever.apply_econ(adjusted_econ, delta)
            if lever.apply_price is not None and adjusted_price is not None:
                adjusted_price = lever.apply_price(adjusted_price, delta)

            sim_output = baseline_sim_output
            if lever.apply_config is not None:
                sim_output = simulate_project(
                    adjusted_cfg,
                    pv_df=pv_df,
                    cycle_df=cycle_df,
                    dod_override=dod_override,
                    need_logs=need_logs,
                )

            metric_value = (
                _compute_pirr(sim_output, adjusted_econ, adjusted_price, wesm_profile_source)
                if metric_key == "pirr_pct" and adjusted_econ is not None and adjusted_price is not None
                else _compute_operational_metric(sim_output, metric_key)
            )
            if metric_value is None or baseline_value is None:
                updated_table.at[idx, column] = None
            else:
                updated_table.at[idx, column] = metric_value - baseline_value

    return updated_table, sorted(set(warnings))


def _build_hourly_summary_workbook(hourly_logs_by_year: Dict[int, HourlyLog]) -> bytes:
    """Create an Excel workbook with one worksheet per project year."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for year_index in sorted(hourly_logs_by_year):
            sheet_name = f"Year {year_index}"
            df = _build_hourly_summary_df(hourly_logs_by_year[year_index])
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()


def _build_daily_summary_df(
    hourly_logs_by_year: Dict[int, HourlyLog],
    step_hours: float,
) -> pd.DataFrame:
    """Aggregate hourly logs into daily MWh summaries for auditing."""

    if step_hours <= 0:
        raise ValueError("step_hours must be positive")

    daily_frames: list[pd.DataFrame] = []
    steps_per_day = int(round(24.0 / step_hours))
    for year_index in sorted(hourly_logs_by_year):
        hourly_df = _build_hourly_summary_df(hourly_logs_by_year[year_index])
        if hourly_df.empty:
            continue

        if "timestamp" in hourly_df.columns:
            hourly_df["day"] = pd.to_datetime(hourly_df["timestamp"]).dt.date
        else:
            hourly_df["day"] = (hourly_df["hour_index"] // steps_per_day) + 1

        energy_df = pd.DataFrame(
            {
                "Year": year_index,
                "Day": hourly_df["day"],
                "PV MWh": hourly_df["pv_mw"] * step_hours,
                "PV→Contract MWh": hourly_df["pv_to_contract_mw"] * step_hours,
                "BESS→Contract MWh": hourly_df["bess_to_contract_mw"] * step_hours,
                "Delivered MWh": hourly_df["delivered_mw"] * step_hours,
                "Shortfall MWh": hourly_df["shortfall_mw"] * step_hours,
                "Charge MWh": hourly_df["charge_mw"] * step_hours,
                "Discharge MWh": hourly_df["discharge_mw"] * step_hours,
                "PV surplus MWh": hourly_df["pv_surplus_mw"] * step_hours,
            }
        )
        daily_frames.append(
            energy_df.groupby(["Year", "Day"], as_index=False)
            .sum(numeric_only=True)
        )

    if not daily_frames:
        return pd.DataFrame()

    return pd.concat(daily_frames, ignore_index=True)


def _build_finance_audit_workbook(
    assumptions_df: pd.DataFrame,
    metrics_summary: pd.DataFrame,
    yearly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    operating_cash_flow_df: pd.DataFrame,
    financing_cash_flow_df: pd.DataFrame,
) -> bytes:
    """Create an Excel workbook with cash-flow audit tables and energy traces."""

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        assumptions_df.to_excel(writer, sheet_name="Assumptions", index=False)
        metrics_summary.to_excel(writer, sheet_name="Metrics summary", index=False)
        operating_cash_flow_df.to_excel(writer, sheet_name="Operating cash flows", index=False)
        financing_cash_flow_df.to_excel(writer, sheet_name="Financing cash flows", index=False)
        yearly_df.to_excel(writer, sheet_name="Yearly energy", index=False)
        monthly_df.to_excel(writer, sheet_name="Monthly energy", index=False)
        if daily_df.empty:
            pd.DataFrame({"Note": ["Daily logs unavailable (hourly logs not generated)."]}).to_excel(
                writer,
                sheet_name="Daily energy",
                index=False,
            )
        else:
            daily_df.to_excel(writer, sheet_name="Daily energy", index=False)
    return output.getvalue()


def _build_finance_assumptions_df(
    normalized_econ_inputs: EconomicInputs,
    price_inputs: PriceInputs,
) -> pd.DataFrame:
    """Summarize economics assumptions for the audit workbook."""

    def _yes_no(flag: bool) -> str:
        return "Yes" if flag else "No"

    def _percent(value: float | None) -> float | None:
        return None if value is None else value * 100.0

    rows: list[dict[str, object]] = []

    def add_row(category: str, label: str, value: object, notes: str = "") -> None:
        rows.append(
            {"Category": category, "Assumption": label, "Value": value, "Units/Notes": notes}
        )

    add_row("Capital", "BESS CAPEX", normalized_econ_inputs.capex_musd, "USD million")
    if normalized_econ_inputs.capex_usd_per_kwh is not None:
        add_row("Capital", "BESS CAPEX per kWh", normalized_econ_inputs.capex_usd_per_kwh, "USD/kWh")
    if normalized_econ_inputs.capex_total_usd is not None:
        add_row("Capital", "BESS CAPEX total", normalized_econ_inputs.capex_total_usd, "USD")
    add_row("Capital", "PV CAPEX", normalized_econ_inputs.pv_capex_musd, "USD million")
    if normalized_econ_inputs.total_capex_musd is not None:
        add_row("Capital", "Total CAPEX", normalized_econ_inputs.total_capex_musd, "USD million")

    add_row(
        "OPEX",
        "Fixed OPEX (% of CAPEX)",
        _percent(normalized_econ_inputs.fixed_opex_pct_of_capex),
        "% of CAPEX per year",
    )
    add_row("OPEX", "Fixed OPEX", normalized_econ_inputs.fixed_opex_musd, "USD million per year")
    if normalized_econ_inputs.opex_php_per_kwh is not None:
        add_row("OPEX", "Variable OPEX input", normalized_econ_inputs.opex_php_per_kwh, "PHP/kWh")
    if normalized_econ_inputs.variable_opex_usd_per_mwh is not None:
        add_row(
            "OPEX",
            "Variable OPEX",
            normalized_econ_inputs.variable_opex_usd_per_mwh,
            "USD/MWh",
        )
    if normalized_econ_inputs.variable_opex_schedule_usd is not None:
        schedule_text = ", ".join(str(value) for value in normalized_econ_inputs.variable_opex_schedule_usd)
        add_row("OPEX", "Variable OPEX schedule", schedule_text, "USD/MWh by year")
    if normalized_econ_inputs.periodic_variable_opex_usd is not None:
        add_row(
            "OPEX",
            "Periodic variable OPEX",
            normalized_econ_inputs.periodic_variable_opex_usd,
            "USD (lump sum)",
        )
    if normalized_econ_inputs.periodic_variable_opex_interval_years is not None:
        add_row(
            "OPEX",
            "Periodic variable OPEX interval",
            normalized_econ_inputs.periodic_variable_opex_interval_years,
            "Years",
        )
    add_row("OPEX", "Variable OPEX basis", normalized_econ_inputs.variable_opex_basis, "")

    add_row("Financing", "FX rate", normalized_econ_inputs.forex_rate_php_per_usd, "PHP per USD")
    add_row("Financing", "Inflation rate", _percent(normalized_econ_inputs.inflation_rate), "%")
    add_row("Financing", "Discount rate", _percent(normalized_econ_inputs.discount_rate), "%")
    add_row("Financing", "WACC", _percent(normalized_econ_inputs.wacc), "%")
    add_row("Financing", "Debt ratio", _percent(normalized_econ_inputs.debt_ratio), "% of total CAPEX")
    add_row("Financing", "Cost of debt", _percent(normalized_econ_inputs.cost_of_debt), "%")
    add_row("Financing", "Tenor", normalized_econ_inputs.tenor_years, "Years")
    add_row("Financing", "Include DevEx year 0", _yes_no(normalized_econ_inputs.include_devex_year0), "")
    add_row("Financing", "DevEx cost (PHP)", normalized_econ_inputs.devex_cost_php, "PHP")
    add_row(
        "Financing",
        "DevEx cost (USD)",
        normalized_econ_inputs.devex_cost_usd,
        "USD (converted)",
    )

    add_row(
        "Pricing",
        "Contract price",
        price_inputs.contract_price_usd_per_mwh,
        "USD/MWh",
    )
    add_row(
        "Pricing",
        "Escalate with inflation",
        _yes_no(price_inputs.escalate_with_inflation),
        "",
    )
    add_row(
        "Pricing",
        "Apply WESM to shortfall",
        _yes_no(price_inputs.apply_wesm_to_shortfall),
        "",
    )
    add_row("Pricing", "Sell to WESM", _yes_no(price_inputs.sell_to_wesm), "")

    return pd.DataFrame(rows)


def run_app():
    bootstrap_session_state()
    debug_mode: bool = False
    render_layout = init_page_layout(
        page_title="Simulation",
        main_title="BESS LAB — PV-only charging, AC-coupled",
        description="Configure inputs, run the simulation, and review per-year results and sensitivities.",
        base_dir=BASE_DIR,
        nav_location="sidebar",
    )
    with st.sidebar:
        st.header("Data Sources")
        st.caption(
            "Uploads are stored per-session. Use the landing page to preload data "
            "or override them below."
        )

        pv_file = st.file_uploader(
            "PV 8760 CSV (hour_index, pv_mw in MW)", type=["csv"], key="inputs_pv_upload"
        )
        cycle_file = st.file_uploader(
            "Cycle model Excel (optional override)", type=["xlsx"], key="inputs_cycle_upload"
        )
        wesm_file = st.file_uploader(
            "WESM hourly price CSV (optional; timestamp/hour_index + deficit/surplus prices)",
            type=["csv"],
            key="inputs_wesm_upload",
        )
        st.caption(
            "If no files are uploaded, built-in defaults are read from ./data/. "
            "Current session caches the latest uploads."
        )

        st.divider()
        st.subheader("Rate limit override")
        render_rate_limit_section()

    pv_df, cycle_df = load_shared_data(BASE_DIR, pv_file, cycle_file)
    pv_df, cycle_df = render_layout(pv_df, cycle_df)

    cached_cfg, _ = get_cached_simulation_config()
    preload_cfg = cached_cfg or SimConfig()

    with st.expander("Load/save inputs (JSON, includes economics)", expanded=False):
        st.caption(
            "Load a JSON payload to prefill the simulation inputs. Missing fields fall back to "
            "current defaults, so older exports remain compatible. Economics and pricing inputs "
            "are optional but preserved when provided."
        )
        uploaded_inputs = st.file_uploader(
            "Upload inputs JSON",
            type=["json"],
            accept_multiple_files=False,
            key="inputs_json_upload",
        )
        pasted_inputs = st.text_area(
            "Or paste inputs JSON",
            placeholder='{"config": {"years": 20, "initial_power_mw": 30}}',
            height=120,
            key="inputs_json_paste",
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
                        st.session_state["inputs_json_pending_payload"] = payload
                        st.success("Inputs loaded. Re-rendering with the new values.")
                        st.rerun()
                    else:
                        st.error("Expected a JSON object with simulation inputs.")
            else:
                st.info("Provide JSON content to load.", icon="ℹ️")

    pending_payload = st.session_state.pop("inputs_json_pending_payload", None)
    if isinstance(pending_payload, dict):
        _apply_inputs_payload(pending_payload, fallback_cfg=preload_cfg)

    form_result = render_simulation_form(pv_df, cycle_df)
    debug_mode = form_result.debug_mode
    cfg = form_result.config
    econ_inputs = form_result.econ_inputs
    price_inputs = form_result.price_inputs
    run_economics = form_result.run_economics
    dod_override = form_result.dod_override
    run_submitted = form_result.run_submitted
    discharge_windows_text = form_result.discharge_windows_text
    charge_windows_text = form_result.charge_windows_text

    with st.expander("Download inputs (JSON, includes economics)", expanded=False):
        inputs_payload = _build_inputs_payload(
            cfg=cfg,
            dod_override=dod_override,
            run_economics=run_economics,
            econ_inputs=econ_inputs,
            price_inputs=price_inputs,
            discharge_windows_text=discharge_windows_text,
            charge_windows_text=charge_windows_text,
        )
        inputs_json = json.dumps(inputs_payload, indent=2).encode("utf-8")
        st.download_button(
            "Download inputs (JSON)",
            data=inputs_json,
            file_name="simulation_inputs.json",
            mime="application/json",
            use_container_width=True,
        )
    inputs_fingerprint = build_inputs_fingerprint(cfg, dod_override, run_economics, econ_inputs, price_inputs)
    bootstrap_session_state(cfg, current_inputs_fingerprint=inputs_fingerprint)

    save_simulation_config(cfg, dod_override)

    cached_results = get_simulation_results()

    def render_exception_alert(message: str, exc: Exception) -> None:
        """Show a user-friendly error with optional debug details."""
        st.error(message)
        if debug_mode:
            with st.expander("Show technical details", expanded=False):
                st.exception(exc)

    if not run_submitted and cached_results is None:
        st.info("Click 'Run simulation' to generate results after updating inputs.")
        st.caption("Use the batch tools or downloads to compare multiple runs.")
        st.page_link(
            "pages/05_Multi_Scenario_Batch.py",
            label="Open Multi-scenario batch",
            help="Run a structured set of variations for side-by-side review.",
        )
        st.page_link(
            "pages/04_BESS_Sizing_Sweep.py",
            label="Open BESS sizing sweep",
            help="Rank feasible usable-energy variants using the latest inputs.",
        )
        st.stop()

    if not form_result.is_valid:
        st.error("Resolve the validation issues above before running a new simulation.")
        if form_result.validation_warnings:
            for msg in form_result.validation_warnings:
                st.warning(msg)
        if form_result.validation_details and debug_mode:
            with st.expander("Debug: validation details", expanded=False):
                st.markdown("\n".join(f"- {msg}" for msg in form_result.validation_details))
        if cached_results is None:
            return

    if run_submitted and form_result.is_valid:
        enforce_rate_limit()

        try:
            with st.spinner("Running simulation..."):
                sim_output = simulate_project(cfg, pv_df, cycle_df, dod_override)
        except ValueError as exc:  # noqa: BLE001
            render_exception_alert("Simulation failed. Please adjust inputs and try again.", exc)
            return
        except Exception as exc:  # noqa: BLE001
            render_exception_alert("Unexpected simulation error. Please retry or contact support.", exc)
            return
        else:
            st.toast("Simulation complete.")

        save_simulation_results(sim_output, dod_override)
        save_last_run_inputs_fingerprint(inputs_fingerprint)
    elif cached_results is not None:
        sim_output = cached_results.sim_output
        st.caption(
            "Showing the latest completed simulation. Click 'Run simulation' to refresh after editing inputs."
        )

    results = sim_output.results
    monthly_results_all = sim_output.monthly_results
    first_year_logs = sim_output.first_year_logs
    final_year_logs = sim_output.final_year_logs
    hourly_logs_by_year = sim_output.hourly_logs_by_year
    hod_count = sim_output.hod_count
    hod_sum_pv = sim_output.hod_sum_pv
    hod_sum_pv_resource = sim_output.hod_sum_pv_resource
    hod_sum_bess = sim_output.hod_sum_bess
    hod_sum_charge = sim_output.hod_sum_charge
    dis_hours_per_day = sim_output.discharge_hours_per_day

    # Yearly table
    res_df = pd.DataFrame([{
        'Year': r.year_index,
        'Expected firm MWh': r.expected_firm_mwh,
        'Delivered firm MWh': r.delivered_firm_mwh,
        'Shortfall MWh': r.shortfall_mwh,
        'Breach days (has any shortfall)': r.breach_days,
        'Charge MWh': r.charge_mwh,
        'Discharge MWh (from BESS)': r.discharge_mwh,
        'Available PV MWh': r.available_pv_mwh,
        'PV→Contract MWh': r.pv_to_contract_mwh,
        'BESS→Contract MWh': r.bess_to_contract_mwh,
        'Avg RTE': r.avg_rte,
        'Eq cycles (year)': r.eq_cycles,
        'Cum cycles': r.cum_cycles,
        'SOH_cycle': r.soh_cycle,
        'SOH_calendar': r.soh_calendar,
        'SOH_total': r.soh_total,
        'EOY usable MWh': r.eoy_usable_mwh,
        'EOY power MW (avail-adjusted)': r.eoy_power_mw,
        'PV curtailed MWh': r.pv_curtailed_mwh,
    } for r in results])

    monthly_df = pd.DataFrame([{
        'Year': m.year_index,
        'Month': m.month_label,
        'Expected firm MWh': m.expected_firm_mwh,
        'Delivered firm MWh': m.delivered_firm_mwh,
        'Shortfall MWh': m.shortfall_mwh,
        'Breach days (has any shortfall)': m.breach_days,
        'Charge MWh': m.charge_mwh,
        'Discharge MWh (from BESS)': m.discharge_mwh,
        'Available PV MWh': m.available_pv_mwh,
        'PV→Contract MWh': m.pv_to_contract_mwh,
        'BESS→Contract MWh': m.bess_to_contract_mwh,
        'Avg RTE': m.avg_rte,
        'Eq cycles (year)': m.eq_cycles,
        'Cum cycles': m.cum_cycles,
        'SOH_cycle': m.soh_cycle,
        'SOH_calendar': m.soh_calendar,
        'SOH_total': m.soh_total,
        'EOY usable MWh': m.eom_usable_mwh,
        'EOY power MW (avail-adjusted)': m.eom_power_mw,
        'PV curtailed MWh': m.pv_curtailed_mwh,
    } for m in monthly_results_all])
    default_df_formatters = {
        'Expected firm MWh': '{:,.1f}',
        'Delivered firm MWh': '{:,.1f}',
        'Shortfall MWh': '{:,.1f}',
        'Charge MWh': '{:,.1f}',
        'Discharge MWh (from BESS)': '{:,.1f}',
        'Available PV MWh': '{:,.1f}',
        'PV→Contract MWh': '{:,.1f}',
        'BESS→Contract MWh': '{:,.1f}',
        'Avg RTE': '{:,.3f}',
        'Eq cycles (year)': '{:,.1f}',
        'Cum cycles': '{:,.1f}',
        'SOH_cycle': '{:,.3f}',
        'SOH_calendar': '{:,.3f}',
        'SOH_total': '{:,.3f}',
        'EOY usable MWh': '{:,.1f}',
        'EOY power MW (avail-adjusted)': '{:,.1f}',
        'PV curtailed MWh': '{:,.1f}',
    }

    # --------- KPIs ---------
    final = results[-1]
    summary = summarize_simulation(sim_output)
    kpis = compute_kpis(cfg, results, summary, sim_output.augmentation_events, sim_output.augmentation_energy_added_mwh)

    econ_outputs: Optional[EconomicOutputs] = None
    cash_outputs: Optional[CashFlowOutputs] = None
    financing_outputs: Optional[FinancingOutputs] = None
    augmentation_costs_usd: Optional[List[float]] = None
    normalized_econ_inputs: Optional[EconomicInputs] = None
    annual_delivered: Optional[List[float]] = None
    annual_bess: Optional[List[float]] = None
    annual_pv_delivered: Optional[List[float]] = None
    annual_pv_excess: Optional[List[float]] = None
    annual_shortfall: Optional[List[float]] = None
    annual_total_generation: Optional[List[float]] = None
    annual_wesm_shortfall_cost_usd: Optional[List[float]] = None
    annual_wesm_surplus_revenue_usd: Optional[List[float]] = None

    if run_economics and econ_inputs and price_inputs:
        cache_latest_economics_payload(
            {
                "economic_inputs": econ_inputs,
                "price_inputs": price_inputs,
                "wesm_profile_source": wesm_file,
            }
        )
        normalized_econ_inputs = normalize_economic_inputs(econ_inputs)
        augmentation_costs_usd = estimate_augmentation_costs_by_year(
            sim_output.augmentation_energy_added_mwh,
            cfg.initial_usable_mwh,
            normalized_econ_inputs.capex_musd,
        )
        if any(augmentation_costs_usd):
            st.caption(
                "Augmentation CAPEX derived from the strategy (proportional to the share of BOL energy added)."
            )

        annual_delivered = [r.delivered_firm_mwh for r in results]
        annual_bess = [r.bess_to_contract_mwh for r in results]
        annual_pv_delivered = [
            float(delivered) - float(bess)
            for delivered, bess in zip(annual_delivered, annual_bess)
        ]
        annual_pv_excess = [r.pv_curtailed_mwh for r in results]
        annual_shortfall = [r.shortfall_mwh for r in results]
        # available_pv_mwh represents total PV generation (MWh) for variable OPEX scaling.
        annual_total_generation = [r.available_pv_mwh for r in results]

        try:
            wesm_profile_source = wesm_file
            if wesm_profile_source is None and (price_inputs.apply_wesm_to_shortfall or price_inputs.sell_to_wesm):
                default_wesm_profile = BASE_DIR / "data" / "wesm_price_profile_historical.csv"
                if default_wesm_profile.exists():
                    wesm_profile_source = str(default_wesm_profile)

            if wesm_profile_source is not None and hourly_logs_by_year:
                wesm_profile_df = read_wesm_profile(
                    [wesm_profile_source],
                    forex_rate_php_per_usd=normalized_econ_inputs.forex_rate_php_per_usd,
                )
                hourly_summary_by_year = {
                    year_index: _build_hourly_summary_df(hourly_logs_by_year[year_index])
                    for year_index in sorted(hourly_logs_by_year)
                }
                (
                    annual_wesm_shortfall_cost_usd,
                    annual_wesm_surplus_revenue_usd,
                ) = aggregate_wesm_profile_to_annual(
                    hourly_summary_by_year,
                    wesm_profile_df,
                    step_hours=cfg.step_hours,
                    apply_inflation=False,
                    inflation_rate=normalized_econ_inputs.inflation_rate,
                )

            econ_outputs = compute_lcoe_lcos_with_augmentation_fallback(
                annual_delivered,
                annual_bess,
                normalized_econ_inputs,
                augmentation_costs_usd=augmentation_costs_usd if augmentation_costs_usd else None,
                annual_total_generation_mwh=annual_total_generation,
            )
            cash_outputs = compute_cash_flows_and_irr(
                annual_delivered,
                annual_bess,
                annual_pv_excess,
                normalized_econ_inputs,
                price_inputs,
                annual_pv_delivered_mwh=annual_pv_delivered,
                annual_shortfall_mwh=annual_shortfall,
                annual_wesm_shortfall_cost_usd=annual_wesm_shortfall_cost_usd,
                annual_wesm_surplus_revenue_usd=annual_wesm_surplus_revenue_usd,
                augmentation_costs_usd=augmentation_costs_usd if augmentation_costs_usd else None,
                annual_total_generation_mwh=annual_total_generation,
            )
            financing_outputs = compute_financing_cash_flows(
                annual_delivered,
                annual_bess,
                annual_pv_excess,
                normalized_econ_inputs,
                price_inputs,
                annual_shortfall_mwh=annual_shortfall,
                annual_wesm_shortfall_cost_usd=annual_wesm_shortfall_cost_usd,
                annual_wesm_surplus_revenue_usd=annual_wesm_surplus_revenue_usd,
                augmentation_costs_usd=augmentation_costs_usd if augmentation_costs_usd else None,
                annual_total_generation_mwh=annual_total_generation,
            )
        except ValueError as exc:  # noqa: BLE001
            render_exception_alert("Economics inputs are invalid. Please review the assumptions.", exc)
            return
        except Exception as exc:  # noqa: BLE001
            render_exception_alert("Unexpected error while computing economics. Please retry.", exc)
            return

    save_simulation_snapshot({
        "Contracted MW": cfg.contracted_mw,
        "Power (BOL MW)": cfg.initial_power_mw,
        "Usable (BOL MWh)": cfg.initial_usable_mwh,
        "Discharge windows": discharge_windows_text,
        "Charge windows": charge_windows_text if charge_windows_text else "Any PV hour",
        "Compliance (%)": kpis.compliance,
        "BESS share of firm (%)": kpis.bess_share_of_firm,
        "Charge/Discharge ratio": kpis.charge_discharge_ratio,
        "PV capture ratio": kpis.pv_capture_ratio,
        "Total project generation (MWh)": kpis.total_project_generation_mwh,
        "BESS share of generation (MWh)": kpis.bess_generation_mwh,
        "PV share of generation (MWh)": kpis.pv_generation_mwh,
        "PV excess (MWh)": kpis.pv_excess_mwh,
        "BESS losses (MWh)": kpis.bess_losses_mwh,
        "Final EOY usable (MWh)": final.eoy_usable_mwh,
        "Final EOY power (MW)": final.eoy_power_mw,
        "Final eq cycles (year)": final.eq_cycles,
        "Final SOH_total": final.soh_total,
    })

    render_primary_metrics(cfg, kpis)
    with st.expander("Sensitivity tornado (optional)", expanded=False):
        metric_options = {
            "% Compliance": "compliance_pct",
            "% Surplus": "surplus_pct",
            "% PIRR": "pirr_pct",
        }
        selected_metric_label = st.selectbox(
            "Metric",
            list(metric_options.keys()),
            key="simple_tornado_metric_select",
        )
        metric_key = metric_options[selected_metric_label]

        default_table = build_simple_lever_table()
        if "simple_tornado_table" not in st.session_state or not isinstance(
            st.session_state.get("simple_tornado_table"), pd.DataFrame
        ):
            st.session_state["simple_tornado_table"] = default_table.copy()
        lever_table = st.session_state["simple_tornado_table"]
        edited_table = st.data_editor(
            lever_table,
            num_rows="fixed",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Lever": st.column_config.TextColumn("Lever", disabled=True),
                "Low change": st.column_config.NumberColumn("Low change", step=0.1, format="%.2f"),
                "High change": st.column_config.NumberColumn("High change", step=0.1, format="%.2f"),
                "Low impact (pp)": st.column_config.NumberColumn(
                    "Low impact (pp)",
                    step=0.1,
                    format="%.2f",
                ),
                "High impact (pp)": st.column_config.NumberColumn(
                    "High impact (pp)",
                    step=0.1,
                    format="%.2f",
                ),
                "Notes": st.column_config.TextColumn("Notes", disabled=True),
            },
        )
        st.session_state["simple_tornado_table"] = edited_table
        run_tornado = st.button("Run tornado", use_container_width=True)
        st.caption(
            "CAPEX/OPEX/Tariff only shift PIRR because the finance model adjusts cash flows, "
            "while compliance/surplus are derived from operational energy outputs."
        )

        if run_tornado:
            baseline_sim_output = sim_output
            wesm_profile_source: Optional[str] = None
            if price_inputs and (price_inputs.apply_wesm_to_shortfall or price_inputs.sell_to_wesm):
                wesm_profile_source = wesm_file
                if wesm_profile_source is None:
                    default_wesm_profile = BASE_DIR / "data" / "wesm_price_profile_historical.csv"
                    if default_wesm_profile.exists():
                        wesm_profile_source = str(default_wesm_profile)

            need_logs = bool(
                metric_key == "pirr_pct"
                and price_inputs is not None
                and (price_inputs.apply_wesm_to_shortfall or price_inputs.sell_to_wesm)
                and wesm_profile_source
            )
            if need_logs and not baseline_sim_output.hourly_logs_by_year:
                baseline_sim_output = simulate_project(
                    cfg,
                    pv_df=pv_df,
                    cycle_df=cycle_df,
                    dod_override=dod_override,
                    need_logs=True,
                )

            updated_table, warnings = _run_simple_tornado(
                base_cfg=cfg,
                pv_df=pv_df,
                cycle_df=cycle_df,
                metric_key=metric_key,
                lever_table=edited_table,
                baseline_sim_output=baseline_sim_output,
                dod_override=dod_override,
                econ_inputs=econ_inputs,
                price_inputs=price_inputs,
                wesm_profile_source=wesm_profile_source,
            )
            st.session_state["simple_tornado_table"] = updated_table

            if warnings:
                st.warning("\n".join(f"• {warning}" for warning in warnings))
            else:
                st.success("Tornado impacts updated.")

        chart_source = prepare_tornado_data(st.session_state["simple_tornado_table"])
        if chart_source["Impact (pp)"].abs().sum() == 0:
            st.info(
                "All impacts are currently zero. Enter low/high changes and run tornado to populate the chart.",
                icon="ℹ️",
            )
        st.altair_chart(build_tornado_chart(chart_source), use_container_width=True)
    optional_diagnostics = st.toggle(
        "Optional diagnostics",
        value=False,
        help="Show advanced flags, design guidance, and SOC/dispatch diagnostics.",
    )

    if run_economics and econ_outputs and cash_outputs and financing_outputs:
        st.markdown("### Economics summary")
        forex_rate_php_per_usd = (
            normalized_econ_inputs.forex_rate_php_per_usd if normalized_econ_inputs else 58.0
        )
        php_per_kwh_factor = forex_rate_php_per_usd / 1000.0
        lcoe_php_per_kwh = econ_outputs.lcoe_usd_per_mwh * php_per_kwh_factor
        lcos_php_per_kwh = econ_outputs.lcos_usd_per_mwh * php_per_kwh_factor
        econ_specs = [
            MetricSpec(
                label="Discounted costs (USD million)",
                value=f"{econ_outputs.discounted_costs_usd / 1_000_000:,.2f}",
                help="CAPEX at year 0 plus discounted OPEX and augmentation across the project horizon.",
            ),
            MetricSpec(
                label="LCOE (PHP/kWh delivered)",
                value=f"{lcoe_php_per_kwh:,.2f}",
                help=(
                    "Total discounted costs ÷ discounted firm energy delivered, converted using "
                    f"PHP {forex_rate_php_per_usd:,.0f}/USD."
                ),
            ),
            MetricSpec(
                label="LCOS (PHP/kWh from BESS)",
                value=f"{lcos_php_per_kwh:,.2f}",
                help=(
                    "Same cost base divided by discounted BESS contribution only, converted with the "
                    f"PHP {forex_rate_php_per_usd:,.0f}/USD rate."
                ),
            ),
        ]
        render_metrics(st.columns(3), econ_specs)

        if normalized_econ_inputs:
            bess_capex_musd = normalized_econ_inputs.capex_musd
            pv_capex_musd = normalized_econ_inputs.pv_capex_musd
            total_capex_musd = (
                normalized_econ_inputs.total_capex_musd
                if normalized_econ_inputs.total_capex_musd is not None
                else bess_capex_musd + pv_capex_musd
            )
            st.caption(
                "CAPEX breakdown: "
                f"BESS ${bess_capex_musd:,.2f}M + PV ${pv_capex_musd:,.2f}M "
                f"= ${total_capex_musd:,.2f}M."
            )

        if normalized_econ_inputs and normalized_econ_inputs.include_devex_year0:
            st.caption(
                "DevEx: Included an additional "
                f"₱{normalized_econ_inputs.devex_cost_php / 1_000_000:,.0f}M "
                f"(≈${normalized_econ_inputs.devex_cost_usd / 1_000_000:,.2f}M) at year 0 across discounted costs, "
                "LCOE/LCOS, NPV, and IRR."
            )
        else:
            st.caption("DevEx not included; upfront spend reflects CAPEX only.")

        use_wesm_shortfall_profile = annual_wesm_shortfall_cost_usd is not None
        use_wesm_surplus_profile = annual_wesm_surplus_revenue_usd is not None
        wesm_profile_active = use_wesm_shortfall_profile or use_wesm_surplus_profile
        revenue_help = "Contract revenue from delivered energy plus optional WESM adjustments."
        if price_inputs.apply_wesm_to_shortfall:
            if wesm_profile_active:
                revenue_help += " Shortfall MWh are deducted using the hourly WESM profile."
            if price_inputs.sell_to_wesm:
                if wesm_profile_active:
                    revenue_help += (
                        " PV surplus is credited using the hourly WESM profile; otherwise surplus is excluded from revenue."
                    )
        cash_specs = [
            MetricSpec(
                label="Discounted revenues (USD million)",
                value=f"{cash_outputs.discounted_revenues_usd / 1_000_000:,.2f}",
                help=revenue_help,
            ),
            MetricSpec(
                label="Project NPV (USD million, WACC)",
                value=f"{financing_outputs.project_npv_usd / 1_000_000:,.2f}",
                help="Discounted project cash flows using WACC (year 0 CAPEX included).",
            ),
            MetricSpec(
                label="PIRR (%)",
                value=f"{financing_outputs.project_irr_pct:,.2f}%"
                if financing_outputs.project_irr_pct == financing_outputs.project_irr_pct
                else "—",
                help="Project IRR computed from operating cash flows and augmentation outflows.",
            ),
        ]
        render_metrics(st.columns(3), cash_specs)

        financing_specs = [
            MetricSpec(
                label="EBITDA (USD million)",
                value=f"{financing_outputs.ebitda_usd / 1_000_000:,.2f}",
                help="Total EBITDA over the project horizon (revenues minus operating OPEX).",
            ),
            MetricSpec(
                label="EBITDA margin (%)",
                value=f"{financing_outputs.ebitda_margin * 100.0:,.2f}%"
                if financing_outputs.ebitda_margin == financing_outputs.ebitda_margin
                else "—",
                help="Total EBITDA divided by total revenue across the project horizon.",
            ),
            MetricSpec(
                label="EIRR (%)",
                value=f"{financing_outputs.equity_irr_pct:,.2f}%"
                if financing_outputs.equity_irr_pct == financing_outputs.equity_irr_pct
                else "—",
                help="Equity IRR after debt service and equity contributions.",
            ),
        ]
        render_metrics(st.columns(3), financing_specs)

        wesm_caption = (
            "WESM pricing disabled; contract shortfalls are not monetized in revenues, NPV, or IRR."
        )
        if price_inputs.apply_wesm_to_shortfall and wesm_profile_active:
            wesm_impact_musd = cash_outputs.discounted_wesm_value_usd / 1_000_000
            surplus_note = (
                " PV surplus credited using the hourly WESM profile due to the sell toggle."
                if price_inputs.sell_to_wesm
                else " PV surplus excluded from revenue while WESM pricing is active."
            )
            wesm_caption = (
                "WESM shortfall costs derived from the hourly profile."
                f" Discounted WESM impact on revenues/NPV/IRR: {wesm_impact_musd:,.2f} USD million."
                + surplus_note
            )

        st.caption(wesm_caption)

    st.markdown("#### Augmentation impact trace")
    augmentation_retired = getattr(
        sim_output, "augmentation_retired_energy_mwh", [0.0 for _ in sim_output.augmentation_energy_added_mwh]
    )
    if len(augmentation_retired) < len(sim_output.augmentation_energy_added_mwh):
        augmentation_retired.extend([0.0] * (len(sim_output.augmentation_energy_added_mwh) - len(augmentation_retired)))
    elif len(augmentation_retired) > len(sim_output.augmentation_energy_added_mwh):
        augmentation_retired = augmentation_retired[: len(sim_output.augmentation_energy_added_mwh)]

    coverage_pct_by_year = [
        (r.delivered_firm_mwh / r.expected_firm_mwh * 100.0) if r.expected_firm_mwh > 0 else float('nan')
        for r in results
    ]
    delivered_by_year = [r.delivered_firm_mwh for r in results]

    aug_rows: List[Dict[str, Any]] = []
    for idx, add_e in enumerate(sim_output.augmentation_energy_added_mwh):
        retired_e = augmentation_retired[idx] if idx < len(augmentation_retired) else 0.0
        if add_e <= 0.0 and retired_e <= 0.0:
            continue

        coverage_pct = coverage_pct_by_year[idx]
        coverage_delta = coverage_pct_by_year[idx] - coverage_pct_by_year[idx - 1] if idx > 0 else float('nan')
        gen_delta = delivered_by_year[idx] - delivered_by_year[idx - 1] if idx > 0 else float('nan')
        add_pct_bol = (add_e / cfg.initial_usable_mwh * 100.0) if cfg.initial_usable_mwh > 0 else float('nan')

        aug_rows.append({
            "Year": idx + 1,
            "Added (MWh BOL)": add_e,
            "Added vs BOL (%)": add_pct_bol,
            "Retired cohorts (MWh BOL)": retired_e,
            "Coverage (%)": coverage_pct,
            "Coverage Δ (pp)": coverage_delta,
            "Generation Δ (MWh)": gen_delta,
        })

    if aug_rows:
        aug_df = pd.DataFrame(aug_rows)
        st.caption("Per-event summary combines augmentation size, cohort retirements, and year-over-year shifts in compliance and delivered energy.")
        augmentation_formatters = {
            "Added (MWh BOL)": "{:.1f}",
            "Added vs BOL (%)": "{:.2f}",
            "Retired cohorts (MWh BOL)": "{:.1f}",
            "Coverage (%)": "{:.2f}",
            "Coverage Δ (pp)": "{:.2f}",
            "Generation Δ (MWh)": "{:.1f}",
        }
        render_formatted_dataframe(aug_df, augmentation_formatters)

        delta_df = aug_df.melt(
            id_vars=["Year"],
            value_vars=["Coverage Δ (pp)", "Generation Δ (MWh)"],
            var_name="Metric",
            value_name="Delta",
        ).dropna(subset=["Delta"])

        if not delta_df.empty:
            delta_chart = alt.Chart(delta_df).mark_bar().encode(
                x=alt.X("Year:O", title="Augmentation year"),
                y=alt.Y("Delta:Q", title="Annual change"),
                color=alt.Color("Metric:N", title=""),
                tooltip=["Year", "Metric", alt.Tooltip("Delta:Q", format=".2f")],
            ).properties(title="Year-over-year shifts at augmentation points")
            st.altair_chart(delta_chart, use_container_width=True)
    else:
        st.info("No augmentation or retirement events were triggered in this run.")


    st.markdown("---")
    st.subheader("Yearly Summary")
    render_formatted_dataframe(res_df, default_df_formatters)

    with st.expander("Monthly summary preview", expanded=False):
        render_formatted_dataframe(monthly_df, default_df_formatters)

    # ---------- EOY Delivered Firm Split (per day): PV vs BESS ----------
    st.subheader("EOY Delivered Firm Split (per day) — PV vs BESS")
    target_daily_mwh = cfg.contracted_mw * dis_hours_per_day
    years_list = [r.year_index for r in results]
    deliv_df = pd.DataFrame({
        'Year': years_list,
        'PV→Contract (MWh/day)': [r.pv_to_contract_mwh/365.0 for r in results],
        'BESS→Contract (MWh/day)': [r.bess_to_contract_mwh/365.0 for r in results],
        'Target firm (MWh/day)': [target_daily_mwh]*len(years_list),
    })
    deliv_long = deliv_df.melt(id_vars='Year', value_vars=['PV→Contract (MWh/day)', 'BESS→Contract (MWh/day)'],
                               var_name='Source', value_name='MWh/day')

    bar2 = alt.Chart(deliv_long).mark_bar().encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('MWh/day:Q', title='MWh/day'),
        color=alt.Color('Source:N', scale=alt.Scale(range=['#86c5da', '#7fd18b']))
    )
    line2 = alt.Chart(deliv_df).mark_line(point=True, color='#f2a900').encode(
        x='Year:O',
        y='Target firm (MWh/day):Q',
    )
    st.altair_chart(bar2 + line2, use_container_width=True)

    if optional_diagnostics:
        # ---------- Flags ----------
        st.subheader("Flags & Guidance")
        flag_totals = {
            'firm_shortfall_hours': sum(r.flags['firm_shortfall_hours'] for r in results),
            'soc_floor_hits': sum(r.flags['soc_floor_hits'] for r in results),
            'soc_ceiling_hits': sum(r.flags['soc_ceiling_hits'] for r in results),
        }
        flag_specs = []
        for key in ["firm_shortfall_hours", "soc_floor_hits", "soc_ceiling_hits"]:
            meta = FLAG_DEFINITIONS[key]
            flag_specs.append(
                MetricSpec(
                    label=meta["label"],
                    value=f"{flag_totals[key]:,}",
                    caption=f"Meaning: {meta['meaning']}\nFix knobs: {meta['knobs']}",
                )
            )
        render_metrics(st.columns(len(flag_specs)), flag_specs)

        insights = build_flag_insights(flag_totals)
        st.markdown("**What the flags suggest:**")
        st.markdown("\n".join(f"- {tip}" for tip in insights))

        st.markdown("---")

        # ---------- Design Advisor (physics-bounded) ----------
        st.subheader("Design Advisor (final-year, physics-bounded)")

        # --- Bounds / guardrails (editable if you like) ---
        RTE_RT_MAX = 0.92              # plausible AC-to-AC roundtrip limit
        SOC_FLOOR_MIN = 0.05           # don't recommend below this
        SOC_CEILING_MAX = 0.98         # don't recommend above this
        DELTA_SOC_MAX = 0.90           # ~5-95%
        EFC_YR_GREEN = 300.0           # vendor guardrail
        EFC_YR_YELLOW = 400.0

        # --- Final-year context ---
        eta_ch_now, eta_dis_now, eta_rt_now = resolve_efficiencies(cfg)
        eta_ratio = eta_dis_now / max(1e-9, eta_ch_now)
        delta_soc_now = max(0.0, cfg.soc_ceiling - cfg.soc_floor)
        delta_soc_cap = min(DELTA_SOC_MAX, SOC_CEILING_MAX - SOC_FLOOR_MIN)
        soh_final = float(final.soh_total)

        target_day = cfg.contracted_mw * dis_hours_per_day                    # MWh/day
        pv_to_contract_day = final.pv_to_contract_mwh / 365.0                 # MWh/day
        bess_share_day = max(0.0, target_day - pv_to_contract_day)            # MWh/day BESS must supply

        deliverable_day_now = cfg.initial_usable_mwh * soh_final * delta_soc_now * eta_dis_now
        shortfall_day_now = max(0.0, target_day - deliverable_day_now)

        colA, colB, colC, colD = st.columns(4)
        colA.metric("Deliverable/day now (final yr)", f"{deliverable_day_now:,.1f} MWh")
        colB.metric("Shortfall/day (final yr)", f"{shortfall_day_now:,.1f} MWh")
        colC.metric("Target/day", f"{target_day:,.1f} MWh")
        colD.metric("EOY power avail (final)", f"{final.eoy_power_mw:,.2f} MW",
                    help="Availability-adjusted final-year power capability.")

        # Quick severity read: give a nudge before diving into options
        gap_ratio = shortfall_day_now / target_day if target_day else 0.0
        if shortfall_day_now <= 0.05:  # effectively on target
            st.success("Final-year deliverable meets the target within rounding noise.")
        elif gap_ratio <= 0.10:
            st.info("Minor gap: adjust one knob below to clear a small shortfall.")
        else:
            st.warning("Material gap: use the action ladder below to close the deficit.")

        # --- 1) Power vs Energy limiter ---
        if final.eoy_power_mw + 1e-9 < cfg.contracted_mw:
            st.error(
                f"Power-limited: final-year available power {final.eoy_power_mw:.2f} MW "
                f"is below contract {cfg.contracted_mw:.2f} MW."
            )
            need = cfg.contracted_mw - final.eoy_power_mw
            st.markdown(
                f"- **Option D (Power)**: Increase power (final-year, avail-adjusted) by **{need:.2f} MW**, "
                f"or reduce contract MW / shift windows."
            )
        else:
            # --- Energy-limited path ---
            st.caption("Energy-limited in final year (power is sufficient).")

            # --- 2) Sequential bounded solve: ΔSOC → RTE → Energy ---
            # a) try to meet target with ΔSOC first (bounded)
            req_delta_soc_at_current = target_day / max(1e-9, cfg.initial_usable_mwh * soh_final * eta_dis_now)
            delta_soc_adopt = min(delta_soc_cap, max(delta_soc_now, req_delta_soc_at_current))

            # b) then RTE (bounded)
            req_eta_dis_at_soc = min(0.9999, max(0.0, target_day / max(1e-9, cfg.initial_usable_mwh * soh_final * delta_soc_adopt)))
            req_rte_rt_at_soc = min(0.9999, max(0.0, (req_eta_dis_at_soc ** 2) / max(1e-9, eta_ratio)))
            rte_rt_adopt = min(RTE_RT_MAX, max(eta_rt_now, req_rte_rt_at_soc))
            eta_dis_adopt = min(0.9999, math.sqrt(rte_rt_adopt * eta_ratio))

            # c) finally BOL energy to close any remaining gap
            ebol_req = target_day / max(1e-9, soh_final * delta_soc_adopt * eta_dis_adopt)
            ebol_delta = max(0.0, ebol_req - cfg.initial_usable_mwh)

            # Helper to render SOC variant text (raise ceiling vs lower floor)
            def soc_variant_text(delta_soc_goal: float) -> str:
                # choice A: keep floor, raise ceiling
                ceil_needed = min(SOC_CEILING_MAX, cfg.soc_floor + delta_soc_goal)
                # choice B: keep ceiling, lower floor
                floor_needed = max(SOC_FLOOR_MIN, cfg.soc_ceiling - delta_soc_goal)
                return (f"(e.g., keep floor at {cfg.soc_floor*100:.0f}% → raise ceiling to **{ceil_needed*100:.0f}%**, "
                        f"or keep ceiling at {cfg.soc_ceiling*100:.0f}% → lower floor to **{floor_needed*100:.0f}%**).")

            # How far each knob alone would push deliverable/day
            deliverable_soc_only = cfg.initial_usable_mwh * soh_final * delta_soc_adopt * eta_dis_now
            deliverable_soc_rte = cfg.initial_usable_mwh * soh_final * delta_soc_adopt * eta_dis_adopt
            deliverable_full = ebol_req * soh_final * delta_soc_adopt * eta_dis_adopt

            # --- 3) PV charge sufficiency check under the adopted RTE ---
            pv_charge_req_day = bess_share_day / max(1e-9, rte_rt_adopt)   # MWh/day needed from PV to charge
            charged_day_now = final.charge_mwh / 365.0                     # MWh/day currently charged
            charge_deficit_day = max(0.0, pv_charge_req_day - charged_day_now)
            extra_charge_hours_day = charge_deficit_day / max(1e-9, final.eoy_power_mw)

            # --- 4) Implied cycles guardrail under the proposed ΔSOC/Ebol ---
            def dod_from_delta_soc(ds: float) -> int:
                return 100 if ds >= 0.90 else (80 if ds >= 0.80 else (40 if ds >= 0.40 else (20 if ds >= 0.20 else 10)))
            dod_key_prop = dod_from_delta_soc(delta_soc_adopt)
            dod_frac_map = {10:0.10,20:0.20,40:0.40,80:0.80,100:1.00}
            dod_frac_prop = dod_frac_map[dod_key_prop]
            efc_year_prop = (bess_share_day * 365.0) / max(1e-9, ebol_req * dod_frac_prop)

            # --- 5) Print bounded options ---
            opts = []

            # OPTION A — ΔSOC only (bounded to cap); if still short, explain why it's insufficient alone
            if delta_soc_now + 1e-9 < delta_soc_cap:
                need_soc = max(0.0, delta_soc_adopt - delta_soc_now) * 100.0
                # re-compute Ebol needed if we keep RTE at current (ΔSOC only)
                ebol_req_soc_only = target_day / max(1e-9, soh_final * delta_soc_adopt * eta_dis_now)
                short_if_only_soc = max(0.0, ebol_req_soc_only - cfg.initial_usable_mwh)
                if short_if_only_soc <= 1e-6:
                    opts.append(f"- **Option A (ΔSOC)**: Widen ΔSOC to **{delta_soc_adopt*100:,.1f}%** {soc_variant_text(delta_soc_adopt)}")
                else:
                    opts.append(f"- **Option A (ΔSOC)**: Widen ΔSOC to **{delta_soc_adopt*100:,.1f}%** {soc_variant_text(delta_soc_adopt)} "
                                f"→ still short on energy by **{short_if_only_soc:,.1f} MWh** (at current RTE).")
            else:
                opts.append(f"- **Option A (ΔSOC)**: Already at cap (**{delta_soc_now*100:,.1f}%**).")

            # OPTION B — ΔSOC (adopted) + RTE (bounded)
            if rte_rt_adopt > eta_rt_now + 1e-9:
                opts.append(f"- **Option B (ΔSOC + RTE)**: Keep ΔSOC at **{delta_soc_adopt*100:,.1f}%** and improve roundtrip RTE to "
                            f"**{rte_rt_adopt*100:,.1f}%** (cap {RTE_RT_MAX*100:.0f}%).")
            else:
                opts.append(f"- **Option B (ΔSOC + RTE)**: RTE already at limit for this option (current {eta_rt_now*100:.1f}%, cap {RTE_RT_MAX*100:.0f}%).")

            # OPTION C — Energy/contract levers when ΔSOC + RTE still fall short
            if ebol_delta > 1e-6:
                contract_with_current_energy = deliverable_soc_rte / max(1e-9, dis_hours_per_day)
                opts.append(
                    f"- **Option C (Energy/contract)**: Need ~+{ebol_delta:,.1f} MWh usable (to {ebol_req:,.1f} MWh) to hit the full target. "
                    f"If adding that at BOL is impractical, consider **staged Threshold/SOH augmentation** or **right-size the contract to ~{contract_with_current_energy:,.2f} MW** under the proposed ΔSOC/RTE."
                )
            else:
                opts.append(f"- **Option C (Energy/contract)**: BOL usable is sufficient under the adopted ΔSOC/RTE.")

            st.markdown("**Bounded recommendations:**")
            st.markdown("\n".join(opts))

            # --- 5b) Action ladder (fastest wins first) ---
            action_ladder: List[str] = []
            if delta_soc_adopt > delta_soc_now + 1e-9:
                action_ladder.append(
                    f"**Widen ΔSOC** to **{delta_soc_adopt*100:,.1f}%** → delivers ~{deliverable_soc_only:,.1f} MWh/day."
                )
            if rte_rt_adopt > eta_rt_now + 1e-9:
                action_ladder.append(
                    f"**Improve roundtrip RTE** to **{rte_rt_adopt*100:,.1f}%** → delivers ~{deliverable_soc_rte:,.1f} MWh/day."
                )
            if ebol_delta > 1e-6:
                contract_with_current_energy = deliverable_soc_rte / max(1e-9, dis_hours_per_day)
                action_ladder.append(
                    f"**Close remaining energy gap**: either plan staged augmentation (~+{ebol_delta:,.1f} MWh usable over life) or resize contract toward **{contract_with_current_energy:,.2f} MW** so ΔSOC/RTE improvements can carry the final year."
                )
            if charge_deficit_day > 1e-3:
                action_ladder.append(
                    f"**Widen charge window** by **+{extra_charge_hours_day:,.2f} h/day** or create shoulder headroom to absorb PV."
                )
            if not action_ladder:
                action_ladder.append("All knobs already at bounds for the target—consider reducing the contract or shifting delivery windows.")

            st.markdown("**Action ladder (work down the list):**")
            st.markdown("\n".join(f"- {idx}) {item}" for idx, item in enumerate(action_ladder, start=1)))

            # --- 6) PV charge sufficiency + charge-hours hint ---
            st.caption(
                f"PV charge required/day for BESS share ≈ **{pv_charge_req_day:,.1f} MWh** "
                f"(BESS share {bess_share_day:,.1f} ÷ RTE {rte_rt_adopt:.2f}). "
                f"Currently charging **{charged_day_now:,.1f} MWh/day**."
            )
            if charge_deficit_day > 1e-3:
                st.warning(
                    f"PV charge **insufficient** by **{charge_deficit_day:,.1f} MWh/day** in final year. "
                    f"At {final.eoy_power_mw:.1f} MW charge power, this needs **+{extra_charge_hours_day:,.2f} h/day** "
                    f"of charge window or equivalent **shoulder discharge** to create headroom while PV is up."
                )
            else:
                st.success("PV charge looks sufficient at the proposed settings.")

            # --- 7) Implied EFC guardrail
            if efc_year_prop > EFC_YR_YELLOW:
                st.error(
                    f"Implied **EqCycles/yr ≈ {efc_year_prop:,.0f}** (ΔSOC bucket {dod_key_prop}%): exceeds typical guardrails. "
                    "Prefer **augmentation** (Threshold/SOH) or reduce ΔSOC."
                )
            elif efc_year_prop > EFC_YR_GREEN:
                st.warning(
                    f"Implied cycles ≈ **{efc_year_prop:,.0f} EFC/yr** at proposed settings; check warranty guardrails "
                    f"(soft limit {EFC_YR_GREEN:.0f}, hard {EFC_YR_YELLOW:.0f})."
                )
            else:
                st.caption(
                    f"Implied cycles ≈ {efc_year_prop:,.0f} EFC/yr at the proposed ΔSOC bucket ({dod_key_prop}%)."
                )

    st.markdown("---")

    # ---------- Average Daily Profile ----------
    st.subheader("Average Daily Profile — PV & BESS contributions to contract; charging shown below zero")

    avg_profiles = build_avg_profile_bundle(
        cfg,
        first_year_logs,
        final_year_logs,
        hod_count,
        hod_sum_pv_resource,
        hod_sum_pv,
        hod_sum_bess,
        hod_sum_charge,
    )

    if avg_profiles.final_year is not None and avg_profiles.first_year is not None and avg_profiles.project is not None:
        tab_final, tab_first, tab_project = st.tabs(["Final year", "Year 1", "Average across project"])
        with tab_final:
            st.altair_chart(build_avg_profile_chart(avg_profiles.final_year), use_container_width=True)
        with tab_first:
            st.altair_chart(build_avg_profile_chart(avg_profiles.first_year), use_container_width=True)
        with tab_project:
            st.altair_chart(build_avg_profile_chart(avg_profiles.project), use_container_width=True)
        st.caption(
            "Stacked bars (narrow width with soft fill): PV→Contract (blue) + BESS→Contract (green) fill the contract box "
            "(gold). Negative area: BESS charging (purple). PV surplus/curtailment shown in light red. PV resource overlay "
            "(tan, dashed outline). Contract step shown with gold outline."
        )
    else:
        st.info("Average daily profiles unavailable — simulation logs not generated.")

    if optional_diagnostics:
        st.markdown("### SOC & dispatch diagnostics")
        diag_logs: Dict[str, HourlyLog] = {}
        if first_year_logs is not None:
            diag_logs["Year 1 (initial)"] = first_year_logs
        if final_year_logs is not None:
            diag_logs[f"Year {cfg.years} (final)"] = final_year_logs

        if diag_logs:
            diag_default = list(diag_logs.keys()).index(f"Year {cfg.years} (final)") if final_year_logs is not None else 0
            selected_label = st.radio(
                "Select which year to visualize",
                options=list(diag_logs.keys()),
                index=diag_default,
                help="Toggle between the first-year baseline and final-year (with degradation/augmentation) logs.",
            )
            selected_logs = diag_logs[selected_label]

            heatmap_bin_hours = st.select_slider(
                "Heatmap resolution",
                options=[1, 2, 3],
                value=1,
                format_func=lambda h: f"{h}-hour bands",
                help="Downsample the heatmap horizontally to shrink the data payload if rendering feels sluggish.",
            )

            st.caption(
                "Heatmap: dark troughs near the SOC floor overnight hint at reliability risk; saturated midday bands near 100% "
                "indicate PV clipping/curtailment. Use the resolution control if the view feels heavy."
            )
            heatmap_pivot = prepare_soc_heatmap_data(selected_logs, cfg.initial_usable_mwh)
            heatmap_source = (
                heatmap_pivot.reset_index()
                .melt(id_vars="day_of_year", var_name="hour", value_name="soc_frac")
                .dropna(subset=["soc_frac"])
            )
            heatmap_source["hour_bin"] = (heatmap_source["hour"].astype(int) // heatmap_bin_hours) * heatmap_bin_hours
            if heatmap_bin_hours > 1:
                heatmap_source = (
                    heatmap_source
                    .groupby(["day_of_year", "hour_bin"], as_index=False)["soc_frac"]
                    .mean()
                    .rename(columns={"hour_bin": "hour"})
                )
            heatmap_source["hour_label"] = heatmap_source["hour"].astype(int).apply(
                lambda h: (
                    f"{h:02d}:00–{(h + heatmap_bin_hours) % 24:02d}:00"
                    if heatmap_bin_hours > 1
                    else f"{h:02d}:00"
                )
            )
            heatmap_source["soc_pct"] = heatmap_source["soc_frac"] * 100.0
            heatmap_source["diagnostic_tip"] = (
                "Low overnight SOC → reliability risk. Flat mid-day SOC near 100% → PV clipping/curtailment headroom."
            )
            if heatmap_source.empty:
                st.info("SOC heatmap unavailable — simulation logs were empty for this scenario.")
            else:
                axis_x = alt.Axis(values=list(range(0, 24, max(heatmap_bin_hours, 3))))
                heatmap_chart = (
                    alt.Chart(heatmap_source)
                    .mark_rect()
                    .encode(
                        x=alt.X("hour:O", title="Hour of day", axis=axis_x),
                        y=alt.Y("day_of_year:O", title="Day of year"),
                        color=alt.Color(
                            "soc_pct:Q",
                            title="SOC (%) of initial usable energy",
                            scale=alt.Scale(scheme="turbo", domain=[0, 100]),
                        ),
                        tooltip=[
                            alt.Tooltip("day_of_year:O", title="Day"),
                            alt.Tooltip("hour_label:N", title="Hour"),
                            alt.Tooltip("soc_pct:Q", title="SOC (%)", format=".1f"),
                            alt.Tooltip("diagnostic_tip:N", title="Reading tip"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(heatmap_chart, use_container_width=True)

            envelope_df = prepare_charge_discharge_envelope(selected_logs)
            st.caption(
                "Charge/discharge envelope shows the median and 5–95% range by hour; widening charge bands or deep discharge "
                "tails highlight operational stress that may erode reliability."
            )
            axis_x = alt.Axis(values=list(range(0, 24, 2)))
            envelope_chart = alt.layer(
                alt.Chart(envelope_df)
                .mark_area(opacity=0.25, color="#caa6ff")
                .encode(
                    x=alt.X("hour:O", title="Hour of day", axis=axis_x),
                    y=alt.Y("charge_low:Q", title="MW"),
                    y2="charge_high:Q",
                    tooltip=[
                        alt.Tooltip("hour:O", title="Hour"),
                        alt.Tooltip("charge_p05:Q", title="Charge p05 (MW)", format=".2f"),
                        alt.Tooltip("charge_p50:Q", title="Charge median (MW)", format=".2f"),
                        alt.Tooltip("charge_p95:Q", title="Charge p95 (MW)", format=".2f"),
                    ],
                ),
                alt.Chart(envelope_df)
                .mark_area(opacity=0.25, color="#7fd18b")
                .encode(
                    x=alt.X("hour:O", title="Hour of day", axis=axis_x),
                    y=alt.Y("discharge_low:Q", title="MW"),
                    y2="discharge_high:Q",
                    tooltip=[
                        alt.Tooltip("hour:O", title="Hour"),
                        alt.Tooltip("discharge_p05:Q", title="Discharge p05 (MW)", format=".2f"),
                        alt.Tooltip("discharge_p50:Q", title="Discharge median (MW)", format=".2f"),
                        alt.Tooltip("discharge_p95:Q", title="Discharge p95 (MW)", format=".2f"),
                    ],
                ),
                alt.Chart(envelope_df)
                .mark_line(color="#7d5ba6", strokeWidth=2)
                .encode(x=alt.X("hour:O", axis=axis_x), y=alt.Y("charge_median_neg:Q", title="MW")),
                alt.Chart(envelope_df)
                .mark_line(color="#2e7b53", strokeWidth=2)
                .encode(x=alt.X("hour:O", axis=axis_x), y=alt.Y("discharge_p50:Q", title="MW")),
            ).properties(height=300)
            st.altair_chart(envelope_chart, use_container_width=True)
        else:
            st.info("SOC heatmap and charge/discharge envelope are hidden because simulation logs are unavailable.")

    st.markdown("---")

    # ---------- Downloads ----------
    st.subheader("Downloads")
    cfg_download = json.dumps(asdict(cfg), indent=2)
    st.download_button(
        "Download simulation config (JSON)",
        cfg_download.encode("utf-8"),
        file_name="bess_config.json",
        mime="application/json",
    )
    cash_flow_builders_available = (
        hasattr(economics, "build_operating_cash_flow_table")
        and hasattr(economics, "build_financing_cash_flow_table")
    )
    if (
        run_economics
        and cash_outputs
        and financing_outputs
        and normalized_econ_inputs
        and price_inputs
        and annual_delivered is not None
        and annual_bess is not None
        and annual_pv_delivered is not None
        and annual_pv_excess is not None
        and annual_shortfall is not None
        and annual_total_generation is not None
        and cash_flow_builders_available
    ):
        daily_df = (
            _build_daily_summary_df(hourly_logs_by_year, cfg.step_hours)
            if hourly_logs_by_year
            else pd.DataFrame()
        )
        operating_cash_flow_df = economics.build_operating_cash_flow_table(
            annual_delivered,
            annual_bess,
            annual_pv_excess,
            normalized_econ_inputs,
            price_inputs,
            annual_pv_delivered_mwh=annual_pv_delivered,
            annual_shortfall_mwh=annual_shortfall,
            annual_wesm_shortfall_cost_usd=annual_wesm_shortfall_cost_usd,
            annual_wesm_surplus_revenue_usd=annual_wesm_surplus_revenue_usd,
            augmentation_costs_usd=augmentation_costs_usd if augmentation_costs_usd else None,
            annual_total_generation_mwh=annual_total_generation,
        )
        financing_cash_flow_df = economics.build_financing_cash_flow_table(
            annual_delivered,
            annual_bess,
            annual_pv_excess,
            normalized_econ_inputs,
            price_inputs,
            annual_shortfall_mwh=annual_shortfall,
            annual_wesm_shortfall_cost_usd=annual_wesm_shortfall_cost_usd,
            annual_wesm_surplus_revenue_usd=annual_wesm_surplus_revenue_usd,
            augmentation_costs_usd=augmentation_costs_usd if augmentation_costs_usd else None,
            annual_total_generation_mwh=annual_total_generation,
        )
        metrics_summary = pd.DataFrame(
            [
                {
                    "Metric": "Project NPV (USD, WACC)",
                    "Value": financing_outputs.project_npv_usd,
                },
                {
                    "Metric": "IRR (%)",
                    "Value": cash_outputs.irr_pct,
                },
                {
                    "Metric": "PIRR (%)",
                    "Value": financing_outputs.project_irr_pct,
                },
                {
                    "Metric": "EIRR (%)",
                    "Value": financing_outputs.equity_irr_pct,
                },
                {
                    "Metric": "Discount rate (%)",
                    "Value": normalized_econ_inputs.discount_rate * 100.0,
                },
                {
                    "Metric": "WACC (%)",
                    "Value": normalized_econ_inputs.wacc * 100.0,
                },
            ]
        )
        assumptions_df = _build_finance_assumptions_df(normalized_econ_inputs, price_inputs)
        finance_workbook = _build_finance_audit_workbook(
            assumptions_df,
            metrics_summary,
            res_df,
            monthly_df,
            daily_df,
            operating_cash_flow_df,
            financing_cash_flow_df,
        )
        st.download_button(
            "Download finance audit workbook (Excel)",
            finance_workbook,
            file_name="bess_finance_audit.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.caption(
            "Workbook includes economics assumptions, yearly/monthly/daily energy traces, and "
            "operating/financing cash-flow tables."
        )
    elif run_economics and not cash_flow_builders_available:
        st.warning(
            "Finance audit workbook unavailable: cash-flow helpers were not found. "
            "Please refresh the app to load the latest economics helpers."
        )
    st.download_button("Download yearly summary (CSV)", res_df.to_csv(index=False).encode('utf-8'),
                       file_name='bess_yearly_summary.csv', mime='text/csv')

    st.download_button("Download monthly summary (CSV)", monthly_df.to_csv(index=False).encode('utf-8'),
                       file_name='bess_monthly_summary.csv', mime='text/csv')

    if hourly_logs_by_year:
        hourly_workbook = _build_hourly_summary_workbook(hourly_logs_by_year)
        st.download_button(
            "Download hourly summary (Excel)",
            hourly_workbook,
            file_name="bess_hourly_summary.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    if final_year_logs is not None:
        hourly_df = _build_hourly_summary_df(final_year_logs)
        st.download_button("Download final-year hourly logs (CSV)", hourly_df.to_csv(index=False).encode('utf-8'),
                           file_name='final_year_hourly_logs.csv', mime='text/csv')

    pdf_bytes = None
    try:
        pdf_bytes = build_pdf_summary(
            cfg,
            results,
            kpis.compliance,
            kpis.bess_share_of_firm,
            kpis.charge_discharge_ratio,
            kpis.pv_capture_ratio,
            kpis.discharge_capacity_factor,
            discharge_windows_text,
            charge_windows_text,
            hod_count,
            hod_sum_pv_resource,
            hod_sum_pv,
            hod_sum_bess,
            hod_sum_charge,
            kpis.total_shortfall_mwh,
            kpis.pv_excess_mwh,
            kpis.total_project_generation_mwh,
            kpis.bess_generation_mwh,
            kpis.pv_generation_mwh,
            kpis.bess_losses_mwh,
            final_year_logs,
            sim_output.augmentation_energy_added_mwh,
            augmentation_retired,
        )
    except Exception as exc:  # noqa: BLE001
        st.warning(f"PDF snapshot unavailable: {exc}")

    if pdf_bytes:
        st.download_button("Download brief PDF snapshot", pdf_bytes,
                           file_name='bess_results_snapshot.pdf', mime='application/pdf')
    else:
        st.info("PDF snapshot will appear after the simulation succeeds.")

    st.info("""
    Notes & Caveats:
    - PV-only charging is enforced; during discharge hours, PV first meets the contract, then surplus PV charges the BESS.
    - Threshold augmentation offers **Capability** and **SOH** triggers. Power is added to keep original C-hours.
    - EOY capability = what the fleet can sustain per day at year-end; Delivered Split = what actually happened per day on average.
    - Design Advisor uses a conservative energy-limited view: Deliverable/day ≈ BOL usable × SOH(final) × ΔSOC × η_dis.
    """)


if __name__ == "__main__":
    run_app()
