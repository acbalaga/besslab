import json
from dataclasses import replace
from typing import Any, Dict, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from app import BASE_DIR
from services.simulation_core import SimConfig
from utils import enforce_rate_limit, parse_numeric_series, read_wesm_profile
from utils.io import read_wesm_forecast_profile_average
from utils.economics import DEVEX_COST_PHP, EconomicInputs, PriceInputs
from utils.bess_advisory import (
    build_power_block_marginals,
    build_sizing_benchmark_table,
    choose_recommended_candidate,
    extract_pareto_frontier,
    rank_recommendation_candidates,
    summarize_pv_sizing_signals,
)
from utils.sweeps import _resolve_ranking_column, sweep_bess_sizes
from utils.ui_layout import init_page_layout
from utils.ui_state import (
    bootstrap_session_state,
    get_cached_simulation_config,
    get_latest_economics_payload,
)

bootstrap_session_state()

render_layout = init_page_layout(
    page_title="BESS Sizing Sweep",
    main_title="BESS sizing sweep (comprehensive advisory)",
    description=(
        "Analyze the PV 8760 source and sweep BESS candidates to generate reliability and economics sizing advice."
    ),
    base_dir=BASE_DIR,
)

ENERGY_POINTS_MIN = 1
ENERGY_POINTS_MAX = 30

BENCHMARK_PRESETS: Dict[str, Dict[str, float]] = {
    "Conservative": {
        "pv_availability": 0.98,
        "bess_availability": 0.95,
        "calendar_fade_rate": 0.015,
        "soc_floor": 0.05,
        "soc_ceiling": 0.95,
        "rte_roundtrip": 0.86,
    },
    "Base": {
        "pv_availability": 0.99,
        "bess_availability": 0.97,
        "calendar_fade_rate": 0.01,
        "soc_floor": 0.0,
        "soc_ceiling": 1.0,
        "rte_roundtrip": 0.89,
    },
    "Aggressive": {
        "pv_availability": 0.995,
        "bess_availability": 0.98,
        "calendar_fade_rate": 0.0075,
        "soc_floor": 0.0,
        "soc_ceiling": 1.0,
        "rte_roundtrip": 0.92,
    },
}
BENCHMARK_PRESET_OPTIONS = ["Conservative", "Base", "Aggressive", "Custom"]

DEFAULT_DURATION_BAND_HOURS = (2.0, 6.0)
DEFAULT_C_RATE_BAND_PER_H = (0.17, 0.5)
DEFAULT_COMPLIANCE_TARGET_PCT = 99.0
DEFAULT_CANDIDATE_PAIRS_MW_MWH: tuple[tuple[float, float], ...] = (
    (5.0, 20.0),
    (5.0, 25.0),
    (10.0, 40.0),
    (10.0, 50.0),
    (15.0, 60.0),
    (15.0, 75.0),
    (20.0, 80.0),
    (20.0, 100.0),
    (25.0, 100.0),
    (25.0, 125.0),
    (30.0, 120.0),
    (30.0, 150.0),
    (35.0, 140.0),
    (35.0, 175.0),
    (40.0, 160.0),
    (40.0, 200.0),
)


def _candidate_label(power_mw: float, energy_mwh: float) -> str:
    return f"{power_mw:.0f} MW / {energy_mwh:.0f} MWh"


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


def _normalize_energy_range(value: Any, default: Tuple[float, float]) -> Tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low = _coerce_float(value[0], default[0])
        high = _coerce_float(value[1], default[1])
        return (low, high) if low <= high else (high, low)
    return default


def _sanitize_benchmark_assumptions(payload: Any, fallback: Dict[str, float]) -> Dict[str, float]:
    """Normalize benchmark assumptions from JSON/session payloads."""

    assumptions = fallback.copy()
    if not isinstance(payload, dict):
        return assumptions
    for key, fallback_value in fallback.items():
        assumptions[key] = _coerce_float(payload.get(key), fallback_value)

    assumptions["pv_availability"] = min(max(assumptions["pv_availability"], 0.0), 1.0)
    assumptions["bess_availability"] = min(max(assumptions["bess_availability"], 0.0), 1.0)
    assumptions["calendar_fade_rate"] = max(assumptions["calendar_fade_rate"], 0.0)
    assumptions["soc_floor"] = min(max(assumptions["soc_floor"], 0.0), 1.0)
    assumptions["soc_ceiling"] = min(max(assumptions["soc_ceiling"], 0.0), 1.0)
    assumptions["rte_roundtrip"] = min(max(assumptions["rte_roundtrip"], 0.0), 1.0)
    if assumptions["soc_floor"] > assumptions["soc_ceiling"]:
        assumptions["soc_floor"], assumptions["soc_ceiling"] = assumptions["soc_ceiling"], assumptions["soc_floor"]
    return assumptions


def _normalize_threshold_band(value: Any, default: Tuple[float, float], *, floor: float = 0.0) -> Tuple[float, float]:
    low, high = _normalize_energy_range(value, default)
    low = max(low, floor)
    high = max(high, floor)
    return (low, high) if low <= high else (high, low)


def _normalize_sweep_inputs(payload: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize JSON payload values to the expected sweep input schema."""

    normalized = defaults.copy()
    normalized["energy_range"] = _normalize_energy_range(
        payload.get("energy_range"),
        defaults["energy_range"],
    )
    normalized["energy_steps"] = max(
        ENERGY_POINTS_MIN,
        min(
            ENERGY_POINTS_MAX,
            _coerce_int(payload.get("energy_steps"), defaults["energy_steps"]),
        ),
    )
    normalized["fixed_power"] = _coerce_float(payload.get("fixed_power"), defaults["fixed_power"])
    normalized["analysis_mode"] = payload.get("analysis_mode", defaults.get("analysis_mode"))
    normalized["power_range"] = _normalize_energy_range(
        payload.get("power_range"),
        defaults.get("power_range", (5.0, 50.0)),
    )
    normalized["wacc_pct"] = _coerce_float(payload.get("wacc_pct"), defaults["wacc_pct"])
    normalized["inflation_pct"] = _coerce_float(payload.get("inflation_pct"), defaults["inflation_pct"])
    normalized["forex_rate_php_per_usd"] = _coerce_float(
        payload.get("forex_rate_php_per_usd"),
        defaults["forex_rate_php_per_usd"],
    )
    normalized["capex_musd"] = _coerce_float(payload.get("capex_musd"), defaults["capex_musd"])
    normalized["capex_energy_usd_per_kwh"] = _coerce_float(
        payload.get("capex_energy_usd_per_kwh"),
        defaults["capex_energy_usd_per_kwh"],
    )
    normalized["capex_power_usd_per_kw"] = _coerce_float(
        payload.get("capex_power_usd_per_kw"),
        defaults["capex_power_usd_per_kw"],
    )
    normalized["capex_base_fixed_musd"] = _coerce_float(
        payload.get("capex_base_fixed_musd"),
        defaults["capex_base_fixed_musd"],
    )
    normalized["pv_capex_musd"] = _coerce_float(payload.get("pv_capex_musd"), defaults["pv_capex_musd"])
    normalized["fixed_opex_pct"] = _coerce_float(payload.get("fixed_opex_pct"), defaults["fixed_opex_pct"])
    normalized["fixed_opex_musd"] = _coerce_float(
        payload.get("fixed_opex_musd"),
        defaults["fixed_opex_musd"],
    )
    opex_mode_payload = payload.get("opex_mode", defaults["opex_mode"])
    if opex_mode_payload not in {"% of CAPEX per year", "PHP/kWh on total generation"}:
        opex_mode_payload = defaults["opex_mode"]
    normalized["opex_mode"] = opex_mode_payload
    normalized["opex_php_per_kwh"] = _coerce_float(
        payload.get("opex_php_per_kwh"),
        defaults["opex_php_per_kwh"],
    )
    if "opex_mode" not in payload and normalized["opex_php_per_kwh"] > 0:
        normalized["opex_mode"] = "PHP/kWh on total generation"
    normalized["include_devex_year0"] = _coerce_bool(
        payload.get("include_devex_year0"),
        defaults["include_devex_year0"],
    )
    normalized["devex_cost_php"] = _coerce_float(
        payload.get("devex_cost_php"),
        defaults["devex_cost_php"],
    )
    normalized["ranking_choice"] = payload.get("ranking_choice", defaults["ranking_choice"])
    normalized["min_soh"] = _coerce_float(payload.get("min_soh"), defaults["min_soh"])
    preset_payload = payload.get("benchmark_preset", defaults["benchmark_preset"])
    normalized["benchmark_preset"] = (
        preset_payload if preset_payload in BENCHMARK_PRESET_OPTIONS else defaults["benchmark_preset"]
    )
    normalized["benchmark_assumptions"] = _sanitize_benchmark_assumptions(
        payload.get("benchmark_assumptions"),
        defaults["benchmark_assumptions"],
    )
    normalized["duration_band_hours"] = _normalize_threshold_band(
        payload.get("duration_band_hours"),
        defaults["duration_band_hours"],
    )
    normalized["c_rate_band_per_h"] = _normalize_threshold_band(
        payload.get("c_rate_band_per_h"),
        defaults["c_rate_band_per_h"],
    )
    normalized["compliance_target_pct"] = max(
        0.0,
        min(100.0, _coerce_float(payload.get("compliance_target_pct"), defaults["compliance_target_pct"])),
    )
    if "contract_price_php_per_kwh" in payload:
        normalized["contract_price_php_per_kwh"] = _coerce_float(
            payload.get("contract_price_php_per_kwh"),
            defaults["contract_price_php_per_kwh"],
        )
    else:
        normalized["contract_price_php_per_kwh"] = round(
            120.0 / 1000.0 * normalized["forex_rate_php_per_usd"],
            2,
        )
    normalized["escalate_prices"] = _coerce_bool(
        payload.get("escalate_prices"),
        defaults["escalate_prices"],
    )
    normalized["wesm_pricing_enabled"] = _coerce_bool(
        payload.get("wesm_pricing_enabled"),
        defaults["wesm_pricing_enabled"],
    )
    normalized["sell_to_wesm"] = _coerce_bool(
        payload.get("sell_to_wesm"),
        defaults["sell_to_wesm"],
    )
    normalized["variable_opex_php_per_kwh"] = _coerce_float(
        payload.get("variable_opex_php_per_kwh"),
        defaults["variable_opex_php_per_kwh"],
    )
    normalized["variable_schedule_choice"] = payload.get(
        "variable_schedule_choice",
        defaults["variable_schedule_choice"],
    )
    normalized["periodic_variable_opex_usd"] = _coerce_float(
        payload.get("periodic_variable_opex_usd"),
        defaults["periodic_variable_opex_usd"],
    )
    normalized["periodic_variable_opex_interval_years"] = _coerce_int(
        payload.get("periodic_variable_opex_interval_years"),
        defaults["periodic_variable_opex_interval_years"],
    )
    schedule_payload = payload.get("variable_opex_schedule_usd")
    if isinstance(schedule_payload, (list, tuple)):
        normalized["variable_opex_schedule_usd"] = [
            _coerce_float(item, 0.0) for item in schedule_payload
        ]
    return normalized


def _resolve_wesm_profile_source(
    uploaded_file: Any,
    cached_payload: Optional[Dict[str, Any]],
    default_profile_path: Any,
    *,
    use_wesm_forecast: bool,
) -> tuple[Optional[Any], Optional[str], bool]:
    """Select a WESM profile source, favoring explicit uploads and cached inputs."""

    if uploaded_file is not None:
        return uploaded_file, "uploaded for this sweep", False
    if isinstance(cached_payload, dict) and cached_payload.get("wesm_profile_source") is not None:
        return cached_payload["wesm_profile_source"], "cached from Inputs & Results", False
    cached_inputs_upload = st.session_state.get("inputs_wesm_upload")
    if cached_inputs_upload is not None:
        return cached_inputs_upload, "cached from Inputs & Results sidebar", False
    if default_profile_path is not None and default_profile_path.exists():
        return (
            str(default_profile_path),
            f"default profile ({default_profile_path})",
            use_wesm_forecast,
        )
    return None, None, False


def _default_wesm_profile_path(use_wesm_forecast: bool) -> Any:
    filename = (
        "wesm_price_profile_forecast.csv"
        if use_wesm_forecast
        else "wesm_price_profile_historical.csv"
    )
    return BASE_DIR / "data" / filename



cfg, dod_override = get_cached_simulation_config()
bootstrap_session_state(cfg)
default_forex_rate_php_per_usd = 58.0
default_contract_php_per_kwh = round(120.0 / 1000.0 * default_forex_rate_php_per_usd, 2)

# Rehydrate PV/cycle inputs from the shared session cache (no external fetches).
pv_df, cycle_df = render_layout()

if cfg is None:
    st.warning(
        "No inputs cached yet. Open the Inputs & Results page, adjust settings, and rerun the simulation to seed the sweep.",
        icon="⚠️",
    )
    cfg = SimConfig()

st.session_state.setdefault("bess_size_sweep_results", None)

default_inputs: Dict[str, Any] = {
    "energy_range": (10.0, 200.0),
    "energy_steps": 20,
    "fixed_power": float(cfg.initial_power_mw),
    "analysis_mode": "Comprehensive matrix (5–50 MW × 10–200 MWh)",
    "power_range": (5.0, 50.0),
    "wacc_pct": 8.0,
    "inflation_pct": 3.0,
    "forex_rate_php_per_usd": default_forex_rate_php_per_usd,
    "capex_musd": 40.0,
    "capex_energy_usd_per_kwh": 0.0,
    "capex_power_usd_per_kw": 0.0,
    "capex_base_fixed_musd": 0.0,
    "pv_capex_musd": 0.0,
    "fixed_opex_pct": 2.0,
    "fixed_opex_musd": 0.0,
    "opex_mode": "% of CAPEX per year",
    "opex_php_per_kwh": 0.0,
    "include_devex_year0": False,
    "devex_cost_php": float(DEVEX_COST_PHP),
    "ranking_choice": "compliance_pct",
    "min_soh": 0.6,
    "benchmark_preset": "Base",
    "benchmark_assumptions": BENCHMARK_PRESETS["Base"].copy(),
    "duration_band_hours": DEFAULT_DURATION_BAND_HOURS,
    "c_rate_band_per_h": DEFAULT_C_RATE_BAND_PER_H,
    "compliance_target_pct": DEFAULT_COMPLIANCE_TARGET_PCT,
    "contract_price_php_per_kwh": default_contract_php_per_kwh,
    "escalate_prices": False,
    "wesm_pricing_enabled": False,
    "sell_to_wesm": False,
    "variable_opex_php_per_kwh": 0.0,
    "variable_schedule_choice": "None",
    "periodic_variable_opex_usd": 0.0,
    "periodic_variable_opex_interval_years": 5,
    "variable_opex_schedule_usd": [],
}
stored_inputs = st.session_state.get("bess_sweep_inputs", {})
if isinstance(stored_inputs, dict):
    default_inputs = _normalize_sweep_inputs(stored_inputs, default_inputs)

cached_econ_payload = get_latest_economics_payload()

with st.expander("Load/save sweep inputs (JSON)", expanded=False):
    st.caption(
        "Upload or paste JSON to restore sweep inputs. Use the download button to save "
        "the current inputs for reuse."
    )
    uploaded_inputs = st.file_uploader(
        "Upload sweep inputs JSON",
        type=["json"],
        accept_multiple_files=False,
        key="bess_sweep_inputs_upload",
    )
    pasted_inputs = st.text_area(
        "Or paste sweep inputs JSON",
        placeholder='{"wacc_pct": 8.0, "capex_musd": 40.0}',
        height=120,
        key="bess_sweep_inputs_paste",
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
                    st.session_state["bess_sweep_inputs"] = _normalize_sweep_inputs(
                        payload, default_inputs
                    )
                    st.success("Sweep inputs loaded. Re-rendering with the new values.")
                    st.rerun()
                else:
                    st.error("Expected a JSON object with sweep input fields.")
        else:
            st.info("Provide JSON content to load.", icon="ℹ️")

with st.container():
    st.info(
        "Comprehensive advisory mode is active with a predefined BESS sizing shortlist "
        "covering 16 power/energy candidates."
    )

    size_col1, size_col2, size_col3, price_col = st.columns(4)
    wesm_profile_df: Optional[pd.DataFrame] = None
    candidate_pairs = list(DEFAULT_CANDIDATE_PAIRS_MW_MWH)
    candidate_labels = [_candidate_label(power_mw, energy_mwh) for power_mw, energy_mwh in candidate_pairs]
    with size_col1:
        st.metric("Sizing candidates", f"{len(candidate_pairs)} predefined pairs")
        st.caption("; ".join(candidate_labels))
    with size_col2:
        wacc_pct = st.number_input(
            "WACC (%)",
            min_value=0.0,
            max_value=30.0,
            value=float(default_inputs["wacc_pct"]),
            step=0.1,
            help="Weighted-average cost of capital (nominal).",
        )
        inflation_pct = st.number_input(
            "Inflation rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=float(default_inputs["inflation_pct"]),
            step=0.1,
            help="Inflation assumption used to derive the real discount rate.",
        )
        discount_rate = max((1 + wacc_pct / 100.0) / (1 + inflation_pct / 100.0) - 1, 0.0)
        forex_rate_php_per_usd = st.number_input(
            "FX rate (PHP/USD)",
            min_value=1.0,
            value=float(default_inputs["forex_rate_php_per_usd"]),
            step=0.5,
            help="Used to convert PHP-denominated inputs (prices, OPEX, DevEx) to USD.",
        )
        capex_musd = st.number_input(
            "Legacy BESS CAPEX (USD million)",
            min_value=0.0,
            value=float(default_inputs["capex_musd"]),
            step=0.1,
            help="Legacy single-scalar BESS CAPEX. Used only when the explicit sizing CAPEX terms below are not set.",
        )
        capex_energy_usd_per_kwh = st.number_input(
            "Energy CAPEX term (USD/kWh)",
            min_value=0.0,
            value=float(default_inputs["capex_energy_usd_per_kwh"]),
            step=1.0,
            help="Applied to candidate usable energy (MWh) during the sweep.",
        )
        capex_power_usd_per_kw = st.number_input(
            "Power CAPEX term (USD/kW)",
            min_value=0.0,
            value=float(default_inputs["capex_power_usd_per_kw"]),
            step=1.0,
            help="Applied to candidate power rating (MW) during the sweep.",
        )
        capex_base_fixed_musd = st.number_input(
            "Fixed BOS/base CAPEX (USD million)",
            min_value=0.0,
            value=float(default_inputs["capex_base_fixed_musd"]),
            step=0.1,
            help="Added once per candidate when explicit CAPEX sizing terms are used.",
        )
        pv_capex_musd = st.number_input(
            "PV CAPEX (USD million)",
            min_value=0.0,
            value=float(default_inputs["pv_capex_musd"]),
            step=0.1,
            help="Standalone PV CAPEX added to the BESS CAPEX input above.",
        )
        total_capex_musd = capex_musd + pv_capex_musd
        st.caption(f"Total project CAPEX (BESS + PV): ${total_capex_musd:,.2f}M.")
        opex_mode = st.radio(
            "OPEX input",
            options=["% of CAPEX per year", "PHP/kWh on total generation"],
            horizontal=True,
            help="Choose a fixed % of CAPEX/year or a PHP/kWh rate applied to total generation.",
            index=["% of CAPEX per year", "PHP/kWh on total generation"].index(
                default_inputs["opex_mode"]
                if default_inputs["opex_mode"] in {"% of CAPEX per year", "PHP/kWh on total generation"}
                else "% of CAPEX per year"
            ),
        )
        fixed_opex_pct = 0.0
        opex_php_per_kwh: Optional[float] = None
        if opex_mode == "% of CAPEX per year":
            fixed_opex_pct = st.number_input(
                "Fixed OPEX (% of CAPEX per year)",
                min_value=0.0,
                max_value=20.0,
                value=float(default_inputs["fixed_opex_pct"]),
                step=0.1,
                help="Annual fixed OPEX expressed as % of CAPEX.",
            ) / 100.0
        else:
            opex_php_per_kwh = st.number_input(
                "OPEX (PHP/kWh on total generation)",
                min_value=0.0,
                value=float(default_inputs["opex_php_per_kwh"]),
                step=0.05,
                help="Converted to USD/MWh using the FX rate; applied to total generation.",
            )
            if opex_php_per_kwh > 0:
                opex_usd_per_mwh = opex_php_per_kwh / forex_rate_php_per_usd * 1000.0
                st.caption(f"Converted OPEX: ${opex_usd_per_mwh:,.2f}/MWh.")
        fixed_opex_musd = st.number_input(
            "Additional fixed OPEX (USD million/yr)",
            min_value=0.0,
            value=float(default_inputs["fixed_opex_musd"]),
            step=0.1,
            help="Extra fixed OPEX not tied to CAPEX percentage.",
        )
        devex_choice = st.radio(
            "DevEx at year 0",
            options=["Exclude", "Include"],
            index=1 if default_inputs["include_devex_year0"] else 0,
            horizontal=True,
            help=(
                "Include or exclude the development expenditure at year 0. The PHP amount is "
                "converted to USD using the FX rate and flows through discounted costs, "
                "LCOE/LCOS, NPV, and IRR."
            ),
        )
        include_devex_year0 = devex_choice == "Include"
        devex_cost_php = st.number_input(
            "DevEx amount (PHP)",
            min_value=0.0,
            value=float(default_inputs["devex_cost_php"]),
            step=1_000_000.0,
            help="Used only when DevEx is included.",
            disabled=not include_devex_year0,
        )
        devex_cost_usd = devex_cost_php / forex_rate_php_per_usd if forex_rate_php_per_usd else 0.0
        if include_devex_year0:
            st.caption(
                "DevEx conversion: "
                f"PHP {devex_cost_php:,.0f} ≈ ${devex_cost_usd / 1_000_000:,.2f}M."
            )

    with size_col3:
        ranking_choice = st.selectbox(
            "Rank feasible candidates by",
            options=[
                "compliance_pct",
                "total_shortfall_mwh",
                "total_project_generation_mwh",
                "bess_generation_mwh",
                "lcoe_usd_per_mwh",
                "npv_costs_usd",
                "npv_usd",
            ],
            format_func=lambda x: {
                "compliance_pct": "Compliance % (higher is better)",
                "total_shortfall_mwh": "Shortfall MWh (lower is better)",
                "total_project_generation_mwh": "Total generation (higher is better)",
                "bess_generation_mwh": "BESS discharge (higher is better)",
                "lcoe_usd_per_mwh": "LCOE ($/MWh, lower is better)",
                "npv_costs_usd": "NPV of costs (USD, lower is better)",
                "npv_usd": "Net NPV (USD, higher is better)",
            }.get(x, x),
            help="Column used to pick the top feasible design.",
            index=[
                "compliance_pct",
                "total_shortfall_mwh",
                "total_project_generation_mwh",
                "bess_generation_mwh",
                "lcoe_usd_per_mwh",
                "npv_costs_usd",
                "npv_usd",
            ].index(
                default_inputs["ranking_choice"]
                if default_inputs["ranking_choice"]
                in [
                    "compliance_pct",
                    "total_shortfall_mwh",
                    "total_project_generation_mwh",
                    "bess_generation_mwh",
                    "lcoe_usd_per_mwh",
                    "npv_costs_usd",
                    "npv_usd",
                ]
                else "compliance_pct"
            ),
        )
        min_soh = st.number_input(
            "Minimum SOH for feasibility",
            min_value=0.2,
            max_value=1.0,
            value=float(default_inputs["min_soh"]),
            step=0.05,
            help="Candidates falling below this total SOH are flagged as infeasible.",
        )
        benchmark_preset = st.selectbox(
            "Benchmark assumption preset",
            options=BENCHMARK_PRESET_OPTIONS,
            index=BENCHMARK_PRESET_OPTIONS.index(
                default_inputs["benchmark_preset"]
                if default_inputs.get("benchmark_preset") in BENCHMARK_PRESET_OPTIONS
                else "Base"
            ),
            help="Preset applies availability, fade, SoC operating window, and round-trip efficiency.",
        )
        preset_defaults = BENCHMARK_PRESETS.get(benchmark_preset, BENCHMARK_PRESETS["Base"])
        stored_benchmark_assumptions = _sanitize_benchmark_assumptions(
            default_inputs.get("benchmark_assumptions"),
            BENCHMARK_PRESETS["Base"],
        )
        if benchmark_preset == "Custom":
            benchmark_assumptions = {
                "pv_availability": st.number_input(
                    "PV availability (fraction)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(stored_benchmark_assumptions["pv_availability"]),
                    step=0.005,
                ),
                "bess_availability": st.number_input(
                    "BESS availability (fraction)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(stored_benchmark_assumptions["bess_availability"]),
                    step=0.005,
                ),
                "calendar_fade_rate": st.number_input(
                    "Calendar fade rate (fraction/year)",
                    min_value=0.0,
                    max_value=0.2,
                    value=float(stored_benchmark_assumptions["calendar_fade_rate"]),
                    step=0.001,
                ),
                "soc_floor": st.number_input(
                    "SoC floor (fraction)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(stored_benchmark_assumptions["soc_floor"]),
                    step=0.01,
                ),
                "soc_ceiling": st.number_input(
                    "SoC ceiling (fraction)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(stored_benchmark_assumptions["soc_ceiling"]),
                    step=0.01,
                ),
                "rte_roundtrip": st.number_input(
                    "Round-trip efficiency (fraction)",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(stored_benchmark_assumptions["rte_roundtrip"]),
                    step=0.005,
                ),
            }
            benchmark_assumptions = _sanitize_benchmark_assumptions(
                benchmark_assumptions,
                stored_benchmark_assumptions,
            )
        else:
            benchmark_assumptions = preset_defaults.copy()

        duration_band_hours = _normalize_threshold_band(
            default_inputs.get("duration_band_hours"),
            DEFAULT_DURATION_BAND_HOURS,
        )
        duration_floor, duration_ceiling = st.slider(
            "Benchmark duration band (h)",
            min_value=0.5,
            max_value=12.0,
            value=(float(duration_band_hours[0]), float(duration_band_hours[1])),
            step=0.25,
            help="Candidates within this duration band pass the benchmark duration gate.",
        )
        c_rate_band = _normalize_threshold_band(
            default_inputs.get("c_rate_band_per_h"),
            DEFAULT_C_RATE_BAND_PER_H,
        )
        c_rate_floor, c_rate_ceiling = st.slider(
            "Benchmark C-rate band (1/h)",
            min_value=0.05,
            max_value=1.0,
            value=(float(c_rate_band[0]), float(c_rate_band[1])),
            step=0.01,
        )
        compliance_target_pct = st.number_input(
            "Benchmark compliance target (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(default_inputs.get("compliance_target_pct", DEFAULT_COMPLIANCE_TARGET_PCT)),
            step=0.1,
            help="Minimum compliance percentage to pass the reliability benchmark gate.",
        )
        st.caption(
            "Discount rate is derived from WACC and inflation to align with the economics helper."
        )

    with price_col:
        contract_price_php_per_kwh = st.number_input(
            "Contract price (PHP/kWh for delivered energy)",
            min_value=0.0,
            value=float(default_inputs["contract_price_php_per_kwh"]),
            step=0.05,
            help="Price converted to USD/MWh internally using the FX rate above.",
        )
        escalate_prices = st.checkbox(
            "Escalate prices with inflation",
            value=bool(default_inputs["escalate_prices"]),
        )
        wesm_pricing_enabled = st.checkbox(
            "Apply WESM pricing to contract shortfalls",
            value=bool(default_inputs["wesm_pricing_enabled"]),
            help="Uses the uploaded (or bundled) hourly WESM profile to price contract shortfalls.",
        )
        sell_to_wesm = st.checkbox(
            "Sell PV surplus to WESM",
            value=bool(default_inputs["sell_to_wesm"]),
            help=(
                "When enabled, PV surplus (excess MWh) is credited at a WESM sale price; otherwise surplus "
                "is excluded from revenue. Pricing comes from the hourly WESM profile."
            ),
            disabled=not wesm_pricing_enabled,
        )

        contract_price = contract_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
        st.caption(f"Converted contract price: ${contract_price:,.2f}/MWh.")
        if wesm_pricing_enabled:
            st.caption(
                "WESM pricing uses the hourly profile (upload or bundled default) for both "
                "shortfall costs and surplus revenue."
            )

        cached_variant = None
        if isinstance(cached_econ_payload, dict):
            cached_variant = cached_econ_payload.get("wesm_profile_variant")
        wesm_file = st.file_uploader(
            "WESM hourly price CSV (optional; timestamp/hour_index + deficit/surplus prices)",
            type=["csv"],
            key="bess_sweep_wesm_upload",
        )
        wesm_profile_variants = ["historical", "forecast"]
        wesm_profile_labels = {
            "historical": "Historical (default)",
            "forecast": "Forecast (8760-hr average)",
        }
        default_variant = cached_variant if cached_variant in wesm_profile_variants else "historical"
        selected_wesm_variant = st.selectbox(
            "Default WESM profile when no file is uploaded",
            options=wesm_profile_variants,
            index=wesm_profile_variants.index(default_variant),
            format_func=wesm_profile_labels.get,
            key="bess_sweep_wesm_profile_variant",
            disabled=not wesm_pricing_enabled,
            help=(
                "Defaults to data/wesm_price_profile_historical.csv or "
                "data/wesm_price_profile_forecast.csv. Forecast values are averaged across years. "
                "Uploaded files always take priority."
            ),
        )
        use_wesm_forecast = selected_wesm_variant == "forecast"
        if wesm_pricing_enabled:
            default_wesm_profile = _default_wesm_profile_path(use_wesm_forecast)
            wesm_profile_source, wesm_profile_label, wesm_profile_is_forecast = _resolve_wesm_profile_source(
                wesm_file,
                cached_econ_payload if isinstance(cached_econ_payload, dict) else None,
                default_wesm_profile,
                use_wesm_forecast=use_wesm_forecast,
            )
            if wesm_profile_source is not None:
                try:
                    if wesm_profile_is_forecast:
                        wesm_profile_df = read_wesm_forecast_profile_average(
                            [wesm_profile_source],
                            forex_rate_php_per_usd=forex_rate_php_per_usd,
                        )
                    else:
                        wesm_profile_df = read_wesm_profile(
                            [wesm_profile_source],
                            forex_rate_php_per_usd=forex_rate_php_per_usd,
                        )
                    st.caption(
                        "WESM hourly profile loaded "
                        f"({wesm_profile_label}). "
                        "Sweep cash flows will use hourly WESM pricing for shortfall costs and "
                        "surplus revenue."
                    )
                except Exception as exc:  # noqa: BLE001
                    st.warning(
                        "WESM profile could not be read. "
                        f"({exc})"
                    )
                    wesm_profile_df = None
            else:
                st.warning(
                    "WESM pricing is enabled but no hourly profile is available. "
                    "Upload a CSV or run Inputs & Results to cache a profile.",
                    icon="⚠️",
                )

    variable_col1, variable_col2 = st.columns(2)
    with variable_col1:
        st.markdown("**Variable OPEX overrides**")
        st.caption(
            "Schedule overrides supersede the base OPEX mode selection above (CAPEX % or PHP/kWh)."
        )
        variable_opex_php_per_kwh = st.number_input(
            "Variable OPEX (PHP/kWh on delivered energy)",
            min_value=0.0,
            value=float(default_inputs["variable_opex_php_per_kwh"]),
            step=0.05,
            help=(
                "Optional per-kWh operating expense applied to annual firm energy. "
                "Escalates with inflation and overrides the base OPEX selection when provided."
            ),
        )
        variable_opex_usd_per_mwh: Optional[float] = None
        if variable_opex_php_per_kwh > 0:
            variable_opex_usd_per_mwh = variable_opex_php_per_kwh / forex_rate_php_per_usd * 1000.0
            st.caption(
                f"Converted variable OPEX: ${variable_opex_usd_per_mwh:,.2f}/MWh (applied to delivered energy)."
            )
    with variable_col2:
        variable_schedule_choice = st.radio(
            "Variable expense schedule",
            options=["None", "Periodic", "Custom"],
            horizontal=True,
            help=(
                "Custom or periodic schedules override the base OPEX mode and per-kWh overrides. "
                "Per-kWh overrides supersede fixed percentages and adders."
            ),
            index=["None", "Periodic", "Custom"].index(
                default_inputs["variable_schedule_choice"]
                if default_inputs["variable_schedule_choice"] in ["None", "Periodic", "Custom"]
                else "None"
            ),
        )
        variable_opex_schedule_usd: Optional[Tuple[float, ...]] = None
        periodic_variable_opex_usd: Optional[float] = None
        periodic_variable_opex_interval_years: Optional[int] = None
        if variable_schedule_choice == "Periodic":
            periodic_variable_opex_usd = st.number_input(
                "Variable expense when periodic (USD)",
                min_value=0.0,
                value=float(default_inputs["periodic_variable_opex_usd"]),
                step=10_000.0,
                help="Amount applied on the selected cadence (year 1, then every N years).",
            )
            periodic_variable_opex_interval_years = st.number_input(
                "Cadence (years)",
                min_value=1,
                value=int(default_inputs["periodic_variable_opex_interval_years"]),
                step=1,
            )
            if periodic_variable_opex_usd <= 0:
                periodic_variable_opex_usd = None
        elif variable_schedule_choice == "Custom":
            default_custom_text = "\n".join(
                str(val) for val in default_inputs.get("variable_opex_schedule_usd", [])
            )
            custom_variable_text = st.text_area(
                "Custom variable expenses (USD/year)",
                value=default_custom_text,
                placeholder="e.g., 250000, 275000, 300000",
                help="Comma or newline separated values applied per project year.",
            )
            st.caption(
                "Use commas or newlines between entries; provide one value per project year "
                f"({cfg.years} entries)."
            )
            if custom_variable_text.strip():
                try:
                    variable_opex_schedule_usd = tuple(
                        parse_numeric_series("Variable expense schedule", custom_variable_text)
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()

    submitted = st.button("Run BESS energy sweep", use_container_width=True)

    energy_values = [float(energy_mwh) for _, energy_mwh in candidate_pairs]
    power_values = [float(power_mw) for power_mw, _ in candidate_pairs]

    st.session_state["bess_sweep_inputs"] = {
        # Keep legacy array fields for backwards compatibility with prior saved JSON payloads.
        "energy_mwh_values": energy_values,
        "power_mw_values": power_values,
        "candidate_pairs_mw_mwh": [
            {"power_mw": float(power_mw), "energy_mwh": float(energy_mwh)}
            for power_mw, energy_mwh in candidate_pairs
        ],
        "wacc_pct": float(wacc_pct),
        "inflation_pct": float(inflation_pct),
        "forex_rate_php_per_usd": float(forex_rate_php_per_usd),
        "capex_musd": float(capex_musd),
        "capex_energy_usd_per_kwh": float(capex_energy_usd_per_kwh),
        "capex_power_usd_per_kw": float(capex_power_usd_per_kw),
        "capex_base_fixed_musd": float(capex_base_fixed_musd),
        "pv_capex_musd": float(pv_capex_musd),
        "fixed_opex_pct": float(fixed_opex_pct * 100.0),
        "fixed_opex_musd": float(fixed_opex_musd),
        "opex_mode": opex_mode,
        "opex_php_per_kwh": float(opex_php_per_kwh or 0.0),
        "include_devex_year0": bool(include_devex_year0),
        "devex_cost_php": float(devex_cost_php),
        "ranking_choice": ranking_choice,
        "min_soh": float(min_soh),
        "benchmark_preset": benchmark_preset,
        "benchmark_assumptions": benchmark_assumptions,
        "duration_band_hours": [float(duration_floor), float(duration_ceiling)],
        "c_rate_band_per_h": [float(c_rate_floor), float(c_rate_ceiling)],
        "compliance_target_pct": float(compliance_target_pct),
        "contract_price_php_per_kwh": float(contract_price_php_per_kwh),
        "escalate_prices": bool(escalate_prices),
        "wesm_pricing_enabled": bool(wesm_pricing_enabled),
        "sell_to_wesm": bool(sell_to_wesm),
        "variable_opex_php_per_kwh": float(variable_opex_php_per_kwh),
        "variable_schedule_choice": variable_schedule_choice,
        "periodic_variable_opex_usd": float(
            periodic_variable_opex_usd
            if periodic_variable_opex_usd is not None
            else default_inputs["periodic_variable_opex_usd"]
        ),
        "periodic_variable_opex_interval_years": int(
            periodic_variable_opex_interval_years
            if periodic_variable_opex_interval_years is not None
            else default_inputs["periodic_variable_opex_interval_years"]
        ),
        "variable_opex_schedule_usd": list(variable_opex_schedule_usd or ()),
    }

    inputs_json = json.dumps(st.session_state["bess_sweep_inputs"], indent=2).encode("utf-8")
    st.download_button(
        "Download sweep inputs (JSON)",
        data=inputs_json,
        file_name="bess_sizing_sweep_inputs.json",
        mime="application/json",
        use_container_width=True,
    )

if submitted:
    enforce_rate_limit()
    if wesm_pricing_enabled and wesm_profile_df is None:
        st.error(
            "WESM pricing is enabled but no hourly WESM profile is available. "
            "Upload a profile or disable WESM pricing to continue."
        )
        st.stop()
    benchmark_cfg = replace(
        cfg,
        pv_availability=benchmark_assumptions["pv_availability"],
        bess_availability=benchmark_assumptions["bess_availability"],
        calendar_fade_rate=benchmark_assumptions["calendar_fade_rate"],
        soc_floor=benchmark_assumptions["soc_floor"],
        soc_ceiling=benchmark_assumptions["soc_ceiling"],
        rte_roundtrip=benchmark_assumptions["rte_roundtrip"],
    )

    if (
        len(power_values) > 1
        and len(energy_values) > 1
        and capex_energy_usd_per_kwh <= 0
        and capex_power_usd_per_kw <= 0
        and capex_base_fixed_musd <= 0
    ):
        st.warning(
            "Using legacy single-scalar BESS CAPEX for a 2D power/energy matrix. "
            "Set energy and/or power CAPEX terms to capture independent sizing effects.",
            icon="⚠️",
        )

    economics_inputs = EconomicInputs(
        capex_musd=capex_musd,
        capex_energy_usd_per_kwh=(capex_energy_usd_per_kwh if capex_energy_usd_per_kwh > 0 else None),
        capex_power_usd_per_kw=(capex_power_usd_per_kw if capex_power_usd_per_kw > 0 else None),
        capex_base_fixed_musd=(capex_base_fixed_musd if capex_base_fixed_musd > 0 else None),
        pv_capex_musd=pv_capex_musd,
        fixed_opex_pct_of_capex=fixed_opex_pct,
        fixed_opex_musd=fixed_opex_musd,
        opex_php_per_kwh=opex_php_per_kwh if opex_mode == "PHP/kWh on total generation" else None,
        inflation_rate=inflation_pct / 100.0,
        discount_rate=discount_rate,
        variable_opex_usd_per_mwh=variable_opex_usd_per_mwh,
        variable_opex_schedule_usd=variable_opex_schedule_usd,
        periodic_variable_opex_usd=periodic_variable_opex_usd,
        periodic_variable_opex_interval_years=periodic_variable_opex_interval_years,
        forex_rate_php_per_usd=forex_rate_php_per_usd,
        devex_cost_php=devex_cost_php,
        include_devex_year0=include_devex_year0,
    )
    price_inputs = PriceInputs(
        contract_price_usd_per_mwh=contract_price,
        escalate_with_inflation=escalate_prices,
        apply_wesm_to_shortfall=wesm_pricing_enabled,
        sell_to_wesm=sell_to_wesm if wesm_pricing_enabled else False,
    )

    total_candidates = len(candidate_pairs)
    progress_state = {"completed": 0}
    progress_bar = st.progress(
        0.0,
        text=f"Running sizing sweep candidates: 0/{total_candidates} completed.",
    )

    def _update_progress(progress_rows: pd.DataFrame) -> None:
        """Update the UI progress bar as candidate batches finish."""

        completed_now = int(len(progress_rows.index))
        progress_state["completed"] = min(
            total_candidates,
            progress_state["completed"] + max(completed_now, 0),
        )
        fraction = progress_state["completed"] / max(total_candidates, 1)
        progress_bar.progress(
            fraction,
            text=(
                "Running sizing sweep candidates: "
                f"{progress_state['completed']}/{total_candidates} completed."
            ),
        )

    with st.spinner("Running BESS energy sweep..."):
        common_kwargs = dict(
            base_cfg=benchmark_cfg,
            pv_df=pv_df,
            cycle_df=cycle_df,
            dod_override=dod_override,
            economics_inputs=economics_inputs,
            price_inputs=price_inputs,
            wesm_profile_df=wesm_profile_df if wesm_pricing_enabled else None,
            wesm_step_hours=getattr(cfg, "step_hours", 1.0),
            ranking_kpi=ranking_choice,
            min_soh=min_soh,
            use_case="reliability",
            progress_callback=_update_progress,
        )

        matrix_frames: list[pd.DataFrame] = []
        for power_mw, energy_mwh in candidate_pairs:
            matrix_frames.append(
                sweep_bess_sizes(
                    **common_kwargs,
                    energy_mwh_values=[float(energy_mwh)],
                    fixed_power_mw=float(power_mw),
                )
            )
        sweep_df = pd.concat(matrix_frames, ignore_index=True) if matrix_frames else pd.DataFrame()

    progress_bar.progress(
        1.0,
        text=f"Sizing sweep finished: {min(progress_state['completed'], total_candidates)}/{total_candidates} completed.",
    )


    if sweep_df.empty:
        st.info("No sweep results generated; widen the ranges and try again.")
        st.session_state["bess_size_sweep_results"] = None
    else:
        st.session_state["bess_size_sweep_results"] = sweep_df
        st.toast("BESS energy sweep complete.")

sweep_df = st.session_state.get("bess_size_sweep_results")
if sweep_df is not None:
    pv_signals = summarize_pv_sizing_signals(pv_df)
    st.subheader("PV 8760 technical signals")
    signal_cols = st.columns(5)
    signal_cols[0].metric("Annual PV energy", f"{pv_signals.annual_energy_mwh:,.0f} MWh")
    signal_cols[1].metric("PV peak", f"{pv_signals.peak_power_mw:,.2f} MW")
    signal_cols[2].metric("Implied CF", f"{pv_signals.implied_capacity_factor_pct:,.1f}%")
    signal_cols[3].metric("Active PV hours", f"{pv_signals.active_hours_pct:,.1f}%")
    signal_cols[4].metric("P95 hourly ramp", f"{pv_signals.p95_hourly_ramp_mw:,.2f} MW")
    st.caption("Sizing signals are shown for the active sweep benchmark settings.")

    assumption_card = pd.DataFrame(
        [
            {"Assumption": "Preset", "Value": benchmark_preset, "Unit": "-"},
            {
                "Assumption": "PV availability",
                "Value": f"{benchmark_assumptions['pv_availability'] * 100:.2f}",
                "Unit": "%",
            },
            {
                "Assumption": "BESS availability",
                "Value": f"{benchmark_assumptions['bess_availability'] * 100:.2f}",
                "Unit": "%",
            },
            {
                "Assumption": "Calendar fade",
                "Value": f"{benchmark_assumptions['calendar_fade_rate'] * 100:.2f}",
                "Unit": "%/year",
            },
            {
                "Assumption": "SoC floor",
                "Value": f"{benchmark_assumptions['soc_floor'] * 100:.1f}",
                "Unit": "%",
            },
            {
                "Assumption": "SoC ceiling",
                "Value": f"{benchmark_assumptions['soc_ceiling'] * 100:.1f}",
                "Unit": "%",
            },
            {
                "Assumption": "Round-trip efficiency",
                "Value": f"{benchmark_assumptions['rte_roundtrip'] * 100:.2f}",
                "Unit": "%",
            },
            {
                "Assumption": "Duration benchmark band",
                "Value": f"{duration_floor:.2f} to {duration_ceiling:.2f}",
                "Unit": "h",
            },
            {
                "Assumption": "C-rate benchmark band",
                "Value": f"{c_rate_floor:.2f} to {c_rate_ceiling:.2f}",
                "Unit": "1/h",
            },
            {
                "Assumption": "Compliance benchmark target",
                "Value": f"{compliance_target_pct:.2f}",
                "Unit": "%",
            },
        ]
    )
    st.markdown("**Assumption card**")
    st.dataframe(assumption_card, use_container_width=True, hide_index=True)

    benchmark_df = build_sizing_benchmark_table(
        sweep_df,
        duration_band_hours=(duration_floor, duration_ceiling),
        c_rate_band_per_h=(c_rate_floor, c_rate_ceiling),
        compliance_target_pct=compliance_target_pct,
    )
    ranking_column, ranking_ascending, _, _ = _resolve_ranking_column("reliability", ranking_choice)
    recommendation = choose_recommended_candidate(
        sweep_df,
        ranking_column=ranking_column,
        ascending=ranking_ascending,
        duration_band_hours=(duration_floor, duration_ceiling),
        c_rate_band_per_h=(c_rate_floor, c_rate_ceiling),
        compliance_target_pct=compliance_target_pct,
    )
    if recommendation is not None:
        st.success(
            "Recommended benchmark candidate: "
            f"{recommendation['power_mw']:.0f} MW / {recommendation['energy_mwh']:.0f} MWh "
            f"({recommendation['duration_h']:.1f} h), compliance {recommendation['compliance_pct']:.2f}%, "
            f"shortfall {recommendation['total_shortfall_mwh']:.1f} MWh."
        )

    st.subheader("Comprehensive sizing visuals")
    viz_df = benchmark_df.copy()
    viz_df["compliance_gap_pct"] = (compliance_target_pct - viz_df["compliance_pct"]).clip(lower=0.0)
    viz_df["candidate_label"] = viz_df.apply(
        lambda row: _candidate_label(float(row["power_mw"]), float(row["energy_mwh"])),
        axis=1,
    )
    candidate_order = [_candidate_label(power_mw, energy_mwh) for power_mw, energy_mwh in DEFAULT_CANDIDATE_PAIRS_MW_MWH]

    reliability_base = alt.Chart(viz_df).encode(
        x=alt.X("candidate_label:N", title="Candidate", sort=candidate_order),
        tooltip=[
            alt.Tooltip("candidate_label:N", title="Candidate"),
            alt.Tooltip("compliance_pct:Q", title="Compliance (%)", format=",.2f"),
            alt.Tooltip("surplus_pct:Q", title="Surplus (%)", format=",.2f"),
            alt.Tooltip("total_shortfall_mwh:Q", title="Shortfall (MWh)", format=",.1f"),
        ],
    )
    reliability_chart = (
        alt.layer(
            reliability_base.mark_line(point=True, color="#1f77b4").encode(
                y=alt.Y("compliance_pct:Q", title="Compliance (%)", axis=alt.Axis(titleColor="#1f77b4")),
            ),
            reliability_base.mark_line(point=True, color="#ff7f0e").encode(
                y=alt.Y("surplus_pct:Q", title="Surplus (%)", axis=alt.Axis(titleColor="#ff7f0e")),
            ),
        )
        .resolve_scale(y="independent")
        .properties(title="Reliability by candidate: compliance and surplus")
    )
    st.altair_chart(reliability_chart, use_container_width=True)

    npv_field = "npv_usd" if "npv_usd" in viz_df.columns and viz_df["npv_usd"].notna().any() else "npv_costs_usd"
    npv_title = "Net NPV (USD)" if npv_field == "npv_usd" else "NPV of costs (USD)"

    tradeoff_chart = (
        alt.Chart(viz_df)
        .mark_circle(size=130, opacity=0.85)
        .encode(
            x=alt.X("total_shortfall_mwh:Q", title="Annual shortfall (MWh)"),
            y=alt.Y(f"{npv_field}:Q", title=npv_title),
            color=alt.Color("compliance_pct:Q", title="Compliance (%)", scale=alt.Scale(scheme="tealblues")),
            size=alt.Size("energy_mwh:Q", title="Usable energy (MWh)"),
            tooltip=[
                alt.Tooltip("candidate_label:N", title="Candidate"),
                alt.Tooltip("power_mw:Q", title="Power (MW)", format=",.0f"),
                alt.Tooltip("energy_mwh:Q", title="Energy (MWh)", format=",.0f"),
                alt.Tooltip("compliance_pct:Q", title="Compliance (%)", format=",.2f"),
                alt.Tooltip("surplus_pct:Q", title="Surplus (%)", format=",.2f"),
                alt.Tooltip("total_shortfall_mwh:Q", title="Shortfall (MWh)", format=",.1f"),
                alt.Tooltip(f"{npv_field}:Q", title=npv_title, format=",.0f"),
            ],
        )
        .properties(title="Sizing behavior scatter: reliability vs economics")
    )
    st.altair_chart(tradeoff_chart, use_container_width=True)

    top_cols = [
        "candidate_label",
        "power_mw",
        "energy_mwh",
        "duration_h",
        "compliance_pct",
        "total_shortfall_mwh",
        "surplus_pct",
        npv_field,
        "irr_pct",
        "benchmark_duration_ok",
        "benchmark_c_rate_ok",
        "benchmark_reliability_ok",
    ]
    available_top_cols = [col for col in top_cols if col in viz_df.columns]
    ranked_df = rank_recommendation_candidates(
        viz_df,
        ranking_column=ranking_column,
        ascending=ranking_ascending,
        duration_band_hours=(duration_floor, duration_ceiling),
        c_rate_band_per_h=(c_rate_floor, c_rate_ceiling),
        compliance_target_pct=compliance_target_pct,
    )
    st.dataframe(ranked_df[available_top_cols].head(len(candidate_order)), use_container_width=True, hide_index=True)

    st.subheader("Pareto frontier and marginal sizing value")
    economic_objective_choice = st.selectbox(
        "Optional economic objective for Pareto filtering",
        options=["None", "npv_usd", "lcoe_usd_per_mwh"],
        index=1 if "npv_usd" in viz_df.columns else 0,
        help=(
            "Base objectives always include maximizing compliance and minimizing shortfall. "
            "You can optionally include NPV (maximize) or LCOE (minimize)."
        ),
        key="pareto_economic_objective",
    )
    selected_economic_objective = None if economic_objective_choice == "None" else economic_objective_choice

    pareto_df = extract_pareto_frontier(viz_df, economic_objective=selected_economic_objective)
    if pareto_df.empty:
        st.info("No frontier candidates were available after filtering to evaluated rows.")
    else:
        pareto_df = pareto_df.copy()
        pareto_df["candidate_label"] = pareto_df.apply(
            lambda row: _candidate_label(float(row["power_mw"]), float(row["energy_mwh"])),
            axis=1,
        )
        pareto_df = pareto_df.sort_values(["total_shortfall_mwh", "compliance_pct"])

        pareto_chart = (
            alt.layer(
                alt.Chart(viz_df)
                .mark_circle(size=90, opacity=0.25, color="#9aa0a6")
                .encode(
                    x=alt.X("total_shortfall_mwh:Q", title="Annual shortfall (MWh)"),
                    y=alt.Y("compliance_pct:Q", title="Compliance (%)"),
                    tooltip=[
                        alt.Tooltip("candidate_label:N", title="Candidate"),
                        alt.Tooltip("total_shortfall_mwh:Q", title="Shortfall (MWh)", format=",.1f"),
                        alt.Tooltip("compliance_pct:Q", title="Compliance (%)", format=",.2f"),
                        alt.Tooltip("surplus_pct:Q", title="Surplus (%)", format=",.2f"),
                    ],
                ),
                alt.Chart(pareto_df)
                .mark_line(point=True, color="#2ca02c", strokeWidth=3)
                .encode(
                    x="total_shortfall_mwh:Q",
                    y="compliance_pct:Q",
                    tooltip=[
                        alt.Tooltip("candidate_label:N", title="Frontier candidate"),
                        alt.Tooltip("total_shortfall_mwh:Q", title="Shortfall (MWh)", format=",.1f"),
                        alt.Tooltip("compliance_pct:Q", title="Compliance (%)", format=",.2f"),
                        alt.Tooltip("surplus_pct:Q", title="Surplus (%)", format=",.2f"),
                    ],
                ),
            )
            .properties(title="Pareto frontier across predefined candidates")
        )
        st.altair_chart(pareto_chart, use_container_width=True)

        marginals_df = build_power_block_marginals(
            pareto_df,
            economic_objective=selected_economic_objective,
        )
        marginals_df["candidate_label"] = marginals_df.apply(
            lambda row: _candidate_label(float(row["power_mw"]), float(row["energy_mwh"])),
            axis=1,
        )

        marginal_chart = (
            alt.Chart(marginals_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("candidate_label:N", title="Frontier candidate", sort=candidate_order),
                y=alt.Y("compliance_gain_per_mwh:Q", title="ΔCompliance / ΔMWh (pct-point per MWh)"),
                color=alt.Color("power_mw:N", title="Power (MW)"),
                tooltip=[
                    alt.Tooltip("candidate_label:N", title="Candidate"),
                    alt.Tooltip("delta_energy_mwh:Q", title="ΔEnergy (MWh)", format=",.1f"),
                    alt.Tooltip("compliance_gain_per_mwh:Q", title="ΔCompliance/ΔMWh", format=",.5f"),
                    alt.Tooltip("shortfall_reduction_per_mwh:Q", title="ΔShortfall/ΔMWh", format=",.5f"),
                ],
            )
            .properties(title="Frontier stepwise compliance gain by added energy")
        )

        elbow_points = marginals_df[marginals_df["is_elbow"]]
        if not elbow_points.empty:
            marginal_chart = marginal_chart + (
                alt.Chart(elbow_points)
                .mark_point(shape="diamond", size=220, color="red")
                .encode(x="candidate_label:N", y="compliance_gain_per_mwh:Q")
            )
        st.altair_chart(marginal_chart, use_container_width=True)

        pareto_table_cols = [
            "candidate_label",
            "power_mw",
            "energy_mwh",
            "compliance_pct",
            "total_shortfall_mwh",
            "delta_energy_mwh",
            "compliance_gain_per_mwh",
            "shortfall_reduction_per_mwh",
            "is_elbow",
        ]
        if selected_economic_objective and selected_economic_objective in marginals_df.columns:
            pareto_table_cols.append(selected_economic_objective)
        if "economic_marginal_value_per_mwh" in marginals_df.columns:
            pareto_table_cols.append("economic_marginal_value_per_mwh")

        st.dataframe(
            marginals_df[[col for col in pareto_table_cols if col in marginals_df.columns]],
            use_container_width=True,
            hide_index=True,
        )

    if recommendation is not None:
        same_power = viz_df[viz_df["power_mw"] == float(recommendation["power_mw"])].sort_values("energy_mwh")
        lower_energy = same_power[same_power["energy_mwh"] < float(recommendation["energy_mwh"])]
        comparison_note = "No lower-energy candidate at this power level was available for comparison."
        if not lower_energy.empty:
            ref = lower_energy.iloc[-1]
            compliance_gain = float(recommendation["compliance_pct"] - ref["compliance_pct"])
            shortfall_reduction = float(ref["total_shortfall_mwh"] - recommendation["total_shortfall_mwh"])
            comparison_note = (
                f"Versus the next lower-energy option at {recommendation['power_mw']:.0f} MW "
                f"({ref['energy_mwh']:.0f} MWh), the recommendation improves compliance by "
                f"{compliance_gain:.2f} points and reduces annual shortfall by {shortfall_reduction:,.1f} MWh."
            )

        st.info(
            "Recommendation rationale: "
            f"{recommendation['power_mw']:.0f} MW / {recommendation['energy_mwh']:.0f} MWh is selected because it "
            "meets reliability-first screening while balancing duration and economics across the matrix. "
            + comparison_note
        )
else:
    st.info(
        "Run the sweep with your latest inputs. Results persist in the session for quick iteration.",
        icon="ℹ️",
    )
