import json
import math
from typing import Any, Dict, Optional, Tuple

import altair as alt
import pandas as pd
import streamlit as st

from app import BASE_DIR
from services.simulation_core import SimConfig
from utils import enforce_rate_limit, parse_numeric_series
from utils.economics import DEVEX_COST_PHP, EconomicInputs, PriceInputs
from utils.sweeps import generate_values, sweep_bess_sizes
from utils.ui_layout import init_page_layout
from utils.ui_state import bootstrap_session_state, get_cached_simulation_config

bootstrap_session_state()

render_layout = init_page_layout(
    page_title="BESS Sizing Sweep",
    main_title="BESS sizing sweep (energy sensitivity)",
    description="Sweep usable energy (MWh) while holding power constant to see feasibility, LCOE, and NPV.",
    base_dir=BASE_DIR,
)

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


def _normalize_sweep_inputs(payload: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize JSON payload values to the expected sweep input schema."""

    normalized = defaults.copy()
    normalized["energy_range"] = _normalize_energy_range(
        payload.get("energy_range"),
        defaults["energy_range"],
    )
    normalized["energy_steps"] = _coerce_int(payload.get("energy_steps"), defaults["energy_steps"])
    normalized["fixed_power"] = _coerce_float(payload.get("fixed_power"), defaults["fixed_power"])
    normalized["wacc_pct"] = _coerce_float(payload.get("wacc_pct"), defaults["wacc_pct"])
    normalized["inflation_pct"] = _coerce_float(payload.get("inflation_pct"), defaults["inflation_pct"])
    normalized["forex_rate_php_per_usd"] = _coerce_float(
        payload.get("forex_rate_php_per_usd"),
        defaults["forex_rate_php_per_usd"],
    )
    normalized["capex_musd"] = _coerce_float(payload.get("capex_musd"), defaults["capex_musd"])
    normalized["pv_capex_musd"] = _coerce_float(payload.get("pv_capex_musd"), defaults["pv_capex_musd"])
    normalized["fixed_opex_pct"] = _coerce_float(payload.get("fixed_opex_pct"), defaults["fixed_opex_pct"])
    normalized["fixed_opex_musd"] = _coerce_float(
        payload.get("fixed_opex_musd"),
        defaults["fixed_opex_musd"],
    )
    normalized["include_devex_year0"] = _coerce_bool(
        payload.get("include_devex_year0"),
        defaults["include_devex_year0"],
    )
    normalized["ranking_choice"] = payload.get("ranking_choice", defaults["ranking_choice"])
    normalized["min_soh"] = _coerce_float(payload.get("min_soh"), defaults["min_soh"])
    normalized["use_blended_price"] = _coerce_bool(
        payload.get("use_blended_price"),
        defaults["use_blended_price"],
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
    if "pv_market_price_php_per_kwh" in payload:
        normalized["pv_market_price_php_per_kwh"] = _coerce_float(
            payload.get("pv_market_price_php_per_kwh"),
            defaults["pv_market_price_php_per_kwh"],
        )
    else:
        normalized["pv_market_price_php_per_kwh"] = round(
            55.0 / 1000.0 * normalized["forex_rate_php_per_usd"],
            2,
        )
    if "blended_price_php_per_kwh" in payload:
        normalized["blended_price_php_per_kwh"] = _coerce_float(
            payload.get("blended_price_php_per_kwh"),
            defaults["blended_price_php_per_kwh"],
        )
    else:
        normalized["blended_price_php_per_kwh"] = normalized["contract_price_php_per_kwh"]
    normalized["escalate_prices"] = _coerce_bool(
        payload.get("escalate_prices"),
        defaults["escalate_prices"],
    )
    normalized["wesm_pricing_enabled"] = _coerce_bool(
        payload.get("wesm_pricing_enabled"),
        defaults["wesm_pricing_enabled"],
    )
    normalized["wesm_deficit_price_php_per_kwh"] = _coerce_float(
        payload.get("wesm_deficit_price_php_per_kwh"),
        defaults["wesm_deficit_price_php_per_kwh"],
    )
    normalized["sell_to_wesm"] = _coerce_bool(
        payload.get("sell_to_wesm"),
        defaults["sell_to_wesm"],
    )
    normalized["wesm_surplus_price_php_per_kwh"] = _coerce_float(
        payload.get("wesm_surplus_price_php_per_kwh"),
        defaults["wesm_surplus_price_php_per_kwh"],
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


def recommend_convergence_point(df: pd.DataFrame) -> Optional[Tuple[float, float, float]]:
    """Identify the BESS capacity where NPV and IRR curves overlap after scaling.

    The NPV and IRR charts use different axes, so we normalize each series to a
    0–1 range and locate the energy point where the curves are closest. A small
    penalty is applied to negative NPVs to avoid recommending designs with weak
    economics even if their normalized values cross. Returns ``(energy_mwh,
    npv_usd, irr_pct)`` when a convergence point can be inferred. Prefers cash-
    flow NPV when available and falls back to discounted-cost NPV otherwise.
    """

    npv_column = "npv_usd" if "npv_usd" in df.columns else "npv_costs_usd"
    columns = ["energy_mwh", npv_column, "irr_pct"]
    if not set(columns).issubset(df.columns):
        return None

    clean_df = df[columns].replace([math.inf, -math.inf], float("nan")).dropna()
    if clean_df.empty:
        return None

    npv_min, npv_max = clean_df[npv_column].min(), clean_df[npv_column].max()
    irr_min, irr_max = clean_df["irr_pct"].min(), clean_df["irr_pct"].max()
    if npv_max == npv_min or irr_max == irr_min:
        return None

    normalized = clean_df.assign(
        npv_norm=lambda x: (x[npv_column] - npv_min) / (npv_max - npv_min),
        irr_norm=lambda x: (x["irr_pct"] - irr_min) / (irr_max - irr_min),
    )

    penalty_scale = max(abs(npv_min), abs(npv_max), 1.0)
    normalized["intersection_score"] = (
        (normalized["npv_norm"] - normalized["irr_norm"]).abs()
        + (normalized[npv_column].clip(upper=0.0).abs() / penalty_scale) * 0.1
    )

    best_row = normalized.nsmallest(1, "intersection_score")
    if best_row.empty:
        return None

    chosen = best_row.iloc[0]
    return float(chosen["energy_mwh"]), float(chosen[npv_column]), float(chosen["irr_pct"])

cfg, dod_override = get_cached_simulation_config()
bootstrap_session_state(cfg)
default_forex_rate_php_per_usd = 58.0
default_contract_php_per_kwh = round(120.0 / 1000.0 * default_forex_rate_php_per_usd, 2)
default_pv_php_per_kwh = round(55.0 / 1000.0 * default_forex_rate_php_per_usd, 2)
wesm_surplus_reference_php_per_kwh = 3.29  # Weighted average WESM price (2025)
wesm_reference_php_per_mwh = 5_583.0  # 2024 Annual Market Assessment Report, PEMC
default_wesm_php_per_kwh = round(wesm_reference_php_per_mwh / 1000.0, 2)
default_wesm_surplus_php_per_kwh = round(wesm_surplus_reference_php_per_kwh, 2)

# Rehydrate PV/cycle inputs from the shared session cache (no external fetches).
pv_df, cycle_df = render_layout()

if cfg is None:
    st.warning(
        "No inputs cached yet. Open the Inputs & Results page, adjust settings, and rerun the simulation to seed the sweep.",
        icon="⚠️",
    )
    cfg = SimConfig()

st.session_state.setdefault("bess_size_sweep_results", None)

default_energy = max(10.0, cfg.initial_usable_mwh)
default_energy_range = (
    max(10.0, default_energy * 0.5),
    min(500.0, default_energy * 1.5),
)
default_inputs: Dict[str, Any] = {
    "energy_range": default_energy_range,
    "energy_steps": 5,
    "fixed_power": float(cfg.initial_power_mw),
    "wacc_pct": 8.0,
    "inflation_pct": 3.0,
    "forex_rate_php_per_usd": default_forex_rate_php_per_usd,
    "capex_musd": 40.0,
    "pv_capex_musd": 0.0,
    "fixed_opex_pct": 2.0,
    "fixed_opex_musd": 0.0,
    "include_devex_year0": False,
    "ranking_choice": "compliance_pct",
    "min_soh": 0.6,
    "use_blended_price": False,
    "contract_price_php_per_kwh": default_contract_php_per_kwh,
    "pv_market_price_php_per_kwh": default_pv_php_per_kwh,
    "blended_price_php_per_kwh": default_contract_php_per_kwh,
    "escalate_prices": False,
    "wesm_pricing_enabled": False,
    "wesm_deficit_price_php_per_kwh": default_wesm_php_per_kwh,
    "sell_to_wesm": False,
    "wesm_surplus_price_php_per_kwh": default_wesm_surplus_php_per_kwh,
    "variable_opex_php_per_kwh": 0.0,
    "variable_schedule_choice": "None",
    "periodic_variable_opex_usd": 0.0,
    "periodic_variable_opex_interval_years": 5,
    "variable_opex_schedule_usd": [],
}
stored_inputs = st.session_state.get("bess_sweep_inputs", {})
if isinstance(stored_inputs, dict):
    default_inputs = _normalize_sweep_inputs(stored_inputs, default_inputs)

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
        placeholder='{"energy_range": [25, 75], "fixed_power": 50}',
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
    size_col1, size_col2, size_col3, price_col = st.columns(4)
    with size_col1:
        energy_range = st.slider(
            "Usable energy range (MWh)",
            min_value=10.0,
            max_value=500.0,
            value=default_inputs["energy_range"],
            step=5.0,
            help="Lower and upper bounds for the usable MWh grid.",
        )
        energy_steps = st.number_input(
            "Energy points",
            min_value=1,
            max_value=15,
            value=int(default_inputs["energy_steps"]),
            help="Number of evenly spaced usable-energy values between the bounds.",
        )
        fixed_power = st.number_input(
            "Fixed discharge power (MW)",
            min_value=0.1,
            max_value=300.0,
            value=float(default_inputs["fixed_power"]),
            step=0.1,
            help="Power rating held constant while sweeping usable energy.",
        )

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
            "BESS CAPEX (USD million)",
            min_value=0.0,
            value=float(default_inputs["capex_musd"]),
            step=0.1,
            help="BESS-only CAPEX. Combined with PV CAPEX for total project spend.",
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
        fixed_opex_pct = st.number_input(
            "Fixed OPEX (% of CAPEX per year)",
            min_value=0.0,
            max_value=20.0,
            value=float(default_inputs["fixed_opex_pct"]),
            step=0.1,
            help="Annual fixed OPEX expressed as % of CAPEX.",
        ) / 100.0
        fixed_opex_musd = st.number_input(
            "Additional fixed OPEX (USD million/yr)",
            min_value=0.0,
            value=float(default_inputs["fixed_opex_musd"]),
            step=0.1,
            help="Extra fixed OPEX not tied to CAPEX percentage.",
        )
        devex_cost_usd = DEVEX_COST_PHP / forex_rate_php_per_usd if forex_rate_php_per_usd else 0.0
        include_devex_year0 = st.checkbox(
            "Include ₱100M DevEx at year 0",
            value=bool(default_inputs["include_devex_year0"]),
            help=(
                "Adds a fixed ₱100 million development expenditure upfront (≈"
                f"${devex_cost_usd / 1_000_000:,.2f}M using PHP {forex_rate_php_per_usd:,.0f}/USD). "
                "Flows through discounted costs, LCOE/LCOS, NPV, and IRR."
            ),
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
        st.caption(
            "Discount rate is derived from WACC and inflation to align with the economics helper."
        )

    with price_col:
        use_blended_price = st.checkbox(
            "Use blended energy price",
            value=bool(default_inputs["use_blended_price"]),
            help=(
                "Apply a single price to all delivered firm energy and excess PV. "
                "Contract/PV-specific inputs are ignored while enabled."
            ),
        )
        contract_price_php_per_kwh = st.number_input(
            "Contract price (PHP/kWh from BESS)",
            min_value=0.0,
            value=float(default_inputs["contract_price_php_per_kwh"]),
            step=0.05,
            help="Price converted to USD/MWh internally using the FX rate above.",
            disabled=use_blended_price,
        )
        pv_market_price_php_per_kwh = st.number_input(
            "PV market price (PHP/kWh for excess PV)",
            min_value=0.0,
            value=float(default_inputs["pv_market_price_php_per_kwh"]),
            step=0.05,
            help="Price converted to USD/MWh internally using the FX rate above.",
            disabled=use_blended_price,
        )
        blended_price_php_per_kwh = st.number_input(
            "Blended energy price (PHP/kWh)",
            min_value=0.0,
            value=float(default_inputs["blended_price_php_per_kwh"]),
            step=0.05,
            help="Applied to all delivered firm energy and marketed PV when blended pricing is enabled.",
            disabled=not use_blended_price,
        )
        escalate_prices = st.checkbox(
            "Escalate prices with inflation",
            value=bool(default_inputs["escalate_prices"]),
        )
        wesm_pricing_enabled = st.checkbox(
            "Apply WESM pricing to contract shortfalls",
            value=bool(default_inputs["wesm_pricing_enabled"]),
            help=(
                "Defaults to PHP 5,583/MWh from the 2024 Annual Market Assessment Report (PEMC);"
                " enter a PHP/kWh rate to override."
            ),
        )
        wesm_deficit_price_php_per_kwh = st.number_input(
            "WESM deficit price for contract shortfalls (PHP/kWh)",
            min_value=0.0,
            value=float(default_inputs["wesm_deficit_price_php_per_kwh"]),
            step=0.05,
            help=(
                "Applied to contract shortfall MWh (annual_shortfall_mwh) as a purchase cost."
                " This is separate from the PV surplus sale price."
                " Defaults to PHP 5,583/MWh from the 2024 Annual Market Assessment Report (PEMC)."
            ),
            disabled=not wesm_pricing_enabled,
        )
        sell_to_wesm = st.checkbox(
            "Sell PV surplus to WESM",
            value=bool(default_inputs["sell_to_wesm"]),
            help=(
                "When enabled, PV surplus (excess MWh) is credited at a WESM sale price; otherwise surplus "
                "is excluded from revenue. This does not change the deficit price applied to shortfalls."
            ),
            disabled=not wesm_pricing_enabled,
        )
        wesm_surplus_price_php_per_kwh = st.number_input(
            "WESM sale price for PV surplus (PHP/kWh)",
            min_value=0.0,
            value=float(default_inputs["wesm_surplus_price_php_per_kwh"]),
            step=0.05,
            help=(
                "Used only when selling PV surplus. Defaults to PHP 3.29/kWh based on the 2025 weighted"
                " average WESM price; adjust to use your own PHP/kWh rate. This does not affect shortfall pricing."
            ),
            disabled=not (wesm_pricing_enabled and sell_to_wesm),
        )

        contract_price = contract_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
        pv_market_price = pv_market_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
        blended_price_usd_per_mwh: Optional[float] = None
        wesm_deficit_price_usd_per_mwh: Optional[float] = None
        wesm_surplus_price_usd_per_mwh: Optional[float] = None
        if use_blended_price:
            blended_price_usd_per_mwh = blended_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
            st.caption(
                "Blended price active for revenues: "
                f"PHP {blended_price_php_per_kwh:,.2f}/kWh "
                f"(≈${blended_price_usd_per_mwh:,.2f}/MWh). Contract/PV prices are ignored."
            )
        else:
            st.caption(
                f"Converted contract price: ${contract_price:,.2f}/MWh | PV market price: ${pv_market_price:,.2f}/MWh"
            )
        if wesm_pricing_enabled:
            wesm_deficit_price_usd_per_mwh = (
                wesm_deficit_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
            )
            wesm_surplus_price_usd_per_mwh = (
                wesm_surplus_price_php_per_kwh / forex_rate_php_per_usd * 1000.0 if sell_to_wesm else None
            )
            st.caption(
                "WESM deficit pricing active for contract shortfalls: "
                f"PHP {wesm_deficit_price_php_per_kwh:,.2f}/kWh "
                f"(≈${wesm_deficit_price_usd_per_mwh:,.2f}/MWh)."
                " Defaults to PHP 5,583/MWh from the 2024 Annual Market Assessment Report (PEMC)."
            )
            if sell_to_wesm and wesm_surplus_price_usd_per_mwh is not None:
                st.caption(
                    "PV surplus credited at a separate WESM sale rate: "
                    f"PHP {wesm_surplus_price_php_per_kwh:,.2f}/kWh "
                    f"(≈${wesm_surplus_price_usd_per_mwh:,.2f}/MWh)."
                    " Edit the PHP/kWh value to use a custom surplus rate."
                )
        else:
            st.caption(
                "WESM pricing is ignored unless the shortfall toggle is enabled."
                " Defaults to PHP 5,583/MWh from the 2024 Annual Market Assessment Report (PEMC)."
            )

    variable_col1, variable_col2 = st.columns(2)
    with variable_col1:
        variable_opex_php_per_kwh = st.number_input(
            "Variable OPEX (PHP/kWh)",
            min_value=0.0,
            value=float(default_inputs["variable_opex_php_per_kwh"]),
            step=0.05,
            help=(
                "Optional per-kWh operating expense applied to annual firm energy. "
                "Escalates with inflation and overrides fixed OPEX when provided."
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
                "Custom or periodic schedules override per-kWh and fixed OPEX assumptions. "
                "Per-kWh costs override fixed percentages and adders."
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

    st.session_state["bess_sweep_inputs"] = {
        "energy_range": energy_range,
        "energy_steps": int(energy_steps),
        "fixed_power": float(fixed_power),
        "wacc_pct": float(wacc_pct),
        "inflation_pct": float(inflation_pct),
        "forex_rate_php_per_usd": float(forex_rate_php_per_usd),
        "capex_musd": float(capex_musd),
        "pv_capex_musd": float(pv_capex_musd),
        "fixed_opex_pct": float(fixed_opex_pct * 100.0),
        "fixed_opex_musd": float(fixed_opex_musd),
        "include_devex_year0": bool(include_devex_year0),
        "ranking_choice": ranking_choice,
        "min_soh": float(min_soh),
        "use_blended_price": bool(use_blended_price),
        "contract_price_php_per_kwh": float(contract_price_php_per_kwh),
        "pv_market_price_php_per_kwh": float(pv_market_price_php_per_kwh),
        "blended_price_php_per_kwh": float(blended_price_php_per_kwh),
        "escalate_prices": bool(escalate_prices),
        "wesm_pricing_enabled": bool(wesm_pricing_enabled),
        "wesm_deficit_price_php_per_kwh": float(wesm_deficit_price_php_per_kwh),
        "sell_to_wesm": bool(sell_to_wesm),
        "wesm_surplus_price_php_per_kwh": float(wesm_surplus_price_php_per_kwh),
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
    energy_values = generate_values(energy_range[0], energy_range[1], int(energy_steps))
    economics_inputs = EconomicInputs(
        capex_musd=capex_musd,
        pv_capex_musd=pv_capex_musd,
        fixed_opex_pct_of_capex=fixed_opex_pct,
        fixed_opex_musd=fixed_opex_musd,
        inflation_rate=inflation_pct / 100.0,
        discount_rate=discount_rate,
        variable_opex_usd_per_mwh=variable_opex_usd_per_mwh,
        variable_opex_schedule_usd=variable_opex_schedule_usd,
        periodic_variable_opex_usd=periodic_variable_opex_usd,
        periodic_variable_opex_interval_years=periodic_variable_opex_interval_years,
        devex_cost_usd=devex_cost_usd,
        include_devex_year0=include_devex_year0,
    )
    price_inputs = PriceInputs(
        contract_price_usd_per_mwh=contract_price,
        pv_market_price_usd_per_mwh=pv_market_price,
        escalate_with_inflation=escalate_prices,
        blended_price_usd_per_mwh=blended_price_usd_per_mwh,
        wesm_deficit_price_usd_per_mwh=wesm_deficit_price_usd_per_mwh,
        wesm_surplus_price_usd_per_mwh=wesm_surplus_price_usd_per_mwh
        if wesm_pricing_enabled and sell_to_wesm
        else None,
        apply_wesm_to_shortfall=wesm_pricing_enabled,
        sell_to_wesm=sell_to_wesm if wesm_pricing_enabled else False,
    )

    with st.spinner("Running BESS energy sweep..."):
        sweep_kwargs = dict(
            base_cfg=cfg,
            pv_df=pv_df,
            cycle_df=cycle_df,
            dod_override=dod_override,
            energy_mwh_values=energy_values,
            fixed_power_mw=fixed_power,
            economics_inputs=economics_inputs,
            price_inputs=price_inputs,
            ranking_kpi=ranking_choice,
            min_soh=min_soh,
            use_case="reliability",
        )

        try:
            sweep_df = sweep_bess_sizes(**sweep_kwargs)
        except TypeError as exc:
            # Backwards-compatibility for environments still running an older sweep implementation
            # that lacks newer keyword arguments. Gracefully retry without the missing inputs.
            message = str(exc)
            if "price_inputs" in message:
                sweep_kwargs.pop("price_inputs", None)
                try:
                    sweep_df = sweep_bess_sizes(**sweep_kwargs)
                except TypeError as inner_exc:
                    if "energy_mwh_values" not in str(inner_exc):
                        raise
                    duration_values = [energy / fixed_power for energy in energy_values if fixed_power > 0]
                    sweep_df = sweep_bess_sizes(
                        cfg,
                        pv_df,
                        cycle_df,
                        dod_override,
                        power_mw_values=[fixed_power],
                        duration_h_values=duration_values,
                        economics_inputs=economics_inputs,
                        ranking_kpi=ranking_choice,
                        min_soh=min_soh,
                        use_case="reliability",
                    )
                sweep_kwargs["price_inputs"] = price_inputs
            elif "energy_mwh_values" in message:
                duration_values = [energy / fixed_power for energy in energy_values if fixed_power > 0]
                sweep_df = sweep_bess_sizes(
                    cfg,
                    pv_df,
                    cycle_df,
                    dod_override,
                    power_mw_values=[fixed_power],
                    duration_h_values=duration_values,
                    economics_inputs=economics_inputs,
                    price_inputs=price_inputs,
                    ranking_kpi=ranking_choice,
                    min_soh=min_soh,
                    use_case="reliability",
                )
            else:
                raise

    if sweep_df.empty:
        st.info("No sweep results generated; widen the ranges and try again.")
        st.session_state["bess_size_sweep_results"] = None
    else:
        st.session_state["bess_size_sweep_results"] = sweep_df
        st.toast("BESS energy sweep complete.")

sweep_df = st.session_state.get("bess_size_sweep_results")
if sweep_df is not None:
    convergence_point = recommend_convergence_point(sweep_df)
    if convergence_point:
        energy_mwh, npv_usd, irr_pct = convergence_point
        npv_label = (
            "net NPV" if "npv_usd" in sweep_df.columns and sweep_df["npv_usd"].notna().any() else "NPV of costs"
        )
        st.info(
            "Convergence point (NPV vs IRR): "
            f"~{energy_mwh:.1f} MWh usable with IRR {irr_pct:.2f}% and {npv_label} ${npv_usd:,.0f}. "
            "Curves are normalized to locate where returns and discounted costs align, "
            "favoring options that avoid very negative NPVs when CAPEX scales linearly "
            "with BESS size and resource availability limits upside energy."
        )

    npv_field = (
        "npv_usd"
        if "npv_usd" in sweep_df.columns and sweep_df["npv_usd"].notna().any()
        else "npv_costs_usd"
    )
    npv_axis_title = "Net NPV (USD)" if npv_field == "npv_usd" else "NPV of costs (USD)"

    chart_df = sweep_df[["energy_mwh", npv_field]].copy()
    chart_df["irr_pct"] = sweep_df.get("irr_pct", float("nan"))
    chart_df["is_best"] = sweep_df.get("is_best", False)
    chart_df = chart_df.sort_values("energy_mwh")

    base_chart = alt.Chart(chart_df).encode(
        x=alt.X("energy_mwh", title="BESS capacity (MWh)", axis=alt.Axis(format=",.0f"))
    )

    point_tooltip = [
        alt.Tooltip("energy_mwh", title="BESS capacity (MWh)", format=",.0f"),
        alt.Tooltip(npv_field, title=npv_axis_title, format=",.0f"),
        alt.Tooltip("irr_pct", title="IRR (%)", format=",.2f"),
    ]

    npv_line = base_chart.mark_line(color="#0b2c66", point=alt.OverlayMarkDef(filled=True, size=90)).encode(
        y=alt.Y(
            npv_field,
            title=npv_axis_title,
            axis=alt.Axis(titleColor="#0b2c66", format=",.0f", orient="left"),
        ),
        tooltip=point_tooltip,
    )

    irr_line = base_chart.mark_line(color="#88c5de", point=alt.OverlayMarkDef(filled=True, size=90)).encode(
        y=alt.Y(
            "irr_pct",
            title="IRR (%)",
            axis=alt.Axis(
                titleColor="#88c5de",
                orient="right",
                format=",.2f",
                labelExpr="datum.label + '%'",
            ),
        ),
        tooltip=point_tooltip,
    )

    zero_line = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#bbbbbb", strokeDash=[4, 4]).encode(
        y=alt.Y("y:Q", axis=None)
    )

    best_point = (
        base_chart.transform_filter(alt.datum.is_best == True)  # noqa: E712 - Altair predicate
        .mark_circle(color="#f57c00", size=120)
        .encode(y=alt.Y(npv_field, axis=None), tooltip=point_tooltip)
    )

    convergence_overlay = None
    if convergence_point:
        convergence_df = pd.DataFrame(
            {"energy_mwh": [convergence_point[0]], npv_field: [convergence_point[1]], "irr_pct": [convergence_point[2]]}
        )
        convergence_overlay = alt.Chart(convergence_df).mark_point(
            color="#b3006e",
            size=140,
            shape="diamond",
            filled=True,
        ).encode(x="energy_mwh", y=alt.Y(npv_field, axis=None), tooltip=point_tooltip)

    layers = [zero_line, npv_line, irr_line, best_point]
    if convergence_overlay is not None:
        layers.append(convergence_overlay)

    st.altair_chart(
        alt.layer(*layers).resolve_scale(y="independent"),
        use_container_width=True,
    )
    st.caption(
        "Dual-axis line chart overlays NPV and IRR across BESS capacities; IRR points are omitted when unavailable. "
        "Net NPV is displayed when cash-flow assumptions are provided, otherwise discounted costs are shown."
    )
else:
    st.info(
        "Run the sweep with your latest inputs. Results persist in the session for quick iteration.",
        icon="ℹ️",
    )
