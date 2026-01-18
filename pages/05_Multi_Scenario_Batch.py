from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from app import BASE_DIR
from services.simulation_core import HourlyLog, SimConfig, Window, parse_windows, simulate_project, summarize_simulation
from utils import enforce_rate_limit, parse_numeric_series, read_wesm_profile
from utils.io import read_wesm_forecast_profile_average
from utils.economics import (
    DEFAULT_COST_OF_DEBT_PCT,
    DEFAULT_DEBT_EQUITY_RATIO,
    DEFAULT_FOREX_RATE_PHP_PER_USD,
    DEFAULT_TENOR_YEARS,
    DEVEX_COST_PHP,
    EconomicInputs,
    PriceInputs,
    aggregate_wesm_profile_to_annual,
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
    get_cached_simulation_config,
    get_latest_economics_payload,
)

render_layout = init_page_layout(
    page_title="Multi-scenario batch",
    main_title="Multi-scenario batch runner",
    description=(
        "Queue multiple simulation variants at once. Per-scenario logs are disabled to conserve memory; "
        "rerun the main page for detailed charts."
    ),
    base_dir=BASE_DIR,
)

bootstrap_session_state()


def _default_wesm_profile_path(use_wesm_forecast: bool) -> Any:
    filename = (
        "wesm_price_profile_forecast.csv"
        if use_wesm_forecast
        else "wesm_price_profile_historical.csv"
    )
    return BASE_DIR / "data" / filename


def _build_hourly_summary_df(logs: HourlyLog) -> pd.DataFrame:
    """Build an hourly summary table for WESM profile aggregation."""

    hour_index = np.arange(len(logs.hod), dtype=int)
    data: Dict[str, Any] = {
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
    return df

def _format_hhmm(hour_value: float) -> str:
    """Return HH:MM text for a fractional hour."""

    hours = int(hour_value)
    minutes = int(round((hour_value - hours) * 60))
    if minutes == 60:
        hours = (hours + 1) % 24
        minutes = 0
    return f"{hours:02d}:{minutes:02d}"


def _windows_to_text(windows: List[Window]) -> str:
    """Serialize Window objects to HH:MM-HH:MM strings."""

    return ", ".join(f"{_format_hhmm(w.start)}-{_format_hhmm(w.end)}" for w in windows)


def _seed_rows(cfg: SimConfig) -> pd.DataFrame:
    """Construct the initial scenario table from cached inputs or defaults."""

    defaults: Dict[str, Any] = {
        "label": "Scenario 1",
        "initial_power_mw": cfg.initial_power_mw,
        "initial_usable_mwh": cfg.initial_usable_mwh,
        "contracted_mw": cfg.contracted_mw,
        "years": cfg.years,
        "pv_degradation_rate": cfg.pv_deg_rate,
        "bess_availability": cfg.bess_availability,
        "rte": cfg.rte_roundtrip,
        "soc_floor": cfg.soc_floor,
        "soc_ceiling": cfg.soc_ceiling,
        "calendar_fade_rate": cfg.calendar_fade_rate,
        "use_calendar_exp_model": cfg.use_calendar_exp_model,
        "discharge_windows": _windows_to_text(cfg.discharge_windows),
        "charge_windows": cfg.charge_windows_text or "",
        "augmentation": cfg.augmentation,
        "aug_trigger_type": cfg.aug_trigger_type,
        "aug_threshold_margin": cfg.aug_threshold_margin,
        "aug_topup_margin": cfg.aug_topup_margin,
        "aug_soh_trigger_pct": cfg.aug_soh_trigger_pct,
        "aug_soh_add_frac_initial": cfg.aug_soh_add_frac_initial,
        "aug_periodic_every_years": cfg.aug_periodic_every_years,
        "aug_periodic_add_frac_of_bol": cfg.aug_periodic_add_frac_of_bol,
        "aug_add_mode": cfg.aug_add_mode,
        "aug_fixed_energy_mwh": cfg.aug_fixed_energy_mwh,
        "aug_retire_old_cohort": cfg.aug_retire_old_cohort,
        "aug_retire_soh_pct": cfg.aug_retire_soh_pct,
        "aug_retire_replacement_mode": cfg.aug_retire_replacement_mode,
        "aug_retire_replacement_pct_bol": cfg.aug_retire_replacement_pct_bol,
        "aug_retire_replacement_fixed_mwh": cfg.aug_retire_replacement_fixed_mwh,
    }
    return pd.DataFrame([defaults])


def _baseline_template(cfg: SimConfig) -> Dict[str, Any]:
    """Create a baseline row that mirrors the cached configuration."""

    row = _seed_rows(cfg).iloc[0].to_dict()
    row["label"] = "Baseline"
    return row


def _high_availability_template(cfg: SimConfig) -> Dict[str, Any]:
    """Create a row favoring availability and wider windows."""

    row = _seed_rows(cfg).iloc[0].to_dict()
    row.update(
        {
            "label": "High availability",
            "bess_availability": max(0.0, min(1.0, cfg.bess_availability + 0.02)),
            "discharge_windows": _windows_to_text(cfg.discharge_windows)
            or "00:00-23:59",
            "charge_windows": cfg.charge_windows_text or "00:00-23:59",
        }
    )
    return row


def _aggressive_augmentation_template(cfg: SimConfig) -> Dict[str, Any]:
    """Create a row that tests frequent augmentation."""

    row = _seed_rows(cfg).iloc[0].to_dict()
    row.update(
        {
            "label": "Aggressive augmentation",
            "augmentation": "Threshold",
            "aug_trigger_type": "Capability",
            "aug_threshold_margin": 0.05,
            "aug_topup_margin": 0.1,
            "aug_add_mode": "Percent",
            "aug_periodic_every_years": max(1, int(cfg.years / 5)),
        }
    )
    return row


def _parse_row_to_config(row: pd.Series, template: SimConfig) -> Tuple[str, SimConfig]:
    """Apply row overrides to a SimConfig copy, validating fields along the way."""

    label = str(row.get("label") or "Scenario")
    config = deepcopy(template)

    dis_windows_text = str(row.get("discharge_windows") or "").strip()
    dis_windows, dis_warnings = parse_windows(dis_windows_text)

    charge_windows_text = str(row.get("charge_windows") or template.charge_windows_text or "").strip()
    charge_windows, charge_warnings = parse_windows(charge_windows_text)
    if dis_warnings or charge_warnings:
        raise ValueError("; ".join(dis_warnings + charge_warnings))
    if not dis_windows:
        raise ValueError("Please provide at least one discharge window.")

    years = int(row.get("years") or template.years)
    soc_floor = float(row.get("soc_floor") or template.soc_floor)
    soc_ceiling = float(row.get("soc_ceiling") or template.soc_ceiling)

    config.years = max(1, years)
    config.initial_power_mw = float(row.get("initial_power_mw") or template.initial_power_mw)
    config.initial_usable_mwh = float(row.get("initial_usable_mwh") or template.initial_usable_mwh)
    config.contracted_mw = float(row.get("contracted_mw") or template.contracted_mw)
    config.pv_deg_rate = float(row.get("pv_degradation_rate") or template.pv_deg_rate)
    config.bess_availability = float(row.get("bess_availability") or template.bess_availability)
    config.rte_roundtrip = float(row.get("rte") or template.rte_roundtrip)
    config.soc_floor = soc_floor
    config.soc_ceiling = soc_ceiling
    config.calendar_fade_rate = float(row.get("calendar_fade_rate") or template.calendar_fade_rate)
    use_exp_value = row.get("use_calendar_exp_model")
    config.use_calendar_exp_model = (
        template.use_calendar_exp_model if pd.isna(use_exp_value) else bool(use_exp_value)
    )
    config.discharge_windows = dis_windows
    config.charge_windows_text = charge_windows_text
    config.charge_windows = charge_windows
    config.augmentation = str(row.get("augmentation") or template.augmentation)
    config.aug_trigger_type = str(row.get("aug_trigger_type") or template.aug_trigger_type)
    config.aug_threshold_margin = float(row.get("aug_threshold_margin") or template.aug_threshold_margin)
    config.aug_topup_margin = float(row.get("aug_topup_margin") or template.aug_topup_margin)
    config.aug_soh_trigger_pct = float(row.get("aug_soh_trigger_pct") or template.aug_soh_trigger_pct)
    config.aug_soh_add_frac_initial = float(
        row.get("aug_soh_add_frac_initial") or template.aug_soh_add_frac_initial
    )
    config.aug_periodic_every_years = int(
        row.get("aug_periodic_every_years") or template.aug_periodic_every_years
    )
    config.aug_periodic_add_frac_of_bol = float(
        row.get("aug_periodic_add_frac_of_bol") or template.aug_periodic_add_frac_of_bol
    )
    config.aug_add_mode = str(row.get("aug_add_mode") or template.aug_add_mode)
    config.aug_fixed_energy_mwh = float(row.get("aug_fixed_energy_mwh") or template.aug_fixed_energy_mwh)
    retire_value = row.get("aug_retire_old_cohort")
    config.aug_retire_old_cohort = template.aug_retire_old_cohort if pd.isna(retire_value) else bool(retire_value)
    config.aug_retire_soh_pct = float(row.get("aug_retire_soh_pct") or template.aug_retire_soh_pct)
    config.aug_retire_replacement_mode = str(
        row.get("aug_retire_replacement_mode") or template.aug_retire_replacement_mode
    )
    config.aug_retire_replacement_pct_bol = float(
        row.get("aug_retire_replacement_pct_bol") or template.aug_retire_replacement_pct_bol
    )
    config.aug_retire_replacement_fixed_mwh = float(
        row.get("aug_retire_replacement_fixed_mwh") or template.aug_retire_replacement_fixed_mwh
    )

    return label, config


def _validate_row(row: pd.Series, idx: int) -> List[str]:
    """Validate a scenario row and return any error messages."""

    errors: List[str] = []
    dis_windows_text = str(row.get("discharge_windows") or "").strip()
    dis_windows, dis_warnings = parse_windows(dis_windows_text)
    errors.extend(dis_warnings)
    if not dis_windows:
        errors.append("Missing discharge windows (HH:MM-HH:MM).")

    years_value = row.get("years")
    try:
        years_int = int(years_value)
        if years_int <= 0:
            errors.append("Years must be positive.")
    except (TypeError, ValueError):
        errors.append("Years must be a positive integer.")

    try:
        soc_floor = float(row.get("soc_floor"))
        soc_ceiling = float(row.get("soc_ceiling"))
        if soc_floor >= soc_ceiling:
            errors.append("SOC floor must be lower than SOC ceiling.")
    except (TypeError, ValueError):
        errors.append("SOC floor/ceiling must be numeric.")

    return [f"Row {idx + 1}: {msg}" for msg in errors]


def _expected_column_order() -> List[str]:
    """Return the canonical column order for scenario entry and rendering."""

    return [
        "label",
        # System sizing
        "initial_power_mw",
        "initial_usable_mwh",
        "contracted_mw",
        "years",
        "pv_degradation_rate",
        # Operations windows + operations levers
        "discharge_windows",
        "charge_windows",
        "bess_availability",
        "rte",
        "soc_floor",
        "soc_ceiling",
        "calendar_fade_rate",
        "use_calendar_exp_model",
        # Augmentation
        "augmentation",
        "aug_trigger_type",
        "aug_threshold_margin",
        "aug_topup_margin",
        "aug_soh_trigger_pct",
        "aug_soh_add_frac_initial",
        "aug_periodic_every_years",
        "aug_periodic_add_frac_of_bol",
        "aug_add_mode",
        "aug_fixed_energy_mwh",
        "aug_retire_old_cohort",
        "aug_retire_soh_pct",
        "aug_retire_replacement_mode",
        "aug_retire_replacement_pct_bol",
        "aug_retire_replacement_fixed_mwh",
    ]


def _normalize_table(df: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    """Align uploaded/pasted scenario tables to the expected schema.

    Missing columns are filled with defaults from the cached configuration to
    avoid reruns dropping user edits. Unknown columns are ignored.
    """

    defaults = _seed_rows(cfg).iloc[0]
    expected_cols = _expected_column_order()
    normalized = df.copy()
    for col in expected_cols:
        if col not in normalized.columns:
            normalized[col] = defaults[col]
    normalized = normalized[expected_cols]
    return normalized.fillna(defaults)


def _read_table_from_text(payload: str, cfg: SimConfig) -> Optional[pd.DataFrame]:
    """Parse CSV or JSON text into a scenario table if possible."""

    text = payload.strip()
    if not text:
        return None

    try:
        df = pd.read_json(text)
    except ValueError:
        try:
            df = pd.read_csv(StringIO(text))
        except Exception as exc:
            st.error(f"Could not parse table input: {exc}")
            return None
    return _normalize_table(df, cfg)


def _render_summary_table(df: pd.DataFrame) -> None:
    """Render a compact summary of key scenario inputs."""

    if df is None or df.empty:
        st.info("No scenarios defined yet. Add rows to see a summary.", icon="ℹ️")
        return

    summary_df = df[
        [
            "label",
            "initial_power_mw",
            "initial_usable_mwh",
            "contracted_mw",
            "years",
            "discharge_windows",
            "augmentation",
        ]
    ].rename(
        columns={
            "label": "Label",
            "initial_power_mw": "Power (MW)",
            "initial_usable_mwh": "Usable (MWh)",
            "contracted_mw": "Contracted (MW)",
            "years": "Years",
            "discharge_windows": "Discharge windows",
            "augmentation": "Augmentation",
        }
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)


caching_cfg, dod_override = get_cached_simulation_config()
cached_cfg: SimConfig = caching_cfg or SimConfig()
bootstrap_session_state(cached_cfg)

# Pull PV/cycle inputs from the shared session cache instead of any external API.
pv_df, cycle_df = render_layout()

econ_defaults = get_latest_economics_payload() or {}
econ_inputs_default: Optional[EconomicInputs] = econ_defaults.get("economic_inputs")
econ_price_default: Optional[PriceInputs] = econ_defaults.get("price_inputs")

with st.expander("Economics (optional)", expanded=False):
    econ_col1, econ_col2, econ_col3 = st.columns(3)
    with econ_col1:
        wacc_pct = st.number_input(
            "WACC (%)",
            min_value=0.0,
            max_value=30.0,
            value=float(econ_inputs_default.wacc * 100.0) if econ_inputs_default else 8.0,
            step=0.1,
        )
        inflation_pct = st.number_input(
            "Inflation rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=float(econ_inputs_default.inflation_rate * 100.0) if econ_inputs_default else 3.0,
            step=0.1,
            help="Used to derive the real discount rate applied to costs and revenues.",
        )
        discount_rate = max((1 + wacc_pct / 100.0) / (1 + inflation_pct / 100.0) - 1, 0.0)
        st.caption(f"Real discount rate derived from WACC and inflation: {discount_rate * 100:.2f}%.")
        forex_rate_php_per_usd = st.number_input(
            "FX rate (PHP/USD)",
            min_value=1.0,
            value=float(econ_inputs_default.forex_rate_php_per_usd)
            if econ_inputs_default
            else float(DEFAULT_FOREX_RATE_PHP_PER_USD),
            step=0.5,
            help="Used to convert PHP-denominated inputs (prices, OPEX, DevEx) to USD.",
        )

    default_contract_php_per_kwh = round(120.0 / 1000.0 * forex_rate_php_per_usd, 2)

    with econ_col2:
        capex_mode_default = "Total CAPEX (USD million)"
        if econ_inputs_default and econ_inputs_default.capex_usd_per_kwh is not None:
            capex_mode_default = "USD/kWh (BOL)"
        capex_mode = st.radio(
            "BESS CAPEX input",
            options=["USD/kWh (BOL)", "Total CAPEX (USD million)"],
            index=["USD/kWh (BOL)", "Total CAPEX (USD million)"].index(capex_mode_default),
            horizontal=True,
            help="Enter BESS CAPEX as a unit rate per kWh of BOL energy or override with a total USD million value.",
        )
        capex_usd_per_kwh = 0.0
        capex_musd = 0.0
        bess_bol_kwh_default = cached_cfg.initial_usable_mwh * 1000.0
        if capex_mode == "USD/kWh (BOL)":
            default_capex_usd_per_kwh = 0.0
            if econ_inputs_default and econ_inputs_default.capex_usd_per_kwh is not None:
                default_capex_usd_per_kwh = float(econ_inputs_default.capex_usd_per_kwh)
            elif bess_bol_kwh_default > 0:
                default_capex_usd_per_kwh = 40_000_000.0 / bess_bol_kwh_default
            capex_usd_per_kwh = st.number_input(
                "CAPEX (USD/kWh, BOL)",
                min_value=0.0,
                value=round(default_capex_usd_per_kwh, 2),
                step=1.0,
                help="Applied to BOL usable energy (kWh) to derive total CAPEX in USD.",
            )
            capex_total_usd = capex_usd_per_kwh * bess_bol_kwh_default
            capex_musd = capex_total_usd / 1_000_000.0
        else:
            default_capex_musd = 40.0
            if econ_inputs_default:
                if econ_inputs_default.capex_musd is not None:
                    default_capex_musd = float(econ_inputs_default.capex_musd)
                elif econ_inputs_default.capex_total_usd is not None:
                    default_capex_musd = float(econ_inputs_default.capex_total_usd) / 1_000_000.0
            capex_musd = st.number_input(
                "Total BESS CAPEX (USD million)",
                min_value=0.0,
                value=float(default_capex_musd),
                step=0.1,
            )
        pv_capex_musd = st.number_input(
            "PV CAPEX (USD million)",
            min_value=0.0,
            value=float(econ_inputs_default.pv_capex_musd) if econ_inputs_default else 0.0,
            step=0.1,
            help="Standalone PV CAPEX added to the BESS CAPEX input above.",
        )
        total_capex_musd = capex_musd + pv_capex_musd
        st.caption(f"Total project CAPEX (BESS + PV): ${total_capex_musd:,.2f}M.")
        opex_mode_default = "% of CAPEX per year"
        if econ_inputs_default and econ_inputs_default.opex_php_per_kwh is not None:
            opex_mode_default = "PHP/kWh on total generation"
        opex_mode = st.radio(
            "OPEX input",
            options=["% of CAPEX per year", "PHP/kWh on total generation"],
            index=["% of CAPEX per year", "PHP/kWh on total generation"].index(opex_mode_default),
            horizontal=True,
            help="Choose a fixed % of CAPEX/year or a PHP/kWh rate applied to total generation.",
        )
        fixed_opex_pct = 0.0
        opex_php_per_kwh: Optional[float] = None
        if opex_mode == "% of CAPEX per year":
            fixed_opex_pct = st.number_input(
                "Fixed OPEX (% of CAPEX per year)",
                min_value=0.0,
                max_value=20.0,
                value=float(econ_inputs_default.fixed_opex_pct_of_capex) if econ_inputs_default else 2.0,
                step=0.1,
            )
        else:
            opex_php_default = (
                float(econ_inputs_default.opex_php_per_kwh)
                if econ_inputs_default and econ_inputs_default.opex_php_per_kwh is not None
                else 0.0
            )
            opex_php_per_kwh = st.number_input(
                "OPEX (PHP/kWh on total generation)",
                min_value=0.0,
                value=opex_php_default,
                step=0.05,
                help="Converted to USD/MWh using the FX rate; applied to total generation.",
            )
            if opex_php_per_kwh > 0:
                opex_usd_per_mwh = opex_php_per_kwh / forex_rate_php_per_usd * 1000.0
                st.caption(f"Converted OPEX: ${opex_usd_per_mwh:,.2f}/MWh.")
        fixed_opex_musd = st.number_input(
            "Additional fixed OPEX (USD million/yr)",
            min_value=0.0,
            value=float(econ_inputs_default.fixed_opex_musd) if econ_inputs_default else 0.0,
            step=0.1,
        )
        devex_choice = st.radio(
            "DevEx at year 0",
            options=["Exclude", "Include"],
            index=1
            if econ_inputs_default and econ_inputs_default.include_devex_year0
            else 0,
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
            value=float(econ_inputs_default.devex_cost_php) if econ_inputs_default else float(DEVEX_COST_PHP),
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

    with econ_col3:
        contract_price_php_per_kwh = st.number_input(
            "Contract price (PHP/kWh for delivered energy)",
            min_value=0.0,
            value=float(econ_price_default.contract_price_usd_per_mwh * forex_rate_php_per_usd / 1000.0)
            if econ_price_default
            else default_contract_php_per_kwh,
            step=0.05,
        )
        escalate_prices = st.checkbox(
            "Escalate prices with inflation",
            value=bool(econ_price_default.escalate_with_inflation) if econ_price_default else False,
        )
        wesm_pricing_enabled = st.checkbox(
            "Apply WESM pricing to contract shortfalls",
            value=bool(econ_price_default.apply_wesm_to_shortfall) if econ_price_default else False,
            help=(
                "Uses the uploaded (or bundled) hourly WESM profile to price contract shortfalls."
            ),
        )
        sell_to_wesm = st.checkbox(
            "Sell PV surplus to WESM",
            value=bool(econ_price_default.sell_to_wesm) if econ_price_default else False,
            help=(
                "When enabled, PV surplus (excess MWh) is credited at a WESM sale price; otherwise surplus "
                "is excluded from revenue. Pricing comes from the hourly WESM profile."
            ),
            disabled=not wesm_pricing_enabled,
        )
    contract_price_usd_per_mwh = contract_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
    st.caption(f"Converted contract price: ${contract_price_usd_per_mwh:,.2f}/MWh.")
    if wesm_pricing_enabled:
        st.caption(
            "WESM pricing uses the hourly profile (upload or bundled default) for both "
            "shortfall costs and surplus revenue."
        )

    default_wesm_variant = econ_defaults.get("wesm_profile_variant")
    wesm_file = st.file_uploader(
        "WESM hourly price CSV (optional; timestamp/hour_index + deficit/surplus prices)",
        type=["csv"],
        key="multi_scenario_wesm_upload",
    )
    wesm_profile_variants = ["historical", "forecast"]
    wesm_profile_labels = {
        "historical": "Historical (default)",
        "forecast": "Forecast (8760-hr average)",
    }
    default_variant = (
        default_wesm_variant if default_wesm_variant in wesm_profile_variants else "historical"
    )
    selected_wesm_variant = st.selectbox(
        "Default WESM profile when no file is uploaded",
        options=wesm_profile_variants,
        index=wesm_profile_variants.index(default_variant),
        format_func=wesm_profile_labels.get,
        key="multi_scenario_wesm_profile_variant",
        disabled=not wesm_pricing_enabled,
        help=(
            "Defaults to data/wesm_price_profile_historical.csv or "
            "data/wesm_price_profile_forecast.csv. Forecast values are averaged across years. "
            "Uploaded files always take priority."
        ),
    )
    use_wesm_forecast = selected_wesm_variant == "forecast"
    if wesm_pricing_enabled:
        st.caption(
            "If no WESM file is uploaded, the default profile in ./data/ is used when available."
        )

    variable_schedule_default = (
        econ_inputs_default.variable_opex_schedule_usd if econ_inputs_default else None
    )
    periodic_variable_amount_default = (
        float(econ_inputs_default.periodic_variable_opex_usd)
        if econ_inputs_default and econ_inputs_default.periodic_variable_opex_usd is not None
        else 0.0
    )
    periodic_variable_cadence_default = (
        int(econ_inputs_default.periodic_variable_opex_interval_years)
        if econ_inputs_default and econ_inputs_default.periodic_variable_opex_interval_years
        else 5
    )
    variable_col1, variable_col2 = st.columns(2)
    with variable_col1:
        st.markdown("**Variable OPEX overrides**")
        st.caption(
            "Use the schedule controls to override the base OPEX mode above (CAPEX % or PHP/kWh). "
            "Amounts are in USD and are treated as nominal per-year values."
        )
        variable_opex_usd_per_mwh: Optional[float] = None
    with variable_col2:
        default_radio = "None"
        if variable_schedule_default:
            default_radio = "Custom"
        elif periodic_variable_amount_default > 0:
            default_radio = "Periodic"
        variable_schedule_choice = st.radio(
            "Variable expense schedule",
            options=["None", "Periodic", "Custom"],
            index=["None", "Periodic", "Custom"].index(default_radio),
            horizontal=True,
            help=(
                "Custom or periodic schedules override the base OPEX mode and per-kWh overrides. "
                "Per-kWh overrides supersede fixed percentages and adders."
            ),
        )
        variable_opex_schedule_usd: Optional[Tuple[float, ...]] = (
            tuple(variable_schedule_default) if variable_schedule_default else None
        )
        periodic_variable_opex_usd: Optional[float] = None
        periodic_variable_opex_interval_years: Optional[int] = None
        if variable_schedule_choice == "Periodic":
            periodic_variable_opex_usd = st.number_input(
                "Variable expense when periodic (USD)",
                min_value=0.0,
                value=periodic_variable_amount_default,
                step=10_000.0,
                help="Amount applied on the selected cadence (year 1, then every N years).",
            )
            periodic_variable_opex_interval_years = st.number_input(
                "Cadence (years)",
                min_value=1,
                value=periodic_variable_cadence_default,
                step=1,
            )
            if periodic_variable_opex_usd <= 0:
                periodic_variable_opex_usd = None
        elif variable_schedule_choice == "Custom":
            default_custom_text = "\n".join(str(val) for val in variable_opex_schedule_usd or [])
            custom_variable_text = st.text_area(
                "Custom variable expenses (USD/year)",
                value=default_custom_text,
                placeholder="e.g., 250000, 275000, 300000",
                help="Comma or newline separated values applied per project year.",
            )
            st.caption(
                "Use commas or newlines between entries; provide one value per project year "
                f"({cached_cfg.years} entries)."
            )
            if custom_variable_text.strip():
                try:
                    variable_opex_schedule_usd = tuple(
                        parse_numeric_series("Variable expense schedule", custom_variable_text)
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()
            else:
                variable_opex_schedule_usd = None

    financing_col1, financing_col2, financing_col3 = st.columns(3)
    with financing_col1:
        default_debt_equity_ratio = DEFAULT_DEBT_EQUITY_RATIO
        if econ_inputs_default and 0.0 < econ_inputs_default.debt_ratio < 1.0:
            default_debt_equity_ratio = econ_inputs_default.debt_ratio / (1.0 - econ_inputs_default.debt_ratio)
        debt_equity_ratio = st.number_input(
            "Debt/Equity ratio (D/E)",
            min_value=0.0,
            value=default_debt_equity_ratio,
            step=0.1,
            help=(
                "Debt divided by equity; 1.0 implies 50% debt and 50% equity. "
                f"Default: {DEFAULT_DEBT_EQUITY_RATIO:.1f} D/E."
            ),
        )
        debt_ratio = debt_equity_ratio / (1.0 + debt_equity_ratio) if debt_equity_ratio > 0 else 0.0
        st.caption(f"Implied debt share of capital: {debt_ratio * 100:.1f}%.")
    with financing_col2:
        cost_of_debt_pct = st.number_input(
            "Cost of debt (%)",
            min_value=0.0,
            max_value=30.0,
            value=float(econ_inputs_default.cost_of_debt * 100.0)
            if econ_inputs_default and econ_inputs_default.cost_of_debt is not None
            else DEFAULT_COST_OF_DEBT_PCT,
            step=0.1,
            help=f"Annual interest rate applied to the debt balance. Default: {DEFAULT_COST_OF_DEBT_PCT:.1f}%.",
        )
    with financing_col3:
        tenor_years = st.number_input(
            "Debt tenor (years)",
            min_value=1,
            value=int(econ_inputs_default.tenor_years)
            if econ_inputs_default and econ_inputs_default.tenor_years
            else DEFAULT_TENOR_YEARS,
            step=1,
            help=f"Years over which debt is amortized using level payments. Default: {DEFAULT_TENOR_YEARS}.",
        )

economic_inputs = EconomicInputs(
    capex_musd=capex_musd,
    capex_usd_per_kwh=capex_usd_per_kwh if capex_mode == "USD/kWh (BOL)" else None,
    capex_total_usd=None,
    bess_bol_kwh=bess_bol_kwh_default if capex_mode == "USD/kWh (BOL)" else None,
    pv_capex_musd=pv_capex_musd,
    fixed_opex_pct_of_capex=fixed_opex_pct,
    fixed_opex_musd=fixed_opex_musd,
    opex_php_per_kwh=opex_php_per_kwh,
    inflation_rate=inflation_pct / 100.0,
    discount_rate=discount_rate,
    variable_opex_usd_per_mwh=variable_opex_usd_per_mwh,
    variable_opex_schedule_usd=variable_opex_schedule_usd,
    periodic_variable_opex_usd=periodic_variable_opex_usd,
    periodic_variable_opex_interval_years=periodic_variable_opex_interval_years,
    forex_rate_php_per_usd=forex_rate_php_per_usd,
    devex_cost_php=devex_cost_php,
    include_devex_year0=include_devex_year0,
    debt_ratio=debt_ratio,
    cost_of_debt=cost_of_debt_pct / 100.0,
    tenor_years=int(tenor_years),
    wacc=wacc_pct / 100.0,
)
price_inputs = PriceInputs(
    contract_price_usd_per_mwh=contract_price_usd_per_mwh,
    escalate_with_inflation=escalate_prices,
    apply_wesm_to_shortfall=wesm_pricing_enabled,
    sell_to_wesm=sell_to_wesm if wesm_pricing_enabled else False,
)
cache_latest_economics_payload(
    {
        "economic_inputs": economic_inputs,
        "price_inputs": price_inputs,
        "wesm_profile_source": wesm_file,
        "wesm_profile_variant": selected_wesm_variant,
    }
)

st.markdown("### Scenario inputs")
st.caption(
    "Use the table below to tweak scenarios. Columns are grouped by **System sizing**, **Operations windows**, "
    "**Augmentation**, and **Economics** knobs."
)
st.caption("Tip: Save edits using the button below to avoid Streamlit reruns resetting values.")

template_cols = st.columns(3)
templates = {
    "Baseline": _baseline_template,
    "High availability": _high_availability_template,
    "Aggressive augmentation": _aggressive_augmentation_template,
}
for col, (label, builder) in zip(template_cols, templates.items()):
    with col:
        if st.button(f"Add {label}", use_container_width=True):
            table_data = st.session_state.get("multi_scenario_table_data")
            if table_data is None or table_data.empty:
                table_data = _seed_rows(cached_cfg)
            new_row = pd.DataFrame([builder(cached_cfg)])
            st.session_state["multi_scenario_table_data"] = pd.concat(
                [table_data, new_row], ignore_index=True
            )

if "multi_scenario_table_data" not in st.session_state:
    st.session_state["multi_scenario_table_data"] = _seed_rows(cached_cfg)

with st.expander("Import scenarios without the table", expanded=False):
    st.caption(
        "Paste CSV/JSON text or upload a file to replace the table below. Columns not provided fall back to defaults."
    )
    uploaded_file = st.file_uploader(
        "Upload CSV or JSON",
        type=["csv", "json"],
        accept_multiple_files=False,
        key="multi_scenario_uploader",
    )
    pasted_text = st.text_area(
        "Or paste CSV/JSON content",
        value="",
        placeholder="label,initial_power_mw,initial_usable_mwh,years\nScenario A,25,100,15",
        key="multi_scenario_paste_area",
        height=120,
    )
    if st.button("Load scenarios", use_container_width=True):
        imported_df: Optional[pd.DataFrame] = None
        if uploaded_file is not None:
            try:
                if uploaded_file.type.endswith("json"):
                    imported_df = pd.read_json(uploaded_file)
                else:
                    imported_df = pd.read_csv(uploaded_file)
            except Exception as exc:
                st.error(f"Upload could not be read: {exc}")
        if imported_df is None and pasted_text.strip():
            imported_df = _read_table_from_text(pasted_text, cached_cfg)
        if imported_df is not None:
            st.session_state["multi_scenario_table_data"] = _normalize_table(imported_df, cached_cfg)
            st.success(f"Loaded {len(imported_df)} scenario(s) from import.")
        else:
            st.info("No import applied. Upload a file or paste table content.", icon="ℹ️")

table_placeholder = st.empty()
WINDOW_PLACEHOLDER = "06:00-10:00, 18:00-22:00"
column_config = {
    "label": st.column_config.TextColumn("Label", help="Identifier for this scenario."),
    "initial_power_mw": st.column_config.NumberColumn(
        "Power (MW)",
        min_value=0.1,
        max_value=500.0,
        step=0.1,
        help="System sizing: initial discharge rating.",
    ),
    "initial_usable_mwh": st.column_config.NumberColumn(
        "Usable energy (MWh)",
        min_value=1.0,
        max_value=1_000.0,
        step=1.0,
        help="System sizing: initial usable energy.",
    ),
    "contracted_mw": st.column_config.NumberColumn(
        "Contracted MW", min_value=0.1, max_value=500.0, step=0.1, help="System sizing: firm delivery target."
    ),
    "years": st.column_config.NumberColumn(
        "Years", min_value=1, max_value=40, step=1, help="Project horizon in years."
    ),
    "pv_degradation_rate": st.column_config.NumberColumn(
        "PV degradation (frac/yr)", min_value=0.0, max_value=0.2, step=0.001, help="Economics/System inputs."
    ),
    "bess_availability": st.column_config.NumberColumn(
        "BESS availability", min_value=0.5, max_value=1.0, step=0.01, help="Operations: availability assumption."
    ),
    "rte": st.column_config.NumberColumn(
        "Round-trip η", min_value=0.5, max_value=0.99, step=0.01, help="Operations: round-trip efficiency (0–1)."
    ),
    "soc_floor": st.column_config.NumberColumn(
        "SOC floor", min_value=0.0, max_value=0.95, step=0.01, help="Operations: minimum SOC as a fraction."
    ),
    "soc_ceiling": st.column_config.NumberColumn(
        "SOC ceiling", min_value=0.05, max_value=1.0, step=0.01, help="Operations: maximum SOC as a fraction."
    ),
    "calendar_fade_rate": st.column_config.NumberColumn(
        "Calendar fade (frac/yr)",
        min_value=0.0,
        max_value=0.1,
        step=0.001,
        help="Operations: annual calendar fade rate applied to usable energy.",
    ),
    "use_calendar_exp_model": st.column_config.CheckboxColumn(
        "Use exponential fade", help="Operations: use exponential decay for calendar fade instead of linear."
    ),
    "discharge_windows": st.column_config.TextColumn(
        "Discharge windows",
        help=f"Operations windows: comma-separated HH:MM-HH:MM ranges (e.g., {WINDOW_PLACEHOLDER}).",
        default=WINDOW_PLACEHOLDER,
    ),
    "charge_windows": st.column_config.TextColumn(
        "Charge windows (optional)",
        help=f"Operations windows: leave blank to allow any PV hour; uses HH:MM-HH:MM (e.g., {WINDOW_PLACEHOLDER}).",
        default=WINDOW_PLACEHOLDER,
    ),
    "augmentation": st.column_config.SelectboxColumn(
        "Augmentation mode",
        options=["None", "Threshold", "Periodic"],
        help="Augmentation: strategy applied across years.",
    ),
    "aug_trigger_type": st.column_config.SelectboxColumn(
        "Trigger type",
        options=["Capability", "SOH"],
        help="Augmentation: used when mode is Threshold.",
    ),
    "aug_threshold_margin": st.column_config.NumberColumn(
        "Capability margin (frac)",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help="Augmentation: allowed margin below contracted energy before triggering.",
    ),
    "aug_topup_margin": st.column_config.NumberColumn(
        "Top-up margin (frac)",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help="Augmentation: energy added when augmenting under capability mode.",
    ),
    "aug_soh_trigger_pct": st.column_config.NumberColumn(
        "SOH trigger (%)",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help="Augmentation: SOH threshold when trigger type is SOH (fraction).",
    ),
    "aug_soh_add_frac_initial": st.column_config.NumberColumn(
        "SOH add frac of BOL",
        min_value=0.0,
        max_value=2.0,
        step=0.01,
        help="Augmentation: fraction of initial BOL energy added under SOH trigger.",
    ),
    "aug_periodic_every_years": st.column_config.NumberColumn(
        "Periodic interval (yrs)",
        min_value=1,
        max_value=40,
        step=1,
        help="Augmentation: augment every N years when mode is Periodic.",
    ),
    "aug_periodic_add_frac_of_bol": st.column_config.NumberColumn(
        "Periodic add (frac of BOL)",
        min_value=0.0,
        max_value=2.0,
        step=0.01,
        help="Augmentation: energy added each period as a fraction of initial BOL energy.",
    ),
    "aug_add_mode": st.column_config.SelectboxColumn(
        "Aug add mode",
        options=["Percent", "Fixed"],
        help="Augmentation: percent of BOL vs fixed MWh when augmenting.",
    ),
    "aug_fixed_energy_mwh": st.column_config.NumberColumn(
        "Fixed aug size (MWh)",
        min_value=0.0,
        max_value=2_000.0,
        step=1.0,
        help="Augmentation: used when Aug add mode is Fixed.",
    ),
    "aug_retire_old_cohort": st.column_config.CheckboxColumn(
        "Retire old cohort",
        help="Augmentation: retire the oldest cohort instead of layering capacity.",
    ),
    "aug_retire_soh_pct": st.column_config.NumberColumn(
        "Retire below SOH",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        help="Augmentation: retire cohorts whose SOH falls below this fraction.",
    ),
    "aug_retire_replacement_mode": st.column_config.SelectboxColumn(
        "Retire replacement mode",
        options=["None", "Percent", "Fixed"],
        help="Augmentation: replacement sizing for retired cohorts.",
    ),
    "aug_retire_replacement_pct_bol": st.column_config.NumberColumn(
        "Retire replacement % BOL",
        min_value=0.0,
        max_value=2.0,
        step=0.01,
        help="Augmentation: replacement energy as a fraction of initial BOL energy.",
    ),
    "aug_retire_replacement_fixed_mwh": st.column_config.NumberColumn(
        "Retire replacement MWh",
        min_value=0.0,
        max_value=2_000.0,
        step=1.0,
        help="Augmentation: fixed replacement energy in MWh.",
    ),
}
column_order = _expected_column_order()

st.caption(f"Time window format examples: {WINDOW_PLACEHOLDER} or 09:00-15:00.")
with st.form("multi_scenario_table_form", clear_on_submit=False):
    edited_df = table_placeholder.data_editor(
        st.session_state["multi_scenario_table_data"],
        use_container_width=True,
        num_rows="dynamic",
        hide_index=True,
        column_config=column_config,
        column_order=column_order,
        key="multi_scenario_table",
    )
    saved = st.form_submit_button("Save scenario table", use_container_width=True)
    if saved:
        st.session_state["multi_scenario_table_data"] = _normalize_table(edited_df, cached_cfg)
        st.success("Saved table edits. They will persist across reruns in this session.")

st.caption("Add rows for each design tweak. Remove rows to pare down the batch run.")
st.markdown("#### Scenario summary")
_render_summary_table(st.session_state["multi_scenario_table_data"])

table_df = st.session_state["multi_scenario_table_data"]

run_container = st.container()
results_container = st.container()

if "multi_scenario_batch_results" in st.session_state:
    st.success("Showing the latest batch results cached in this session.")


def _run_batch() -> pd.DataFrame | None:
    """Run the configured scenarios and return a results DataFrame."""

    if table_df is None or table_df.empty:
        st.warning("Add at least one scenario before running.", icon="⚠️")
        return None

    econ_payload = get_latest_economics_payload()
    validation_errors: List[str] = []
    for idx, row in table_df.reset_index(drop=True).iterrows():
        validation_errors.extend(_validate_row(row, idx))

    if validation_errors:
        st.error("Please fix the issues below before running.")
        for msg in validation_errors:
            st.caption(f"• {msg}")
        return None

    enforce_rate_limit()
    total_scenarios = len(table_df)
    progress = st.progress(0.0, text="Starting batch...")
    status_placeholder = st.empty()
    status_rows: List[Dict[str, str]] = []
    for _, row in table_df.iterrows():
        status_rows.append({"Label": str(row.get("label") or "Scenario"), "Status": "⏳ Pending"})
    status_placeholder.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)
    results: List[Dict[str, Any]] = []

    wesm_profile_df: Optional[pd.DataFrame] = None
    base_econ_inputs: Optional[EconomicInputs] = None
    base_price_inputs: Optional[PriceInputs] = None
    wesm_profile_source: Optional[Any] = None
    wesm_profile_variant = "historical"
    if isinstance(econ_payload, dict):
        base_econ_inputs = econ_payload.get("economic_inputs")
        base_price_inputs = econ_payload.get("price_inputs")
        wesm_profile_source = econ_payload.get("wesm_profile_source")
        wesm_profile_variant = econ_payload.get("wesm_profile_variant", "historical")
        if base_econ_inputs and base_price_inputs:
            wesm_enabled = base_price_inputs.apply_wesm_to_shortfall or base_price_inputs.sell_to_wesm
            if not wesm_enabled:
                wesm_profile_source = None
            if base_econ_inputs.capex_usd_per_kwh is not None and base_econ_inputs.bess_bol_kwh is None:
                base_econ_inputs = replace(
                    base_econ_inputs,
                    bess_bol_kwh=cached_cfg.initial_usable_mwh * 1000.0,
                )
            normalized_base_inputs = normalize_economic_inputs(base_econ_inputs)
            use_wesm_forecast = wesm_profile_variant == "forecast"
            default_wesm_profile = _default_wesm_profile_path(use_wesm_forecast)
            if (
                wesm_profile_source is None
                and wesm_enabled
                and default_wesm_profile.exists()
            ):
                wesm_profile_source = str(default_wesm_profile)
            if wesm_profile_source is not None and wesm_enabled:
                try:
                    if use_wesm_forecast and wesm_profile_source == str(default_wesm_profile):
                        wesm_profile_df = read_wesm_forecast_profile_average(
                            [wesm_profile_source],
                            forex_rate_php_per_usd=normalized_base_inputs.forex_rate_php_per_usd,
                        )
                    else:
                        wesm_profile_df = read_wesm_profile(
                            [wesm_profile_source],
                            forex_rate_php_per_usd=normalized_base_inputs.forex_rate_php_per_usd,
                        )
                except Exception as exc:  # noqa: BLE001
                    st.warning(f"WESM profile could not be read; falling back to static pricing. ({exc})")
                    wesm_profile_df = None

    for idx, (_, row_series) in enumerate(table_df.reset_index(drop=True).iterrows(), start=1):
        label = str(row_series.get("label") or f"Scenario {idx}")
        try:
            _, cfg = _parse_row_to_config(row_series, cached_cfg)
        except ValueError as exc:  # noqa: BLE001
            status_rows[idx - 1]["Status"] = f"❌ {exc}"
            status_placeholder.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)
            continue

        status_rows[idx - 1]["Status"] = "🚀 Running"
        status_placeholder.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)
        progress.progress((idx - 1) / total_scenarios, text=f"Running {label}...")
        try:
            with st.spinner(f"Running {label} simulation..."):
                need_logs = bool(wesm_profile_df is not None)
                sim_output = simulate_project(
                    cfg, pv_df=pv_df, cycle_df=cycle_df, dod_override=dod_override, need_logs=need_logs
                )
        except ValueError as exc:  # noqa: BLE001
            status_rows[idx - 1]["Status"] = f"❌ {exc}"
            status_placeholder.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)
            continue

        summary = summarize_simulation(sim_output)
        final_year = sim_output.results[-1]
        economics_fields: Dict[str, Any] = {}
        economics_columns = [
            "Discounted costs (USD million)",
            "LCOE (PHP/kWh delivered)",
            "LCOS (PHP/kWh from BESS)",
            "Discounted revenues (USD million)",
            "Project NPV (USD million, WACC)",
            "PIRR (%)",
            "EBITDA (USD million)",
            "EBITDA margin (%)",
            "EIRR (%)",
        ]
        if (
            isinstance(econ_payload, dict)
            and econ_payload.get("economic_inputs")
            and econ_payload.get("price_inputs")
        ):
            try:
                econ_inputs: EconomicInputs = econ_payload["economic_inputs"]
                price_inputs: PriceInputs = econ_payload["price_inputs"]
                if econ_inputs.capex_usd_per_kwh is not None:
                    econ_inputs = replace(
                        econ_inputs,
                        bess_bol_kwh=cfg.initial_usable_mwh * 1000.0,
                    )
                normalized_econ_inputs = normalize_economic_inputs(econ_inputs)
                augmentation_costs_usd = estimate_augmentation_costs_by_year(
                    sim_output.augmentation_energy_added_mwh,
                    cfg.initial_usable_mwh,
                    normalized_econ_inputs.capex_musd,
                )
                annual_delivered = [r.delivered_firm_mwh for r in sim_output.results]
                annual_bess = [r.bess_to_contract_mwh for r in sim_output.results]
                annual_pv_delivered = [
                    float(delivered) - float(bess)
                    for delivered, bess in zip(annual_delivered, annual_bess)
                ]
                annual_pv_excess = [r.pv_curtailed_mwh for r in sim_output.results]
                annual_shortfall = [r.shortfall_mwh for r in sim_output.results]
                annual_total_generation = [r.available_pv_mwh for r in sim_output.results]
                annual_wesm_shortfall_cost_usd: Optional[List[float]] = None
                annual_wesm_surplus_revenue_usd: Optional[List[float]] = None
                if wesm_profile_df is not None and sim_output.hourly_logs_by_year:
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
                        step_hours=cfg.step_hours,
                        apply_inflation=False,
                        inflation_rate=normalized_econ_inputs.inflation_rate,
                    )
                econ_outputs = compute_lcoe_lcos_with_augmentation_fallback(
                    annual_delivered_mwh=annual_delivered,
                    annual_bess_mwh=annual_bess,
                    inputs=normalized_econ_inputs,
                    augmentation_costs_usd=augmentation_costs_usd if any(augmentation_costs_usd) else None,
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
                    augmentation_costs_usd=augmentation_costs_usd if any(augmentation_costs_usd) else None,
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
                    augmentation_costs_usd=augmentation_costs_usd if any(augmentation_costs_usd) else None,
                    annual_total_generation_mwh=annual_total_generation,
                )
                php_per_kwh_factor = normalized_econ_inputs.forex_rate_php_per_usd / 1000.0
                economics_fields = {
                    "Discounted costs (USD million)": econ_outputs.discounted_costs_usd / 1_000_000.0,
                    "LCOE (PHP/kWh delivered)": econ_outputs.lcoe_usd_per_mwh * php_per_kwh_factor,
                    "LCOS (PHP/kWh from BESS)": econ_outputs.lcos_usd_per_mwh * php_per_kwh_factor,
                    "Discounted revenues (USD million)": cash_outputs.discounted_revenues_usd / 1_000_000.0,
                    "Project NPV (USD million, WACC)": financing_outputs.project_npv_usd / 1_000_000.0,
                    "PIRR (%)": financing_outputs.project_irr_pct,
                    "EBITDA (USD million)": financing_outputs.ebitda_usd / 1_000_000.0,
                    "EBITDA margin (%)": financing_outputs.ebitda_margin * 100.0,
                    "EIRR (%)": financing_outputs.equity_irr_pct,
                }
            except ValueError:
                economics_fields = {column: float("nan") for column in economics_columns}
            except Exception:  # noqa: BLE001
                economics_fields = {column: float("nan") for column in economics_columns}

        results.append(
            {
                "Label": label,
                "Years": cfg.years,
                "Power (MW)": cfg.initial_power_mw,
                "Usable MWh": cfg.initial_usable_mwh,
                "Contracted MW": cfg.contracted_mw,
                "Compliance (%)": summary.compliance,
                "BESS share of firm (%)": summary.bess_share_of_firm,
                "Charge/Discharge ratio": summary.charge_discharge_ratio,
                "PV capture ratio": summary.pv_capture_ratio,
                "Shortfall MWh": summary.total_shortfall_mwh,
                "Total generation (MWh)": summary.total_project_generation_mwh,
                "BESS discharge (MWh)": summary.bess_generation_mwh,
                "PV contribution (MWh)": summary.pv_generation_mwh,
                "PV excess (MWh)": summary.pv_excess_mwh,
                "BESS losses (MWh)": summary.bess_losses_mwh,
                "Avg eq cycles/yr": summary.avg_eq_cycles_per_year,
                "Final SOH_total": final_year.soh_total,
                "EOY usable MWh": final_year.eoy_usable_mwh,
                "EOY power MW": final_year.eoy_power_mw,
                "Augmentations": sim_output.augmentation_events,
                **economics_fields,
            },
        )
        status_rows[idx - 1]["Status"] = "✅ Complete"
        status_placeholder.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)
        progress.progress(idx / total_scenarios, text=f"Finished {label} ({idx}/{total_scenarios})")

    progress.progress(1.0, text="Batch complete.")
    st.balloons()
    st.toast("Batch run complete.")
    if not results:
        st.warning("No runs finished successfully. Please review the errors above.", icon="⚠️")
        return None
    return pd.DataFrame(results)


with run_container:
    if st.button("Run scenarios", use_container_width=True):
        with st.spinner("Running batch scenarios..."):
            batch_df = _run_batch()
        if batch_df is not None:
            st.session_state["multi_scenario_batch_results"] = batch_df


with results_container:
    batch_results: pd.DataFrame | None = st.session_state.get("multi_scenario_batch_results")
    if batch_results is not None and not batch_results.empty:
        formatted = batch_results.style.format(
            {
                "Compliance (%)": "{:,.2f}",
                "BESS share of firm (%)": "{:,.2f}",
                "Charge/Discharge ratio": "{:,.3f}",
                "PV capture ratio": "{:,.3f}",
                "Shortfall MWh": "{:,.1f}",
                "Total generation (MWh)": "{:,.1f}",
                "BESS discharge (MWh)": "{:,.1f}",
                "PV contribution (MWh)": "{:,.1f}",
                "PV excess (MWh)": "{:,.1f}",
                "BESS losses (MWh)": "{:,.1f}",
                "Avg eq cycles/yr": "{:,.2f}",
                "Final SOH_total": "{:,.3f}",
                "EOY usable MWh": "{:,.1f}",
                "EOY power MW": "{:,.2f}",
                "Discounted costs (USD million)": "{:,.2f}",
                "LCOE (PHP/kWh delivered)": "{:,.2f}",
                "LCOS (PHP/kWh from BESS)": "{:,.2f}",
                "Discounted revenues (USD million)": "{:,.2f}",
                "Project NPV (USD million, WACC)": "{:,.2f}",
                "PIRR (%)": "{:,.2f}",
                "EBITDA (USD million)": "{:,.2f}",
                "EBITDA margin (%)": "{:,.2f}",
                "EIRR (%)": "{:,.2f}",
            }
        )
        st.dataframe(formatted, use_container_width=True, hide_index=True)

        csv_bytes = batch_results.to_csv(index=False).encode("utf-8")
        json_bytes = batch_results.to_json(orient="records", indent=2).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="multi_scenario_batch_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Download results as JSON",
            data=json_bytes,
            file_name="multi_scenario_batch_results.json",
            mime="application/json",
            use_container_width=True,
        )
        inputs_df = st.session_state.get("multi_scenario_table_data", pd.DataFrame())
        if not inputs_df.empty:
            inputs_csv = inputs_df.to_csv(index=False).encode("utf-8")
            inputs_json = inputs_df.to_json(orient="records", indent=2).encode("utf-8")
            st.download_button(
                "Download input table (CSV)",
                data=inputs_csv,
                file_name="multi_scenario_inputs.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.download_button(
                "Download input table (JSON)",
                data=inputs_json,
                file_name="multi_scenario_inputs.json",
                mime="application/json",
                use_container_width=True,
            )
        st.caption("Economics columns use the assumptions entered in the Economics section above.")
    else:
        st.info("No batch results yet. Add rows above and click Run scenarios.", icon="ℹ️")

st.caption(
    "Scenarios inherit other settings from the latest cached configuration. Logs are off for each run "
    "to keep memory within typical 4GB limits."
)
