from __future__ import annotations

from copy import deepcopy
from io import StringIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from app import BASE_DIR
from services.simulation_core import SimConfig, Window, parse_windows, simulate_project, summarize_simulation
from utils import enforce_rate_limit, parse_numeric_series
from utils.economics import (
    EconomicInputs,
    PriceInputs,
    compute_lcoe_lcos_with_augmentation_fallback,
)
from utils.ui_layout import init_page_layout

render_layout = init_page_layout(
    page_title="Multi-scenario batch",
    main_title="Multi-scenario batch runner",
    description=(
        "Queue multiple simulation variants at once. Per-scenario logs are disabled to conserve memory; "
        "rerun the main page for detailed charts."
    ),
    base_dir=BASE_DIR,
)


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


cached_cfg: SimConfig = st.session_state.get("latest_sim_config", SimConfig())
dod_override = st.session_state.get("latest_dod_override", "Auto (infer)")
forex_rate_php_per_usd = 58.0
default_contract_php_per_kwh = round(120.0 / 1000.0 * forex_rate_php_per_usd, 2)
default_pv_php_per_kwh = round(55.0 / 1000.0 * forex_rate_php_per_usd, 2)

# Pull PV/cycle inputs from the shared session cache instead of any external API.
pv_df, cycle_df = render_layout()

econ_defaults = st.session_state.get("latest_economics_payload", {})
econ_inputs_default: Optional[EconomicInputs] = econ_defaults.get("economic_inputs")
econ_price_default: Optional[PriceInputs] = econ_defaults.get("price_inputs")

with st.expander("Economics (optional)", expanded=False):
    econ_col1, econ_col2 = st.columns(2)
    with econ_col1:
        capex_musd = st.number_input(
            "Total CAPEX (USD million)",
            min_value=0.0,
            value=float(econ_inputs_default.capex_musd) if econ_inputs_default else 40.0,
            step=0.1,
        )
        fixed_opex_pct = st.number_input(
            "Fixed OPEX (% of CAPEX per year)",
            min_value=0.0,
            max_value=20.0,
            value=float(econ_inputs_default.fixed_opex_pct_of_capex) if econ_inputs_default else 2.0,
            step=0.1,
        )
        fixed_opex_musd = st.number_input(
            "Additional fixed OPEX (USD million/yr)",
            min_value=0.0,
            value=float(econ_inputs_default.fixed_opex_musd) if econ_inputs_default else 0.0,
            step=0.1,
        )
    with econ_col2:
        inflation_pct = st.number_input(
            "Inflation rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=float(econ_inputs_default.inflation_rate * 100) if econ_inputs_default else 3.0,
            step=0.1,
        )
        discount_rate_pct = st.number_input(
            "Discount rate (%)",
            min_value=0.0,
            max_value=30.0,
            value=float(econ_inputs_default.discount_rate * 100) if econ_inputs_default else 5.0,
            step=0.1,
        )
        st.caption(
            "Discount rate is applied directly. Use WACC-derived values if you prefer real/nominal alignment."
        )

    blended_price_default_php_per_kwh = (
        float(econ_price_default.blended_price_usd_per_mwh * forex_rate_php_per_usd / 1000.0)
        if econ_price_default and econ_price_default.blended_price_usd_per_mwh is not None
        else default_contract_php_per_kwh
    )
    blended_price_default_active = bool(
        econ_price_default and econ_price_default.blended_price_usd_per_mwh is not None
    )
    price_col1, price_col2 = st.columns(2)
    with price_col1:
        use_blended_price = st.checkbox(
            "Use blended energy price",
            value=blended_price_default_active,
            help=(
                "Apply a single price to all delivered firm energy and excess PV. "
                "Contract/PV-specific inputs are ignored while enabled."
            ),
        )
        contract_price_php_per_kwh = st.number_input(
            "Contract price (PHP/kWh from BESS)",
            min_value=0.0,
            value=float(econ_price_default.contract_price_usd_per_mwh * forex_rate_php_per_usd / 1000.0)
            if econ_price_default
            else default_contract_php_per_kwh,
            step=0.05,
            disabled=use_blended_price,
        )
    with price_col2:
        pv_market_price_php_per_kwh = st.number_input(
            "PV market price (PHP/kWh for excess PV)",
            min_value=0.0,
            value=float(econ_price_default.pv_market_price_usd_per_mwh * forex_rate_php_per_usd / 1000.0)
            if econ_price_default
            else default_pv_php_per_kwh,
            step=0.05,
            disabled=use_blended_price,
        )
        blended_price_php_per_kwh = st.number_input(
            "Blended energy price (PHP/kWh)",
            min_value=0.0,
            value=blended_price_default_php_per_kwh,
            step=0.05,
            help="Applied to all delivered firm energy and marketed PV when blended pricing is enabled.",
            disabled=not use_blended_price,
        )
    escalate_prices = st.checkbox(
        "Escalate prices with inflation",
        value=bool(econ_price_default.escalate_with_inflation) if econ_price_default else False,
    )
    blended_price_usd_per_mwh: Optional[float] = None
    contract_price_usd_per_mwh = contract_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
    pv_market_price_usd_per_mwh = pv_market_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
    if use_blended_price:
        blended_price_usd_per_mwh = blended_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
        st.caption(
            "Blended price active for revenues: "
            f"PHP {blended_price_php_per_kwh:,.2f}/kWh "
            f"(≈${blended_price_usd_per_mwh:,.2f}/MWh). Contract/PV prices are ignored."
        )
    else:
        st.caption(
            f"Converted contract price: ${contract_price_usd_per_mwh:,.2f}/MWh | "
            f"PV market price: ${pv_market_price_usd_per_mwh:,.2f}/MWh"
        )

    variable_opex_default_php = (
        float(econ_inputs_default.variable_opex_usd_per_mwh * forex_rate_php_per_usd / 1000.0)
        if econ_inputs_default and econ_inputs_default.variable_opex_usd_per_mwh is not None
        else 0.0
    )
    variable_schedule_default = econ_inputs_default.variable_opex_schedule_usd if econ_inputs_default else None
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
        variable_opex_php_per_kwh = st.number_input(
            "Variable OPEX (PHP/kWh)",
            min_value=0.0,
            value=variable_opex_default_php,
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
                "Custom or periodic schedules override per-kWh and fixed OPEX assumptions. "
                "Per-kWh costs override fixed percentages and adders."
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
            else:
                variable_opex_schedule_usd = None

economic_inputs = EconomicInputs(
    capex_musd=capex_musd,
    fixed_opex_pct_of_capex=fixed_opex_pct,
    fixed_opex_musd=fixed_opex_musd,
    inflation_rate=inflation_pct / 100.0,
    discount_rate=discount_rate_pct / 100.0,
    variable_opex_usd_per_mwh=variable_opex_usd_per_mwh,
    variable_opex_schedule_usd=variable_opex_schedule_usd,
    periodic_variable_opex_usd=periodic_variable_opex_usd,
    periodic_variable_opex_interval_years=periodic_variable_opex_interval_years,
)
price_inputs = PriceInputs(
    contract_price_usd_per_mwh=contract_price_usd_per_mwh,
    pv_market_price_usd_per_mwh=pv_market_price_usd_per_mwh,
    escalate_with_inflation=escalate_prices,
    blended_price_usd_per_mwh=blended_price_usd_per_mwh,
)
st.session_state["latest_economics_payload"] = {
    "economic_inputs": economic_inputs,
    "price_inputs": price_inputs,
}

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

    econ_payload = st.session_state.get("latest_economics_payload")
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
                sim_output = simulate_project(
                    cfg, pv_df=pv_df, cycle_df=cycle_df, dod_override=dod_override, need_logs=False
                )
        except ValueError as exc:  # noqa: BLE001
            status_rows[idx - 1]["Status"] = f"❌ {exc}"
            status_placeholder.dataframe(pd.DataFrame(status_rows), hide_index=True, use_container_width=True)
            continue

        summary = summarize_simulation(sim_output)
        final_year = sim_output.results[-1]
        economics_fields: Dict[str, Any] = {}
        if isinstance(econ_payload, dict) and econ_payload.get("economic_inputs"):
            try:
                econ_outputs = compute_lcoe_lcos_with_augmentation_fallback(
                    annual_delivered_mwh=[r.delivered_firm_mwh for r in sim_output.results],
                    annual_bess_mwh=[r.bess_to_contract_mwh for r in sim_output.results],
                    inputs=econ_payload["economic_inputs"],
                    augmentation_costs_usd=econ_payload.get("augmentation_costs_usd"),
                )
                economics_fields["LCOE ($/MWh)"] = econ_outputs.lcoe_usd_per_mwh
                economics_fields["LCOS ($/MWh)"] = econ_outputs.lcos_usd_per_mwh
            except ValueError:
                economics_fields["LCOE ($/MWh)"] = float("nan")
                economics_fields["LCOS ($/MWh)"] = float("nan")

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
                "LCOE ($/MWh)": "{:,.2f}",
                "LCOS ($/MWh)": "{:,.2f}",
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
        if st.session_state.get("latest_economics_payload"):
            st.caption("Economic columns populate when economics inputs are cached from the main page.")
        else:
            st.caption("LCOE/LCOS will populate once economics assumptions have been cached on the main page.")
    else:
        st.info("No batch results yet. Add rows above and click Run scenarios.", icon="ℹ️")

st.caption(
    "Scenarios inherit other settings from the latest cached configuration. Logs are off for each run "
    "to keep memory within typical 4GB limits."
)
