from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from app import (
    BASE_DIR,
    SimConfig,
    Window,
    parse_windows,
    simulate_project,
    summarize_simulation,
)
from utils import enforce_rate_limit
from utils.economics import (
    EconomicInputs,
    PriceInputs,
    compute_lcoe_lcos_with_augmentation_fallback,
)
from utils.ui_state import get_shared_data

st.set_page_config(page_title="Multi-scenario batch", layout="wide")

st.title("Multi-scenario batch runner")
st.caption(
    "Queue multiple simulation variants at once. Per-scenario logs are disabled to conserve memory; "
    "rerun the main page for detailed charts."
)


def _parse_numeric_series(raw_text: str, label: str) -> list[float]:
    """Parse a comma or newline-delimited series of floats for form inputs."""

    tokens = [t.strip() for t in raw_text.replace(",", "\n").splitlines() if t.strip()]
    series: list[float] = []
    for token in tokens:
        try:
            series.append(float(token))
        except ValueError as exc:  # noqa: BLE001
            st.error(f"{label} contains a non-numeric entry: '{token}'")
            raise
    return series


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


def _parse_row_to_config(row: pd.Series, template: SimConfig) -> Tuple[str, SimConfig]:
    """Apply row overrides to a SimConfig copy, validating fields along the way."""

    label = str(row.get("label") or "Scenario")
    config = deepcopy(template)

    dis_windows_text = str(row.get("discharge_windows") or "").strip()
    dis_windows = parse_windows(dis_windows_text)
    if not dis_windows:
        raise ValueError("Provide at least one discharge window (HH:MM-HH:MM).")

    charge_windows_text = str(row.get("charge_windows") or template.charge_windows_text or "").strip()

    years = int(row.get("years") or template.years)
    soc_floor = float(row.get("soc_floor") or template.soc_floor)
    soc_ceiling = float(row.get("soc_ceiling") or template.soc_ceiling)
    if soc_floor >= soc_ceiling:
        raise ValueError("SOC floor must be lower than the SOC ceiling.")

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


pv_df, cycle_df = get_shared_data(BASE_DIR)
cached_cfg: SimConfig = st.session_state.get("latest_sim_config", SimConfig())
dod_override = st.session_state.get("latest_dod_override", "Auto (infer)")
forex_rate_php_per_usd = 58.0
default_contract_php_per_kwh = round(120.0 / 1000.0 * forex_rate_php_per_usd, 2)
default_pv_php_per_kwh = round(55.0 / 1000.0 * forex_rate_php_per_usd, 2)

st.page_link("app.py", label="Back to Inputs & Results", help="Tune inputs before batching scenarios.")
st.page_link("pages/03_Scenario_Comparisons.py", label="Scenario comparisons table")
st.page_link("pages/04_BESS_Sizing_Sweep.py", label="BESS sizing sweep")
st.page_link("pages/00_Home.py", label="Home (Guide)")
st.markdown("---")

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
            if custom_variable_text.strip():
                try:
                    variable_opex_schedule_usd = tuple(
                        _parse_numeric_series(custom_variable_text, "Variable expense schedule")
                    )
                except ValueError:
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

table_placeholder = st.empty()
default_rows = _seed_rows(cached_cfg)
edited_df = table_placeholder.data_editor(
    default_rows,
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "label": st.column_config.TextColumn("Label", help="Identifier for this scenario."),
        "initial_power_mw": st.column_config.NumberColumn(
            "Power (MW)", min_value=0.1, max_value=500.0, step=0.1, help="Initial discharge rating."
        ),
        "initial_usable_mwh": st.column_config.NumberColumn(
            "Usable energy (MWh)", min_value=1.0, max_value=1_000.0, step=1.0
        ),
        "contracted_mw": st.column_config.NumberColumn(
            "Contracted MW", min_value=0.1, max_value=500.0, step=0.1, help="Firm delivery target."
        ),
        "years": st.column_config.NumberColumn(
            "Years", min_value=1, max_value=40, step=1, help="Project horizon in years."
        ),
        "pv_degradation_rate": st.column_config.NumberColumn(
            "PV degradation (frac/yr)", min_value=0.0, max_value=0.2, step=0.001
        ),
        "bess_availability": st.column_config.NumberColumn(
            "BESS availability", min_value=0.5, max_value=1.0, step=0.01
        ),
        "rte": st.column_config.NumberColumn(
            "Round-trip η", min_value=0.5, max_value=0.99, step=0.01, help="Round-trip efficiency (0–1)."
        ),
        "soc_floor": st.column_config.NumberColumn(
            "SOC floor", min_value=0.0, max_value=0.95, step=0.01, help="Minimum SOC as a fraction."
        ),
        "soc_ceiling": st.column_config.NumberColumn(
            "SOC ceiling", min_value=0.05, max_value=1.0, step=0.01, help="Maximum SOC as a fraction."
        ),
        "calendar_fade_rate": st.column_config.NumberColumn(
            "Calendar fade (frac/yr)",
            min_value=0.0,
            max_value=0.1,
            step=0.001,
            help="Annual calendar fade rate applied to usable energy.",
        ),
        "use_calendar_exp_model": st.column_config.CheckboxColumn(
            "Use exponential fade", help="Use exponential decay for calendar fade instead of linear."
        ),
        "discharge_windows": st.column_config.TextColumn(
            "Discharge windows",
            help="Comma-separated HH:MM-HH:MM ranges (e.g., 10:00-14:00, 18:00-22:00).",
        ),
        "charge_windows": st.column_config.TextColumn(
            "Charge windows (optional)",
            help="Leave blank to allow any PV hour; uses the same HH:MM-HH:MM format.",
        ),
        "augmentation": st.column_config.SelectboxColumn(
            "Augmentation mode",
            options=["None", "Threshold", "Periodic"],
            help="Choose the augmentation strategy applied across years.",
        ),
        "aug_trigger_type": st.column_config.SelectboxColumn(
            "Trigger type",
            options=["Capability", "SOH"],
            help="Used when augmentation mode is Threshold.",
        ),
        "aug_threshold_margin": st.column_config.NumberColumn(
            "Capability margin (frac)",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help="Allowed margin below contracted energy before triggering augmentation (capability mode).",
        ),
        "aug_topup_margin": st.column_config.NumberColumn(
            "Top-up margin (frac)",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help="Energy added when augmenting under capability mode.",
        ),
        "aug_soh_trigger_pct": st.column_config.NumberColumn(
            "SOH trigger (%)",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help="SOH threshold for augmentation when trigger type is SOH (fraction).",
        ),
        "aug_soh_add_frac_initial": st.column_config.NumberColumn(
            "SOH add frac of BOL",
            min_value=0.0,
            max_value=2.0,
            step=0.01,
            help="Fraction of initial BOL energy to add when augmenting under SOH trigger.",
        ),
        "aug_periodic_every_years": st.column_config.NumberColumn(
            "Periodic interval (yrs)",
            min_value=1,
            max_value=40,
            step=1,
            help="Augment every N years when mode is Periodic.",
        ),
        "aug_periodic_add_frac_of_bol": st.column_config.NumberColumn(
            "Periodic add (frac of BOL)",
            min_value=0.0,
            max_value=2.0,
            step=0.01,
            help="Energy added each period as a fraction of initial BOL energy.",
        ),
        "aug_add_mode": st.column_config.SelectboxColumn(
            "Aug add mode",
            options=["Percent", "Fixed"],
            help="Percent of BOL vs fixed MWh when augmenting.",
        ),
        "aug_fixed_energy_mwh": st.column_config.NumberColumn(
            "Fixed aug size (MWh)",
            min_value=0.0,
            max_value=2_000.0,
            step=1.0,
            help="Used when Aug add mode is Fixed.",
        ),
        "aug_retire_old_cohort": st.column_config.CheckboxColumn(
            "Retire old cohort",
            help="When augmenting, retire the oldest cohort instead of layering capacity.",
        ),
        "aug_retire_soh_pct": st.column_config.NumberColumn(
            "Retire below SOH",
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            help="Retire cohorts whose SOH falls below this fraction.",
        ),
    },
    key="multi_scenario_table",
)

st.caption("Tip: Add rows for each design tweak. Remove rows to pare down the batch run.")

run_container = st.container()
results_container = st.container()

if "multi_scenario_batch_results" in st.session_state:
    st.success("Showing the latest batch results cached in this session.")


def _run_batch() -> pd.DataFrame | None:
    """Run the configured scenarios and return a results DataFrame."""

    if edited_df is None or edited_df.empty:
        st.warning("Add at least one scenario before running.", icon="⚠️")
        return None

    econ_payload = st.session_state.get("latest_economics_payload")
    scenarios: List[Tuple[str, SimConfig]] = []
    errors: List[str] = []
    for idx, row in edited_df.reset_index(drop=True).iterrows():
        try:
            scenarios.append(_parse_row_to_config(row, cached_cfg))
        except ValueError as exc:  # noqa: BLE001
            errors.append(f"Row {idx + 1}: {exc}")

    if errors:
        st.error("Please fix the highlighted rows before running.")
        for msg in errors:
            st.caption(f"• {msg}")
        return None

    enforce_rate_limit()
    progress = st.progress(0.0, text="Starting batch...")
    results: List[Dict[str, Any]] = []

    for idx, (label, cfg) in enumerate(scenarios, start=1):
        progress.progress((idx - 1) / len(scenarios), text=f"Running {label}...")
        try:
            sim_output = simulate_project(
                cfg, pv_df=pv_df, cycle_df=cycle_df, dod_override=dod_override, need_logs=False
            )
        except ValueError as exc:  # noqa: BLE001
            progress.empty()
            st.error(f"{label}: {exc}")
            return None

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
        progress.progress(idx / len(scenarios), text=f"Finished {label} ({idx}/{len(scenarios)})")

    progress.progress(1.0, text="Batch complete.")
    st.balloons()
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
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="multi_scenario_batch_results.csv",
            mime="text/csv",
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
