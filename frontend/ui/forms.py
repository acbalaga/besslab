"""Streamlit form rendering for simulation inputs.

Centralizing the form setup keeps `app.run_app` focused on orchestration and
lets other pages reuse the same config construction logic.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from services.simulation_core import (
    AUGMENTATION_SCHEDULE_BASIS,
    AugmentationScheduleEntry,
    SimConfig,
    build_schedule_from_editor,
    infer_step_hours_from_pv,
    parse_windows,
    validate_pv_profile_duration,
)
from utils import get_rate_limit_password, parse_numeric_series
from utils.economics import DEVEX_COST_PHP, EconomicInputs, PriceInputs
from utils.ui_state import (
    get_manual_aug_schedule_rows,
    get_rate_limit_state,
    save_manual_aug_schedule_rows,
    set_rate_limit_bypass,
)


@dataclass
class SimulationFormResult:
    config: SimConfig
    econ_inputs: Optional[EconomicInputs]
    price_inputs: Optional[PriceInputs]
    run_economics: bool
    dod_override: Optional[str]
    run_submitted: bool
    discharge_windows_text: str
    charge_windows_text: str


def render_rate_limit_section() -> None:
    """Allow users to enter a password to bypass the per-session rate limit."""
    expected_password = get_rate_limit_password()
    rate_limit_state = get_rate_limit_state()
    rate_limit_password = st.text_input(
        "Remove rate limit (password)",
        type="password",
        help=("Enter the configured password to disable the session rate limit. If no secret is set, use 'besslab'."),
        key="inputs_rate_limit_password",
    )
    if rate_limit_password:
        if rate_limit_password == expected_password:
            set_rate_limit_bypass(True)
            st.success("Rate limit disabled for this session.")
        else:
            set_rate_limit_bypass(False)
            st.error("Incorrect password. Rate limit still active.")
    elif rate_limit_state.bypass:
        st.caption("Rate limit disabled for this session.")


def render_simulation_form(pv_df: pd.DataFrame, cycle_df: pd.DataFrame) -> SimulationFormResult:
    """Render the main simulation inputs and return the assembled configuration."""
    st.subheader("Inputs")

    # Project & PV
    with st.expander("Project & PV", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            years = st.selectbox(
                "Project life (years)",
                list(range(10, 36, 5)),
                index=2,
                help="Extend to test augmentation schedules and end effects.",
            )
        with c2:
            pv_deg = (
                st.number_input(
                    "PV degradation %/yr",
                    0.0,
                    5.0,
                    0.6,
                    0.1,
                    help="Applied multiplicatively per year (e.g., 0.6% → (1−0.006)^year).",
                )
                / 100.0
            )
        with c3:
            pv_avail = st.slider("PV availability", 0.90, 1.00, 0.98, 0.01, help="Uptime factor applied to PV output.")

    # Availability (kept outside the main form so toggles rerender immediately)
    with st.expander("Availability", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            bess_avail = st.slider(
                "BESS availability",
                0.90,
                1.00,
                0.99,
                0.01,
                help="Uptime factor applied to BESS power capability.",
            )
        with c2:
            use_split_rte = st.checkbox(
                "Use separate charge/discharge efficiencies",
                value=False,
                help="Select to enter distinct charge and discharge efficiencies instead of a single round-trip value.",
            )
            if use_split_rte:
                charge_eff = st.slider(
                    "Charge efficiency (AC-AC)",
                    0.70,
                    0.99,
                    0.94,
                    0.01,
                    help="Applied when absorbing energy; multiplied with discharge efficiency to form the round-trip value.",
                )
                discharge_eff = st.slider(
                    "Discharge efficiency (AC-AC)",
                    0.70,
                    0.99,
                    0.94,
                    0.01,
                    help="Applied when delivering energy; multiplied with charge efficiency to form the round-trip value.",
                )
                rte = charge_eff * discharge_eff
                st.caption(f"Implied round-trip efficiency: {rte:.3f} (charge × discharge).")
            else:
                rte = st.slider(
                    "Round-trip efficiency (single, at POI)",
                    0.70,
                    0.99,
                    0.88,
                    0.01,
                    help="Single RTE; internally split √RTE for charge/discharge.",
                )
                charge_eff = None
                discharge_eff = None

    with st.form("inputs_form"):
        # BESS Specs
        with st.expander("BESS Specs (high-level)", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                init_power = st.number_input(
                    "Power rating (MW)", 1.0, None, 30.0, 1.0, help="Initial nameplate power (POI context), before availability."
                )
            with c2:
                init_energy = st.number_input(
                    "Usable energy at BOL (MWh)", 1.0, None, 120.0, 1.0, help="Initial usable energy (POI context)."
                )
            with c3:
                soc_floor = st.slider(
                    "SOC floor (%)", 0, 50, 10, 1, help="Reserve to protect cycling; lowers daily swing."
                ) / 100.0
                soc_ceiling = st.slider(
                    "SOC ceiling (%)", 50, 100, 98, 1, help="Upper limit to protect cycling; raises daily swing when higher."
                ) / 100.0

        # Dispatch
        with st.expander("Dispatch Strategy", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                contracted_mw = st.number_input(
                    "Contracted MW (firm)", 0.0, None, 30.0, 1.0, help="Firm capacity to meet during discharge windows."
                )
            with c2:
                discharge_windows_text = st.text_input(
                    "Discharge windows (HH:MM-HH:MM, comma-separated)",
                    "10:00-14:00, 18:00-22:00",
                    help="Ex: 10:00-14:00, 18:00-22:00",
                )
            with c3:
                charge_windows_text = st.text_input(
                    "Charge windows (blank = any PV hours)", "", help="PV-only charging; blank allows any PV hour."
                )

        # Degradation
        with st.expander("Degradation modeling", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                cal_fade = (
                    st.number_input(
                        "Calendar fade %/yr (empirical)",
                        0.0,
                        5.0,
                        1.0,
                        0.1,
                        help="Multiplicative retention: (1 − rate)^year.",
                    )
                    / 100.0
                )
            with c2:
                dod_override = st.selectbox(
                    "Degradation DoD basis",
                    ["Auto (infer)", "10%", "20%", "40%", "80%", "100%"],
                    help="Use the cycle table at a fixed DoD, or let the app infer based on median daily discharge.",
                )

        manual_schedule_entries: List[AugmentationScheduleEntry] = []
        manual_schedule_errors: List[str] = []
        manual_schedule_rows = get_manual_aug_schedule_rows(int(years))

        # Augmentation (conditional, with explainers)
        with st.expander("Augmentation strategy", expanded=False):
            aug_mode = st.selectbox("Strategy", ["None", "Threshold", "Periodic", "Manual"], index=0)

            aug_size_mode = "Percent"
            aug_fixed_energy = 0.0
            retire_enabled = False
            retire_soh = 0.60

            if aug_mode == "Manual":
                st.caption("Define explicit augmentation events by year. Save the table to persist edits across reruns.")
                with st.form("manual_aug_schedule_form", clear_on_submit=False):
                    manual_schedule_df = st.data_editor(
                        pd.DataFrame(manual_schedule_rows),
                        key="manual_aug_schedule_editor",
                        column_config={
                            "Year": st.column_config.NumberColumn("Year", min_value=1, step=1),
                            "Basis": st.column_config.SelectboxColumn("Basis", options=AUGMENTATION_SCHEDULE_BASIS),
                            "Amount": st.column_config.NumberColumn(
                                "Amount", min_value=0.0, format="%.3f", help="Percent or MW/MWh depending on basis."
                            ),
                        },
                        num_rows="dynamic",
                        hide_index=True,
                        use_container_width=True,
                    )
                    saved_manual_schedule = st.form_submit_button("Save augmentation table", use_container_width=True)
                if saved_manual_schedule:
                    save_manual_aug_schedule_rows(manual_schedule_df.to_dict("records"), int(years))
                    manual_schedule_rows = get_manual_aug_schedule_rows(int(years))
                    st.success("Saved augmentation events for this session.")
                manual_schedule_df = pd.DataFrame(manual_schedule_rows)
                manual_schedule_entries, manual_schedule_errors = build_schedule_from_editor(manual_schedule_df, int(years))
                if manual_schedule_errors:
                    for err in manual_schedule_errors:
                        st.error(err)
                elif not manual_schedule_entries:
                    st.warning("Add at least one row to run a manual augmentation schedule.")
                aug_thr_margin = 0.0
                aug_topup = 0.0
                aug_every = 5
                aug_frac = 0.10
                aug_trigger_type = "Capability"
                aug_soh_trig = 0.80
                aug_soh_add = 0.10
            elif aug_mode == "Threshold":
                trigger = st.selectbox(
                    "Trigger type",
                    ["Capability", "SOH"],
                    index=0,
                    help="Capability: Compare EOY capability vs target MWh/day.  SOH: Compare fleet SOH vs threshold.",
                )
                if trigger == "Capability":
                    c1, c2 = st.columns(2)
                    with c1:
                        aug_thr_margin = (
                            st.number_input(
                                "Allowance margin (%)",
                                0.0,
                                None,
                                0.0,
                                0.5,
                                help="Trigger when capability < target × (1 − margin).",
                            )
                            / 100.0
                        )
                    with c2:
                        aug_topup = (
                            st.number_input(
                                "Top-up margin (%)",
                                0.0,
                                None,
                                5.0,
                                0.5,
                                help="Augment up to target × (1 + margin) when triggered.",
                            )
                            / 100.0
                        )
                    aug_every = 5
                    aug_frac = 0.10
                    aug_trigger_type = "Capability"
                    aug_soh_trig = 0.80
                    aug_soh_add = 0.10
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        aug_soh_trig = (
                            st.number_input("SOH trigger (%)", 50.0, 100.0, 80.0, 1.0, help="Augment when SOH falls below this threshold.")
                            / 100.0
                        )
                    with c2:
                        aug_soh_add = (
                            st.number_input(
                                "Add % of initial BOL energy",
                                0.0,
                                None,
                                10.0,
                                1.0,
                                help="Added energy as % of initial BOL. Power added to keep original C-hours.",
                            )
                            / 100.0
                        )
                    aug_thr_margin = 0.0
                    aug_topup = 0.0
                    aug_every = 5
                    aug_frac = 0.10
                    aug_trigger_type = "SOH"
            elif aug_mode == "Periodic":
                c1, c2 = st.columns(2)
                with c1:
                    aug_every = st.number_input(
                        "Every N years", 1, None, 5, 1, help="Add capacity on this cadence (e.g., every 5 years)."
                    )
                with c2:
                    aug_frac = (
                        st.number_input(
                            "Add % of current BOL-ref energy",
                            0.0,
                            None,
                            10.0,
                            1.0,
                            help="Top-up energy relative to current BOL reference.",
                        )
                        / 100.0
                    )
                aug_thr_margin = 0.0
                aug_topup = 0.0
                aug_trigger_type = "Capability"
                aug_soh_trig = 0.80
                aug_soh_add = 0.10
            else:
                aug_thr_margin = 0.0
                aug_topup = 0.0
                aug_every = 5
                aug_frac = 0.10
                aug_trigger_type = "Capability"
                aug_soh_trig = 0.80
                aug_soh_add = 0.10

            if aug_mode not in ["None", "Manual"]:
                aug_size_mode = st.selectbox(
                    "Augmentation sizing",
                    ["Percent", "Fixed"],
                    format_func=lambda k: "% basis" if k == "Percent" else "Fixed energy (MWh)",
                    help="Choose whether to size augmentation as a percent or a fixed MWh add.",
                )
                if aug_size_mode == "Fixed":
                    aug_fixed_energy = st.number_input(
                        "Fixed energy added per event (MWh, BOL basis)",
                        min_value=0.0,
                        value=0.0,
                        step=1.0,
                        help="Adds this BOL-equivalent energy whenever augmentation is triggered.",
                    )

            if aug_mode != "None":
                retire_enabled = st.checkbox(
                    "Retire low-SOH cohorts when augmenting",
                    value=False,
                    help="Remove cohorts once their SOH falls below the retirement threshold.",
                )
                if retire_enabled:
                    retire_soh = (
                        st.number_input(
                            "Retirement SOH threshold (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=60.0,
                            step=1.0,
                            help="Cohorts at or below this SOH are retired before applying augmentation.",
                        )
                        / 100.0
                    )

        if aug_mode == "Manual" and (manual_schedule_errors or not manual_schedule_entries):
            st.error("Manual augmentation requires at least one valid year and no duplicate years.")
            st.stop()

        discharge_windows, discharge_window_warnings = parse_windows(discharge_windows_text)
        charge_windows, charge_window_warnings = parse_windows(charge_windows_text)
        for msg in discharge_window_warnings + charge_window_warnings:
            st.warning(msg)

        if not discharge_windows:
            st.error("Please provide at least one discharge window.")
            st.stop()

        cfg = SimConfig(
            years=int(years),
            pv_deg_rate=float(pv_deg),
            pv_availability=float(pv_avail),
            bess_availability=float(bess_avail),
            rte_roundtrip=float(rte),
            use_split_rte=bool(use_split_rte),
            charge_efficiency=float(charge_eff) if use_split_rte else None,
            discharge_efficiency=float(discharge_eff) if use_split_rte else None,
            soc_floor=float(soc_floor),
            soc_ceiling=float(soc_ceiling),
            initial_power_mw=float(init_power),
            initial_usable_mwh=float(init_energy),
            contracted_mw=float(contracted_mw),
            discharge_windows=discharge_windows,
            charge_windows_text=charge_windows_text,
            charge_windows=charge_windows,
            max_cycles_per_day_cap=1.2,
            calendar_fade_rate=float(cal_fade),
            use_calendar_exp_model=True,
            augmentation=aug_mode,
            aug_trigger_type=aug_trigger_type,
            aug_threshold_margin=float(aug_thr_margin),
            aug_topup_margin=float(aug_topup),
            aug_soh_trigger_pct=float(aug_soh_trig),
            aug_soh_add_frac_initial=float(aug_soh_add),
            aug_periodic_every_years=int(aug_every),
            aug_periodic_add_frac_of_bol=float(aug_frac),
            aug_add_mode=aug_size_mode,
            aug_fixed_energy_mwh=float(aug_fixed_energy),
            aug_retire_old_cohort=bool(retire_enabled),
            aug_retire_soh_pct=float(retire_soh),
            augmentation_schedule=list(manual_schedule_entries) if aug_mode == "Manual" else [],
        )

        inferred_step = infer_step_hours_from_pv(pv_df)
        if inferred_step is not None:
            cfg.step_hours = inferred_step

        duration_error = validate_pv_profile_duration(pv_df, cfg.step_hours)
        if duration_error:
            st.error(duration_error)
            st.stop()

        st.markdown("### Optional economics (NPV, IRR, LCOE, LCOS)")
        run_economics = st.checkbox(
            "Compute economics using simulation outputs",
            value=False,
            help=("Enable to enter financial assumptions and derive LCOE/LCOS, NPV, and IRR from the simulated annual energy streams."),
        )
        econ_inputs: Optional[EconomicInputs] = None
        price_inputs: Optional[PriceInputs] = None
        forex_rate_php_per_usd = 58.0
        devex_cost_usd = DEVEX_COST_PHP / forex_rate_php_per_usd

        default_contract_php_per_kwh = round(120.0 / 1000.0 * forex_rate_php_per_usd, 2)
        default_pv_php_per_kwh = round(55.0 / 1000.0 * forex_rate_php_per_usd, 2)
        wesm_surplus_reference_php_per_kwh = 3.29
        default_wesm_surplus_php_per_kwh = round(wesm_surplus_reference_php_per_kwh, 2)
        wesm_reference_php_per_mwh = 5_583.0
        default_wesm_php_per_kwh = round(wesm_reference_php_per_mwh / 1000.0, 2)

        if run_economics:
            wesm_pricing_enabled = False
            sell_to_wesm = False

            econ_col1, econ_col2, econ_col3 = st.columns(3)
            with econ_col1:
                wacc_pct = st.number_input("WACC (%)", min_value=0.0, max_value=30.0, value=8.0, step=0.1)
                inflation_pct = st.number_input(
                    "Inflation rate (%)",
                    min_value=0.0,
                    max_value=20.0,
                    value=3.0,
                    step=0.1,
                    help="Used to derive the real discount rate applied to costs and revenues.",
                )
                discount_rate = max((1 + wacc_pct / 100.0) / (1 + inflation_pct / 100.0) - 1, 0.0)
                st.caption(f"Real discount rate derived from WACC and inflation: {discount_rate * 100:.2f}%.")
            with econ_col2:
                capex_musd = st.number_input("Total CAPEX (USD million)", min_value=0.0, value=40.0, step=0.1)
                fixed_opex_pct = (
                    st.number_input(
                        "Fixed OPEX (% of CAPEX per year)",
                        min_value=0.0,
                        max_value=20.0,
                        value=2.0,
                        step=0.1,
                    )
                    / 100.0
                )
                fixed_opex_musd = st.number_input(
                    "Additional fixed OPEX (USD million/yr)", min_value=0.0, value=0.0, step=0.1
                )
                include_devex_year0 = st.checkbox(
                    "Include ₱100M DevEx at year 0",
                    value=False,
                    help=(
                        "Adds a fixed ₱100 million development expenditure upfront (≈"
                        f"${devex_cost_usd / 1_000_000:,.2f}M using PHP {forex_rate_php_per_usd:,.0f}/USD). "
                        "Flows through discounted costs, LCOE/LCOS, NPV, and IRR."
                    ),
                )
            with econ_col3:
                use_blended_price = st.checkbox(
                    "Use blended energy price",
                    value=False,
                    help=(
                        "Apply a single price to all delivered firm energy and excess PV. "
                        "Contract/PV-specific inputs are ignored while enabled."
                    ),
                )
                contract_price_php_per_kwh = st.number_input(
                    "Contract price (PHP/kWh from BESS)",
                    min_value=0.0,
                    value=default_contract_php_per_kwh,
                    step=0.05,
                    help="Price converted to USD/MWh internally using PHP 58/USD.",
                    disabled=use_blended_price,
                )
                pv_market_price_php_per_kwh = st.number_input(
                    "PV market price (PHP/kWh for excess PV)",
                    min_value=0.0,
                    value=default_pv_php_per_kwh,
                    step=0.05,
                    help="Price converted to USD/MWh internally using PHP 58/USD.",
                    disabled=use_blended_price,
                )
                blended_price_php_per_kwh = st.number_input(
                    "Blended energy price (PHP/kWh)",
                    min_value=0.0,
                    value=default_contract_php_per_kwh,
                    step=0.05,
                    help=("Applied to all delivered firm energy and marketed PV when blended pricing is enabled."),
                    disabled=not use_blended_price,
                )
                escalate_prices = st.checkbox("Escalate prices with inflation", value=False)
                wesm_pricing_enabled = st.checkbox(
                    "Apply WESM price to shortfalls",
                    value=False,
                    help=(
                        "Defaults to PHP 5,583/MWh from the 2024 Annual Market Assessment Report (PEMC); "
                        "enter a PHP/kWh rate to override."
                    ),
                )
                wesm_price_php_per_kwh = st.number_input(
                    "Average WESM price for shortfalls (PHP/kWh)",
                    min_value=0.0,
                    value=default_wesm_php_per_kwh,
                    step=0.05,
                    help="Applied to shortfall MWh as either a purchase cost or sale credit.",
                    disabled=not wesm_pricing_enabled,
                )
                sell_to_wesm = st.checkbox(
                    "Sell PV surplus to WESM",
                    value=False,
                    help=(
                        "When enabled, PV surplus (excess MWh) is credited at a WESM sale price; otherwise surplus "
                        "is excluded from revenue. Shortfalls always incur a WESM cost while this section is enabled."
                    ),
                    disabled=not wesm_pricing_enabled,
                )
                wesm_surplus_price_php_per_kwh = st.number_input(
                    "WESM sale price for PV surplus (PHP/kWh)",
                    min_value=0.0,
                    value=default_wesm_surplus_php_per_kwh,
                    step=0.05,
                    help=(
                        "Used only when selling PV surplus. Defaults to PHP 3.29/kWh based on the 2025 weighted "
                        "average WESM price; adjust to use your own PHP/kWh rate."
                    ),
                    disabled=not (wesm_pricing_enabled and sell_to_wesm),
                )

                contract_price = contract_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
                pv_market_price = pv_market_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
                blended_price_usd_per_mwh: Optional[float] = None
                wesm_price_usd_per_mwh: Optional[float] = None
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
                    wesm_price_usd_per_mwh = wesm_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
                    wesm_surplus_price_usd_per_mwh = (
                        wesm_surplus_price_php_per_kwh / forex_rate_php_per_usd * 1000.0 if sell_to_wesm else None
                    )
                    st.caption(
                        "WESM pricing active for shortfalls: "
                        f"PHP {wesm_price_php_per_kwh:,.2f}/kWh (≈${wesm_price_usd_per_mwh:,.2f}/MWh)."
                    )
                    if sell_to_wesm and wesm_surplus_price_usd_per_mwh is not None:
                        st.caption(
                            "PV surplus credited at a separate WESM sale rate: "
                            f"PHP {wesm_surplus_price_php_per_kwh:,.2f}/kWh "
                            f"(≈${wesm_surplus_price_usd_per_mwh:,.2f}/MWh)."
                            " Edit the PHP/kWh value to use a custom surplus rate."
                        )

            variable_col1, variable_col2 = st.columns(2)
            with variable_col1:
                variable_opex_php_per_kwh = st.number_input(
                    "Variable OPEX (PHP/kWh)",
                    min_value=0.0,
                    value=0.0,
                    step=0.05,
                    help=(
                        "Optional per-kWh operating expense applied to annual firm energy. "
                        "Escalates with inflation and overrides fixed OPEX when provided."
                    ),
                )
                variable_opex_usd_per_mwh: Optional[float] = None
                if variable_opex_php_per_kwh > 0:
                    variable_opex_usd_per_mwh = variable_opex_php_per_kwh / forex_rate_php_per_usd * 1000.0
                    st.caption(f"Converted variable OPEX: ${variable_opex_usd_per_mwh:,.2f}/MWh (applied to delivered energy).")
            with variable_col2:
                variable_schedule_choice = st.radio(
                    "Variable expense schedule",
                    options=["None", "Periodic", "Custom"],
                    horizontal=True,
                    help="Custom or periodic schedules override per-kWh and fixed OPEX assumptions.",
                )
                variable_opex_schedule_usd: Optional[Tuple[float, ...]] = None
                periodic_variable_opex_usd: Optional[float] = None
                periodic_variable_opex_interval_years: Optional[int] = None
                if variable_schedule_choice == "Periodic":
                    periodic_variable_opex_usd = st.number_input(
                        "Variable expense when periodic (USD)",
                        min_value=0.0,
                        value=0.0,
                        step=10_000.0,
                        help="Amount applied on the selected cadence (year 1, then every N years).",
                    )
                    periodic_variable_opex_interval_years = st.number_input("Cadence (years)", min_value=1, value=5, step=1)
                    if periodic_variable_opex_usd <= 0:
                        periodic_variable_opex_usd = None
                elif variable_schedule_choice == "Custom":
                    custom_variable_text = st.text_area(
                        "Custom variable expenses (USD/year)",
                        placeholder="e.g., 250000, 275000, 300000",
                        help=("Comma or newline separated values applied per project year. Length must match the simulation horizon."),
                    )
                    st.caption(f"Use commas or newlines between amounts; provide one value per project year ({cfg.years} entries).")
                    if custom_variable_text.strip():
                        try:
                            variable_opex_schedule_usd = tuple(
                                parse_numeric_series("Variable expense schedule", custom_variable_text)
                            )
                        except ValueError as exc:
                            st.error(str(exc))
                            st.stop()

            econ_inputs = EconomicInputs(
                capex_musd=capex_musd,
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
                wesm_price_usd_per_mwh=wesm_price_usd_per_mwh,
                wesm_surplus_price_usd_per_mwh=wesm_surplus_price_usd_per_mwh if wesm_pricing_enabled and sell_to_wesm else None,
                apply_wesm_to_shortfall=wesm_pricing_enabled,
                sell_to_wesm=sell_to_wesm if wesm_pricing_enabled else False,
            )

        run_cols = st.columns([2, 1])
        with run_cols[0]:
            run_submitted = st.form_submit_button(
                "Run simulation",
                use_container_width=True,
                help="Prevents auto-reruns while you adjust inputs; click to compute results.",
            )
        with run_cols[1]:
            st.caption("Edit parameters freely, then run when ready.")

    return SimulationFormResult(
        config=cfg,
        econ_inputs=econ_inputs,
        price_inputs=price_inputs,
        run_economics=run_economics,
        dod_override=dod_override,
        run_submitted=run_submitted,
        discharge_windows_text=discharge_windows_text,
        charge_windows_text=charge_windows_text,
    )
