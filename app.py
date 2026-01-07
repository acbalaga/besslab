import json
import math
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from services.simulation_core import HourlyLog, SimConfig, YearResult, resolve_efficiencies, simulate_project, summarize_simulation
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
from utils import (
    FLAG_DEFINITIONS,
    build_flag_insights,
    enforce_rate_limit,
)
from utils.economics import (
    CashFlowOutputs,
    EconomicInputs,
    EconomicOutputs,
    FinancingOutputs,
    PriceInputs,
    compute_cash_flows_and_irr,
    compute_financing_cash_flows,
    compute_lcoe_lcos_with_augmentation_fallback,
    estimate_augmentation_costs_by_year,
    normalize_economic_inputs,
)
from utils.ui_layout import init_page_layout
from utils.ui_state import (
    bootstrap_session_state,
    get_base_dir,
    get_simulation_results,
    load_shared_data,
    save_simulation_config,
    save_simulation_results,
    save_simulation_snapshot,
)


BASE_DIR = get_base_dir()


def run_app():
    bootstrap_session_state()
    debug_mode: bool = False
    render_layout = init_page_layout(
        page_title="Simulation",
        main_title="BESS LAB — PV-only charging, AC-coupled",
        description="Configure inputs, run the simulation, and review per-year results and sensitivities.",
        base_dir=BASE_DIR,
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
        st.caption(
            "If no files are uploaded, built-in defaults are read from ./data/. "
            "Current session caches the latest uploads."
        )

        st.divider()
        st.subheader("Rate limit override")
        render_rate_limit_section()

    pv_df, cycle_df = load_shared_data(BASE_DIR, pv_file, cycle_file)

    pv_df, cycle_df = render_layout(pv_df, cycle_df)

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
    bootstrap_session_state(cfg)

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
    elif cached_results is not None:
        sim_output = cached_results.sim_output
        st.caption(
            "Showing the latest completed simulation. Click 'Run simulation' to refresh after editing inputs."
        )

    results = sim_output.results
    monthly_results_all = sim_output.monthly_results
    first_year_logs = sim_output.first_year_logs
    final_year_logs = sim_output.final_year_logs
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

    if run_economics and econ_inputs and price_inputs:
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

        if normalized_econ_inputs and normalized_econ_inputs.include_devex_year0:
            st.caption(
                "DevEx: Included an additional "
                f"₱{normalized_econ_inputs.devex_cost_php / 1_000_000:,.0f}M "
                f"(≈${normalized_econ_inputs.devex_cost_usd / 1_000_000:,.2f}M) at year 0 across discounted costs, "
                "LCOE/LCOS, NPV, and IRR."
            )
        else:
            st.caption("DevEx not included; upfront spend reflects CAPEX only.")

        revenue_help = (
            "Revenues apply the blended energy price to all BESS deliveries and excess PV; "
            "contract/PV-specific rates are ignored in this mode."
            if price_inputs.blended_price_usd_per_mwh is not None
            else "Contract revenue from BESS deliveries plus market revenue from excess PV."
        )
        if price_inputs.apply_wesm_to_shortfall and price_inputs.wesm_price_usd_per_mwh is not None:
            revenue_help += (
                f" Shortfall MWh are deducted as a WESM cost at ${price_inputs.wesm_price_usd_per_mwh:,.2f}/MWh."
            )
            if price_inputs.sell_to_wesm:
                surplus_sale_rate = (
                    price_inputs.wesm_surplus_price_usd_per_mwh
                    if price_inputs.wesm_surplus_price_usd_per_mwh is not None
                    else price_inputs.wesm_price_usd_per_mwh
                )
                revenue_help += (
                    f" PV surplus is credited at ${surplus_sale_rate:,.2f}/MWh while selling to WESM; otherwise surplus is excluded from revenue."
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
        if price_inputs.apply_wesm_to_shortfall and price_inputs.wesm_price_usd_per_mwh is not None:
            wesm_price_php_per_kwh = price_inputs.wesm_price_usd_per_mwh / forex_rate_php_per_usd * 1000.0
            wesm_surplus_price_php_per_kwh = (
                price_inputs.wesm_surplus_price_usd_per_mwh / forex_rate_php_per_usd * 1000.0
                if price_inputs.sell_to_wesm and price_inputs.wesm_surplus_price_usd_per_mwh is not None
                else float("nan")
            )
            wesm_impact_musd = cash_outputs.discounted_wesm_value_usd / 1_000_000
            surplus_rate_usd = (
                price_inputs.wesm_surplus_price_usd_per_mwh
                if price_inputs.wesm_surplus_price_usd_per_mwh is not None
                else price_inputs.wesm_price_usd_per_mwh
            )
            surplus_note = (
                " PV surplus credited at the WESM sale rate due to the sell toggle"
                f" (PHP {wesm_surplus_price_php_per_kwh:,.2f}/kWh ≈ ${surplus_rate_usd:,.2f}/MWh)."
                if price_inputs.sell_to_wesm
                else " PV surplus excluded from revenue while WESM pricing is active."
            )
            wesm_caption = (
                "WESM shortfall cost applied at PHP "
                f"{wesm_price_php_per_kwh:,.2f}/kWh (≈${price_inputs.wesm_price_usd_per_mwh:,.2f}/MWh)."
                f" Discounted WESM impact on revenues/NPV/IRR: {wesm_impact_musd:,.2f} USD million." + surplus_note
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
    st.download_button("Download yearly summary (CSV)", res_df.to_csv(index=False).encode('utf-8'),
                       file_name='bess_yearly_summary.csv', mime='text/csv')

    st.download_button("Download monthly summary (CSV)", monthly_df.to_csv(index=False).encode('utf-8'),
                       file_name='bess_monthly_summary.csv', mime='text/csv')

    if final_year_logs is not None:
        hourly_df = pd.DataFrame({
            'hour_index': np.arange(len(final_year_logs.hod)),
            'hod': final_year_logs.hod,
            'pv_to_contract_mw': final_year_logs.pv_to_contract_mw,
            'bess_to_contract_mw': final_year_logs.bess_to_contract_mw,
            'charge_mw': final_year_logs.charge_mw,
            'discharge_mw': final_year_logs.discharge_mw,
            'soc_mwh': final_year_logs.soc_mwh,
            'pv_surplus_mw': np.maximum(
                final_year_logs.pv_mw - final_year_logs.pv_to_contract_mw - final_year_logs.charge_mw,
                0.0,
            ),
        })
        st.download_button("Download final-year hourly logs (CSV)", hourly_df.to_csv(index=False).encode('utf-8'),
                           file_name='final_year_hourly_logs.csv', mime='text/csv')

    pdf_bytes = None
    try:
        pdf_bytes = build_pdf_summary(cfg, results, kpis.compliance, kpis.bess_share_of_firm, kpis.charge_discharge_ratio,
                                      kpis.pv_capture_ratio, kpis.discharge_capacity_factor,
                                      discharge_windows_text, charge_windows_text,
                                      hod_count, hod_sum_pv_resource, hod_sum_pv, hod_sum_bess, hod_sum_charge,
                                      kpis.total_shortfall_mwh, kpis.pv_excess_mwh, kpis.total_project_generation_mwh,
                                      kpis.bess_generation_mwh, kpis.pv_generation_mwh, kpis.bess_losses_mwh)
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
