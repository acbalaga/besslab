from typing import Optional
import math

import altair as alt
import streamlit as st

from app import BASE_DIR, SimConfig
from utils import enforce_rate_limit
from utils.economics import EconomicInputs, PriceInputs
from utils.sweeps import generate_values, sweep_bess_sizes
from utils.ui_state import get_shared_data

st.set_page_config(page_title="BESS Sizing Sweep", layout="wide")

st.title("BESS sizing sweep (energy sensitivity)")
st.caption("Sweep over usable energy (MWh) while holding power constant to see feasibility, LCOE, and NPV.")

pv_df, cycle_df = get_shared_data(BASE_DIR)
cfg: Optional[SimConfig] = st.session_state.get("latest_sim_config")
dod_override = st.session_state.get("latest_dod_override", "Auto (infer)")
forex_rate_php_per_usd = 58.0
default_contract_php_per_kwh = round(120.0 / 1000.0 * forex_rate_php_per_usd, 2)
default_pv_php_per_kwh = round(55.0 / 1000.0 * forex_rate_php_per_usd, 2)

if cfg is None:
    st.warning(
        "No inputs cached yet. Open the Inputs & Results page, adjust settings, and rerun the simulation to seed the sweep.",
        icon="⚠️",
    )
    cfg = SimConfig()

st.page_link("app.py", label="Back to Inputs & Results", help="Update inputs before rerunning the sweep.")

st.markdown("---")

st.session_state.setdefault("bess_size_sweep_results", None)

with st.form("size_sweep_form_page"):
    size_col1, size_col2, size_col3, price_col = st.columns(4)
    with size_col1:
        default_energy = max(10.0, cfg.initial_usable_mwh)
        energy_range = st.slider(
            "Usable energy range (MWh)",
            min_value=10.0,
            max_value=500.0,
            value=(max(10.0, default_energy * 0.5), min(500.0, default_energy * 1.5)),
            step=5.0,
            help="Lower and upper bounds for the usable MWh grid.",
        )
        energy_steps = st.number_input(
            "Energy points",
            min_value=1,
            max_value=15,
            value=5,
            help="Number of evenly spaced usable-energy values between the bounds.",
        )
        fixed_power = st.number_input(
            "Fixed discharge power (MW)",
            min_value=0.1,
            max_value=300.0,
            value=float(cfg.initial_power_mw),
            step=0.1,
            help="Power rating held constant while sweeping usable energy.",
        )

    with size_col2:
        wacc_pct = st.number_input(
            "WACC (%)",
            min_value=0.0,
            max_value=30.0,
            value=8.0,
            step=0.1,
            help="Weighted-average cost of capital (nominal).",
        )
        inflation_pct = st.number_input(
            "Inflation rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            step=0.1,
            help="Inflation assumption used to derive the real discount rate.",
        )
        discount_rate = max((1 + wacc_pct / 100.0) / (1 + inflation_pct / 100.0) - 1, 0.0)
        capex_musd = st.number_input(
            "Total CAPEX (USD million)",
            min_value=0.0,
            value=40.0,
            step=0.1,
            help="All-in CAPEX for the project. Expressed in USD millions for compact entry.",
        )
        fixed_opex_pct = st.number_input(
            "Fixed OPEX (% of CAPEX per year)",
            min_value=0.0,
            max_value=20.0,
            value=2.0,
            step=0.1,
            help="Annual fixed OPEX expressed as % of CAPEX.",
        ) / 100.0
        fixed_opex_musd = st.number_input(
            "Additional fixed OPEX (USD million/yr)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            help="Extra fixed OPEX not tied to CAPEX percentage.",
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
            ],
            format_func=lambda x: {
                "compliance_pct": "Compliance % (higher is better)",
                "total_shortfall_mwh": "Shortfall MWh (lower is better)",
                "total_project_generation_mwh": "Total generation (higher is better)",
                "bess_generation_mwh": "BESS discharge (higher is better)",
                "lcoe_usd_per_mwh": "LCOE ($/MWh, lower is better)",
                "npv_costs_usd": "NPV of costs (USD, lower is better)",
            }.get(x, x),
            help="Column used to pick the top feasible design.",
        )
        min_soh = st.number_input(
            "Minimum SOH for feasibility",
            min_value=0.2,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Candidates falling below this total SOH are flagged as infeasible.",
        )
        st.caption(
            "Discount rate is derived from WACC and inflation to align with the economics helper."
        )

    with price_col:
        contract_price_php_per_kwh = st.number_input(
            "Contract price (PHP/kWh from BESS)",
            min_value=0.0,
            value=default_contract_php_per_kwh,
            step=0.05,
            help="Price converted to USD/MWh internally using PHP 58/USD.",
        )
        pv_market_price_php_per_kwh = st.number_input(
            "PV market price (PHP/kWh for excess PV)",
            min_value=0.0,
            value=default_pv_php_per_kwh,
            step=0.05,
            help="Price converted to USD/MWh internally using PHP 58/USD.",
        )
        escalate_prices = st.checkbox(
            "Escalate prices with inflation",
            value=False,
        )

        contract_price = contract_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
        pv_market_price = pv_market_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
        st.caption(
            f"Converted contract price: ${contract_price:,.2f}/MWh | PV market price: ${pv_market_price:,.2f}/MWh"
        )

    submitted = st.form_submit_button("Run BESS energy sweep", use_container_width=True)

if submitted:
    enforce_rate_limit()
    energy_values = generate_values(energy_range[0], energy_range[1], int(energy_steps))
    economics_inputs = EconomicInputs(
        capex_musd=capex_musd,
        fixed_opex_pct_of_capex=fixed_opex_pct,
        fixed_opex_musd=fixed_opex_musd,
        inflation_rate=inflation_pct / 100.0,
        discount_rate=discount_rate,
    )
    price_inputs = PriceInputs(
        contract_price_usd_per_mwh=contract_price,
        pv_market_price_usd_per_mwh=pv_market_price,
        escalate_with_inflation=escalate_prices,
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

sweep_df = st.session_state.get("bess_size_sweep_results")
if sweep_df is not None:
    best_row = sweep_df[sweep_df["is_best"]]
    if best_row.empty:
        st.warning("No feasible candidates met the SOH/cycle thresholds.")
    else:
        best = best_row.iloc[0]
        lcoe_text = ""
        if not math.isnan(best.get("lcoe_usd_per_mwh", float("nan"))):
            lcoe_text = f" — LCOE {best['lcoe_usd_per_mwh']:.0f} $/MWh"
        st.success(
            "Best feasible: "
            f"{best['energy_mwh']:.1f} MWh usable @ {best['power_mw']:.1f} MW "
            f"({best['duration_h']:.2f} h){lcoe_text}"
        )

    st.dataframe(
        sweep_df[
            [
                "power_mw",
                "duration_h",
                "energy_mwh",
                "compliance_pct",
                "total_shortfall_mwh",
                "avg_eq_cycles_per_year",
                "min_soh_total",
                "lcoe_usd_per_mwh",
                "npv_costs_usd",
                "feasible",
                "is_best",
            ]
        ],
        use_container_width=True,
    )

    chart_df = sweep_df[["energy_mwh", "lcoe_usd_per_mwh"]].copy()
    if "irr_pct" in sweep_df.columns:
        chart_df["irr_pct"] = sweep_df["irr_pct"]
    else:
        chart_df["irr_pct"] = float("nan")

    chart_df = chart_df.sort_values("energy_mwh")

    base_chart = alt.Chart(chart_df).encode(
        x=alt.X("energy_mwh", title="Usable energy (MWh)")
    )

    point_tooltip = [
        alt.Tooltip("energy_mwh", title="Usable energy (MWh)", format=",.0f"),
        alt.Tooltip("lcoe_usd_per_mwh", title="LCOE ($/MWh)", format=",.0f"),
        alt.Tooltip("irr_pct", title="IRR (%)", format=",.2f"),
    ]

    lcoe_points = base_chart.mark_point(filled=True, color="#d62728", size=80).encode(
        y=alt.Y(
            "lcoe_usd_per_mwh",
            title="LCOE ($/MWh)",
            axis=alt.Axis(titleColor="#d62728"),
        ),
        tooltip=point_tooltip,
    )

    irr_points = base_chart.mark_point(filled=True, color="#9467bd", size=80, shape="diamond").encode(
        y=alt.Y(
            "irr_pct",
            title="IRR (%)",
            axis=alt.Axis(titleColor="#9467bd", orient="right"),
        ),
        tooltip=point_tooltip,
    )

    st.altair_chart(
        alt.layer(lcoe_points, irr_points).resolve_scale(y="independent"),
        use_container_width=True,
    )
    st.caption(
        "Dual-axis scatter shows LCOE and IRR for each BESS size; IRR points may be omitted when unavailable."
    )
else:
    st.info(
        "Run the sweep with your latest inputs. Results persist in the session for quick iteration.",
        icon="ℹ️",
    )

st.markdown("---")
st.subheader("Navigate across the workspace")
st.page_link("pages/00_Home.py", label="Home (Guide)")
st.page_link("app.py", label="Simulation (Inputs & Results)")
st.page_link("pages/03_Scenario_Comparisons.py", label="Scenario comparisons")
