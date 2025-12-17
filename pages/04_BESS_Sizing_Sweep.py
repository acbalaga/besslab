from typing import Optional, Tuple
import math

import altair as alt
import pandas as pd
import streamlit as st

from app import BASE_DIR, SimConfig
from utils import enforce_rate_limit
from utils.economics import EconomicInputs, PriceInputs
from utils.sweeps import generate_values, sweep_bess_sizes
from utils.ui_state import get_shared_data

st.set_page_config(page_title="BESS Sizing Sweep", layout="wide")

st.title("BESS sizing sweep (energy sensitivity)")
st.caption("Sweep over usable energy (MWh) while holding power constant to see feasibility, LCOE, and NPV.")


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


def recommend_convergence_point(df: pd.DataFrame) -> Optional[Tuple[float, float, float]]:
    """Identify the BESS capacity where NPV and IRR curves overlap after scaling.

    The NPV and IRR charts use different axes, so we normalize each series to a
    0–1 range and locate the energy point where the curves are closest. A small
    penalty is applied to negative NPVs to avoid recommending designs with weak
    economics even if their normalized values cross. Returns ``(energy_mwh,
    npv_usd, irr_pct)`` when a convergence point can be inferred.
    """

    columns = ["energy_mwh", "npv_costs_usd", "irr_pct"]
    if not set(columns).issubset(df.columns):
        return None

    clean_df = df[columns].replace([math.inf, -math.inf], float("nan")).dropna()
    if clean_df.empty:
        return None

    npv_min, npv_max = clean_df["npv_costs_usd"].min(), clean_df["npv_costs_usd"].max()
    irr_min, irr_max = clean_df["irr_pct"].min(), clean_df["irr_pct"].max()
    if npv_max == npv_min or irr_max == irr_min:
        return None

    normalized = clean_df.assign(
        npv_norm=lambda x: (x["npv_costs_usd"] - npv_min) / (npv_max - npv_min),
        irr_norm=lambda x: (x["irr_pct"] - irr_min) / (irr_max - irr_min),
    )

    penalty_scale = max(abs(npv_min), abs(npv_max), 1.0)
    normalized["intersection_score"] = (
        (normalized["npv_norm"] - normalized["irr_norm"]).abs()
        + (normalized["npv_costs_usd"].clip(upper=0.0).abs() / penalty_scale) * 0.1
    )

    best_row = normalized.nsmallest(1, "intersection_score")
    if best_row.empty:
        return None

    chosen = best_row.iloc[0]
    return float(chosen["energy_mwh"]), float(chosen["npv_costs_usd"]), float(chosen["irr_pct"])

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
            help="Applied to all delivered firm energy and marketed PV when blended pricing is enabled.",
            disabled=not use_blended_price,
        )
        escalate_prices = st.checkbox(
            "Escalate prices with inflation",
            value=False,
        )

        contract_price = contract_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
        pv_market_price = pv_market_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
        blended_price_usd_per_mwh: Optional[float] = None
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
            periodic_variable_opex_interval_years = st.number_input(
                "Cadence (years)",
                min_value=1,
                value=5,
                step=1,
            )
            if periodic_variable_opex_usd <= 0:
                periodic_variable_opex_usd = None
        elif variable_schedule_choice == "Custom":
            custom_variable_text = st.text_area(
                "Custom variable expenses (USD/year)",
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
        variable_opex_usd_per_mwh=variable_opex_usd_per_mwh,
        variable_opex_schedule_usd=variable_opex_schedule_usd,
        periodic_variable_opex_usd=periodic_variable_opex_usd,
        periodic_variable_opex_interval_years=periodic_variable_opex_interval_years,
    )
    price_inputs = PriceInputs(
        contract_price_usd_per_mwh=contract_price,
        pv_market_price_usd_per_mwh=pv_market_price,
        escalate_with_inflation=escalate_prices,
        blended_price_usd_per_mwh=blended_price_usd_per_mwh,
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
    convergence_point = recommend_convergence_point(sweep_df)
    if convergence_point:
        energy_mwh, npv_usd, irr_pct = convergence_point
        st.info(
            "Convergence point (NPV vs IRR): "
            f"~{energy_mwh:.1f} MWh usable with IRR {irr_pct:.2f}% and NPV ${npv_usd:,.0f}. "
            "Curves are normalized to locate where returns and discounted costs align, "
            "favoring options that avoid very negative NPVs when CAPEX scales linearly "
            "with BESS size and resource availability limits upside energy."
        )

    convergence_point = recommend_convergence_point(sweep_df)
    if convergence_point:
        energy_mwh, npv_usd, irr_pct = convergence_point
        st.info(
            "Convergence point (NPV vs IRR): "
            f"~{energy_mwh:.1f} MWh usable with IRR {irr_pct:.2f}% and NPV ${npv_usd:,.0f}. "
            "Curves are normalized to locate where returns and discounted costs align, "
            "favoring options that avoid very negative NPVs when CAPEX scales linearly "
            "with BESS size and resource availability limits upside energy."
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

    chart_df = sweep_df[["energy_mwh", "npv_costs_usd"]].copy()
    chart_df["irr_pct"] = sweep_df.get("irr_pct", float("nan"))
    chart_df = chart_df.sort_values("energy_mwh")

    base_chart = alt.Chart(chart_df).encode(
        x=alt.X("energy_mwh", title="BESS capacity (MWh)", axis=alt.Axis(format=",.0f"))
    )

    point_tooltip = [
        alt.Tooltip("energy_mwh", title="BESS capacity (MWh)", format=",.0f"),
        alt.Tooltip("npv_costs_usd", title="NPV (USD)", format=",.0f"),
        alt.Tooltip("irr_pct", title="IRR (%)", format=",.2f"),
    ]

    npv_line = base_chart.mark_line(color="#0b2c66", point=alt.OverlayMarkDef(filled=True, size=90)).encode(
        y=alt.Y(
            "npv_costs_usd",
            title="NPV (USD)",
            axis=alt.Axis(titleColor="#0b2c66", format=",.0f"),
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

    st.altair_chart(
        alt.layer(npv_line, irr_line).resolve_scale(y="independent"),
        use_container_width=True,
    )
    st.caption(
        "Dual-axis line chart overlays NPV (USD) and IRR (%) across BESS capacities; IRR points are omitted when unavailable."
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
