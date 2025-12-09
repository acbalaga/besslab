from __future__ import annotations

from typing import List, Sequence

import streamlit as st

from utils.economics import (
    CashFlowOutputs,
    EconomicInputs,
    PriceInputs,
    compute_cash_flows_and_irr,
    compute_lcoe_lcos,
)
from utils.ui_state import get_latest_economics_payload, hide_root_page_from_sidebar

st.set_page_config(page_title="Economics module — LCOE/LCOS helper", layout="wide")
hide_root_page_from_sidebar()

st.title("Economics helper module (LCOE / LCOS)")
st.caption(
    "Run the same economics engine used in the Simulation page with any energy series. "
    "You can pull the latest Simulation inputs, tweak them here, and compute LCOE/LCOS "
    "without rerunning the full dispatch model."
)


def _series_to_text(values: Sequence[float]) -> str:
    if not values:
        return ""
    return "\n".join(f"{v:.2f}" for v in values)


def _parse_numeric_series(raw_text: str, label: str) -> List[float]:
    tokens = [t.strip() for t in raw_text.replace(",", "\n").splitlines() if t.strip()]
    series: List[float] = []
    for token in tokens:
        try:
            series.append(float(token))
        except ValueError as exc:  # noqa: BLE001
            st.error(f"{label} contains a non-numeric entry: '{token}'")
            raise
    return series


def _render_outputs(econ_output, cash_output: CashFlowOutputs) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Discounted costs (USD million)",
        f"{econ_output.discounted_costs_usd / 1_000_000:,.2f}",
        help="CAPEX at year 0 plus discounted OPEX and augmentation across the project horizon.",
    )
    col2.metric(
        "LCOE ($/MWh delivered)",
        f"{econ_output.lcoe_usd_per_mwh:,.2f}",
        help="Total discounted costs ÷ discounted firm energy delivered.",
    )
    col3.metric(
        "LCOS ($/MWh from BESS)",
        f"{econ_output.lcos_usd_per_mwh:,.2f}",
        help="Same cost base divided by discounted BESS contribution only.",
    )

    cf1, cf2, cf3 = st.columns(3)
    cf1.metric(
        "Discounted revenues (USD million)",
        f"{cash_output.discounted_revenues_usd / 1_000_000:,.2f}",
        help="Contract revenue from BESS deliveries plus market revenue from excess PV.",
    )
    cf2.metric(
        "NPV (USD million)",
        f"{cash_output.npv_usd / 1_000_000:,.2f}",
        help="Discounted cash flows using the chosen discount rate (year 0 CAPEX included).",
    )
    cf3.metric(
        "Project IRR (%)",
        f"{cash_output.irr_pct:,.2f}%" if cash_output.irr_pct == cash_output.irr_pct else "—",
        help="IRR computed from annual revenues and OPEX/augmentation outflows.",
    )


latest_payload = get_latest_economics_payload()
if latest_payload:
    st.success(
        "Using energy and economics inputs from your most recent Simulation run. "
        "Adjust any field below to rerun with overrides.",
    )
else:
    st.warning(
        "No Simulation inputs detected yet. Open the Simulation page, run a scenario, "
        "and return here to reuse the outputs automatically.",
        icon="⚠️",
    )

with st.form("economics_form"):
    use_sim_defaults = st.checkbox(
        "Start with the latest Simulation inputs", value=bool(latest_payload)
    )

    if use_sim_defaults and latest_payload:
        delivered_default = latest_payload.get("annual_delivered_mwh", [])
        bess_default = latest_payload.get("annual_bess_mwh", [])
        pv_excess_default = latest_payload.get("annual_pv_excess_mwh", [])
        augmentation_default = latest_payload.get("augmentation_costs_usd", [])
        econ_defaults = latest_payload.get("economic_inputs", {})
        price_defaults = latest_payload.get("price_inputs", {})
    else:
        delivered_default = [120_000.0, 118_000.0, 116_000.0]
        bess_default = [60_000.0, 58_000.0, 57_000.0]
        pv_excess_default = [10_000.0, 9_000.0, 8_000.0]
        augmentation_default: List[float] = []
        econ_defaults = {}
        price_defaults = {}

    st.markdown("### Energy and augmentation series")
    series_col1, series_col2 = st.columns(2)
    with series_col1:
        delivered_text = st.text_area(
            "Annual firm energy delivered to contract (MWh)",
            value=_series_to_text(delivered_default),
            help="One value per project year. Commas or newlines are accepted.",
            height=140,
        )
        bess_text = st.text_area(
            "Annual BESS energy delivered (MWh)",
            value=_series_to_text(bess_default),
            help="Must align with the contract energy series above.",
            height=140,
        )
    with series_col2:
        pv_excess_text = st.text_area(
            "Annual excess PV sold to market (MWh)",
            value=_series_to_text(pv_excess_default),
            help="Required for cash-flow metrics. Leave blank to treat as zero.",
            height=140,
        )
        augmentation_text = st.text_area(
            "Augmentation CAPEX by year (USD)",
            value=_series_to_text(augmentation_default),
            help="Optional. Empty means no augmentation spend is included.",
            height=140,
        )

    st.markdown("### Economic and price assumptions")
    econ_col1, econ_col2, econ_col3 = st.columns(3)
    with econ_col1:
        wacc_pct = st.number_input(
            "WACC (%)",
            min_value=0.0,
            max_value=30.0,
            value=float(econ_defaults.get("wacc_pct", 8.0)),
            step=0.1,
        )
        inflation_pct = st.number_input(
            "Inflation rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=float(econ_defaults.get("inflation_rate_pct", 3.0)),
            step=0.1,
            help="Used to derive the real discount rate applied to costs and revenues.",
        )
        discount_rate = max((1 + wacc_pct / 100.0) / (1 + inflation_pct / 100.0) - 1, 0.0)
        st.caption(f"Real discount rate derived from WACC and inflation: {discount_rate * 100:.2f}%.")
    with econ_col2:
        capex_musd = st.number_input(
            "Total CAPEX (USD million)",
            min_value=0.0,
            value=float(econ_defaults.get("capex_musd", 40.0)),
            step=0.1,
        )
        fixed_opex_pct = (
            st.number_input(
                "Fixed OPEX (% of CAPEX per year)",
                min_value=0.0,
                max_value=20.0,
                value=float(econ_defaults.get("fixed_opex_pct_of_capex", 2.0)),
                step=0.1,
            )
            / 100.0
        )
        fixed_opex_musd = st.number_input(
            "Additional fixed OPEX (USD million/yr)",
            min_value=0.0,
            value=float(econ_defaults.get("fixed_opex_musd", 0.0)),
            step=0.1,
        )
    with econ_col3:
        contract_price = st.number_input(
            "Contract price (USD/MWh from BESS)",
            min_value=0.0,
            value=float(price_defaults.get("contract_price_usd_per_mwh", 120.0)),
            step=1.0,
        )
        pv_market_price = st.number_input(
            "PV market price (USD/MWh for excess PV)",
            min_value=0.0,
            value=float(price_defaults.get("pv_market_price_usd_per_mwh", 55.0)),
            step=1.0,
        )
        escalate_prices = st.checkbox(
            "Escalate prices with inflation",
            value=bool(price_defaults.get("escalate_with_inflation", False)),
        )

    submitted = st.form_submit_button("Run economics module", use_container_width=True)

if submitted:
    try:
        annual_delivered = _parse_numeric_series(delivered_text, "Contract energy")
        annual_bess = _parse_numeric_series(bess_text, "BESS energy")
        pv_excess = _parse_numeric_series(pv_excess_text, "Excess PV") if pv_excess_text.strip() else []
        augmentation_costs = (
            _parse_numeric_series(augmentation_text, "Augmentation costs") if augmentation_text.strip() else []
        )
    except ValueError:
        st.stop()

    if len(annual_delivered) != len(annual_bess):
        st.error("Contract and BESS energy series must have the same number of years.")
        st.stop()
    if pv_excess and len(pv_excess) != len(annual_delivered):
        st.error("Excess PV series must align with the contract energy series.")
        st.stop()
    if augmentation_costs and len(augmentation_costs) != len(annual_delivered):
        st.error("Augmentation costs must match the number of years or be left blank.")
        st.stop()

    economic_inputs = EconomicInputs(
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

    econ_output = compute_lcoe_lcos(
        annual_delivered,
        annual_bess,
        economic_inputs,
        augmentation_costs_usd=augmentation_costs if augmentation_costs else None,
    )
    cash_output = compute_cash_flows_and_irr(
        annual_delivered,
        annual_bess,
        pv_excess or [0.0 for _ in annual_delivered],
        economic_inputs,
        price_inputs,
        augmentation_costs_usd=augmentation_costs if augmentation_costs else None,
    )

    st.markdown("---")
    st.subheader("Results")
    _render_outputs(econ_output, cash_output)

st.markdown("---")
st.subheader("Navigate across the workspace")
st.page_link("pages/00_Home.py", label="Home (Guide)")
st.page_link("pages/01_Simulation.py", label="Simulation (Inputs & Results)")
st.page_link("pages/03_Scenario_Comparisons.py", label="Scenario comparisons")
st.page_link("pages/04_BESS_Sizing_Sweep.py", label="BESS sizing sweep")
