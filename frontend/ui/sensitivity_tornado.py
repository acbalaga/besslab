from __future__ import annotations

from dataclasses import replace
from typing import Optional

import altair as alt
import pandas as pd

from utils.economics import EconomicInputs, PriceInputs


def prepare_tornado_data(table: pd.DataFrame) -> pd.DataFrame:
    """Convert a lever impact table into the long format expected by the tornado chart."""

    scenario_label_map = {
        "Low impact (pp)": "Low impact",
        "High impact (pp)": "High impact",
    }
    numeric_table = table.copy()
    numeric_table["Low impact (pp)"] = pd.to_numeric(
        numeric_table["Low impact (pp)"], errors="coerce"
    ).fillna(0.0)
    numeric_table["High impact (pp)"] = pd.to_numeric(
        numeric_table["High impact (pp)"], errors="coerce"
    ).fillna(0.0)
    numeric_table["sort_key"] = numeric_table[["Low impact (pp)", "High impact (pp)"]].abs().max(axis=1)
    melted = numeric_table.melt(
        id_vars=["Lever", "Notes", "sort_key"],
        value_vars=["Low impact (pp)", "High impact (pp)"],
        var_name="Scenario",
        value_name="Impact (pp)",
    )
    melted["Scenario"] = melted["Scenario"].map(scenario_label_map).fillna(melted["Scenario"])
    return melted.sort_values("sort_key", ascending=True)


def build_tornado_chart(source: pd.DataFrame) -> alt.Chart:
    """Render a tornado bar chart from a melted impact table."""

    if source.empty:
        return alt.Chart(pd.DataFrame({"Impact (pp)": [], "Lever": []})).mark_bar()

    impact_domain = (
        float(source["Impact (pp)"].min()),
        float(source["Impact (pp)"].max()),
    )
    extent = max(abs(impact_domain[0]), abs(impact_domain[1]), 1.0)
    zero_line = alt.Chart(pd.DataFrame({"zero": [0]})).mark_rule(color="#6b6b6b").encode(x="zero:Q")

    bars = (
        alt.Chart(source)
        .mark_bar()
        .encode(
            x=alt.X(
                "Impact (pp):Q",
                title="Impact (percentage points)",
                scale=alt.Scale(domain=[-extent, extent]),
            ),
            y=alt.Y("Lever:N", sort=None, title="Sensitivity lever"),
            color=alt.Color(
                "Scenario:N",
                scale=alt.Scale(range=["#d95f02", "#1b9e77"]),
                title="Scenario",
            ),
            tooltip=[
                alt.Tooltip("Lever:N"),
                alt.Tooltip("Scenario:N"),
                alt.Tooltip("Impact (pp):Q", format=".2f"),
                alt.Tooltip("Notes:N"),
            ],
        )
    )
    return (bars + zero_line).properties(height=420)


def build_simple_lever_table() -> pd.DataFrame:
    """Return the minimal lever table for the main Inputs & Results tornado."""

    return pd.DataFrame(
        [
            {
                "Lever": "CAPEX",
                "Low change": -10.0,
                "High change": 10.0,
                "Low impact (pp)": None,
                "High impact (pp)": None,
                "Notes": "Finance-only (% change to CAPEX input).",
            },
            {
                "Lever": "OPEX",
                "Low change": -10.0,
                "High change": 10.0,
                "Low impact (pp)": None,
                "High impact (pp)": None,
                "Notes": "Finance-only (% change to fixed or variable OPEX inputs).",
            },
            {
                "Lever": "Tariff",
                "Low change": -5.0,
                "High change": 5.0,
                "Low impact (pp)": None,
                "High impact (pp)": None,
                "Notes": "Finance-only (% change to contract or blended price).",
            },
            {
                "Lever": "Availability",
                "Low change": -2.0,
                "High change": 2.0,
                "Low impact (pp)": None,
                "High impact (pp)": None,
                "Notes": "Availability delta (pp).",
            },
            {
                "Lever": "RTE",
                "Low change": -2.0,
                "High change": 2.0,
                "Low impact (pp)": None,
                "High impact (pp)": None,
                "Notes": "Roundtrip efficiency delta (pp).",
            },
            {
                "Lever": "BESS size (MWh)",
                "Low change": -10.0,
                "High change": 10.0,
                "Low impact (pp)": None,
                "High impact (pp)": None,
                "Notes": "Usable energy delta (MWh).",
            },
            {
                "Lever": "BESS capacity (MW)",
                "Low change": -5.0,
                "High change": 5.0,
                "Low impact (pp)": None,
                "High impact (pp)": None,
                "Notes": "Power rating delta (MW).",
            },
            {
                "Lever": "Dispatch window",
                "Low change": -1.0,
                "High change": 1.0,
                "Low impact (pp)": None,
                "High impact (pp)": None,
                "Notes": "Adjusts discharge window duration (hours).",
            },
        ]
    )


def apply_capex_delta(inputs: EconomicInputs, delta_pct: float) -> EconomicInputs:
    """Apply a percent delta to CAPEX inputs used in the finance model."""

    multiplier = 1.0 + delta_pct / 100.0
    capex_total_usd = inputs.capex_total_usd
    capex_usd_per_kwh = inputs.capex_usd_per_kwh
    capex_musd = inputs.capex_musd
    pv_capex_musd = inputs.pv_capex_musd
    if capex_total_usd is not None:
        capex_total_usd = max(capex_total_usd * multiplier, 0.0)
        capex_musd = capex_total_usd / 1_000_000.0
    elif capex_usd_per_kwh is not None:
        capex_usd_per_kwh = max(capex_usd_per_kwh * multiplier, 0.0)
        capex_musd = capex_musd * multiplier
    else:
        capex_musd = max(capex_musd * multiplier, 0.0)
    pv_capex_musd = max(pv_capex_musd * multiplier, 0.0)
    return replace(
        inputs,
        capex_total_usd=capex_total_usd,
        capex_usd_per_kwh=capex_usd_per_kwh,
        capex_musd=capex_musd,
        pv_capex_musd=pv_capex_musd,
    )


def apply_opex_delta(inputs: EconomicInputs, delta_pct: float) -> EconomicInputs:
    """Apply a percent delta to the active OPEX input path."""

    multiplier = 1.0 + delta_pct / 100.0
    fixed_opex_pct = inputs.fixed_opex_pct_of_capex
    fixed_opex_musd = inputs.fixed_opex_musd
    opex_php_per_kwh: Optional[float] = inputs.opex_php_per_kwh

    if opex_php_per_kwh is not None:
        opex_php_per_kwh = max(opex_php_per_kwh * multiplier, 0.0)
    elif fixed_opex_pct > 0:
        fixed_opex_pct = max(fixed_opex_pct * multiplier, 0.0)
    elif fixed_opex_musd > 0:
        fixed_opex_musd = max(fixed_opex_musd * multiplier, 0.0)

    return replace(
        inputs,
        fixed_opex_pct_of_capex=fixed_opex_pct,
        fixed_opex_musd=fixed_opex_musd,
        opex_php_per_kwh=opex_php_per_kwh,
    )


def apply_tariff_delta(inputs: PriceInputs, delta_pct: float) -> PriceInputs:
    """Apply a percent delta to the active tariff (blended or contract/PV prices)."""

    multiplier = 1.0 + delta_pct / 100.0
    if inputs.blended_price_usd_per_mwh is not None:
        return replace(
            inputs,
            blended_price_usd_per_mwh=max(inputs.blended_price_usd_per_mwh * multiplier, 0.0),
        )

    return replace(
        inputs,
        contract_price_usd_per_mwh=max(inputs.contract_price_usd_per_mwh * multiplier, 0.0),
        pv_market_price_usd_per_mwh=max(inputs.pv_market_price_usd_per_mwh * multiplier, 0.0),
    )
