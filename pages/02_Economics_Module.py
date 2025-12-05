from __future__ import annotations

from io import StringIO
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from utils.economics import EconomicInputs, compute_lcoe_lcos

st.set_page_config(page_title="Economics module — LCOE/LCOS helper", layout="wide")

st.title("Economics helper module (LCOE / LCOS)")
st.caption(
    "This page surfaces the standalone economics helper so you can open it in a separate tab "
    "while running scenarios in the main app."
)

st.markdown(
    """
### How to import and use
```python
from utils.economics import EconomicInputs, compute_lcoe_lcos

inputs = EconomicInputs(
    capex_musd=40.0,
    fixed_opex_pct_of_capex=2.5,
    fixed_opex_musd=0.0,
    variable_opex_usd_per_mwh=1.0,
    discount_rate=0.05,
)
outputs = compute_lcoe_lcos(
    annual_delivered_mwh=[120_000, 118_000, 117_000],
    annual_bess_mwh=[60_000, 58_000, 57_000],
    inputs=inputs,
)
print(outputs.lcoe_usd_per_mwh, outputs.lcos_usd_per_mwh)
```
    """
)

st.markdown("### Download or view the module")
module_path = Path(__file__).resolve().parent.parent / "utils" / "economics.py"
source_text = module_path.read_text(encoding="utf-8")

st.download_button(
    label="Download utils/economics.py",
    data=source_text,
    file_name="economics.py",
    mime="text/x-python",
    use_container_width=True,
)

with st.expander("Preview utils/economics.py", expanded=False):
    st.code(source_text, language="python")

st.info(
    "Use the sidebar link in the main BESSLab page to come back here. "
    "Right-click the sidebar link to open this helper in a new tab."
)


st.markdown("---")
st.markdown("### LCOE vs. initial BESS capacity")
st.caption(
    "Paste quick what-if data to visualize how levelized cost shifts with initial usable"
    " BESS energy. The lowest point is highlighted as the \"sweet spot.\""
)

default_series = """bess_mwh,lcoe_usd_per_mwh
80,65.0
120,64.5
160,65.3
200,67.2
240,72.1
"""

data_text = st.text_area(
    "BESS capacity (MWh) and LCOE ($/MWh)",
    value=default_series,
    help=(
        "Provide two columns named 'bess_mwh' and 'lcoe_usd_per_mwh'."
        " Values can be comma- or tab-separated."
    ),
    height=180,
)


def _parse_lcoe_series(raw_text: str) -> pd.DataFrame:
    """Return a clean DataFrame for plotting the LCOE sensitivity chart."""

    cleaned_text = raw_text.strip()
    if not cleaned_text:
        return pd.DataFrame(columns=["bess_mwh", "lcoe_usd_per_mwh"])

    return pd.read_csv(StringIO(cleaned_text))


chart_container = st.container()
try:
    lcoe_df = _parse_lcoe_series(data_text)
    lcoe_df["bess_mwh"] = pd.to_numeric(lcoe_df["bess_mwh"], errors="coerce")
    lcoe_df["lcoe_usd_per_mwh"] = pd.to_numeric(
        lcoe_df["lcoe_usd_per_mwh"], errors="coerce"
    )
    lcoe_df = lcoe_df.dropna(subset=["bess_mwh", "lcoe_usd_per_mwh"]).sort_values(
        "bess_mwh"
    )
except Exception as exc:  # pragma: no cover - defensive around free-form input
    chart_container.error(
        "Could not parse the provided series. Ensure the headers are present and numeric values"
        f" are valid (details: {exc})."
    )
else:
    if lcoe_df.empty:
        chart_container.info("Provide at least one row to plot the sensitivity curve.")
    else:
        sweet_spot = lcoe_df.loc[lcoe_df["lcoe_usd_per_mwh"].idxmin()]
        sweet_capacity = sweet_spot["bess_mwh"]
        sweet_lcoe = sweet_spot["lcoe_usd_per_mwh"]

        chart_container.metric(
            "Sweet spot",
            f"{sweet_capacity:,.0f} MWh",
            help="The capacity with the lowest LCOE in the provided series.",
        )

        base_chart = alt.Chart(lcoe_df).encode(
            x=alt.X("bess_mwh", title="Initial BESS capacity (MWh)"),
            y=alt.Y("lcoe_usd_per_mwh", title="LCOE ($/MWh)"),
        )

        line = base_chart.mark_line(color="#d62728").interactive()
        points = base_chart.mark_circle(color="#d62728", size=80)
        sweet_rule = alt.Chart(pd.DataFrame({
            "bess_mwh": [sweet_capacity],
            "lcoe_usd_per_mwh": [sweet_lcoe],
            "label": [f"Sweet spot: {sweet_capacity:,.0f} MWh"]
        })).encode(
            x="bess_mwh",
            y="lcoe_usd_per_mwh",
            tooltip=["bess_mwh", "lcoe_usd_per_mwh"],
            text="label",
        ).mark_text(dy=-10, color="#d62728")

        chart = (line + points + sweet_rule).properties(height=320)
        chart_container.altair_chart(chart, use_container_width=True)

        chart_container.caption(
            "Data is treated as-is; no interpolation or unit conversions are applied."
        )
