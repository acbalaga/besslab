import streamlit as st
from pathlib import Path

from economics import EconomicInputs, compute_lcoe_lcos

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
from economics import EconomicInputs, compute_lcoe_lcos

inputs = EconomicInputs(
    capex_musd=40.0,
    fixed_opex_pct_of_capex=2.5,
    fixed_opex_musd=0.0,
    variable_opex_usd_per_mwh=1.0,
    discount_rate=0.05,
)
outputs = compute_lcoe_lcos(
    delivered_mwh_by_year=[120_000, 118_000, 117_000],
    bess_mwh_by_year=[60_000, 58_000, 57_000],
    inputs=inputs,
)
print(outputs.lcoe_usd_per_mwh, outputs.lcos_usd_per_mwh)
```
    """
)

st.markdown("### Download or view the module")
module_path = Path(__file__).resolve().parent.parent / "economics.py"
source_text = module_path.read_text(encoding="utf-8")

st.download_button(
    label="Download economics.py",
    data=source_text,
    file_name="economics.py",
    mime="text/x-python",
    use_container_width=True,
)

with st.expander("Preview economics.py", expanded=False):
    st.code(source_text, language="python")

st.info(
    "Use the sidebar link in the main BESSLab page to come back here. "
    "Right-click the sidebar link to open this helper in a new tab."
)
