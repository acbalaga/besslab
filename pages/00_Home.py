import streamlit as st

from app import BASE_DIR
from utils.ui_state import get_shared_data

st.set_page_config(page_title="BESSLab — Guide", layout="wide")

st.title("BESSLab guide and tips")

pv_df, cycle_df = get_shared_data(BASE_DIR)
st.caption(
    f"Session data ready: PV profile with {len(pv_df):,} rows and cycle model with {len(cycle_df)} rows."
)

st.markdown(
    """
## Welcome to BESSLab
This guide summarizes the multipage workflow for the PV-only, AC-coupled BESS model.

### Getting started
1) Upload a PV 8760 CSV (`hour_index, pv_mw`) on the landing page or keep the bundled sample.
2) (Optional) Upload a cycle-model Excel file to supply your own degradation table.
3) Configure contracted MW, duration (hours), and dispatch windows in the Inputs page.
4) Adjust efficiency, SOC limits, availability, augmentation triggers, and rate-limit settings as needed.
5) Review the summary cards and charts, then download CSV or PDF outputs.
6) Rerun Design Advisor recommendations or SOC sensitivity sweeps after major input changes.

### What you will see
- Contract compliance over the project life.
- Energy split between direct PV delivery and the BESS, including shortfall flags.
- End-of-year capability bars, typical daily profiles, and economics sensitivities.
- Design Advisor guidance and SOC-window sensitivity sweeps.
- A scenario table for saving and comparing input sets, plus hourly/monthly/yearly downloads.

### Helpful extras
- Use the sidebar link to open the economics helper page (LCOE/LCOS) and download the module.
- Disable the rate limit by entering the password in the sidebar (default: `besslab`).
- Window strings accept minutes (e.g., `05:30-09:00`), parsed as fractional hours.

### If results look off
- For shortfalls, widen the SOC window, improve efficiency, or enable augmentation.
- If the battery often empties, increase duration (MWh), raise the SOC ceiling, or allow more charge time.
- If it frequently tops out, lower the SOC ceiling slightly or allow more discharge.
- For unexpected economics results, confirm price/cost units and rerun the sensitivity heatmaps.

### Notes
- `hour_index` can start at 0 or 1; the app aligns it automatically.
- Defaults are loaded when no files are uploaded.
- The model enforces PV-only charging; no grid charging is considered.
- Sensitivity sweeps clear when inputs change; rerun them after adjustments.
    """
)

st.page_link("pages/01_Simulation.py", label="Jump to Inputs & Results")
