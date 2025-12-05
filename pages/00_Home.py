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
A quick-start guide for exploring **PV-only, AC-coupled BESS** behavior.

### How to get started
1) Upload a **PV 8760 CSV** (`hour_index, pv_mw`) or keep the sample file on the landing page.
2) (Optional) Upload a **cycle-model Excel** file to use your own degradation table.
3) Set your **contracted MW**, **duration (hours)**, and **dispatch windows** in the Inputs page.
4) Adjust efficiency, SOC limits, availability, augmentation triggers, and rate-limit settings if needed.
5) Review the summary cards and charts, then download the CSV or PDF outputs you need.
6) Run **Design Advisor** recommendations or **SOC sensitivity sweeps** when compliance slips.

### What you will see
- Whether the contract is met across the project life.
- How much energy comes from **PV directly** vs. **the BESS**, including shortfall flags.
- End-of-year capability bars, typical daily profiles, and **economics sensitivities**.
- Friendly suggestions from the **Design Advisor** plus SOC-window **sensitivity sweeps**.
- A **scenario table** for saving and comparing different input sets, alongside hourly/monthly/yearly downloads.

### Handy extras
- Use the sidebar link to open the **economics helper** page (LCOE/LCOS) and download the module.
- Turn off the **rate limit** by entering the password in the sidebar (default: `besslab`).
- Window strings accept minutes (e.g., `05:30-09:00`), which are parsed as fractional hours.

### If results look off
- Shortfalls? Try widening the SOC window, improving efficiency, or enabling augmentation.
- Frequent empty battery? Increase duration (MWh), raise the SOC ceiling, or allow more charge time.
- Battery keeps topping out? Lower the SOC ceiling slightly or add a bit more discharge window.
- Unexpected economics results? Confirm price/cost units and rerun the sensitivity heatmaps.

### Helpful notes
- `hour_index` can start at 0 or 1; the app will align it.
- The app uses included defaults when you do not upload files.
- No grid charging is modeled—this is a PV-only pre-feasibility view.
- Sensitivity sweeps will clear when inputs change; rerun them after major adjustments.

**Questions or ideas?** Feedback is welcome to keep improving the tool.
    """
)

st.page_link("pages/01_Simulation.py", label="Jump to Inputs & Results", icon="➡️")
