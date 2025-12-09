import streamlit as st

from app import BASE_DIR
from utils.ui_state import get_shared_data

st.set_page_config(page_title="Home", layout="wide")

st.title("BESSLab guide and tips")

pv_df, cycle_df = get_shared_data(BASE_DIR)
st.caption(
    f"Session data ready: PV profile with {len(pv_df):,} rows and cycle model with {len(cycle_df)} rows."
)

st.markdown(
    """
## Welcome to BESSLab
Use this in-app guide to navigate the PV-only, AC-coupled BESS model and its multipage workflow.

### Prepare your data
- PV profile CSV with columns `hour_index, pv_mw` (0–8759 or 1–8760). The app auto-aligns the starting index.
- Optional cycle-model Excel file to override the built-in degradation table.
- Dispatch windows accept minutes (e.g., `05:30-09:00`), parsed as fractional hours.

### Run the workflow
1) Upload your PV CSV (and optional cycle model) on the Inputs & Results page, or use the bundled samples.
2) Set contracted MW, duration (hours), discharge windows, and any charge windows.
3) Adjust efficiency, SOC limits, availability, augmentation triggers, rate-limit settings, and (optionally) enable the economics helper for LCOE/LCOS, NPV, and IRR.
4) Run the simulation and review compliance, energy splits, end-of-year capability, daily profiles, and flag guidance.
5) Use the physics-bounded Design Advisor and SOC/economics sensitivity sweeps to explore mitigations when targets are missed.
6) Save the latest run to the Scenario comparisons table, download yearly/monthly/hourly CSVs, export the config (JSON), or grab a PDF snapshot.
7) Jump to the BESS sizing sweep page to explore usable-energy variants (holding power fixed) using your cached inputs.

### Multipage navigation
- **Inputs & Results:** Primary simulation, Design Advisor, sensitivities, and downloads.
- **Scenario comparisons:** Label and store snapshots from successive runs for side-by-side review.
- **BESS sizing sweep:** Sweep usable energy, rank feasible designs, and chart LCOE/IRR trends.

### Troubleshooting and tips
- Keep `hour_index` consecutive; the app warns on gaps/out-of-range rows and drops them automatically.
- For persistent shortfalls, widen the SOC window, improve efficiency, or enable augmentation.
- If charge opportunities look tight, widen charge windows or raise the SOC ceiling.
- Sensitivities clear when inputs change—rerun them after major edits.
- PV-only charging is enforced; grid charging is not modeled.
- Disable the session rate limit by entering the password in the sidebar (default: `besslab`).
    """
)

st.page_link("app.py", label="Jump to Inputs & Results")

st.markdown("---")
st.subheader("Navigate across the workspace")
st.page_link("app.py", label="Simulation (Inputs & Results)")
st.page_link("pages/03_Scenario_Comparisons.py", label="Scenario comparisons")
st.page_link("pages/04_BESS_Sizing_Sweep.py", label="BESS sizing sweep")
