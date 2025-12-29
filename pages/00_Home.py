import streamlit as st

from utils.ui_layout import init_page_layout

render_layout = init_page_layout(
    page_title="Home",
    main_title="BESSLab guide and tips",
    description="Overview of the multipage workflow plus quick navigation links.",
)
render_layout()

st.markdown(
    """
## Welcome to BESSLab
Use this in-app guide to navigate the PV-only, AC-coupled BESS model and its multipage workflow.

### Prepare your data
- PV profile CSV with columns `hour_index, pv_mw` (0–8759 or 1–8760). Add a `timestamp` column to keep sub-hourly cadences or leap-year coverage; the app infers the timestep and fills gaps.
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
- Keep `hour_index` consecutive; the app warns on gaps/out-of-range rows and drops them automatically. Timestamped uploads are also supported for sub-hourly or leap-year data.
- For persistent shortfalls, widen the SOC window, improve efficiency, or enable augmentation.
- If charge opportunities look tight, widen charge windows or raise the SOC ceiling.
- Sensitivities clear when inputs change—rerun them after major edits.
- PV-only charging is enforced; grid charging is not modeled.
- Disable the session rate limit by entering the password in the sidebar (default: `besslab`).
    """
)
