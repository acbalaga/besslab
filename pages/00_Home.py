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
- Optional dispatch requirement CSV with columns `hour_index, required_mw` to drive a full-year contract requirement profile.
- Dispatch windows accept minutes (e.g., `05:30-09:00`), parsed as fractional hours, and uploads are cached on the Landing page so you can reuse them across pages or fresh sessions.

### Run the workflow
1) Upload your PV CSV (and optional cycle model) on the Landing page (or directly on Inputs & Results), or use the bundled samples.
2) Set contracted MW, duration (hours), discharge windows, and any charge windows (minutes are preserved).
3) Adjust efficiency (single RTE or split charge/discharge), SOC limits, availability, augmentation triggers (threshold, periodic, or manual schedules), rate-limit settings, and (optionally) enable economics for LCOE/LCOS, NPV, and IRR. Set the contract energy price and optional WESM pricing to reflect your offtake structure.
4) Run the simulation and review compliance, energy splits, end-of-year capability, daily profiles, and flag guidance.
5) Use the physics-bounded Design Advisor and SOC/economics sensitivity sweeps to explore mitigations when targets are missed. Re-run sweeps after input changes to refresh results.
6) Download yearly/monthly/hourly CSVs, export the inputs (JSON, including economics), grab a PDF snapshot, or pull the finance audit workbook for traceable cash-flow reviews. Use the sweep and batch tools to compare scenarios in a structured way.
7) Jump to the BESS sizing sweep page to explore usable-energy variants (holding power fixed) using your cached inputs.

### Units and conventions
- Power: MW (nameplate and contracted).
- Energy: MWh (BOL, usable, and delivered).
- Prices and costs: USD/MWh unless explicitly labeled as PHP; use the FX rate input to convert PHP ↔ USD.
- Efficiencies and SOC: fractions (0–1) in calculations; UI sliders display percent where relevant.
- Time: hourly or sub-hourly inputs supported; timestamped uploads allow leap-year coverage.

### Multipage navigation
- **Inputs & Results:** Primary simulation, Design Advisor, sensitivities, and downloads.
- **BESS sizing sweep:** Sweep usable energy, rank feasible designs, and chart LCOE/IRR trends.
- **Multi-scenario batch:** Run predefined variations and compare results in one table (recommended for scenario reviews).

### Troubleshooting and tips
- Keep `hour_index` consecutive; the app warns on gaps/out-of-range rows and drops them automatically. Timestamped uploads are also supported for sub-hourly or leap-year data.
- For persistent shortfalls, widen the SOC window, improve efficiency, or enable augmentation.
- If charge opportunities look tight, widen charge windows or raise the SOC ceiling.
- Sensitivities clear when inputs change—rerun them after major edits.
- PV-only charging is enforced; grid charging is not modeled.
- Disable the session rate limit by entering the password in the sidebar (default: `besslab`).
    """
)
