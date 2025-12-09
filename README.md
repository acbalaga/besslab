# BESSLab — PV-only, AC-coupled (Pre-Feasibility BESS Sizing Tool)

An interactive Streamlit app for exploring **battery energy storage** sizing and behavior when it charges only from **PV** and delivers through an **AC-coupled** point of interconnection.

## What you can do
- Validate whether a chosen **contracted MW × duration** can be met using PV first and the BESS for any shortfall.
- Explore how performance changes with **round-trip efficiency, state-of-charge limits, availability, degradation**, and **calendar/cycle fade**.
- Toggle **augmentation strategies** (threshold- or SOH-based) to keep discharge capability on target across the project life.
- Use the **Design Advisor** for bounded suggestions when the system misses the target, plus run **sensitivity sweeps** on SOC windows.
- View clear charts for **end-of-year capability**, **PV vs. BESS energy delivered**, **typical daily profiles**, and **economics sensitivities**.
- Save **scenario snapshots** to compare different input sets side by side.
- Download **yearly, monthly, hourly**, and **PDF** summaries for sharing, along with sensitivity tables where applicable.
- Open the built-in **economics helper** page (LCOE/LCOS) from the sidebar and download the module for offline use.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the provided local URL in your browser to launch the app. To remove the session rate limit in an open deployment, enter the password in the sidebar (default: `besslab`).

### Installation tips
- Use a virtual environment to isolate dependencies: `python -m venv .venv && source .venv/bin/activate` before installing.
- The bundled sample data lives in `data/` and loads automatically if you skip uploads.

## Inputs and file formats
- **PV profile (CSV):** `hour_index, pv_mw` with consecutive hours (0–8759 or 1–8760). The app auto-aligns a 1-based index and drops out-of-range values.
- **Cycle-model (Excel, optional):** Override the built-in degradation table by uploading your own.
- **Contract + dispatch windows:** Specify MW, duration, and window strings. Minutes are accepted and interpreted as fractional hours in the window parser (e.g., `05:30-09:00`).
- **Assumptions:** Configure round-trip efficiency, availability, SOC min/max, augmentation triggers, rate limits, and Design-Advisor bounds.

If no files are uploaded, the app uses the sample data in `./data/`.

## Using the app
1. **Upload or use defaults.** Provide a PV 8760 CSV (`hour_index, pv_mw`) and, optionally, a cycle-model Excel file. If you skip uploads, the app uses included sample data.
2. **Set your target.** Enter the contracted power (MW) and desired duration (hours), plus discharge and optional charge windows.
3. **Adjust assumptions.** Use sidebar controls for efficiency, state-of-charge limits, availability, and augmentation options. Enable the economics helper to compute LCOE/LCOS, NPV, and IRR alongside the simulation.
4. **Review results.** Check compliance, flags, end-of-year capability, daily profiles, and energy split between PV and the BESS.
5. **Run sensitivities.** Generate SOC-window sweeps, economics heatmaps, and the physics-bounded Design Advisor suggestions when the system misses the target.
6. **Save & export.** Capture scenarios to the comparison table, download yearly/monthly/hourly CSVs, export the simulation config (JSON), and grab a PDF snapshot for sharing.

## App pages and workflows
- **Inputs & Results (main page):** Run simulations, view KPIs and charts, download CSV/PDF outputs, and trigger SOC/economics sensitivities plus the Design Advisor.
- **Home (guide):** In-app walkthrough of the multipage workflow with data-format reminders and troubleshooting tips.
- **Scenario comparisons:** Save the latest run as a snapshot, adjust inputs on the main page, and build a table of labeled scenarios for side-by-side review.
- **BESS sizing sweep:** Sweep usable energy (holding power fixed) using the latest inputs, rank feasible candidates by compliance, shortfall, generation, LCOE, or cost metrics, and visualize LCOE/IRR trends.

## Run a quick BESS sizing sweep from the CLI
The grid-search helper can be exercised without Streamlit using the bundled sample data:

```bash
python -m utils.legacy.bess_size_sweeps
```

The command sweeps a handful of power/duration combinations, prints the KPI table, and flags the best feasible candidate. Use this as a template—adjust the power/duration lists or replace the sample CSV/XLSX with your own inputs.

## Tips for best results
- Keep `hour_index` consecutive (0–8759 or 1–8760); the app auto-corrects the starting index.
- If you see frequent shortfalls, try widening the SOC range, improving efficiency, or enabling augmentation.
- Use the Design Advisor panel for quick, bounded suggestions when the system struggles to meet the contract.
- Minutes in window strings (e.g., `05:30-09:00`) are supported and parsed into fractional hours.
- The economics helper and sensitivity heatmaps assume positive cost/price inputs; double-check units before running sweeps.

## Feedback
This tool originated from studies at **Emerging Power Inc. (EPI)**. Feedback and improvement ideas are welcome—please reach out with suggestions or issues.
