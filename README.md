# BESSLab — PV-only, AC-coupled (Pre-Feasibility BESS Sizing Tool)

An interactive Streamlit app for exploring **battery energy storage** sizing and behavior when it charges only from **PV** and delivers through an **AC-coupled** point of interconnection.

## What you can do
- Validate whether a chosen **contracted MW × duration** can be met using PV first and the BESS for any shortfall, with minute-level dispatch windows.
- Swap between fixed windows and **dispatch requirement profiles** (hourly MW targets) when modeling full-year contract obligations.
- Explore how performance changes with **round-trip efficiency (single or split charge/discharge), state-of-charge limits, availability, degradation**, and **calendar/cycle fade** assumptions.
- Toggle **augmentation strategies** (threshold, periodic, or explicit manual schedules) to keep discharge capability on target across the project life.
- Use the **Design Advisor** for bounded suggestions when the system misses the target, plus run **sensitivity sweeps** on SOC windows and economics.
- Capture **stress-test sensitivities** with tornado charts to quantify which levers move compliance, deficit, surplus, or IRR the most.
- View clear charts for **end-of-year capability**, **PV vs. BESS energy delivered**, **typical daily profiles**, and **economics sensitivities**.
- Compare multiple runs with the **BESS sizing sweep** and **Multi-scenario batch** tools seeded from cached inputs.
- Download **yearly, monthly, hourly**, **PDF**, and **finance audit** workbooks for sharing, along with sensitivity tables where applicable.
- Use the built-in **economics module** (LCOE/LCOS/NPV/IRR) across pages, including optional **WESM shortfall/surplus pricing**, export auditable cash-flow tables, and adjust contract pricing as needed.
- Reuse uploads across pages and sessions via the landing page cache; bypass the session rate limit with a password when deploying publicly.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the provided local URL in your browser to launch the app. To remove the session rate limit in an open deployment, enter the password in the sidebar (default: `besslab`).

### Installation tips
- Use a virtual environment to isolate dependencies: `python -m venv .venv && source .venv/bin/activate` before installing.
- The bundled sample data lives in `data/` and loads automatically if you skip uploads. The sample PV profile is hourly (8,760 rows); timestamped uploads can be sub-hourly or leap-year length.

## Inputs and file formats
- **PV profile (CSV):** `hour_index, pv_mw` with consecutive hours (0–8759 or 1–8760). Include a `timestamp` column or supply a frequency when loading to preserve sub-hourly cadences and leap years; the app fills missing periods with 0 MW and reuses the one-year profile for multi-year projects.
- **Cycle-model (Excel, optional):** Override the built-in degradation table by uploading your own.
- **Dispatch requirement profile (CSV, optional):** `hour_index, required_mw` to define hourly contracted MW targets across the year (used when dispatch mode is set to “profile” or “hourly schedule”).
- **Contract + dispatch windows:** Specify MW, duration, and window strings. Minutes are accepted and interpreted as fractional hours in the window parser (e.g., `05:30-09:00`).
- **Assumptions:** Configure round-trip efficiency, availability, SOC min/max, augmentation triggers, rate limits, and Design-Advisor bounds.
- **Economics inputs (optional):** Provide CAPEX/OPEX assumptions and the contract energy price, plus an FX rate when working in PHP. Export JSON inputs to reuse economics settings across pages.
- **WESM pricing (CSV, optional):** Hourly `timestamp` or `hour_index` plus `wesm_deficit_price_usd_per_mwh` and/or `wesm_surplus_price_usd_per_mwh` (PHP/kWh columns are accepted and converted using the FX rate). Month/hour average profiles can also be used and will be expanded to hourly values.

If no files are uploaded, the app uses the sample data in `./data/`.

## Units and conventions
- **Power:** MW (nameplate and contracted).
- **Energy:** MWh (BOL, usable, and delivered).
- **Prices and costs:** USD/MWh unless explicitly labeled as PHP; use the FX rate input to convert PHP ↔ USD.
- **Efficiencies and SOC:** Fractions (0–1) in calculations; UI sliders display percent where relevant.
- **Time:** Hourly or sub-hourly inputs supported; timestamped uploads allow leap-year coverage.

## Using the app
1. **Upload or reuse defaults.** Provide a PV 8760 CSV (`hour_index, pv_mw`) and, optionally, a cycle-model Excel file. If you skip uploads, the app uses bundled sample data. The landing page caches uploads by hash so they can be reused across pages or fresh browser sessions until you clear the cache.
2. **Set your target.** Enter the contracted power (MW) and desired duration (hours), plus discharge and optional charge windows. Minutes are preserved and parsed as fractional hours.
3. **Adjust assumptions.** Use sidebar controls for efficiency (single or split), state-of-charge limits, availability, augmentation options (threshold, periodic, or explicit schedules), and calendar/cycle fade. Choose a dispatch mode (fixed windows or hourly/profile schedule) if you have a requirement profile. Enable economics to compute LCOE/LCOS, NPV, and IRR alongside the simulation, configure contract pricing, and optionally include WESM pricing for shortfalls/surplus.
4. **Review results.** Check compliance, flags, end-of-year capability, daily profiles, PV capture, and energy splits between PV and the BESS.
5. **Run sensitivities.** Generate SOC-window sweeps, economics heatmaps, the physics-bounded Design Advisor suggestions when the system misses the target, and stress-test key levers with the tornado chart. Re-run sweeps after changing inputs.
6. **Save & export.** Download yearly/monthly/hourly CSVs, export the simulation inputs (JSON, including economics), grab a PDF snapshot for sharing, and pull the finance audit workbook for traceable cash-flow reviews. Use the sweep, batch, and stress-test tools for structured comparisons.

## App pages and workflows
- **Landing:** Upload PV/cycle files once, warm the cache, and carry those uploads across other pages (or new sessions) without reloading defaults.
- **Inputs & Results (main page):** Run simulations, view KPIs and charts, download CSV/PDF outputs, and trigger SOC/economics sensitivities plus the Design Advisor. Disable the session rate limit with the sidebar password when load-testing deployments.
- **Home (guide):** In-app walkthrough of the multipage workflow with data-format reminders and troubleshooting tips.
- **Sensitivity & stress test:** Capture baseline KPI values, apply upside/downside deltas, and export a tornado chart summary for stakeholder reviews.
- **BESS sizing sweep:** Sweep usable energy (holding power fixed) using the latest inputs, rank feasible candidates by compliance, shortfall, generation, LCOE, or cost metrics, and visualize LCOE/IRR trends.
- **Multi-scenario batch:** Run a structured set of parameter variations and compare the resulting KPIs in one table (preferred for scenario reviews). Start from the seeded templates, tweak availability, SOC windows, augmentation, or economics, and export the table for external review.

## Run a quick BESS sizing sweep from the CLI
The grid-search helper can be exercised without Streamlit using the bundled sample data:

```bash
python -m utils.sweeps
```

The command sweeps a handful of power/duration combinations, prints the KPI table, and flags the best feasible candidate. Use this as a template—adjust the power/duration lists or replace the sample CSV/XLSX with your own inputs.
Legacy entrypoints can still use `python -m bess_size_sweeps` if needed.

## Repository structure
- **app.py**: Main Inputs & Results Streamlit page (simulation, charts, downloads).
- **pages/**: Multipage Streamlit entrypoints (Landing, Home, sweeps, batch, sensitivity).
- **frontend/**: Reusable Streamlit UI components and chart helpers.
- **services/**: Core simulation engine and domain services.
- **utils/**: Shared helpers for I/O, economics, sweeps, flags, and session state.
- **data/**: Bundled sample PV, cycle, and WESM profiles.
- **tests/**: Regression and utility tests for core modules.

## Tips for best results
- Keep `hour_index` consecutive (0–8759 or 1–8760); the app auto-corrects the starting index. Timestamped uploads can be sub-hourly and can include leap-year days—the timestep is inferred automatically.
- If you see frequent shortfalls, try widening the SOC range, improving efficiency, or enabling augmentation.
- Use the Design Advisor panel for quick, bounded suggestions when the system struggles to meet the contract.
- Minutes in window strings (e.g., `05:30-09:00`) are supported and parsed into fractional hours.
- The economics helper and sensitivity heatmaps assume positive cost/price inputs; double-check units before running sweeps.
- WESM pricing requires an hourly price profile (upload or bundled default); disable the toggle if you are only modeling contract pricing.

## Feedback
This tool originated from studies at **Emerging Power Inc. (EPI)**. Feedback and improvement ideas are welcome—please reach out with suggestions or issues.
