# BESSLab — PV-only, AC-coupled (Pre-Feasibility BESS Sizing Tool)

An interactive Streamlit app for exploring **battery energy storage** sizing and behavior when it charges only from **PV** and delivers through an **AC-coupled** point of interconnection.

## What you can do
- Validate whether a chosen **contracted MW × duration** can be met using PV first and the BESS for any shortfall, with minute-level dispatch windows.
- Explore how performance changes with **round-trip efficiency (single or split charge/discharge), state-of-charge limits, availability, degradation**, and **calendar/cycle fade** assumptions.
- Toggle **augmentation strategies** (threshold, periodic, or explicit manual schedules) to keep discharge capability on target across the project life.
- Use the **Design Advisor** for bounded suggestions when the system misses the target, plus run **sensitivity sweeps** on SOC windows and economics.
- View clear charts for **end-of-year capability**, **PV vs. BESS energy delivered**, **typical daily profiles**, and **economics sensitivities**.
- Compare multiple runs with the **BESS sizing sweep** and **Multi-scenario batch** tools seeded from cached inputs.
- Download **yearly, monthly, hourly**, and **PDF** summaries for sharing, along with sensitivity tables where applicable.
- Open the built-in **economics helper** page (LCOE/LCOS) from the sidebar, download the standalone module for offline use, and override energy prices with a blended rate when needed.
- Reuse uploads across pages and sessions via the landing page cache; bypass the session rate limit with a password when deploying publicly.

## Quick start
```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the provided local URL in your browser to launch the app. To remove the session rate limit in an open deployment, enter the password in the sidebar (default: `besslab`).

## Web front-end (React/Vite SPA)
A lightweight React SPA lives in `frontend/` and mirrors the Streamlit flows (uploads, single simulation, sizing sweep, and batch runs) while consuming the FastAPI endpoints.

1. Install dependencies:
   ```bash
   cd frontend
   npm install
   ```
2. Create a `.env.local` (or `.env`) with your API base URL:
   ```
   VITE_API_BASE_URL=http://localhost:8000
   ```
3. Start the dev server and open the printed localhost URL:
   ```bash
   npm run dev
   ```

The SPA posts uploads to `/uploads`, passes the returned tokens into `/simulate`, `/sweep`, and `/batch`, and renders KPIs/charts for compliance, coverage, round-trip efficiency, SOH, augmentation events, and economics flags. Use the cached upload tokens from the landing page to reuse PV/cycle files across routes.

### Installation tips
- Use a virtual environment to isolate dependencies: `python -m venv .venv && source .venv/bin/activate` before installing.
- The bundled sample data lives in `data/` and loads automatically if you skip uploads. The sample PV profile is hourly (8,760 rows); timestamped uploads can be sub-hourly or leap-year length.

## Inputs and file formats
- **PV profile (CSV):** `hour_index, pv_mw` with consecutive hours (0–8759 or 1–8760). Include a `timestamp` column or supply a frequency when loading to preserve sub-hourly cadences and leap years; the app fills missing periods with 0 MW and reuses the one-year profile for multi-year projects.
- **Cycle-model (Excel, optional):** Override the built-in degradation table by uploading your own.
- **Contract + dispatch windows:** Specify MW, duration, and window strings. Minutes are accepted and interpreted as fractional hours in the window parser (e.g., `05:30-09:00`).
- **Assumptions:** Configure round-trip efficiency, availability, SOC min/max, augmentation triggers, rate limits, and Design-Advisor bounds.

If no files are uploaded, the app uses the sample data in `./data/`.

## Using the app
1. **Upload or reuse defaults.** Provide a PV 8760 CSV (`hour_index, pv_mw`) and, optionally, a cycle-model Excel file. If you skip uploads, the app uses bundled sample data. The landing page caches uploads by hash so they can be reused across pages or fresh browser sessions until you clear the cache.
2. **Set your target.** Enter the contracted power (MW) and desired duration (hours), plus discharge and optional charge windows. Minutes are preserved and parsed as fractional hours.
3. **Adjust assumptions.** Use sidebar controls for efficiency (single or split), state-of-charge limits, availability, augmentation options (threshold, periodic, or explicit schedules), and calendar/cycle fade. Enable the economics helper to compute LCOE/LCOS, NPV, and IRR alongside the simulation, and override prices with a single blended energy rate if desired.
4. **Review results.** Check compliance, flags, end-of-year capability, daily profiles, PV capture, and energy splits between PV and the BESS.
5. **Run sensitivities.** Generate SOC-window sweeps, economics heatmaps, and the physics-bounded Design Advisor suggestions when the system misses the target. Re-run sweeps after changing inputs.
6. **Save & export.** Download yearly/monthly/hourly CSVs, export the simulation config (JSON), and grab a PDF snapshot for sharing. Use the sweep and batch tools for structured comparisons.

## App pages and workflows
- **Landing:** Upload PV/cycle files once, warm the cache, and carry those uploads across other pages (or new sessions) without reloading defaults.
- **Inputs & Results (main page):** Run simulations, view KPIs and charts, download CSV/PDF outputs, and trigger SOC/economics sensitivities plus the Design Advisor. Disable the session rate limit with the sidebar password when load-testing deployments.
- **Home (guide):** In-app walkthrough of the multipage workflow with data-format reminders and troubleshooting tips.
- **BESS sizing sweep:** Sweep usable energy (holding power fixed) using the latest inputs, rank feasible candidates by compliance, shortfall, generation, LCOE, or cost metrics, and visualize LCOE/IRR trends.
- **Multi-scenario batch:** Run a structured set of parameter variations and compare the resulting KPIs in one table (preferred for scenario reviews). Start from the seeded templates, tweak availability, SOC windows, augmentation, or economics, and export the table for external review.

## Run a quick BESS sizing sweep from the CLI
The grid-search helper can be exercised without Streamlit using the bundled sample data:

```bash
python -m utils.legacy.bess_size_sweeps
```

The command sweeps a handful of power/duration combinations, prints the KPI table, and flags the best feasible candidate. Use this as a template—adjust the power/duration lists or replace the sample CSV/XLSX with your own inputs.

## Lightweight API (FastAPI)
Run the lightweight REST API outside Streamlit to reuse the simulation core:

```bash
uvicorn api.server:app --host 0.0.0.0 --port 8000
```

- Health check: `curl http://localhost:8000/health`
- Run a simulation with bundled sample data:

  ```bash
  curl -X POST http://localhost:8000/simulate \
    -H "Content-Type: application/json" \
    -d '{"config":{"contracted_mw":20,"discharge_windows_text":"10:00-14:00,18:00-22:00"},"data":{"use_sample_pv":true,"use_sample_cycle":true},"dod_override":"Auto (infer)"}'
  ```

- Run a small sweep over two BESS sizes:

  ```bash
  curl -X POST http://localhost:8000/sweep \
    -H "Content-Type: application/json" \
    -d '{"config":{"contracted_mw":20,"discharge_windows_text":"10:00-14:00,18:00-22:00"},"data":{"use_sample_pv":true,"use_sample_cycle":true},"power_values":[20,30],"duration_values":[2,4]}'
  ```

Use `/uploads` to cache JSON PV or cycle tables across requests, and `/batch` to run multiple configurations against shared inputs.

## Tips for best results
- Keep `hour_index` consecutive (0–8759 or 1–8760); the app auto-corrects the starting index. Timestamped uploads can be sub-hourly and can include leap-year days—the timestep is inferred automatically.
- If you see frequent shortfalls, try widening the SOC range, improving efficiency, or enabling augmentation.
- Use the Design Advisor panel for quick, bounded suggestions when the system struggles to meet the contract.
- Minutes in window strings (e.g., `05:30-09:00`) are supported and parsed into fractional hours.
- The economics helper and sensitivity heatmaps assume positive cost/price inputs; double-check units before running sweeps.

## Feedback
This tool originated from studies at **Emerging Power Inc. (EPI)**. Feedback and improvement ideas are welcome—please reach out with suggestions or issues.
