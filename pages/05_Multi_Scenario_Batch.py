from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from app import (
    BASE_DIR,
    SimConfig,
    Window,
    parse_windows,
    simulate_project,
    summarize_simulation,
)
from utils import enforce_rate_limit
from utils.economics import compute_lcoe_lcos_with_augmentation_fallback
from utils.ui_state import get_shared_data

st.set_page_config(page_title="Multi-scenario batch", layout="wide")

st.title("Multi-scenario batch runner")
st.caption(
    "Queue multiple simulation variants at once. Per-scenario logs are disabled to conserve memory; "
    "rerun the main page for detailed charts."
)


def _format_hhmm(hour_value: float) -> str:
    """Return HH:MM text for a fractional hour."""

    hours = int(hour_value)
    minutes = int(round((hour_value - hours) * 60))
    if minutes == 60:
        hours = (hours + 1) % 24
        minutes = 0
    return f"{hours:02d}:{minutes:02d}"


def _windows_to_text(windows: List[Window]) -> str:
    """Serialize Window objects to HH:MM-HH:MM strings."""

    return ", ".join(f"{_format_hhmm(w.start)}-{_format_hhmm(w.end)}" for w in windows)


def _seed_rows(cfg: SimConfig) -> pd.DataFrame:
    """Construct the initial scenario table from cached inputs or defaults."""

    defaults: Dict[str, Any] = {
        "label": "Scenario 1",
        "initial_power_mw": cfg.initial_power_mw,
        "initial_usable_mwh": cfg.initial_usable_mwh,
        "contracted_mw": cfg.contracted_mw,
        "years": cfg.years,
        "pv_degradation_rate": cfg.pv_deg_rate,
        "bess_availability": cfg.bess_availability,
        "rte": cfg.rte_roundtrip,
        "soc_floor": cfg.soc_floor,
        "soc_ceiling": cfg.soc_ceiling,
        "discharge_windows": _windows_to_text(cfg.discharge_windows),
        "charge_windows": cfg.charge_windows_text or "",
    }
    return pd.DataFrame([defaults])


def _parse_row_to_config(row: pd.Series, template: SimConfig) -> Tuple[str, SimConfig]:
    """Apply row overrides to a SimConfig copy, validating fields along the way."""

    label = str(row.get("label") or "Scenario")
    config = deepcopy(template)

    dis_windows_text = str(row.get("discharge_windows") or "").strip()
    dis_windows = parse_windows(dis_windows_text)
    if not dis_windows:
        raise ValueError("Provide at least one discharge window (HH:MM-HH:MM).")

    charge_windows_text = str(row.get("charge_windows") or template.charge_windows_text or "").strip()

    years = int(row.get("years") or template.years)
    soc_floor = float(row.get("soc_floor") or template.soc_floor)
    soc_ceiling = float(row.get("soc_ceiling") or template.soc_ceiling)
    if soc_floor >= soc_ceiling:
        raise ValueError("SOC floor must be lower than the SOC ceiling.")

    config.years = max(1, years)
    config.initial_power_mw = float(row.get("initial_power_mw") or template.initial_power_mw)
    config.initial_usable_mwh = float(row.get("initial_usable_mwh") or template.initial_usable_mwh)
    config.contracted_mw = float(row.get("contracted_mw") or template.contracted_mw)
    config.pv_deg_rate = float(row.get("pv_degradation_rate") or template.pv_deg_rate)
    config.bess_availability = float(row.get("bess_availability") or template.bess_availability)
    config.rte_roundtrip = float(row.get("rte") or template.rte_roundtrip)
    config.soc_floor = soc_floor
    config.soc_ceiling = soc_ceiling
    config.discharge_windows = dis_windows
    config.charge_windows_text = charge_windows_text

    return label, config


pv_df, cycle_df = get_shared_data(BASE_DIR)
cached_cfg: SimConfig = st.session_state.get("latest_sim_config", SimConfig())
dod_override = st.session_state.get("latest_dod_override", "Auto (infer)")

st.page_link("app.py", label="Back to Inputs & Results", help="Tune inputs before batching scenarios.")
st.page_link("pages/03_Scenario_Comparisons.py", label="Scenario comparisons table")
st.page_link("pages/04_BESS_Sizing_Sweep.py", label="BESS sizing sweep")
st.page_link("pages/00_Home.py", label="Home (Guide)")
st.markdown("---")

table_placeholder = st.empty()
default_rows = _seed_rows(cached_cfg)
edited_df = table_placeholder.data_editor(
    default_rows,
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "label": st.column_config.TextColumn("Label", help="Identifier for this scenario."),
        "initial_power_mw": st.column_config.NumberColumn(
            "Power (MW)", min_value=0.1, max_value=500.0, step=0.1, help="Initial discharge rating."
        ),
        "initial_usable_mwh": st.column_config.NumberColumn(
            "Usable energy (MWh)", min_value=1.0, max_value=1_000.0, step=1.0
        ),
        "contracted_mw": st.column_config.NumberColumn(
            "Contracted MW", min_value=0.1, max_value=500.0, step=0.1, help="Firm delivery target."
        ),
        "years": st.column_config.NumberColumn(
            "Years", min_value=1, max_value=40, step=1, help="Project horizon in years."
        ),
        "pv_degradation_rate": st.column_config.NumberColumn(
            "PV degradation (frac/yr)", min_value=0.0, max_value=0.2, step=0.001
        ),
        "bess_availability": st.column_config.NumberColumn(
            "BESS availability", min_value=0.5, max_value=1.0, step=0.01
        ),
        "rte": st.column_config.NumberColumn(
            "Round-trip η", min_value=0.5, max_value=0.99, step=0.01, help="Round-trip efficiency (0–1)."
        ),
        "soc_floor": st.column_config.NumberColumn(
            "SOC floor", min_value=0.0, max_value=0.95, step=0.01, help="Minimum SOC as a fraction."
        ),
        "soc_ceiling": st.column_config.NumberColumn(
            "SOC ceiling", min_value=0.05, max_value=1.0, step=0.01, help="Maximum SOC as a fraction."
        ),
        "discharge_windows": st.column_config.TextColumn(
            "Discharge windows",
            help="Comma-separated HH:MM-HH:MM ranges (e.g., 10:00-14:00, 18:00-22:00).",
        ),
        "charge_windows": st.column_config.TextColumn(
            "Charge windows (optional)",
            help="Leave blank to allow any PV hour; uses the same HH:MM-HH:MM format.",
        ),
    },
    key="multi_scenario_table",
)

st.caption("Tip: Add rows for each design tweak. Remove rows to pare down the batch run.")

run_container = st.container()
results_container = st.container()

if "multi_scenario_batch_results" in st.session_state:
    st.success("Showing the latest batch results cached in this session.")


def _run_batch() -> pd.DataFrame | None:
    """Run the configured scenarios and return a results DataFrame."""

    if edited_df is None or edited_df.empty:
        st.warning("Add at least one scenario before running.", icon="⚠️")
        return None

    econ_payload = st.session_state.get("latest_economics_payload")
    scenarios: List[Tuple[str, SimConfig]] = []
    errors: List[str] = []
    for idx, row in edited_df.reset_index(drop=True).iterrows():
        try:
            scenarios.append(_parse_row_to_config(row, cached_cfg))
        except ValueError as exc:  # noqa: BLE001
            errors.append(f"Row {idx + 1}: {exc}")

    if errors:
        st.error("Please fix the highlighted rows before running.")
        for msg in errors:
            st.caption(f"• {msg}")
        return None

    enforce_rate_limit()
    progress = st.progress(0.0, text="Starting batch...")
    results: List[Dict[str, Any]] = []

    for idx, (label, cfg) in enumerate(scenarios, start=1):
        progress.progress((idx - 1) / len(scenarios), text=f"Running {label}...")
        try:
            sim_output = simulate_project(
                cfg, pv_df=pv_df, cycle_df=cycle_df, dod_override=dod_override, need_logs=False
            )
        except ValueError as exc:  # noqa: BLE001
            progress.empty()
            st.error(f"{label}: {exc}")
            return None

        summary = summarize_simulation(sim_output)
        final_year = sim_output.results[-1]
        economics_fields: Dict[str, Any] = {}
        if isinstance(econ_payload, dict) and econ_payload.get("economic_inputs"):
            try:
                econ_outputs = compute_lcoe_lcos_with_augmentation_fallback(
                    annual_delivered_mwh=[r.delivered_firm_mwh for r in sim_output.results],
                    annual_bess_mwh=[r.bess_to_contract_mwh for r in sim_output.results],
                    inputs=econ_payload["economic_inputs"],
                    augmentation_costs_usd=econ_payload.get("augmentation_costs_usd"),
                )
                economics_fields["LCOE ($/MWh)"] = econ_outputs.lcoe_usd_per_mwh
                economics_fields["LCOS ($/MWh)"] = econ_outputs.lcos_usd_per_mwh
            except ValueError:
                economics_fields["LCOE ($/MWh)"] = float("nan")
                economics_fields["LCOS ($/MWh)"] = float("nan")

        results.append(
            {
                "Label": label,
                "Years": cfg.years,
                "Power (MW)": cfg.initial_power_mw,
                "Usable MWh": cfg.initial_usable_mwh,
                "Contracted MW": cfg.contracted_mw,
                "Compliance (%)": summary.compliance,
                "BESS share of firm (%)": summary.bess_share_of_firm,
                "Charge/Discharge ratio": summary.charge_discharge_ratio,
                "PV capture ratio": summary.pv_capture_ratio,
                "Shortfall MWh": summary.total_shortfall_mwh,
                "Avg eq cycles/yr": summary.avg_eq_cycles_per_year,
                "Final SOH_total": final_year.soh_total,
                "EOY usable MWh": final_year.eoy_usable_mwh,
                "EOY power MW": final_year.eoy_power_mw,
                "Augmentations": sim_output.augmentation_events,
                **economics_fields,
            },
        )
        progress.progress(idx / len(scenarios), text=f"Finished {label} ({idx}/{len(scenarios)})")

    progress.progress(1.0, text="Batch complete.")
    st.balloons()
    return pd.DataFrame(results)


with run_container:
    if st.button("Run scenarios", use_container_width=True):
        with st.spinner("Running batch scenarios..."):
            batch_df = _run_batch()
        if batch_df is not None:
            st.session_state["multi_scenario_batch_results"] = batch_df


with results_container:
    batch_results: pd.DataFrame | None = st.session_state.get("multi_scenario_batch_results")
    if batch_results is not None and not batch_results.empty:
        formatted = batch_results.style.format(
            {
                "Compliance (%)": "{:,.2f}",
                "BESS share of firm (%)": "{:,.2f}",
                "Charge/Discharge ratio": "{:,.3f}",
                "PV capture ratio": "{:,.3f}",
                "Shortfall MWh": "{:,.1f}",
                "Avg eq cycles/yr": "{:,.2f}",
                "Final SOH_total": "{:,.3f}",
                "EOY usable MWh": "{:,.1f}",
                "EOY power MW": "{:,.2f}",
                "LCOE ($/MWh)": "{:,.2f}",
                "LCOS ($/MWh)": "{:,.2f}",
            }
        )
        st.dataframe(formatted, use_container_width=True, hide_index=True)

        csv_bytes = batch_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download results as CSV",
            data=csv_bytes,
            file_name="multi_scenario_batch_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.session_state.get("latest_economics_payload"):
            st.caption("Economic columns populate when economics inputs are cached from the main page.")
        else:
            st.caption("LCOE/LCOS will populate once economics assumptions have been cached on the main page.")
    else:
        st.info("No batch results yet. Add rows above and click Run scenarios.", icon="ℹ️")

st.caption(
    "Scenarios inherit other settings from the latest cached configuration. Logs are off for each run "
    "to keep memory within typical 4GB limits."
)
