from typing import Optional
import streamlit as st

from app import BASE_DIR, SimConfig
from utils import enforce_rate_limit
from utils.sweeps import generate_values, sweep_bess_sizes
from utils.ui_state import get_shared_data

st.set_page_config(page_title="BESS sizing sweep", layout="wide")

st.title("BESS sizing sweep (power × duration)")
st.caption(
    "Grid-search the current inputs across power and duration to highlight feasible designs and KPIs."
)

pv_df, cycle_df = get_shared_data(BASE_DIR)
cfg: Optional[SimConfig] = st.session_state.get("latest_sim_config")
dod_override = st.session_state.get("latest_dod_override", "Auto (infer)")

if cfg is None:
    st.warning(
        "No inputs cached yet. Open the Inputs & Results page, adjust settings, and rerun the simulation to seed the sweep.",
        icon="⚠️",
    )
    cfg = SimConfig()

st.page_link("pages/01_Simulation.py", label="Back to Inputs & Results", help="Update inputs before rerunning the sweep.")

st.markdown("---")

st.session_state.setdefault("bess_size_sweep_results", None)

with st.form("size_sweep_form_page"):
    size_col1, size_col2, size_col3 = st.columns(3)
    with size_col1:
        power_range = st.slider(
            "Power range (MW)",
            min_value=1.0,
            max_value=200.0,
            value=(
                max(1.0, cfg.initial_power_mw * 0.5),
                cfg.initial_power_mw * 1.5,
            ),
            step=1.0,
            help="Lower and upper bounds for the MW grid.",
        )
        power_steps = st.number_input(
            "Power points",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of evenly spaced MW values between the bounds.",
        )

    with size_col2:
        default_duration = max(1.0, cfg.initial_usable_mwh / max(cfg.initial_power_mw, 0.1))
        duration_range = st.slider(
            "Duration range (hours)",
            min_value=0.5,
            max_value=12.0,
            value=(
                max(0.5, default_duration * 0.5),
                min(12.0, default_duration * 1.5),
            ),
            step=0.25,
            help="Lower and upper bounds for duration at rated power.",
        )
        duration_steps = st.number_input(
            "Duration points",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of evenly spaced durations between the bounds.",
        )

    with size_col3:
        ranking_choice = st.selectbox(
            "Rank feasible candidates by",
            options=[
                "compliance_pct",
                "total_shortfall_mwh",
                "total_project_generation_mwh",
                "bess_generation_mwh",
            ],
            format_func=lambda x: {
                "compliance_pct": "Compliance % (higher is better)",
                "total_shortfall_mwh": "Shortfall MWh (lower is better)",
                "total_project_generation_mwh": "Total generation (higher is better)",
                "bess_generation_mwh": "BESS discharge (higher is better)",
            }.get(x, x),
            help="Column used to pick the top feasible design.",
        )
        min_soh = st.number_input(
            "Minimum SOH for feasibility",
            min_value=0.2,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Candidates falling below this total SOH are flagged as infeasible.",
        )

    submitted = st.form_submit_button("Run BESS size sweep", use_container_width=True)

if submitted:
    enforce_rate_limit()
    power_values = generate_values(power_range[0], power_range[1], int(power_steps))
    duration_values = generate_values(duration_range[0], duration_range[1], int(duration_steps))

    with st.spinner("Running BESS size grid..."):
        sweep_df = sweep_bess_sizes(
            base_cfg=cfg,
            pv_df=pv_df,
            cycle_df=cycle_df,
            dod_override=dod_override,
            power_mw_values=power_values,
            duration_h_values=duration_values,
            ranking_kpi=ranking_choice,
            min_soh=min_soh,
            use_case="reliability",
        )

    if sweep_df.empty:
        st.info("No sweep results generated; widen the ranges and try again.")
        st.session_state["bess_size_sweep_results"] = None
    else:
        st.session_state["bess_size_sweep_results"] = sweep_df

sweep_df = st.session_state.get("bess_size_sweep_results")
if sweep_df is not None:
    best_row = sweep_df[sweep_df["is_best"]]
    if best_row.empty:
        st.warning("No feasible candidates met the SOH/cycle thresholds.")
    else:
        best = best_row.iloc[0]
        st.success(
            f"Best feasible: {best['power_mw']:.1f} MW × {best['duration_h']:.2f} h "
            f"({best['energy_mwh']:.1f} MWh)"
        )

    st.dataframe(
        sweep_df[
            [
                "power_mw",
                "duration_h",
                "energy_mwh",
                "compliance_pct",
                "total_shortfall_mwh",
                "avg_eq_cycles_per_year",
                "min_soh_total",
                "feasible",
                "is_best",
            ]
        ],
        use_container_width=True,
    )
else:
    st.info(
        "Run the sweep with your latest inputs. Results persist in the session for quick iteration.",
        icon="ℹ️",
    )

st.markdown("---")
st.subheader("Navigate across the workspace")
st.page_link("pages/00_Home.py", label="Home (Guide)")
st.page_link("pages/01_Simulation.py", label="Simulation (Inputs & Results)")
st.page_link("pages/02_Economics_Module.py", label="Economics helper")
st.page_link("pages/03_Scenario_Comparisons.py", label="Scenario comparisons")
