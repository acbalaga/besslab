import pandas as pd
import streamlit as st

from utils.ui_state import hide_root_page_from_sidebar

st.set_page_config(page_title="Scenario comparisons", layout="wide")

hide_root_page_from_sidebar()

st.title("Scenario comparisons")
st.caption(
    "Save the latest simulation snapshot, then adjust inputs on the main page to build a table of"
    " configurations and KPIs."
)

if "scenario_comparisons" not in st.session_state:
    st.session_state["scenario_comparisons"] = []

latest_snapshot = st.session_state.get("latest_simulation_snapshot")
default_label = f"Scenario {len(st.session_state['scenario_comparisons']) + 1}"
scenario_label = st.text_input(
    "Label for this scenario",
    value=default_label,
    key="scenario_label_input_page",
    help="Applies to the next snapshot saved to the table.",
)

cols = st.columns([3, 2])
with cols[0]:
    if latest_snapshot is None:
        st.info(
            "Run the simulation from the Inputs & Results page to populate the latest snapshot.",
            icon="ℹ️",
        )
    else:
        st.success("Latest run is ready to save.")

    if st.button("Add current scenario to table", disabled=latest_snapshot is None, use_container_width=True):
        scenario_entry = {"Label": scenario_label or default_label, **latest_snapshot}
        st.session_state["scenario_comparisons"].append(scenario_entry)
        st.success("Scenario saved. Adjust inputs on the main page and add another to compare.")

with cols[1]:
    st.page_link(
        "app.py",
        label="Back to Inputs & Results",
        help="Adjust inputs and rerun the simulation to refresh the snapshot.",
    )
    st.page_link(
        "pages/04_BESS_Sizing_Sweep.py",
        label="Go to BESS sizing sweep",
        help="Explore power × duration grids using the current inputs.",
    )

st.markdown("---")

if latest_snapshot:
    st.markdown("**Latest snapshot (read-only)**")
    snapshot_df = pd.DataFrame([latest_snapshot]).T.reset_index()
    snapshot_df.columns = ["Metric", "Value"]
    st.dataframe(snapshot_df, hide_index=True, use_container_width=True)

if st.session_state["scenario_comparisons"]:
    compare_df = pd.DataFrame(st.session_state["scenario_comparisons"])
    st.markdown("**Saved scenarios**")
    st.dataframe(
        compare_df.style.format(
            {
                "Compliance (%)": "{:,.2f}",
                "BESS share of firm (%)": "{:,.1f}",
                "Charge/Discharge ratio": "{:,.3f}",
                "PV capture ratio": "{:,.3f}",
                "Total project generation (MWh)": "{:,.1f}",
                "BESS share of generation (MWh)": "{:,.1f}",
                "PV share of generation (MWh)": "{:,.1f}",
                "PV excess (MWh)": "{:,.1f}",
                "BESS losses (MWh)": "{:,.1f}",
                "Final EOY usable (MWh)": "{:,.1f}",
                "Final EOY power (MW)": "{:,.2f}",
                "Final eq cycles (year)": "{:,.1f}",
                "Final SOH_total": "{:,.3f}",
            }
        ),
        use_container_width=True,
    )
    if st.button("Clear saved scenarios", type="secondary", use_container_width=True):
        st.session_state["scenario_comparisons"] = []
        st.info("Cleared. Rerun the simulation to capture a fresh snapshot.")
else:
    st.info(
        "No saved scenarios yet. Run the main simulation, then click 'Add current scenario to table'.",
        icon="ℹ️",
    )

st.markdown("---")
st.subheader("Navigate across the workspace")
st.page_link("pages/00_Home.py", label="Home (Guide)")
st.page_link("app.py", label="Simulation (Inputs & Results)")
st.page_link("pages/04_BESS_Sizing_Sweep.py", label="BESS sizing sweep")
