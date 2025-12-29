import pandas as pd
import streamlit as st

from utils.ui_layout import init_page_layout

render_layout = init_page_layout(
    page_title="Scenario comparisons (deprecated)",
    main_title="Scenario comparisons (deprecated)",
    description=(
        "Snapshot saving is no longer supported. Use batch tools or exports from Inputs & Results to "
        "review multiple runs side by side."
    ),
)
render_layout()

st.warning(
    "This page is deprecated and will be removed in a future release. "
    "Use the Multi-scenario batch page for structured sweeps or download CSV/PDF results from "
    "Inputs & Results for offline comparisons.",
    icon="⚠️",
)

if st.session_state.get("scenario_comparisons"):
    compare_df = pd.DataFrame(st.session_state["scenario_comparisons"])
    st.markdown("**Read-only snapshots from this session**")
    st.caption(
        "Capturing new snapshots is disabled. Export results or rerun the batch tools to rebuild comparisons."
    )
    st.dataframe(compare_df, use_container_width=True)
else:
    st.info(
        "Snapshot capture has been retired. Run the Multi-scenario batch page or export results from Inputs & Results "
        "to compare scenarios.",
        icon="ℹ️",
    )
