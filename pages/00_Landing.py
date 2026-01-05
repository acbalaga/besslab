import streamlit as st

from app import BASE_DIR
from utils.ui_layout import init_page_layout
from utils.ui_state import bootstrap_session_state, load_shared_data

bootstrap_session_state()

render_layout = init_page_layout(
    page_title="Landing",
    main_title="BESSLab landing (uploads & session cache)",
    description=(
        "Upload PV and (optionally) a cycle model once, then navigate the workspace with shared session data and cached uploads."
    ),
    base_dir=BASE_DIR,
)

with st.expander("Upload your data", expanded=True):
    st.caption(
        "Session uploads are reused across pages and fresh sessions when the file hash matches. Skip the upload to use bundled samples from ./data/."
    )
    upload_cols = st.columns(2)
    with upload_cols[0]:
        pv_file = st.file_uploader(
            "PV 8760 CSV (hour_index, pv_mw in MW)",
            type=["csv"],
            key="landing_pv_upload",
            help="Include a timestamp column to preserve sub-hourly cadence or leap-year coverage.",
        )
    with upload_cols[1]:
        cycle_file = st.file_uploader(
            "Cycle model Excel (optional override)",
            type=["xlsx"],
            key="landing_cycle_upload",
            help="Overrides the built-in degradation table when provided.",
        )

pv_df, cycle_df = load_shared_data(BASE_DIR, pv_file, cycle_file)
pv_df, cycle_df = render_layout(pv_df, cycle_df)

st.success(
    f"PV profile loaded with {len(pv_df):,} rows; cycle model contains {len(cycle_df)} rows."
)
st.caption(
    "Uploads are cached across pages (and reused on a fresh session when hashes match); "
    "select 'Clear cache' from the Streamlit menu or remove uploads to revert to bundled defaults."
)

st.markdown("### Next steps")
st.page_link("app.py", label="Go to Inputs & Results", help="Configure the simulation and run it.")
st.page_link(
    "pages/00_Home.py",
    label="Open the in-app guide",
    help="Review workflow tips before running scenarios.",
)
