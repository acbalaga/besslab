"""Shared UI helpers for session-scoped data and configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import streamlit as st

from utils import read_cycle_model, read_pv_profile

PV_SESSION_KEY = "shared_pv_profile_df"
CYCLE_SESSION_KEY = "shared_cycle_model_df"


def get_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _set_session_df(key: str, df: pd.DataFrame) -> None:
    st.session_state[key] = df.copy()


def load_shared_data(
    base_dir: Path,
    pv_file,
    cycle_file,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load PV and cycle data from uploads or defaults and cache in the session."""

    default_pv_paths = [str(base_dir / "data" / "PV_8760_MW.csv")]
    default_cycle_paths = [str(base_dir / "data" / "cycle_model.xlsx")]

    pv_df: Optional[pd.DataFrame] = None
    cycle_df: Optional[pd.DataFrame] = None

    if pv_file is not None:
        pv_df = pd.read_csv(pv_file)
        _set_session_df(PV_SESSION_KEY, pv_df)
    elif PV_SESSION_KEY in st.session_state:
        pv_df = st.session_state[PV_SESSION_KEY]
    else:
        pv_df = read_pv_profile(default_pv_paths)
        _set_session_df(PV_SESSION_KEY, pv_df)

    if cycle_file is not None:
        cycle_df = pd.read_excel(cycle_file)
        _set_session_df(CYCLE_SESSION_KEY, cycle_df)
    elif CYCLE_SESSION_KEY in st.session_state:
        cycle_df = st.session_state[CYCLE_SESSION_KEY]
    else:
        cycle_df = read_cycle_model(default_cycle_paths)
        _set_session_df(CYCLE_SESSION_KEY, cycle_df)

    return pv_df, cycle_df


def get_shared_data(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return cached PV and cycle data, falling back to defaults when absent."""

    default_pv_paths = [str(base_dir / "data" / "PV_8760_MW.csv")]
    default_cycle_paths = [str(base_dir / "data" / "cycle_model.xlsx")]

    pv_df = st.session_state.get(PV_SESSION_KEY)
    cycle_df = st.session_state.get(CYCLE_SESSION_KEY)

    if pv_df is None:
        pv_df = read_pv_profile(default_pv_paths)
        _set_session_df(PV_SESSION_KEY, pv_df)
    if cycle_df is None:
        cycle_df = read_cycle_model(default_cycle_paths)
        _set_session_df(CYCLE_SESSION_KEY, cycle_df)

    return pv_df, cycle_df


def hide_root_page_from_sidebar() -> None:
    """Hide the launcher script entry from Streamlit's sidebar navigation."""

    st.markdown(
        """
        <style>
        /* Fall back to simply hiding the first nav item. */
        [data-testid="stSidebarNav"] li:first-child { display: none; }

        /* Also hide any link that points to the root app script. */
        [data-testid="stSidebarNav"] a[href="/"] { display: none !important; }
        [data-testid="stSidebarNav"] li a[href="/"] { display: none !important; }
        </style>
        <script>
        // Remove the launcher link if Streamlit renders it with a root href.
        const nav = window.parent?.document?.querySelector('[data-testid="stSidebarNav"]');
        if (nav) {
            const rootLink = nav.querySelector('a[href="/"]');
            const rootItem = rootLink?.closest('li');
            if (rootItem) rootItem.style.display = 'none';
        }
        </script>
        """,
        unsafe_allow_html=True,
    )
