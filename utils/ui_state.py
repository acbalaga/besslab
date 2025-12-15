"""Shared UI helpers for session-scoped data and configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from utils import read_cycle_model, read_pv_profile

PV_SESSION_KEY = "shared_pv_profile_df"
CYCLE_SESSION_KEY = "shared_cycle_model_df"
ECON_PAYLOAD_KEY = "latest_economics_payload"


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
        pv_df = read_pv_profile([pv_file])
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


def cache_latest_economics_payload(payload: Dict[str, Any]) -> None:
    """Store the latest economics-ready payload for reuse across pages.

    The Simulation page seeds this with energy series and price/economic
    assumptions so other pages (e.g., the standalone Economics helper) can
    rehydrate the same inputs without rerunning the dispatch model.
    """

    st.session_state[ECON_PAYLOAD_KEY] = payload


def get_latest_economics_payload() -> Optional[Dict[str, Any]]:
    """Return the most recent economics payload cached in the session."""

    return st.session_state.get(ECON_PAYLOAD_KEY)


