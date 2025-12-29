"""Reusable layout helpers for Streamlit pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from utils.ui_state import DATA_SOURCE_SESSION_KEY, get_base_dir, get_shared_data

NavRenderer = Callable[[Optional[pd.DataFrame], Optional[pd.DataFrame]], Tuple[pd.DataFrame, pd.DataFrame]]


@dataclass(frozen=True)
class _NavigationLink:
    label: str
    target: str
    help_text: Optional[str] = None


_NAV_LINKS = (
    _NavigationLink("Landing / Uploads", "pages/00_Landing.py", "Seed shared uploads for the session."),
    _NavigationLink("Home (Guide)", "pages/00_Home.py"),
    _NavigationLink("Inputs & Results", "app.py", "Main simulation workspace."),
    _NavigationLink("BESS sizing sweep", "pages/04_BESS_Sizing_Sweep.py"),
    _NavigationLink("Multi-scenario batch", "pages/05_Multi_Scenario_Batch.py"),
)


def _render_navigation_block(container: DeltaGenerator) -> None:
    """Render standardized navigation links for the workspace."""

    container.markdown("#### Navigate")
    for link in _NAV_LINKS:
        container.page_link(link.target, label=link.label, help=link.help_text)


def _render_status_block(
    container: DeltaGenerator,
    pv_df: pd.DataFrame,
    cycle_df: pd.DataFrame,
) -> None:
    """Show concise session status for shared uploads and the rate limit."""

    rate_limit_bypassed = bool(st.session_state.get("rate_limit_bypass", False))
    recent_runs = len(st.session_state.get("recent_runs", []))
    rate_limit_state = "Bypassed" if rate_limit_bypassed else "Active"
    rate_limit_detail = (
        "Password accepted for this session."
        if rate_limit_bypassed
        else f"{recent_runs} runs recorded in the last 10 minutes."
    )

    container.markdown("#### Session status")
    container.caption(f"PV rows loaded: {len(pv_df):,}")
    container.caption(f"Cycle rows loaded: {len(cycle_df):,}")
    data_source = st.session_state.get(DATA_SOURCE_SESSION_KEY, {})
    pv_source = data_source.get("pv", "default")
    cycle_source = data_source.get("cycle", "default")
    container.caption(f"Data source: PV ({pv_source}), cycle ({cycle_source}).")
    container.caption(f"Rate limit: {rate_limit_state} ({rate_limit_detail})")


def init_page_layout(
    *,
    page_title: str,
    main_title: str,
    description: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> NavRenderer:
    """Initialize the page layout with shared navigation and status blocks.

    The helper sets ``st.set_page_config`` immediately, reserves a header slot at
    the top of the page, and returns a renderer that can be called after data
    loading completes. Passing ``pv_df`` and ``cycle_df`` to the renderer avoids
    redundant reads when uploads are handled elsewhere; otherwise shared data is
    fetched from session cache or defaults.
    """

    st.set_page_config(page_title=page_title, layout="wide")
    header_container = st.container()
    resolved_base_dir = base_dir or get_base_dir()

    def _render(
        pv_df: Optional[pd.DataFrame] = None,
        cycle_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        shared_pv_df, shared_cycle_df = (
            (pv_df, cycle_df) if pv_df is not None and cycle_df is not None else get_shared_data(resolved_base_dir)
        )

        with header_container:
            st.title(main_title)
            if description:
                st.caption(description)

            nav_col, status_col = st.columns([3, 2])
            _render_navigation_block(nav_col)
            _render_status_block(status_col, shared_pv_df, shared_cycle_df)

        st.divider()
        return shared_pv_df, shared_cycle_df

    return _render
