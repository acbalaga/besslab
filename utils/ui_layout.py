"""Reusable layout helpers for Streamlit pages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from utils.io import PVProfileSummary, summarize_pv_profile
from utils.ui_state import get_base_dir, get_data_source_status, get_rate_limit_state, get_shared_data

NavRenderer = Callable[[Optional[pd.DataFrame], Optional[pd.DataFrame]], Tuple[pd.DataFrame, pd.DataFrame]]


@dataclass(frozen=True)
class _NavigationLink:
    label: str
    target: str
    help_text: Optional[str] = None


_NAV_LINKS = (
    _NavigationLink("Landing / Uploads", "pages/00_Landing.py", "Seed shared uploads for the session."),
    _NavigationLink("Home (Guide)", "pages/00_Home.py", "Workflow walkthrough, tips, and data-format reminders."),
    _NavigationLink("Inputs & Results", "app.py", "Main simulation workspace and downloads."),
    _NavigationLink(
        "Sensitivity & stress test",
        "pages/06_Sensitivity_Stress_Test.py",
        "Capture sensitivity ranges and view a tornado chart for key KPIs.",
    ),
    _NavigationLink(
        "BESS sizing sweep",
        "pages/04_BESS_Sizing_Sweep.py",
        "Sweep usable energy while holding power constant.",
    ),
    _NavigationLink(
        "Multi-scenario batch",
        "pages/05_Multi_Scenario_Batch.py",
        "Run structured variations and compare KPIs in one table.",
    ),
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

    pv_summary = summarize_pv_profile(pv_df)
    rate_limit_state = get_rate_limit_state()
    rate_limit_bypassed = rate_limit_state.bypass
    recent_runs = len(rate_limit_state.recent_runs)
    rate_limit_state = "Bypassed" if rate_limit_bypassed else "Active"
    rate_limit_detail = (
        "Password accepted for this session."
        if rate_limit_bypassed
        else f"{recent_runs} runs recorded in the last 10 minutes."
    )

    container.markdown("#### Session status")
    container.caption(f"PV rows loaded: {len(pv_df):,}")
    container.caption(f"Cycle rows loaded: {len(cycle_df):,}")
    data_source = get_data_source_status()
    pv_source = data_source.get("pv", "default")
    cycle_source = data_source.get("cycle", "default")
    container.caption(f"Data source: PV ({pv_source}), cycle ({cycle_source}).")
    container.caption(f"Rate limit: {rate_limit_state} ({rate_limit_detail})")

    summary_container = container.expander("PV profile summary", expanded=False)
    _render_pv_summary_table(summary_container, pv_summary)


def _format_pv_summary_value(
    value: Optional[float],
    *,
    unit: Optional[str] = None,
    decimals: int = 3,
) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    suffix = f" {unit}" if unit else ""
    return f"{value:.{decimals}f}{suffix}"


def _format_pv_summary_range(summary: PVProfileSummary) -> str:
    if summary.uses_timestamp:
        if summary.start_timestamp is None or summary.end_timestamp is None:
            return "n/a"
        return f"{summary.start_timestamp.isoformat()} → {summary.end_timestamp.isoformat()}"

    if summary.hour_index_range is None:
        return "n/a"
    start_hour, end_hour = summary.hour_index_range
    return f"{start_hour:,} → {end_hour:,}"


def _build_pv_summary_table(summary: PVProfileSummary) -> pd.DataFrame:
    rows = [
        ("Timestep (hours)", _format_pv_summary_value(summary.timestep_hours)),
        ("Range", _format_pv_summary_range(summary)),
        ("Missing steps (filled)", f"{summary.missing_steps:,}"),
        ("Total steps", f"{summary.total_steps:,}"),
        ("PV min (MW)", _format_pv_summary_value(summary.pv_min_mw, unit="MW")),
        ("PV max (MW)", _format_pv_summary_value(summary.pv_max_mw, unit="MW")),
        ("PV mean (MW)", _format_pv_summary_value(summary.pv_mean_mw, unit="MW")),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


def _render_pv_summary_table(container: DeltaGenerator, summary: PVProfileSummary) -> None:
    container.table(_build_pv_summary_table(summary))


def render_pv_profile_summary(
    container: DeltaGenerator,
    pv_df: pd.DataFrame,
    *,
    title: Optional[str] = None,
) -> None:
    """Render a compact PV profile summary table."""

    summary = summarize_pv_profile(pv_df)
    if title:
        container.markdown(title)
    _render_pv_summary_table(container, summary)


def init_page_layout(
    *,
    page_title: str,
    main_title: str,
    description: Optional[str] = None,
    base_dir: Optional[Path] = None,
    nav_location: Literal["header", "sidebar"] = "header",
) -> NavRenderer:
    """Initialize the page layout with shared navigation and status blocks.

    The helper sets ``st.set_page_config`` immediately, reserves a header slot at
    the top of the page, and returns a renderer that can be called after data
    loading completes. Passing ``pv_df`` and ``cycle_df`` to the renderer avoids
    redundant reads when uploads are handled elsewhere; otherwise shared data is
    fetched from session cache or defaults. ``nav_location`` controls whether the
    navigation links are shown in the header or the sidebar.
    """

    st.set_page_config(page_title=page_title, layout="wide")
    header_container = st.container()
    resolved_base_dir = base_dir or get_base_dir()
    if nav_location == "sidebar":
        with st.sidebar:
            _render_navigation_block(st)

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

            if nav_location == "header":
                nav_col, status_col = st.columns([3, 2])
                _render_navigation_block(nav_col)
                _render_status_block(status_col, shared_pv_df, shared_cycle_df)
            else:
                _render_status_block(st, shared_pv_df, shared_cycle_df)

        st.divider()
        return shared_pv_df, shared_cycle_df

    return _render
