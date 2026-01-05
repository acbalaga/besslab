"""Shared rendering helpers for Streamlit pages."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

Formatter = Union[str, Callable[[Any], str]]


@dataclass(frozen=True)
class MetricSpec:
    """Specification for a Streamlit metric card."""

    label: str
    value: str
    help: Optional[str] = None
    caption: Optional[str] = None


def render_metrics(columns: Sequence[DeltaGenerator], specs: Sequence[MetricSpec]) -> None:
    """Render metric cards from specs to keep layout and captions consistent."""

    for col, spec in zip(columns, specs):
        col.metric(spec.label, spec.value, help=spec.help)
        if spec.caption:
            col.caption(spec.caption)


def render_formatted_dataframe(
    df: pd.DataFrame,
    formatters: Mapping[str, Formatter],
    *,
    use_container_width: bool = True,
    **dataframe_kwargs: Any,
) -> None:
    """Render a dataframe with shared number formatting to avoid repeated style blocks."""

    st.dataframe(
        df.style.format(formatters),
        use_container_width=use_container_width,
        **dataframe_kwargs,
    )
