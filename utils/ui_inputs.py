"""Shared UI input parsing helpers used across Streamlit pages."""

from __future__ import annotations

from typing import List

import streamlit as st


def parse_numeric_series(label: str, raw_text: str) -> List[float]:
    """Parse comma/newline separated floats for form inputs with consistent errors.

    Centralizes validation so all pages surface the same Streamlit error message
    and exception behavior when non-numeric entries are provided.
    """

    tokens = [token.strip() for token in raw_text.replace(",", "\n").splitlines() if token.strip()]
    series: List[float] = []
    for token in tokens:
        try:
            series.append(float(token))
        except ValueError:
            message = f"{label} contains a non-numeric entry: '{token}'"
            st.error(message)
            raise ValueError(message)
    return series
