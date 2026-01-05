"""Rate limit helpers for Streamlit sessions."""

from __future__ import annotations

import os
import time

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from utils.ui_state import get_rate_limit_state, update_rate_limit_state


def enforce_rate_limit(
    max_runs: int = 60,
    window_seconds: int = 600,
    min_spacing_seconds: float = 2.0,
) -> None:
    """Simple session-based rate limit to deter abuse on open deployments.

    The `min_spacing_seconds` guard prevents multiple Streamlit reruns triggered by
    a single UI action (e.g., widget update + state change) from being counted as
    separate runs, which otherwise exhausts the allowance during small batches.
    """
    state = get_rate_limit_state()
    if state.bypass:
        return

    now = time.time()
    recent = list(state.recent_runs)
    recent = [t for t in recent if now - t < window_seconds]
    if len(recent) >= max_runs:
        wait_for = int(window_seconds - (now - min(recent)))
        st.error(
            "Rate limit reached. Please wait a few minutes before running more calculations."
        )
        st.info(
            f"You can retry in approximately {max(wait_for, 1)} seconds."
        )
        st.stop()

    last_recorded = state.last_rate_limit_ts
    if last_recorded is None or now - last_recorded >= min_spacing_seconds:
        recent.append(now)
        state.last_rate_limit_ts = now

    state.recent_runs = recent
    update_rate_limit_state(state)


def get_rate_limit_password() -> str:
    """Return the password used to disable rate limiting.

    The lookup order is Streamlit secrets → environment variable → built-in default.
    """

    try:
        secret_password = st.secrets.get("rate_limit_password")
    except StreamlitSecretNotFoundError:
        secret_password = None

    return secret_password or os.environ.get("BESSLAB_RATE_LIMIT_PASSWORD") or "besslab"
