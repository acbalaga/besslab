"""Shared UI helpers for session-scoped data and configuration."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st

from utils import read_cycle_model, read_pv_profile

PV_SESSION_KEY = "shared_pv_profile_df"
CYCLE_SESSION_KEY = "shared_cycle_model_df"
ECON_PAYLOAD_KEY = "latest_economics_payload"
PV_HASH_SESSION_KEY = "shared_pv_profile_hash"
CYCLE_HASH_SESSION_KEY = "shared_cycle_model_hash"
DATA_SOURCE_SESSION_KEY = "shared_data_source_status"


def get_base_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _set_session_df(key: str, df: pd.DataFrame) -> None:
    st.session_state[key] = df.copy()


@st.cache_resource
def _shared_upload_cache() -> Dict[str, Dict[str, Optional[bytes]]]:
    """Persist upload payloads across Streamlit sessions.

    The cache enables a fresh browser session to rehydrate the most recent
    uploads without re-reading defaults when a hash match exists.
    """

    return {"pv": {"hash": None, "content": None}, "cycle": {"hash": None, "content": None}}


@st.cache_data(show_spinner=False)
def _cache_pv_upload(file_hash: str, content: bytes) -> pd.DataFrame:
    """Parse and cache PV profiles keyed by upload hash.

    PV profiles carry hour_index (dimensionless) and ``pv_mw`` in MW. Caching
    avoids repeatedly decoding file-like objects on reruns and across pages.
    """

    return read_pv_profile([io.BytesIO(content)])


@st.cache_data(show_spinner=False)
def _cache_cycle_upload(file_hash: str, content: bytes) -> pd.DataFrame:
    """Parse and cache cycle models keyed by upload hash.

    Cycle model tables use DoD*_Cycles (cycles) and DoD*_Ret(%) (% retained).
    Cached parsing keeps upload reuse cheap when navigating between pages.
    """

    return read_cycle_model([io.BytesIO(content)])


def _hash_upload(upload: Any) -> Tuple[str, bytes]:
    content = upload.getvalue()
    file_hash = hashlib.sha256(content).hexdigest()
    return file_hash, content


def _record_data_source(pv_source: str, cycle_source: str) -> None:
    st.session_state[DATA_SOURCE_SESSION_KEY] = {"pv": pv_source, "cycle": cycle_source}


def load_shared_data(
    base_dir: Path,
    pv_file,
    cycle_file,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load PV and cycle data from uploads or defaults and cache in the session.

    Uploaded files are hashed to track reuse; parsed DataFrames are cached via
    ``st.cache_data`` keyed by hash to avoid repeated decoding. The hashes are
    also stored in ``st.session_state`` so subsequent pages can rehydrate from
    the upload cache before falling back to bundled defaults.
    """

    default_pv_paths = [str(base_dir / "data" / "PV_8760_MW.csv")]
    default_cycle_paths = [str(base_dir / "data" / "cycle_model.xlsx")]
    upload_cache = _shared_upload_cache()

    pv_df: Optional[pd.DataFrame] = None
    cycle_df: Optional[pd.DataFrame] = None
    pv_source = "default"
    cycle_source = "default"

    if pv_file is not None:
        pv_hash, pv_content = _hash_upload(pv_file)
        upload_cache["pv"] = {"hash": pv_hash, "content": pv_content}
        pv_df = _cache_pv_upload(pv_hash, pv_content)
        st.session_state[PV_HASH_SESSION_KEY] = pv_hash
        _set_session_df(PV_SESSION_KEY, pv_df)
        pv_source = "upload"
    elif PV_HASH_SESSION_KEY in st.session_state:
        pv_hash = st.session_state[PV_HASH_SESSION_KEY]
        cached_payload = upload_cache.get("pv", {})
        if cached_payload.get("hash") == pv_hash and cached_payload.get("content") is not None:
            pv_df = _cache_pv_upload(pv_hash, cached_payload["content"])
            _set_session_df(PV_SESSION_KEY, pv_df)
            pv_source = "cache"
        elif PV_SESSION_KEY in st.session_state:
            pv_df = st.session_state[PV_SESSION_KEY]
            pv_source = "session"
        else:
            pv_df = read_pv_profile(default_pv_paths)
            _set_session_df(PV_SESSION_KEY, pv_df)
    elif upload_cache.get("pv", {}).get("hash"):
        cached_payload = upload_cache["pv"]
        pv_df = _cache_pv_upload(cached_payload["hash"], cached_payload["content"])
        st.session_state[PV_HASH_SESSION_KEY] = cached_payload["hash"]
        _set_session_df(PV_SESSION_KEY, pv_df)
        pv_source = "cache"
    elif PV_SESSION_KEY in st.session_state:
        pv_df = st.session_state[PV_SESSION_KEY]
        pv_source = "session"
    else:
        pv_df = read_pv_profile(default_pv_paths)
        _set_session_df(PV_SESSION_KEY, pv_df)

    if cycle_file is not None:
        cycle_hash, cycle_content = _hash_upload(cycle_file)
        upload_cache["cycle"] = {"hash": cycle_hash, "content": cycle_content}
        cycle_df = _cache_cycle_upload(cycle_hash, cycle_content)
        st.session_state[CYCLE_HASH_SESSION_KEY] = cycle_hash
        _set_session_df(CYCLE_SESSION_KEY, cycle_df)
        cycle_source = "upload"
    elif CYCLE_HASH_SESSION_KEY in st.session_state:
        cycle_hash = st.session_state[CYCLE_HASH_SESSION_KEY]
        cached_payload = upload_cache.get("cycle", {})
        if cached_payload.get("hash") == cycle_hash and cached_payload.get("content") is not None:
            cycle_df = _cache_cycle_upload(cycle_hash, cached_payload["content"])
            _set_session_df(CYCLE_SESSION_KEY, cycle_df)
            cycle_source = "cache"
        elif CYCLE_SESSION_KEY in st.session_state:
            cycle_df = st.session_state[CYCLE_SESSION_KEY]
            cycle_source = "session"
        else:
            cycle_df = read_cycle_model(default_cycle_paths)
            _set_session_df(CYCLE_SESSION_KEY, cycle_df)
    elif upload_cache.get("cycle", {}).get("hash"):
        cached_payload = upload_cache["cycle"]
        cycle_df = _cache_cycle_upload(cached_payload["hash"], cached_payload["content"])
        st.session_state[CYCLE_HASH_SESSION_KEY] = cached_payload["hash"]
        _set_session_df(CYCLE_SESSION_KEY, cycle_df)
        cycle_source = "cache"
    elif CYCLE_SESSION_KEY in st.session_state:
        cycle_df = st.session_state[CYCLE_SESSION_KEY]
        cycle_source = "session"
    else:
        cycle_df = read_cycle_model(default_cycle_paths)
        _set_session_df(CYCLE_SESSION_KEY, cycle_df)

    _record_data_source(pv_source, cycle_source)
    return pv_df, cycle_df


def get_shared_data(base_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return cached PV and cycle data, rehydrating from upload cache when available."""

    default_pv_paths = [str(base_dir / "data" / "PV_8760_MW.csv")]
    default_cycle_paths = [str(base_dir / "data" / "cycle_model.xlsx")]
    upload_cache = _shared_upload_cache()

    pv_df = st.session_state.get(PV_SESSION_KEY)
    cycle_df = st.session_state.get(CYCLE_SESSION_KEY)
    pv_source = "session" if pv_df is not None else "default"
    cycle_source = "session" if cycle_df is not None else "default"

    pv_cache_hash = st.session_state.get(PV_HASH_SESSION_KEY) or upload_cache.get("pv", {}).get("hash")
    if pv_df is None and pv_cache_hash and upload_cache.get("pv", {}).get("content") is not None:
        pv_df = _cache_pv_upload(pv_cache_hash, upload_cache["pv"]["content"])
        st.session_state[PV_HASH_SESSION_KEY] = pv_cache_hash
        _set_session_df(PV_SESSION_KEY, pv_df)
        pv_source = "cache"
    if pv_df is None:
        pv_df = read_pv_profile(default_pv_paths)
        _set_session_df(PV_SESSION_KEY, pv_df)

    cycle_cache_hash = st.session_state.get(CYCLE_HASH_SESSION_KEY) or upload_cache.get("cycle", {}).get("hash")
    if cycle_df is None and cycle_cache_hash and upload_cache.get("cycle", {}).get("content") is not None:
        cycle_df = _cache_cycle_upload(cycle_cache_hash, upload_cache["cycle"]["content"])
        st.session_state[CYCLE_HASH_SESSION_KEY] = cycle_cache_hash
        _set_session_df(CYCLE_SESSION_KEY, cycle_df)
        cycle_source = "cache"
    if cycle_df is None:
        cycle_df = read_cycle_model(default_cycle_paths)
        _set_session_df(CYCLE_SESSION_KEY, cycle_df)

    _record_data_source(pv_source, cycle_source)
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
