"""Shared UI helpers for session-scoped data and configuration."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from services.simulation_core import SimConfig, SimulationOutput
from utils import read_cycle_model, read_pv_profile

PV_SESSION_KEY = "shared_pv_profile_df"
CYCLE_SESSION_KEY = "shared_cycle_model_df"
ECON_PAYLOAD_KEY = "latest_economics_payload"
PV_HASH_SESSION_KEY = "shared_pv_profile_hash"
CYCLE_HASH_SESSION_KEY = "shared_cycle_model_hash"
DATA_SOURCE_SESSION_KEY = "shared_data_source_status"
SIM_CONFIG_KEY = "latest_sim_config"
SIM_CONFIG_FINGERPRINT_KEY = "latest_sim_config_fingerprint"
DOD_OVERRIDE_KEY = "latest_dod_override"
SIM_RESULTS_KEY = "latest_simulation_results"
SIM_SNAPSHOT_KEY = "latest_simulation_snapshot"
MANUAL_AUG_SCHEDULE_KEY = "manual_aug_schedule_rows"
RATE_LIMIT_BYPASS_KEY = "rate_limit_bypass"
RATE_LIMIT_RECENT_RUNS_KEY = "recent_runs"
RATE_LIMIT_LAST_TS_KEY = "last_rate_limit_ts"

DEFAULT_MANUAL_SCHEDULE = [{"Year": 5, "Basis": "Threshold (Capability)", "Amount": 10.0}]


@dataclass
class SimulationResultsState:
    """Cache of the latest simulation output stored in session state."""

    sim_output: SimulationOutput
    dod_override: str


@dataclass
class RateLimitState:
    """Typed container for rate-limit tracking across reruns."""

    bypass: bool = False
    recent_runs: List[float] = field(default_factory=list)
    last_rate_limit_ts: Optional[float] = None


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


def get_data_source_status() -> Dict[str, str]:
    """Return the latest recorded data sources for PV and cycle inputs."""

    return st.session_state.get(DATA_SOURCE_SESSION_KEY, {"pv": "default", "cycle": "default"})


def get_cached_uploads() -> Dict[str, Dict[str, Optional[bytes]]]:
    """Return the shared upload cache payload (hash + content) for PV and cycle files."""

    return _shared_upload_cache()


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
    upload_cache = get_cached_uploads()

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
    upload_cache = get_cached_uploads()

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


def _config_fingerprint(cfg: SimConfig) -> str:
    cfg_dict = asdict(cfg)
    return hashlib.sha256(json.dumps(cfg_dict, sort_keys=True, default=str).encode()).hexdigest()


def get_cached_simulation_config(default_dod: str = "Auto (infer)") -> Tuple[Optional[SimConfig], str]:
    """Return the latest simulation config and DoD override stored in session."""

    cfg: Optional[SimConfig] = st.session_state.get(SIM_CONFIG_KEY)
    dod_override = st.session_state.get(DOD_OVERRIDE_KEY, default_dod)
    return cfg, dod_override


def save_simulation_config(cfg: SimConfig, dod_override: str) -> None:
    """Persist the latest simulation config and DoD override with a fingerprint."""

    st.session_state[SIM_CONFIG_KEY] = cfg
    st.session_state[DOD_OVERRIDE_KEY] = dod_override
    st.session_state[SIM_CONFIG_FINGERPRINT_KEY] = _config_fingerprint(cfg)


def _prune_manual_schedule(rows: List[Dict[str, Any]], max_years: Optional[int]) -> List[Dict[str, Any]]:
    sanitized_rows = rows or DEFAULT_MANUAL_SCHEDULE.copy()
    if max_years is None:
        return sanitized_rows

    filtered: List[Dict[str, Any]] = []
    for row in sanitized_rows:
        try:
            year_value = int(row.get("Year", 0))
        except (TypeError, ValueError):
            continue
        if year_value <= max_years:
            filtered.append(
                {
                    "Year": year_value,
                    "Basis": row.get("Basis", DEFAULT_MANUAL_SCHEDULE[0]["Basis"]),
                    "Amount": float(row.get("Amount", DEFAULT_MANUAL_SCHEDULE[0]["Amount"])),
                }
            )

    if filtered:
        return filtered

    fallback_year = min(max_years, DEFAULT_MANUAL_SCHEDULE[0]["Year"])
    return [{"Year": fallback_year, "Basis": DEFAULT_MANUAL_SCHEDULE[0]["Basis"], "Amount": DEFAULT_MANUAL_SCHEDULE[0]["Amount"]}]


def get_manual_aug_schedule_rows(max_years: Optional[int] = None) -> List[Dict[str, Any]]:
    """Return manual augmentation rows, seeding defaults and pruning invalid years when needed."""

    rows: List[Dict[str, Any]] = st.session_state.get(MANUAL_AUG_SCHEDULE_KEY, DEFAULT_MANUAL_SCHEDULE.copy())
    pruned = _prune_manual_schedule(rows, max_years)
    st.session_state[MANUAL_AUG_SCHEDULE_KEY] = pruned
    return pruned


def save_manual_aug_schedule_rows(rows: List[Dict[str, Any]], max_years: Optional[int]) -> None:
    """Persist manual augmentation rows after pruning any that exceed the project horizon."""

    pruned = _prune_manual_schedule(rows, max_years)
    st.session_state[MANUAL_AUG_SCHEDULE_KEY] = pruned


def get_simulation_results() -> Optional[SimulationResultsState]:
    """Return the cached simulation results, if any."""

    cached = st.session_state.get(SIM_RESULTS_KEY)
    if isinstance(cached, SimulationResultsState):
        return cached
    if isinstance(cached, dict) and "sim_output" in cached:
        return SimulationResultsState(
            sim_output=cached.get("sim_output"),
            dod_override=cached.get("dod_override", "Auto (infer)"),
        )
    return None


def save_simulation_results(sim_output: SimulationOutput, dod_override: str) -> None:
    """Store the latest simulation output with its DoD override."""

    st.session_state[SIM_RESULTS_KEY] = SimulationResultsState(sim_output=sim_output, dod_override=dod_override)


def clear_simulation_results() -> None:
    """Remove cached simulation results and snapshots."""

    st.session_state.pop(SIM_RESULTS_KEY, None)
    st.session_state.pop(SIM_SNAPSHOT_KEY, None)


def save_simulation_snapshot(snapshot: Dict[str, Any]) -> None:
    """Persist the latest simulation snapshot for cross-page reuse."""

    st.session_state[SIM_SNAPSHOT_KEY] = snapshot


def get_simulation_snapshot() -> Optional[Dict[str, Any]]:
    """Return the latest simulation snapshot if available."""

    return st.session_state.get(SIM_SNAPSHOT_KEY)


def get_rate_limit_state() -> RateLimitState:
    """Return a typed rate-limit state with defaults populated."""

    bypass = bool(st.session_state.get(RATE_LIMIT_BYPASS_KEY, False))
    recent_runs = list(st.session_state.get(RATE_LIMIT_RECENT_RUNS_KEY, []))
    last_ts = st.session_state.get(RATE_LIMIT_LAST_TS_KEY)
    if last_ts is not None:
        last_ts = float(last_ts)
    state = RateLimitState(bypass=bypass, recent_runs=recent_runs, last_rate_limit_ts=last_ts)
    update_rate_limit_state(state)
    return state


def update_rate_limit_state(state: RateLimitState) -> None:
    """Write a rate-limit state back to session."""

    st.session_state[RATE_LIMIT_BYPASS_KEY] = state.bypass
    st.session_state[RATE_LIMIT_RECENT_RUNS_KEY] = list(state.recent_runs)
    st.session_state[RATE_LIMIT_LAST_TS_KEY] = state.last_rate_limit_ts


def set_rate_limit_bypass(enabled: bool) -> None:
    """Enable or disable the rate-limit bypass flag."""

    state = get_rate_limit_state()
    state.bypass = enabled
    update_rate_limit_state(state)


def bootstrap_session_state(expected_config: Optional[SimConfig] = None) -> None:
    """Initialize known session keys and clear stale caches when inputs change."""

    _ = get_manual_aug_schedule_rows(getattr(expected_config, "years", None))
    state = get_rate_limit_state()
    update_rate_limit_state(state)

    cached_cfg = st.session_state.get(SIM_CONFIG_KEY)
    cached_fp = st.session_state.get(SIM_CONFIG_FINGERPRINT_KEY)
    expected_fp = _config_fingerprint(expected_config) if expected_config is not None else None

    if cached_cfg is not None and cached_fp is None:
        st.session_state[SIM_CONFIG_FINGERPRINT_KEY] = _config_fingerprint(cached_cfg)

    if expected_fp is not None and cached_fp is not None and cached_fp != expected_fp:
        clear_simulation_results()

    if expected_fp is not None:
        st.session_state[SIM_CONFIG_FINGERPRINT_KEY] = expected_fp
