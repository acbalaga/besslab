"""Streamlit form rendering for simulation inputs.

Centralizing the form setup keeps `app.run_app` focused on orchestration and
lets other pages reuse the same config construction logic.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from services.simulation_core import (
    AUGMENTATION_SCHEDULE_BASIS,
    AugmentationScheduleEntry,
    SimConfig,
    Window,
    build_schedule_from_editor,
    infer_step_hours_from_pv,
    parse_windows,
    validate_pv_profile_duration,
)
from utils import get_rate_limit_password, parse_numeric_series
from utils.dispatch_schedule import resolve_contracted_mw_schedule
from utils.economics import (
    DEFAULT_COST_OF_DEBT_PCT,
    DEFAULT_DEBT_EQUITY_RATIO,
    DEFAULT_FOREX_RATE_PHP_PER_USD,
    DEFAULT_TENOR_YEARS,
    DEVEX_COST_PHP,
    EconomicInputs,
    PriceInputs,
)
from utils.ui_state import (
    get_manual_aug_schedule_rows,
    get_rate_limit_state,
    save_manual_aug_schedule_rows,
    set_rate_limit_bypass,
)


@dataclass
class SimulationFormResult:
    config: SimConfig
    econ_inputs: Optional[EconomicInputs]
    price_inputs: Optional[PriceInputs]
    run_economics: bool
    dod_override: Optional[str]
    run_submitted: bool
    discharge_windows_text: str
    charge_windows_text: str
    dispatch_mode: str
    dispatch_schedule: Dict[str, Any]
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)
    validation_details: List[str] = field(default_factory=list)
    debug_mode: bool = False
    is_valid: bool = True


@dataclass
class DispatchWindowValidation:
    discharge_windows: List[Tuple[int, int]]
    charge_windows: List[Tuple[int, int]]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)

@dataclass
class DispatchScheduleValidation:
    schedule_payload: Dict[str, Any]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: List[str] = field(default_factory=list)


DISPATCH_MODE_FIXED = "Fixed MW + windows"
DISPATCH_MODE_HOURLY = "Hourly capacity schedule"
DISPATCH_MODE_PERIOD = "Period table"
DISPATCH_MODE_OPTIONS = [DISPATCH_MODE_FIXED, DISPATCH_MODE_HOURLY, DISPATCH_MODE_PERIOD]

def _default_hourly_schedule_rows() -> List[Dict[str, float]]:
    return [{"Hour": hour, "Capacity (MW)": 0.0} for hour in range(24)]


def _coerce_columnar_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    """Normalize columnar dict payloads into a DataFrame.

    Streamlit can emit dicts with mixed column value types (lists, dicts, arrays).
    Convert every column into a Series to avoid pandas' mixed-dict ambiguity.
    """
    series_map: Dict[str, pd.Series] = {}
    for key, value in payload.items():
        if isinstance(value, pd.Series):
            series_map[key] = value
        elif isinstance(value, dict):
            series_map[key] = pd.Series(value)
        elif isinstance(value, (list, tuple, np.ndarray)):
            series_map[key] = pd.Series(list(value))
        else:
            series_map[key] = pd.Series([value])
    return pd.DataFrame(series_map)


def _normalize_hourly_schedule_payload(hourly_default: Any) -> pd.DataFrame:
    """Return a 24-row DataFrame for hourly dispatch inputs.

    Streamlit's data editor can yield dict-like payloads (column dicts or row dicts).
    Normalize those shapes to avoid pandas' ambiguous ordering errors.
    """
    default_rows = _default_hourly_schedule_rows()
    if hourly_default is None:
        return pd.DataFrame(default_rows)

    if isinstance(hourly_default, pd.DataFrame):
        df = hourly_default.copy()
    elif isinstance(hourly_default, list):
        df = pd.DataFrame.from_records(hourly_default)
    elif isinstance(hourly_default, dict):
        if "data" in hourly_default:
            return _normalize_hourly_schedule_payload(hourly_default["data"])
        editor_state_keys = {"edited_rows", "added_rows", "deleted_rows"}
        if set(hourly_default.keys()).issubset(editor_state_keys):
            return pd.DataFrame(default_rows)
        if all(isinstance(value, dict) for value in hourly_default.values()):
            df = pd.DataFrame.from_records(list(hourly_default.values()))
        else:
            df = _coerce_columnar_frame(hourly_default)
    else:
        df = pd.DataFrame()

    if df.empty:
        return pd.DataFrame(default_rows)

    rename_map: Dict[str, str] = {}
    for column in df.columns:
        if not isinstance(column, str):
            continue
        if column.lower() == "hour":
            rename_map[column] = "Hour"
        elif column.lower().startswith("capacity"):
            rename_map[column] = "Capacity (MW)"
    if rename_map:
        df = df.rename(columns=rename_map)

    if "Hour" not in df.columns or "Capacity (MW)" not in df.columns:
        return pd.DataFrame(default_rows)

    df = df[["Hour", "Capacity (MW)"]].copy()
    df["Hour"] = pd.to_numeric(df["Hour"], errors="coerce")
    df["Capacity (MW)"] = pd.to_numeric(df["Capacity (MW)"], errors="coerce")
    df = df.dropna(subset=["Hour"])
    df["Hour"] = df["Hour"].astype(int)
    # Keep the first entry per hour to avoid duplicates if Streamlit returns row dicts.
    df = df.drop_duplicates(subset=["Hour"], keep="first")

    hours_df = pd.DataFrame({"Hour": list(range(24))})
    normalized = hours_df.merge(df, on="Hour", how="left")
    normalized["Capacity (MW)"] = normalized["Capacity (MW)"].fillna(0.0)
    return normalized


def _format_window(window: Window) -> str:
    if window.source:
        return window.source
    start_hour = int(window.start)
    start_minute = int(round((window.start - start_hour) * 60))
    end_hour = int(window.end)
    end_minute = int(round((window.end - end_hour) * 60))
    return f"{start_hour:02d}:{start_minute:02d}-{end_hour:02d}:{end_minute:02d}"


def _intervals(window: Window) -> List[Tuple[float, float]]:
    if window.start <= window.end:
        return [(window.start, window.end)]
    return [(window.start, 24.0), (0.0, window.end)]


def _collect_window_warnings(window_type: str, windows: List[Window], include_overlap: bool = True) -> List[str]:
    """Return warnings about zero-length, duplicate, or overlapping windows."""
    local_warnings: List[str] = []
    seen: Dict[Tuple[float, float], List[Window]] = {}

    for window in windows:
        key = (window.start, window.end)
        seen.setdefault(key, []).append(window)
        if window.start == window.end:
            window_label = _format_window(window)
            local_warnings.append(
                f"{window_type} window '{window_label}' has zero length (start == end). "
                "Adjust the end time or remove the window."
            )

    for key, matches in seen.items():
        if len(matches) > 1:
            window_label = _format_window(matches[0])
            local_warnings.append(
                f"Duplicate {window_type} window '{window_label}' detected ({len(matches)} occurrences). "
                "Remove duplicates or adjust the times."
            )

    if include_overlap:
        for idx, left in enumerate(windows):
            for right in windows[idx + 1 :]:
                for left_interval in _intervals(left):
                    for right_interval in _intervals(right):
                        if max(left_interval[0], right_interval[0]) < min(left_interval[1], right_interval[1]):
                            local_warnings.append(
                                f"{window_type} windows '{_format_window(left)}' and "
                                f"'{_format_window(right)}' overlap. Split or adjust end times to avoid overlap."
                            )
                            break
                    else:
                        continue
                    break

    return local_warnings


def _parse_hhmm_time(value: str) -> float:
    """Parse HH[:MM] into an hour-of-day float."""
    parts = value.split(":")
    if len(parts) == 1:
        hour = int(parts[0])
        minute = 0
    elif len(parts) == 2:
        hour = int(parts[0])
        minute = int(parts[1])
    else:
        raise ValueError("Too many ':' characters")
    if not (0 <= hour <= 23) or not (0 <= minute <= 59):
        raise ValueError("Hour must be 0-23 and minute 0-59")
    return hour + minute / 60.0


def _cell_to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _cell_to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _window_overlaps(windows: List[Window]) -> List[str]:
    errors: List[str] = []
    for idx, left in enumerate(windows):
        for right in windows[idx + 1 :]:
            for left_interval in _intervals(left):
                for right_interval in _intervals(right):
                    if max(left_interval[0], right_interval[0]) < min(left_interval[1], right_interval[1]):
                        errors.append(
                            f"Overlapping period rows '{_format_window(left)}' and '{_format_window(right)}' detected."
                        )
                        break
                else:
                    continue
                break
    return errors


def _missing_hour_warnings(windows: List[Window], context: str) -> List[str]:
    uncovered = []
    for hour in range(24):
        hod = hour + 0.5
        if not any(window.contains(hod) for window in windows):
            uncovered.append(hour)
    if not uncovered:
        return []
    preview = ", ".join(f"{hour:02d}:00" for hour in uncovered[:6])
    tail = "…" if len(uncovered) > 6 else ""
    return [
        (
            f"{context} does not cover all 24 hours. Uncovered hours include {preview}{tail}. "
            "Uncovered hours will default to 0 MW."
        )
    ]


def validate_dispatch_windows(
    discharge_text: str, charge_text: str, require_discharge_windows: bool = True
) -> DispatchWindowValidation:
    """Parse and validate dispatch windows, returning user-facing feedback."""

    discharge_windows, discharge_window_warnings = parse_windows(discharge_text)
    charge_windows, charge_window_warnings = parse_windows(charge_text)
    warnings = list(discharge_window_warnings + charge_window_warnings)
    errors: List[str] = []
    details = [
        f"Raw discharge text: {discharge_text or '<blank>'}",
        f"Raw charge text: {charge_text or '<blank>'}",
        f"Parsed discharge windows: {discharge_windows or 'none'}",
        f"Parsed charge windows: {charge_windows or 'none'}",
    ]

    warnings.extend(_collect_window_warnings("Discharge", discharge_windows))
    warnings.extend(_collect_window_warnings("Charge", charge_windows))

    if require_discharge_windows and not discharge_windows:
        errors.append("Provide at least one discharge window in HH:MM-HH:MM format (e.g., 10:00-14:00).")

    return DispatchWindowValidation(
        discharge_windows=discharge_windows,
        charge_windows=charge_windows,
        errors=errors,
        warnings=warnings,
        details=details,
    )


def validate_dispatch_schedule(
    dispatch_mode: str,
    hourly_schedule_df: Optional[pd.DataFrame],
    period_schedule_df: Optional[pd.DataFrame],
) -> DispatchScheduleValidation:
    """Validate dispatch schedule inputs without mutating config."""
    errors: List[str] = []
    warnings: List[str] = []
    details: List[str] = [f"Dispatch schedule mode: {dispatch_mode}"]
    schedule_payload: Dict[str, Any] = {"mode": dispatch_mode}

    if dispatch_mode == DISPATCH_MODE_HOURLY:
        if hourly_schedule_df is None or hourly_schedule_df.empty:
            errors.append("Provide a full 24-hour capacity schedule (one MW value per hour).")
            return DispatchScheduleValidation(schedule_payload, errors, warnings, details)

        hourly_values: List[Optional[float]] = []
        missing_hours: List[int] = []
        for _, row in hourly_schedule_df.iterrows():
            hour_value = _cell_to_float(row.get("Hour"))
            if hour_value is None:
                continue
            if not (0 <= hour_value <= 23):
                errors.append(f"Hourly schedule includes invalid hour value {hour_value}. Use 0-23.")
                continue
            capacity_value = _cell_to_float(row.get("Capacity (MW)"))
            if capacity_value is None:
                missing_hours.append(int(hour_value))
                hourly_values.append(None)
                continue
            if capacity_value < 0:
                errors.append(f"Hourly schedule hour {int(hour_value):02d}:00 has negative MW.")
            hourly_values.append(float(capacity_value))

        if len(hourly_values) != 24:
            errors.append("Hourly schedule must include exactly 24 rows (hours 00-23).")
        if missing_hours:
            missing_preview = ", ".join(f"{hour:02d}:00" for hour in sorted(missing_hours[:6]))
            tail = "…" if len(missing_hours) > 6 else ""
            errors.append(
                f"Hourly schedule is missing MW values for {len(missing_hours)} hours ({missing_preview}{tail})."
            )
        schedule_payload["hourly_mw"] = hourly_values
        details.append(f"Hourly schedule rows: {len(hourly_values)}")

    if dispatch_mode == DISPATCH_MODE_PERIOD:
        if period_schedule_df is None or period_schedule_df.empty:
            errors.append("Add at least one period row with start/end time and capacity MW.")
            return DispatchScheduleValidation(schedule_payload, errors, warnings, details)

        period_rows: List[Dict[str, Any]] = []
        windows: List[Window] = []
        for row_idx, row in period_schedule_df.iterrows():
            start_text = _cell_to_str(row.get("Start time"))
            end_text = _cell_to_str(row.get("End time"))
            capacity_value = _cell_to_float(row.get("Capacity (MW)"))

            if not any([start_text, end_text, capacity_value is not None]):
                continue

            row_label = f"Row {row_idx + 1}"
            if start_text is None or end_text is None or capacity_value is None:
                errors.append(f"{row_label}: Provide start time, end time, and capacity MW.")
                continue
            try:
                start = _parse_hhmm_time(start_text)
                end = _parse_hhmm_time(end_text)
            except ValueError as exc:
                errors.append(f"{row_label}: {exc}. Use HH:MM (00:00-23:59).")
                continue
            if capacity_value < 0:
                errors.append(f"{row_label}: Capacity MW must be non-negative.")
            windows.append(Window(start=start, end=end, source=f"{start_text}-{end_text}"))
            period_rows.append(
                {"start_time": start_text, "end_time": end_text, "capacity_mw": float(capacity_value)}
            )

        errors.extend(_window_overlaps(windows))
        warnings.extend(_collect_window_warnings("Period", windows, include_overlap=False))
        warnings.extend(_missing_hour_warnings(windows, "Period schedule"))

        schedule_payload["period_table"] = period_rows
        details.append(f"Period rows parsed: {len(period_rows)}")

    return DispatchScheduleValidation(schedule_payload, errors, warnings, details)


def validate_manual_augmentation_schedule(
    aug_mode: str, manual_schedule_entries: List[AugmentationScheduleEntry], manual_schedule_errors: List[str]
) -> Tuple[List[str], List[str]]:
    """Return manual augmentation errors and debug details without halting the app."""

    if aug_mode != "Manual":
        return [], []

    errors = list(manual_schedule_errors)
    if not manual_schedule_entries:
        errors.append("Add at least one valid manual augmentation event before running the simulation.")

    details = [
        f"Manual augmentation rows: {len(manual_schedule_entries)}",
        f"Manual augmentation errors: {errors or 'none'}",
    ]

    return errors, details


def render_rate_limit_section() -> None:
    """Allow users to enter a password to bypass the per-session rate limit."""
    expected_password = get_rate_limit_password()
    rate_limit_state = get_rate_limit_state()
    rate_limit_password = st.text_input(
        "Remove rate limit (password)",
        type="password",
        help=("Enter the configured password to disable the session rate limit. If no secret is set, use 'besslab'."),
        key="inputs_rate_limit_password",
    )
    if rate_limit_password:
        if rate_limit_password == expected_password:
            set_rate_limit_bypass(True)
            st.success("Rate limit disabled for this session.")
        else:
            set_rate_limit_bypass(False)
            st.error("Incorrect password. Rate limit still active.")
    elif rate_limit_state.bypass:
        st.caption("Rate limit disabled for this session.")


def render_simulation_form(pv_df: pd.DataFrame, cycle_df: pd.DataFrame) -> SimulationFormResult:
    """Render the main simulation inputs and return the assembled configuration."""
    st.subheader("Inputs")
    debug_mode = st.checkbox(
        "Debug mode",
        value=False,
        help="Show validation details and recent inputs without exposing stack traces to all users.",
        key="inputs_debug_mode",
    )
    validation_errors: List[str] = []
    validation_warnings: List[str] = []
    validation_details: List[str] = []

    with st.expander("Technical", expanded=True):
        # Project & PV
        with st.expander("Project & PV", expanded=True):
            c1, c2, c3 = st.columns(3)
            with c1:
                years = st.selectbox(
                    "Project life (years)",
                    list(range(10, 36, 5)),
                    index=2,
                    help="Extend to test augmentation schedules and end effects.",
                    key="inputs_years",
                )
            with c2:
                pv_deg = (
                    st.number_input(
                        "PV degradation %/yr",
                        0.0,
                        5.0,
                        0.6,
                        0.1,
                        help="Applied multiplicatively per year (e.g., 0.6% → (1−0.006)^year).",
                        key="inputs_pv_deg_pct",
                    )
                    / 100.0
                )
            with c3:
                pv_avail = st.slider(
                    "PV availability",
                    0.90,
                    1.00,
                    0.98,
                    0.01,
                    help="Uptime factor applied to PV output.",
                    key="inputs_pv_avail",
                )

        # Availability (kept outside the main form so toggles rerender immediately)
        with st.expander("Availability", expanded=True):
            c1, c2 = st.columns(2)
            with c1:
                bess_avail = st.slider(
                    "BESS availability",
                    0.90,
                    1.00,
                    0.99,
                    0.01,
                    help="Uptime factor applied to BESS power capability.",
                    key="inputs_bess_avail",
                )
            with c2:
                use_split_rte = st.checkbox(
                    "Use separate charge/discharge efficiencies",
                    value=False,
                    help="Select to enter distinct charge and discharge efficiencies instead of a single round-trip value.",
                    key="inputs_use_split_rte",
                )
                if use_split_rte:
                    charge_eff = st.slider(
                        "Charge efficiency (AC-AC)",
                        0.70,
                        0.99,
                        0.94,
                        0.01,
                        help="Applied when absorbing energy; multiplied with discharge efficiency to form the round-trip value.",
                        key="inputs_charge_eff",
                    )
                    discharge_eff = st.slider(
                        "Discharge efficiency (AC-AC)",
                        0.70,
                        0.99,
                        0.94,
                        0.01,
                        help="Applied when delivering energy; multiplied with charge efficiency to form the round-trip value.",
                        key="inputs_discharge_eff",
                    )
                    rte = charge_eff * discharge_eff
                    st.caption(f"Implied round-trip efficiency: {rte:.3f} (charge × discharge).")
                else:
                    rte = st.slider(
                        "Round-trip efficiency (single, at POI)",
                        0.70,
                        0.99,
                        0.88,
                        0.01,
                        help="Single RTE; internally split √RTE for charge/discharge.",
                        key="inputs_rte",
                    )
                    charge_eff = None
                    discharge_eff = None

        manual_schedule_entries: List[AugmentationScheduleEntry] = []
        manual_schedule_errors: List[str] = []
        manual_schedule_rows = get_manual_aug_schedule_rows(int(years))

        # Augmentation (kept outside the main form so dropdown changes reveal inputs immediately).
        with st.expander("Augmentation strategy", expanded=False):
            aug_mode = st.selectbox(
                "Strategy",
                ["None", "Threshold", "Periodic", "Manual"],
                index=0,
                key="augmentation_strategy_mode",
            )

            aug_size_mode = "Percent"
            aug_fixed_energy = 0.0
            retire_enabled = False
            retire_soh = 0.60
            retire_replace_mode = "None"
            retire_replace_pct = 0.0
            retire_replace_fixed_mwh = 0.0

            if aug_mode == "Manual":
                st.caption("Define explicit augmentation events by year. Save the table to persist edits across reruns.")
                with st.form("manual_aug_schedule_form", clear_on_submit=False):
                    manual_schedule_df = st.data_editor(
                        pd.DataFrame(manual_schedule_rows),
                        key="manual_aug_schedule_editor",
                        column_config={
                            "Year": st.column_config.NumberColumn("Year", min_value=1, step=1),
                            "Basis": st.column_config.SelectboxColumn("Basis", options=AUGMENTATION_SCHEDULE_BASIS),
                            "Amount": st.column_config.NumberColumn(
                                "Amount", min_value=0.0, format="%.3f", help="Percent or MW/MWh depending on basis."
                            ),
                        },
                        num_rows="dynamic",
                        hide_index=True,
                        use_container_width=True,
                    )
                    saved_manual_schedule = st.form_submit_button("Save augmentation table", use_container_width=True)
                if saved_manual_schedule:
                    save_manual_aug_schedule_rows(manual_schedule_df.to_dict("records"), int(years))
                    manual_schedule_rows = get_manual_aug_schedule_rows(int(years))
                    st.success("Saved augmentation events for this session.")
                manual_schedule_df = pd.DataFrame(manual_schedule_rows)
                manual_schedule_entries, manual_schedule_errors = build_schedule_from_editor(manual_schedule_df, int(years))
                if manual_schedule_errors:
                    for err in manual_schedule_errors:
                        st.error(err)
                elif not manual_schedule_entries:
                    st.warning("Add at least one row to run a manual augmentation schedule.")
                aug_thr_margin = 0.0
                aug_topup = 0.0
                aug_every = 5
                aug_frac = 0.10
                aug_trigger_type = "Capability"
                aug_soh_trig = 0.80
                aug_soh_add = 0.10
            elif aug_mode == "Threshold":
                trigger = st.selectbox(
                    "Trigger type",
                    ["Capability", "SOH"],
                    index=0,
                    help="Capability: Compare EOY capability vs target MWh/day.  SOH: Compare fleet SOH vs threshold.",
                    key="inputs_aug_trigger_type",
                )
                if trigger == "Capability":
                    c1, c2 = st.columns(2)
                    with c1:
                        aug_thr_margin = (
                            st.number_input(
                                "Allowance margin (%)",
                                0.0,
                                None,
                                0.0,
                                0.5,
                                help="Trigger when capability < target × (1 − margin).",
                                key="inputs_aug_threshold_margin_pct",
                            )
                            / 100.0
                        )
                    with c2:
                        aug_topup = (
                            st.number_input(
                                "Top-up margin (%)",
                                0.0,
                                None,
                                5.0,
                                0.5,
                                help="Augment up to target × (1 + margin) when triggered.",
                                key="inputs_aug_topup_margin_pct",
                            )
                            / 100.0
                        )
                    aug_every = 5
                    aug_frac = 0.10
                    aug_trigger_type = "Capability"
                    aug_soh_trig = 0.80
                    aug_soh_add = 0.10
                else:
                    c1, c2 = st.columns(2)
                    with c1:
                        aug_soh_trig = (
                            st.number_input(
                                "SOH trigger (%)",
                                50.0,
                                100.0,
                                80.0,
                                1.0,
                                help="Augment when SOH falls below this threshold.",
                                key="inputs_aug_soh_trigger_pct",
                            )
                            / 100.0
                        )
                    with c2:
                        aug_soh_add = (
                            st.number_input(
                                "Add % of initial BOL energy",
                                0.0,
                                None,
                                10.0,
                                1.0,
                                help="Added energy as % of initial BOL. Power added to keep original C-hours.",
                                key="inputs_aug_soh_add_pct",
                            )
                            / 100.0
                        )
                    aug_thr_margin = 0.0
                    aug_topup = 0.0
                    aug_every = 5
                    aug_frac = 0.10
                    aug_trigger_type = "SOH"
            elif aug_mode == "Periodic":
                c1, c2 = st.columns(2)
                with c1:
                    aug_every = st.number_input(
                        "Every N years",
                        1,
                        None,
                        5,
                        1,
                        help="Add capacity on this cadence (e.g., every 5 years).",
                        key="inputs_aug_periodic_every_years",
                    )
                with c2:
                    aug_frac = (
                        st.number_input(
                            "Add % of current BOL-ref energy",
                            0.0,
                            None,
                            10.0,
                            1.0,
                            help="Top-up energy relative to current BOL reference.",
                            key="inputs_aug_periodic_add_pct",
                        )
                        / 100.0
                    )
                aug_thr_margin = 0.0
                aug_topup = 0.0
                aug_trigger_type = "Capability"
                aug_soh_trig = 0.80
                aug_soh_add = 0.10
            else:
                aug_thr_margin = 0.0
                aug_topup = 0.0
                aug_every = 5
                aug_frac = 0.10
                aug_trigger_type = "Capability"
                aug_soh_trig = 0.80
                aug_soh_add = 0.10

            if aug_mode not in ["None", "Manual"]:
                aug_size_mode = st.selectbox(
                    "Augmentation sizing",
                    ["Percent", "Fixed"],
                    format_func=lambda k: "% basis" if k == "Percent" else "Fixed energy (MWh)",
                    help="Choose whether to size augmentation as a percent or a fixed MWh add.",
                    key="inputs_aug_size_mode",
                )
                if aug_size_mode == "Fixed":
                    aug_fixed_energy = st.number_input(
                        "Fixed energy added per event (MWh, BOL basis)",
                        min_value=0.0,
                        value=0.0,
                        step=1.0,
                        help="Adds this BOL-equivalent energy whenever augmentation is triggered.",
                        key="inputs_aug_fixed_energy_mwh",
                    )

            if aug_mode != "None":
                retire_enabled = st.checkbox(
                    "Retire low-SOH cohorts when augmenting",
                    value=False,
                    help="Remove cohorts once their SOH falls below the retirement threshold.",
                    key="inputs_aug_retire_enabled",
                )
                if retire_enabled:
                    retire_soh = (
                        st.number_input(
                            "Retirement SOH threshold (%)",
                            min_value=0.0,
                            max_value=100.0,
                            value=60.0,
                            step=1.0,
                            help="Cohorts at or below this SOH are retired before applying augmentation.",
                            key="inputs_aug_retire_soh_pct",
                        )
                        / 100.0
                    )
                    retire_replace_mode = st.selectbox(
                        "Replacement after retirement",
                        ["None", "Percent", "Fixed"],
                        format_func=lambda k: "None"
                        if k == "None"
                        else ("% of BOL energy" if k == "Percent" else "Fixed energy (MWh)"),
                        help="Optionally replace retired cohorts with new energy on a BOL basis.",
                        key="inputs_aug_retire_replace_mode",
                    )
                    if retire_replace_mode == "Percent":
                        retire_replace_pct = (
                            st.number_input(
                                "Replacement % of BOL energy",
                                min_value=0.0,
                                max_value=200.0,
                                value=0.0,
                                step=1.0,
                                help="Add this fraction of initial BOL energy when retirement happens.",
                                key="inputs_aug_retire_replace_pct",
                            )
                            / 100.0
                        )
                    elif retire_replace_mode == "Fixed":
                        retire_replace_fixed_mwh = st.number_input(
                            "Replacement energy (MWh, BOL basis)",
                            min_value=0.0,
                            value=0.0,
                            step=1.0,
                            help="Add this BOL-equivalent energy when retirement happens.",
                            key="inputs_aug_retire_replace_fixed_mwh",
                        )

        with st.container():
            # BESS Specs
            with st.expander("BESS Specs (high-level)", expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    init_power = st.number_input(
                        "Power rating (MW)",
                        1.0,
                        None,
                        30.0,
                        1.0,
                        help="Initial nameplate power (POI context), before availability.",
                        key="inputs_initial_power_mw",
                    )
                with c2:
                    init_energy = st.number_input(
                        "Usable energy at BOL (MWh)",
                        1.0,
                        None,
                        120.0,
                        1.0,
                        help="Initial usable energy (POI context).",
                        key="inputs_initial_usable_mwh",
                    )
                with c3:
                    soc_floor = st.slider(
                        "SOC floor (%)",
                        0,
                        50,
                        10,
                        1,
                        help="Reserve to protect cycling; lowers daily swing.",
                        key="inputs_soc_floor_pct",
                    ) / 100.0
                    soc_ceiling = st.slider(
                        "SOC ceiling (%)",
                        50,
                        100,
                        98,
                        1,
                        help="Upper limit to protect cycling; raises daily swing when higher.",
                        key="inputs_soc_ceiling_pct",
                    ) / 100.0

            # Degradation
            with st.expander("Degradation modeling", expanded=False):
                c1, c2 = st.columns(2)
                with c1:
                    cal_fade = (
                        st.number_input(
                            "Calendar fade %/yr (empirical)",
                            0.0,
                            5.0,
                            1.0,
                            0.1,
                            help="Multiplicative retention: (1 − rate)^year.",
                            key="inputs_calendar_fade_pct",
                        )
                        / 100.0
                    )
                with c2:
                    dod_override = st.selectbox(
                        "Degradation DoD basis",
                        ["Auto (infer)", "10%", "20%", "40%", "80%", "100%"],
                        help="Use the cycle table at a fixed DoD, or let the app infer based on median daily discharge.",
                        key="inputs_dod_override",
                    )

    with st.expander("Market", expanded=True):
        # Dispatch
        with st.expander("Dispatch Strategy", expanded=True):
            dispatch_mode = st.radio(
                "Dispatch input mode",
                options=DISPATCH_MODE_OPTIONS,
                horizontal=True,
                help=(
                    "Use fixed windows for legacy firm-capacity runs, or provide an hourly/period schedule for "
                    "variable dispatch. Hourly/period schedules are validated and stored for export."
                ),
                key="inputs_dispatch_mode",
            )
            contracted_mw = st.number_input(
                "Contracted MW (firm)",
                0.0,
                None,
                30.0,
                1.0,
                help="Firm capacity to meet during discharge windows (used by fixed-window simulations today).",
                key="inputs_contracted_mw",
            )

            discharge_windows_text = st.session_state.get(
                "inputs_discharge_windows", "10:00-14:00, 18:00-22:00"
            )
            charge_windows_text = st.session_state.get("inputs_charge_windows", "")
            hourly_schedule_df: Optional[pd.DataFrame] = None
            period_schedule_df: Optional[pd.DataFrame] = None

            if dispatch_mode == DISPATCH_MODE_FIXED:
                c1, c2 = st.columns(2)
                with c1:
                    discharge_windows_text = st.text_input(
                        "Discharge windows (HH:MM-HH:MM, comma-separated)",
                        "10:00-14:00, 18:00-22:00",
                        help=(
                            "Ex: 10:00-14:00, 18:00-22:00. Wrap-around windows like 22:00-02:00 are allowed."
                        ),
                        key="inputs_discharge_windows",
                    )
                with c2:
                    charge_windows_text = st.text_input(
                        "Charge windows (blank = any PV hours)",
                        "",
                        help="PV-only charging; blank allows any PV hour. Wrap-around windows are allowed.",
                        key="inputs_charge_windows",
                    )
            elif dispatch_mode == DISPATCH_MODE_HOURLY:
                st.caption(
                    "Provide a MW value for each hour (00:00-23:00). Use 0 for off hours; blanks are flagged "
                    "as missing hours."
                )
                hourly_default = st.session_state.get("inputs_dispatch_hourly_schedule")
                hourly_schedule_df = st.data_editor(
                    _normalize_hourly_schedule_payload(hourly_default),
                    num_rows="fixed",
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Hour": st.column_config.NumberColumn("Hour", disabled=True),
                        "Capacity (MW)": st.column_config.NumberColumn(
                            "Capacity (MW)", min_value=0.0, step=0.01, format="%.2f"
                        ),
                    },
                    key="inputs_dispatch_hourly_schedule",
                )
            else:
                st.caption(
                    "Add periods with HH:MM start/end and a MW capacity. Wrap-around windows like 22:00-02:00 "
                    "are allowed. Overlaps are not allowed; gaps imply 0 MW for uncovered hours."
                )
                period_default = st.session_state.get("inputs_dispatch_period_schedule", [])
                period_schedule_df = st.data_editor(
                    pd.DataFrame(period_default, columns=["Start time", "End time", "Capacity (MW)"]),
                    num_rows="dynamic",
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Start time": st.column_config.TextColumn("Start time", help="HH:MM (00:00-23:59)"),
                        "End time": st.column_config.TextColumn("End time", help="HH:MM (00:00-23:59)"),
                        "Capacity (MW)": st.column_config.NumberColumn(
                            "Capacity (MW)", min_value=0.0, step=0.5
                        ),
                    },
                    key="inputs_dispatch_period_schedule",
                )

    econ_inputs: Optional[EconomicInputs] = None
    price_inputs: Optional[PriceInputs] = None

    with st.expander("Financial", expanded=False):
        # Keep the economics toggle outside the main input block so checking it reveals inputs immediately.
        st.markdown("### Optional economics (NPV, IRR, LCOE, LCOS)")
        run_economics = st.checkbox(
            "Compute economics using simulation outputs",
            value=False,
            help=(
                "Enable to enter financial assumptions and derive LCOE/LCOS, NPV, and IRR from the simulated annual energy streams."
            ),
            key="inputs_run_economics",
        )

        forex_rate_php_per_usd = DEFAULT_FOREX_RATE_PHP_PER_USD

        if run_economics:
            wesm_pricing_enabled = False
            sell_to_wesm = False

            econ_col1, econ_col2, econ_col3 = st.columns(3)
            with econ_col1:
                wacc_pct = st.number_input(
                    "WACC (%)",
                    min_value=0.0,
                    max_value=30.0,
                    value=8.0,
                    step=0.1,
                    key="inputs_wacc_pct",
                )
                inflation_pct = st.number_input(
                    "Inflation rate (%)",
                    min_value=0.0,
                    max_value=20.0,
                    value=3.0,
                    step=0.1,
                    help="Used to derive the real discount rate applied to costs and revenues.",
                    key="inputs_inflation_pct",
                )
                discount_rate = max((1 + wacc_pct / 100.0) / (1 + inflation_pct / 100.0) - 1, 0.0)
                st.caption(f"Real discount rate derived from WACC and inflation: {discount_rate * 100:.2f}%.")
                forex_rate_php_per_usd = st.number_input(
                    "FX rate (PHP/USD)",
                    min_value=1.0,
                    value=float(DEFAULT_FOREX_RATE_PHP_PER_USD),
                    step=0.5,
                    help="Used to convert PHP-denominated inputs (prices, OPEX, DevEx) to USD.",
                    key="inputs_forex_rate_php_per_usd",
                )
            default_contract_php_per_kwh = round(120.0 / 1000.0 * forex_rate_php_per_usd, 2)
            with econ_col2:
                capex_mode = st.radio(
                    "BESS CAPEX input",
                    options=["USD/kWh (BOL)", "Total CAPEX (USD million)"],
                    horizontal=True,
                    help=(
                        "Enter BESS CAPEX as a unit rate per kWh of BOL energy or override with a total USD million value."
                    ),
                    key="inputs_capex_mode",
                )
                capex_usd_per_kwh = 0.0
                capex_musd = 0.0
                bess_bol_kwh = init_energy * 1000.0
                if capex_mode == "USD/kWh (BOL)":
                    default_capex_usd_per_kwh = 0.0
                    if bess_bol_kwh > 0:
                        default_capex_usd_per_kwh = 40_000_000.0 / bess_bol_kwh
                    capex_usd_per_kwh = st.number_input(
                        "CAPEX (USD/kWh, BOL)",
                        min_value=0.0,
                        value=round(default_capex_usd_per_kwh, 2),
                        step=1.0,
                        help="Applied to BOL usable energy (kWh) to derive total CAPEX in USD.",
                        key="inputs_capex_usd_per_kwh",
                    )
                    capex_total_usd = capex_usd_per_kwh * bess_bol_kwh
                    capex_musd = capex_total_usd / 1_000_000.0
                    st.caption(f"Implied BESS CAPEX: ${capex_musd:,.2f}M.")
                else:
                    capex_musd = st.number_input(
                        "Total BESS CAPEX (USD million)",
                        min_value=0.0,
                        value=40.0,
                        step=0.1,
                        help="Single-number override for BESS CAPEX in USD million.",
                        key="inputs_capex_musd",
                    )
                pv_capex_musd = st.number_input(
                    "PV CAPEX (USD million)",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    help="Standalone PV CAPEX; combined with BESS CAPEX for total project spend.",
                    key="inputs_pv_capex_musd",
                )
                total_capex_musd = capex_musd + pv_capex_musd
                st.caption(f"Total project CAPEX (BESS + PV): ${total_capex_musd:,.2f}M.")
                opex_mode = st.radio(
                    "OPEX input",
                    options=["% of CAPEX per year", "PHP/kWh on total generation"],
                    horizontal=True,
                    help="Choose a fixed % of CAPEX/year or a PHP/kWh rate applied to total generation.",
                    key="inputs_opex_mode",
                )
                fixed_opex_pct = 0.0
                opex_php_per_kwh = None
                if opex_mode == "% of CAPEX per year":
                    fixed_opex_pct = st.number_input(
                        "Fixed OPEX (% of CAPEX per year)",
                        min_value=0.0,
                        max_value=20.0,
                        value=2.0,
                        step=0.1,
                        help="Enter the percent value (e.g., 2.0 for 2%).",
                        key="inputs_fixed_opex_pct",
                    )
                else:
                    opex_php_per_kwh = st.number_input(
                        "OPEX (PHP/kWh on total generation)",
                        min_value=0.0,
                        value=0.0,
                        step=0.05,
                        help="Converted to USD/MWh using the FX rate; applied to total generation.",
                        key="inputs_opex_php_per_kwh",
                    )
                    if opex_php_per_kwh > 0:
                        opex_usd_per_mwh = opex_php_per_kwh / forex_rate_php_per_usd * 1000.0
                        st.caption(f"Converted OPEX: ${opex_usd_per_mwh:,.2f}/MWh.")
                fixed_opex_musd = st.number_input(
                    "Additional fixed OPEX (USD million/yr)",
                    min_value=0.0,
                    value=0.0,
                    step=0.1,
                    key="inputs_fixed_opex_musd",
                )
                include_devex_default = bool(st.session_state.get("inputs_include_devex_year0", False))
                devex_choice = st.radio(
                    "DevEx at year 0",
                    options=["Exclude", "Include"],
                    index=1 if include_devex_default else 0,
                    horizontal=True,
                    help=(
                        "Include or exclude the development expenditure at year 0. The PHP amount is "
                        "converted to USD using the FX rate and flows through discounted costs, "
                        "LCOE/LCOS, NPV, and IRR."
                    ),
                    key="inputs_devex_choice",
                )
                include_devex_year0 = devex_choice == "Include"
                st.session_state["inputs_include_devex_year0"] = include_devex_year0
                devex_cost_php = st.number_input(
                    "DevEx amount (PHP)",
                    min_value=0.0,
                    value=float(DEVEX_COST_PHP),
                    step=1_000_000.0,
                    help="Used only when DevEx is included.",
                    disabled=not include_devex_year0,
                    key="inputs_devex_cost_php",
                )
                devex_cost_usd = devex_cost_php / forex_rate_php_per_usd if forex_rate_php_per_usd else 0.0
                if include_devex_year0:
                    st.caption(
                        "DevEx conversion: "
                        f"PHP {devex_cost_php:,.0f} ≈ ${devex_cost_usd / 1_000_000:,.2f}M."
                    )
            with econ_col3:
                contract_price_php_per_kwh = st.number_input(
                    "Contract price (PHP/kWh for delivered energy)",
                    min_value=0.0,
                    value=default_contract_php_per_kwh,
                    step=0.05,
                    help="Price converted to USD/MWh using the FX rate above.",
                    key="inputs_contract_price_php_per_kwh",
                )
                escalate_prices = st.checkbox(
                    "Escalate prices with inflation",
                    value=False,
                    key="inputs_escalate_prices",
                )
                wesm_pricing_enabled = st.checkbox(
                    "Apply WESM pricing to contract shortfalls",
                    value=False,
                    help=(
                        "Uses the uploaded (or bundled) hourly WESM profile to price contract shortfalls."
                    ),
                    key="inputs_wesm_pricing_enabled",
                )
                sell_to_wesm = st.checkbox(
                    "Sell PV surplus to WESM",
                    value=False,
                    help=(
                        "When enabled, PV surplus (excess MWh) is credited at a WESM sale price; otherwise surplus "
                        "is excluded from revenue. Pricing comes from the hourly WESM profile."
                    ),
                    disabled=not wesm_pricing_enabled,
                    key="inputs_sell_to_wesm",
                )

                contract_price = contract_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
                st.caption(f"Converted contract price: ${contract_price:,.2f}/MWh.")
                if wesm_pricing_enabled:
                    st.caption(
                        "WESM pricing uses the hourly profile (upload or bundled default) for both "
                        "shortfall costs and surplus revenue."
                    )

            financing_col1, financing_col2, financing_col3 = st.columns(3)
            with financing_col1:
                debt_equity_ratio = st.number_input(
                    "Debt/Equity ratio (D/E)",
                    min_value=0.0,
                    value=DEFAULT_DEBT_EQUITY_RATIO,
                    step=0.1,
                    help=(
                        "Debt divided by equity; 1.0 implies 50% debt and 50% equity. "
                        f"Default: {DEFAULT_DEBT_EQUITY_RATIO:.1f} D/E."
                    ),
                    key="inputs_debt_equity_ratio",
                )
                debt_ratio = debt_equity_ratio / (1.0 + debt_equity_ratio) if debt_equity_ratio > 0 else 0.0
                st.caption(f"Implied debt share of capital: {debt_ratio * 100:.1f}%.")
            with financing_col2:
                cost_of_debt_pct = st.number_input(
                    "Cost of debt (%)",
                    min_value=0.0,
                    max_value=30.0,
                    value=DEFAULT_COST_OF_DEBT_PCT,
                    step=0.1,
                    help=f"Annual interest rate applied to the debt balance. Default: {DEFAULT_COST_OF_DEBT_PCT:.1f}%.",
                    key="inputs_cost_of_debt_pct",
                )
            with financing_col3:
                tenor_years = st.number_input(
                    "Debt tenor (years)",
                    min_value=1,
                    value=DEFAULT_TENOR_YEARS,
                    step=1,
                    help=f"Years over which debt is amortized using level payments. Default: {DEFAULT_TENOR_YEARS}.",
                    key="inputs_tenor_years",
                )

            variable_col1, variable_col2 = st.columns(2)
            with variable_col1:
                st.markdown("**Variable OPEX overrides**")
                st.caption(
                    "Use the schedule controls to override the % of CAPEX or PHP/kWh inputs above. "
                    "Amounts are in USD and are treated as nominal per-year values."
                )
                variable_opex_usd_per_mwh: Optional[float] = None
            with variable_col2:
                variable_schedule_choice = st.radio(
                    "Variable expense schedule",
                    options=["None", "Periodic", "Custom"],
                    horizontal=True,
                    help="Custom or periodic schedules override per-kWh and fixed OPEX assumptions.",
                    key="inputs_variable_schedule_choice",
                )
                variable_opex_schedule_usd: Optional[Tuple[float, ...]] = None
                periodic_variable_opex_usd: Optional[float] = None
                periodic_variable_opex_interval_years: Optional[int] = None
                if variable_schedule_choice == "Periodic":
                    periodic_variable_opex_usd = st.number_input(
                        "Variable expense when periodic (USD)",
                        min_value=0.0,
                        value=0.0,
                        step=10_000.0,
                        help="Amount applied on the selected cadence (year 1, then every N years).",
                        key="inputs_periodic_variable_opex_usd",
                    )
                    periodic_variable_opex_interval_years = st.number_input(
                        "Cadence (years)",
                        min_value=1,
                        value=5,
                        step=1,
                        key="inputs_periodic_variable_opex_interval_years",
                    )
                    if periodic_variable_opex_usd <= 0:
                        periodic_variable_opex_usd = None
                elif variable_schedule_choice == "Custom":
                    custom_variable_text = st.text_area(
                        "Custom variable expenses (USD/year)",
                        placeholder="e.g., 250000, 275000, 300000",
                        help=("Comma or newline separated values applied per project year. Length must match the simulation horizon."),
                        key="inputs_variable_opex_custom_text",
                    )
                    st.caption(f"Use commas or newlines between amounts; provide one value per project year ({years} entries).")
                    if custom_variable_text.strip():
                        try:
                            variable_opex_schedule_usd = tuple(
                                parse_numeric_series("Variable expense schedule", custom_variable_text)
                            )
                        except ValueError as exc:
                            error_message = str(exc)
                            st.error(error_message)
                            validation_errors.append(error_message)
                            variable_opex_schedule_usd = None

            econ_inputs = EconomicInputs(
                capex_musd=capex_musd,
                capex_usd_per_kwh=capex_usd_per_kwh if capex_mode == "USD/kWh (BOL)" else None,
                capex_total_usd=None,
                bess_bol_kwh=bess_bol_kwh if capex_mode == "USD/kWh (BOL)" else None,
                pv_capex_musd=pv_capex_musd,
                fixed_opex_pct_of_capex=fixed_opex_pct,
                fixed_opex_musd=fixed_opex_musd,
                opex_php_per_kwh=opex_php_per_kwh,
                inflation_rate=inflation_pct / 100.0,
                discount_rate=discount_rate,
                variable_opex_usd_per_mwh=variable_opex_usd_per_mwh,
                variable_opex_schedule_usd=variable_opex_schedule_usd,
                periodic_variable_opex_usd=periodic_variable_opex_usd,
                periodic_variable_opex_interval_years=periodic_variable_opex_interval_years,
                forex_rate_php_per_usd=forex_rate_php_per_usd,
                devex_cost_php=devex_cost_php,
                include_devex_year0=include_devex_year0,
                debt_ratio=debt_ratio,
                cost_of_debt=cost_of_debt_pct / 100.0,
                tenor_years=int(tenor_years),
                wacc=wacc_pct / 100.0,
            )
            price_inputs = PriceInputs(
                contract_price_usd_per_mwh=contract_price,
                escalate_with_inflation=escalate_prices,
                apply_wesm_to_shortfall=wesm_pricing_enabled,
                sell_to_wesm=sell_to_wesm if wesm_pricing_enabled else False,
            )

    dispatch_validation = validate_dispatch_windows(
        discharge_windows_text,
        charge_windows_text,
        require_discharge_windows=dispatch_mode == DISPATCH_MODE_FIXED,
    )
    schedule_validation = validate_dispatch_schedule(dispatch_mode, hourly_schedule_df, period_schedule_df)
    validation_warnings.extend(dispatch_validation.warnings)
    validation_warnings.extend(schedule_validation.warnings)
    manual_errors, manual_details = validate_manual_augmentation_schedule(
        aug_mode, manual_schedule_entries, manual_schedule_errors
    )
    validation_errors.extend(manual_errors)
    validation_errors.extend(schedule_validation.errors)
    validation_details.extend(manual_details)
    validation_details.extend(dispatch_validation.details)
    validation_details.extend(schedule_validation.details)

    for msg in validation_warnings:
        st.warning(msg)
    for msg in validation_errors:
        st.error(msg)

    cfg = SimConfig(
        years=int(years),
        pv_deg_rate=float(pv_deg),
        pv_availability=float(pv_avail),
        bess_availability=float(bess_avail),
        rte_roundtrip=float(rte),
        use_split_rte=bool(use_split_rte),
        charge_efficiency=float(charge_eff) if use_split_rte else None,
        discharge_efficiency=float(discharge_eff) if use_split_rte else None,
        soc_floor=float(soc_floor),
        soc_ceiling=float(soc_ceiling),
        initial_power_mw=float(init_power),
        initial_usable_mwh=float(init_energy),
        contracted_mw=float(contracted_mw),
        discharge_windows=dispatch_validation.discharge_windows,
        charge_windows_text=charge_windows_text,
        charge_windows=dispatch_validation.charge_windows,
        max_cycles_per_day_cap=1.2,
        calendar_fade_rate=float(cal_fade),
        use_calendar_exp_model=True,
        augmentation=aug_mode,
        aug_trigger_type=aug_trigger_type,
        aug_threshold_margin=float(aug_thr_margin),
        aug_topup_margin=float(aug_topup),
        aug_soh_trigger_pct=float(aug_soh_trig),
        aug_soh_add_frac_initial=float(aug_soh_add),
        aug_periodic_every_years=int(aug_every),
        aug_periodic_add_frac_of_bol=float(aug_frac),
        aug_add_mode=aug_size_mode,
        aug_fixed_energy_mwh=float(aug_fixed_energy),
        aug_retire_old_cohort=bool(retire_enabled),
        aug_retire_soh_pct=float(retire_soh),
        aug_retire_replacement_mode=retire_replace_mode,
        aug_retire_replacement_pct_bol=float(retire_replace_pct),
        aug_retire_replacement_fixed_mwh=float(retire_replace_fixed_mwh),
        augmentation_schedule=list(manual_schedule_entries) if aug_mode == "Manual" else [],
    )
    cfg.contracted_mw_schedule = resolve_contracted_mw_schedule(schedule_validation.schedule_payload)

    inferred_step = infer_step_hours_from_pv(pv_df)
    if inferred_step is not None:
        cfg.step_hours = inferred_step

    duration_error = validate_pv_profile_duration(pv_df, cfg.step_hours)
    if duration_error:
        validation_errors.append(duration_error)
        validation_details.append(f"PV profile validation error: {duration_error}")

    run_cols = st.columns([2, 1])
    with run_cols[0]:
        run_submitted = st.button(
            "Run simulation",
            use_container_width=True,
            help="Click to compute results with the current inputs.",
        )
    with run_cols[1]:
        st.caption("Edit parameters freely, then run when ready.")

    is_valid = not validation_errors
    if validation_errors and run_submitted:
        st.info("Fix the validation errors above before re-running the simulation.")

    if debug_mode:
        recent_inputs: Dict[str, Any] = {
            "years": years,
            "initial_power_mw": init_power,
            "initial_usable_mwh": init_energy,
            "contracted_mw": contracted_mw,
            "discharge_windows_text": discharge_windows_text,
            "charge_windows_text": charge_windows_text or "<blank>",
            "dispatch_mode": dispatch_mode,
            "dispatch_schedule": schedule_validation.schedule_payload,
            "augmentation_mode": aug_mode,
            "augmentation_schedule_rows": len(manual_schedule_entries),
            "use_split_rte": use_split_rte,
        }
        with st.expander("Debug: validation details and recent inputs", expanded=False):
            st.write("Validation details:")
            if validation_details:
                st.markdown("\n".join(f"- {msg}" for msg in validation_details))
            else:
                st.caption("No validation details available.")
            st.write("Recent inputs:")
            st.json(recent_inputs)

    return SimulationFormResult(
        config=cfg,
        econ_inputs=econ_inputs,
        price_inputs=price_inputs,
        run_economics=run_economics,
        dod_override=dod_override,
        run_submitted=run_submitted,
        discharge_windows_text=discharge_windows_text,
        charge_windows_text=charge_windows_text,
        dispatch_mode=dispatch_mode,
        dispatch_schedule=schedule_validation.schedule_payload,
        validation_errors=validation_errors,
        validation_warnings=validation_warnings,
        validation_details=validation_details,
        debug_mode=debug_mode,
        is_valid=is_valid,
    )
