"""Helpers for contracted capacity schedules."""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from services.simulation_core import SimConfig, Window


HOURS_PER_DAY = 24


def normalize_hourly_schedule(values: Optional[Sequence[Any]]) -> Optional[List[float]]:
    """Return a sanitized 24-value hourly schedule or None if invalid."""

    if not values:
        return None
    if len(values) != HOURS_PER_DAY:
        return None

    normalized: List[float] = []
    for value in values:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        try:
            normalized.append(float(value))
        except (TypeError, ValueError):
            return None
    return normalized


def parse_hhmm_time(token: str) -> float:
    """Parse an HH:MM time token into a fractional hour (0-24)."""

    parts = token.split(":")
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


def _intervals(window: Window) -> List[Tuple[float, float]]:
    if window.start <= window.end:
        return [(window.start, window.end)]
    return [(window.start, float(HOURS_PER_DAY)), (0.0, window.end)]


def find_window_overlaps(windows: Sequence[Window]) -> List[str]:
    """Return overlap error strings for a list of windows."""

    errors: List[str] = []
    for idx, left in enumerate(windows):
        for right in windows[idx + 1 :]:
            for left_interval in _intervals(left):
                for right_interval in _intervals(right):
                    if max(left_interval[0], right_interval[0]) < min(left_interval[1], right_interval[1]):
                        left_label = left.source or f"{left.start:.2f}-{left.end:.2f}"
                        right_label = right.source or f"{right.start:.2f}-{right.end:.2f}"
                        errors.append(f"Overlapping period rows '{left_label}' and '{right_label}' detected.")
                        break
                else:
                    continue
                break
    return errors


def build_hourly_schedule_from_period_table(period_table: Optional[Sequence[Dict[str, Any]]]) -> Optional[List[float]]:
    """Convert period-table rows into an hourly MW schedule (24 values)."""

    if not period_table:
        return None

    hourly = [0.0 for _ in range(HOURS_PER_DAY)]
    for row in period_table:
        if not isinstance(row, dict):
            raise ValueError("Period table rows must be objects with start_time, end_time, and capacity_mw.")
        start_text = row.get("start_time")
        end_text = row.get("end_time")
        capacity_value = row.get("capacity_mw")
        if start_text is None or end_text is None or capacity_value is None:
            raise ValueError("Period table rows must include start_time, end_time, and capacity_mw.")

        start = parse_hhmm_time(str(start_text))
        end = parse_hhmm_time(str(end_text))
        capacity = float(capacity_value)
        if capacity < 0:
            raise ValueError("Period table capacity values must be non-negative.")

        segments = [(start, end)] if start <= end else [(start, float(HOURS_PER_DAY)), (0.0, end)]
        for seg_start, seg_end in segments:
            start_hour = int(math.floor(seg_start))
            end_hour = int(math.ceil(seg_end))
            for hour in range(start_hour, end_hour):
                hour_start = float(hour)
                hour_end = float(hour + 1)
                overlap = max(0.0, min(seg_end, hour_end) - max(seg_start, hour_start))
                if overlap > 0:
                    hourly[hour % HOURS_PER_DAY] += overlap * capacity

    return hourly


def resolve_contracted_mw_schedule(dispatch_schedule: Optional[Dict[str, Any]]) -> Optional[List[float]]:
    """Return a 24-hour schedule from a dispatch payload if available."""

    if not isinstance(dispatch_schedule, dict):
        return None

    hourly = normalize_hourly_schedule(dispatch_schedule.get("hourly_mw"))
    if hourly:
        return hourly

    period_table = dispatch_schedule.get("period_table")
    if period_table:
        return build_hourly_schedule_from_period_table(period_table)

    return None


def build_contracted_mw_profile(cfg: SimConfig) -> List[float]:
    """Return the contracted MW schedule by hour, falling back to discharge windows."""

    schedule = normalize_hourly_schedule(cfg.contracted_mw_schedule)
    if schedule:
        return schedule

    return [
        cfg.contracted_mw if any(window.contains(hour) for window in cfg.discharge_windows) else 0.0
        for hour in range(HOURS_PER_DAY)
    ]


def parse_schedule_cell(value: Any) -> Optional[List[float]]:
    """Parse a schedule cell value into a list of hourly MW values."""

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None

    raw_value: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            raw_value = json.loads(text)
        except json.JSONDecodeError:
            raw_value = [item.strip() for item in text.replace(";", ",").split(",") if item.strip()]

    if isinstance(raw_value, (list, tuple)):
        return normalize_hourly_schedule(raw_value)

    return None


def parse_period_table_cell(value: Any) -> Optional[List[Dict[str, Any]]]:
    """Parse a period table cell into a list of period dictionaries."""

    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None

    raw_value: Any = value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            raw_value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Period table must be a JSON array of rows.") from exc

    if isinstance(raw_value, list):
        return [row for row in raw_value if isinstance(row, dict)]

    return None
