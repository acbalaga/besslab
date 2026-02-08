from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


@dataclass
class Window:
    start: float  # hour-of-day, inclusive
    end: float  # hour-of-day, exclusive
    source: Optional[str] = None  # original user-entered string for warnings

    def contains(self, hod: float) -> bool:
        if self.start <= self.end:
            return self.start <= hod < self.end
        return hod >= self.start or hod < self.end


@dataclass
class AugmentationScheduleEntry:
    """Explicit augmentation action for a given project year."""

    year: int
    basis: str  # Percent of BOL, percent of current, fixed energy, or fixed power
    value: float

    def to_dict(self) -> Dict[str, Any]:
        return {"year": self.year, "basis": self.basis, "value": self.value}


AUGMENTATION_SCHEDULE_BASIS = [
    "Percent of BOL energy",
    "Percent of current BOL-ref energy",
    "Fixed energy (MWh)",
    "Fixed power (MW)",
]


def parse_windows(text: str) -> Tuple[List[Window], List[str]]:
    """Parse comma-separated HH:MM-HH:MM windows into `Window` objects.

    Returns the parsed windows and any warning messages instead of emitting
    UI side-effects so callers can decide how to surface validation issues.
    """

    if not text.strip():
        return [], []

    warnings: List[str] = []
    windows: List[Window] = []

    def _parse_time(token: str) -> float:
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

    for part in [p.strip() for p in text.split(",") if p.strip()]:
        try:
            start_text, end_text = part.split("-")
            h1 = _parse_time(start_text)
            h2 = _parse_time(end_text)
            if not (0.0 <= h1 < 24.0 and 0.0 <= h2 < 24.0):
                warnings.append(f"Invalid window hour in '{part}' (00:00-23:59). Skipped.")
                continue
            windows.append(Window(h1, h2, source=part))
        except Exception:
            warnings.append(f"Could not parse window '{part}'. Use 'HH:MM-HH:MM'. Skipped.")

    return windows, warnings


def infer_step_hours_from_pv(pv_df: pd.DataFrame, timestamp_col: str = "timestamp") -> Optional[float]:
    """Infer the timestep (hours) from PV timestamps when present."""

    if timestamp_col not in pv_df.columns:
        return None

    timestamps = pd.to_datetime(pv_df[timestamp_col], errors="coerce").dropna().sort_values()
    if len(timestamps) < 2:
        return None

    inferred = pd.infer_freq(timestamps)
    freq_td = None
    if inferred is not None:
        try:
            offset = to_offset(inferred)
            freq_td = pd.Timedelta(offset.nanos)
        except ValueError:
            freq_td = pd.Timedelta(inferred)
    if freq_td is None:
        diffs = timestamps.diff().dropna()
        if diffs.empty:
            return None
        freq_td = diffs.median()

    if freq_td <= pd.Timedelta(0):
        return None

    return float(freq_td / pd.Timedelta(hours=1))


def validate_pv_profile_duration(
    pv_df: pd.DataFrame, step_hours: float, timestamp_col: str = "timestamp"
) -> Optional[str]:
    """Return an error message if PV rows do not cover one year at the given step."""

    if step_hours <= 0:
        return "Timestep (step_hours) must be positive."

    expected_365 = int(round(24.0 * 365.0 / step_hours))
    expected_366 = int(round(24.0 * 366.0 / step_hours))
    allowed_counts = {expected_365, expected_366}

    if timestamp_col in pv_df.columns:
        timestamps = pd.to_datetime(pv_df[timestamp_col], errors="coerce").dropna().sort_values()
        if len(timestamps) >= 2:
            freq_td = pd.Timedelta(hours=step_hours)
            expected_range = pd.date_range(timestamps.min(), timestamps.max(), freq=freq_td)
            allowed_counts.add(len(expected_range))

    if len(pv_df) not in allowed_counts:
        return (
            "PV profile should represent one year. "
            f"Received {len(pv_df):,} rows, expected {expected_365:,} (365-day) or "
            f"{expected_366:,} (leap-year) rows for a {step_hours}-hour timestep."
        )

    return None


def infer_dod_bucket(daily_dis_mwh: np.ndarray, usable_mwh_available: float) -> int:
    """Infer an effective DoD bucket from daily discharge energy."""

    if usable_mwh_available <= 0:
        return 100
    if len(daily_dis_mwh) == 0:
        return 10
    med = float(np.median(daily_dis_mwh))
    if med <= 0:
        return 10
    r = med / max(1e-9, usable_mwh_available)
    if r >= 0.9:
        return 100
    if r >= 0.8:
        return 80
    if r >= 0.4:
        return 40
    if r >= 0.2:
        return 20
    return 10


def cycle_retention_lookup(cycle_df: pd.DataFrame, dod_key: int, cumulative_cycles: float) -> float:
    c_col = f"DoD{dod_key}_Cycles"
    r_col = f"DoD{dod_key}_Ret(%)"
    if c_col not in cycle_df.columns or r_col not in cycle_df.columns:
        return 1.0
    df = cycle_df[[c_col, r_col]].dropna().sort_values(c_col)
    x = df[c_col].to_numpy(float)
    y = df[r_col].to_numpy(float)
    if len(x) == 0:
        return 1.0
    if cumulative_cycles <= x[0]:
        ret = y[0]
    elif cumulative_cycles >= x[-1]:
        ret = y[-1]
    else:
        ret = np.interp(cumulative_cycles, x, y)
    return max(0.0, float(ret)) / 100.0


@dataclass
class SimConfig:
    years: int = 20
    step_hours: float = 1.0
    pv_deg_rate: float = 0.006
    pv_availability: float = 0.98
    bess_availability: float = 0.99
    rte_roundtrip: float = 0.88  # single (η_rt)
    use_split_rte: bool = False
    charge_efficiency: Optional[float] = None
    discharge_efficiency: Optional[float] = None
    soc_floor: float = 0.10
    soc_ceiling: float = 0.90
    initial_power_mw: float = 30.0
    initial_usable_mwh: float = 120.0
    contracted_mw: float = 30.0
    contracted_mw_schedule: Optional[List[float]] = None
    contracted_mw_profile: Optional[List[float]] = None
    discharge_windows: List[Window] = field(default_factory=lambda: [Window(10, 14), Window(18, 22)])
    charge_windows_text: str = ""
    charge_windows: List[Window] = field(default_factory=list)
    max_cycles_per_day_cap: float = 1.2
    calendar_fade_rate: float = 0.01
    use_calendar_exp_model: bool = True
    # Augmentation knobs
    augmentation: str = "None"  # 'None'|'Threshold'|'Periodic'
    aug_trigger_type: str = "Capability"  # 'Capability'|'SOH'
    aug_threshold_margin: float = 0.00  # capability mode
    aug_topup_margin: float = 0.05  # capability mode
    aug_soh_trigger_pct: float = 0.80  # SOH mode (e.g., 0.80 = 80%)
    aug_soh_add_frac_initial: float = 0.10  # SOH mode: add % of initial BOL energy
    aug_periodic_every_years: int = 5
    aug_periodic_add_frac_of_bol: float = 0.10
    aug_add_mode: str = "Percent"  # 'Percent'|'Fixed'
    aug_fixed_energy_mwh: float = 0.0  # fixed augmentation size when aug_add_mode='Fixed'
    aug_retire_old_cohort: bool = False
    aug_retire_soh_pct: float = 0.60
    aug_retire_replacement_mode: str = "None"  # 'None'|'Percent'|'Fixed'
    aug_retire_replacement_pct_bol: float = 0.0  # fractional, based on initial BOL energy
    aug_retire_replacement_fixed_mwh: float = 0.0  # fixed BOL energy replacement (MWh)
    augmentation_schedule: List[AugmentationScheduleEntry] = field(default_factory=list)


@dataclass
class YearResult:
    year_index: int
    expected_firm_mwh: float
    delivered_firm_mwh: float
    shortfall_mwh: float
    breach_days: int
    charge_mwh: float
    discharge_mwh: float
    available_pv_mwh: float
    pv_to_contract_mwh: float
    bess_to_contract_mwh: float
    avg_rte: float
    eq_cycles: float
    cum_cycles: float
    soh_cycle: float
    soh_calendar: float
    soh_total: float
    eoy_usable_mwh: float
    eoy_power_mw: float
    pv_curtailed_mwh: float
    flags: Dict[str, int]


@dataclass
class HourlyLog:
    hod: np.ndarray
    pv_mw: np.ndarray
    pv_to_contract_mw: np.ndarray
    bess_to_contract_mw: np.ndarray
    delivered_mw: np.ndarray
    shortfall_mw: np.ndarray
    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soc_mwh: np.ndarray
    timestamp: Optional[np.ndarray] = None


@dataclass
class MonthResult:
    year_index: int
    month_index: int
    month_label: str
    expected_firm_mwh: float
    delivered_firm_mwh: float
    shortfall_mwh: float
    breach_days: int
    charge_mwh: float
    discharge_mwh: float
    available_pv_mwh: float
    pv_to_contract_mwh: float
    bess_to_contract_mwh: float
    avg_rte: float
    eq_cycles: float
    cum_cycles: float
    soh_cycle: float
    soh_calendar: float
    soh_total: float
    eom_usable_mwh: float
    eom_power_mw: float
    pv_curtailed_mwh: float
    flags: Dict[str, int]


@dataclass
class SimulationOutput:
    cfg: SimConfig
    discharge_hours_per_day: float
    results: List[YearResult]
    monthly_results: List[MonthResult]
    first_year_logs: Optional[HourlyLog]
    final_year_logs: Optional[HourlyLog]
    hod_count: np.ndarray
    hod_sum_pv: np.ndarray
    hod_sum_pv_resource: np.ndarray
    hod_sum_bess: np.ndarray
    hod_sum_charge: np.ndarray
    augmentation_energy_added_mwh: List[float]
    augmentation_retired_energy_mwh: List[float]
    augmentation_events: int
    hourly_logs_by_year: Dict[int, HourlyLog] = field(default_factory=dict)


def resolve_efficiencies(cfg: SimConfig) -> Tuple[float, float, float]:
    """Return (charge, discharge, roundtrip) efficiencies with consistent bounds."""

    def _bound(value: float) -> float:
        return max(0.05, min(value, 0.9999))

    if cfg.use_split_rte and cfg.charge_efficiency is not None and cfg.discharge_efficiency is not None:
        eta_ch = _bound(cfg.charge_efficiency)
        eta_dis = _bound(cfg.discharge_efficiency)
        eta_rt = _bound(eta_ch * eta_dis)
        return eta_ch, eta_dis, eta_rt

    eta_rt = _bound(cfg.rte_roundtrip)
    eta_ch = eta_rt ** 0.5
    eta_dis = eta_rt ** 0.5
    return eta_ch, eta_dis, eta_rt


@dataclass
class SimulationSummary:
    compliance: float
    bess_share_of_firm: float
    charge_discharge_ratio: float
    pv_capture_ratio: float
    discharge_capacity_factor: float
    total_project_generation_mwh: float
    bess_generation_mwh: float
    pv_generation_mwh: float
    pv_excess_mwh: float
    bess_losses_mwh: float
    total_shortfall_mwh: float
    avg_eq_cycles_per_year: float
    cap_ratio_final: float


@dataclass
class BatteryCohort:
    """Represents a tranche of capacity with its own degradation history."""

    energy_mwh_bol: float
    start_year: int
    cum_cycles: float = 0.0

    def age_years(self, years_elapsed: float) -> float:
        """Return cohort age in years at a given simulation timestamp."""
        return max(0.0, years_elapsed - self.start_year)


@dataclass
class SimState:
    pv_df: pd.DataFrame
    cycle_df: pd.DataFrame
    cfg: SimConfig
    current_power_mw: float
    current_usable_mwh_bolref: float
    # reference for keeping original C-hours when augmenting
    initial_bol_energy_mwh: float
    initial_bol_power_mw: float
    cum_cycles: float = 0.0
    cohorts: List[BatteryCohort] = field(default_factory=list)
    last_dod_key: Optional[int] = None


def calc_calendar_soh(year_idx: int, rate: float, exp_model: bool) -> float:
    return max(0.0, (1.0 - rate) ** year_idx) if exp_model else max(0.0, 1.0 - rate * year_idx)


def calc_calendar_soh_fraction(years_elapsed: float, rate: float, exp_model: bool) -> float:
    return max(0.0, (1.0 - rate) ** years_elapsed) if exp_model else max(0.0, 1.0 - rate * years_elapsed)


def compute_fleet_soh(
    cohorts: List[BatteryCohort],
    cycle_df: pd.DataFrame,
    dod_key: int,
    years_elapsed: float,
    calendar_rate: float,
    use_calendar_exp_model: bool,
    cycles_at_point: Optional[List[float]] = None,
) -> Tuple[float, float, float]:
    """Return energy-weighted (cycle, calendar, total) SOH for the fleet."""

    total_energy = sum(c.energy_mwh_bol for c in cohorts)
    if total_energy <= 0:
        return 1.0, 1.0, 1.0

    if cycles_at_point is None:
        cycles_at_point = [c.cum_cycles for c in cohorts]

    weighted_cycle = 0.0
    weighted_calendar = 0.0
    weighted_total = 0.0

    for cohort, cycles in zip(cohorts, cycles_at_point):
        cohort_cycle_soh = cycle_retention_lookup(cycle_df, dod_key, cycles)
        cohort_calendar_soh = calc_calendar_soh_fraction(
            cohort.age_years(years_elapsed), calendar_rate, use_calendar_exp_model
        )
        cohort_total_soh = cohort_cycle_soh * cohort_calendar_soh
        weighted_cycle += cohort_cycle_soh * cohort.energy_mwh_bol
        weighted_calendar += cohort_calendar_soh * cohort.energy_mwh_bol
        weighted_total += cohort_total_soh * cohort.energy_mwh_bol

    return (
        weighted_cycle / total_energy,
        weighted_calendar / total_energy,
        weighted_total / total_energy,
    )


def compute_cohort_total_soh(
    cohort: BatteryCohort,
    cycle_df: pd.DataFrame,
    dod_key: int,
    years_elapsed: float,
    calendar_rate: float,
    use_calendar_exp_model: bool,
) -> float:
    """Return total SOH (cycle × calendar) for a single cohort."""

    cycle_soh = cycle_retention_lookup(cycle_df, dod_key, cohort.cum_cycles)
    calendar_soh = calc_calendar_soh_fraction(
        cohort.age_years(years_elapsed), calendar_rate, use_calendar_exp_model
    )
    return max(0.0, cycle_soh * calendar_soh)


def in_any_window(hod: int, windows: List[Window]) -> bool:
    return any(w.contains(hod) for w in windows)


def _normalize_contracted_mw_schedule(cfg: SimConfig) -> Optional[np.ndarray]:
    """Return a 24-value hourly schedule if provided; otherwise None.

    A schedule must include one value per hour-of-day (0-23). Any other
    length is treated as partial coverage and falls back to discharge windows.
    """

    if not cfg.contracted_mw_schedule:
        return None

    schedule = np.asarray(cfg.contracted_mw_schedule, dtype=float)
    if schedule.size != 24:
        return None

    return np.nan_to_num(schedule, nan=0.0)


def _normalize_contracted_mw_profile(cfg: SimConfig, expected_length: int) -> Optional[np.ndarray]:
    """Return a per-timestep contracted MW profile aligned to the PV profile length."""

    if not cfg.contracted_mw_profile:
        return None

    profile = np.asarray(cfg.contracted_mw_profile, dtype=float)
    if profile.size != expected_length:
        if profile.size > 0 and expected_length % profile.size == 0:
            profile = np.tile(profile, int(expected_length / profile.size))
        else:
            return None

    return np.nan_to_num(profile, nan=0.0)


def _target_mw_from_schedule(hod: float, dt: float, schedule_mw: np.ndarray) -> float:
    """Map an hourly schedule to a timestep target using overlap weighting.

    If a timestep spans multiple hours (dt != 1), the target is the overlap-
    weighted average of the hourly buckets. Wrap-around across midnight is
    handled by continuing to index the 24-hour schedule modulo 24.
    """

    if dt <= 0:
        return 0.0

    start = hod % 24.0
    end = start + dt
    total = 0.0
    hour = int(np.floor(start))
    current = start

    while current < end - 1e-9:
        next_hour = min(end, hour + 1)
        overlap = next_hour - current
        total += overlap * schedule_mw[hour % 24]
        current = next_hour
        hour += 1

    return total / dt


def _daily_target_mwh(cfg: SimConfig, discharge_hours_per_day: float) -> float:
    """Return daily target energy, using schedule when available."""

    schedule = _normalize_contracted_mw_schedule(cfg)
    if schedule is None:
        profile = None
        if cfg.contracted_mw_profile:
            profile = np.asarray(cfg.contracted_mw_profile, dtype=float)
        if profile is None or profile.size == 0:
            return cfg.contracted_mw * discharge_hours_per_day
        total_hours = profile.size * cfg.step_hours
        days = max(total_hours / 24.0, 1e-9)
        return float(np.nan_to_num(profile, nan=0.0).sum() * cfg.step_hours / days)

    return float(schedule.sum())


def simulate_year(state: SimState, year_idx: int, dod_key: Optional[int], need_logs: bool = False) -> Tuple[YearResult, HourlyLog, List[MonthResult]]:
    cfg = state.cfg
    dt = cfg.step_hours

    pv_scale = (1.0 - cfg.pv_deg_rate) ** (year_idx - 1)
    pv_mw = state.pv_df["pv_mw"].to_numpy(float) * pv_scale * cfg.pv_availability

    pow_cap_mw = state.current_power_mw * cfg.bess_availability

    eta_ch, eta_dis, _ = resolve_efficiencies(cfg)

    dis_windows = cfg.discharge_windows
    ch_windows = cfg.charge_windows
    schedule_mw = _normalize_contracted_mw_schedule(cfg)

    dod_for_lookup = dod_key if dod_key else 100
    soh_cycle_start, soh_calendar_start, soh_total_start = compute_fleet_soh(
        state.cohorts,
        state.cycle_df,
        dod_for_lookup,
        years_elapsed=max(year_idx - 1, 0),
        calendar_rate=cfg.calendar_fade_rate,
        use_calendar_exp_model=cfg.use_calendar_exp_model,
    )
    usable_mwh_start = state.current_usable_mwh_bolref * soh_total_start * cfg.bess_availability

    soc_mwh = usable_mwh_start * 0.5
    soc_min = usable_mwh_start * cfg.soc_floor
    soc_max = usable_mwh_start * cfg.soc_ceiling

    n_hours = len(pv_mw)
    profile_mw = _normalize_contracted_mw_profile(cfg, n_hours)
    if "timestamp" in state.pv_df.columns:
        calendar_index = pd.to_datetime(state.pv_df["timestamp"], errors="coerce")
    else:
        calendar_index = pd.date_range("2020-01-01", periods=n_hours, freq=pd.Timedelta(hours=dt))
    calendar_index = pd.DatetimeIndex(calendar_index)
    if calendar_index.isna().any():
        raise ValueError("PV timestamps contain invalid entries after cleaning.")

    day_index = ((calendar_index.normalize() - calendar_index[0].normalize()) / pd.Timedelta("1D")).astype(int)
    month_index = calendar_index.month - 1
    daily_dis_mwh = np.zeros(day_index.max() + 1)
    hod = (calendar_index.hour + calendar_index.minute / 60.0 + calendar_index.second / 3600.0).to_numpy()

    pv_to_contract_mw_log = np.zeros(n_hours)
    bess_to_contract_mw_log = np.zeros(n_hours)
    delivered_mw_log = np.zeros(n_hours)
    shortfall_mw_log = np.zeros(n_hours)
    charge_mw_log = np.zeros(n_hours)
    discharge_mw_log = np.zeros(n_hours)
    soc_log = np.zeros(n_hours)
    shortfall_day_flags = np.zeros(day_index.max() + 1, dtype=bool)

    month_expected = np.zeros(12)
    month_delivered = np.zeros(12)
    month_shortfall = np.zeros(12)
    month_charge = np.zeros(12)
    month_discharge = np.zeros(12)
    month_pv_available = np.zeros(12)
    month_pv_contract = np.zeros(12)
    month_bess_contract = np.zeros(12)
    month_pv_curtailed = np.zeros(12)
    month_flag_shortfall_hours = np.zeros(12, dtype=int)
    month_flag_soc_floor_hits = np.zeros(12, dtype=int)
    month_flag_soc_ceiling_hits = np.zeros(12, dtype=int)

    expected_firm_mwh = (
        charged_mwh
    ) = (
        discharged_mwh
    ) = (
        pv_available_mwh
    ) = pv_to_contract_mwh = bess_to_contract_mwh = pv_curtailed_mwh = 0.0
    flag_shortfall_hours = flag_soc_floor_hits = flag_soc_ceiling_hits = 0

    for h in range(n_hours):
        is_ch = True if not ch_windows else in_any_window(int(hod[h]), ch_windows)
        pv_avail_mw = max(0.0, pv_mw[h])

        pv_available_mwh += pv_avail_mw * dt
        month_pv_available[month_index[h]] += pv_avail_mw * dt

        if profile_mw is not None:
            target_mw = float(profile_mw[h])
        elif schedule_mw is not None:
            # Schedule overrides discharge windows; partial coverage falls back to windows above.
            target_mw = _target_mw_from_schedule(hod[h], dt, schedule_mw)
        else:
            is_dis = in_any_window(int(hod[h]), dis_windows)
            target_mw = cfg.contracted_mw if is_dis else 0.0
        expected_firm_mwh += target_mw * dt

        pv_to_contract_mw = min(pv_avail_mw, target_mw)
        pv_avail_after_contract = pv_avail_mw - pv_to_contract_mw

        residual_mw = max(0.0, target_mw - pv_to_contract_mw)
        dis_mw = min(residual_mw, pow_cap_mw)
        if dis_mw > 0:
            e_req = dis_mw * dt / max(1e-9, eta_dis)
            e_can = max(0.0, (soc_mwh - soc_min))
            if e_req > e_can:
                delivered = e_can * eta_dis / dt
                if delivered + pv_to_contract_mw < target_mw - 1e-9:
                    flag_shortfall_hours += 1
                    month_flag_shortfall_hours[month_index[h]] += 1
                    shortfall_day_flags[day_index[h]] = True
                dis_mw = delivered
                e_req = e_can
            soc_mwh -= e_req
            discharged_mwh += dis_mw * dt
            daily_dis_mwh[day_index[h]] += dis_mw * dt
            bess_to_contract_mwh += dis_mw * dt
            discharge_mw_log[h] = dis_mw
            bess_to_contract_mw_log[h] = dis_mw
            if abs(soc_mwh - soc_min) < 1e-6:
                flag_soc_floor_hits += 1
                month_flag_soc_floor_hits[month_index[h]] += 1

        pv_to_contract_mwh += pv_to_contract_mw * dt

        ch_cap = pow_cap_mw if is_ch else 0.0
        ch_mw = 0.0
        if ch_cap > 0 and pv_avail_after_contract > 0:
            e_room = max(0.0, soc_max - soc_mwh)
            p_soc_lim = e_room / max(1e-9, eta_ch * dt)
            ch_mw = min(ch_cap, pv_avail_after_contract, p_soc_lim)
            if ch_mw > 0:
                soc_mwh += ch_mw * dt * eta_ch
                charged_mwh += ch_mw * dt
                charge_mw_log[h] = ch_mw
                if abs(soc_mwh - soc_max) < 1e-6:
                    flag_soc_ceiling_hits += 1
                    month_flag_soc_ceiling_hits[month_index[h]] += 1

        pv_curtailed_mwh += max(0.0, pv_avail_after_contract - ch_mw) * dt
        pv_to_contract_mw_log[h] = pv_to_contract_mw
        soc_log[h] = soc_mwh

        month_expected[month_index[h]] += target_mw * dt
        delivered_hour = pv_to_contract_mw + dis_mw
        delivered_mw_log[h] = delivered_hour
        shortfall_mw_log[h] = max(0.0, target_mw - delivered_hour)
        month_delivered[month_index[h]] += delivered_hour * dt
        month_shortfall[month_index[h]] += max(0.0, target_mw - delivered_hour) * dt
        month_charge[month_index[h]] += ch_mw * dt
        month_discharge[month_index[h]] += dis_mw * dt
        month_pv_contract[month_index[h]] += pv_to_contract_mw * dt
        month_bess_contract[month_index[h]] += dis_mw * dt
        month_pv_curtailed[month_index[h]] += max(0.0, pv_avail_after_contract - ch_mw) * dt

    avg_rte = (discharged_mwh / charged_mwh) if charged_mwh > 0 else np.nan

    dod_key_eff = dod_key if dod_key is not None else infer_dod_bucket(daily_dis_mwh, usable_mwh_start)
    state.last_dod_key = dod_key_eff
    dod_frac = {10: 0.10, 20: 0.20, 40: 0.40, 80: 0.80, 100: 1.00}[dod_key_eff]
    usable_for_cycles = max(1e-9, usable_mwh_start * dod_frac)
    eq_cycles_year = discharged_mwh / usable_for_cycles
    cum_cycles_increment = eq_cycles_year
    cum_cycles_new = state.cum_cycles + cum_cycles_increment

    cohort_cycles_eoy = [c.cum_cycles + cum_cycles_increment for c in state.cohorts]
    soh_cycle, soh_calendar, soh_total = compute_fleet_soh(
        state.cohorts,
        state.cycle_df,
        dod_key_eff,
        years_elapsed=year_idx,
        calendar_rate=cfg.calendar_fade_rate,
        use_calendar_exp_model=cfg.use_calendar_exp_model,
        cycles_at_point=cohort_cycles_eoy,
    )

    eoy_usable_mwh = state.current_usable_mwh_bolref * soh_total
    eoy_power_mw = pow_cap_mw

    delivered_firm_mwh = pv_to_contract_mwh + bess_to_contract_mwh
    shortfall_mwh = max(0.0, expected_firm_mwh - delivered_firm_mwh)
    breach_days = int(shortfall_day_flags.sum())

    yr = YearResult(
        year_index=year_idx,
        expected_firm_mwh=expected_firm_mwh,
        delivered_firm_mwh=delivered_firm_mwh,
        shortfall_mwh=shortfall_mwh,
        breach_days=breach_days,
        charge_mwh=charged_mwh,
        discharge_mwh=discharged_mwh,
        available_pv_mwh=pv_available_mwh,
        pv_to_contract_mwh=pv_to_contract_mwh,
        bess_to_contract_mwh=bess_to_contract_mwh,
        avg_rte=float(avg_rte) if not np.isnan(avg_rte) else float("nan"),
        eq_cycles=float(eq_cycles_year),
        cum_cycles=float(cum_cycles_new),
        soh_cycle=float(soh_cycle),
        soh_calendar=float(soh_calendar),
        soh_total=float(soh_total),
        eoy_usable_mwh=float(eoy_usable_mwh),
        eoy_power_mw=float(eoy_power_mw),
        pv_curtailed_mwh=float(pv_curtailed_mwh),
        flags={
            "firm_shortfall_hours": int(flag_shortfall_hours),
            "soc_floor_hits": int(flag_soc_floor_hits),
            "soc_ceiling_hits": int(flag_soc_ceiling_hits),
        },
    )

    month_for_day = np.zeros(day_index.max() + 1, dtype=int)
    for d in range(day_index.max() + 1):
        idx = np.argmax(day_index == d)
        month_for_day[d] = month_index[idx]

    month_shortfall_days = [int(shortfall_day_flags[month_for_day == m].sum()) for m in range(12)]

    monthly_results: List[MonthResult] = []
    cum_cycles_running = state.cum_cycles
    cohort_cycles_running = [c.cum_cycles for c in state.cohorts]
    for m in range(12):
        eq_cycles_month = month_discharge[m] / usable_for_cycles
        cum_cycles_running += eq_cycles_month
        cohort_cycles_running = [c + eq_cycles_month for c in cohort_cycles_running]
        years_elapsed = year_idx - 1 + (m + 1) / 12.0
        soh_cycle_month, soh_calendar_month, soh_total_month = compute_fleet_soh(
            state.cohorts,
            state.cycle_df,
            dod_key_eff,
            years_elapsed=years_elapsed,
            calendar_rate=cfg.calendar_fade_rate,
            use_calendar_exp_model=cfg.use_calendar_exp_model,
            cycles_at_point=cohort_cycles_running,
        )
        monthly_results.append(
            MonthResult(
                year_index=year_idx,
                month_index=m + 1,
                month_label=calendar.month_name[m + 1],
                expected_firm_mwh=month_expected[m],
                delivered_firm_mwh=month_delivered[m],
                shortfall_mwh=month_shortfall[m],
                breach_days=month_shortfall_days[m],
                charge_mwh=month_charge[m],
                discharge_mwh=month_discharge[m],
                available_pv_mwh=month_pv_available[m],
                pv_to_contract_mwh=month_pv_contract[m],
                bess_to_contract_mwh=month_bess_contract[m],
                avg_rte=float(month_discharge[m] / month_charge[m]) if month_charge[m] > 0 else float("nan"),
                eq_cycles=float(eq_cycles_month),
                cum_cycles=float(cum_cycles_running),
                soh_cycle=float(soh_cycle_month),
                soh_calendar=float(soh_calendar_month),
                soh_total=float(soh_total_month),
                eom_usable_mwh=float(state.current_usable_mwh_bolref * soh_total_month),
                eom_power_mw=float(pow_cap_mw),
                pv_curtailed_mwh=float(month_pv_curtailed[m]),
                flags={
                    "firm_shortfall_hours": int(month_flag_shortfall_hours[m]),
                    "soc_floor_hits": int(month_flag_soc_floor_hits[m]),
                    "soc_ceiling_hits": int(month_flag_soc_ceiling_hits[m]),
                },
            )
        )

    for idx, cohort in enumerate(state.cohorts):
        cohort.cum_cycles = cohort_cycles_eoy[idx]

    state.cum_cycles = cum_cycles_new

    logs = HourlyLog(
        hod=hod,
        pv_mw=pv_mw,
        pv_to_contract_mw=pv_to_contract_mw_log,
        bess_to_contract_mw=bess_to_contract_mw_log,
        delivered_mw=delivered_mw_log,
        shortfall_mw=shortfall_mw_log,
        charge_mw=charge_mw_log,
        discharge_mw=discharge_mw_log,
        soc_mwh=soc_log,
        timestamp=calendar_index.to_numpy(),
    )
    return yr, logs, monthly_results


def _compute_retirement_replacement_energy(
    retired_energy_bol: float, state: SimState, cfg: SimConfig
) -> Tuple[float, float]:
    """Return replacement power/energy for retired cohorts (BOL basis)."""

    if retired_energy_bol <= 0:
        return 0.0, 0.0

    if cfg.aug_retire_replacement_mode == "Percent":
        add_energy_bol = state.initial_bol_energy_mwh * cfg.aug_retire_replacement_pct_bol
    elif cfg.aug_retire_replacement_mode == "Fixed":
        add_energy_bol = cfg.aug_retire_replacement_fixed_mwh
    else:
        return 0.0, 0.0

    add_energy_bol = max(0.0, float(add_energy_bol))
    if add_energy_bol <= 0:
        return 0.0, 0.0

    c_hours = max(1e-9, state.initial_bol_energy_mwh / max(1e-9, state.initial_bol_power_mw))
    add_power = add_energy_bol / c_hours
    return float(add_power), float(add_energy_bol)


def retire_cohorts_if_needed(state: SimState, cfg: SimConfig, year_idx: int) -> Tuple[float, float, float]:
    """Retire cohorts whose SOH is below the configured threshold.

    Returns a tuple of (retired_energy_bol, replacement_power_mw, replacement_energy_bol).
    """

    if not cfg.aug_retire_old_cohort:
        return 0.0, 0.0, 0.0

    dod_key = state.last_dod_key or 80
    c_hours = max(1e-9, state.initial_bol_energy_mwh / max(1e-9, state.initial_bol_power_mw))

    remaining_cohorts: List[BatteryCohort] = []
    retired_energy_bol = 0.0

    for cohort in state.cohorts:
        total_soh = compute_cohort_total_soh(
            cohort,
            state.cycle_df,
            dod_key,
            years_elapsed=year_idx,
            calendar_rate=cfg.calendar_fade_rate,
            use_calendar_exp_model=cfg.use_calendar_exp_model,
        )
        if total_soh <= cfg.aug_retire_soh_pct + 1e-9:
            retired_energy_bol += cohort.energy_mwh_bol
        else:
            remaining_cohorts.append(cohort)

    if retired_energy_bol <= 0:
        return 0.0, 0.0, 0.0

    state.cohorts = remaining_cohorts
    state.current_usable_mwh_bolref = max(0.0, state.current_usable_mwh_bolref - retired_energy_bol)
    state.current_power_mw = max(0.0, state.current_power_mw - retired_energy_bol / c_hours)
    add_power, add_energy_bol = _compute_retirement_replacement_energy(retired_energy_bol, state, cfg)
    return retired_energy_bol, add_power, add_energy_bol


def _find_manual_schedule_entry(schedule: List[AugmentationScheduleEntry], year_idx: int) -> Optional[AugmentationScheduleEntry]:
    for entry in schedule:
        if entry.year == year_idx:
            return entry
    return None


def _compute_manual_augmentation(entry: AugmentationScheduleEntry, state: SimState) -> Tuple[float, float]:
    c_hours = max(1e-9, state.initial_bol_energy_mwh / max(1e-9, state.initial_bol_power_mw))
    basis = entry.basis or AUGMENTATION_SCHEDULE_BASIS[0]
    value = max(0.0, float(entry.value))

    if basis == "Percent of BOL energy":
        add_energy_bol = state.initial_bol_energy_mwh * value / 100.0
        add_power = add_energy_bol / c_hours
    elif basis == "Percent of current BOL-ref energy":
        add_energy_bol = state.current_usable_mwh_bolref * value / 100.0
        add_power = add_energy_bol / c_hours
    elif basis == "Fixed power (MW)":
        add_power = value
        add_energy_bol = add_power * c_hours
    else:
        add_energy_bol = value
        add_power = add_energy_bol / c_hours

    return float(add_power), float(add_energy_bol)


def describe_schedule(entries: List[AugmentationScheduleEntry]) -> str:
    if not entries:
        return "None"

    parts = [f"Y{entry.year}: {entry.value:g} ({entry.basis})" for entry in sorted(entries, key=lambda e: e.year)]
    return "; ".join(parts)


def build_schedule_from_editor(df: pd.DataFrame, max_years: int) -> Tuple[List[AugmentationScheduleEntry], List[str]]:
    """Convert schedule table rows into validated augmentation entries."""

    if df is None or df.empty:
        return [], []

    entries: List[AugmentationScheduleEntry] = []
    errors: List[str] = []
    seen_years: set[int] = set()

    for idx, row in df.iterrows():
        year_val = row.get("Year")
        amount_val = row.get("Amount")
        basis_val = row.get("Basis") or AUGMENTATION_SCHEDULE_BASIS[0]

        if pd.isna(year_val) and pd.isna(amount_val):
            continue
        if pd.isna(year_val) or pd.isna(amount_val):
            errors.append(f"Row {idx + 1}: please provide both a Year and Amount.")
            continue

        try:
            year = int(year_val)
        except (TypeError, ValueError):
            errors.append(f"Row {idx + 1}: Year must be an integer.")
            continue

        if year < 1 or year > max_years:
            errors.append(f"Row {idx + 1}: Year must be between 1 and {max_years}.")
            continue
        if year in seen_years:
            errors.append(f"Duplicate year {year} detected; each year can only appear once.")
            continue

        seen_years.add(year)
        entries.append(AugmentationScheduleEntry(year=year, basis=str(basis_val), value=float(amount_val)))

    entries.sort(key=lambda e: e.year)
    return entries, errors


def apply_augmentation(state: SimState, cfg: SimConfig, yr: YearResult, discharge_hours_per_day: float) -> Tuple[float, float]:
    """Return (add_power_MW, add_energy_MWh at BOL)."""

    scheduled_entry = _find_manual_schedule_entry(cfg.augmentation_schedule, yr.year_index)
    if scheduled_entry is not None:
        return _compute_manual_augmentation(scheduled_entry, state)

    if cfg.augmentation == "None":
        return 0.0, 0.0

    if cfg.augmentation == "Threshold" and cfg.aug_trigger_type == "Capability":
        target_energy_per_day = _daily_target_mwh(cfg, discharge_hours_per_day)
        eoy_cap_per_day = min(yr.eoy_usable_mwh, yr.eoy_power_mw * discharge_hours_per_day)
        if eoy_cap_per_day + 1e-6 < target_energy_per_day * (1.0 - cfg.aug_threshold_margin):
            short_mwh = target_energy_per_day * (1.0 + cfg.aug_topup_margin) - eoy_cap_per_day
            add_energy_bol = max(0.0, short_mwh)
            if cfg.aug_add_mode == "Fixed" and cfg.aug_fixed_energy_mwh > 0:
                add_energy_bol = cfg.aug_fixed_energy_mwh
            c_hours = max(1e-9, state.initial_bol_energy_mwh / max(1e-9, state.initial_bol_power_mw))
            add_power = add_energy_bol / c_hours
            return float(add_power), float(add_energy_bol)
        return 0.0, 0.0

    if cfg.augmentation == "Threshold" and cfg.aug_trigger_type == "SOH":
        if yr.soh_total <= cfg.aug_soh_trigger_pct + 1e-9:
            add_energy_bol = cfg.aug_soh_add_frac_initial * state.initial_bol_energy_mwh
            if cfg.aug_add_mode == "Fixed" and cfg.aug_fixed_energy_mwh > 0:
                add_energy_bol = cfg.aug_fixed_energy_mwh
            c_hours = max(1e-9, state.initial_bol_energy_mwh / max(1e-9, state.initial_bol_power_mw))
            add_power = add_energy_bol / c_hours
            return float(add_power), float(add_energy_bol)
        return 0.0, 0.0

    if cfg.augmentation == "Periodic":
        if (yr.year_index % max(1, cfg.aug_periodic_every_years)) == 0:
            add_energy_bol = cfg.aug_periodic_add_frac_of_bol * state.current_usable_mwh_bolref
            if cfg.aug_add_mode == "Fixed" and cfg.aug_fixed_energy_mwh > 0:
                add_energy_bol = cfg.aug_fixed_energy_mwh
            c_hours = max(1e-9, state.initial_bol_energy_mwh / max(1e-9, state.initial_bol_power_mw))
            add_power = add_energy_bol / c_hours
            return float(add_power), float(add_energy_bol)
        return 0.0, 0.0

    return 0.0, 0.0


def simulate_project(cfg: SimConfig, pv_df: pd.DataFrame, cycle_df: pd.DataFrame, dod_override: str, need_logs: bool = True) -> SimulationOutput:
    schedule_mw = _normalize_contracted_mw_schedule(cfg)
    profile_mw = _normalize_contracted_mw_profile(cfg, len(pv_df))
    if not cfg.discharge_windows and schedule_mw is None and profile_mw is None:
        raise ValueError("Please provide at least one discharge window or an active dispatch schedule.")
    if profile_mw is not None and profile_mw.size > 0:
        total_hours = profile_mw.size * cfg.step_hours
        days = max(total_hours / 24.0, 1e-9)
        dis_hours_per_day = float(np.count_nonzero(profile_mw > 0.0) * cfg.step_hours / days)
    elif schedule_mw is not None:
        # Hours with non-zero schedule are treated as the active discharge period.
        dis_hours_per_day = float(np.count_nonzero(schedule_mw > 0.0))
    else:
        dis_hours_per_day = 0.0
        for w in cfg.discharge_windows:
            dis_hours_per_day += (w.end - w.start) if w.start <= w.end else (24 - w.start + w.end)

    state = SimState(
        pv_df=pv_df,
        cycle_df=cycle_df,
        cfg=cfg,
        current_power_mw=cfg.initial_power_mw,
        current_usable_mwh_bolref=cfg.initial_usable_mwh,
        initial_bol_energy_mwh=cfg.initial_usable_mwh,
        initial_bol_power_mw=cfg.initial_power_mw,
        cohorts=[BatteryCohort(energy_mwh_bol=cfg.initial_usable_mwh, start_year=0)],
    )

    results: List[YearResult] = []
    augmentation_energy_added: List[float] = [0.0 for _ in range(cfg.years)]
    augmentation_retired_energy: List[float] = [0.0 for _ in range(cfg.years)]
    monthly_results_all: List[MonthResult] = []
    dod_key_override = None if dod_override == "Auto (infer)" else int(dod_override.strip("%"))
    first_year_logs: Optional[HourlyLog] = None
    final_year_logs = None
    hourly_logs_by_year: Dict[int, HourlyLog] = {}
    hod_count = np.zeros(24, dtype=float)
    hod_sum_pv = np.zeros(24, dtype=float)
    hod_sum_pv_resource = np.zeros(24, dtype=float)
    hod_sum_bess = np.zeros(24, dtype=float)
    hod_sum_charge = np.zeros(24, dtype=float)
    augmentation_events = 0

    for y in range(1, cfg.years + 1):
        yr, logs, monthly_results = simulate_year(state, y, dod_key_override, need_logs=(need_logs and y == cfg.years))
        hours = np.mod(logs.hod.astype(int), 24)
        np.add.at(hod_count, hours, 1)
        np.add.at(hod_sum_pv, hours, logs.pv_to_contract_mw)
        np.add.at(hod_sum_pv_resource, hours, logs.pv_mw)
        np.add.at(hod_sum_bess, hours, logs.bess_to_contract_mw)
        np.add.at(hod_sum_charge, hours, logs.charge_mw)
        if y == 1 and need_logs:
            first_year_logs = logs
        if y == cfg.years and need_logs:
            final_year_logs = logs
        if need_logs:
            hourly_logs_by_year[y] = logs
        state.cum_cycles = yr.cum_cycles
        results.append(yr)
        monthly_results_all.extend(monthly_results)
        retired_energy, retire_add_p, retire_add_e = retire_cohorts_if_needed(state, cfg, y)
        augmentation_retired_energy[y - 1] = retired_energy
        if retire_add_p > 0 or retire_add_e > 0:
            augmentation_events += 1
            augmentation_energy_added[y - 1] += retire_add_e
            state.current_power_mw += retire_add_p
            state.current_usable_mwh_bolref += retire_add_e
            state.cohorts.append(BatteryCohort(energy_mwh_bol=retire_add_e, start_year=y))
        add_p, add_e = apply_augmentation(state, cfg, yr, dis_hours_per_day)
        if add_p > 0 or add_e > 0:
            augmentation_events += 1
            augmentation_energy_added[y - 1] += add_e
            state.current_power_mw += add_p
            state.current_usable_mwh_bolref += add_e
            state.cohorts.append(BatteryCohort(energy_mwh_bol=add_e, start_year=y))

    return SimulationOutput(
        cfg=cfg,
        discharge_hours_per_day=dis_hours_per_day,
        results=results,
        monthly_results=monthly_results_all,
        first_year_logs=first_year_logs,
        final_year_logs=final_year_logs,
        hod_count=hod_count,
        hod_sum_pv=hod_sum_pv,
        hod_sum_pv_resource=hod_sum_pv_resource,
        hod_sum_bess=hod_sum_bess,
        hod_sum_charge=hod_sum_charge,
        augmentation_energy_added_mwh=augmentation_energy_added,
        augmentation_retired_energy_mwh=augmentation_retired_energy,
        augmentation_events=augmentation_events,
        hourly_logs_by_year=hourly_logs_by_year,
    )


def summarize_simulation(sim_output: SimulationOutput) -> SimulationSummary:
    results = sim_output.results
    final = results[-1]
    expected_total = sum(r.expected_firm_mwh for r in results)
    actual_total = sum(r.delivered_firm_mwh for r in results)
    compliance = (actual_total / expected_total * 100.0) if expected_total > 0 else float("nan")
    total_discharge_mwh = sum(r.discharge_mwh for r in results)
    total_charge_mwh = sum(r.charge_mwh for r in results)
    bess_generation_mwh = sum(r.bess_to_contract_mwh for r in results)
    pv_generation_mwh = sum(r.pv_to_contract_mwh for r in results)
    pv_excess_mwh = sum(r.pv_curtailed_mwh for r in results)
    charge_discharge_ratio = (total_charge_mwh / total_discharge_mwh) if total_discharge_mwh > 0 else float("nan")
    bess_share_of_firm = (bess_generation_mwh / actual_total * 100.0) if actual_total > 0 else float("nan")
    pv_capture_ratio = (
        total_charge_mwh / (total_charge_mwh + pv_excess_mwh) if (total_charge_mwh + pv_excess_mwh) > 0 else float("nan")
    )
    hours_in_discharge_windows_year = sim_output.discharge_hours_per_day * 365.0
    discharge_capacity_factor = (
        final.discharge_mwh / (final.eoy_power_mw * hours_in_discharge_windows_year) if final.eoy_power_mw > 0 else float("nan")
    )
    total_project_generation_mwh = actual_total
    bess_losses_mwh = max(total_charge_mwh - total_discharge_mwh, 0.0)
    total_shortfall_mwh = sum(r.shortfall_mwh for r in results)
    avg_eq_cycles_per_year = float(np.mean([r.eq_cycles for r in results]))
    cap_daily_final = min(final.eoy_usable_mwh, final.eoy_power_mw * sim_output.discharge_hours_per_day)
    daily_target_mwh = _daily_target_mwh(sim_output.cfg, sim_output.discharge_hours_per_day)
    cap_ratio_final = (cap_daily_final / daily_target_mwh) if daily_target_mwh > 0 else float("nan")

    return SimulationSummary(
        compliance=compliance,
        bess_share_of_firm=bess_share_of_firm,
        charge_discharge_ratio=charge_discharge_ratio,
        pv_capture_ratio=pv_capture_ratio,
        discharge_capacity_factor=discharge_capacity_factor,
        total_project_generation_mwh=total_project_generation_mwh,
        bess_generation_mwh=bess_generation_mwh,
        pv_generation_mwh=pv_generation_mwh,
        pv_excess_mwh=pv_excess_mwh,
        bess_losses_mwh=bess_losses_mwh,
        total_shortfall_mwh=total_shortfall_mwh,
        avg_eq_cycles_per_year=avg_eq_cycles_per_year,
        cap_ratio_final=cap_ratio_final,
    )
