import math
import calendar
import json
from dataclasses import dataclass, field, asdict
from io import BytesIO, StringIO
from typing import Any, List, Tuple, Optional, Dict

import streamlit as st
from streamlit.delta_generator import DeltaGenerator
import numpy as np
import pandas as pd
from pathlib import Path
from pandas.tseries.frequencies import to_offset
import altair as alt
from fpdf import FPDF
from utils import (
    FLAG_DEFINITIONS,
    build_flag_insights,
    enforce_rate_limit,
    get_rate_limit_password,
    parse_numeric_series,
    read_cycle_model,
    read_pv_profile,
)
from utils.economics import (
    CashFlowOutputs,
    DEVEX_COST_PHP,
    EconomicOutputs,
    EconomicInputs,
    PriceInputs,
    compute_cash_flows_and_irr,
    compute_lcoe_lcos_with_augmentation_fallback,
    estimate_augmentation_costs_by_year,
)
from utils.ui_layout import init_page_layout
from utils.ui_state import get_base_dir, load_shared_data


BASE_DIR = get_base_dir()

@dataclass
class Window:
    start: float  # hour-of-day, inclusive
    end: float    # hour-of-day, exclusive

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

def parse_windows(text: str) -> List[Window]:
    if not text.strip():
        return []
    wins = []

    def _parse_time(token: str) -> float:
        parts = token.split(':')
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

    for part in [p.strip() for p in text.split(',') if p.strip()]:
        try:
            a, b = part.split('-')
            h1 = _parse_time(a)
            h2 = _parse_time(b)
            if not (0.0 <= h1 < 24.0 and 0.0 <= h2 < 24.0):
                st.warning(f"Invalid window hour in '{part}' (00:00-23:59). Skipped.")
                continue
            wins.append(Window(h1, h2))
        except Exception:
            st.warning(f"Could not parse window '{part}'. Use 'HH:MM-HH:MM'. Skipped.")
    return wins


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

# --------- Degradation helpers ---------

def infer_dod_bucket(daily_dis_mwh: np.ndarray, usable_mwh_available: float) -> int:
    """Infer an effective DoD bucket from daily discharge energy.

    The median is computed across the daily data provided, which for annual
    simulations represents the entire year's profile rather than a single
    month. Anchoring the inference on the full-year distribution yields a
    stable DoD assumption for downstream monthly and annual cycle calculations
    instead of letting seasonal swings toggle the bucket from month to month.

    The usable energy basis should reflect the battery's available capability
    (after degradation/availability), not just the original BOL rating. Using
    an inflated reference skews the implied DoD downward and overstates the
    equivalent cycles computed later in the simulation.
    """

    if usable_mwh_available <= 0: return 100
    if len(daily_dis_mwh) == 0: return 10
    med = float(np.median(daily_dis_mwh))
    if med <= 0: return 10
    r = med / max(1e-9, usable_mwh_available)
    if r >= 0.9: return 100
    if r >= 0.8: return 80
    if r >= 0.4: return 40
    if r >= 0.2: return 20
    return 10

def cycle_retention_lookup(cycle_df: pd.DataFrame, dod_key: int, cumulative_cycles: float) -> float:
    c_col = f"DoD{dod_key}_Cycles"; r_col = f"DoD{dod_key}_Ret(%)"
    if c_col not in cycle_df.columns or r_col not in cycle_df.columns:
        return 1.0
    df = cycle_df[[c_col, r_col]].dropna().sort_values(c_col)
    x = df[c_col].to_numpy(float); y = df[r_col].to_numpy(float)
    if len(x) == 0: return 1.0
    if cumulative_cycles <= x[0]: ret = y[0]
    elif cumulative_cycles >= x[-1]: ret = y[-1]
    else: ret = np.interp(cumulative_cycles, x, y)
    return max(0.0, float(ret)) / 100.0

# --------- Simulation core ---------

@dataclass
class SimConfig:
    years: int = 20
    step_hours: float = 1.0
    pv_deg_rate: float = 0.006
    pv_availability: float = 0.98
    bess_availability: float = 0.99
    rte_roundtrip: float = 0.88           # single (η_rt)
    use_split_rte: bool = False
    charge_efficiency: Optional[float] = None
    discharge_efficiency: Optional[float] = None
    soc_floor: float = 0.10
    soc_ceiling: float = 0.90
    initial_power_mw: float = 30.0
    initial_usable_mwh: float = 120.0
    contracted_mw: float = 30.0
    discharge_windows: List[Window] = field(default_factory=lambda: [Window(10,14), Window(18,22)])
    charge_windows_text: str = ""
    max_cycles_per_day_cap: float = 1.2
    calendar_fade_rate: float = 0.01
    use_calendar_exp_model: bool = True
    # Augmentation knobs
    augmentation: str = "None"  # 'None'|'Threshold'|'Periodic'
    aug_trigger_type: str = "Capability"  # 'Capability'|'SOH'
    aug_threshold_margin: float = 0.00    # capability mode
    aug_topup_margin: float = 0.05        # capability mode
    aug_soh_trigger_pct: float = 0.80     # SOH mode (e.g., 0.80 = 80%)
    aug_soh_add_frac_initial: float = 0.10  # SOH mode: add % of initial BOL energy
    aug_periodic_every_years: int = 5
    aug_periodic_add_frac_of_bol: float = 0.10
    aug_add_mode: str = "Percent"  # 'Percent'|'Fixed'
    aug_fixed_energy_mwh: float = 0.0  # fixed augmentation size when aug_add_mode='Fixed'
    aug_retire_old_cohort: bool = False
    aug_retire_soh_pct: float = 0.60
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
    charge_mw: np.ndarray
    discharge_mw: np.ndarray
    soc_mwh: np.ndarray


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


def resolve_efficiencies(cfg: SimConfig) -> Tuple[float, float, float]:
    """Return (charge, discharge, roundtrip) efficiencies with consistent bounds.

    When ``use_split_rte`` is enabled and both charge/discharge efficiencies
    are supplied, the function preserves their ratio while bounding each term
    to a reasonable range. Otherwise a symmetric split (√RTE) is used to keep
    downstream calculations aligned with the single-input UI.
    """

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
    """Return energy-weighted (cycle, calendar, total) SOH for the fleet.

    cycles_at_point lets callers supply cohort-specific cumulative cycles at a
    particular timestep (e.g., inside the monthly loop) without mutating the
    stored cohort state. Defaults to using the cohort.cum_cycles values.
    """

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


def _draw_metric_card(pdf: FPDF, x: float, y: float, w: float, h: float, title: str, value: str, subtitle: str,
                      fill_rgb: Tuple[int, int, int]) -> None:
    pdf.set_fill_color(*fill_rgb)
    pdf.set_draw_color(230, 232, 235)
    pdf.rect(x, y, w, h, style="DF")
    pdf.set_xy(x + 2, y + 2)
    pdf.set_text_color(50, 50, 50)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(w - 4, 5, title, ln=1)

    pdf.set_xy(x + 2, y + 9)
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(15, 15, 15)
    pdf.cell(w - 4, 7, value, ln=1)

    pdf.set_xy(x + 2, y + h - 6)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(w - 4, 4, subtitle)
    pdf.set_text_color(0, 0, 0)


def _draw_sparkline(pdf: FPDF, x: float, y: float, w: float, h: float, series: List[Tuple[str, List[float], Tuple[int, int, int]]],
                    y_label: str) -> None:
    pdf.set_draw_color(230, 232, 235)
    pdf.rect(x, y, w, h)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_xy(x, y - 5)
    pdf.cell(w, 4, y_label)

    if not series or not series[0][1]:
        return

    max_len = max(len(vals) for _, vals, _ in series)
    if max_len < 2:
        return

    all_values = [v for _, vals, _ in series for v in vals]
    min_v = min(all_values)
    max_v = max(all_values)
    span = max(1e-9, max_v - min_v)

    for label, vals, color in series:
        if len(vals) < 2:
            continue
        pdf.set_draw_color(*color)
        step_x = w / max(1, len(vals) - 1)
        points = []
        for idx, val in enumerate(vals):
            px = x + idx * step_x
            py = y + h - ((val - min_v) / span * h)
            points.append((px, py))
        for i in range(len(points) - 1):
            pdf.line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
        pdf.set_xy(points[-1][0] - 8, points[-1][1] - 3)
        pdf.cell(16, 4, label, align="C")


def _fmt(val: float, suffix: str = "") -> str:
    return f"{val:,.2f}{suffix}" if abs(val) >= 10 else f"{val:,.3f}{suffix}"


def _percent(numerator: float, denominator: float) -> float:
    return (numerator / denominator * 100.0) if denominator > 0 else float("nan")


def _average_profile_from_aggregates(cfg: SimConfig, hod_count: np.ndarray, hod_sum_pv_resource: np.ndarray,
                                     hod_sum_pv: np.ndarray, hod_sum_bess: np.ndarray, hod_sum_charge: np.ndarray
                                     ) -> Dict[str, List[float]]:
    """Return the average daily profile across the full project using hourly aggregates.

    The aggregates include every simulated hour (all years), so the average reflects
    degradation and augmentation across life. Hour counts guard against divide-by-zero
    for sparse inputs.
    """
    contracted_by_hour = [
        cfg.contracted_mw if any(w.contains(h) for w in cfg.discharge_windows) else 0.0
        for h in range(24)
    ]

    with np.errstate(invalid='ignore', divide='ignore'):
        avg_pv_resource = np.divide(
            hod_sum_pv_resource,
            hod_count,
            out=np.zeros_like(hod_sum_pv_resource),
            where=hod_count > 0,
        )
        avg_pv_to_contract = np.divide(
            hod_sum_pv,
            hod_count,
            out=np.zeros_like(hod_sum_pv),
            where=hod_count > 0,
        )
        avg_bess_to_contract = np.divide(
            hod_sum_bess,
            hod_count,
            out=np.zeros_like(hod_sum_bess),
            where=hod_count > 0,
        )
        avg_charge = np.divide(
            hod_sum_charge,
            hod_count,
            out=np.zeros_like(hod_sum_charge),
            where=hod_count > 0,
        )

    return {
        "hours": list(range(24)),
        "contracted": [float(v) for v in contracted_by_hour],
        "pv_resource": avg_pv_resource.tolist(),
        "pv_to_contract": avg_pv_to_contract.tolist(),
        "bess_to_contract": avg_bess_to_contract.tolist(),
        "charge": avg_charge.tolist(),
    }


def _draw_table(pdf: FPDF, x: float, y: float, col_widths: List[float], rows: List[List[str]],
                header_fill: Tuple[int, int, int]=(245, 248, 255), row_fill: Tuple[int, int, int]=(255, 255, 255),
                border: bool=True, header_bold: bool=True, font_size: int=9) -> float:
    """Render a simple table and return the updated y position (bottom of table)."""
    pdf.set_xy(x, y)
    pdf.set_font("Helvetica", "B" if header_bold else "", font_size)
    pdf.set_fill_color(*header_fill)
    pdf.set_draw_color(220, 223, 228)
    pdf.set_text_color(20, 20, 20)
    # header
    for idx, cell in enumerate(rows[0]):
        pdf.cell(col_widths[idx], 6, cell, border=1 if border else 0, ln=0, align="L", fill=True)
    pdf.ln(6)
    pdf.set_font("Helvetica", "", font_size)
    pdf.set_fill_color(*row_fill)
    # body
    for row in rows[1:]:
        pdf.set_x(x)
        for idx, cell in enumerate(row):
            pdf.cell(col_widths[idx], 6, cell, border=1 if border else 0, ln=0, align="L", fill=True)
        pdf.ln(6)
    return pdf.get_y()


def build_pdf_summary(cfg: SimConfig, results: List[YearResult], compliance: float, bess_share: float,
                      charge_discharge_ratio: float, pv_capture_ratio: float,
                      discharge_capacity_factor: float, discharge_windows_text: str,
                      charge_windows_text: str, hod_count: np.ndarray, hod_sum_pv_resource: np.ndarray,
                      hod_sum_pv: np.ndarray, hod_sum_bess: np.ndarray, hod_sum_charge: np.ndarray,
                      total_shortfall_mwh: float, pv_excess_mwh: float, total_generation_mwh: float,
                      bess_generation_mwh: float, pv_generation_mwh: float, bess_losses_mwh: float) -> bytes:
    if not results:
        pdf = FPDF(format="A4")
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "BESS Lab - One-page Summary")
        pdf.ln(8)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(90, 90, 90)
        pdf.multi_cell(0, 5, "PDF snapshot unavailable because no results were generated. Run a simulation to view the summary.")
        pdf_bytes = pdf.output(dest='S')
        return pdf_bytes.encode('latin-1') if isinstance(pdf_bytes, str) else bytes(pdf_bytes)

    final = results[-1]
    first = results[0]
    eta_ch, eta_dis, eta_rt = resolve_efficiencies(cfg)
    pdf = FPDF(format="A4")
    pdf.add_page()
    margin = 12
    usable_width = 210 - 2 * margin

    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 10, "BESS Lab - One-page Summary", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Project life: {cfg.years} years  |  Contracted: {cfg.contracted_mw:.1f} MW  |  PV-only charging", ln=1)
    pdf.cell(0, 6, f"Discharge windows: {discharge_windows_text}  |  Charge windows: {charge_windows_text or 'Any PV hour'}", ln=1)
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Inputs used", ln=1)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, f"Initial power: {cfg.initial_power_mw:.1f} MW  |  Initial usable: {cfg.initial_usable_mwh:.1f} MWh  |  RTE: {eta_rt:.2f}", ln=1)
    if cfg.use_split_rte:
        pdf.cell(0, 5, f"Charge efficiency: {eta_ch:.2f}  |  Discharge efficiency: {eta_dis:.2f}", ln=1)
    pdf.cell(0, 5, f"PV availability: {cfg.pv_availability:.2f}  |  BESS availability: {cfg.bess_availability:.2f}  |  SoC window: {cfg.soc_floor:.2f}-{cfg.soc_ceiling:.2f}", ln=1)
    pdf.cell(0, 5, f"PV deg: {cfg.pv_deg_rate:.3f}/yr  |  Calendar fade: {cfg.calendar_fade_rate:.3f}/yr  |  Augmentation: {cfg.augmentation}", ln=1)
    if cfg.augmentation_schedule:
        pdf.cell(0, 5, f"Manual augmentation: {describe_schedule(cfg.augmentation_schedule)}", ln=1)
    pdf.ln(2)

    card_width = (usable_width - 10) / 3
    card_height = 22
    x0 = margin
    y0 = pdf.get_y()
    _draw_metric_card(pdf, x0, y0, card_width, card_height, "Delivery compliance", f"{compliance:,.2f}%", "Across full life",
                      (225, 245, 255))
    deficit_pct = _percent(total_shortfall_mwh, sum(r.expected_firm_mwh for r in results))
    surplus_pct = _percent(pv_excess_mwh, sum(r.available_pv_mwh for r in results))
    _draw_metric_card(pdf, x0 + card_width + 5, y0, card_width, card_height, "Delivery deficit",
                      f"{deficit_pct:,.2f}%", "Shortfall vs expected", (255, 238, 238))
    _draw_metric_card(pdf, x0 + 2 * (card_width + 5), y0, card_width, card_height, "PV surplus",
                      f"{surplus_pct:,.2f}%", "Curtailment share", (255, 245, 235))

    y_cards2 = y0 + card_height + 4
    _draw_metric_card(pdf, x0, y_cards2, card_width, card_height, "PV capture ratio",
                      _fmt(pv_capture_ratio), "PV used vs available", (255, 245, 235))
    _draw_metric_card(pdf, x0 + card_width + 5, y_cards2, card_width, card_height, "Discharge CF (final)",
                      _fmt(discharge_capacity_factor), "Avg MW / contracted", (238, 245, 255))
    _draw_metric_card(pdf, x0 + 2 * (card_width + 5), y_cards2, card_width, card_height, "SOH total (final)",
                      _fmt(final.soh_total, ""), "Cycle & calendar combined", (240, 240, 240))

    pdf.set_y(y_cards2 + card_height + 6)
    chart_width = (usable_width - 5) / 2
    chart_height = 55
    chart_x_left = margin
    chart_y = pdf.get_y()

    avg_profile = _average_profile_from_aggregates(cfg, hod_count, hod_sum_pv_resource, hod_sum_pv, hod_sum_bess, hod_sum_charge)
    pdf.set_xy(chart_x_left, chart_y)
    pdf.set_draw_color(230, 232, 235)
    pdf.rect(chart_x_left, chart_y, usable_width, chart_height)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_xy(chart_x_left + 2, chart_y - 5)
    pdf.cell(usable_width, 4, "Average daily profile (MW) - project-wide")
    hours = avg_profile["hours"]
    if len(hours) >= 2:
        max_power = max(
            max(avg_profile["pv_resource"] or [0]),
            max(avg_profile["pv_to_contract"] or [0]),
            max(avg_profile["bess_to_contract"] or [0]),
            max(-min(avg_profile["charge"] or [0]), 0),
            max(avg_profile["contracted"] or [0]),
        )
        span = max(1e-6, max_power * 1.1)
        step_x = usable_width / max(1, len(hours) - 1)
        # axis lines
        zero_y = chart_y + chart_height - (0.0 / span * chart_height)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(chart_x_left, zero_y, chart_x_left + usable_width, zero_y)
        # helper to plot series with optional fill to zero
        def _polyline(values: List[float], color: Tuple[int, int, int], invert: bool=False) -> None:
            points = []
            for idx, val in enumerate(values):
                px = chart_x_left + idx * step_x
                adj_val = -val if invert else val
                py = chart_y + chart_height - (adj_val / span * chart_height)
                points.append((px, py))
            pdf.set_draw_color(*color)
            for i in range(len(points) - 1):
                pdf.line(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1])
        _polyline(avg_profile["pv_resource"], (199, 129, 0))
        _polyline(avg_profile["pv_to_contract"], (134, 197, 218))
        _polyline(avg_profile["bess_to_contract"], (127, 209, 139))
        _polyline([max(c, 0.0) for c in avg_profile["contracted"]], (242, 169, 0))
        _polyline(avg_profile["charge"], (202, 166, 255), invert=True)

        pdf.set_font("Helvetica", "", 7)
        legends = [
            ("PV resource", (199, 129, 0)),
            ("PV->Contract", (134, 197, 218)),
            ("BESS->Contract", (127, 209, 139)),
            ("Contract", (242, 169, 0)),
            ("Charge", (202, 166, 255)),
        ]
        legend_y = chart_y + chart_height + 2
        for idx, (label, color) in enumerate(legends):
            pdf.set_draw_color(*color)
            x_leg = chart_x_left + idx * (usable_width / len(legends))
            pdf.line(x_leg, legend_y, x_leg + 6, legend_y)
            pdf.set_xy(x_leg + 7, legend_y - 2)
            pdf.cell(30, 4, label)

    pdf.set_y(chart_y + chart_height + 8)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Design + final year snapshot", ln=1)
    pdf.set_font("Helvetica", "", 9)
    design_rows = [
        ["Metric", "Value"],
        ["Initial usable (MWh)", f"{cfg.initial_usable_mwh:,.1f}"],
        ["Initial power (MW)", f"{cfg.initial_power_mw:,.1f}"],
        ["Augmentation", cfg.augmentation],
        ["SoC window", f"{cfg.soc_floor:.2f}-{cfg.soc_ceiling:.2f}"],
        ["Round-trip efficiency", f"{eta_rt:.2f}"],
        ["Charge / discharge eff.", f"{eta_ch:.2f} / {eta_dis:.2f}" if cfg.use_split_rte else "Symmetric"],
        ["Calendar fade (/yr)", f"{cfg.calendar_fade_rate:.3f}"],
        ["EOY usable (Y1 -> final)", f"{first.eoy_usable_mwh:,.1f} -> {final.eoy_usable_mwh:,.1f}"],
        ["EOY power (Y1 -> final)", f"{first.eoy_power_mw:,.2f} -> {final.eoy_power_mw:,.2f}"],
        ["PV->Contract (final yr)", f"{final.pv_to_contract_mwh:,.1f}"],
        ["BESS->Contract (final yr)", f"{final.bess_to_contract_mwh:,.1f}"],
        ["Eq cycles (yr / cum)", f"{final.eq_cycles:,.1f} / {final.cum_cycles:,.1f}"],
    ]
    if cfg.augmentation_schedule:
        design_rows.insert(4, ["Manual augmentation", describe_schedule(cfg.augmentation_schedule)])
    table_widths = [usable_width * 0.45, usable_width * 0.55]
    table_bottom = _draw_table(pdf, margin, pdf.get_y(), table_widths, design_rows)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Generation summary", ln=1)
    pdf.set_font("Helvetica", "", 9)
    gen_rows = [
        ["Metric", "Value"],
        ["Total delivered (project)", f"{total_generation_mwh:,.1f} MWh"],
        ["Total shortfall", f"{total_shortfall_mwh:,.1f} MWh"],
        ["PV->Contract (project)", f"{pv_generation_mwh:,.1f} MWh"],
        ["BESS->Contract (project)", f"{bess_generation_mwh:,.1f} MWh"],
        ["PV surplus/curtailment", f"{pv_excess_mwh:,.1f} MWh"],
        ["BESS losses (charging ineff.)", f"{bess_losses_mwh:,.1f} MWh"],
        ["Charge/Discharge ratio", _fmt(charge_discharge_ratio)],
        ["PV capture ratio", _fmt(pv_capture_ratio)],
    ]
    _draw_table(pdf, margin, pdf.get_y(), table_widths, gen_rows)

    pdf.ln(1)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "Generation summary", ln=1)
    pdf.set_font("Helvetica", "", 9)
    gen_lines = [
        f"Total delivered (project): {total_generation_mwh:,.1f} MWh  |  Shortfall: {total_shortfall_mwh:,.1f} MWh",
        f"PV->Contract: {pv_generation_mwh:,.1f} MWh  |  BESS->Contract: {bess_generation_mwh:,.1f} MWh",
        f"PV surplus/curtailment: {pv_excess_mwh:,.1f} MWh  |  BESS losses (charging inefficiency): {bess_losses_mwh:,.1f} MWh",
        f"Charge/Discharge ratio: {_fmt(charge_discharge_ratio)}  |  PV capture ratio: {_fmt(pv_capture_ratio)}",
    ]
    for line in gen_lines:
        pdf.multi_cell(usable_width, 5, line)

    pdf.ln(2)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(0, 4, "Auto-generated from current Streamlit inputs. Keep everything on one page by focusing on the metrics that shape bankability and warranty conversations.")

    pdf_bytes = pdf.output(dest='S')
    return pdf_bytes.encode('latin-1') if isinstance(pdf_bytes, str) else bytes(pdf_bytes)

def in_any_window(hod: int, windows: List[Window]) -> bool:
    return any(w.contains(hod) for w in windows)

def simulate_year(state: SimState, year_idx: int, dod_key: Optional[int], need_logs: bool=False) -> Tuple[YearResult, HourlyLog, List[MonthResult]]:
    cfg = state.cfg; dt = cfg.step_hours

    pv_scale = (1.0 - cfg.pv_deg_rate) ** (year_idx - 1)
    pv_mw = state.pv_df['pv_mw'].to_numpy(float) * pv_scale * cfg.pv_availability

    pow_cap_mw = state.current_power_mw * cfg.bess_availability

    eta_ch, eta_dis, _ = resolve_efficiencies(cfg)

    ch_windows = parse_windows(cfg.charge_windows_text)
    dis_windows = cfg.discharge_windows

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
    if "timestamp" in state.pv_df.columns:
        calendar_index = pd.to_datetime(state.pv_df["timestamp"], errors="coerce")
    else:
        calendar_index = pd.date_range(
            "2020-01-01", periods=n_hours, freq=pd.Timedelta(hours=dt)
        )
    calendar_index = pd.DatetimeIndex(calendar_index)
    if calendar_index.isna().any():
        raise ValueError("PV timestamps contain invalid entries after cleaning.")

    day_index = ((calendar_index.normalize() - calendar_index[0].normalize()) / pd.Timedelta("1D")).astype(int)
    month_index = calendar_index.month - 1
    daily_dis_mwh = np.zeros(day_index.max() + 1)
    hod = (calendar_index.hour + calendar_index.minute / 60.0 + calendar_index.second / 3600.0).to_numpy()

    pv_to_contract_mw_log = np.zeros(n_hours)
    bess_to_contract_mw_log = np.zeros(n_hours)
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

    expected_firm_mwh = charged_mwh = discharged_mwh = pv_available_mwh = pv_to_contract_mwh = bess_to_contract_mwh = pv_curtailed_mwh = 0.0
    flag_shortfall_hours = flag_soc_floor_hits = flag_soc_ceiling_hits = 0

    for h in range(n_hours):
        is_dis = in_any_window(int(hod[h]), dis_windows)
        is_ch = True if not ch_windows else in_any_window(int(hod[h]), ch_windows)
        pv_avail_mw = max(0.0, pv_mw[h])

        pv_available_mwh += pv_avail_mw * dt
        month_pv_available[month_index[h]] += pv_avail_mw * dt

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
                dis_mw = delivered; e_req = e_can
            soc_mwh -= e_req
            discharged_mwh += dis_mw * dt
            daily_dis_mwh[day_index[h]] += dis_mw * dt
            bess_to_contract_mwh += dis_mw * dt
            discharge_mw_log[h] = dis_mw
            bess_to_contract_mw_log[h] = dis_mw
            if abs(soc_mwh - soc_min) < 1e-6: flag_soc_floor_hits += 1
            if abs(soc_mwh - soc_min) < 1e-6: month_flag_soc_floor_hits[month_index[h]] += 1

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
                if abs(soc_mwh - soc_max) < 1e-6: flag_soc_ceiling_hits += 1
                if abs(soc_mwh - soc_max) < 1e-6: month_flag_soc_ceiling_hits[month_index[h]] += 1

        pv_curtailed_mwh += max(0.0, pv_avail_after_contract - ch_mw) * dt
        pv_to_contract_mw_log[h] = pv_to_contract_mw
        soc_log[h] = soc_mwh

        month_expected[month_index[h]] += target_mw * dt
        delivered_hour = pv_to_contract_mw + dis_mw
        month_delivered[month_index[h]] += delivered_hour * dt
        month_shortfall[month_index[h]] += max(0.0, target_mw - delivered_hour) * dt
        month_charge[month_index[h]] += ch_mw * dt
        month_discharge[month_index[h]] += dis_mw * dt
        month_pv_contract[month_index[h]] += pv_to_contract_mw * dt
        month_bess_contract[month_index[h]] += dis_mw * dt
        month_pv_curtailed[month_index[h]] += max(0.0, pv_avail_after_contract - ch_mw) * dt

    avg_rte = (discharged_mwh / charged_mwh) if charged_mwh > 0 else np.nan

    dod_key_eff = (
        dod_key
        if dod_key is not None
        else infer_dod_bucket(daily_dis_mwh, usable_mwh_start)
    )
    # The inferred DoD reflects the full year's median daily discharge and is
    # reused for monthly reporting so that both annual and monthly cycle counts
    # share a consistent depth assumption.
    state.last_dod_key = dod_key_eff
    dod_frac = {10:0.10,20:0.20,40:0.40,80:0.80,100:1.00}[dod_key_eff]
    usable_for_cycles = max(1e-9, usable_mwh_start * dod_frac)
    eq_cycles_year = discharged_mwh / usable_for_cycles
    # Add the year's equivalent cycles once and reuse that increment for
    # every cohort and the fleet-level counter. Keeping the increment in a
    # single variable avoids accidental double-handling if state.cum_cycles
    # is also updated by the caller.
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
        avg_rte=float(avg_rte) if not np.isnan(avg_rte) else float('nan'),
        eq_cycles=float(eq_cycles_year),
        cum_cycles=float(cum_cycles_new),
        soh_cycle=float(soh_cycle),
        soh_calendar=float(soh_calendar),
        soh_total=float(soh_total),
        eoy_usable_mwh=float(eoy_usable_mwh),
        eoy_power_mw=float(eoy_power_mw),
        pv_curtailed_mwh=float(pv_curtailed_mwh),
        flags={'firm_shortfall_hours':int(flag_shortfall_hours),'soc_floor_hits':int(flag_soc_floor_hits),'soc_ceiling_hits':int(flag_soc_ceiling_hits)},
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
        monthly_results.append(MonthResult(
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
            avg_rte=float(month_discharge[m] / month_charge[m]) if month_charge[m] > 0 else float('nan'),
            eq_cycles=float(eq_cycles_month),
            cum_cycles=float(cum_cycles_running),
            soh_cycle=float(soh_cycle_month),
            soh_calendar=float(soh_calendar_month),
            soh_total=float(soh_total_month),
            eom_usable_mwh=float(state.current_usable_mwh_bolref * soh_total_month),
            eom_power_mw=float(pow_cap_mw),
            pv_curtailed_mwh=float(month_pv_curtailed[m]),
            flags={
                'firm_shortfall_hours': int(month_flag_shortfall_hours[m]),
                'soc_floor_hits': int(month_flag_soc_floor_hits[m]),
                'soc_ceiling_hits': int(month_flag_soc_ceiling_hits[m]),
            },
        ))

    for idx, cohort in enumerate(state.cohorts):
        cohort.cum_cycles = cohort_cycles_eoy[idx]

    # Keep the fleet-level cumulative cycles in sync with the cohorts to
    # avoid adding the year's increment again elsewhere.
    state.cum_cycles = cum_cycles_new

    logs = HourlyLog(
        hod=hod,
        pv_mw=pv_mw,
        pv_to_contract_mw=pv_to_contract_mw_log,
        bess_to_contract_mw=bess_to_contract_mw_log,
        charge_mw=charge_mw_log,
        discharge_mw=discharge_mw_log,
        soc_mwh=soc_log,
    )
    return yr, logs, monthly_results


def retire_cohorts_if_needed(state: SimState, cfg: SimConfig, year_idx: int) -> float:
    """Retire cohorts whose SOH is below the configured threshold.

    Returns
    -------
    float
        Total BOL energy (MWh) removed from service.
    """

    if not cfg.aug_retire_old_cohort:
        return 0.0

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
        return 0.0

    state.cohorts = remaining_cohorts
    state.current_usable_mwh_bolref = max(0.0, state.current_usable_mwh_bolref - retired_energy_bol)
    state.current_power_mw = max(0.0, state.current_power_mw - retired_energy_bol / c_hours)
    return retired_energy_bol


def _find_manual_schedule_entry(
    schedule: List[AugmentationScheduleEntry], year_idx: int
) -> Optional[AugmentationScheduleEntry]:
    for entry in schedule:
        if entry.year == year_idx:
            return entry
    return None


def _compute_manual_augmentation(
    entry: AugmentationScheduleEntry, state: SimState
) -> Tuple[float, float]:
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
    else:  # Fixed energy (MWh) or fallback
        add_energy_bol = value
        add_power = add_energy_bol / c_hours

    return float(add_power), float(add_energy_bol)


def describe_schedule(entries: List[AugmentationScheduleEntry]) -> str:
    if not entries:
        return "None"

    parts = [
        f"Y{entry.year}: {entry.value:g} ({entry.basis})"
        for entry in sorted(entries, key=lambda e: e.year)
    ]
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
        entries.append(
            AugmentationScheduleEntry(
                year=year, basis=str(basis_val), value=float(amount_val)
            )
        )

    entries.sort(key=lambda e: e.year)
    return entries, errors


def apply_augmentation(state: SimState, cfg: SimConfig, yr: YearResult, discharge_hours_per_day: float) -> Tuple[float, float]:
    """Return (add_power_MW, add_energy_MWh at BOL)."""
    scheduled_entry = _find_manual_schedule_entry(cfg.augmentation_schedule, yr.year_index)
    if scheduled_entry is not None:
        return _compute_manual_augmentation(scheduled_entry, state)

    if cfg.augmentation == 'None':
        return 0.0, 0.0

    if cfg.augmentation == 'Threshold' and cfg.aug_trigger_type == 'Capability':
        target_energy_per_day = cfg.contracted_mw * discharge_hours_per_day
        eoy_cap_per_day = min(yr.eoy_usable_mwh, yr.eoy_power_mw * discharge_hours_per_day)
        if eoy_cap_per_day + 1e-6 < target_energy_per_day * (1.0 - cfg.aug_threshold_margin):
            short_mwh = target_energy_per_day * (1.0 + cfg.aug_topup_margin) - eoy_cap_per_day
            add_energy_bol = max(0.0, short_mwh)
            if cfg.aug_add_mode == 'Fixed' and cfg.aug_fixed_energy_mwh > 0:
                add_energy_bol = cfg.aug_fixed_energy_mwh
            c_hours = max(1e-9, state.initial_bol_energy_mwh / max(1e-9, state.initial_bol_power_mw))
            add_power = add_energy_bol / c_hours
            return float(add_power), float(add_energy_bol)
        return 0.0, 0.0

    if cfg.augmentation == 'Threshold' and cfg.aug_trigger_type == 'SOH':
        if yr.soh_total <= cfg.aug_soh_trigger_pct + 1e-9:
            add_energy_bol = cfg.aug_soh_add_frac_initial * state.initial_bol_energy_mwh
            if cfg.aug_add_mode == 'Fixed' and cfg.aug_fixed_energy_mwh > 0:
                add_energy_bol = cfg.aug_fixed_energy_mwh
            c_hours = max(1e-9, state.initial_bol_energy_mwh / max(1e-9, state.initial_bol_power_mw))
            add_power = add_energy_bol / c_hours
            return float(add_power), float(add_energy_bol)
        return 0.0, 0.0

    if cfg.augmentation == 'Periodic':
        if (yr.year_index % max(1, cfg.aug_periodic_every_years)) == 0:
            add_energy_bol = cfg.aug_periodic_add_frac_of_bol * state.current_usable_mwh_bolref
            if cfg.aug_add_mode == 'Fixed' and cfg.aug_fixed_energy_mwh > 0:
                add_energy_bol = cfg.aug_fixed_energy_mwh
            c_hours = max(1e-9, state.initial_bol_energy_mwh / max(1e-9, state.initial_bol_power_mw))
            add_power = add_energy_bol / c_hours
            return float(add_power), float(add_energy_bol)
        return 0.0, 0.0

    return 0.0, 0.0


def simulate_project(cfg: SimConfig, pv_df: pd.DataFrame, cycle_df: pd.DataFrame, dod_override: str,
                     need_logs: bool = True) -> SimulationOutput:
    if not cfg.discharge_windows:
        raise ValueError("Please provide at least one discharge window.")

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
    dod_key_override = None if dod_override == "Auto (infer)" else int(dod_override.strip('%'))
    first_year_logs: Optional[HourlyLog] = None
    final_year_logs = None
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
        state.cum_cycles = yr.cum_cycles
        results.append(yr)
        monthly_results_all.extend(monthly_results)
        retired_energy = retire_cohorts_if_needed(state, cfg, y)
        augmentation_retired_energy[y - 1] = retired_energy
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
    )


def summarize_simulation(sim_output: SimulationOutput) -> SimulationSummary:
    results = sim_output.results
    final = results[-1]
    expected_total = sum(r.expected_firm_mwh for r in results)
    actual_total = sum(r.delivered_firm_mwh for r in results)
    compliance = (actual_total / expected_total * 100.0) if expected_total > 0 else float('nan')
    total_discharge_mwh = sum(r.discharge_mwh for r in results)
    total_charge_mwh = sum(r.charge_mwh for r in results)
    bess_generation_mwh = sum(r.bess_to_contract_mwh for r in results)
    pv_generation_mwh = sum(r.pv_to_contract_mwh for r in results)
    pv_excess_mwh = sum(r.pv_curtailed_mwh for r in results)
    charge_discharge_ratio = (total_charge_mwh / total_discharge_mwh) if total_discharge_mwh > 0 else float('nan')
    bess_share_of_firm = (bess_generation_mwh / actual_total * 100.0) if actual_total > 0 else float('nan')
    pv_capture_ratio = (total_charge_mwh / (total_charge_mwh + pv_excess_mwh)) if (total_charge_mwh + pv_excess_mwh) > 0 else float('nan')
    hours_in_discharge_windows_year = sim_output.discharge_hours_per_day * 365.0
    discharge_capacity_factor = (final.discharge_mwh / (final.eoy_power_mw * hours_in_discharge_windows_year)) if final.eoy_power_mw > 0 else float('nan')
    total_project_generation_mwh = actual_total
    bess_losses_mwh = max(total_charge_mwh - total_discharge_mwh, 0.0)
    total_shortfall_mwh = sum(r.shortfall_mwh for r in results)
    avg_eq_cycles_per_year = float(np.mean([r.eq_cycles for r in results]))
    cap_daily_final = min(final.eoy_usable_mwh, final.eoy_power_mw * sim_output.discharge_hours_per_day)
    cap_ratio_final = cap_daily_final / (sim_output.cfg.contracted_mw * sim_output.discharge_hours_per_day) if sim_output.discharge_hours_per_day > 0 else float('nan')

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

# --------- Streamlit UI ---------

def prepare_soc_heatmap_data(logs: HourlyLog, initial_energy_mwh: float) -> pd.DataFrame:
    """Return a day-of-year × hour-of-day SOC fraction pivot for heatmap rendering.

    Using a pivot keeps aggregation vectorized and consistent across years,
    while normalizing by the initial usable energy highlights reliability
    headroom and clipping regardless of degradation.
    """
    hours_total = len(logs.hod)
    if hours_total == 0:
        return pd.DataFrame(
            index=pd.Index([], name="day_of_year"),
            columns=pd.Index(range(24), name="hour"),
        )

    hours_per_day = 24
    day_count = int(math.ceil(hours_total / hours_per_day))
    df = pd.DataFrame(
        {
            "hour_index": np.arange(hours_total, dtype=int),
            "hour": logs.hod.astype(int),
            "soc_frac": logs.soc_mwh / max(initial_energy_mwh, 1e-9),
        }
    )
    df["day_of_year"] = (df["hour_index"] // hours_per_day) + 1

    pivot = (
        df.pivot_table(index="day_of_year", columns="hour", values="soc_frac", aggfunc="mean")
        .reindex(index=pd.RangeIndex(1, day_count + 1, name="day_of_year"))
        .reindex(columns=pd.RangeIndex(0, 24, name="hour"), fill_value=np.nan)
    )
    return pivot


def prepare_charge_discharge_envelope(logs: HourlyLog) -> pd.DataFrame:
    """Aggregate charge/discharge distributions by hour for envelope plotting."""
    df = pd.DataFrame(
        {
            "hour": logs.hod.astype(int),
            "charge_mw": logs.charge_mw,
            "discharge_mw": logs.discharge_mw,
        }
    )
    grouped = df.groupby("hour").agg(
        charge_p05=("charge_mw", lambda s: s.quantile(0.05)),
        charge_p50=("charge_mw", "median"),
        charge_p95=("charge_mw", lambda s: s.quantile(0.95)),
        discharge_p05=("discharge_mw", lambda s: s.quantile(0.05)),
        discharge_p50=("discharge_mw", "median"),
        discharge_p95=("discharge_mw", lambda s: s.quantile(0.95)),
    )
    envelope = (
        grouped.reindex(pd.RangeIndex(0, 24, name="hour"))
        .fillna(0.0)
        .reset_index()
    )
    envelope["charge_low"] = -envelope["charge_p95"]
    envelope["charge_high"] = -envelope["charge_p05"]
    envelope["charge_median_neg"] = -envelope["charge_p50"]
    envelope["discharge_low"] = envelope["discharge_p05"]
    envelope["discharge_high"] = envelope["discharge_p95"]
    return envelope


def run_app():
    render_layout = init_page_layout(
        page_title="Simulation",
        main_title="BESS LAB — PV-only charging, AC-coupled",
        description="Configure inputs, run the simulation, and review per-year results and sensitivities.",
        base_dir=BASE_DIR,
    )
    with st.sidebar:
        st.header("Data Sources")
        st.caption(
            "Uploads are stored per-session. Use the landing page to preload data "
            "or override them below."
        )

        pv_file = st.file_uploader(
            "PV 8760 CSV (hour_index, pv_mw in MW)", type=["csv"], key="inputs_pv_upload"
        )
        cycle_file = st.file_uploader(
            "Cycle model Excel (optional override)", type=["xlsx"], key="inputs_cycle_upload"
        )
        pv_df, cycle_df = load_shared_data(BASE_DIR, pv_file, cycle_file)

        st.caption(
            "If no files are uploaded, built-in defaults are read from ./data/. "
            "Current session caches the latest uploads."
        )

        st.divider()
        st.subheader("Rate limit override")
        rate_limit_password = st.text_input(
            "Remove rate limit (password)",
            type="password",
            help=(
                "Enter the configured password to disable the session rate limit. "
                "If no secret is set, use 'besslab'."
            ),
            key="inputs_rate_limit_password",
        )
        expected_password = get_rate_limit_password()

        if rate_limit_password:
            if rate_limit_password == expected_password:
                st.session_state["rate_limit_bypass"] = True
                st.success("Rate limit disabled for this session.")
            else:
                st.session_state["rate_limit_bypass"] = False
                st.error("Incorrect password. Rate limit still active.")
        elif st.session_state.get("rate_limit_bypass", False):
            st.caption("Rate limit disabled for this session.")

    pv_df, cycle_df = render_layout(pv_df, cycle_df)

    st.subheader("Inputs")

    # Project & PV
    with st.expander("Project & PV", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            years = st.selectbox("Project life (years)", list(range(10, 36, 5)), index=2,
                help="Extend to test augmentation schedules and end effects.")
        with c2:
            pv_deg = st.number_input("PV degradation %/yr", 0.0, 5.0, 0.6, 0.1,
                help="Applied multiplicatively per year (e.g., 0.6% → (1−0.006)^year).") / 100.0
        with c3:
            pv_avail = st.slider("PV availability", 0.90, 1.00, 0.98, 0.01,
                help="Uptime factor applied to PV output.")

    # Availability
    with st.expander("Availability", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            bess_avail = st.slider("BESS availability", 0.90, 1.00, 0.99, 0.01,
                help="Uptime factor applied to BESS power capability.")
        with c2:
            use_split_rte = st.checkbox(
                "Use separate charge/discharge efficiencies",
                value=False,
                help="Select to enter distinct charge and discharge efficiencies instead of a single round-trip value.",
            )
            if use_split_rte:
                charge_eff = st.slider(
                    "Charge efficiency (AC-AC)",
                    0.70,
                    0.99,
                    0.94,
                    0.01,
                    help="Applied when absorbing energy; multiplied with discharge efficiency to form the round-trip value.",
                )
                discharge_eff = st.slider(
                    "Discharge efficiency (AC-AC)",
                    0.70,
                    0.99,
                    0.94,
                    0.01,
                    help="Applied when delivering energy; multiplied with charge efficiency to form the round-trip value.",
                )
                rte = charge_eff * discharge_eff
                st.caption(f"Implied round-trip efficiency: {rte:.3f} (charge × discharge).")
            else:
                rte = st.slider("Round-trip efficiency (single, at POI)", 0.70, 0.99, 0.88, 0.01,
                    help="Single RTE; internally split √RTE for charge/discharge.")
                charge_eff = None
                discharge_eff = None

    # BESS Specs
    with st.expander("BESS Specs (high-level)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            init_power = st.number_input("Power rating (MW)", 1.0, None, 30.0, 1.0,
                help="Initial nameplate power (POI context), before availability.")
        with c2:
            init_energy = st.number_input("Usable energy at BOL (MWh)", 1.0, None, 120.0, 1.0,
                help="Initial usable energy (POI context).")
        with c3:
            soc_floor = st.slider("SOC floor (%)", 0, 50, 10, 1,
                help="Reserve to protect cycling; lowers daily swing.") / 100.0
            soc_ceiling = st.slider("SOC ceiling (%)", 50, 100, 98, 1,
                help="Upper limit to protect cycling; raises daily swing when higher.") / 100.0

    # Dispatch
    with st.expander("Dispatch Strategy", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            contracted_mw = st.number_input("Contracted MW (firm)", 0.0, None, 30.0, 1.0,
                help="Firm capacity to meet during discharge windows.")
        with c2:
            discharge_windows_text = st.text_input("Discharge windows (HH:MM-HH:MM, comma-separated)",
                "10:00-14:00, 18:00-22:00",
                help="Ex: 10:00-14:00, 18:00-22:00")
        with c3:
            charge_windows_text = st.text_input("Charge windows (blank = any PV hours)", "",
                help="PV-only charging; blank allows any PV hour (even during discharge if PV surplus exists).")

    # Degradation
    with st.expander("Degradation modeling", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            cal_fade = st.number_input("Calendar fade %/yr (empirical)", 0.0, 5.0, 1.0, 0.1,
                help="Multiplicative retention: (1 − rate)^year.") / 100.0
        with c2:
            dod_override = st.selectbox("Degradation DoD basis",
                ["Auto (infer)", "10%", "20%", "40%", "80%", "100%"],
                help="Use the cycle table at a fixed DoD, or let the app infer based on median daily discharge.")

    manual_schedule_rows = st.session_state.get("manual_aug_schedule_rows")
    if manual_schedule_rows is None:
        manual_schedule_rows = [
            {"Year": 5, "Basis": AUGMENTATION_SCHEDULE_BASIS[0], "Amount": 10.0}
        ]
        st.session_state["manual_aug_schedule_rows"] = list(manual_schedule_rows)

    manual_schedule_df = pd.DataFrame(manual_schedule_rows)
    manual_schedule_entries: List[AugmentationScheduleEntry] = []
    manual_schedule_errors: List[str] = []

    # Augmentation (conditional, with explainers)
    with st.expander("Augmentation strategy", expanded=False):
        aug_mode = st.selectbox("Strategy", ["None", "Threshold", "Periodic", "Manual"], index=0)

        aug_size_mode = "Percent"
        aug_fixed_energy = 0.0
        retire_enabled = False
        retire_soh = 0.60

        if aug_mode == "Manual":
            st.caption(
                "Define explicit augmentation events by year. Each row adds capacity before other augmentation logic runs."
            )
            manual_schedule_df = st.data_editor(
                manual_schedule_df,
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
            st.session_state["manual_aug_schedule_rows"] = manual_schedule_df.to_dict("records")
            manual_schedule_entries, manual_schedule_errors = build_schedule_from_editor(
                manual_schedule_df, int(years)
            )
            if manual_schedule_errors:
                for err in manual_schedule_errors:
                    st.error(err)
            elif not manual_schedule_entries:
                st.warning("Add at least one row to run a manual augmentation schedule.")
            aug_thr_margin = 0.0; aug_topup = 0.0
            aug_every = 5; aug_frac = 0.10
            aug_trigger_type = "Capability"
            aug_soh_trig = 0.80; aug_soh_add = 0.10
        elif aug_mode == "Threshold":
            trigger = st.selectbox("Trigger type", ["Capability", "SOH"], index=0,
                help="Capability: Compare EOY capability vs target MWh/day.  SOH: Compare fleet SOH vs threshold.")
            if trigger == "Capability":
                c1, c2 = st.columns(2)
                with c1:
                    aug_thr_margin = st.number_input("Allowance margin (%)", 0.0, None, 0.0, 0.5,
                        help="Trigger when capability < target × (1 − margin).") / 100.0
                with c2:
                    aug_topup = st.number_input("Top-up margin (%)", 0.0, None, 5.0, 0.5,
                        help="Augment up to target × (1 + margin) when triggered.") / 100.0
                aug_every = 5; aug_frac = 0.10
                aug_trigger_type = "Capability"
                aug_soh_trig = 0.80; aug_soh_add = 0.10
            else:
                c1, c2 = st.columns(2)
                with c1:
                    aug_soh_trig = st.number_input("SOH trigger (%)", 50.0, 100.0, 80.0, 1.0,
                        help="If fleet SOH at year-end ≤ this threshold, augment.") / 100.0
                with c2:
                    aug_soh_add = st.number_input("Add % of initial BOL energy", 0.0, None, 10.0, 1.0,
                        help="Added energy as % of initial BOL. Power added to keep original C-hours.") / 100.0
                aug_thr_margin = 0.0; aug_topup = 0.0
                aug_every = 5; aug_frac = 0.10
                aug_trigger_type = "SOH"
        elif aug_mode == "Periodic":
            c1, c2 = st.columns(2)
            with c1:
                aug_every = st.number_input("Every N years", 1, None, 5, 1,
                    help="Add capacity on this cadence (e.g., every 5 years).")
            with c2:
                aug_frac = st.number_input("Add % of current BOL-ref energy", 0.0, None, 10.0, 1.0,
                    help="Top-up energy relative to current BOL reference.") / 100.0
            aug_thr_margin = 0.0; aug_topup = 0.0
            aug_trigger_type = "Capability"
            aug_soh_trig = 0.80; aug_soh_add = 0.10
        else:
            aug_thr_margin = 0.0; aug_topup = 0.0
            aug_every = 5; aug_frac = 0.10
            aug_trigger_type = "Capability"
            aug_soh_trig = 0.80; aug_soh_add = 0.10

        if aug_mode not in ["None", "Manual"]:
            aug_size_mode = st.selectbox(
                "Augmentation sizing",
                ["Percent", "Fixed"],
                format_func=lambda k: "% basis" if k == "Percent" else "Fixed energy (MWh)",
                help="Choose whether to size augmentation as a percent or a fixed MWh add.",
            )
            if aug_size_mode == "Fixed":
                aug_fixed_energy = st.number_input(
                    "Fixed energy added per event (MWh, BOL basis)",
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                    help="Adds this BOL-equivalent energy whenever augmentation is triggered.",
                )

        if aug_mode != "None":
            retire_enabled = st.checkbox(
                "Retire low-SOH cohorts when augmenting",
                value=False,
                help="Remove cohorts once their SOH falls below the retirement threshold.",
            )
            if retire_enabled:
                retire_soh = st.number_input(
                    "Retirement SOH threshold (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=60.0,
                    step=1.0,
                    help="Cohorts at or below this SOH are retired before applying augmentation.",
                ) / 100.0


    if aug_mode == "Manual" and (manual_schedule_errors or not manual_schedule_entries):
        st.error("Manual augmentation requires at least one valid year and no duplicate years.")
        st.stop()

    # Build config
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
        discharge_windows=parse_windows(discharge_windows_text),
        charge_windows_text=charge_windows_text,
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
        augmentation_schedule=list(manual_schedule_entries) if aug_mode == "Manual" else [],
    )

    inferred_step = infer_step_hours_from_pv(pv_df)
    if inferred_step is not None:
        cfg.step_hours = inferred_step

    duration_error = validate_pv_profile_duration(pv_df, cfg.step_hours)
    if duration_error:
        st.error(duration_error)
        st.stop()

    st.markdown("### Optional economics (NPV, IRR, LCOE, LCOS)")
    run_economics = st.checkbox(
        "Compute economics using simulation outputs",
        value=False,
        help=(
            "Enable to enter financial assumptions and derive LCOE/LCOS, NPV, and IRR "
            "from the simulated annual energy streams."
        ),
    )
    econ_inputs: Optional[EconomicInputs] = None
    price_inputs: Optional[PriceInputs] = None
    forex_rate_php_per_usd = 58.0
    devex_cost_usd = DEVEX_COST_PHP / forex_rate_php_per_usd

    default_contract_php_per_kwh = round(120.0 / 1000.0 * forex_rate_php_per_usd, 2)
    default_pv_php_per_kwh = round(55.0 / 1000.0 * forex_rate_php_per_usd, 2)
    wesm_reference_php_per_mwh = 5_583.0  # 2024 Annual Market Assessment Report, PEMC
    default_wesm_php_per_kwh = round(wesm_reference_php_per_mwh / 1000.0, 2)

    if run_economics:
        wesm_price_usd_per_mwh: Optional[float] = None
        wesm_price_php_per_kwh = default_wesm_php_per_kwh
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
            )
            inflation_pct = st.number_input(
                "Inflation rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=3.0,
                step=0.1,
                help="Used to derive the real discount rate applied to costs and revenues.",
            )
            discount_rate = max((1 + wacc_pct / 100.0) / (1 + inflation_pct / 100.0) - 1, 0.0)
            st.caption(f"Real discount rate derived from WACC and inflation: {discount_rate * 100:.2f}%.")
        with econ_col2:
            capex_musd = st.number_input(
                "Total CAPEX (USD million)",
                min_value=0.0,
                value=40.0,
                step=0.1,
            )
            fixed_opex_pct = (
                st.number_input(
                    "Fixed OPEX (% of CAPEX per year)",
                    min_value=0.0,
                    max_value=20.0,
                    value=2.0,
                    step=0.1,
                )
                / 100.0
            )
            fixed_opex_musd = st.number_input(
                "Additional fixed OPEX (USD million/yr)",
                min_value=0.0,
                value=0.0,
                step=0.1,
            )
            include_devex_year0 = st.checkbox(
                "Include ₱100M DevEx at year 0",
                value=False,
                help=(
                    "Adds a fixed ₱100 million development expenditure upfront (≈"
                    f"${devex_cost_usd / 1_000_000:,.2f}M using PHP {forex_rate_php_per_usd:,.0f}/USD). "
                    "Flows through discounted costs, LCOE/LCOS, NPV, and IRR."
                ),
            )
        with econ_col3:
            use_blended_price = st.checkbox(
                "Use blended energy price",
                value=False,
                help=(
                    "Apply a single price to all delivered firm energy and excess PV. "
                    "Contract/PV-specific inputs are ignored while enabled."
                ),
            )
            contract_price_php_per_kwh = st.number_input(
                "Contract price (PHP/kWh from BESS)",
                min_value=0.0,
                value=default_contract_php_per_kwh,
                step=0.05,
                help="Price converted to USD/MWh internally using PHP 58/USD.",
                disabled=use_blended_price,
            )
            pv_market_price_php_per_kwh = st.number_input(
                "PV market price (PHP/kWh for excess PV)",
                min_value=0.0,
                value=default_pv_php_per_kwh,
                step=0.05,
                help="Price converted to USD/MWh internally using PHP 58/USD.",
                disabled=use_blended_price,
            )
            blended_price_php_per_kwh = st.number_input(
                "Blended energy price (PHP/kWh)",
                min_value=0.0,
                value=default_contract_php_per_kwh,
                step=0.05,
                help=(
                    "Applied to all delivered firm energy and marketed PV when blended pricing is enabled."
                ),
                disabled=not use_blended_price,
            )
            escalate_prices = st.checkbox(
                "Escalate prices with inflation",
                value=False,
            )
            wesm_pricing_enabled = st.checkbox(
                "Apply WESM price to shortfalls",
                value=False,
                help=(
                    "Defaults to PHP 5,583/MWh from the 2024 Annual Market Assessment Report (PEMC);"
                    " enter a PHP/kWh rate to override."
                ),
            )
            wesm_price_php_per_kwh = st.number_input(
                "Average WESM price (PHP/kWh)",
                min_value=0.0,
                value=default_wesm_php_per_kwh,
                step=0.05,
                help=(
                    "Applied to shortfall MWh as either a purchase cost or sale credit."
                    " Defaults to PHP 5,583/MWh from the 2024 Annual Market Assessment Report (PEMC)."
                ),
                disabled=not wesm_pricing_enabled,
            )
            sell_to_wesm = st.checkbox(
                "Sell PV surplus to WESM",
                value=False,
                help=(
                    "When enabled, PV surplus (excess MWh) is credited at the WESM price; otherwise surplus"
                    " is excluded from revenue. Shortfalls always incur a WESM cost while this section is enabled."
                ),
                disabled=not wesm_pricing_enabled,
            )

            contract_price = contract_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
            pv_market_price = pv_market_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
            blended_price_usd_per_mwh: Optional[float] = None
            if use_blended_price:
                blended_price_usd_per_mwh = blended_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
                st.caption(
                    "Blended price active for revenues: "
                    f"PHP {blended_price_php_per_kwh:,.2f}/kWh "
                    f"(≈${blended_price_usd_per_mwh:,.2f}/MWh). Contract/PV prices are ignored."
                )
            else:
                st.caption(
                    f"Converted contract price: ${contract_price:,.2f}/MWh | PV market price: ${pv_market_price:,.2f}/MWh"
                )
            if wesm_pricing_enabled:
                wesm_price_usd_per_mwh = wesm_price_php_per_kwh / forex_rate_php_per_usd * 1000.0
                st.caption(
                    "WESM pricing active for shortfalls: "
                    f"PHP {wesm_price_php_per_kwh:,.2f}/kWh (≈${wesm_price_usd_per_mwh:,.2f}/MWh)."
                    " Defaults to PHP 5,583/MWh from the 2024 Annual Market Assessment Report (PEMC)."
                )

        variable_col1, variable_col2 = st.columns(2)
        with variable_col1:
            variable_opex_php_per_kwh = st.number_input(
                "Variable OPEX (PHP/kWh)",
                min_value=0.0,
                value=0.0,
                step=0.05,
                help=(
                    "Optional per-kWh operating expense applied to annual firm energy. "
                    "Escalates with inflation and overrides fixed OPEX when provided."
                ),
            )
            variable_opex_usd_per_mwh: Optional[float] = None
            if variable_opex_php_per_kwh > 0:
                variable_opex_usd_per_mwh = variable_opex_php_per_kwh / forex_rate_php_per_usd * 1000.0
                st.caption(
                    f"Converted variable OPEX: ${variable_opex_usd_per_mwh:,.2f}/MWh (applied to delivered energy)."
                )
        with variable_col2:
            variable_schedule_choice = st.radio(
                "Variable expense schedule",
                options=["None", "Periodic", "Custom"],
                horizontal=True,
                help=(
                    "Custom or periodic schedules override per-kWh and fixed OPEX assumptions. "
                    "Per-kWh costs override fixed percentages and adders."
                ),
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
                )
                periodic_variable_opex_interval_years = st.number_input(
                    "Cadence (years)",
                    min_value=1,
                    value=5,
                    step=1,
                )
                if periodic_variable_opex_usd <= 0:
                    periodic_variable_opex_usd = None
            elif variable_schedule_choice == "Custom":
                custom_variable_text = st.text_area(
                    "Custom variable expenses (USD/year)",
                    placeholder="e.g., 250000, 275000, 300000",
                    help=(
                        "Comma or newline separated values applied per project year. "
                        "Length must match the simulation horizon."
                    ),
                )
                if custom_variable_text.strip():
                    try:
                        variable_opex_schedule_usd = tuple(
                            parse_numeric_series("Variable expense schedule", custom_variable_text)
                        )
                    except ValueError:
                        st.stop()

        econ_inputs = EconomicInputs(
            capex_musd=capex_musd,
            fixed_opex_pct_of_capex=fixed_opex_pct,
            fixed_opex_musd=fixed_opex_musd,
            inflation_rate=inflation_pct / 100.0,
            discount_rate=discount_rate,
            variable_opex_usd_per_mwh=variable_opex_usd_per_mwh,
            variable_opex_schedule_usd=variable_opex_schedule_usd,
            periodic_variable_opex_usd=periodic_variable_opex_usd,
            periodic_variable_opex_interval_years=periodic_variable_opex_interval_years,
            devex_cost_usd=devex_cost_usd,
            include_devex_year0=include_devex_year0,
        )
        price_inputs = PriceInputs(
            contract_price_usd_per_mwh=contract_price,
            pv_market_price_usd_per_mwh=pv_market_price,
            escalate_with_inflation=escalate_prices,
            blended_price_usd_per_mwh=blended_price_usd_per_mwh,
            wesm_price_usd_per_mwh=wesm_price_usd_per_mwh,
            apply_wesm_to_shortfall=wesm_pricing_enabled,
            sell_to_wesm=sell_to_wesm if wesm_pricing_enabled else False,
        )

    # Store the latest input set for use in other pages (e.g., sweeps) without
    # tying the data to this page's widgets.
    st.session_state["latest_sim_config"] = cfg
    st.session_state["latest_dod_override"] = dod_override

    run_cols = st.columns([2, 1])
    with run_cols[0]:
        run_clicked = st.button(
            "Run simulation",
            use_container_width=True,
            help="Prevents auto-reruns while you adjust inputs; click to compute results.",
        )
    with run_cols[1]:
        st.caption("Edit parameters freely, then run when ready.")

    cached_results = st.session_state.get("latest_simulation_results")

    if not run_clicked and cached_results is None:
        st.info("Click 'Run simulation' to generate results after updating inputs.")
        st.caption("Use the batch tools or downloads to compare multiple runs.")
        st.page_link(
            "pages/05_Multi_Scenario_Batch.py",
            label="Open Multi-scenario batch",
            help="Run a structured set of variations for side-by-side review.",
        )
        st.page_link(
            "pages/04_BESS_Sizing_Sweep.py",
            label="Open BESS sizing sweep",
            help="Rank feasible usable-energy variants using the latest inputs.",
        )
        st.stop()

    if run_clicked or cached_results is None:
        enforce_rate_limit()

        try:
            sim_output = simulate_project(cfg, pv_df, cycle_df, dod_override)
        except ValueError as exc:  # noqa: BLE001
            st.error(str(exc))
            st.stop()

        st.session_state["latest_simulation_results"] = {
            "sim_output": sim_output,
            "dod_override": dod_override,
        }
    else:
        sim_output = cached_results["sim_output"]
        st.caption(
            "Showing the latest completed simulation. Click 'Run simulation' to refresh after editing inputs."
        )

    results = sim_output.results
    monthly_results_all = sim_output.monthly_results
    first_year_logs = sim_output.first_year_logs
    final_year_logs = sim_output.final_year_logs
    hod_count = sim_output.hod_count
    hod_sum_pv = sim_output.hod_sum_pv
    hod_sum_pv_resource = sim_output.hod_sum_pv_resource
    hod_sum_bess = sim_output.hod_sum_bess
    hod_sum_charge = sim_output.hod_sum_charge
    dis_hours_per_day = sim_output.discharge_hours_per_day

    # Yearly table
    res_df = pd.DataFrame([{
        'Year': r.year_index,
        'Expected firm MWh': r.expected_firm_mwh,
        'Delivered firm MWh': r.delivered_firm_mwh,
        'Shortfall MWh': r.shortfall_mwh,
        'Breach days (has any shortfall)': r.breach_days,
        'Charge MWh': r.charge_mwh,
        'Discharge MWh (from BESS)': r.discharge_mwh,
        'Available PV MWh': r.available_pv_mwh,
        'PV→Contract MWh': r.pv_to_contract_mwh,
        'BESS→Contract MWh': r.bess_to_contract_mwh,
        'Avg RTE': r.avg_rte,
        'Eq cycles (year)': r.eq_cycles,
        'Cum cycles': r.cum_cycles,
        'SOH_cycle': r.soh_cycle,
        'SOH_calendar': r.soh_calendar,
        'SOH_total': r.soh_total,
        'EOY usable MWh': r.eoy_usable_mwh,
        'EOY power MW (avail-adjusted)': r.eoy_power_mw,
        'PV curtailed MWh': r.pv_curtailed_mwh,
    } for r in results])

    monthly_df = pd.DataFrame([{
        'Year': m.year_index,
        'Month': m.month_label,
        'Expected firm MWh': m.expected_firm_mwh,
        'Delivered firm MWh': m.delivered_firm_mwh,
        'Shortfall MWh': m.shortfall_mwh,
        'Breach days (has any shortfall)': m.breach_days,
        'Charge MWh': m.charge_mwh,
        'Discharge MWh (from BESS)': m.discharge_mwh,
        'Available PV MWh': m.available_pv_mwh,
        'PV→Contract MWh': m.pv_to_contract_mwh,
        'BESS→Contract MWh': m.bess_to_contract_mwh,
        'Avg RTE': m.avg_rte,
        'Eq cycles (year)': m.eq_cycles,
        'Cum cycles': m.cum_cycles,
        'SOH_cycle': m.soh_cycle,
        'SOH_calendar': m.soh_calendar,
        'SOH_total': m.soh_total,
        'EOY usable MWh': m.eom_usable_mwh,
        'EOY power MW (avail-adjusted)': m.eom_power_mw,
        'PV curtailed MWh': m.pv_curtailed_mwh,
    } for m in monthly_results_all])

    # --------- KPIs ---------
    final = results[-1]
    summary = summarize_simulation(sim_output)
    compliance = summary.compliance
    bess_share_of_firm = summary.bess_share_of_firm
    charge_discharge_ratio = summary.charge_discharge_ratio
    pv_capture_ratio = summary.pv_capture_ratio
    discharge_capacity_factor = summary.discharge_capacity_factor
    total_project_generation_mwh = summary.total_project_generation_mwh
    bess_generation_mwh = summary.bess_generation_mwh
    pv_generation_mwh = summary.pv_generation_mwh
    pv_excess_mwh = summary.pv_excess_mwh
    bess_losses_mwh = summary.bess_losses_mwh
    total_shortfall_mwh = summary.total_shortfall_mwh
    avg_eq_cycles_per_year = summary.avg_eq_cycles_per_year
    cap_ratio_final = summary.cap_ratio_final
    expected_total_mwh = sum(r.expected_firm_mwh for r in results)
    coverage_by_year = [
        (r.delivered_firm_mwh / r.expected_firm_mwh) if r.expected_firm_mwh > 0 else float('nan')
        for r in results
    ]
    min_yearly_coverage = float(np.nanmin(coverage_by_year)) if coverage_by_year else float('nan')
    final_soh_pct = final.soh_total * 100.0
    eoy_capacity_margin_pct = cap_ratio_final * 100.0
    augmentation_events = sim_output.augmentation_events
    augmentation_energy_mwh = float(np.sum(sim_output.augmentation_energy_added_mwh)) if sim_output.augmentation_energy_added_mwh else 0.0

    econ_outputs: Optional[EconomicOutputs] = None
    cash_outputs: Optional[CashFlowOutputs] = None
    augmentation_costs_usd: Optional[List[float]] = None

    if run_economics and econ_inputs and price_inputs:
        augmentation_costs_usd = estimate_augmentation_costs_by_year(
            sim_output.augmentation_energy_added_mwh,
            cfg.initial_usable_mwh,
            econ_inputs.capex_musd,
        )
        if any(augmentation_costs_usd):
            st.caption(
                "Augmentation CAPEX derived from the strategy (proportional to the share of BOL energy added)."
            )

        annual_delivered = [r.delivered_firm_mwh for r in results]
        annual_bess = [r.bess_to_contract_mwh for r in results]
        annual_pv_excess = [r.pv_curtailed_mwh for r in results]
        annual_shortfall = [r.shortfall_mwh for r in results]

        try:
            econ_outputs = compute_lcoe_lcos_with_augmentation_fallback(
                annual_delivered,
                annual_bess,
                econ_inputs,
                augmentation_costs_usd=augmentation_costs_usd if augmentation_costs_usd else None,
            )
            cash_outputs = compute_cash_flows_and_irr(
                annual_delivered,
                annual_bess,
                annual_pv_excess,
                econ_inputs,
                price_inputs,
                annual_shortfall_mwh=annual_shortfall,
                augmentation_costs_usd=augmentation_costs_usd if augmentation_costs_usd else None,
            )
        except ValueError as exc:  # noqa: BLE001
            st.error(str(exc))
            st.stop()

    st.session_state["latest_simulation_snapshot"] = {
        'Contracted MW': cfg.contracted_mw,
        'Power (BOL MW)': cfg.initial_power_mw,
        'Usable (BOL MWh)': cfg.initial_usable_mwh,
        'Discharge windows': discharge_windows_text,
        'Charge windows': charge_windows_text if charge_windows_text else 'Any PV hour',
        'Compliance (%)': compliance,
        'BESS share of firm (%)': bess_share_of_firm,
        'Charge/Discharge ratio': charge_discharge_ratio,
        'PV capture ratio': pv_capture_ratio,
        'Total project generation (MWh)': total_project_generation_mwh,
        'BESS share of generation (MWh)': bess_generation_mwh,
        'PV share of generation (MWh)': pv_generation_mwh,
        'PV excess (MWh)': pv_excess_mwh,
        'BESS losses (MWh)': bess_losses_mwh,
        'Final EOY usable (MWh)': final.eoy_usable_mwh,
        'Final EOY power (MW)': final.eoy_power_mw,
        'Final eq cycles (year)': final.eq_cycles,
        'Final SOH_total': final.soh_total,
    }

    def _fmt_percent(value: float, as_fraction: bool = False) -> str:
        if math.isnan(value):
            return "—"
        pct_value = value * 100.0 if as_fraction else value
        return f"{pct_value:,.2f}%"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Delivery compliance", _fmt_percent(compliance), help="Total firm energy delivered vs contracted across project life.")
    c2.metric("Worst-year coverage", _fmt_percent(min_yearly_coverage, as_fraction=True), help="Lowest annual delivery vs contract shows weakest year.")
    c3.metric("Final SOH_total", _fmt_percent(final_soh_pct, as_fraction=False), help="End-of-life usable fraction after cycle + calendar fade.")
    c4.metric("EOY deliverable vs contract", _fmt_percent(eoy_capacity_margin_pct, as_fraction=False), help="Final-year daily deliverable vs target day (MW×h window).")
    c5.metric(
        "Augmentations triggered",
        f"{augmentation_events} events",
        help=f"Energy added over life: {augmentation_energy_mwh:,.0f} MWh (BOL basis).",
    )

    if run_economics and econ_outputs and cash_outputs:
        st.markdown("### Economics summary")
        econ_c1, econ_c2, econ_c3 = st.columns(3)
        econ_c1.metric(
            "Discounted costs (USD million)",
            f"{econ_outputs.discounted_costs_usd / 1_000_000:,.2f}",
            help="CAPEX at year 0 plus discounted OPEX and augmentation across the project horizon.",
        )
        php_per_kwh_factor = forex_rate_php_per_usd / 1000.0
        lcoe_php_per_kwh = econ_outputs.lcoe_usd_per_mwh * php_per_kwh_factor
        lcos_php_per_kwh = econ_outputs.lcos_usd_per_mwh * php_per_kwh_factor

        econ_c2.metric(
            "LCOE (PHP/kWh delivered)",
            f"{lcoe_php_per_kwh:,.2f}",
            help=(
                "Total discounted costs ÷ discounted firm energy delivered, converted using "
                f"PHP {forex_rate_php_per_usd:,.0f}/USD."
            ),
        )
        econ_c3.metric(
            "LCOS (PHP/kWh from BESS)",
            f"{lcos_php_per_kwh:,.2f}",
            help=(
                "Same cost base divided by discounted BESS contribution only, converted with the "
                f"PHP {forex_rate_php_per_usd:,.0f}/USD rate."
            ),
        )

        if econ_inputs.include_devex_year0:
            st.caption(
                "DevEx: Included an additional ₱100M "
                f"(≈${econ_inputs.devex_cost_usd / 1_000_000:,.2f}M) at year 0 across discounted costs, "
                "LCOE/LCOS, NPV, and IRR."
            )
        else:
            st.caption("DevEx not included; upfront spend reflects CAPEX only.")

        cash_c1, cash_c2, cash_c3 = st.columns(3)
        revenue_help = (
            "Revenues apply the blended energy price to all BESS deliveries and excess PV; "
            "contract/PV-specific rates are ignored in this mode."
            if price_inputs.blended_price_usd_per_mwh is not None
            else "Contract revenue from BESS deliveries plus market revenue from excess PV."
        )
        if price_inputs.apply_wesm_to_shortfall and price_inputs.wesm_price_usd_per_mwh is not None:
            revenue_help += (
                f" Shortfall MWh are deducted as a WESM cost at ${price_inputs.wesm_price_usd_per_mwh:,.2f}/MWh."
            )
            if price_inputs.sell_to_wesm:
                revenue_help += (
                    " PV surplus is credited at the same WESM rate; otherwise surplus is excluded from revenue."
                )
        cash_c1.metric(
            "Discounted revenues (USD million)",
            f"{cash_outputs.discounted_revenues_usd / 1_000_000:,.2f}",
            help=revenue_help,
        )
        cash_c2.metric(
            "NPV (USD million)",
            f"{cash_outputs.npv_usd / 1_000_000:,.2f}",
            help="Discounted cash flows using the chosen discount rate (year 0 CAPEX included).",
        )
        cash_c3.metric(
            "Project IRR (%)",
            f"{cash_outputs.irr_pct:,.2f}%" if cash_outputs.irr_pct == cash_outputs.irr_pct else "—",
            help="IRR computed from annual revenues and OPEX/augmentation outflows.",
        )

        wesm_caption = (
            "WESM pricing disabled; contract shortfalls are not monetized in revenues, NPV, or IRR."
        )
        if price_inputs.apply_wesm_to_shortfall and price_inputs.wesm_price_usd_per_mwh is not None:
            wesm_impact_musd = cash_outputs.discounted_wesm_value_usd / 1_000_000
            surplus_note = (
                " PV surplus credited at WESM due to the sell toggle."
                if price_inputs.sell_to_wesm
                else " PV surplus excluded from revenue while WESM pricing is active."
            )
            wesm_caption = (
                "WESM shortfall cost applied at PHP "
                f"{wesm_price_php_per_kwh:,.2f}/kWh (≈${price_inputs.wesm_price_usd_per_mwh:,.2f}/MWh)."
                f" Discounted WESM impact on revenues/NPV/IRR: {wesm_impact_musd:,.2f} USD million." + surplus_note
            )

        st.caption(wesm_caption)

    st.markdown("#### Augmentation impact trace")
    augmentation_retired = getattr(
        sim_output, "augmentation_retired_energy_mwh", [0.0 for _ in sim_output.augmentation_energy_added_mwh]
    )
    if len(augmentation_retired) < len(sim_output.augmentation_energy_added_mwh):
        augmentation_retired.extend([0.0] * (len(sim_output.augmentation_energy_added_mwh) - len(augmentation_retired)))
    elif len(augmentation_retired) > len(sim_output.augmentation_energy_added_mwh):
        augmentation_retired = augmentation_retired[: len(sim_output.augmentation_energy_added_mwh)]

    coverage_pct_by_year = [
        (r.delivered_firm_mwh / r.expected_firm_mwh * 100.0) if r.expected_firm_mwh > 0 else float('nan')
        for r in results
    ]
    delivered_by_year = [r.delivered_firm_mwh for r in results]

    aug_rows: List[Dict[str, Any]] = []
    for idx, add_e in enumerate(sim_output.augmentation_energy_added_mwh):
        retired_e = augmentation_retired[idx] if idx < len(augmentation_retired) else 0.0
        if add_e <= 0.0 and retired_e <= 0.0:
            continue

        coverage_pct = coverage_pct_by_year[idx]
        coverage_delta = coverage_pct_by_year[idx] - coverage_pct_by_year[idx - 1] if idx > 0 else float('nan')
        gen_delta = delivered_by_year[idx] - delivered_by_year[idx - 1] if idx > 0 else float('nan')
        add_pct_bol = (add_e / cfg.initial_usable_mwh * 100.0) if cfg.initial_usable_mwh > 0 else float('nan')

        aug_rows.append({
            "Year": idx + 1,
            "Added (MWh BOL)": add_e,
            "Added vs BOL (%)": add_pct_bol,
            "Retired cohorts (MWh BOL)": retired_e,
            "Coverage (%)": coverage_pct,
            "Coverage Δ (pp)": coverage_delta,
            "Generation Δ (MWh)": gen_delta,
        })

    if aug_rows:
        aug_df = pd.DataFrame(aug_rows)
        st.caption("Per-event summary combines augmentation size, cohort retirements, and year-over-year shifts in compliance and delivered energy.")
        st.dataframe(
            aug_df.style.format({
                "Added (MWh BOL)": "{:.1f}",
                "Added vs BOL (%)": "{:.2f}",
                "Retired cohorts (MWh BOL)": "{:.1f}",
                "Coverage (%)": "{:.2f}",
                "Coverage Δ (pp)": "{:.2f}",
                "Generation Δ (MWh)": "{:.1f}",
            }),
            use_container_width=True,
        )

        delta_df = aug_df.melt(
            id_vars=["Year"],
            value_vars=["Coverage Δ (pp)", "Generation Δ (MWh)"],
            var_name="Metric",
            value_name="Delta",
        ).dropna(subset=["Delta"])

        if not delta_df.empty:
            delta_chart = alt.Chart(delta_df).mark_bar().encode(
                x=alt.X("Year:O", title="Augmentation year"),
                y=alt.Y("Delta:Q", title="Annual change"),
                color=alt.Color("Metric:N", title=""),
                tooltip=["Year", "Metric", alt.Tooltip("Delta:Q", format=".2f")],
            ).properties(title="Year-over-year shifts at augmentation points")
            st.altair_chart(delta_chart, use_container_width=True)
    else:
        st.info("No augmentation or retirement events were triggered in this run.")


    st.markdown("---")
    st.subheader("Yearly Summary")
    st.dataframe(res_df.style.format({
        'Expected firm MWh': '{:,.1f}',
        'Delivered firm MWh': '{:,.1f}',
        'Shortfall MWh': '{:,.1f}',
        'Charge MWh': '{:,.1f}',
        'Discharge MWh (from BESS)': '{:,.1f}',
        'Available PV MWh': '{:,.1f}',
        'PV→Contract MWh': '{:,.1f}',
        'BESS→Contract MWh': '{:,.1f}',
        'Avg RTE': '{:,.3f}',
        'Eq cycles (year)': '{:,.1f}',
        'Cum cycles': '{:,.1f}',
        'SOH_cycle': '{:,.3f}',
        'SOH_calendar': '{:,.3f}',
        'SOH_total': '{:,.3f}',
        'EOY usable MWh': '{:,.1f}',
        'EOY power MW (avail-adjusted)': '{:,.1f}',
        'PV curtailed MWh': '{:,.1f}',
    }))

    with st.expander("Monthly summary preview", expanded=False):
        st.dataframe(monthly_df.style.format({
            'Expected firm MWh': '{:,.1f}',
            'Delivered firm MWh': '{:,.1f}',
            'Shortfall MWh': '{:,.1f}',
            'Charge MWh': '{:,.1f}',
            'Discharge MWh (from BESS)': '{:,.1f}',
            'Available PV MWh': '{:,.1f}',
            'PV→Contract MWh': '{:,.1f}',
            'BESS→Contract MWh': '{:,.1f}',
            'Avg RTE': '{:,.3f}',
            'Eq cycles (year)': '{:,.1f}',
            'Cum cycles': '{:,.1f}',
            'SOH_cycle': '{:,.3f}',
            'SOH_calendar': '{:,.3f}',
            'SOH_total': '{:,.3f}',
            'EOY usable MWh': '{:,.1f}',
            'EOY power MW (avail-adjusted)': '{:,.1f}',
            'PV curtailed MWh': '{:,.1f}',
        }))

    # ---------- EOY Delivered Firm Split (per day): PV vs BESS ----------
    st.subheader("EOY Delivered Firm Split (per day) — PV vs BESS")
    target_daily_mwh = cfg.contracted_mw * dis_hours_per_day
    years_list = [r.year_index for r in results]
    deliv_df = pd.DataFrame({
        'Year': years_list,
        'PV→Contract (MWh/day)': [r.pv_to_contract_mwh/365.0 for r in results],
        'BESS→Contract (MWh/day)': [r.bess_to_contract_mwh/365.0 for r in results],
        'Target firm (MWh/day)': [target_daily_mwh]*len(years_list),
    })
    deliv_long = deliv_df.melt(id_vars='Year', value_vars=['PV→Contract (MWh/day)', 'BESS→Contract (MWh/day)'],
                               var_name='Source', value_name='MWh/day')

    bar2 = alt.Chart(deliv_long).mark_bar().encode(
        x=alt.X('Year:O', title='Year'),
        y=alt.Y('MWh/day:Q', title='MWh/day'),
        color=alt.Color('Source:N', scale=alt.Scale(range=['#86c5da', '#7fd18b']))
    )
    line2 = alt.Chart(deliv_df).mark_line(point=True, color='#f2a900').encode(
        x='Year:O',
        y='Target firm (MWh/day):Q',
    )
    st.altair_chart(bar2 + line2, use_container_width=True)

    # ---------- Flags ----------
    st.subheader("Flags & Guidance")
    flag_totals = {
        'firm_shortfall_hours': sum(r.flags['firm_shortfall_hours'] for r in results),
        'soc_floor_hits': sum(r.flags['soc_floor_hits'] for r in results),
        'soc_ceiling_hits': sum(r.flags['soc_ceiling_hits'] for r in results),
    }
    f1, f2, f3 = st.columns(3)
    for key, col in zip(
        ["firm_shortfall_hours", "soc_floor_hits", "soc_ceiling_hits"],
        [f1, f2, f3],
    ):
        meta = FLAG_DEFINITIONS[key]
        col.metric(meta["label"], f"{flag_totals[key]:,}")
        col.caption(f"Meaning: {meta['meaning']}\nFix knobs: {meta['knobs']}")

    insights = build_flag_insights(flag_totals)
    st.markdown("**What the flags suggest:**")
    st.markdown("\n".join(f"- {tip}" for tip in insights))

    st.markdown("---")

    # ---------- Design Advisor (physics-bounded) ----------
    st.subheader("Design Advisor (final-year, physics-bounded)")

    # --- Bounds / guardrails (editable if you like) ---
    RTE_RT_MAX = 0.92              # plausible AC-to-AC roundtrip limit
    SOC_FLOOR_MIN = 0.05           # don't recommend below this
    SOC_CEILING_MAX = 0.98         # don't recommend above this
    DELTA_SOC_MAX = 0.90           # ~5-95%
    EFC_YR_GREEN = 300.0           # vendor guardrail
    EFC_YR_YELLOW = 400.0

    # --- Final-year context ---
    eta_ch_now, eta_dis_now, eta_rt_now = resolve_efficiencies(cfg)
    eta_ratio = eta_dis_now / max(1e-9, eta_ch_now)
    delta_soc_now = max(0.0, cfg.soc_ceiling - cfg.soc_floor)
    delta_soc_cap = min(DELTA_SOC_MAX, SOC_CEILING_MAX - SOC_FLOOR_MIN)
    soh_final = float(final.soh_total)

    target_day = cfg.contracted_mw * dis_hours_per_day                    # MWh/day
    pv_to_contract_day = final.pv_to_contract_mwh / 365.0                 # MWh/day
    bess_share_day = max(0.0, target_day - pv_to_contract_day)            # MWh/day BESS must supply

    deliverable_day_now = cfg.initial_usable_mwh * soh_final * delta_soc_now * eta_dis_now
    shortfall_day_now = max(0.0, target_day - deliverable_day_now)

    colA, colB, colC, colD = st.columns(4)
    colA.metric("Deliverable/day now (final yr)", f"{deliverable_day_now:,.1f} MWh")
    colB.metric("Shortfall/day (final yr)", f"{shortfall_day_now:,.1f} MWh")
    colC.metric("Target/day", f"{target_day:,.1f} MWh")
    colD.metric("EOY power avail (final)", f"{final.eoy_power_mw:,.2f} MW",
                help="Availability-adjusted final-year power capability.")

    # Quick severity read: give a nudge before diving into options
    gap_ratio = shortfall_day_now / target_day if target_day else 0.0
    if shortfall_day_now <= 0.05:  # effectively on target
        st.success("Final-year deliverable meets the target within rounding noise.")
    elif gap_ratio <= 0.10:
        st.info("Minor gap: adjust one knob below to clear a small shortfall.")
    else:
        st.warning("Material gap: use the action ladder below to close the deficit.")

    # --- 1) Power vs Energy limiter ---
    if final.eoy_power_mw + 1e-9 < cfg.contracted_mw:
        st.error(
            f"Power-limited: final-year available power {final.eoy_power_mw:.2f} MW "
            f"is below contract {cfg.contracted_mw:.2f} MW."
        )
        need = cfg.contracted_mw - final.eoy_power_mw
        st.markdown(
            f"- **Option D (Power)**: Increase power (final-year, avail-adjusted) by **{need:.2f} MW**, "
            f"or reduce contract MW / shift windows."
        )
    else:
        # --- Energy-limited path ---
        st.caption("Energy-limited in final year (power is sufficient).")

        # --- 2) Sequential bounded solve: ΔSOC → RTE → Energy ---
        # a) try to meet target with ΔSOC first (bounded)
        req_delta_soc_at_current = target_day / max(1e-9, cfg.initial_usable_mwh * soh_final * eta_dis_now)
        delta_soc_adopt = min(delta_soc_cap, max(delta_soc_now, req_delta_soc_at_current))

        # b) then RTE (bounded)
        req_eta_dis_at_soc = min(0.9999, max(0.0, target_day / max(1e-9, cfg.initial_usable_mwh * soh_final * delta_soc_adopt)))
        req_rte_rt_at_soc = min(0.9999, max(0.0, (req_eta_dis_at_soc ** 2) / max(1e-9, eta_ratio)))
        rte_rt_adopt = min(RTE_RT_MAX, max(eta_rt_now, req_rte_rt_at_soc))
        eta_dis_adopt = min(0.9999, math.sqrt(rte_rt_adopt * eta_ratio))

        # c) finally BOL energy to close any remaining gap
        ebol_req = target_day / max(1e-9, soh_final * delta_soc_adopt * eta_dis_adopt)
        ebol_delta = max(0.0, ebol_req - cfg.initial_usable_mwh)

        # Helper to render SOC variant text (raise ceiling vs lower floor)
        def soc_variant_text(delta_soc_goal: float) -> str:
            # choice A: keep floor, raise ceiling
            ceil_needed = min(SOC_CEILING_MAX, cfg.soc_floor + delta_soc_goal)
            # choice B: keep ceiling, lower floor
            floor_needed = max(SOC_FLOOR_MIN, cfg.soc_ceiling - delta_soc_goal)
            return (f"(e.g., keep floor at {cfg.soc_floor*100:.0f}% → raise ceiling to **{ceil_needed*100:.0f}%**, "
                    f"or keep ceiling at {cfg.soc_ceiling*100:.0f}% → lower floor to **{floor_needed*100:.0f}%**).")

        # How far each knob alone would push deliverable/day
        deliverable_soc_only = cfg.initial_usable_mwh * soh_final * delta_soc_adopt * eta_dis_now
        deliverable_soc_rte = cfg.initial_usable_mwh * soh_final * delta_soc_adopt * eta_dis_adopt
        deliverable_full = ebol_req * soh_final * delta_soc_adopt * eta_dis_adopt

        # --- 3) PV charge sufficiency check under the adopted RTE ---
        pv_charge_req_day = bess_share_day / max(1e-9, rte_rt_adopt)   # MWh/day needed from PV to charge
        charged_day_now = final.charge_mwh / 365.0                     # MWh/day currently charged
        charge_deficit_day = max(0.0, pv_charge_req_day - charged_day_now)
        extra_charge_hours_day = charge_deficit_day / max(1e-9, final.eoy_power_mw)

        # --- 4) Implied cycles guardrail under the proposed ΔSOC/Ebol ---
        def dod_from_delta_soc(ds: float) -> int:
            return 100 if ds >= 0.90 else (80 if ds >= 0.80 else (40 if ds >= 0.40 else (20 if ds >= 0.20 else 10)))
        dod_key_prop = dod_from_delta_soc(delta_soc_adopt)
        dod_frac_map = {10:0.10,20:0.20,40:0.40,80:0.80,100:1.00}
        dod_frac_prop = dod_frac_map[dod_key_prop]
        efc_year_prop = (bess_share_day * 365.0) / max(1e-9, ebol_req * dod_frac_prop)

        # --- 5) Print bounded options ---
        opts = []

        # OPTION A — ΔSOC only (bounded to cap); if still short, explain why it's insufficient alone
        if delta_soc_now + 1e-9 < delta_soc_cap:
            need_soc = max(0.0, delta_soc_adopt - delta_soc_now) * 100.0
            # re-compute Ebol needed if we keep RTE at current (ΔSOC only)
            ebol_req_soc_only = target_day / max(1e-9, soh_final * delta_soc_adopt * eta_dis_now)
            short_if_only_soc = max(0.0, ebol_req_soc_only - cfg.initial_usable_mwh)
            if short_if_only_soc <= 1e-6:
                opts.append(f"- **Option A (ΔSOC)**: Widen ΔSOC to **{delta_soc_adopt*100:,.1f}%** {soc_variant_text(delta_soc_adopt)}")
            else:
                opts.append(f"- **Option A (ΔSOC)**: Widen ΔSOC to **{delta_soc_adopt*100:,.1f}%** {soc_variant_text(delta_soc_adopt)} "
                            f"→ still short on energy by **{short_if_only_soc:,.1f} MWh** (at current RTE).")
        else:
            opts.append(f"- **Option A (ΔSOC)**: Already at cap (**{delta_soc_now*100:,.1f}%**).")

        # OPTION B — ΔSOC (adopted) + RTE (bounded)
        if rte_rt_adopt > eta_rt_now + 1e-9:
            opts.append(f"- **Option B (ΔSOC + RTE)**: Keep ΔSOC at **{delta_soc_adopt*100:,.1f}%** and improve roundtrip RTE to "
                        f"**{rte_rt_adopt*100:,.1f}%** (cap {RTE_RT_MAX*100:.0f}%).")
        else:
            opts.append(f"- **Option B (ΔSOC + RTE)**: RTE already at limit for this option (current {eta_rt_now*100:.1f}%, cap {RTE_RT_MAX*100:.0f}%).")

        # OPTION C — Energy/contract levers when ΔSOC + RTE still fall short
        if ebol_delta > 1e-6:
            contract_with_current_energy = deliverable_soc_rte / max(1e-9, dis_hours_per_day)
            opts.append(
                f"- **Option C (Energy/contract)**: Need ~+{ebol_delta:,.1f} MWh usable (to {ebol_req:,.1f} MWh) to hit the full target. "
                f"If adding that at BOL is impractical, consider **staged Threshold/SOH augmentation** or **right-size the contract to ~{contract_with_current_energy:,.2f} MW** under the proposed ΔSOC/RTE."
            )
        else:
            opts.append(f"- **Option C (Energy/contract)**: BOL usable is sufficient under the adopted ΔSOC/RTE.")

        st.markdown("**Bounded recommendations:**")
        st.markdown("\n".join(opts))

        # --- 5b) Action ladder (fastest wins first) ---
        action_ladder: List[str] = []
        if delta_soc_adopt > delta_soc_now + 1e-9:
            action_ladder.append(
                f"**Widen ΔSOC** to **{delta_soc_adopt*100:,.1f}%** → delivers ~{deliverable_soc_only:,.1f} MWh/day."
            )
        if rte_rt_adopt > eta_rt_now + 1e-9:
            action_ladder.append(
                f"**Improve roundtrip RTE** to **{rte_rt_adopt*100:,.1f}%** → delivers ~{deliverable_soc_rte:,.1f} MWh/day."
            )
        if ebol_delta > 1e-6:
            contract_with_current_energy = deliverable_soc_rte / max(1e-9, dis_hours_per_day)
            action_ladder.append(
                f"**Close remaining energy gap**: either plan staged augmentation (~+{ebol_delta:,.1f} MWh usable over life) or resize contract toward **{contract_with_current_energy:,.2f} MW** so ΔSOC/RTE improvements can carry the final year."
            )
        if charge_deficit_day > 1e-3:
            action_ladder.append(
                f"**Widen charge window** by **+{extra_charge_hours_day:,.2f} h/day** or create shoulder headroom to absorb PV."
            )
        if not action_ladder:
            action_ladder.append("All knobs already at bounds for the target—consider reducing the contract or shifting delivery windows.")

        st.markdown("**Action ladder (work down the list):**")
        st.markdown("\n".join(f"- {idx}) {item}" for idx, item in enumerate(action_ladder, start=1)))

        # --- 6) PV charge sufficiency + charge-hours hint ---
        st.caption(
            f"PV charge required/day for BESS share ≈ **{pv_charge_req_day:,.1f} MWh** "
            f"(BESS share {bess_share_day:,.1f} ÷ RTE {rte_rt_adopt:.2f}). "
            f"Currently charging **{charged_day_now:,.1f} MWh/day**."
        )
        if charge_deficit_day > 1e-3:
            st.warning(
                f"PV charge **insufficient** by **{charge_deficit_day:,.1f} MWh/day** in final year. "
                f"At {final.eoy_power_mw:.1f} MW charge power, this needs **+{extra_charge_hours_day:,.2f} h/day** "
                f"of charge window or equivalent **shoulder discharge** to create headroom while PV is up."
            )
        else:
            st.success("PV charge looks sufficient at the proposed settings.")

        # --- 7) Implied EFC guardrail
        if efc_year_prop > EFC_YR_YELLOW:
            st.error(
                f"Implied **EqCycles/yr ≈ {efc_year_prop:,.0f}** (ΔSOC bucket {dod_key_prop}%): exceeds typical guardrails. "
                "Prefer **augmentation** (Threshold/SOH) or reduce ΔSOC."
            )
        elif efc_year_prop > EFC_YR_GREEN:
            st.warning(
                f"Implied cycles ≈ **{efc_year_prop:,.0f} EFC/yr** at proposed settings; check warranty guardrails "
                f"(soft limit {EFC_YR_GREEN:.0f}, hard {EFC_YR_YELLOW:.0f})."
            )
        else:
            st.caption(
                f"Implied cycles ≈ {efc_year_prop:,.0f} EFC/yr at the proposed ΔSOC bucket ({dod_key_prop}%)."
            )

    st.markdown("---")

    # ---------- Average Daily Profile ----------
    st.subheader("Average Daily Profile — PV & BESS contributions to contract; charging shown below zero")

    def _avg_profile_df_from_logs(logs: HourlyLog, cfg: SimConfig) -> pd.DataFrame:
        contracted_series = np.array([
            cfg.contracted_mw if any(w.contains(int(h)) for w in cfg.discharge_windows) else 0.0
            for h in logs.hod
        ], dtype=float)
        df_hr = pd.DataFrame({
            'hod': logs.hod.astype(int),
            'pv_resource_mw': logs.pv_mw,
            'pv_to_contract_mw': logs.pv_to_contract_mw,
            'bess_to_contract_mw': logs.bess_to_contract_mw,
            'charge_mw': logs.charge_mw,
            'contracted_mw': contracted_series,
        })
        avg = df_hr.groupby('hod', as_index=False).mean().rename(columns={'hod': 'hour'})
        avg['pv_surplus_mw'] = np.maximum(avg['pv_resource_mw'] - avg['pv_to_contract_mw'] - avg['charge_mw'], 0.0)
        avg['charge_mw_neg'] = -avg['charge_mw']
        return avg[[
            'hour',
            'pv_resource_mw',
            'pv_to_contract_mw',
            'bess_to_contract_mw',
            'pv_surplus_mw',
            'charge_mw_neg',
            'contracted_mw',
        ]]

    def _render_avg_profile_chart(avg_df: pd.DataFrame) -> None:
        axis_x = alt.Axis(values=list(range(0, 24, 2)))
        x_hour = alt.X('hour:O', title='Hour of Day', axis=axis_x)

        base = alt.Chart(avg_df).encode(x=x_hour)

        contrib_long = avg_df.melt(id_vars=['hour'],
                                   value_vars=['pv_to_contract_mw', 'bess_to_contract_mw'],
                                   var_name='Source', value_name='MW')
        contrib_long['Source'] = contrib_long['Source'].replace({
            'pv_to_contract_mw': 'PV→Contract',
            'bess_to_contract_mw': 'BESS→Contract',
        })
        contrib_long['SourceOrder'] = contrib_long['Source'].map({
            'PV→Contract': 0,
            'BESS→Contract': 1,
        })
        contrib_fill = (
            alt.Chart(contrib_long)
            .mark_bar(opacity=0.28, size=16)
            .encode(
                x=x_hour,
                y=alt.Y('MW:Q', stack='zero'),
                color=alt.Color('Source:N', scale=alt.Scale(domain=['PV→Contract', 'BESS→Contract'],
                                                           range=['#86c5da', '#7fd18b']), legend=None),
                order=alt.Order('SourceOrder:Q', sort='ascending')
            )
        )
        contrib_chart = (
            alt.Chart(contrib_long)
            .mark_bar(opacity=0.9, size=16)
            .encode(
                x=x_hour,
                y=alt.Y('MW:Q', title='MW', stack='zero'),
                color=alt.Color('Source:N', scale=alt.Scale(domain=['PV→Contract', 'BESS→Contract'],
                                                           range=['#86c5da', '#7fd18b'])),
                order=alt.Order('SourceOrder:Q', sort='ascending')
            )
        )

        pv_resource_area = (
            base
            .mark_area(
                opacity=0.18,
                color='#f2d7a0',
                line=alt.LineConfig(color='#c78100', strokeDash=[6, 3], strokeWidth=2)
            )
            .encode(
                x=x_hour,
                y=alt.Y('pv_resource_mw:Q', title='MW'),
                tooltip=[alt.Tooltip('pv_resource_mw:Q', title='PV resource (MW)', format='.2f')]
            )
        )

        pv_surplus_area = (
            base
            .mark_area(color='#f7c5c5', opacity=0.45)
            .encode(
                x=x_hour,
                y=alt.Y('pv_surplus_mw:Q', title='MW'),
                tooltip=[alt.Tooltip('pv_surplus_mw:Q', title='PV surplus (MW)', format='.2f')]
            )
        )

        area_chg = base.mark_area(opacity=0.5).encode(y='charge_mw_neg:Q', color=alt.value('#caa6ff'))

        contract_steps = avg_df[['hour', 'contracted_mw']].copy()
        contract_outline = pd.concat([
            contract_steps,
            pd.DataFrame({'hour': [contract_steps['hour'].max() + 1], 'contracted_mw': [0.0]})
        ], ignore_index=True)
        contract_box = (
            alt.Chart(contract_steps)
            .mark_bar(color='#f2a900', opacity=0.1, size=26)
            .encode(
                x=x_hour,
                y=alt.Y('contracted_mw:Q', title='MW'),
                y2=alt.value(0)
            )
        )
        line_contract = (
            alt.Chart(contract_outline)
            .mark_line(color='#f2a900', strokeWidth=2, interpolate='step-after')
            .encode(x=x_hour, y='contracted_mw:Q')
        )

        chart_avg_profile = alt.layer(
            contract_box,
            line_contract,
            pv_resource_area,
            pv_surplus_area,
            area_chg,
            contrib_fill,
            contrib_chart,
        ).properties(height=360)

        st.altair_chart(chart_avg_profile, use_container_width=True)

    if final_year_logs is not None and first_year_logs is not None:
        avg_first_year = _avg_profile_df_from_logs(first_year_logs, cfg)
        avg_final_year = _avg_profile_df_from_logs(final_year_logs, cfg)

        contracted_by_hour = np.array([
            cfg.contracted_mw if any(w.contains(h) for w in cfg.discharge_windows) else 0.0
            for h in range(24)
        ], dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            avg_charge = np.divide(
                hod_sum_charge,
                hod_count,
                out=np.zeros_like(hod_sum_charge),
                where=hod_count > 0,
            )
            avg_project = pd.DataFrame({
                'hour': np.arange(24),
                'pv_resource_mw': np.divide(hod_sum_pv_resource, hod_count, out=np.zeros_like(hod_sum_pv_resource), where=hod_count > 0),
                'pv_to_contract_mw': np.divide(hod_sum_pv, hod_count, out=np.zeros_like(hod_sum_pv), where=hod_count > 0),
                'bess_to_contract_mw': np.divide(hod_sum_bess, hod_count, out=np.zeros_like(hod_sum_bess), where=hod_count > 0),
                'charge_mw': avg_charge,
                'charge_mw_neg': -avg_charge,
                'contracted_mw': contracted_by_hour,
            })
            avg_project['pv_surplus_mw'] = np.maximum(
                avg_project['pv_resource_mw'] - avg_project['pv_to_contract_mw'] - avg_project['charge_mw'],
                0.0,
            )

        tab_final, tab_first, tab_project = st.tabs([
            "Final year",
            "Year 1",
            "Average across project",
        ])
        with tab_final:
            _render_avg_profile_chart(avg_final_year)
        with tab_first:
            _render_avg_profile_chart(avg_first_year)
        with tab_project:
            _render_avg_profile_chart(avg_project)
        st.caption(
            "Stacked bars (narrow width with soft fill): PV→Contract (blue) + BESS→Contract (green) fill the contract box "
            "(gold). Negative area: BESS charging (purple). PV surplus/curtailment shown in light red. PV resource overlay "
            "(tan, dashed outline). Contract step shown with gold outline."
        )
    else:
        st.info("Average daily profiles unavailable — simulation logs not generated.")

    st.markdown("### SOC & dispatch diagnostics")
    diag_logs: Dict[str, HourlyLog] = {}
    if first_year_logs is not None:
        diag_logs["Year 1 (initial)"] = first_year_logs
    if final_year_logs is not None:
        diag_logs[f"Year {cfg.years} (final)"] = final_year_logs

    if diag_logs:
        diag_default = list(diag_logs.keys()).index(f"Year {cfg.years} (final)") if final_year_logs is not None else 0
        selected_label = st.radio(
            "Select which year to visualize",
            options=list(diag_logs.keys()),
            index=diag_default,
            help="Toggle between the first-year baseline and final-year (with degradation/augmentation) logs.",
        )
        selected_logs = diag_logs[selected_label]

        heatmap_bin_hours = st.select_slider(
            "Heatmap resolution",
            options=[1, 2, 3],
            value=1,
            format_func=lambda h: f"{h}-hour bands",
            help="Downsample the heatmap horizontally to shrink the data payload if rendering feels sluggish.",
        )

        st.caption(
            "Heatmap: dark troughs near the SOC floor overnight hint at reliability risk; saturated midday bands near 100% "
            "indicate PV clipping/curtailment. Use the resolution control if the view feels heavy."
        )
        heatmap_pivot = prepare_soc_heatmap_data(selected_logs, cfg.initial_usable_mwh)
        heatmap_source = (
            heatmap_pivot.reset_index()
            .melt(id_vars="day_of_year", var_name="hour", value_name="soc_frac")
            .dropna(subset=["soc_frac"])
        )
        heatmap_source["hour_bin"] = (heatmap_source["hour"].astype(int) // heatmap_bin_hours) * heatmap_bin_hours
        if heatmap_bin_hours > 1:
            heatmap_source = (
                heatmap_source
                .groupby(["day_of_year", "hour_bin"], as_index=False)["soc_frac"]
                .mean()
                .rename(columns={"hour_bin": "hour"})
            )
        heatmap_source["hour_label"] = heatmap_source["hour"].astype(int).apply(
            lambda h: (
                f"{h:02d}:00–{(h + heatmap_bin_hours) % 24:02d}:00"
                if heatmap_bin_hours > 1
                else f"{h:02d}:00"
            )
        )
        heatmap_source["soc_pct"] = heatmap_source["soc_frac"] * 100.0
        heatmap_source["diagnostic_tip"] = (
            "Low overnight SOC → reliability risk. Flat mid-day SOC near 100% → PV clipping/curtailment headroom."
        )
        if heatmap_source.empty:
            st.info("SOC heatmap unavailable — simulation logs were empty for this scenario.")
        else:
            axis_x = alt.Axis(values=list(range(0, 24, max(heatmap_bin_hours, 3))))
            heatmap_chart = (
                alt.Chart(heatmap_source)
                .mark_rect()
                .encode(
                    x=alt.X("hour:O", title="Hour of day", axis=axis_x),
                    y=alt.Y("day_of_year:O", title="Day of year"),
                    color=alt.Color(
                        "soc_pct:Q",
                        title="SOC (%) of initial usable energy",
                        scale=alt.Scale(scheme="turbo", domain=[0, 100]),
                    ),
                    tooltip=[
                        alt.Tooltip("day_of_year:O", title="Day"),
                        alt.Tooltip("hour_label:N", title="Hour"),
                        alt.Tooltip("soc_pct:Q", title="SOC (%)", format=".1f"),
                        alt.Tooltip("diagnostic_tip:N", title="Reading tip"),
                    ],
                )
                .properties(height=320)
            )
            st.altair_chart(heatmap_chart, use_container_width=True)

        envelope_df = prepare_charge_discharge_envelope(selected_logs)
        st.caption(
            "Charge/discharge envelope shows the median and 5–95% range by hour; widening charge bands or deep discharge "
            "tails highlight operational stress that may erode reliability."
        )
        axis_x = alt.Axis(values=list(range(0, 24, 2)))
        envelope_chart = alt.layer(
            alt.Chart(envelope_df)
            .mark_area(opacity=0.25, color="#caa6ff")
            .encode(
                x=alt.X("hour:O", title="Hour of day", axis=axis_x),
                y=alt.Y("charge_low:Q", title="MW"),
                y2="charge_high:Q",
                tooltip=[
                    alt.Tooltip("hour:O", title="Hour"),
                    alt.Tooltip("charge_p05:Q", title="Charge p05 (MW)", format=".2f"),
                    alt.Tooltip("charge_p50:Q", title="Charge median (MW)", format=".2f"),
                    alt.Tooltip("charge_p95:Q", title="Charge p95 (MW)", format=".2f"),
                ],
            ),
            alt.Chart(envelope_df)
            .mark_area(opacity=0.25, color="#7fd18b")
            .encode(
                x=alt.X("hour:O", title="Hour of day", axis=axis_x),
                y=alt.Y("discharge_low:Q", title="MW"),
                y2="discharge_high:Q",
                tooltip=[
                    alt.Tooltip("hour:O", title="Hour"),
                    alt.Tooltip("discharge_p05:Q", title="Discharge p05 (MW)", format=".2f"),
                    alt.Tooltip("discharge_p50:Q", title="Discharge median (MW)", format=".2f"),
                    alt.Tooltip("discharge_p95:Q", title="Discharge p95 (MW)", format=".2f"),
                ],
            ),
            alt.Chart(envelope_df)
            .mark_line(color="#7d5ba6", strokeWidth=2)
            .encode(x=alt.X("hour:O", axis=axis_x), y=alt.Y("charge_median_neg:Q", title="MW")),
            alt.Chart(envelope_df)
            .mark_line(color="#2e7b53", strokeWidth=2)
            .encode(x=alt.X("hour:O", axis=axis_x), y=alt.Y("discharge_p50:Q", title="MW")),
        ).properties(height=300)
        st.altair_chart(envelope_chart, use_container_width=True)
    else:
        st.info("SOC heatmap and charge/discharge envelope are hidden because simulation logs are unavailable.")

    st.markdown("---")

    # ---------- Downloads ----------
    st.subheader("Downloads")
    cfg_download = json.dumps(asdict(cfg), indent=2)
    st.download_button(
        "Download simulation config (JSON)",
        cfg_download.encode("utf-8"),
        file_name="bess_config.json",
        mime="application/json",
    )
    st.download_button("Download yearly summary (CSV)", res_df.to_csv(index=False).encode('utf-8'),
                       file_name='bess_yearly_summary.csv', mime='text/csv')

    st.download_button("Download monthly summary (CSV)", monthly_df.to_csv(index=False).encode('utf-8'),
                       file_name='bess_monthly_summary.csv', mime='text/csv')

    if final_year_logs is not None:
        hourly_df = pd.DataFrame({
            'hour_index': np.arange(len(final_year_logs.hod)),
            'hod': final_year_logs.hod,
            'pv_to_contract_mw': final_year_logs.pv_to_contract_mw,
            'bess_to_contract_mw': final_year_logs.bess_to_contract_mw,
            'charge_mw': final_year_logs.charge_mw,
            'discharge_mw': final_year_logs.discharge_mw,
            'soc_mwh': final_year_logs.soc_mwh,
            'pv_surplus_mw': np.maximum(
                final_year_logs.pv_mw - final_year_logs.pv_to_contract_mw - final_year_logs.charge_mw,
                0.0,
            ),
        })
        st.download_button("Download final-year hourly logs (CSV)", hourly_df.to_csv(index=False).encode('utf-8'),
                           file_name='final_year_hourly_logs.csv', mime='text/csv')

    pdf_bytes = None
    try:
        pdf_bytes = build_pdf_summary(cfg, results, compliance, bess_share_of_firm, charge_discharge_ratio,
                                      pv_capture_ratio, discharge_capacity_factor,
                                      discharge_windows_text, charge_windows_text,
                                      hod_count, hod_sum_pv_resource, hod_sum_pv, hod_sum_bess, hod_sum_charge,
                                      total_shortfall_mwh, pv_excess_mwh, total_project_generation_mwh,
                                      bess_generation_mwh, pv_generation_mwh, bess_losses_mwh)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"PDF snapshot unavailable: {exc}")

    if pdf_bytes:
        st.download_button("Download brief PDF snapshot", pdf_bytes,
                           file_name='bess_results_snapshot.pdf', mime='application/pdf')
    else:
        st.info("PDF snapshot will appear after the simulation succeeds.")

    st.info("""
    Notes & Caveats:
    - PV-only charging is enforced; during discharge hours, PV first meets the contract, then surplus PV charges the BESS.
    - Threshold augmentation offers **Capability** and **SOH** triggers. Power is added to keep original C-hours.
    - EOY capability = what the fleet can sustain per day at year-end; Delivered Split = what actually happened per day on average.
    - Design Advisor uses a conservative energy-limited view: Deliverable/day ≈ BOL usable × SOH(final) × ΔSOC × η_dis.
    """)


if __name__ == "__main__":
    run_app()
