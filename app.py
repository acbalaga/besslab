# app.py — BESSLab (PV-only charging, AC-coupled) v0.3.0
# - NEW: Design Advisor + "Required to Meet" calculator (final-year aware)
# - NEW: KPI traffic-light hints vs practical benchmarks
# - Keeps: README/Help, Threshold & SOH-trigger augmentation, EOY capability + PV/BESS split,
#          multi-period daily profiles, flags, downloads

# ---- Abuse protection (rate limit) ----
import os
import time
import streamlit as st
from streamlit.delta_generator import DeltaGenerator


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
    if st.session_state.get("rate_limit_bypass", False):
        return

    now = time.time()
    recent = st.session_state.get("recent_runs", [])
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

    last_recorded = st.session_state.get("last_rate_limit_ts")
    if last_recorded is None or now - last_recorded >= min_spacing_seconds:
        recent.append(now)
        st.session_state["last_rate_limit_ts"] = now

    st.session_state["recent_runs"] = recent
# ---- end gate ----


def get_rate_limit_password() -> str:
    """Return the password used to disable rate limiting.

    The lookup order is Streamlit secrets → environment variable → built-in default.
    """

    try:
        secret_password = st.secrets.get("rate_limit_password")
    except StreamlitSecretNotFoundError:
        secret_password = None

    return secret_password or os.environ.get("BESSLAB_RATE_LIMIT_PASSWORD") or "besslab"

import math
import calendar
import json
from dataclasses import dataclass, field, replace, asdict
from io import BytesIO, StringIO
from typing import Any, List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from pathlib import Path
import altair as alt
from fpdf import FPDF
from economics import EconomicInputs, EconomicOutputs, compute_lcoe_lcos
from economics_helpers import compute_lcoe_lcos_with_augmentation_fallback
from streamlit.errors import StreamlitSecretNotFoundError
from sensitivity_sweeps import build_soc_windows, generate_values, run_sensitivity_grid

BASE_DIR = Path(__file__).resolve().parent
USD_TO_PHP = 58.0

# --------- Flag metadata ---------
FLAG_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "firm_shortfall_hours": {
        "label": "Firm shortfall hours",
        "meaning": "In-window hours when PV + BESS could not meet contracted MW.",
        "knobs": "Increase energy/power, relax windows, augment, or reduce contract.",
        "insight": (
            "Prioritize contract-window alignment and energy sufficiency; sustained shortfalls"
            " usually point to inadequate discharge capacity or overly tight windows."
        ),
    },
    "soc_floor_hits": {
        "label": "SOC floor hits",
        "meaning": "SOC hit the minimum reserve.",
        "knobs": "Raise ceiling, lower floor, increase energy, improve RTE, widen charge windows.",
        "insight": (
            "Consistent floor hits indicate the reserve band is too narrow or charging windows"
            " are constrained; evaluate reserve settings and ensure charging energy is available"
            " before dispatch windows."
        ),
    },
    "soc_ceiling_hits": {
        "label": "SOC ceiling hits",
        "meaning": "Battery reached upper SOC limit (limited charging).",
        "knobs": "Increase shoulder discharge, lower ceiling, narrow charge window if unnecessary.",
        "insight": (
            "Ceiling hits suggest unused charging headroom; consider shifting discharge earlier"
            " or narrowing charge windows so PV energy is redirected toward contract support."
        ),
    },
}


def build_flag_insights(flag_totals: Dict[str, int]) -> List[str]:
    """Translate flag counts into short, actionable insights."""

    insights: List[str] = []
    ordered_keys = sorted(flag_totals, key=flag_totals.get, reverse=True)

    for key in ordered_keys:
        count = flag_totals.get(key, 0)
        if count <= 0:
            continue

        meta = FLAG_DEFINITIONS.get(key)
        if not meta:
            continue

        insights.append(f"{meta['label']} occurred {count:,} times. {meta['insight']}")

    if flag_totals.get("soc_floor_hits", 0) > 0 and flag_totals.get("soc_ceiling_hits", 0) > 0:
        insights.append(
            "Both SOC floor and ceiling were hit; revisit charge/discharge windows and reserve bands"
            " to reduce cycling extremes."
        )

    if not insights:
        insights.append("No flags were triggered across the simulated years.")

    return insights

# --------- Utilities ---------

def read_pv_profile(path_candidates: List[Any]) -> pd.DataFrame:
    """Read and validate a PV profile with ['hour_index','pv_mw'] columns in MW."""

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if not {"hour_index", "pv_mw"}.issubset(df.columns):
            raise ValueError("CSV must contain columns: hour_index, pv_mw")

        df = df[["hour_index", "pv_mw"]].copy()
        df["hour_index"] = pd.to_numeric(df["hour_index"], errors="coerce")
        df["pv_mw"] = pd.to_numeric(df["pv_mw"], errors="coerce")

        invalid_rows = ~np.isfinite(df["hour_index"]) | ~np.isfinite(df["pv_mw"])
        if invalid_rows.any():
            st.error(
                "PV CSV contains non-numeric or missing hour_index/pv_mw entries; "
                f"dropping {invalid_rows.sum()} rows."
            )
            df = df.loc[~invalid_rows].copy()

        if df.empty:
            raise ValueError("No valid PV rows after cleaning.")

        if (df["hour_index"] % 1 != 0).any():
            st.error("hour_index must be integer hours (0-8759).")
            raise ValueError("Non-integer hour_index encountered.")

        df["hour_index"] = df["hour_index"].astype(int)

        if df["hour_index"].min() == 1 and 0 not in df["hour_index"].values:
            df["hour_index"] = df["hour_index"] - 1

        out_of_range = (df["hour_index"] < 0) | (df["hour_index"] >= 8760)
        if out_of_range.any():
            st.warning(
                "hour_index values outside 0-8759 were dropped: "
                f"{sorted(df.loc[out_of_range, 'hour_index'].unique().tolist())}"
            )
            df = df.loc[~out_of_range].copy()

        if df.empty:
            raise ValueError("No valid PV rows after removing out-of-range hours.")

        duplicate_mask = df["hour_index"].duplicated(keep=False)
        if duplicate_mask.any():
            st.warning(
                "Duplicate hour_index values found; averaging pv_mw for each hour."
            )
            df = (
                df.groupby("hour_index", as_index=False)["pv_mw"].mean()
                .sort_values("hour_index")
                .reset_index(drop=True)
            )
        else:
            df = df.sort_values("hour_index").drop_duplicates("hour_index")

        full_index = pd.Index(range(8760), name="hour_index")
        df = df.set_index("hour_index")
        missing_hours = full_index.difference(df.index)
        if len(missing_hours) > 0:
            st.warning(
                f"PV CSV is missing {len(missing_hours)} hours; filling gaps with 0 MW."
            )
            df = df.reindex(full_index, fill_value=0.0)
        else:
            df = df.reindex(full_index)

        if len(df) != 8760:
            st.warning(
                f"PV CSV has {len(df)} rows after cleaning (expected 8760). Proceeding anyway."
            )

        df = df.reset_index()
        df["pv_mw"] = df["pv_mw"].astype(float)
        return df

    last_err = None
    for candidate in path_candidates:
        if candidate is None:
            continue
        try:
            df = pd.read_csv(candidate)
            return _clean(df)
        except Exception as e:
            last_err = e
    raise RuntimeError(
        "Failed to read PV profile. "
        f"Looked for: {path_candidates}. Last error: {last_err}"
    )


def compute_pv_surplus(pv_resource: pd.Series, pv_to_contract: pd.Series, charge_mw: pd.Series) -> pd.Series:
    """Return PV surplus/curtailment after serving contract and charging.

    Negative values are clipped to zero to avoid showing deficit as surplus, while
    preserving vectorized performance for large hourly datasets.
    """

    return np.maximum(pv_resource - pv_to_contract - charge_mw, 0.0)


def read_cycle_model(path_candidates: List[str]) -> pd.DataFrame:
    """Read cycle model Excel with column pairs DoD*_Cycles / DoD*_Ret(%)."""
    last_err = None
    for p in path_candidates:
        try:
            df = pd.read_excel(p)
            keep = []
            for dod in [10, 20, 40, 80, 100]:
                c1 = f"DoD{dod}_Cycles"; c2 = f"DoD{dod}_Ret(%)"
                if c1 in df.columns and c2 in df.columns:
                    keep += [c1, c2]
            if not keep:
                raise ValueError("No DoD*_Cycles / DoD*_Ret(%) pairs found.")
            return df[keep].copy()
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read cycle model. Looked for: {path_candidates}. Last error: {last_err}")

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

# --------- Degradation helpers ---------

def infer_dod_bucket(daily_dis_mwh: np.ndarray, usable_mwh_available: float) -> int:
    """Infer an effective DoD bucket from daily discharge energy.

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
class ScenarioConfig:
    label: str
    bess_specs: Dict[str, Any]
    dispatch: Dict[str, Any]
    augmentation: Dict[str, Any]
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


def build_pdf_summary(cfg: SimConfig, results: List[YearResult], compliance: float, bess_share: float,
                      charge_discharge_ratio: float, pv_capture_ratio: float,
                      discharge_capacity_factor: float, discharge_windows_text: str,
                      charge_windows_text: str) -> bytes:
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
    pdf.ln(2)

    card_width = (usable_width - 10) / 3
    card_height = 22
    x0 = margin
    y0 = pdf.get_y()
    _draw_metric_card(pdf, x0, y0, card_width, card_height, "Delivery compliance", f"{compliance:,.2f}%", "Across full life",
                      (225, 245, 255))
    _draw_metric_card(pdf, x0 + card_width + 5, y0, card_width, card_height, "BESS share of firm",
                      f"{bess_share:,.1f}%", "Portion of contract served", (235, 248, 240))
    _draw_metric_card(pdf, x0 + 2 * (card_width + 5), y0, card_width, card_height, "Charge/Discharge ratio",
                      _fmt(charge_discharge_ratio), "Energy in vs out", (245, 238, 255))

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

    expected = [r.expected_firm_mwh for r in results]
    delivered = [r.delivered_firm_mwh for r in results]
    _draw_sparkline(
        pdf,
        chart_x_left,
        chart_y,
        chart_width,
        chart_height,
        [
            ("Exp", expected, (255, 160, 122)),
            ("Del", delivered, (76, 175, 80)),
        ],
        "Annual firm energy (MWh)",
    )

    soh_series = [r.soh_total for r in results]
    pdf.set_xy(chart_x_left + chart_width + 5, chart_y)
    _draw_sparkline(
        pdf,
        chart_x_left + chart_width + 5,
        chart_y,
        chart_width,
        chart_height,
        [
            ("Total", soh_series, (66, 133, 244)),
            ("Cycle", [r.soh_cycle for r in results], (142, 68, 173)),
            ("Cal", [r.soh_calendar for r in results], (255, 193, 7)),
        ],
        "State of health (fraction)",
    )

    pdf.set_y(chart_y + chart_height + 8)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Design + final year snapshot", ln=1)
    pdf.set_font("Helvetica", "", 9)
    param_lines = [
        f"Initial usable: {cfg.initial_usable_mwh:.1f} MWh | Initial power: {cfg.initial_power_mw:.1f} MW",
        f"Augmentation: {cfg.augmentation} | SoC window: {cfg.soc_floor:.2f}-{cfg.soc_ceiling:.2f}",
        f"Round-trip efficiency: {cfg.rte_roundtrip:.2f} | Calendar fade: {cfg.calendar_fade_rate:.3f}/yr",
        f"EOY usable: {final.eoy_usable_mwh:,.1f} MWh (Year 1: {first.eoy_usable_mwh:,.1f})",
        f"EOY power: {final.eoy_power_mw:,.2f} MW (Year 1: {first.eoy_power_mw:,.2f})",
        f"PV->Contract: {final.pv_to_contract_mwh:,.1f} MWh/yr | BESS->Contract: {final.bess_to_contract_mwh:,.1f} MWh/yr",
        f"Eq cycles this year: {final.eq_cycles:,.1f} | Cum cycles: {final.cum_cycles:,.1f}",
    ]

    if cfg.augmentation_schedule:
        param_lines.insert(2, f"Manual augmentation schedule: {describe_schedule(cfg.augmentation_schedule)}")

    pdf.set_x(margin)
    for line in param_lines:
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

    eta_rt = max(0.05, min(cfg.rte_roundtrip, 0.9999))
    eta_ch = eta_rt ** 0.5; eta_dis = eta_rt ** 0.5

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
    usable_mwh_start = state.current_usable_mwh_bolref * soh_total_start

    soc_mwh = usable_mwh_start * 0.5
    soc_min = usable_mwh_start * cfg.soc_floor
    soc_max = usable_mwh_start * cfg.soc_ceiling

    n_hours = len(pv_mw)
    day_index = np.array([i // 24 for i in range(n_hours)])
    calendar_index = pd.date_range("2020-01-01", periods=n_hours, freq=pd.Timedelta(hours=dt))
    month_index = calendar_index.month - 1
    daily_dis_mwh = np.zeros(day_index.max() + 1)
    hod = np.arange(n_hours) % 24

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
    state.last_dod_key = dod_key_eff
    dod_frac = {10:0.10,20:0.20,40:0.40,80:0.80,100:1.00}[dod_key_eff]
    usable_for_cycles = max(1e-9, state.current_usable_mwh_bolref * dod_frac)
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
    if pv_df is None or pv_df.empty:
        raise ValueError("PV profile is missing or empty; please upload a valid 8760.")

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


def run_app():


    st.set_page_config(page_title="BESSLab by ACB", layout="wide")
    st.markdown(
        """
        <style>
        /* Hide default Streamlit sidebar navigation to declutter the layout */
        [data-testid="stSidebarNav"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("BESS LAB — PV-only charging, AC-coupled")

    # README / Help
    with st.expander("Help & Guide (click to open)", expanded=False):
        st.markdown("""
    ## Welcome to BESSLab
    A quick-start guide for exploring **PV-only, AC-coupled BESS** behavior.

    ### How to get started
    1) Upload a **PV 8760 CSV** (`hour_index, pv_mw`) or keep the sample file.
    2) (Optional) Upload a **cycle-model Excel** file to use your own degradation table.
    3) Set your **contracted MW**, **duration (hours)**, and **dispatch windows** in the sidebar.
    4) Adjust efficiency, SOC limits, availability, augmentation triggers, and rate-limit settings if needed.
    5) Review the summary cards and charts, then download the CSV or PDF outputs you need.
    6) Run **Design Advisor** recommendations or **SOC sensitivity sweeps** when compliance slips.

    ### What you will see
    - Whether the contract is met across the project life.
    - How much energy comes from **PV directly** vs. **the BESS**, including shortfall flags.
    - End-of-year capability bars, typical daily profiles, and **economics sensitivities**.
    - Friendly suggestions from the **Design Advisor** plus SOC-window **sensitivity sweeps**.
    - A **scenario table** for saving and comparing different input sets, alongside hourly/monthly/yearly downloads.

    ### Handy extras
    - Use the sidebar link to open the **economics helper** page (LCOE/LCOS) and download the module.
    - Turn off the **rate limit** by entering the password in the sidebar (default: `besslab`).
    - Window strings accept minutes (e.g., `05:30-09:00`), which are parsed as fractional hours.

    ### If results look off
    - Shortfalls? Try widening the SOC window, improving efficiency, or enabling augmentation.
    - Frequent empty battery? Increase duration (MWh), raise the SOC ceiling, or allow more charge time.
    - Battery keeps topping out? Lower the SOC ceiling slightly or add a bit more discharge window.
    - Unexpected economics results? Confirm price/cost units and rerun the sensitivity heatmaps.

    ### Helpful notes
    - `hour_index` can start at 0 or 1; the app will align it.
    - The app uses included defaults when you do not upload files.
    - No grid charging is modeled—this is a PV-only pre-feasibility view.
    - Sensitivity sweeps will clear when inputs change; rerun them after major adjustments.

    **Questions or ideas?** Feedback is welcome to keep improving the tool.
    """)


    with st.sidebar:
        st.header("Data Sources")
        default_pv_paths = [str(BASE_DIR / 'data' / 'PV_8760_MW.csv')]
        default_cycle_paths = [str(BASE_DIR / 'data' / 'cycle_model.xlsx')]

        pv_file = st.file_uploader("PV 8760 CSV (hour_index, pv_mw in MW)", type=['csv'])
        try:
            pv_df = read_pv_profile([pv_file] if pv_file is not None else default_pv_paths)
        except Exception as exc:
            st.error(f"Unable to load PV profile: {exc}")
            st.stop()

        cycle_file = st.file_uploader("Cycle model Excel (optional override)", type=['xlsx'])
        cycle_df = pd.read_excel(cycle_file) if cycle_file is not None else read_cycle_model(default_cycle_paths)

        st.caption("If no files are uploaded, built-in defaults are read from ./data/")

        st.divider()
        st.subheader("Rate limit override")
        rate_limit_password = st.text_input(
            "Remove rate limit (password)",
            type="password",
            help=(
                "Enter the configured password to disable the session rate limit. "
                "If no secret is set, use 'besslab'."
            ),
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
            rte = st.slider("Round-trip efficiency (single, at POI)", 0.70, 0.99, 0.88, 0.01,
                help="Single RTE; internally split √RTE for charge/discharge.")

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

    with st.expander("Scenario set", expanded=False):
        st.caption("Capture different BESS specs/dispatch/augmentation combinations for later reuse.")

        def build_scenario(label: str) -> ScenarioConfig:
            return ScenarioConfig(
                label=label,
                bess_specs={
                    "power_mw": float(init_power),
                    "usable_mwh": float(init_energy),
                    "soc_floor": float(soc_floor),
                    "soc_ceiling": float(soc_ceiling),
                },
                dispatch={
                    "contracted_mw": float(contracted_mw),
                    "discharge_windows_text": discharge_windows_text,
                    "charge_windows_text": charge_windows_text,
                },
                augmentation={
                    "mode": aug_mode,
                    "trigger_type": aug_trigger_type,
                    "threshold_margin": float(aug_thr_margin),
                    "topup_margin": float(aug_topup),
                    "soh_trigger_pct": float(aug_soh_trig),
                    "soh_add_frac_initial": float(aug_soh_add),
                    "periodic_every_years": int(aug_every),
                    "periodic_add_frac_of_bol": float(aug_frac),
                    "add_mode": aug_size_mode,
                    "fixed_energy_mwh": float(aug_fixed_energy),
                    "retire_old_cohort": bool(retire_enabled),
                    "retire_soh_pct": float(retire_soh),
                },
                augmentation_schedule=list(manual_schedule_entries) if aug_mode == "Manual" else [],
            )

        scenarios: List[ScenarioConfig] = st.session_state.setdefault("scenarios", [])
        default_label = f"Scenario {len(scenarios) + 1}"
        new_label = st.text_input("Scenario label", value=default_label)

        add_cols = st.columns([1, 1])
        add_disabled = bool(aug_mode == "Manual" and (manual_schedule_errors or not manual_schedule_entries))
        with add_cols[0]:
            if st.button("Add current inputs", use_container_width=True, disabled=add_disabled):
                scenarios.append(build_scenario(new_label or default_label))
                st.success("Scenario added to the session set.")
            elif add_disabled:
                st.info("Resolve schedule validation issues before saving a manual augmentation scenario.")

        if scenarios:
            scenario_options = {f"{i + 1}. {sc.label}": i for i, sc in enumerate(scenarios)}
            selected_key = st.selectbox("Select a scenario", list(scenario_options.keys()))
            selected_idx = scenario_options[selected_key]
            selected = scenarios[selected_idx]

            with add_cols[1]:
                if st.button("Duplicate selected", use_container_width=True):
                    dup_label = f"{selected.label} (copy)"
                    scenarios.append(replace(selected, label=dup_label))
                    st.success(f"Duplicated as '{dup_label}'.")

            del_col1, _del_col2 = st.columns([1, 1])
            with del_col1:
                if st.button("Delete selected", use_container_width=True):
                    scenarios.pop(selected_idx)
                    st.warning("Scenario removed from the session set.")

            table_rows = []
            for sc in scenarios:
                table_rows.append({
                    "Label": sc.label,
                    "Power (MW)": sc.bess_specs.get("power_mw"),
                    "Usable (MWh)": sc.bess_specs.get("usable_mwh"),
                    "SOC floor": sc.bess_specs.get("soc_floor"),
                    "SOC ceiling": sc.bess_specs.get("soc_ceiling"),
                    "Contracted MW": sc.dispatch.get("contracted_mw"),
                    "Discharge windows": sc.dispatch.get("discharge_windows_text"),
                    "Charge windows": sc.dispatch.get("charge_windows_text") or "Any PV hour",
                    "Aug strategy": sc.augmentation.get("mode"),
                    "Aug trigger": sc.augmentation.get("trigger_type"),
                    "Aug schedule": describe_schedule(sc.augmentation_schedule),
                })

            st.data_editor(
                pd.DataFrame(table_rows),
                hide_index=True,
                disabled=True,
                use_container_width=True,
            )
        else:
            st.info("No scenarios saved yet. Use 'Add current inputs' to capture this configuration.")

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

    scenarios_run_col, scenarios_hint_col = st.columns([2, 1])
    with scenarios_run_col:
        run_all_clicked = st.button(
            "Run all scenarios", use_container_width=True, disabled=not scenarios
        )
    with scenarios_hint_col:
        if scenarios:
            st.caption("Batch run saved scenarios to compare KPIs side-by-side.")
        else:
            st.caption("Save scenarios above to enable batch runs.")

    if run_all_clicked:
        enforce_rate_limit()
        comparison_rows = []
        scored_runs = []
        with st.spinner("Running all saved scenarios..."):
            for sc in scenarios:
                sc_charge_windows = sc.dispatch.get("charge_windows_text", "")
                scenario_cfg = replace(
                    cfg,
                    initial_power_mw=float(sc.bess_specs.get("power_mw", cfg.initial_power_mw)),
                    initial_usable_mwh=float(sc.bess_specs.get("usable_mwh", cfg.initial_usable_mwh)),
                    soc_floor=float(sc.bess_specs.get("soc_floor", cfg.soc_floor)),
                    soc_ceiling=float(sc.bess_specs.get("soc_ceiling", cfg.soc_ceiling)),
                    contracted_mw=float(sc.dispatch.get("contracted_mw", cfg.contracted_mw)),
                    discharge_windows=parse_windows(sc.dispatch.get("discharge_windows_text", "")),
                    charge_windows_text=sc_charge_windows,
                    augmentation=sc.augmentation.get("mode", cfg.augmentation),
                    aug_trigger_type=sc.augmentation.get("trigger_type", cfg.aug_trigger_type),
                    aug_threshold_margin=float(sc.augmentation.get("threshold_margin", cfg.aug_threshold_margin)),
                    aug_topup_margin=float(sc.augmentation.get("topup_margin", cfg.aug_topup_margin)),
                    aug_soh_trigger_pct=float(sc.augmentation.get("soh_trigger_pct", cfg.aug_soh_trigger_pct)),
                    aug_soh_add_frac_initial=float(sc.augmentation.get("soh_add_frac_initial", cfg.aug_soh_add_frac_initial)),
                    aug_periodic_every_years=int(sc.augmentation.get("periodic_every_years", cfg.aug_periodic_every_years)),
                    aug_periodic_add_frac_of_bol=float(sc.augmentation.get("periodic_add_frac_of_bol", cfg.aug_periodic_add_frac_of_bol)),
                    aug_add_mode=sc.augmentation.get("add_mode", cfg.aug_add_mode),
                    aug_fixed_energy_mwh=float(sc.augmentation.get("fixed_energy_mwh", cfg.aug_fixed_energy_mwh)),
                    aug_retire_old_cohort=bool(sc.augmentation.get("retire_old_cohort", cfg.aug_retire_old_cohort)),
                    aug_retire_soh_pct=float(sc.augmentation.get("retire_soh_pct", cfg.aug_retire_soh_pct)),
                    augmentation_schedule=list(sc.augmentation_schedule),
                )
                try:
                    scenario_output = simulate_project(
                        scenario_cfg, pv_df, cycle_df, dod_override, need_logs=False
                    )
                except ValueError as exc:  # noqa: BLE001
                    st.warning(f"Skipping '{sc.label}': {exc}")
                    continue

                scenario_summary = summarize_simulation(scenario_output)
                scored_runs.append({
                    'label': sc.label,
                    'cfg': scenario_cfg,
                    'summary': scenario_summary,
                    'results': scenario_output.results,
                })
                comparison_rows.append({
                    'Label': sc.label,
                    'Compliance (%)': scenario_summary.compliance,
                    'BESS share of firm (%)': scenario_summary.bess_share_of_firm,
                    'Total shortfall (MWh)': scenario_summary.total_shortfall_mwh,
                    'Charge/Discharge ratio': scenario_summary.charge_discharge_ratio,
                    'PV capture ratio': scenario_summary.pv_capture_ratio,
                    'Augmentation events': scenario_output.augmentation_events,
                })

        if comparison_rows:
            comp_df = pd.DataFrame(comparison_rows)
            st.dataframe(
                comp_df.style.format({
                    'Compliance (%)': '{:,.2f}',
                    'BESS share of firm (%)': '{:,.1f}',
                    'Total shortfall (MWh)': '{:,.1f}',
                    'Charge/Discharge ratio': '{:,.3f}',
                    'PV capture ratio': '{:,.3f}',
                }),
                use_container_width=True,
            )

            def _score_coverage(min_coverage: float) -> float:
                if min_coverage < 0.99:
                    return float('nan')
                if min_coverage >= 0.999:
                    return 5.0
                if min_coverage >= 0.995:
                    return 4.0
                if min_coverage >= 0.990:
                    return 3.0
                if min_coverage >= 0.980:
                    return 2.0
                return 1.0

            def _score_shortfall(shortfall: float, min_s: float, max_s: float) -> float:
                if math.isclose(max_s, min_s):
                    return 5.0
                raw = 1.0 + 4.0 * (max_s - shortfall) / max(1e-9, (max_s - min_s))
                return max(1.0, min(5.0, raw))

            def _score_grid_margins(violations: int, near_limit_frac: float) -> float:
                if violations == 0 and near_limit_frac < 0.01:
                    return 5.0
                if violations == 0 and near_limit_frac < 0.05:
                    return 4.0
                if (1 <= violations <= 5) or (0.05 <= near_limit_frac < 0.10):
                    return 3.0
                if violations > 5 and near_limit_frac < 0.20:
                    return 2.0
                return 1.0

            def _score_soh_margin(margin: float) -> float:
                if margin >= 0.10:
                    return 5.0
                if margin >= 0.05:
                    return 4.0
                if margin >= 0.0:
                    return 3.0
                if margin >= -0.05:
                    return 2.0
                return 1.0

            def _score_cycles_ratio(ratio: float) -> float:
                if ratio <= 0.60:
                    return 5.0
                if ratio <= 0.75:
                    return 4.0
                if ratio <= 0.90:
                    return 3.0
                if ratio <= 1.00:
                    return 2.0
                return 1.0

            def _score_stress_fraction(fraction: float) -> float:
                if fraction <= 0.01:
                    return 5.0
                if fraction <= 0.05:
                    return 4.0
                if fraction <= 0.10:
                    return 3.0
                if fraction <= 0.20:
                    return 2.0
                return 1.0

            if scored_runs:
                passing_shortfalls = []
                coverage_lookup = {}
                for run in scored_runs:
                    coverage_year = [
                        (r.delivered_firm_mwh / r.expected_firm_mwh) if r.expected_firm_mwh > 0 else float('nan')
                        for r in run['results']
                    ]
                    coverage_min = float(np.nanmin(coverage_year)) if coverage_year else float('nan')
                    coverage_lookup[run['label']] = coverage_min
                    if not math.isnan(coverage_min) and coverage_min >= 0.99:
                        passing_shortfalls.append(run['summary'].total_shortfall_mwh)

                min_s = min(passing_shortfalls) if passing_shortfalls else 0.0
                max_s = max(passing_shortfalls) if passing_shortfalls else 0.0
                soh_requirement = 0.80
                cycles_allowed = 400.0

                score_rows = []
                for run in scored_runs:
                    summary = run['summary']
                    coverage_min = coverage_lookup.get(run['label'], float('nan'))
                    c1 = _score_coverage(coverage_min)
                    c2 = _score_shortfall(summary.total_shortfall_mwh, min_s, max_s) if not math.isnan(c1) else float('nan')
                    total_violation_hours = sum(r.flags['firm_shortfall_hours'] for r in run['results'])
                    near_limit_hours = sum(r.flags['soc_floor_hits'] + r.flags['soc_ceiling_hits'] for r in run['results'])
                    total_hours = run['cfg'].years * 8760
                    near_limit_frac = near_limit_hours / max(1, total_hours)
                    c3 = _score_grid_margins(total_violation_hours, near_limit_frac)
                    contract_score = (
                        0.6 * c1 + 0.25 * c2 + 0.15 * c3
                    ) if not any(math.isnan(x) for x in [c1, c2, c3]) else float('nan')

                    final_soh = run['results'][-1].soh_total if run['results'] else float('nan')
                    soh_margin = final_soh - soh_requirement if not math.isnan(final_soh) else float('nan')
                    t1 = _score_soh_margin(soh_margin)
                    cycle_ratio = summary.avg_eq_cycles_per_year / cycles_allowed if cycles_allowed > 0 else float('nan')
                    t2 = _score_cycles_ratio(cycle_ratio)
                    stress_fraction = near_limit_hours / max(1, total_hours)
                    t3 = _score_stress_fraction(stress_fraction)
                    tech_score = (
                        0.5 * t1 + 0.3 * t2 + 0.2 * t3
                    ) if not any(math.isnan(x) for x in [t1, t2, t3]) else float('nan')

                    if math.isnan(contract_score) and math.isnan(tech_score):
                        continue

                    score_rows.append({
                        'Label': run['label'],
                        'C1 (coverage)': c1,
                        'C2 (shortfall energy)': c2,
                        'C3 (grid margins)': c3,
                        'Contract & grid score (C)': contract_score,
                        'T1 (SoH margin)': t1,
                        'T2 (cycle loading)': t2,
                        'T3 (C-rate stress)': t3,
                        'Technical robustness (T)': tech_score,
                    })

                if score_rows:
                    score_df = pd.DataFrame(score_rows)
                    numeric_cols = [c for c in score_df.columns if c != 'Label']
                    formatters = {col: '{:,.2f}' for col in numeric_cols}
                    st.dataframe(score_df.style.format(formatters), use_container_width=True)

                    heat_df = score_df.melt(
                        id_vars='Label',
                        value_vars=['Contract & grid score (C)', 'Technical robustness (T)'],
                        var_name='Metric',
                        value_name='Score',
                    ).dropna(subset=['Score'])
                    heat_chart = alt.Chart(heat_df).mark_rect().encode(
                        x=alt.X('Metric:N', title=''),
                        y=alt.Y('Label:N', sort=None, title='Scenario'),
                        color=alt.Color('Score:Q', scale=alt.Scale(domain=[1, 5], scheme='blues'), title='Score (1–5)'),
                        tooltip=['Label', 'Metric', alt.Tooltip('Score:Q', format='.2f')],
                    ).properties(title='Scenario feasibility & reliability heatmap')
                    st.altair_chart(heat_chart, use_container_width=True)


    def render_scenario_comparisons(container: st.delta_generator.DeltaGenerator) -> None:
        """Show the scenario comparison controls using the latest stored snapshot."""

        with container:
            st.markdown("---")
            comparison_tab = st.tabs(["Scenario comparisons"])[0]
            with comparison_tab:
                st.caption(
                    "Save different input sets to compare how capacity and dispatch choices affect KPIs."
                )
                if "scenario_comparisons" not in st.session_state:
                    st.session_state["scenario_comparisons"] = []

                scenario_snapshot = st.session_state.get("latest_scenario_snapshot")
                default_label = f"Scenario {len(st.session_state['scenario_comparisons']) + 1}"
                scenario_label = st.text_input(
                    "Label for this scenario", default_label, key="scenario_label_input"
                )

                if scenario_snapshot is None:
                    st.info("Run the simulation to populate metrics before saving a scenario.")

                if st.button("Add current scenario to table", disabled=scenario_snapshot is None):
                    scenario_entry = {"Label": scenario_label or default_label, **scenario_snapshot}
                    st.session_state["scenario_comparisons"].append(scenario_entry)
                    st.success("Scenario saved. Adjust inputs and add another to compare.")

                if st.session_state["scenario_comparisons"]:
                    compare_df = pd.DataFrame(st.session_state["scenario_comparisons"])
                    st.dataframe(compare_df.style.format({
                        'Compliance (%)': '{:,.2f}',
                        'BESS share of firm (%)': '{:,.1f}',
                        'Charge/Discharge ratio': '{:,.3f}',
                        'PV capture ratio': '{:,.3f}',
                        'Total project generation (MWh)': '{:,.1f}',
                        'BESS share of generation (MWh)': '{:,.1f}',
                        'PV share of generation (MWh)': '{:,.1f}',
                        'PV excess (MWh)': '{:,.1f}',
                        'BESS losses (MWh)': '{:,.1f}',
                        'Final EOY usable (MWh)': '{:,.1f}',
                        'Final EOY power (MW)': '{:,.2f}',
                        'Final eq cycles (year)': '{:,.1f}',
                        'Final SOH_total': '{:,.3f}',
                    }))
                    if st.button("Clear saved scenarios"):
                        st.session_state["scenario_comparisons"] = []
                else:
                    st.info(
                        "No saved scenarios yet. Tune the inputs above and click 'Add current scenario to table'."
                    )

    run_cols = st.columns([2, 1])
    with run_cols[0]:
        run_clicked = st.button(
            "Run simulation",
            use_container_width=True,
            help="Prevents auto-reruns while you adjust inputs; click to compute results.",
        )
    with run_cols[1]:
        st.caption(
            "Edit parameters freely, then run when ready. Scenarios can still be batch-run above."
        )

    comparison_placeholder = st.container()

    if not run_clicked:
        render_scenario_comparisons(comparison_placeholder)
        st.info("Click 'Run simulation' to generate results after updating inputs.")
        st.stop()

    enforce_rate_limit()

    try:
        sim_output = simulate_project(cfg, pv_df, cycle_df, dod_override)
    except ValueError as exc:  # noqa: BLE001
        st.error(str(exc))
        st.stop()

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

    st.session_state["latest_scenario_snapshot"] = {
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

    comparison_placeholder.empty()
    render_scenario_comparisons(comparison_placeholder)

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

    with st.expander("Sensitivity sweeps (PV oversize, SOC window, RTE)", expanded=False):
        st.caption(
            "Run quick grids of PV/SOC/RTE combos without blocking the main simulation, then "
            "surface the most promising settings."
        )

        sweep_col1, sweep_col2, sweep_col3 = st.columns(3)
        with sweep_col1:
            pv_range = st.slider(
                "PV oversize multipliers (×)",
                min_value=0.5,
                max_value=3.0,
                value=(1.0, 1.5),
                step=0.05,
                help="Scales the PV profile before running each sweep case.",
            )
            pv_steps = st.number_input(
                "PV oversize points",
                min_value=1,
                max_value=6,
                value=3,
                help="Number of evenly spaced multipliers between the selected bounds.",
            )

        with sweep_col2:
            soc_floor_range = st.slider(
                "SOC floor range",
                min_value=0.0,
                max_value=0.9,
                value=(max(0.0, cfg.soc_floor - 0.05), min(0.9, cfg.soc_floor + 0.05)),
                step=0.01,
                help="Lower bound sweep for minimum SOC.",
            )
            soc_ceiling_range = st.slider(
                "SOC ceiling range",
                min_value=0.1,
                max_value=1.0,
                value=(max(0.1, cfg.soc_ceiling - 0.05), min(1.0, cfg.soc_ceiling + 0.05)),
                step=0.01,
                help="Upper bound sweep for maximum SOC.",
            )
            soc_steps = st.number_input(
                "SOC window grid points",
                min_value=1,
                max_value=5,
                value=2,
                help="Number of evenly spaced floors and ceilings to cross-multiply.",
            )

        with sweep_col3:
            rte_range = st.slider(
                "Round-trip efficiency sweep",
                min_value=0.6,
                max_value=1.0,
                value=(max(0.6, cfg.rte_roundtrip - 0.05), min(1.0, cfg.rte_roundtrip + 0.05)),
                step=0.01,
                help="Efficiency values are applied directly to the simulator.",
            )
            rte_steps = st.number_input(
                "RTE points",
                min_value=1,
                max_value=6,
                value=3,
                help="Number of values between the bounds (inclusive).",
            )

        pv_factors = generate_values(pv_range[0], pv_range[1], int(pv_steps))
        soc_windows = build_soc_windows(
            soc_floor_range, soc_ceiling_range, int(soc_steps), int(soc_steps)
        )
        rte_values = generate_values(rte_range[0], rte_range[1], int(rte_steps))

        if not soc_windows:
            st.warning("SOC sweep skipped because all floor/ceiling combinations are invalid.")

        run_sweeps = st.button(
            "Run sensitivity sweeps", use_container_width=True, disabled=not soc_windows
        )

        if "sensitivity_sweep_results" not in st.session_state:
            st.session_state["sensitivity_sweep_results"] = None

        if run_sweeps:
            enforce_rate_limit()
            with st.spinner("Running sensitivity sweep grid..."):
                sweep_df = run_sensitivity_grid(
                    base_cfg=cfg,
                    pv_df=pv_df,
                    cycle_df=cycle_df,
                    dod_override=dod_override,
                    pv_oversize_factors=pv_factors,
                    soc_windows=soc_windows,
                    rte_values=rte_values,
                    simulate_project_fn=simulate_project,
                    summarize_fn=summarize_simulation,
                )

            if sweep_df.empty:
                st.info("No sweep results generated; adjust ranges and retry.")
                st.session_state["sensitivity_sweep_results"] = None
            else:
                base_compliance = summary.compliance
                base_shortfall = summary.total_shortfall_mwh

                sweep_df = sweep_df.assign(
                    compliance_delta_pct=sweep_df["compliance_pct"] - base_compliance,
                    shortfall_delta_mwh=base_shortfall - sweep_df["shortfall_mwh"],
                )
                st.session_state["sensitivity_sweep_results"] = sweep_df

        sweep_df = st.session_state.get("sensitivity_sweep_results")

        if sweep_df is not None:
            ranked_df = sweep_df.sort_values(
                ["compliance_pct", "shortfall_mwh"], ascending=[False, True]
            )
            top_rows = ranked_df.head(5)
            st.markdown("**Top sweep picks (sorted by delivery, then shortfall)**")
            st.dataframe(
                top_rows.round(
                    {
                        "compliance_pct": 2,
                        "bess_share_pct": 2,
                        "shortfall_mwh": 1,
                        "compliance_delta_pct": 2,
                        "shortfall_delta_mwh": 1,
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

            pv_span = (
                sweep_df.groupby("pv_oversize_factor")["compliance_pct"].mean().max()
                - sweep_df.groupby("pv_oversize_factor")["compliance_pct"].mean().min()
            )
            rte_span = (
                sweep_df.groupby("rte_roundtrip")["compliance_pct"].mean().max()
                - sweep_df.groupby("rte_roundtrip")["compliance_pct"].mean().min()
            )
            st.caption(
                "PV oversize swing across tested points: "
                f"{pv_span:,.2f} compliance points. "
                "RTE swing: "
                f"{rte_span:,.2f} compliance points."
            )

            metric_options = {
                "Compliance delta vs base (pct-pts)": {
                    "field": "compliance_delta_pct",
                    "format": ".2f",
                    "scale": alt.Scale(scheme="redyellowgreen", domainMid=0),
                    "title": "Δ compliance (pct-pts)",
                    "higher_is_better": True,
                    "is_delta": True,
                },
                "Total shortfall (MWh)": {
                    "field": "shortfall_mwh",
                    "format": ",.0f",
                    "scale": alt.Scale(scheme="blues", reverse=True),
                    "title": "Total shortfall (MWh)",
                    "higher_is_better": False,
                    "is_delta": False,
                },
                "Shortfall delta vs base (MWh)": {
                    "field": "shortfall_delta_mwh",
                    "format": ",.0f",
                    "scale": alt.Scale(scheme="redyellowgreen", domainMid=0),
                    "title": "Δ shortfall (MWh)",
                    "higher_is_better": True,
                    "is_delta": True,
                },
                "BESS share of firm (%)": {
                    "field": "bess_share_pct",
                    "format": ".2f",
                    "scale": alt.Scale(scheme="greens"),
                    "title": "BESS share of firm (%)",
                    "higher_is_better": True,
                    "is_delta": False,
                },
            }

            selected_metric = st.radio(
                "Heatmap focus",
                list(metric_options.keys()),
                horizontal=True,
                index=0,
            )
            metric_cfg = metric_options[selected_metric]
            metric_field = metric_cfg["field"]

            soc_options = {
                f"{floor:.2f}–{ceiling:.2f}": (floor, ceiling)
                for floor, ceiling in sorted(
                    sweep_df[["soc_floor", "soc_ceiling"]].drop_duplicates().itertuples(
                        index=False, name=None
                    )
                )
            }
            selected_soc_label = st.selectbox(
                "SOC window shown in heatmap", list(soc_options.keys()), index=0
            )
            selected_floor, selected_ceiling = soc_options[selected_soc_label]

            st.markdown(
                "**Heatmap (PV oversize × RTE) for the selected SOC window, following the LCOE/LCOS style.**"
            )
            group = sweep_df[
                (sweep_df["soc_floor"] == selected_floor)
                & (sweep_df["soc_ceiling"] == selected_ceiling)
            ]
            if group.empty:
                st.info("No sweep points available for the selected SOC window.")
            else:
                base_chart = alt.Chart(group).encode(
                    x=alt.X("rte_roundtrip:Q", title="Round-trip efficiency"),
                    y=alt.Y("pv_oversize_factor:Q", title="PV oversize (×)"),
                    color=alt.Color(
                        f"{metric_field}:Q",
                        title=metric_cfg["title"],
                        scale=metric_cfg["scale"],
                    ),
                    tooltip=[
                        alt.Tooltip("rte_roundtrip", title="RTE", format=".3f"),
                        alt.Tooltip("pv_oversize_factor", title="PV oversize", format=".2f"),
                        alt.Tooltip("compliance_pct", title="Compliance %", format=".2f"),
                        alt.Tooltip("bess_share_pct", title="BESS share %", format=".2f"),
                        alt.Tooltip("shortfall_mwh", title="Shortfall (MWh)", format=",.1f"),
                        alt.Tooltip("compliance_delta_pct", title="Δ compliance", format=".2f"),
                        alt.Tooltip("shortfall_delta_mwh", title="Δ shortfall", format=",.0f"),
                    ],
                )

                text_layer = base_chart.mark_text(color="black").encode(
                    text=alt.Text(f"{metric_field}:Q", format=metric_cfg["format"])
                )

                best_row = (
                    group.sort_values(metric_field, ascending=not metric_cfg["higher_is_better"])
                    .iloc[0]
                )
                best_point = alt.Chart(pd.DataFrame([best_row])).mark_point(
                    shape="star", size=120, color="black"
                ).encode(x="rte_roundtrip:Q", y="pv_oversize_factor:Q")

                st.altair_chart(
                    base_chart.mark_rect() + text_layer + best_point,
                    use_container_width=True,
                )

                st.caption(
                    "Deltas are measured against the base run and the chart mirrors the LCOE/LCOS heatmap layout "
                    "for faster scanning."
                )
        else:
            st.info("Run the sensitivity sweeps to populate the heatmap and ranking table.")

    with st.expander("Economics — LCOE / LCOS (separate module)", expanded=False):
        econ_inputs_col1, econ_inputs_col2 = st.columns(2)
        with econ_inputs_col1:
            wacc_pct = st.number_input(
                "WACC (%)",
                min_value=0.0,
                max_value=30.0,
                value=8.0,
                step=0.1,
                help="Weighted-average cost of capital (nominal).",
            )
            inflation_pct = st.number_input(
                "Inflation rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=3.0,
                step=0.1,
                help="Long-run inflation assumption used to derive the real discount rate.",
            )
            discount_rate = max((1 + wacc_pct / 100.0) / (1 + inflation_pct / 100.0) - 1, 0.0)

            capex_musd = st.number_input(
                "Total CAPEX (USD million)",
                min_value=0.0,
                value=40.0,
                step=0.1,
                help="All-in CAPEX for the project. Expressed in USD millions for compact entry.",
            )
        with econ_inputs_col2:
            fixed_opex_pct = (
                st.number_input(
                    "Fixed OPEX (% of CAPEX per year)",
                    min_value=0.0,
                    max_value=20.0,
                    value=2.0,
                    step=0.1,
                    help="Annual fixed OPEX expressed as % of CAPEX.",
                )
                / 100.0
            )
            fixed_opex_musd = st.number_input(
                "Additional fixed OPEX (USD million/yr)",
                min_value=0.0,
                value=0.0,
                step=0.1,
                help="Extra fixed OPEX not tied to CAPEX percentage.",
            )

        def build_economics_output_for_run(
            sim_output: SimulationOutput, cfg_for_run: SimConfig
        ) -> Tuple[float, float, List[float], Any]:
            """Compute economics for a simulation using current discount and cost inputs."""

            results_for_run = sim_output.results
            capex_musd_run = capex_musd

            augmentation_unit_rate_usd_per_kwh = 0.0
            if cfg_for_run.initial_usable_mwh > 0 and capex_musd_run > 0:
                augmentation_unit_rate_usd_per_kwh = (capex_musd_run * 1_000_000.0) / (
                    cfg_for_run.initial_usable_mwh * 1_000.0
                )

            augmentation_energy_added = list(
                getattr(sim_output, "augmentation_energy_added_mwh", [])
            )
            if len(augmentation_energy_added) < len(results_for_run):
                augmentation_energy_added.extend([0.0] * (len(results_for_run) - len(augmentation_energy_added)))
            elif len(augmentation_energy_added) > len(results_for_run):
                augmentation_energy_added = augmentation_energy_added[: len(results_for_run)]

            augmentation_costs_usd = [
                add_e * 1_000.0 * augmentation_unit_rate_usd_per_kwh
                for add_e in augmentation_energy_added
            ]

            economics_inputs = EconomicInputs(
                capex_musd=capex_musd_run,
                fixed_opex_pct_of_capex=fixed_opex_pct,
                fixed_opex_musd=fixed_opex_musd,
                inflation_rate=inflation_pct / 100.0,
                discount_rate=discount_rate,
            )

            economics_output_run = compute_lcoe_lcos_with_augmentation_fallback(
                [r.delivered_firm_mwh for r in results_for_run],
                [r.bess_to_contract_mwh for r in results_for_run],
                economics_inputs,
                augmentation_costs_usd=augmentation_costs_usd,
            )

            return (
                capex_musd_run,
                augmentation_unit_rate_usd_per_kwh,
                augmentation_costs_usd,
                economics_output_run,
            )

        (
            capex_musd,
            augmentation_unit_rate_usd_per_kwh,
            augmentation_costs_usd,
            economics_output,
        ) = build_economics_output_for_run(sim_output, cfg)

        # Reuse simulation-year outputs for downstream economics sensitivity charts.
        results_for_run = sim_output.results

        def _fmt_optional(value: float, scale: float = 1.0, prefix: str = "") -> str:
            return "—" if math.isnan(value) else f"{prefix}{value / scale:,.2f}"

        def _usd_per_mwh_to_php_per_kwh(value: float) -> float:
            if not math.isfinite(value):
                return float("nan")
            return value * USD_TO_PHP / 1000.0

        def _compute_break_even_rates(econ_output: EconomicOutputs) -> Dict[str, float]:
            blended = _usd_per_mwh_to_php_per_kwh(econ_output.lcoe_usd_per_mwh)
            bess_rate = _usd_per_mwh_to_php_per_kwh(econ_output.lcos_usd_per_mwh)

            return {"blended_php_per_kwh": blended, "bess_php_per_kwh": bess_rate}

        break_even_rates = _compute_break_even_rates(economics_output)

        econ_c1, econ_c2, econ_c3 = st.columns(3)
        econ_c1.metric(
            "Discounted costs (USD million)",
            _fmt_optional(economics_output.discounted_costs_usd, scale=1_000_000),
            help="CAPEX at year 0 plus discounted OPEX across the project horizon.",
        )
        econ_c2.metric(
            "LCOE ($/MWh delivered)",
            _fmt_optional(economics_output.lcoe_usd_per_mwh),
            help="Total discounted costs ÷ discounted firm energy delivered.",
        )
        econ_c3.metric(
            "LCOS ($/MWh from BESS)",
            _fmt_optional(economics_output.lcos_usd_per_mwh),
            help="Same cost base but divided by discounted BESS contribution only.",
        )
        if sum(augmentation_costs_usd) > 0:
            st.caption(
                "Augmentation CAPEX included "
                f"(discounted: ${economics_output.discounted_augmentation_costs_usd / 1_000_000:,.2f}M; "
                f"unit cost ${augmentation_unit_rate_usd_per_kwh:,.0f}/kWh)."
            )
        st.caption(
            f"Real discount rate derived from WACC {wacc_pct:.2f}% and inflation {inflation_pct:.2f}%: {discount_rate * 100:.2f}%. "
            "Discounting starts in year 1 for OPEX and energy; CAPEX is treated as a year-0 spend."
        )

        st.markdown("**Break-even selling rates (PHP/kWh @ 58 PHP/USD)**")
        rate_c1, rate_c2 = st.columns(2)
        rate_c1.metric(
            "Blended effective rate",
            _fmt_optional(break_even_rates["blended_php_per_kwh"], prefix="₱"),
            help="LCOE converted to PHP/kWh using the specified exchange rate.",
        )
        rate_c2.metric(
            "BESS selling rate",
            _fmt_optional(break_even_rates["bess_php_per_kwh"], prefix="₱"),
            help="LCOS converted to PHP/kWh for BESS-originated energy.",
        )

        st.markdown("---")
        st.markdown("#### LCOE & LCOS sensitivity (±20%)")
        st.caption(
            "Adjust each cost lever by ±20% while holding other assumptions constant to see how "
            "LCOE and LCOS respond."
        )

        annual_delivered_mwh = [r.delivered_firm_mwh for r in results_for_run]
        annual_bess_mwh = [r.bess_to_contract_mwh for r in results_for_run]

        def _recompute_economics(
            capex_musd_override: float,
            fixed_opex_pct_override: float,
            fixed_opex_musd_override: float,
            discount_rate_override: float,
        ) -> EconomicOutputs:
            """Return LCOE and LCOS after applying the provided economics inputs."""

            updated_inputs = EconomicInputs(
                capex_musd=capex_musd_override,
                fixed_opex_pct_of_capex=fixed_opex_pct_override,
                fixed_opex_musd=fixed_opex_musd_override,
                inflation_rate=inflation_pct / 100.0,
                discount_rate=discount_rate_override,
            )

            return compute_lcoe_lcos_with_augmentation_fallback(
                annual_delivered_mwh,
                annual_bess_mwh,
                updated_inputs,
                augmentation_costs_usd=augmentation_costs_usd,
            )

        lever_definitions = [
            ("CAPEX (USD million)", "capex_musd", capex_musd),
            ("Fixed OPEX (% of CAPEX)", "fixed_opex_pct", fixed_opex_pct),
            ("Fixed OPEX (USD million/yr)", "fixed_opex_musd", fixed_opex_musd),
            ("Discount rate", "discount_rate", discount_rate),
        ]

        sensitivity_rows: List[Dict[str, Any]] = []
        factors = [-0.2, 0.0, 0.2]
        change_labels = {factor: f"{factor:+.0%}" for factor in factors}

        with st.spinner("Computing sensitivity heatmaps..."):
            for lever_label, lever_key, base_value in lever_definitions:
                if base_value < 0:
                    continue

                for factor in factors:
                    capex_test = capex_musd
                    fixed_pct_test = fixed_opex_pct
                    fixed_musd_test = fixed_opex_musd
                    discount_rate_test = discount_rate

                    scaled_value = max(0.0, base_value * (1.0 + factor))

                    if lever_key == "capex_musd":
                        capex_test = scaled_value
                    elif lever_key == "fixed_opex_pct":
                        fixed_pct_test = scaled_value
                    elif lever_key == "fixed_opex_musd":
                        fixed_musd_test = scaled_value
                    elif lever_key == "discount_rate":
                        discount_rate_test = scaled_value

                    economics_outputs = _recompute_economics(
                        capex_test,
                        fixed_pct_test,
                        fixed_musd_test,
                        discount_rate_test,
                    )

                    sensitivity_rows.append(
                        {
                            "Lever": lever_label,
                            "Change": change_labels[factor],
                            "order": factor,
                            "lcoe_usd_per_mwh": economics_outputs.lcoe_usd_per_mwh,
                            "lcos_usd_per_mwh": economics_outputs.lcos_usd_per_mwh,
                        }
                    )

        sensitivity_df = pd.DataFrame(sensitivity_rows)
        if sensitivity_df.empty:
            st.info("Unable to compute sensitivity results with the current inputs.")
        else:
            change_order = [change_labels[f] for f in factors]
            lever_order = [lever_label for lever_label, _, _ in lever_definitions]

            for metric_col, title in [
                ("lcoe_usd_per_mwh", "LCOE sensitivity ($/MWh delivered)"),
                ("lcos_usd_per_mwh", "LCOS sensitivity ($/MWh from BESS)"),
            ]:
                metric_df = sensitivity_df.dropna(subset=[metric_col])
                if metric_df.empty:
                    st.warning(f"No data available to plot {title} heatmap.")
                    continue

                base_chart = alt.Chart(metric_df).encode(
                    x=alt.X("Change:N", sort=change_order, title="Lever adjustment"),
                    y=alt.Y("Lever:N", sort=lever_order, title=""),
                    color=alt.Color(
                        f"{metric_col}:Q",
                        title=title,
                        scale=alt.Scale(scheme="blues"),
                    ),
                    tooltip=[
                        alt.Tooltip("Lever", title="Lever"),
                        alt.Tooltip("Change", title="Adjustment"),
                        alt.Tooltip(metric_col, title=title, format=".1f"),
                    ],
                ).properties(title=title, height=220)

                text_layer = base_chart.mark_text(baseline="middle", color="black").encode(
                    text=alt.Text(f"{metric_col}:Q", format=".1f")
                )

                st.altair_chart(base_chart.mark_rect() + text_layer, use_container_width=True)

            st.caption(
                "Each cell recomputes LCOE/LCOS by scaling one lever ±20% and keeping all other assumptions fixed."
            )

    # --------- KPI Traffic-lights ----------
    st.markdown("### KPI Health")
    def light_icon(color: str) -> str:
        return {"green": "🟢", "yellow": "🟡", "red": "🔴"}[color]

    def eval_coverage(x: float) -> str:
        if math.isnan(x):
            return "red"
        return "green" if x >= 0.90 else ("yellow" if x >= 0.85 else "red")

    def eval_final_soh(pct: float) -> str:
        if math.isnan(pct):
            return "red"
        return "green" if pct >= 75.0 else ("yellow" if pct >= 65.0 else "red")

    def eval_cycles_per_year(e: float) -> str:  # vendor guardrail ~300–400 EFC/yr
        if math.isnan(e):
            return "red"
        return "green" if e <= 300 else ("yellow" if e <= 400 else "red")

    def eval_augmentations(count: int) -> str:
        return "green" if count <= 1 else ("yellow" if count <= 3 else "red")

    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(f"{light_icon(eval_coverage(min_yearly_coverage))} **Worst-year coverage**: {_fmt_percent(min_yearly_coverage, as_fraction=True)}")
    k1.caption("≥90% green · 85–90% yellow")
    k2.markdown(f"{light_icon(eval_final_soh(final_soh_pct))} **Final SOH_total**: {_fmt_percent(final_soh_pct, as_fraction=False)}")
    k2.caption("≥75% green · 65–74% yellow")
    k3.markdown(f"{light_icon(eval_cycles_per_year(avg_eq_cycles_per_year))} **EqCycles/yr (avg)**: {avg_eq_cycles_per_year:.1f}")
    k3.caption("≤300 green · 300–400 yellow")
    k4.markdown(f"{light_icon(eval_augmentations(augmentation_events))} **Augmentation count**: {augmentation_events}")
    k4.caption("0–1 green · 2–3 yellow")

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
    eta_rt_now = max(0.05, min(cfg.rte_roundtrip, 0.9999))  # roundtrip now
    eta_dis_now = eta_rt_now ** 0.5
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
        req_eta_dis_at_soc = target_day / max(1e-9, cfg.initial_usable_mwh * soh_final * delta_soc_adopt)
        req_rte_rt_at_soc = min(0.9999, max(0.0, req_eta_dis_at_soc ** 2))
        rte_rt_adopt = min(RTE_RT_MAX, max(eta_rt_now, req_rte_rt_at_soc))

        # c) finally BOL energy to close any remaining gap
        ebol_req = target_day / max(1e-9, soh_final * delta_soc_adopt * (rte_rt_adopt ** 0.5))
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
        deliverable_soc_only = cfg.initial_usable_mwh * soh_final * delta_soc_adopt * (eta_rt_now ** 0.5)
        deliverable_soc_rte = cfg.initial_usable_mwh * soh_final * delta_soc_adopt * (rte_rt_adopt ** 0.5)
        deliverable_full = ebol_req * soh_final * delta_soc_adopt * (rte_rt_adopt ** 0.5)

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
            ebol_req_soc_only = target_day / max(1e-9, soh_final * delta_soc_adopt * (eta_rt_now ** 0.5))
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
        avg['pv_surplus_mw'] = compute_pv_surplus(
            avg['pv_resource_mw'], avg['pv_to_contract_mw'], avg['charge_mw']
        )
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
        # Extend hourly bounds to support step-style contract lines that cover the full 24-hour window.
        base_x = alt.X(
            'hour:Q',
            title='Hour of Day',
            scale=alt.Scale(domain=[0, 24], nice=False),
            axis=alt.Axis(values=list(range(0, 25, 2)))
        )

        avg_df = avg_df.copy()
        avg_df['hour_end'] = avg_df['hour'] + 1
        base = alt.Chart(avg_df).encode(x=base_x)

        contrib_long = avg_df.melt(id_vars=['hour', 'hour_end'],
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
            .mark_bar(opacity=0.28)
            .encode(
                x=base_x,
                x2='hour_end:Q',
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
                x=base_x,
                x2='hour_end:Q',
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
                x=base_x,
                y=alt.Y('pv_resource_mw:Q', title='MW'),
                tooltip=[alt.Tooltip('pv_resource_mw:Q', title='PV resource (MW)', format='.2f')]
            )
        )

        pv_surplus_area = (
            base
            .mark_area(color='#f7c5c5', opacity=0.45)
            .encode(
                x=alt.X('hour:O', title='Hour of Day', axis=None),
                y=alt.Y('pv_surplus_mw:Q', title='MW'),
                tooltip=[alt.Tooltip('pv_surplus_mw:Q', title='PV surplus (MW)', format='.2f')]
            )
        )

        area_chg = (
            base
            .mark_area(opacity=0.55, color='#caa6ff')
            .encode(y='charge_mw_neg:Q')
        )

        contract_steps = avg_df[['hour', 'contracted_mw']].copy()
        # Append a terminal point at hour 24 so the step line stops at the final bar edge.
        contract_steps = pd.concat([
            contract_steps,
            pd.DataFrame({'hour': [24], 'contracted_mw': [0.0]})
        ], ignore_index=True)
        # Drop the last hour of each discharge window from the visualization (e.g., 4–7 shows 4,5,6).
        contract_steps['contracted_mw'] = contract_steps['contracted_mw'].where(contract_steps['hour'] < 24, 0.0)

        contract_box = (
            alt.Chart(avg_df)
            .mark_rect(color='#f2a900', opacity=0.08, stroke='#f2a900', strokeWidth=1.5)
            .encode(
                x=base_x,
                x2='hour_end:Q',
                y=alt.value(0),
                y2='contracted_mw:Q'
            )
        )
        line_contract = (
            alt.Chart(contract_steps)
            .mark_line(color='#f2a900', strokeWidth=2, interpolate='step-after')
            .encode(x=base_x, y='contracted_mw:Q')
        )

        st.altair_chart(contract_box + contrib_fill + contrib_chart + area_chg + pv_resource_area + pv_surplus_area + line_contract,
                        use_container_width=True)

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
            avg_project['pv_surplus_mw'] = compute_pv_surplus(
                avg_project['pv_resource_mw'],
                avg_project['pv_to_contract_mw'],
                avg_project['charge_mw'],
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
            'pv_surplus_mw': compute_pv_surplus(
                final_year_logs.pv_mw,
                final_year_logs.pv_to_contract_mw,
                final_year_logs.charge_mw,
            ),
        })
        st.download_button("Download final-year hourly logs (CSV)", hourly_df.to_csv(index=False).encode('utf-8'),
                           file_name='final_year_hourly_logs.csv', mime='text/csv')

    pdf_bytes = None
    try:
        pdf_bytes = build_pdf_summary(cfg, results, compliance, bess_share_of_firm, charge_discharge_ratio,
                                      pv_capture_ratio, discharge_capacity_factor,
                                      discharge_windows_text, charge_windows_text)
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
