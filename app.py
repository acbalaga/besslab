# app.py — BESSLab (PV-only charging, AC-coupled) v0.3.0
# - NEW: Design Advisor + "Required to Meet" calculator (final-year aware)
# - NEW: KPI traffic-light hints vs practical benchmarks
# - Keeps: README/Help, Threshold & SOH-trigger augmentation, EOY capability + PV/BESS split,
#          multi-period daily profiles, flags, downloads

# ---- Abuse protection (rate limit) ----
import time
import streamlit as st


def enforce_rate_limit(max_runs: int = 20, window_seconds: int = 300) -> None:
    """Simple session-based rate limit to deter abuse on open deployments."""
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

    recent.append(now)
    st.session_state["recent_runs"] = recent


enforce_rate_limit()
# ---- end gate ----

import math
import calendar
from dataclasses import dataclass, field
from io import BytesIO
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from pathlib import Path
import altair as alt
from fpdf import FPDF

BASE_DIR = Path(__file__).resolve().parent

# --------- Utilities ---------

def read_pv_profile(path_candidates: List[str]) -> pd.DataFrame:
    """Read PV profile with columns ['hour_index','pv_mw'] in MW. 0..8759 or 1..8760 accepted."""
    last_err = None
    for p in path_candidates:
        try:
            df = pd.read_csv(p)
            if not {'hour_index', 'pv_mw'}.issubset(df.columns):
                raise ValueError("CSV must contain columns: hour_index, pv_mw")
            df = df[['hour_index', 'pv_mw']].copy()
            if df['hour_index'].min() == 1:
                df['hour_index'] = df['hour_index'] - 1
            if df.shape[0] != 8760:
                st.warning(f"PV CSV has {df.shape[0]} rows (expected 8760). Proceeding anyway.")
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read PV profile. Looked for: {path_candidates}. Last error: {last_err}")

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

def infer_dod_bucket(daily_dis_mwh: np.ndarray, usable_mwh_bol: float) -> int:
    if usable_mwh_bol <= 0: return 100
    if len(daily_dis_mwh) == 0: return 10
    med = float(np.median(daily_dis_mwh))
    if med <= 0: return 10
    r = med / max(1e-9, usable_mwh_bol)
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
    soc_ceiling: float = 0.98
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

@dataclass
class YearResult:
    year_index: int
    expected_firm_mwh: float
    delivered_firm_mwh: float
    shortfall_mwh: float
    breach_days: int
    charge_mwh: float
    discharge_mwh: float
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

def calc_calendar_soh(year_idx: int, rate: float, exp_model: bool) -> float:
    return max(0.0, (1.0 - rate) ** year_idx) if exp_model else max(0.0, 1.0 - rate * year_idx)


def calc_calendar_soh_fraction(years_elapsed: float, rate: float, exp_model: bool) -> float:
    return max(0.0, (1.0 - rate) ** years_elapsed) if exp_model else max(0.0, 1.0 - rate * years_elapsed)


def build_pdf_summary(cfg: SimConfig, results: List[YearResult], compliance: float, bess_share: float,
                      charge_discharge_ratio: float, pv_capture_ratio: float,
                      discharge_capacity_factor: float, discharge_windows_text: str,
                      charge_windows_text: str) -> bytes:
    final = results[-1]
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "BESS Lab - Results Snapshot", ln=1)

    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Project life: {cfg.years} years", ln=1)
    pdf.cell(0, 8, f"Contracted MW: {cfg.contracted_mw:.2f} | Initial power: {cfg.initial_power_mw:.2f} MW | Initial usable: {cfg.initial_usable_mwh:.2f} MWh", ln=1)
    pdf.cell(0, 8, f"Discharge windows: {discharge_windows_text}", ln=1)
    pdf.cell(0, 8, f"Charge windows: {charge_windows_text if charge_windows_text else 'Any PV hour'}", ln=1)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Lifecycle KPIs", ln=1)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"Delivery compliance: {compliance:,.2f}%", ln=1)
    pdf.cell(0, 8, f"BESS share of firm: {bess_share:,.1f}%", ln=1)
    pdf.cell(0, 8, f"Charge/Discharge ratio: {charge_discharge_ratio:,.3f}", ln=1)
    pdf.cell(0, 8, f"PV capture ratio: {pv_capture_ratio:,.3f}", ln=1)
    pdf.cell(0, 8, f"Discharge capacity factor (final year): {discharge_capacity_factor:,.3f}", ln=1)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Final-year highlights", ln=1)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, f"EOY usable energy: {final.eoy_usable_mwh:,.1f} MWh", ln=1)
    pdf.cell(0, 8, f"EOY power (avail-adjusted): {final.eoy_power_mw:,.2f} MW", ln=1)
    pdf.cell(0, 8, f"PV to Contract: {final.pv_to_contract_mwh:,.1f} MWh/year", ln=1)
    pdf.cell(0, 8, f"BESS to Contract: {final.bess_to_contract_mwh:,.1f} MWh/year", ln=1)
    pdf.cell(0, 8, f"Eq cycles this year: {final.eq_cycles:,.1f} | Cum cycles: {final.cum_cycles:,.1f}", ln=1)
    pdf.cell(0, 8, f"SOH_total: {final.soh_total:,.3f} (cycle: {final.soh_cycle:,.3f}, calendar: {final.soh_calendar:,.3f})", ln=1)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Notes", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, "Snapshot generated directly from the current Streamlit inputs. Share as a quick reference; rerun the app to update with new parameters.")

    return pdf.output(dest='S').encode('latin-1')

def in_any_window(hod: int, windows: List[Window]) -> bool:
    return any(w.contains(hod) for w in windows)

def simulate_year(state: SimState, year_idx: int, dod_key: Optional[int], need_logs: bool=False) -> Tuple[YearResult, HourlyLog, List[MonthResult]]:
    cfg = state.cfg; dt = cfg.step_hours

    pv_scale = (1.0 - cfg.pv_deg_rate) ** (year_idx - 1)
    pv_mw = state.pv_df['pv_mw'].to_numpy(float) * pv_scale * cfg.pv_availability

    pow_cap_mw = state.current_power_mw * cfg.bess_availability
    soh_cal_start = calc_calendar_soh(max(year_idx - 1, 0), cfg.calendar_fade_rate, cfg.use_calendar_exp_model)
    soh_cal_eoy = calc_calendar_soh(year_idx, cfg.calendar_fade_rate, cfg.use_calendar_exp_model)

    eta_rt = max(0.05, min(cfg.rte_roundtrip, 0.9999))
    eta_ch = eta_rt ** 0.5; eta_dis = eta_rt ** 0.5

    ch_windows = parse_windows(cfg.charge_windows_text)
    dis_windows = cfg.discharge_windows

    dod_for_lookup = dod_key if dod_key else 100
    soh_cycle_pre = cycle_retention_lookup(state.cycle_df, dod_for_lookup, state.cum_cycles)
    usable_mwh_start = state.current_usable_mwh_bolref * soh_cal_start * soh_cycle_pre

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
    month_pv_contract = np.zeros(12)
    month_bess_contract = np.zeros(12)
    month_pv_curtailed = np.zeros(12)
    month_flag_shortfall_hours = np.zeros(12, dtype=int)
    month_flag_soc_floor_hits = np.zeros(12, dtype=int)
    month_flag_soc_ceiling_hits = np.zeros(12, dtype=int)

    expected_firm_mwh = charged_mwh = discharged_mwh = pv_to_contract_mwh = bess_to_contract_mwh = pv_curtailed_mwh = 0.0
    flag_shortfall_hours = flag_soc_floor_hits = flag_soc_ceiling_hits = 0

    for h in range(n_hours):
        is_dis = in_any_window(int(hod[h]), dis_windows)
        is_ch = True if not ch_windows else in_any_window(int(hod[h]), ch_windows)
        pv_avail_mw = max(0.0, pv_mw[h])

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

    dod_key_eff = dod_key if dod_key is not None else infer_dod_bucket(daily_dis_mwh, state.current_usable_mwh_bolref)
    dod_frac = {10:0.10,20:0.20,40:0.40,80:0.80,100:1.00}[dod_key_eff]
    usable_for_cycles = max(1e-9, state.current_usable_mwh_bolref * dod_frac)
    eq_cycles_year = discharged_mwh / usable_for_cycles
    cum_cycles_new = state.cum_cycles + eq_cycles_year
    soh_cycle = cycle_retention_lookup(state.cycle_df, dod_key_eff, cum_cycles_new)
    soh_total = soh_cal_eoy * soh_cycle

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
        pv_to_contract_mwh=pv_to_contract_mwh,
        bess_to_contract_mwh=bess_to_contract_mwh,
        avg_rte=float(avg_rte) if not np.isnan(avg_rte) else float('nan'),
        eq_cycles=float(eq_cycles_year),
        cum_cycles=float(cum_cycles_new),
        soh_cycle=float(soh_cycle),
        soh_calendar=float(soh_cal_eoy),
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
    for m in range(12):
        eq_cycles_month = month_discharge[m] / usable_for_cycles
        cum_cycles_running += eq_cycles_month
        soh_cycle_month = cycle_retention_lookup(state.cycle_df, dod_key_eff, cum_cycles_running)
        years_elapsed = year_idx - 1 + (m + 1) / 12.0
        soh_calendar_month = calc_calendar_soh_fraction(years_elapsed, cfg.calendar_fade_rate, cfg.use_calendar_exp_model)
        soh_total_month = soh_calendar_month * soh_cycle_month
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

def apply_augmentation(state: SimState, cfg: SimConfig, yr: YearResult, discharge_hours_per_day: float) -> Tuple[float, float]:
    """Return (add_power_MW, add_energy_MWh at BOL)."""
    if cfg.augmentation == 'None':
        return 0.0, 0.0

    if cfg.augmentation == 'Threshold' and cfg.aug_trigger_type == 'Capability':
        target_energy_per_day = cfg.contracted_mw * discharge_hours_per_day
        eoy_cap_per_day = min(yr.eoy_usable_mwh, yr.eoy_power_mw * discharge_hours_per_day)
        if eoy_cap_per_day + 1e-6 < target_energy_per_day * (1.0 - cfg.aug_threshold_margin):
            short_mwh = target_energy_per_day * (1.0 + cfg.aug_topup_margin) - eoy_cap_per_day
            short_mwh = max(0.0, short_mwh)
            c_hours = max(1e-9, state.initial_bol_energy_mwh / max(1e-9, state.initial_bol_power_mw))
            add_energy_bol = short_mwh
            add_power = add_energy_bol / c_hours
            return float(add_power), float(add_energy_bol)
        return 0.0, 0.0

    if cfg.augmentation == 'Threshold' and cfg.aug_trigger_type == 'SOH':
        if yr.soh_total <= cfg.aug_soh_trigger_pct + 1e-9:
            add_energy_bol = cfg.aug_soh_add_frac_initial * state.initial_bol_energy_mwh
            c_hours = max(1e-9, state.initial_bol_energy_mwh / max(1e-9, state.initial_bol_power_mw))
            add_power = add_energy_bol / c_hours
            return float(add_power), float(add_energy_bol)
        return 0.0, 0.0

    if cfg.augmentation == 'Periodic':
        if (yr.year_index % max(1, cfg.aug_periodic_every_years)) == 0:
            add_energy_bol = cfg.aug_periodic_add_frac_of_bol * state.current_usable_mwh_bolref
            c_hours = max(1e-9, state.initial_bol_energy_mwh / max(1e-9, state.initial_bol_power_mw))
            add_power = add_energy_bol / c_hours
            return float(add_power), float(add_energy_bol)
        return 0.0, 0.0

    return 0.0, 0.0

# --------- Streamlit UI ---------


def run_app():


    st.set_page_config(page_title="BESSLab by ACB", layout="wide")
    st.title("BESS LAB — PV-only charging, AC-coupled")

    # README / Help
    with st.expander("Help & Guide (click to open)", expanded=False):
        st.markdown("""
    ## BESS Lab version 1 — Help
    by Alfred Balaga

    **Who this is for.**  
    This Streamlit app helps engineers and analysts explore **PV-only charging, AC-coupled BESS** behavior during pre-feasibility studies.
    It originated from studies and technical services work at **Emerging Power Inc. (EPI)** and is shared publicly to support learning, transparency, and community improvements.

    ### What the app does
    - Checks if your **contracted MW × duration** can be met using PV\→Contract first and **BESS for the residual**.
    - Accounts for **PV degradation**, **availability**, **RTE**, **SOC limits**, **calendar + cycle fade**, and (optional) **augmentation**.
    - Surfaces **flags** (shortfalls, SOC hits) and **KPIs** (compliance, capture, cycles, etc.).
    - Visualizes **EOY capability vs target**, **EOY delivered split (PV vs BESS)**, and a **final-year average daily profile**.

    ### Data inputs
    (Already pre-loaded, can be updated upon request)
    - **PV 8760 CSV**: columns `hour_index, pv_mw` (MW at the BESS coupling bus). `hour_index` can be 0–8759 or 1–8760 (1-based will be auto-shifted).  
    - **Cycle model XLSX**: DoD tables (`DoD10_Cycles / DoD10_Ret(%)`, …, `DoD100_*`). If not uploaded, we use the internal table.

    ### Key assumptions (pre-feasibility)
    - **PV-only charge**: no charging from the grid. During discharge windows, **PV serves the contract first**; any PV surplus may charge the BESS.
    - **Single RTE** at POI (internally split √RTE for charge/discharge).  
    - **Availability** is applied to PV energy and BESS power.  
    - **Degradation** = calendar (multiplicative retention) × cycle (from DoD curves).  
    - **Augmentation (optional)**: Threshold (Capability or SOH) or Periodic; **newer cohorts preferentially take more duty** (keeps C-hours).

    ### KPIs (how to read them)
    - **Delivery compliance (%)** = delivered firm ÷ expected firm over the project life.  
    - **BESS share of firm (%)** = portion of firm MWh delivered by BESS (vs PV).  
    - **Charge/Discharge ratio** ≈ 1/RTE (AC context).  
    - **PV capture ratio** = charged ÷ (charged + curtailed).  
    - **Discharge capacity factor (final)** = final-year BESS discharge MWh ÷ (avail-adj MW × discharge-window hours).  
    - **Eq cycles/yr** (guardrail ~300–400): from discharged MWh vs DoD-bucket energy.

    ### Flags (with quick fixes)
    - **Firm shortfall hours**: in-window hours when PV + BESS < contract.  
      *Try:* widen ΔSOC; add BOL MWh; improve RTE; widen charge windows; add augmentation.  
    - **SOC floor hits**: energy-limited (running out).  
      *Try:* raise ceiling / lower floor within limits; add energy; improve RTE.  
    - **SOC ceiling hits**: can’t accept more charge.  
      *Try:* add shoulder discharge; lower ceiling; narrow charge window if unnecessary.

    ### Charts
    - **EOY Capability vs Target**: bars show **energy- vs power-limited** portions; line = contract/day.  
    - **EOY Delivered Split (PV vs BESS)**: average per day; line = contract/day.  
    - **Average Daily Profiles**: view PV→Contract + BESS→Contract (above zero) with charging shown below zero for Year 1, Final Year, and the average across the project; contract line overlaid.

    ### Design Advisor (physics-bounded)
    - Detects **power- vs energy-limit** first.  
    - Suggests bounded deltas (caps: **RTE ≤ 92%**, **ΔSOC ≤ 90%**, **5% ≤ floor**, **ceiling ≤ 98%**).  
    - Checks **PV charge sufficiency** and estimates **extra charge hours/day** needed.  
    - Warns when implied **EqCycles/yr** exceed guardrails (recommend augmentation instead of over-cycling).

    ### Known limitations (by design)
    - No grid charging; no price optimization; no network constraints.  
    - Hourly granularity; sub-hourly only where warranted in later versions.  
    - Warranty, safety, and interconnection compliance are **out of scope** here—refer to OEM docs and standards.

    ### Versioning & feedback
    - You’ll see the version (e.g., `v0.3.x`) in the header.  
    - Send feedback/issues to my work email. I'll triage and iterate.

    *©acbalaga. GNU General Public License v3.0 (GPL-3.0).*
    """)


    with st.sidebar:
        st.header("Data Sources")
        default_pv_paths = [str(BASE_DIR / 'data' / 'PV_8760_MW.csv')]
        default_cycle_paths = [str(BASE_DIR / 'data' / 'cycle_model.xlsx')]

        pv_file = st.file_uploader("PV 8760 CSV (hour_index, pv_mw in MW)", type=['csv'])
        pv_df = pd.read_csv(pv_file) if pv_file is not None else read_pv_profile(default_pv_paths)

        cycle_file = st.file_uploader("Cycle model Excel (optional override)", type=['xlsx'])
        cycle_df = pd.read_excel(cycle_file) if cycle_file is not None else read_cycle_model(default_cycle_paths)

        st.caption("If no files are uploaded, built-in defaults are read from ./data/")

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

    # Augmentation (conditional, with explainers)
    with st.expander("Augmentation strategy", expanded=False):
        aug_mode = st.selectbox("Strategy", ["None", "Threshold", "Periodic"], index=0)
        if aug_mode == "Threshold":
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
    )

    if not cfg.discharge_windows:
        st.error("Please provide at least one discharge window."); st.stop()

    # Discharge hours/day
    dis_hours_per_day = 0.0
    for w in cfg.discharge_windows:
        dis_hours_per_day += (w.end - w.start) if w.start <= w.end else (24 - w.start + w.end)

    # Initial state
    state = SimState(
        pv_df=pv_df,
        cycle_df=cycle_df,
        cfg=cfg,
        current_power_mw=cfg.initial_power_mw,
        current_usable_mwh_bolref=cfg.initial_usable_mwh,
        initial_bol_energy_mwh=cfg.initial_usable_mwh,
        initial_bol_power_mw=cfg.initial_power_mw,
    )

    # Simulate years
    results: List[YearResult] = []
    monthly_results_all: List[MonthResult] = []
    dod_key_override = None if dod_override == "Auto (infer)" else int(dod_override.strip('%'))
    first_year_logs: Optional[HourlyLog] = None
    final_year_logs = None
    hod_count = np.zeros(24, dtype=float)
    hod_sum_pv = np.zeros(24, dtype=float)
    hod_sum_bess = np.zeros(24, dtype=float)
    hod_sum_charge = np.zeros(24, dtype=float)
    for y in range(1, cfg.years + 1):
        yr, logs, monthly_results = simulate_year(state, y, dod_key_override, need_logs=(y == cfg.years))
        hours = np.mod(logs.hod.astype(int), 24)
        np.add.at(hod_count, hours, 1)
        np.add.at(hod_sum_pv, hours, logs.pv_to_contract_mw)
        np.add.at(hod_sum_bess, hours, logs.bess_to_contract_mw)
        np.add.at(hod_sum_charge, hours, logs.charge_mw)
        if y == 1: first_year_logs = logs
        if y == cfg.years: final_year_logs = logs
        state.cum_cycles = yr.cum_cycles
        results.append(yr)
        monthly_results_all.extend(monthly_results)
        add_p, add_e = apply_augmentation(state, cfg, yr, dis_hours_per_day)
        if add_p > 0 or add_e > 0:
            state.current_power_mw += add_p
            state.current_usable_mwh_bolref += add_e

    # Yearly table
    res_df = pd.DataFrame([{
        'Year': r.year_index,
        'Expected firm MWh': r.expected_firm_mwh,
        'Delivered firm MWh': r.delivered_firm_mwh,
        'Shortfall MWh': r.shortfall_mwh,
        'Breach days (has any shortfall)': r.breach_days,
        'Charge MWh': r.charge_mwh,
        'Discharge MWh (from BESS)': r.discharge_mwh,
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
    expected_total = sum(r.expected_firm_mwh for r in results)
    actual_total = sum(r.delivered_firm_mwh for r in results)
    compliance = (actual_total / expected_total * 100.0) if expected_total > 0 else float('nan')
    total_discharge_mwh = sum(r.discharge_mwh for r in results)
    total_charge_mwh = sum(r.charge_mwh for r in results)
    charge_discharge_ratio = (total_charge_mwh / total_discharge_mwh) if total_discharge_mwh > 0 else float('nan')
    bess_share_of_firm = (sum(r.bess_to_contract_mwh for r in results) / actual_total * 100.0) if actual_total > 0 else float('nan')
    pv_capture_ratio = (total_charge_mwh / (total_charge_mwh + sum(r.pv_curtailed_mwh for r in results))) if (total_charge_mwh + sum(r.pv_curtailed_mwh for r in results)) > 0 else float('nan')
    hours_in_discharge_windows_year = dis_hours_per_day * 365.0
    discharge_capacity_factor = (final.discharge_mwh / (final.eoy_power_mw * hours_in_discharge_windows_year)) if final.eoy_power_mw > 0 else float('nan')

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Delivery compliance (%)", f"{compliance:,.2f}")
    c2.metric("BESS share of firm (%)", f"{bess_share_of_firm:,.1f}", help="Portion of contracted energy supplied by BESS vs PV.")
    c3.metric("Charge/Discharge ratio", f"{charge_discharge_ratio:,.3f}", help="AC charged MWh ÷ AC discharged MWh.")
    c4.metric("PV capture ratio", f"{pv_capture_ratio:,.3f}", help="Charged MWh ÷ (Charged MWh + Curtailed MWh).")
    c5.metric("Discharge cap. factor (final yr)", f"{discharge_capacity_factor:,.3f}", help="Final-year BESS discharge MWh ÷ (avail-adjusted MW × discharge-window hours).")

    # --------- KPI Traffic-lights ----------
    st.markdown("### KPI Health (traffic-light hints)")
    def light_icon(color: str) -> str:
        return {"green":"🟢", "yellow":"🟡", "red":"🔴"}[color]

    def eval_rte(rte_single: float) -> str:
        return "green" if rte_single >= 0.85 else ("yellow" if rte_single >= 0.80 else "red")

    def eval_avail(av: float) -> str:
        return "green" if av >= 0.98 else ("yellow" if av >= 0.97 else "red")

    def eval_capture(x: float) -> str:
        return "green" if x >= 0.60 else ("yellow" if x >= 0.50 else "red")

    def eval_cf(x: float) -> str:
        # mid-merit 4–6h typical band
        return "green" if 0.35 <= x <= 0.60 else ("yellow" if 0.30 <= x < 0.35 or 0.60 < x <= 0.70 else "red")

    def eval_cycles_per_year(e):  # vendor guardrail ~300–400 EFC/yr
        return "green" if e <= 300 else ("yellow" if e <= 400 else "red")

    def eval_final_cap_margin(cap_ratio: float) -> str:
        return "green" if cap_ratio >= 1.05 else ("yellow" if cap_ratio >= 1.00 else "red")

    avg_eq_cycles_per_year = float(np.mean([r.eq_cycles for r in results]))
    cap_daily_final = min(final.eoy_usable_mwh, final.eoy_power_mw * dis_hours_per_day)
    cap_ratio_final = cap_daily_final / (cfg.contracted_mw * dis_hours_per_day) if dis_hours_per_day > 0 else float('nan')

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.markdown(f"{light_icon(eval_rte(cfg.rte_roundtrip))} **RTE (single)**: {cfg.rte_roundtrip:.2f}")
    k1.caption("≥0.85 green · 0.80–0.85 yellow")
    k2.markdown(f"{light_icon(eval_avail(cfg.bess_availability))} **BESS availability**: {cfg.bess_availability:.2f}")
    k2.caption("≥0.98 green · 0.97–0.98 yellow")
    k3.markdown(f"{light_icon(eval_capture(pv_capture_ratio))} **PV capture**: {pv_capture_ratio:.3f}")
    k3.caption("≥0.60 green · 0.50–0.60 yellow")
    k4.markdown(f"{light_icon(eval_cf(discharge_capacity_factor))} **Discharge CF (final)**: {discharge_capacity_factor:.3f}")
    k4.caption("0.35–0.60 green for 4–6h mid-merit")
    k5.markdown(f"{light_icon(eval_cycles_per_year(avg_eq_cycles_per_year))} **EqCycles/yr (avg)**: {avg_eq_cycles_per_year:.1f}")
    k5.caption("≤300 green · 300–400 yellow")
    k6.markdown(f"{light_icon(eval_final_cap_margin(cap_ratio_final))} **EOY cap / target**: {cap_ratio_final:.3f}")
    k6.caption("≥1.05 green · 1.00–1.05 yellow")

    st.markdown("---")
    st.subheader("Yearly Summary")
    st.dataframe(res_df.style.format({
        'Expected firm MWh': '{:,.1f}',
        'Delivered firm MWh': '{:,.1f}',
        'Shortfall MWh': '{:,.1f}',
        'Charge MWh': '{:,.1f}',
        'Discharge MWh (from BESS)': '{:,.1f}',
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
    f1.metric("Firm shortfall hours", f"{flag_totals['firm_shortfall_hours']:,}")
    f1.caption("Meaning: In-window hours when PV + BESS could not meet contracted MW.\nFix knobs: increase energy/power, relax windows, augment, or reduce contract.")
    f2.metric("SOC floor hits", f"{flag_totals['soc_floor_hits']:,}")
    f2.caption("Meaning: SOC hit the minimum reserve.\nFix knobs: raise ceiling, lower floor, increase energy, improve RTE, widen charge windows.")
    f3.metric("SOC ceiling hits", f"{flag_totals['soc_ceiling_hits']:,}")
    f3.caption("Meaning: Battery reached upper SOC limit (limited charging).\nFix knobs: increase shoulder discharge, lower ceiling, narrow charge window if unnecessary.")

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

        # OPTION C — Energy at BOL to close the gap (with adopted ΔSOC & RTE)
        if ebol_delta > 1e-6:
            opts.append(f"- **Option C (Energy)**: Increase BOL usable by **{ebol_delta:,.1f} MWh** (to **{ebol_req:,.1f} MWh**).")
        else:
            opts.append(f"- **Option C (Energy)**: BOL usable is sufficient under the adopted ΔSOC/RTE.")

        st.markdown("**Bounded recommendations:**")
        st.markdown("\n".join(opts))

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

        # --- 7) Cycles guardrail hint ---
        if efc_year_prop > EFC_YR_YELLOW:
            st.error(
                f"Implied **EqCycles/yr ≈ {efc_year_prop:,.0f}** (ΔSOC bucket {dod_key_prop}): exceeds typical guardrails. "
                "Prefer **augmentation** (Threshold/SOH) or reduce ΔSOC."
            )
        elif efc_year_prop > EFC_YR_GREEN:
            st.warning(f"Implied **EqCycles/yr ≈ {efc_year_prop:,.0f}**: near vendor guardrail; consider augmentation.")
        else:
            st.info(f"Implied **EqCycles/yr ≈ {efc_year_prop:,.0f}** under proposed ΔSOC/Ebol looks reasonable.")

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
            'pv_to_contract_mw': logs.pv_to_contract_mw,
            'bess_to_contract_mw': logs.bess_to_contract_mw,
            'charge_mw': logs.charge_mw,
            'contracted_mw': contracted_series,
        })
        avg = df_hr.groupby('hod', as_index=False).mean().rename(columns={'hod': 'hour'})
        avg['charge_mw_neg'] = -avg['charge_mw']
        return avg[['hour', 'pv_to_contract_mw', 'bess_to_contract_mw', 'charge_mw_neg', 'contracted_mw']]

    def _render_avg_profile_chart(avg_df: pd.DataFrame) -> None:
        base = alt.Chart(avg_df).encode(x=alt.X('hour:O', title='Hour of Day'))

        contrib_long = avg_df.melt(id_vars=['hour'],
                                   value_vars=['pv_to_contract_mw', 'bess_to_contract_mw'],
                                   var_name='Source', value_name='MW')
        contrib_long['Source'] = contrib_long['Source'].replace({
            'pv_to_contract_mw': 'PV→Contract',
            'bess_to_contract_mw': 'BESS→Contract',
        })
        contrib_chart = (
            alt.Chart(contrib_long)
            .mark_area(opacity=0.65)
            .encode(
                x=alt.X('hour:O', title='Hour of Day'),
                y=alt.Y('MW:Q', title='MW', stack='zero'),
                color=alt.Color('Source:N', scale=alt.Scale(domain=['PV→Contract', 'BESS→Contract'],
                                                           range=['#86c5da', '#7fd18b']))
            )
        )

        area_chg = base.mark_area(opacity=0.5).encode(y='charge_mw_neg:Q', color=alt.value('#caa6ff'))
        line_contract = base.mark_line(color='#f2a900', strokeWidth=2).encode(y='contracted_mw:Q')
        st.altair_chart(contrib_chart + area_chg + line_contract, use_container_width=True)

    if final_year_logs is not None and first_year_logs is not None:
        avg_first_year = _avg_profile_df_from_logs(first_year_logs, cfg)
        avg_final_year = _avg_profile_df_from_logs(final_year_logs, cfg)

        contracted_by_hour = np.array([
            cfg.contracted_mw if any(w.contains(h) for w in cfg.discharge_windows) else 0.0
            for h in range(24)
        ], dtype=float)
        with np.errstate(invalid='ignore', divide='ignore'):
            avg_project = pd.DataFrame({
                'hour': np.arange(24),
                'pv_to_contract_mw': np.divide(hod_sum_pv, hod_count, out=np.zeros_like(hod_sum_pv), where=hod_count > 0),
                'bess_to_contract_mw': np.divide(hod_sum_bess, hod_count, out=np.zeros_like(hod_sum_bess), where=hod_count > 0),
                'charge_mw_neg': -np.divide(hod_sum_charge, hod_count, out=np.zeros_like(hod_sum_charge), where=hod_count > 0),
                'contracted_mw': contracted_by_hour,
            })

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
        st.caption("Positive areas: PV→Contract (blue) + BESS→Contract (green). Negative area: BESS charging (purple). Contract line overlaid (gold).")
    else:
        st.info("Average daily profiles unavailable — simulation logs not generated.")

    st.markdown("---")

    # ---------- Downloads ----------
    st.subheader("Downloads")
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
        })
        st.download_button("Download final-year hourly logs (CSV)", hourly_df.to_csv(index=False).encode('utf-8'),
                           file_name='final_year_hourly_logs.csv', mime='text/csv')

    pdf_bytes = build_pdf_summary(cfg, results, compliance, bess_share_of_firm, charge_discharge_ratio,
                                  pv_capture_ratio, discharge_capacity_factor,
                                  discharge_windows_text, charge_windows_text)
    st.download_button("Download brief PDF snapshot", pdf_bytes,
                       file_name='bess_results_snapshot.pdf', mime='application/pdf')

    st.info("""
    Notes & Caveats:
    - PV-only charging is enforced; during discharge hours, PV first meets the contract, then surplus PV charges the BESS.
    - Threshold augmentation offers **Capability** and **SOH** triggers. Power is added to keep original C-hours.
    - EOY capability = what the fleet can sustain per day at year-end; Delivered Split = what actually happened per day on average.
    - Design Advisor uses a conservative energy-limited view: Deliverable/day ≈ BOL usable × SOH(final) × ΔSOC × η_dis.
    """)

    st.markdown("---")
    comparison_tab = st.tabs(["Scenario comparisons"])[0]
    with comparison_tab:
        st.caption("Save different input sets to compare how capacity and dispatch choices affect KPIs.")
        if "scenario_comparisons" not in st.session_state:
            st.session_state["scenario_comparisons"] = []

        default_label = f"Scenario {len(st.session_state['scenario_comparisons']) + 1}"
        scenario_label = st.text_input("Label for this scenario", default_label)
        if st.button("Add current scenario to table"):
            st.session_state["scenario_comparisons"].append({
                'Label': scenario_label or default_label,
                'Contracted MW': cfg.contracted_mw,
                'Power (BOL MW)': cfg.initial_power_mw,
                'Usable (BOL MWh)': cfg.initial_usable_mwh,
                'Discharge windows': discharge_windows_text,
                'Charge windows': charge_windows_text if charge_windows_text else 'Any PV hour',
                'Compliance (%)': compliance,
                'BESS share of firm (%)': bess_share_of_firm,
                'Charge/Discharge ratio': charge_discharge_ratio,
                'PV capture ratio': pv_capture_ratio,
                'Final EOY usable (MWh)': final.eoy_usable_mwh,
                'Final EOY power (MW)': final.eoy_power_mw,
                'Final eq cycles (year)': final.eq_cycles,
                'Final SOH_total': final.soh_total,
            })
            st.success("Scenario saved. Adjust inputs and add another to compare.")

        if st.session_state["scenario_comparisons"]:
            compare_df = pd.DataFrame(st.session_state["scenario_comparisons"])
            st.dataframe(compare_df.style.format({
                'Compliance (%)': '{:,.2f}',
                'BESS share of firm (%)': '{:,.1f}',
                'Charge/Discharge ratio': '{:,.3f}',
                'PV capture ratio': '{:,.3f}',
                'Final EOY usable (MWh)': '{:,.1f}',
                'Final EOY power (MW)': '{:,.2f}',
                'Final eq cycles (year)': '{:,.1f}',
                'Final SOH_total': '{:,.3f}',
            }))
            if st.button("Clear saved scenarios"):
                st.session_state["scenario_comparisons"] = []
        else:
            st.info("No saved scenarios yet. Tune the inputs above and click 'Add current scenario to table'.")

if __name__ == "__main__":
    run_app()
