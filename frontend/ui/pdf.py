"""PDF rendering helpers for the BESS Lab Streamlit app.

These utilities centralize layout for the one-page summary so pages and
download handlers can reuse the same rendering without duplicating chart
or table logic.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from fpdf import FPDF

from services.simulation_core import HourlyLog, SimConfig, YearResult, describe_schedule, resolve_efficiencies


def _draw_metric_card(
    pdf: FPDF,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    value: str,
    subtitle: str,
    fill_rgb: Tuple[int, int, int],
) -> None:
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


def _draw_sparkline(
    pdf: FPDF,
    x: float,
    y: float,
    w: float,
    h: float,
    series: List[Tuple[str, List[float], Tuple[int, int, int]]],
    y_label: str,
) -> None:
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


def _draw_section_header(pdf: FPDF, title: str, margin: float, usable_width: float) -> None:
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 7, title, ln=1)
    pdf.set_draw_color(220, 223, 228)
    pdf.line(margin, pdf.get_y(), margin + usable_width, pdf.get_y())
    pdf.ln(2)


def _draw_subsection_title(pdf: FPDF, title: str) -> None:
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(35, 35, 35)
    pdf.cell(0, 5, title, ln=1)
    pdf.set_text_color(0, 0, 0)


def _draw_histogram(
    pdf: FPDF,
    x: float,
    y: float,
    w: float,
    h: float,
    values: np.ndarray,
    bin_edges: np.ndarray,
    color: Tuple[int, int, int],
    label: str,
) -> None:
    pdf.set_draw_color(230, 232, 235)
    pdf.rect(x, y, w, h)
    if values.size == 0:
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(120, 120, 120)
        pdf.set_xy(x + 2, y + h / 2 - 2)
        pdf.cell(w - 4, 4, "No data", align="C")
        pdf.set_text_color(0, 0, 0)
        return

    counts, _ = np.histogram(values, bins=bin_edges)
    max_count = counts.max(initial=0)
    if max_count == 0:
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(120, 120, 120)
        pdf.set_xy(x + 2, y + h / 2 - 2)
        pdf.cell(w - 4, 4, "No events", align="C")
        pdf.set_text_color(0, 0, 0)
        return

    bar_width = w / max(1, len(counts))
    pdf.set_fill_color(*color)
    for idx, count in enumerate(counts):
        bar_height = (count / max_count) * (h - 6)
        bar_x = x + idx * bar_width + 0.5
        bar_y = y + h - bar_height - 2
        pdf.rect(bar_x, bar_y, bar_width - 1, bar_height, style="F")

    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(80, 80, 80)
    pdf.set_xy(x, y + h + 1)
    pdf.cell(w, 4, label, align="C")
    pdf.set_text_color(0, 0, 0)


def _fmt(val: float, suffix: str = "") -> str:
    return f"{val:,.2f}{suffix}" if abs(val) >= 10 else f"{val:,.3f}{suffix}"


def _percent(numerator: float, denominator: float) -> float:
    return (numerator / denominator * 100.0) if denominator > 0 else float("nan")


def _safe_sum(values: List[float]) -> float:
    return float(np.sum(values)) if values else 0.0


def _format_optional(value: Optional[float], suffix: str = "") -> str:
    if value is None or np.isnan(value):
        return "N/A"
    return _fmt(value, suffix)


def _average_profile_from_aggregates(
    cfg: SimConfig,
    hod_count: np.ndarray,
    hod_sum_pv_resource: np.ndarray,
    hod_sum_pv: np.ndarray,
    hod_sum_bess: np.ndarray,
    hod_sum_charge: np.ndarray,
) -> Dict[str, List[float]]:
    """Return the average daily profile across the full project using hourly aggregates."""
    contracted_by_hour = [
        cfg.contracted_mw if any(w.contains(h) for w in cfg.discharge_windows) else 0.0
        for h in range(24)
    ]

    with np.errstate(invalid="ignore", divide="ignore"):
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


def _draw_table(
    pdf: FPDF,
    x: float,
    y: float,
    col_widths: List[float],
    rows: List[List[str]],
    header_fill: Tuple[int, int, int] = (245, 248, 255),
    row_fill: Tuple[int, int, int] = (255, 255, 255),
    border: bool = True,
    header_bold: bool = True,
    font_size: int = 9,
) -> float:
    """Render a simple table and return the updated y position (bottom of table)."""
    pdf.set_xy(x, y)
    pdf.set_font("Helvetica", "B" if header_bold else "", font_size)
    pdf.set_fill_color(*header_fill)
    pdf.set_draw_color(220, 223, 228)
    pdf.set_text_color(20, 20, 20)
    for idx, cell in enumerate(rows[0]):
        pdf.cell(col_widths[idx], 6, cell, border=1 if border else 0, ln=0, align="L", fill=True)
    pdf.ln(6)
    pdf.set_font("Helvetica", "", font_size)
    pdf.set_fill_color(*row_fill)
    for row in rows[1:]:
        pdf.set_x(x)
        for idx, cell in enumerate(row):
            pdf.cell(col_widths[idx], 6, cell, border=1 if border else 0, ln=0, align="L", fill=True)
        pdf.ln(6)
    return pdf.get_y()


def build_pdf_summary(
    cfg: SimConfig,
    results: List[YearResult],
    compliance: float,
    bess_share: float,
    charge_discharge_ratio: float,
    pv_capture_ratio: float,
    discharge_capacity_factor: float,
    discharge_windows_text: str,
    charge_windows_text: str,
    hod_count: np.ndarray,
    hod_sum_pv_resource: np.ndarray,
    hod_sum_pv: np.ndarray,
    hod_sum_bess: np.ndarray,
    hod_sum_charge: np.ndarray,
    total_shortfall_mwh: float,
    pv_excess_mwh: float,
    total_generation_mwh: float,
    bess_generation_mwh: float,
    pv_generation_mwh: float,
    bess_losses_mwh: float,
    final_year_logs: Optional[HourlyLog] = None,
    augmentation_energy_added_mwh: Optional[List[float]] = None,
    augmentation_retired_energy_mwh: Optional[List[float]] = None,
) -> bytes:
    """Render a multi-section PDF snapshot summarizing the latest simulation."""
    if not results:
        pdf = FPDF(format="A4")
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "BESS Lab - One-page Summary")
        pdf.ln(8)
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(90, 90, 90)
        pdf.multi_cell(
            0,
            5,
            "PDF snapshot unavailable because no results were generated. "
            "Run a simulation to view the summary.",
        )
        pdf_bytes = pdf.output(dest="S")
        return pdf_bytes.encode("latin-1") if isinstance(pdf_bytes, str) else bytes(pdf_bytes)

    final = results[-1]
    first = results[0]
    eta_ch, eta_dis, eta_rt = resolve_efficiencies(cfg)
    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    margin = 12
    usable_width = 210 - 2 * margin
    total_expected_mwh = _safe_sum([r.expected_firm_mwh for r in results])
    total_available_pv_mwh = _safe_sum([r.available_pv_mwh for r in results])
    total_delivered_mwh = _safe_sum([r.delivered_firm_mwh for r in results])
    total_charge_mwh = _safe_sum([r.charge_mwh for r in results])
    total_discharge_mwh = _safe_sum([r.discharge_mwh for r in results])
    total_throughput_mwh = total_charge_mwh + total_discharge_mwh
    avg_cycles_per_year = float(np.mean([r.eq_cycles for r in results])) if results else float("nan")
    deficit_pct = _percent(total_shortfall_mwh, total_expected_mwh)
    surplus_pct = _percent(pv_excess_mwh, total_available_pv_mwh)

    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 10, "BESS Lab - Performance & Lifecycle Summary", ln=1)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(
        0,
        6,
        f"Project life: {cfg.years} years  |  Contracted: {cfg.contracted_mw:.1f} MW  |  PV-only charging",
        ln=1,
    )
    pdf.cell(
        0,
        6,
        f"Discharge windows: {discharge_windows_text}  |  Charge windows: {charge_windows_text or 'Any PV hour'}",
        ln=1,
    )
    pdf.ln(3)

    _draw_section_header(pdf, "1. Technical Inputs", margin, usable_width)
    pdf.set_font("Helvetica", "", 9)
    inputs_rows = [
        ["Metric", "Value"],
        ["Initial usable energy", f"{cfg.initial_usable_mwh:,.1f} MWh"],
        ["Initial power", f"{cfg.initial_power_mw:,.1f} MW"],
        ["Contracted power", f"{cfg.contracted_mw:,.1f} MW"],
        ["Round-trip efficiency", f"{eta_rt:.2f}"],
        ["Charge / Discharge efficiency", f"{eta_ch:.2f} / {eta_dis:.2f}" if cfg.use_split_rte else "Symmetric"],
        ["SoC window", f"{cfg.soc_floor:.2f} - {cfg.soc_ceiling:.2f}"],
        ["PV availability", f"{cfg.pv_availability:.2f}"],
        ["BESS availability", f"{cfg.bess_availability:.2f}"],
        ["PV degradation rate", f"{cfg.pv_deg_rate:.3f} / yr"],
        ["Calendar fade rate", f"{cfg.calendar_fade_rate:.3f} / yr"],
        ["Augmentation strategy", cfg.augmentation],
    ]
    if cfg.augmentation_schedule:
        inputs_rows.append(["Manual augmentation schedule", describe_schedule(cfg.augmentation_schedule)])
    table_widths = [usable_width * 0.42, usable_width * 0.58]
    _draw_table(pdf, margin, pdf.get_y(), table_widths, inputs_rows, font_size=8)
    pdf.ln(3)

    _draw_section_header(pdf, "2. Operational Performance", margin, usable_width)
    _draw_subsection_title(pdf, "2.1 Energy charged/discharged, throughput, cycles/year")
    performance_rows = [
        ["Metric", "Project total", "Final year"],
        ["Charge energy", f"{total_charge_mwh:,.1f} MWh", f"{final.charge_mwh:,.1f} MWh"],
        ["Discharge energy", f"{total_discharge_mwh:,.1f} MWh", f"{final.discharge_mwh:,.1f} MWh"],
        ["Throughput (charge+discharge)", f"{total_throughput_mwh:,.1f} MWh", f"{(final.charge_mwh + final.discharge_mwh):,.1f} MWh"],
        ["Equivalent cycles", _format_optional(avg_cycles_per_year), f"{final.eq_cycles:,.1f}"],
    ]
    perf_widths = [usable_width * 0.45, usable_width * 0.27, usable_width * 0.28]
    _draw_table(pdf, margin, pdf.get_y(), perf_widths, performance_rows, font_size=8)
    pdf.ln(2)

    _draw_subsection_title(pdf, "2.2 SOC distribution, clipping, curtailment interaction")
    soc_values = np.array([])
    if final_year_logs is not None and final.eoy_usable_mwh > 0:
        soc_values = np.clip(final_year_logs.soc_mwh / final.eoy_usable_mwh, 0, 1)
    hist_x = margin
    hist_y = pdf.get_y() + 2
    hist_w = usable_width * 0.6
    hist_h = 30
    soc_bins = np.linspace(0, 1, 11)
    _draw_histogram(
        pdf,
        hist_x,
        hist_y,
        hist_w,
        hist_h,
        soc_values,
        soc_bins,
        (134, 197, 218),
        "Final-year SOC distribution (fraction of usable)",
    )
    pdf.set_xy(hist_x + hist_w + 4, hist_y)
    pdf.set_font("Helvetica", "", 8)
    curtailment_pct_final = _percent(final.pv_curtailed_mwh, final.available_pv_mwh)
    clipping_rows = [
        ["Indicator", "Value"],
        ["PV curtailment (final)", f"{curtailment_pct_final:,.2f}%"],
        ["PV curtailment (project)", f"{surplus_pct:,.2f}%"],
        ["SOC ceiling hits (project)", f"{sum(r.flags.get('soc_ceiling_hits', 0) for r in results):,}"],
        ["SOC floor hits (project)", f"{sum(r.flags.get('soc_floor_hits', 0) for r in results):,}"],
    ]
    clipping_widths = [usable_width * 0.2, usable_width * 0.2]
    _draw_table(pdf, hist_x + hist_w + 4, hist_y + 2, clipping_widths, clipping_rows, font_size=7)
    pdf.ln(hist_h + 10)

    _draw_subsection_title(pdf, "2.3 Peak power delivered vs limits; hours at max discharge/charge")
    discharge_peak = None
    charge_peak = None
    hours_at_discharge_limit = None
    hours_at_charge_limit = None
    if final_year_logs is not None:
        discharge_peak = float(np.max(final_year_logs.discharge_mw))
        charge_peak = float(np.max(final_year_logs.charge_mw))
        power_limit = final.eoy_power_mw if final.eoy_power_mw > 0 else cfg.initial_power_mw
        discharge_mask = final_year_logs.discharge_mw >= 0.98 * power_limit
        charge_mask = final_year_logs.charge_mw >= 0.98 * power_limit
        hours_at_discharge_limit = float(discharge_mask.sum()) * cfg.step_hours
        hours_at_charge_limit = float(charge_mask.sum()) * cfg.step_hours
    peak_rows = [
        ["Metric", "Value"],
        ["Discharge peak", _format_optional(discharge_peak, " MW")],
        ["Charge peak", _format_optional(charge_peak, " MW")],
        ["Hours near discharge limit", _format_optional(hours_at_discharge_limit, " h")],
        ["Hours near charge limit", _format_optional(hours_at_charge_limit, " h")],
    ]
    _draw_table(pdf, margin, pdf.get_y(), table_widths, peak_rows, font_size=8)
    pdf.ln(2)

    _draw_subsection_title(pdf, "2.4 Constraint binding analysis")
    total_hours = cfg.years * 8760 / max(cfg.step_hours, 1e-9)
    flag_totals = {
        "Firm shortfall hours": sum(r.flags.get("firm_shortfall_hours", 0) for r in results),
        "SOC floor hits": sum(r.flags.get("soc_floor_hits", 0) for r in results),
        "SOC ceiling hits": sum(r.flags.get("soc_ceiling_hits", 0) for r in results),
    }
    constraint_rows = [["Constraint", "Hours", "Share"]]
    for label, count in sorted(flag_totals.items(), key=lambda item: item[1], reverse=True):
        constraint_rows.append([label, f"{count:,}", f"{_percent(count, total_hours):,.2f}%"])
    constraint_widths = [usable_width * 0.5, usable_width * 0.2, usable_width * 0.3]
    _draw_table(pdf, margin, pdf.get_y(), constraint_widths, constraint_rows, font_size=8)
    pdf.ln(3)

    _draw_section_header(pdf, "3. Contract Compliance / Adequacy", margin, usable_width)
    if cfg.contracted_mw <= 0:
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(
            0,
            4,
            "No contracted power specified. Compliance metrics are shown for reference, but adequacy checks are not applied.",
        )
    _draw_subsection_title(pdf, "3.1 Compliance %, deficit %, surplus %")
    compliance_rows = [
        ["Metric", "Value"],
        ["Compliance (project)", f"{compliance:,.2f}%"],
        ["Deficit share (project)", f"{deficit_pct:,.2f}%"],
        ["PV surplus / curtailment share", f"{surplus_pct:,.2f}%"],
        ["Total delivered", f"{total_delivered_mwh:,.1f} MWh"],
        ["Expected delivered", f"{total_expected_mwh:,.1f} MWh"],
    ]
    _draw_table(pdf, margin, pdf.get_y(), table_widths, compliance_rows, font_size=8)
    pdf.ln(2)

    _draw_subsection_title(pdf, "3.2 Deficit magnitude distribution (final-year hourly)")
    shortfall_values = np.array([])
    if final_year_logs is not None:
        shortfall_values = final_year_logs.shortfall_mw[final_year_logs.shortfall_mw > 0]
    if cfg.contracted_mw > 0:
        bin_edges = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0]) * cfg.contracted_mw
    else:
        max_shortfall = float(shortfall_values.max(initial=0.0))
        bin_edges = np.linspace(0.0, max_shortfall if max_shortfall > 0 else 1.0, 6)
    _draw_histogram(
        pdf,
        margin,
        pdf.get_y(),
        usable_width,
        28,
        shortfall_values,
        bin_edges,
        (255, 196, 196),
        "Hourly shortfall MW (bins)",
    )
    pdf.ln(35)

    _draw_subsection_title(pdf, "3.3 Suggested reserve SOC / oversizing implications (heuristic)")
    p95_shortfall_mw = None
    reserve_soc_pct = None
    if shortfall_values.size > 0 and final.eoy_usable_mwh > 0:
        p95_shortfall_mw = float(np.percentile(shortfall_values, 95))
        reserve_mwh = p95_shortfall_mw * cfg.step_hours
        reserve_soc_pct = reserve_mwh / final.eoy_usable_mwh * 100.0
    reserve_rows = [
        ["Indicator", "Value"],
        ["P95 hourly shortfall", _format_optional(p95_shortfall_mw, " MW")],
        ["Heuristic reserve SOC", _format_optional(reserve_soc_pct, "%")],
        ["Heuristic energy top-up", _format_optional(p95_shortfall_mw, " MW")],
    ]
    _draw_table(pdf, margin, pdf.get_y(), table_widths, reserve_rows, font_size=8)
    pdf.ln(3)

    _draw_section_header(pdf, "4. Degradation and Lifecycle", margin, usable_width)
    _draw_subsection_title(pdf, "4.1 SOH trajectory and drivers")
    years = [r.year_index for r in results]
    soh_cycle_series = [r.soh_cycle for r in results]
    soh_calendar_series = [r.soh_calendar for r in results]
    soh_total_series = [r.soh_total for r in results]
    _draw_sparkline(
        pdf,
        margin,
        pdf.get_y() + 4,
        usable_width,
        28,
        [
            ("cycle", soh_cycle_series, (202, 166, 255)),
            ("calendar", soh_calendar_series, (255, 196, 120)),
            ("total", soh_total_series, (127, 209, 139)),
        ],
        "SOH (fraction of BOL)",
    )
    pdf.ln(35)
    soh_rows = [
        ["Metric", "Value"],
        ["Initial SOH total", f"{first.soh_total:,.3f}"],
        ["Final SOH total", f"{final.soh_total:,.3f}"],
        ["Final SOH cycle", f"{final.soh_cycle:,.3f}"],
        ["Final SOH calendar", f"{final.soh_calendar:,.3f}"],
    ]
    _draw_table(pdf, margin, pdf.get_y(), table_widths, soh_rows, font_size=8)
    pdf.ln(2)

    _draw_subsection_title(pdf, "4.2 Predicted EOL date under each strategy")
    eol_threshold = 0.80
    eol_year = next((r.year_index for r in results if r.soh_total <= eol_threshold), None)
    eol_rows = [
        ["Assumption", "Value"],
        ["EOL SOH threshold", f"{eol_threshold:.2f} (placeholder)"],
        ["First year below threshold", str(eol_year) if eol_year is not None else "Not reached"],
    ]
    if cfg.augmentation == "Threshold" and cfg.aug_trigger_type == "SOH":
        eol_rows.append(["SOH augmentation trigger", f"{cfg.aug_soh_trigger_pct:.2f}"])
    _draw_table(pdf, margin, pdf.get_y(), table_widths, eol_rows, font_size=8)
    pdf.ln(2)

    _draw_subsection_title(pdf, "4.3 Augmentation schedule and KPI impact")
    added_energy = augmentation_energy_added_mwh or []
    retired_energy = augmentation_retired_energy_mwh or [0.0 for _ in added_energy]
    if len(retired_energy) < len(added_energy):
        retired_energy.extend([0.0] * (len(added_energy) - len(retired_energy)))
    augmentation_rows = [["Year", "Added MWh", "Retired MWh"]]
    for idx, added in enumerate(added_energy):
        augmentation_rows.append([str(idx + 1), f"{added:,.1f}", f"{retired_energy[idx]:,.1f}"])
    if len(augmentation_rows) == 1:
        augmentation_rows.append(["-", "N/A", "N/A"])
    aug_widths = [usable_width * 0.2, usable_width * 0.4, usable_width * 0.4]
    _draw_table(pdf, margin, pdf.get_y(), aug_widths, augmentation_rows, font_size=8)
    pdf.ln(2)
    impact_rows = [
        ["KPI", "Value"],
        ["Total augmentation added", f"{sum(added_energy):,.1f} MWh" if added_energy else "N/A"],
        ["Final usable energy", f"{final.eoy_usable_mwh:,.1f} MWh"],
        ["Final power", f"{final.eoy_power_mw:,.1f} MW"],
        ["Final SOH total", f"{final.soh_total:,.3f}"],
    ]
    _draw_table(pdf, margin, pdf.get_y(), table_widths, impact_rows, font_size=8)
    pdf.ln(2)

    pdf.add_page()
    _draw_section_header(pdf, "Appendix: Energy & Generation Summary", margin, usable_width)
    card_width = (usable_width - 10) / 3
    card_height = 22
    x0 = margin
    y0 = pdf.get_y()
    _draw_metric_card(
        pdf,
        x0,
        y0,
        card_width,
        card_height,
        "Delivery compliance",
        f"{compliance:,.2f}%",
        "Across full life",
        (225, 245, 255),
    )
    _draw_metric_card(
        pdf,
        x0 + card_width + 5,
        y0,
        card_width,
        card_height,
        "Delivery deficit",
        f"{deficit_pct:,.2f}%",
        "Shortfall vs expected",
        (255, 238, 238),
    )
    _draw_metric_card(
        pdf,
        x0 + 2 * (card_width + 5),
        y0,
        card_width,
        card_height,
        "PV surplus",
        f"{surplus_pct:,.2f}%",
        "Curtailment share",
        (255, 245, 235),
    )

    y_cards2 = y0 + card_height + 4
    _draw_metric_card(
        pdf,
        x0,
        y_cards2,
        card_width,
        card_height,
        "PV capture ratio",
        _fmt(pv_capture_ratio),
        "PV used vs available",
        (255, 245, 235),
    )
    _draw_metric_card(
        pdf,
        x0 + card_width + 5,
        y_cards2,
        card_width,
        card_height,
        "Discharge CF (final)",
        _fmt(discharge_capacity_factor),
        "Avg MW / contracted",
        (238, 245, 255),
    )
    _draw_metric_card(
        pdf,
        x0 + 2 * (card_width + 5),
        y_cards2,
        card_width,
        card_height,
        "SOH total (final)",
        _fmt(final.soh_total, ""),
        "Cycle & calendar combined",
        (240, 240, 240),
    )

    pdf.set_y(y_cards2 + card_height + 6)
    chart_height = 55
    chart_x_left = margin
    chart_y = pdf.get_y()

    avg_profile = _average_profile_from_aggregates(
        cfg, hod_count, hod_sum_pv_resource, hod_sum_pv, hod_sum_bess, hod_sum_charge
    )
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
        zero_y = chart_y + chart_height - (0.0 / span * chart_height)
        pdf.set_draw_color(200, 200, 200)
        pdf.line(chart_x_left, zero_y, chart_x_left + usable_width, zero_y)

        def _polyline(values: List[float], color: Tuple[int, int, int], invert: bool = False) -> None:
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

    pdf.ln(2)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(
        0,
        4,
        "Auto-generated from current Streamlit inputs. Heuristic reserve SOC and EOL assumptions are placeholders and "
        "should be replaced with warranty or contract-specific thresholds.",
    )

    pdf_bytes = pdf.output(dest="S")
    return pdf_bytes.encode("latin-1") if isinstance(pdf_bytes, str) else bytes(pdf_bytes)
