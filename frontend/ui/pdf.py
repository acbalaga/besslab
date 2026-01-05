"""PDF rendering helpers for the BESS Lab Streamlit app.

These utilities centralize layout for the one-page summary so pages and
download handlers can reuse the same rendering without duplicating chart
or table logic.
"""

from typing import Dict, List, Tuple

import numpy as np
from fpdf import FPDF

from services.simulation_core import SimConfig, YearResult, describe_schedule, resolve_efficiencies


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


def _fmt(val: float, suffix: str = "") -> str:
    return f"{val:,.2f}{suffix}" if abs(val) >= 10 else f"{val:,.3f}{suffix}"


def _percent(numerator: float, denominator: float) -> float:
    return (numerator / denominator * 100.0) if denominator > 0 else float("nan")


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
) -> bytes:
    """Render a one-page PDF snapshot summarizing the latest simulation."""
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
    pdf.add_page()
    margin = 12
    usable_width = 210 - 2 * margin

    pdf.set_font("Helvetica", "B", 15)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 10, "BESS Lab - One-page Summary", ln=1)
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
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Inputs used", ln=1)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(
        0,
        5,
        f"Initial power: {cfg.initial_power_mw:.1f} MW  |  Initial usable: {cfg.initial_usable_mwh:.1f} MWh  |  RTE: {eta_rt:.2f}",
        ln=1,
    )
    if cfg.use_split_rte:
        pdf.cell(0, 5, f"Charge efficiency: {eta_ch:.2f}  |  Discharge efficiency: {eta_dis:.2f}", ln=1)
    pdf.cell(
        0,
        5,
        f"PV availability: {cfg.pv_availability:.2f}  |  BESS availability: {cfg.bess_availability:.2f}  |  SoC window: {cfg.soc_floor:.2f}-{cfg.soc_ceiling:.2f}",
        ln=1,
    )
    pdf.cell(
        0,
        5,
        f"PV deg: {cfg.pv_deg_rate:.3f}/yr  |  Calendar fade: {cfg.calendar_fade_rate:.3f}/yr  |  Augmentation: {cfg.augmentation}",
        ln=1,
    )
    if cfg.augmentation_schedule:
        pdf.cell(0, 5, f"Manual augmentation: {describe_schedule(cfg.augmentation_schedule)}", ln=1)
    pdf.ln(2)

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
    deficit_pct = _percent(total_shortfall_mwh, sum(r.expected_firm_mwh for r in results))
    surplus_pct = _percent(pv_excess_mwh, sum(r.available_pv_mwh for r in results))
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
    pdf.multi_cell(
        0,
        4,
        "Auto-generated from current Streamlit inputs. Keep everything on one page by focusing on the metrics that "
        "shape bankability and warranty conversations.",
    )

    pdf_bytes = pdf.output(dest="S")
    return pdf_bytes.encode("latin-1") if isinstance(pdf_bytes, str) else bytes(pdf_bytes)
