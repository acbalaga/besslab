"""Flag metadata and insights helpers."""

from __future__ import annotations

from typing import Dict, List

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
