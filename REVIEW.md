# Code Review

## Calendar fade applied too early
- **Location:** `app.py`, `calc_calendar_soh` and the start-of-year energy calculation in `simulate_year` (lines 212–236).
- **Issue:** `calc_calendar_soh` returns `(1 - rate) ** year_idx` (or `1 - rate * year_idx`), and `simulate_year` multiplies the beginning-of-year usable energy by this value even for `year_idx == 1`. This reduces BOL energy by one year of calendar fade before any dispatch occurs, making the very first simulated year start with an already degraded battery.
- **Impact:** All compliance, augmentation, and KPI calculations underestimate available energy and power from Year 1 onward. The error compounds across years and can trigger unnecessary augmentation.
- **Suggestion:** Use `year_idx - 1` when computing the calendar retention that applies to the start of the current year (e.g., `calc_calendar_soh(year_idx - 1, ...)` or adjust the helper to treat Year 1 as unity) so that BOL energy is preserved at the beginning of the simulation.

## "Breach days" is only a binary flag
- **Location:** `app.py`, `simulate_year`, lines 315–318.
- **Issue:** `breach_days` is calculated as `int((shortfall_mwh > 0) * 1)`, so the yearly table displays either 0 or 1 regardless of how many days contained shortfalls. This contradicts the column label ("Breach days") and hides the magnitude of reliability issues.
- **Impact:** Users cannot distinguish between a single missed hour and systematic multi-day shortfalls, which undermines the usefulness of the summary table and any downstream analytics.
- **Suggestion:** Track shortfall occurrences per day (similar to how `daily_dis_mwh` is accumulated) and count the number of unique days where `flag_shortfall_hours` occurred, or at least return the total count of shortfall hours.

## Dispatch window parser ignores minutes
- **Location:** `app.py`, `parse_windows`, lines 96–110.
- **Issue:** Inputs are documented as `"HH:MM-HH:MM"`, but the parser drops everything after the colon by casting only the hour component to `int`. Any minute component silently truncates to the hour, so an input like `"05:30-09:00"` becomes `05:00-09:00` without warning.
- **Impact:** Users cannot represent non-integer-hour windows even though the UI advertises minute precision, leading to inaccurate charge/discharge schedules and potentially large energy errors per day.
- **Suggestion:** Parse the minute component (e.g., convert to fractional hours or minutes) or restrict the UI to hour-only inputs and update the help text accordingly.
