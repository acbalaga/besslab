"""Input parsing utilities for PV profiles and degradation models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class PVProfileSummary:
    """Summary statistics and metadata for a PV profile."""

    timestep_hours: Optional[float]
    start_timestamp: Optional[pd.Timestamp]
    end_timestamp: Optional[pd.Timestamp]
    hour_index_range: Optional[Tuple[int, int]]
    missing_steps: int
    total_steps: int
    pv_min_mw: float
    pv_max_mw: float
    pv_mean_mw: float
    uses_timestamp: bool


def summarize_pv_profile(df: pd.DataFrame, *, timestamp_col: str = "timestamp") -> PVProfileSummary:
    """Summarize timestep, coverage, missing steps, and PV statistics.

    Missing steps are computed against an inferred cadence. If the profile was
    already gap-filled before loading, missing steps may legitimately be zero.
    """

    if df.empty:
        raise ValueError("PV profile summary requires a non-empty DataFrame.")
    if "pv_mw" not in df.columns:
        raise ValueError("PV profile summary requires a pv_mw column.")

    pv_series = pd.to_numeric(df["pv_mw"], errors="coerce").dropna()
    if pv_series.empty:
        pv_min = float("nan")
        pv_max = float("nan")
        pv_mean = float("nan")
    else:
        pv_min = float(pv_series.min())
        pv_max = float(pv_series.max())
        pv_mean = float(pv_series.mean())

    uses_timestamp = timestamp_col in df.columns

    if uses_timestamp:
        ts_series = pd.to_datetime(df[timestamp_col], errors="coerce").dropna()
        if ts_series.empty:
            raise ValueError("PV profile summary requires valid timestamps.")

        ts_series = ts_series.sort_values()
        start_ts = ts_series.iloc[0]
        end_ts = ts_series.iloc[-1]

        diffs = ts_series.diff().dropna()
        timestep_hours = None
        if not diffs.empty:
            median_diff = diffs.median()
            if pd.notna(median_diff) and median_diff > pd.Timedelta(0):
                timestep_hours = float(median_diff / pd.Timedelta(hours=1))

        unique_ts = ts_series.drop_duplicates()
        total_steps = len(unique_ts)
        missing_steps = 0
        if timestep_hours is not None:
            expected_index = pd.date_range(
                start_ts,
                end_ts,
                freq=pd.to_timedelta(timestep_hours, unit="h"),
            )
            total_steps = len(expected_index)
            missing_steps = max(total_steps - len(unique_ts), 0)

        return PVProfileSummary(
            timestep_hours=timestep_hours,
            start_timestamp=start_ts,
            end_timestamp=end_ts,
            hour_index_range=None,
            missing_steps=missing_steps,
            total_steps=total_steps,
            pv_min_mw=pv_min,
            pv_max_mw=pv_max,
            pv_mean_mw=pv_mean,
            uses_timestamp=True,
        )

    if "hour_index" not in df.columns:
        raise ValueError("PV profile summary requires an hour_index column when timestamps are absent.")

    hour_series = pd.to_numeric(df["hour_index"], errors="coerce").dropna()
    if hour_series.empty:
        raise ValueError("PV profile summary requires valid hour_index values.")

    hour_series = hour_series.astype(int).sort_values().drop_duplicates()
    start_hour = int(hour_series.iloc[0])
    end_hour = int(hour_series.iloc[-1])

    timestep_hours = None
    missing_steps = 0
    total_steps = len(hour_series)
    if len(hour_series) > 1:
        diffs = hour_series.diff().dropna()
        median_step = diffs.median()
        if pd.notna(median_step) and median_step > 0:
            timestep_hours = float(median_step)
            step_value = int(round(median_step))
            if step_value > 0:
                expected_steps = int(((end_hour - start_hour) // step_value) + 1)
                total_steps = expected_steps
                missing_steps = max(expected_steps - len(hour_series), 0)

    return PVProfileSummary(
        timestep_hours=timestep_hours,
        start_timestamp=None,
        end_timestamp=None,
        hour_index_range=(start_hour, end_hour),
        missing_steps=missing_steps,
        total_steps=total_steps,
        pv_min_mw=pv_min,
        pv_max_mw=pv_max,
        pv_mean_mw=pv_mean,
        uses_timestamp=False,
    )


def read_pv_profile(
    path_candidates: List[Any],
    *,
    freq: Optional[str] = None,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Read and validate a PV profile.

    Defaults to the legacy hourly behavior (0–8759 hour_index, padded to 8,760
    rows). When a timestamp column or an explicit ``freq`` is provided, the
    function preserves the native resolution and validates against the implied
    cadence without coercing to 8,760 points. Missing periods in a detected
    range are filled with 0 MW to keep downstream arrays aligned.
    """

    def _clean_hour_index(df: pd.DataFrame, freq_td: Optional[pd.Timedelta]) -> pd.DataFrame:
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
            st.error("hour_index must be integer hours when no timestamp column is provided.")
            raise ValueError("Non-integer hour_index encountered.")

        df["hour_index"] = df["hour_index"].astype(int)

        if df["hour_index"].min() == 1 and 0 not in df["hour_index"].values:
            df["hour_index"] = df["hour_index"] - 1

        out_of_range = df["hour_index"] < 0
        if freq_td is None:
            out_of_range |= df["hour_index"] >= 8760
        if out_of_range.any():
            st.warning(
                "hour_index values outside the expected range were dropped: "
                f"{sorted(df.loc[out_of_range, 'hour_index'].unique().tolist())}"
            )
            df = df.loc[~out_of_range].copy()

        if df.empty:
            raise ValueError("No valid PV rows after removing out-of-range hours.")

        duplicate_mask = df["hour_index"].duplicated(keep=False)
        if duplicate_mask.any():
            st.warning(
                "Duplicate hour_index values found; averaging pv_mw for each step."
            )
            df = (
                df.groupby("hour_index", as_index=False)["pv_mw"].mean()
                .sort_values("hour_index")
                .reset_index(drop=True)
            )
        else:
            df = df.sort_values("hour_index").drop_duplicates("hour_index")

        if freq_td is None:
            full_index = pd.Index(range(8760), name="hour_index")
        else:
            full_index = pd.Index(
                range(df["hour_index"].min(), df["hour_index"].max() + 1),
                name="hour_index",
            )

        df = df.set_index("hour_index")
        missing_steps = full_index.difference(df.index)
        if len(missing_steps) > 0:
            st.warning(
                f"PV CSV is missing {len(missing_steps)} steps; filling gaps with 0 MW."
            )
            df = df.reindex(full_index, fill_value=0.0)
        else:
            df = df.reindex(full_index)

        if freq_td is None and len(df) != 8760:
            st.warning(
                f"PV CSV has {len(df)} rows after cleaning (expected 8760). Proceeding anyway."
            )

        df = df.reset_index()
        df["pv_mw"] = df["pv_mw"].astype(float)
        return df

    def _clean_timestamp(df: pd.DataFrame, freq_td: Optional[pd.Timedelta]) -> pd.DataFrame:
        if "pv_mw" not in df.columns:
            raise ValueError("CSV must contain column: pv_mw")
        if timestamp_col not in df.columns:
            raise ValueError(f"CSV must contain a '{timestamp_col}' column when using timestamps.")

        df = df[[timestamp_col, "pv_mw"]].copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df["pv_mw"] = pd.to_numeric(df["pv_mw"], errors="coerce")

        invalid_rows = df[timestamp_col].isna() | ~np.isfinite(df["pv_mw"])
        if invalid_rows.any():
            st.error(
                "PV CSV contains invalid timestamps or non-numeric pv_mw entries; "
                f"dropping {invalid_rows.sum()} rows."
            )
            df = df.loc[~invalid_rows].copy()

        if df.empty:
            raise ValueError("No valid PV rows after cleaning.")

        df = (
            df.groupby(timestamp_col, as_index=False)["pv_mw"].mean()
            .sort_values(timestamp_col)
            .reset_index(drop=True)
        )
        df = df.set_index(timestamp_col)

        inferred = freq_td
        if inferred is None:
            inferred_str = pd.infer_freq(df.index)
            inferred = pd.Timedelta(inferred_str) if inferred_str is not None else None

        if inferred is None:
            diffs = df.index.to_series().diff().dropna()
            inferred = diffs.median() if not diffs.empty else None

        if inferred is None or inferred <= pd.Timedelta(0):
            raise ValueError("Could not infer PV timestamp frequency.")

        expected_index = pd.date_range(df.index.min(), df.index.max(), freq=inferred)
        missing_steps = expected_index.difference(df.index)
        if len(missing_steps) > 0:
            st.warning(
                f"PV CSV is missing {len(missing_steps)} timestamps; filling gaps with 0 MW."
            )
            df = df.reindex(expected_index, fill_value=0.0)
        else:
            df = df.reindex(expected_index)

        df = df.rename_axis("timestamp").reset_index()
        df["hour_index"] = range(len(df))
        df["pv_mw"] = df["pv_mw"].astype(float)
        return df[["hour_index", "timestamp", "pv_mw"]]

    last_err = None
    for candidate in path_candidates:
        try:
            df = pd.read_csv(candidate)
            freq_td = pd.Timedelta(freq) if freq is not None else None
            if timestamp_col in df.columns or freq_td is not None:
                if timestamp_col in df.columns:
                    return _clean_timestamp(df, freq_td)
                return _clean_hour_index(df, freq_td)
            return _clean_hour_index(df, freq_td)
        except Exception as e:  # pragma: no cover - errors handled via last_err
            last_err = e
    raise RuntimeError(
        "Failed to read PV profile. "
        f"Looked for: {path_candidates}. Last error: {last_err}"
    )


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
        except Exception as e:  # pragma: no cover - errors handled via last_err
            last_err = e
    raise RuntimeError(
        f"Failed to read cycle model. Looked for: {path_candidates}. Last error: {last_err}"
    )


def read_wesm_profile(
    path_candidates: List[Any],
    *,
    forex_rate_php_per_usd: float,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Read and validate an hourly WESM price profile.

    Expected schema (hourly resolution):
    - timestamp or hour_index column for alignment
    - wesm_deficit_price_usd_per_mwh and/or wesm_surplus_price_usd_per_mwh

    If the profile provides PHP/kWh columns (wesm_deficit_price_php_per_kwh and
    wesm_surplus_price_php_per_kwh), they are converted to USD/MWh using the
    provided FX rate (PHP per USD) to keep downstream economics in USD/MWh.
    Profiles provided with month/hour averages (e.g., legacy CSVs with month +
    hr columns) are expanded to an hourly profile for a reference year so that
    hourly simulation logs can be priced consistently.
    """

    if forex_rate_php_per_usd <= 0:
        raise ValueError("forex_rate_php_per_usd must be greater than zero")

    def _normalize_column_name(name: str) -> str:
        return str(name).strip().replace("\ufeff", "").lower()

    def _find_legacy_php_mwh_column(columns: list[str]) -> str | None:
        for col in columns:
            if "central" in col and "php/mwh" in col:
                return col
        return None

    def _expand_month_hour_profile(df: pd.DataFrame) -> pd.DataFrame:
        """Expand a month/hour average profile to an hourly profile.

        Assumes the source provides prices for every month-hour combination and
        uses a non-leap reference year (2023) to generate hourly timestamps.
        """

        month_map = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        month_labels = df["month"].astype(str).str.strip().str.lower().str[:3]
        month_num = month_labels.map(month_map)
        if month_num.isna().any():
            raise ValueError("WESM profile month values must use standard month names.")

        hour_col = pd.to_numeric(df["hr"], errors="coerce")
        if hour_col.isna().any():
            raise ValueError("WESM profile hr values must be numeric.")
        hour_of_day = hour_col.astype(int) - 1
        if (hour_of_day < 0).any() or (hour_of_day > 23).any():
            raise ValueError("WESM profile hr values must be in the 1-24 range.")

        month_hour_prices = df.assign(
            month_num=month_num.astype(int),
            hour_of_day=hour_of_day,
        )[
            [
                "month_num",
                "hour_of_day",
                "wesm_deficit_price_usd_per_mwh",
                "wesm_surplus_price_usd_per_mwh",
            ]
        ]

        hourly_index = pd.date_range(
            "2023-01-01 00:00:00",
            "2023-12-31 23:00:00",
            freq="h",
        )
        hourly_frame = pd.DataFrame(
            {
                "timestamp": hourly_index,
                "month_num": hourly_index.month,
                "hour_of_day": hourly_index.hour,
            }
        )
        expanded = hourly_frame.merge(
            month_hour_prices,
            on=["month_num", "hour_of_day"],
            how="left",
            validate="many_to_one",
        )
        if expanded["wesm_deficit_price_usd_per_mwh"].isna().any():
            raise ValueError("Month/hour WESM profile did not cover all hours.")
        expanded["hour_index"] = range(len(expanded))
        return expanded[
            [
                "timestamp",
                "hour_index",
                "wesm_deficit_price_usd_per_mwh",
                "wesm_surplus_price_usd_per_mwh",
            ]
        ]

    def _clean_profile(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={col: _normalize_column_name(col) for col in df.columns})
        has_timestamp = timestamp_col in df.columns
        has_hour_index = "hour_index" in df.columns
        has_month_hour = "month" in df.columns and "hr" in df.columns
        if not (has_timestamp or has_hour_index or has_month_hour):
            raise ValueError(
                "WESM profile must include timestamp/hour_index or month/hr columns."
            )

        if has_timestamp:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        if has_hour_index:
            df["hour_index"] = pd.to_numeric(df["hour_index"], errors="coerce")

        price_columns = {
            "wesm_deficit_price_usd_per_mwh",
            "wesm_surplus_price_usd_per_mwh",
            "wesm_deficit_price_php_per_kwh",
            "wesm_surplus_price_php_per_kwh",
            "wesm_deficit_price_php_per_mwh",
            "wesm_surplus_price_php_per_mwh",
        }
        available_price_cols = price_columns.intersection(df.columns)
        if not available_price_cols:
            legacy_central = _find_legacy_php_mwh_column(list(df.columns))
            if legacy_central is not None:
                # Legacy WESM files provide Central/High/Low PHP/MWh. Use Central for both
                # deficit and surplus when no explicit split is provided.
                df["wesm_deficit_price_php_per_mwh"] = df[legacy_central]
                df["wesm_surplus_price_php_per_mwh"] = df[legacy_central]
                available_price_cols = price_columns.intersection(df.columns)
        if not available_price_cols:
            raise ValueError(
                "WESM profile must include deficit/surplus price columns in USD/MWh or PHP/kWh."
            )

        for col in available_price_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        invalid_rows = False
        if has_timestamp and df[timestamp_col].isna().any():
            invalid_rows = True
        if has_hour_index and df["hour_index"].isna().any():
            invalid_rows = True
        for col in available_price_cols:
            if df[col].isna().any():
                invalid_rows = True

        if invalid_rows:
            st.error("WESM profile contains invalid timestamps, hour_index values, or prices.")
            drop_cols = list(available_price_cols)
            if has_timestamp:
                drop_cols.append(timestamp_col)
            if has_hour_index:
                drop_cols.append("hour_index")
            df = df.dropna(subset=drop_cols)

        if df.empty:
            raise ValueError("No valid WESM price rows after cleaning.")

        if "wesm_deficit_price_php_per_kwh" in df.columns:
            df["wesm_deficit_price_usd_per_mwh"] = (
                df["wesm_deficit_price_php_per_kwh"] / forex_rate_php_per_usd * 1000.0
            )
        if "wesm_surplus_price_php_per_kwh" in df.columns:
            df["wesm_surplus_price_usd_per_mwh"] = (
                df["wesm_surplus_price_php_per_kwh"] / forex_rate_php_per_usd * 1000.0
            )
        if "wesm_deficit_price_php_per_mwh" in df.columns:
            df["wesm_deficit_price_usd_per_mwh"] = (
                df["wesm_deficit_price_php_per_mwh"] / forex_rate_php_per_usd
            )
        if "wesm_surplus_price_php_per_mwh" in df.columns:
            df["wesm_surplus_price_usd_per_mwh"] = (
                df["wesm_surplus_price_php_per_mwh"] / forex_rate_php_per_usd
            )

        if "wesm_deficit_price_usd_per_mwh" not in df.columns:
            raise ValueError("WESM profile requires wesm_deficit_price_usd_per_mwh values.")

        if "wesm_surplus_price_usd_per_mwh" not in df.columns:
            df["wesm_surplus_price_usd_per_mwh"] = df["wesm_deficit_price_usd_per_mwh"]

        if has_timestamp:
            df = (
                df.groupby(timestamp_col, as_index=False)
                .mean(numeric_only=True)
                .sort_values(timestamp_col)
                .reset_index(drop=True)
            )
        if has_hour_index:
            df = (
                df.groupby("hour_index", as_index=False)
                .mean(numeric_only=True)
                .sort_values("hour_index")
                .reset_index(drop=True)
            )
            if (df["hour_index"] % 1 != 0).any():
                raise ValueError("hour_index must be integer hours.")
            df["hour_index"] = df["hour_index"].astype(int)

        if has_month_hour and not (has_timestamp or has_hour_index):
            df = _expand_month_hour_profile(df)

        return df

    last_err = None
    for candidate in path_candidates:
        try:
            if hasattr(candidate, "seek"):
                candidate.seek(0)
            df = pd.read_csv(candidate)
            return _clean_profile(df)
        except Exception as e:  # pragma: no cover - errors handled via last_err
            last_err = e
    raise ValueError(
        "Failed to read WESM profile. "
        f"Looked for: {path_candidates}. Last error: {last_err}"
    )


def read_wesm_forecast_profile_average(
    path_candidates: List[Any],
    *,
    forex_rate_php_per_usd: float,
    hours_in_year: int = 8760,
) -> pd.DataFrame:
    """Read a multi-year WESM forecast and return an 8760-hour average profile.

    The forecast CSV is expected to contain sequential hourly rows across
    multiple years. We average each hour-of-year across all years to generate
    a single 8760-hour profile for pricing. Forecast files commonly supply
    banded prices; we map the central price to both deficit and surplus to
    align with the shortfall/surplus pricing workflow.
    """

    if forex_rate_php_per_usd <= 0:
        raise ValueError("forex_rate_php_per_usd must be greater than zero")
    if hours_in_year <= 0:
        raise ValueError("hours_in_year must be greater than zero")

    def _normalize_column_name(name: str) -> str:
        return str(name).strip().replace("\ufeff", "").lower()

    def _find_central_column(columns: list[str], unit_token: str) -> str | None:
        for col in columns:
            normalized = _normalize_column_name(col).replace(" ", "")
            if "central" in normalized and unit_token in normalized:
                return col
        return None

    def _clean_profile(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={col: _normalize_column_name(col) for col in df.columns})
        if "hour_index" not in df.columns:
            raise ValueError("Forecast WESM profile must include an hour_index column.")

        df["hour_index"] = pd.to_numeric(df["hour_index"], errors="coerce")
        if df["hour_index"].isna().any():
            st.error("Forecast WESM profile contains invalid hour_index values.")
            df = df.dropna(subset=["hour_index"])

        raw_columns = list(df.columns)
        usd_col = _find_central_column(raw_columns, "usd/mwh")
        if usd_col is None:
            usd_col = _find_central_column(raw_columns, "usdpermwh")
        php_mwh_col = _find_central_column(raw_columns, "php/mwh")
        if php_mwh_col is None:
            php_mwh_col = _find_central_column(raw_columns, "phppermwh")
        php_kwh_col = _find_central_column(raw_columns, "php/kwh")
        if php_kwh_col is None:
            php_kwh_col = _find_central_column(raw_columns, "phpperkwh")

        if usd_col is not None:
            df["wesm_deficit_price_usd_per_mwh"] = pd.to_numeric(df[usd_col], errors="coerce")
        elif php_mwh_col is not None:
            df["wesm_deficit_price_usd_per_mwh"] = (
                pd.to_numeric(df[php_mwh_col], errors="coerce") / forex_rate_php_per_usd
            )
        elif php_kwh_col is not None:
            df["wesm_deficit_price_usd_per_mwh"] = (
                pd.to_numeric(df[php_kwh_col], errors="coerce") / forex_rate_php_per_usd * 1000.0
            )
        else:
            raise ValueError(
                "Forecast WESM profile must include a central price column in USD/MWh, PHP/MWh, or PHP/kWh."
            )

        if df["wesm_deficit_price_usd_per_mwh"].isna().any():
            st.error("Forecast WESM profile contains invalid price values.")
            df = df.dropna(subset=["wesm_deficit_price_usd_per_mwh"])

        if df.empty:
            raise ValueError("No valid WESM forecast rows after cleaning.")

        if (df["hour_index"] < 0).any():
            raise ValueError("Forecast WESM hour_index values must be non-negative.")

        df["hour_of_year"] = df["hour_index"].astype(int) % hours_in_year
        averaged = (
            df.groupby("hour_of_year", as_index=False)["wesm_deficit_price_usd_per_mwh"]
            .mean()
            .sort_values("hour_of_year")
            .reset_index(drop=True)
        )
        if len(averaged) != hours_in_year:
            raise ValueError("Forecast WESM profile does not cover every hour-of-year.")

        averaged["hour_index"] = averaged["hour_of_year"].astype(int)
        averaged["wesm_surplus_price_usd_per_mwh"] = averaged[
            "wesm_deficit_price_usd_per_mwh"
        ]
        return averaged[
            [
                "hour_index",
                "wesm_deficit_price_usd_per_mwh",
                "wesm_surplus_price_usd_per_mwh",
            ]
        ]

    last_err = None
    for candidate in path_candidates:
        try:
            if hasattr(candidate, "seek"):
                candidate.seek(0)
            df = pd.read_csv(candidate)
            return _clean_profile(df)
        except Exception as e:  # pragma: no cover - errors handled via last_err
            last_err = e
    raise ValueError(
        "Failed to read WESM forecast profile. "
        f"Looked for: {path_candidates}. Last error: {last_err}"
    )


def read_wesm_profile_bands(
    path_candidates: List[Any],
    *,
    forex_rate_php_per_usd: float,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """Read WESM price bands (high/central/low) and return USD/MWh values.

    Expected schema (hourly resolution):
    - timestamp or hour_index column for alignment
    - high/central/low price columns in USD/MWh or PHP/MWh (or PHP/kWh)

    Month/hour average profiles are expanded to hourly timestamps for alignment.
    """

    if forex_rate_php_per_usd <= 0:
        raise ValueError("forex_rate_php_per_usd must be greater than zero")

    def _normalize_column_name(name: str) -> str:
        return str(name).strip().replace("\ufeff", "").lower()

    def _expand_month_hour_profile(df: pd.DataFrame) -> pd.DataFrame:
        """Expand a month/hour average profile to an hourly profile."""

        month_map = {
            "jan": 1,
            "feb": 2,
            "mar": 3,
            "apr": 4,
            "may": 5,
            "jun": 6,
            "jul": 7,
            "aug": 8,
            "sep": 9,
            "oct": 10,
            "nov": 11,
            "dec": 12,
        }
        month_labels = df["month"].astype(str).str.strip().str.lower().str[:3]
        month_num = month_labels.map(month_map)
        if month_num.isna().any():
            raise ValueError("WESM profile month values must use standard month names.")

        hour_col = pd.to_numeric(df["hr"], errors="coerce")
        if hour_col.isna().any():
            raise ValueError("WESM profile hr values must be numeric.")
        hour_of_day = hour_col.astype(int) - 1
        if (hour_of_day < 0).any() or (hour_of_day > 23).any():
            raise ValueError("WESM profile hr values must be in the 1-24 range.")

        month_hour_prices = df.assign(
            month_num=month_num.astype(int),
            hour_of_day=hour_of_day,
        )[
            [
                "month_num",
                "hour_of_day",
                "wesm_price_high_usd_per_mwh",
                "wesm_price_central_usd_per_mwh",
                "wesm_price_low_usd_per_mwh",
            ]
        ]

        hourly_index = pd.date_range(
            "2023-01-01 00:00:00",
            "2023-12-31 23:00:00",
            freq="h",
        )
        hourly_frame = pd.DataFrame(
            {
                "timestamp": hourly_index,
                "month_num": hourly_index.month,
                "hour_of_day": hourly_index.hour,
            }
        )
        expanded = hourly_frame.merge(
            month_hour_prices,
            on=["month_num", "hour_of_day"],
            how="left",
            validate="many_to_one",
        )
        if expanded[["wesm_price_high_usd_per_mwh", "wesm_price_central_usd_per_mwh", "wesm_price_low_usd_per_mwh"]].isna().any().any():
            raise ValueError("Month/hour WESM profile did not cover all hours.")
        expanded["hour_index"] = range(len(expanded))
        return expanded[
            [
                "timestamp",
                "hour_index",
                "wesm_price_high_usd_per_mwh",
                "wesm_price_central_usd_per_mwh",
                "wesm_price_low_usd_per_mwh",
            ]
        ]

    def _find_band_column(columns: list[str], band: str, unit_token: str) -> str | None:
        for col in columns:
            normalized = _normalize_column_name(col).replace(" ", "")
            if band in normalized and unit_token in normalized:
                return col
        return None

    def _clean_profile(df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns={col: _normalize_column_name(col) for col in df.columns})
        has_timestamp = timestamp_col in df.columns
        has_hour_index = "hour_index" in df.columns
        has_month_hour = "month" in df.columns and "hr" in df.columns
        if not (has_timestamp or has_hour_index or has_month_hour):
            raise ValueError(
                "WESM profile must include timestamp/hour_index or month/hr columns."
            )

        if has_timestamp:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        if has_hour_index:
            df["hour_index"] = pd.to_numeric(df["hour_index"], errors="coerce")

        raw_columns = list(df.columns)
        bands = ("high", "central", "low")
        price_map: dict[str, str] = {}
        for band in bands:
            usd_col = _find_band_column(raw_columns, band, "usd/mwh")
            if usd_col is None:
                usd_col = _find_band_column(raw_columns, band, "usdpermwh")
            php_mwh_col = _find_band_column(raw_columns, band, "php/mwh")
            if php_mwh_col is None:
                php_mwh_col = _find_band_column(raw_columns, band, "phppermwh")
            php_kwh_col = _find_band_column(raw_columns, band, "php/kwh")
            if php_kwh_col is None:
                php_kwh_col = _find_band_column(raw_columns, band, "phpperkwh")

            if usd_col is not None:
                price_map[f"wesm_price_{band}_usd_per_mwh"] = usd_col
            elif php_mwh_col is not None:
                df[f"wesm_price_{band}_usd_per_mwh"] = (
                    pd.to_numeric(df[php_mwh_col], errors="coerce") / forex_rate_php_per_usd
                )
            elif php_kwh_col is not None:
                df[f"wesm_price_{band}_usd_per_mwh"] = (
                    pd.to_numeric(df[php_kwh_col], errors="coerce") / forex_rate_php_per_usd * 1000.0
                )
            else:
                df[f"wesm_price_{band}_usd_per_mwh"] = np.nan

        for target_col, source_col in price_map.items():
            df[target_col] = pd.to_numeric(df[source_col], errors="coerce")

        price_cols = [
            "wesm_price_high_usd_per_mwh",
            "wesm_price_central_usd_per_mwh",
            "wesm_price_low_usd_per_mwh",
        ]
        if df[price_cols].isna().all().all():
            raise ValueError("WESM profile must include high/central/low price columns.")

        invalid_rows = False
        if has_timestamp and df[timestamp_col].isna().any():
            invalid_rows = True
        if has_hour_index and df["hour_index"].isna().any():
            invalid_rows = True
        if df[price_cols].isna().any().any():
            invalid_rows = True

        if invalid_rows:
            st.error("WESM profile contains invalid timestamps, hour_index values, or prices.")
            drop_cols = price_cols[:]
            if has_timestamp:
                drop_cols.append(timestamp_col)
            if has_hour_index:
                drop_cols.append("hour_index")
            df = df.dropna(subset=drop_cols)

        if df.empty:
            raise ValueError("No valid WESM price rows after cleaning.")

        if has_timestamp:
            df = (
                df.groupby(timestamp_col, as_index=False)
                .mean(numeric_only=True)
                .sort_values(timestamp_col)
                .reset_index(drop=True)
            )
        if has_hour_index:
            df = (
                df.groupby("hour_index", as_index=False)
                .mean(numeric_only=True)
                .sort_values("hour_index")
                .reset_index(drop=True)
            )
            if (df["hour_index"] % 1 != 0).any():
                raise ValueError("hour_index must be integer hours.")
            df["hour_index"] = df["hour_index"].astype(int)

        if has_month_hour and not (has_timestamp or has_hour_index):
            df = _expand_month_hour_profile(df)

        return df[
            [col for col in [timestamp_col, "hour_index"] if col in df.columns]
            + price_cols
        ]

    last_err = None
    for candidate in path_candidates:
        try:
            if hasattr(candidate, "seek"):
                candidate.seek(0)
            df = pd.read_csv(candidate)
            return _clean_profile(df)
        except Exception as e:  # pragma: no cover - errors handled via last_err
            last_err = e
    raise ValueError(
        "Failed to read WESM price bands. "
        f"Looked for: {path_candidates}. Last error: {last_err}"
    )
