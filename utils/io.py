"""Input parsing utilities for PV profiles and degradation models."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st


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
    """

    if forex_rate_php_per_usd <= 0:
        raise ValueError("forex_rate_php_per_usd must be greater than zero")

    def _clean_profile(df: pd.DataFrame) -> pd.DataFrame:
        has_timestamp = timestamp_col in df.columns
        has_hour_index = "hour_index" in df.columns
        if not (has_timestamp or has_hour_index):
            raise ValueError("WESM profile must include a timestamp or hour_index column.")

        if has_timestamp:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        if has_hour_index:
            df["hour_index"] = pd.to_numeric(df["hour_index"], errors="coerce")

        price_columns = {
            "wesm_deficit_price_usd_per_mwh",
            "wesm_surplus_price_usd_per_mwh",
            "wesm_deficit_price_php_per_kwh",
            "wesm_surplus_price_php_per_kwh",
        }
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

        if "wesm_deficit_price_usd_per_mwh" not in df.columns:
            raise ValueError("WESM profile requires wesm_deficit_price_usd_per_mwh values.")

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

        return df

    last_err = None
    for candidate in path_candidates:
        try:
            df = pd.read_csv(candidate)
            return _clean_profile(df)
        except Exception as e:  # pragma: no cover - errors handled via last_err
            last_err = e
    raise RuntimeError(
        "Failed to read WESM profile. "
        f"Looked for: {path_candidates}. Last error: {last_err}"
    )
