"""Input parsing utilities for PV profiles and degradation models."""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
import streamlit as st


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
        try:
            df = pd.read_csv(candidate)
            return _clean(df)
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
