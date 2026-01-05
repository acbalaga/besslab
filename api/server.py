from __future__ import annotations

import os
import uuid
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, root_validator, validator

from services.simulation_core import (
    AUGMENTATION_SCHEDULE_BASIS,
    AugmentationScheduleEntry,
    SimConfig,
    SimulationOutput,
    SimulationSummary,
    Window,
    infer_step_hours_from_pv,
    parse_windows,
    simulate_project,
    summarize_simulation,
    validate_pv_profile_duration,
)
from utils.sweeps import sweep_bess_sizes

_DEFAULT_CFG = SimConfig()
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class WindowPayload(BaseModel):
    start: float
    end: float

    def to_window(self) -> Window:
        return Window(self.start, self.end)


class AugmentationSchedulePayload(BaseModel):
    year: int
    basis: str
    value: float

    @validator("basis")
    def _validate_basis(cls, value: str) -> str:
        if value not in AUGMENTATION_SCHEDULE_BASIS:
            raise ValueError(f"basis must be one of {AUGMENTATION_SCHEDULE_BASIS}")
        return value

    def to_entry(self) -> AugmentationScheduleEntry:
        return AugmentationScheduleEntry(year=self.year, basis=self.basis, value=self.value)


class SimConfigPayload(BaseModel):
    """Pydantic mirror of :class:`SimConfig` for FastAPI requests."""

    years: int = _DEFAULT_CFG.years
    step_hours: float = _DEFAULT_CFG.step_hours
    pv_deg_rate: float = _DEFAULT_CFG.pv_deg_rate
    pv_availability: float = _DEFAULT_CFG.pv_availability
    bess_availability: float = _DEFAULT_CFG.bess_availability
    rte_roundtrip: float = _DEFAULT_CFG.rte_roundtrip
    use_split_rte: bool = _DEFAULT_CFG.use_split_rte
    charge_efficiency: Optional[float] = _DEFAULT_CFG.charge_efficiency
    discharge_efficiency: Optional[float] = _DEFAULT_CFG.discharge_efficiency
    soc_floor: float = _DEFAULT_CFG.soc_floor
    soc_ceiling: float = _DEFAULT_CFG.soc_ceiling
    initial_power_mw: float = _DEFAULT_CFG.initial_power_mw
    initial_usable_mwh: float = _DEFAULT_CFG.initial_usable_mwh
    contracted_mw: float = _DEFAULT_CFG.contracted_mw
    discharge_windows_text: Optional[str] = None
    discharge_windows: Optional[List[WindowPayload]] = None
    charge_windows_text: str = _DEFAULT_CFG.charge_windows_text
    charge_windows: Optional[List[WindowPayload]] = None
    max_cycles_per_day_cap: float = _DEFAULT_CFG.max_cycles_per_day_cap
    calendar_fade_rate: float = _DEFAULT_CFG.calendar_fade_rate
    use_calendar_exp_model: bool = _DEFAULT_CFG.use_calendar_exp_model
    augmentation: str = _DEFAULT_CFG.augmentation
    aug_trigger_type: str = _DEFAULT_CFG.aug_trigger_type
    aug_threshold_margin: float = _DEFAULT_CFG.aug_threshold_margin
    aug_topup_margin: float = _DEFAULT_CFG.aug_topup_margin
    aug_soh_trigger_pct: float = _DEFAULT_CFG.aug_soh_trigger_pct
    aug_soh_add_frac_initial: float = _DEFAULT_CFG.aug_soh_add_frac_initial
    aug_periodic_every_years: int = _DEFAULT_CFG.aug_periodic_every_years
    aug_periodic_add_frac_of_bol: float = _DEFAULT_CFG.aug_periodic_add_frac_of_bol
    aug_add_mode: str = _DEFAULT_CFG.aug_add_mode
    aug_fixed_energy_mwh: float = _DEFAULT_CFG.aug_fixed_energy_mwh
    aug_retire_old_cohort: bool = _DEFAULT_CFG.aug_retire_old_cohort
    aug_retire_soh_pct: float = _DEFAULT_CFG.aug_retire_soh_pct
    augmentation_schedule: List[AugmentationSchedulePayload] = Field(default_factory=list)

    def build(self) -> Tuple[SimConfig, List[str]]:
        """Return a :class:`SimConfig` and any window warnings."""

        warnings: List[str] = []
        discharge_windows = self._resolve_windows(
            explicit=self.discharge_windows,
            text=self.discharge_windows_text,
            fallback=_DEFAULT_CFG.discharge_windows,
            warnings=warnings,
            field_name="discharge_windows",
        )
        charge_windows = self._resolve_windows(
            explicit=self.charge_windows,
            text=self.charge_windows_text,
            fallback=_DEFAULT_CFG.charge_windows,
            warnings=warnings,
            field_name="charge_windows",
            allow_empty=True,
        )

        schedule_entries = [entry.to_entry() for entry in self.augmentation_schedule]

        cfg = SimConfig(
            years=self.years,
            step_hours=self.step_hours,
            pv_deg_rate=self.pv_deg_rate,
            pv_availability=self.pv_availability,
            bess_availability=self.bess_availability,
            rte_roundtrip=self.rte_roundtrip,
            use_split_rte=self.use_split_rte,
            charge_efficiency=self.charge_efficiency if self.use_split_rte else None,
            discharge_efficiency=self.discharge_efficiency if self.use_split_rte else None,
            soc_floor=self.soc_floor,
            soc_ceiling=self.soc_ceiling,
            initial_power_mw=self.initial_power_mw,
            initial_usable_mwh=self.initial_usable_mwh,
            contracted_mw=self.contracted_mw,
            discharge_windows=discharge_windows,
            charge_windows_text=self.charge_windows_text,
            charge_windows=charge_windows,
            max_cycles_per_day_cap=self.max_cycles_per_day_cap,
            calendar_fade_rate=self.calendar_fade_rate,
            use_calendar_exp_model=self.use_calendar_exp_model,
            augmentation=self.augmentation,
            aug_trigger_type=self.aug_trigger_type,
            aug_threshold_margin=self.aug_threshold_margin,
            aug_topup_margin=self.aug_topup_margin,
            aug_soh_trigger_pct=self.aug_soh_trigger_pct,
            aug_soh_add_frac_initial=self.aug_soh_add_frac_initial,
            aug_periodic_every_years=self.aug_periodic_every_years,
            aug_periodic_add_frac_of_bol=self.aug_periodic_add_frac_of_bol,
            aug_add_mode=self.aug_add_mode,
            aug_fixed_energy_mwh=self.aug_fixed_energy_mwh,
            aug_retire_old_cohort=self.aug_retire_old_cohort,
            aug_retire_soh_pct=self.aug_retire_soh_pct,
            augmentation_schedule=schedule_entries,
        )

        return cfg, warnings

    @staticmethod
    def _resolve_windows(
        explicit: Optional[List[WindowPayload]],
        text: Optional[str],
        fallback: List[Window],
        warnings: List[str],
        field_name: str,
        allow_empty: bool = False,
    ) -> List[Window]:
        if explicit:
            return [w.to_window() for w in explicit]

        if text:
            windows, window_warnings = parse_windows(text)
            warnings.extend(window_warnings)
            if windows:
                return windows
            if not allow_empty:
                raise HTTPException(status_code=400, detail=f"{field_name} could not be parsed.")

        if not fallback and not allow_empty:
            raise HTTPException(status_code=400, detail=f"{field_name} must include at least one window.")
        return list(fallback)


class PVRow(BaseModel):
    pv_mw: float
    hour_index: Optional[int] = None
    timestamp: Optional[str] = None


class DataSource(BaseModel):
    pv_upload_id: Optional[str] = None
    cycle_upload_id: Optional[str] = None
    use_sample_pv: bool = True
    use_sample_cycle: bool = True
    pv_rows: Optional[List[PVRow]] = None
    cycle_rows: Optional[List[Dict[str, Any]]] = None

    @root_validator
    def _at_least_one_source(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not any(
            [
                values.get("pv_upload_id"),
                values.get("cycle_upload_id"),
                values.get("pv_rows"),
                values.get("cycle_rows"),
                values.get("use_sample_pv"),
                values.get("use_sample_cycle"),
            ]
        ):
            raise ValueError("Provide PV/cycle data or enable sample inputs.")
        return values


class SimulationRequest(BaseModel):
    config: SimConfigPayload = Field(default_factory=SimConfigPayload)
    data: DataSource = Field(default_factory=DataSource)
    dod_override: str = "Auto (infer)"
    include_hourly_logs: bool = False


class SweepRequest(BaseModel):
    config: SimConfigPayload = Field(default_factory=SimConfigPayload)
    data: DataSource = Field(default_factory=DataSource)
    dod_override: str = "Auto (infer)"
    power_values: Optional[List[float]] = None
    duration_values: Optional[List[float]] = None
    energy_values: Optional[List[float]] = None
    fixed_power_mw: Optional[float] = None
    min_soh: float = 0.6
    min_compliance_pct: Optional[float] = None
    max_shortfall_mwh: Optional[float] = None
    max_candidates: Optional[int] = 100
    on_exceed: Literal["raise", "return", "batch"] = "raise"
    batch_size: Optional[int] = None

    @root_validator
    def _ensure_candidate_space(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if not any([values.get("power_values"), values.get("duration_values"), values.get("energy_values")]):
            raise ValueError("Provide at least one of power_values, duration_values, or energy_values.")
        return values


class BatchRun(BaseModel):
    name: Optional[str] = None
    config: SimConfigPayload


class BatchRequest(BaseModel):
    data: DataSource = Field(default_factory=DataSource)
    dod_override: str = "Auto (infer)"
    runs: List[BatchRun]

    @root_validator
    def _require_runs(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        runs = values.get("runs") or []
        if not runs:
            raise ValueError("Provide at least one run in 'runs'.")
        return values


class UploadPayload(BaseModel):
    kind: Literal["pv", "cycle"]
    name: Optional[str] = None
    pv_rows: Optional[List[PVRow]] = None
    cycle_rows: Optional[List[Dict[str, Any]]] = None

    @root_validator
    def _validate_rows(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        kind = values.get("kind")
        if kind == "pv" and not values.get("pv_rows"):
            raise ValueError("pv_rows are required when kind='pv'.")
        if kind == "cycle" and not values.get("cycle_rows"):
            raise ValueError("cycle_rows are required when kind='cycle'.")
        return values


class UploadStore:
    """In-memory upload cache for PV and cycle data frames."""

    def __init__(self) -> None:
        self._pv: Dict[str, pd.DataFrame] = {}
        self._cycle: Dict[str, pd.DataFrame] = {}
        self._lock = Lock()

    def store_pv(self, df: pd.DataFrame, name: Optional[str] = None) -> str:
        upload_id = name or str(uuid.uuid4())
        with self._lock:
            self._pv[upload_id] = df.copy()
        return upload_id

    def store_cycle(self, df: pd.DataFrame, name: Optional[str] = None) -> str:
        upload_id = name or str(uuid.uuid4())
        with self._lock:
            self._cycle[upload_id] = df.copy()
        return upload_id

    def get_pv(self, upload_id: str) -> pd.DataFrame:
        with self._lock:
            if upload_id not in self._pv:
                raise HTTPException(status_code=404, detail=f"PV upload '{upload_id}' not found.")
            return self._pv[upload_id].copy()

    def get_cycle(self, upload_id: str) -> pd.DataFrame:
        with self._lock:
            if upload_id not in self._cycle:
                raise HTTPException(status_code=404, detail=f"Cycle upload '{upload_id}' not found.")
            return self._cycle[upload_id].copy()


@lru_cache(maxsize=1)
def _sample_pv() -> pd.DataFrame:
    return pd.read_csv(_DATA_DIR / "PV_8760_MW.csv")


@lru_cache(maxsize=1)
def _sample_cycle() -> pd.DataFrame:
    return pd.read_excel(_DATA_DIR / "cycle_model.xlsx")


def _pv_rows_to_df(rows: List[PVRow]) -> pd.DataFrame:
    df = pd.DataFrame([row.dict() for row in rows])
    if "pv_mw" not in df.columns:
        raise HTTPException(status_code=400, detail="PV rows must include 'pv_mw'.")
    if "hour_index" not in df.columns or df["hour_index"].isnull().all():
        df["hour_index"] = range(len(df))
    df["pv_mw"] = pd.to_numeric(df["pv_mw"], errors="coerce")
    df["hour_index"] = pd.to_numeric(df["hour_index"], errors="coerce")
    if df["pv_mw"].isnull().any() or df["hour_index"].isnull().any():
        raise HTTPException(status_code=400, detail="pv_mw and hour_index must be numeric.")
    df = df.sort_values("hour_index").reset_index(drop=True)
    return df[["hour_index", "pv_mw"] + ([col for col in ["timestamp"] if col in df.columns])]


def _cycle_rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        raise HTTPException(status_code=400, detail="cycle_rows cannot be empty.")
    return pd.DataFrame(rows)


def _serialize_log(log: Optional[Any]) -> Optional[Dict[str, Any]]:
    if log is None:
        return None
    return {
        "hod": log.hod.tolist(),
        "pv_mw": log.pv_mw.tolist(),
        "pv_to_contract_mw": log.pv_to_contract_mw.tolist(),
        "bess_to_contract_mw": log.bess_to_contract_mw.tolist(),
        "charge_mw": log.charge_mw.tolist(),
        "discharge_mw": log.discharge_mw.tolist(),
        "soc_mwh": log.soc_mwh.tolist(),
    }


def _serialize_config(cfg: SimConfig) -> Dict[str, Any]:
    data = asdict(cfg)
    data["discharge_windows"] = [{"start": w.start, "end": w.end} for w in cfg.discharge_windows]
    data["charge_windows"] = [{"start": w.start, "end": w.end} for w in cfg.charge_windows]
    data["augmentation_schedule"] = [entry.to_dict() for entry in cfg.augmentation_schedule]
    return data


def _serialize_simulation_output(sim_output: SimulationOutput) -> Dict[str, Any]:
    return {
        "config": _serialize_config(sim_output.cfg),
        "discharge_hours_per_day": sim_output.discharge_hours_per_day,
        "results": [asdict(r) for r in sim_output.results],
        "monthly_results": [asdict(r) for r in sim_output.monthly_results],
        "first_year_logs": _serialize_log(sim_output.first_year_logs),
        "final_year_logs": _serialize_log(sim_output.final_year_logs),
        "hod_count": sim_output.hod_count.tolist(),
        "hod_sum_pv": sim_output.hod_sum_pv.tolist(),
        "hod_sum_pv_resource": sim_output.hod_sum_pv_resource.tolist(),
        "hod_sum_bess": sim_output.hod_sum_bess.tolist(),
        "hod_sum_charge": sim_output.hod_sum_charge.tolist(),
        "augmentation_energy_added_mwh": sim_output.augmentation_energy_added_mwh,
        "augmentation_retired_energy_mwh": sim_output.augmentation_retired_energy_mwh,
        "augmentation_events": sim_output.augmentation_events,
    }


def _serialize_summary(summary: SimulationSummary) -> Dict[str, Any]:
    return asdict(summary)


def _resolve_inputs(data: DataSource, uploads: UploadStore) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    warnings: List[str] = []
    if data.pv_rows:
        pv_df = _pv_rows_to_df(data.pv_rows)
    elif data.pv_upload_id:
        pv_df = uploads.get_pv(data.pv_upload_id)
    elif data.use_sample_pv:
        pv_df = _sample_pv().copy()
        warnings.append("Using bundled sample PV profile (data/PV_8760_MW.csv).")
    else:
        raise HTTPException(status_code=400, detail="PV data not provided.")

    if data.cycle_rows:
        cycle_df = _cycle_rows_to_df(data.cycle_rows)
    elif data.cycle_upload_id:
        cycle_df = uploads.get_cycle(data.cycle_upload_id)
    elif data.use_sample_cycle:
        cycle_df = _sample_cycle().copy()
        warnings.append("Using bundled sample cycle table (data/cycle_model.xlsx).")
    else:
        raise HTTPException(status_code=400, detail="Cycle data not provided.")

    return pv_df, cycle_df, warnings


def _validate_and_hydrate_config(cfg_payload: SimConfigPayload, pv_df: pd.DataFrame) -> Tuple[SimConfig, List[str]]:
    cfg, warnings = cfg_payload.build()
    inferred_step = infer_step_hours_from_pv(pv_df)
    if inferred_step is not None:
        cfg.step_hours = inferred_step
        warnings.append(f"step_hours inferred from timestamps: {inferred_step}")
    duration_error = validate_pv_profile_duration(pv_df, cfg.step_hours)
    if duration_error:
        raise HTTPException(status_code=400, detail=duration_error)
    return cfg, warnings


uploads = UploadStore()
app = FastAPI(
    title="BESSLab API",
    description="Lightweight REST API for running BESSLab simulations outside Streamlit.",
    version="0.1.0",
)


_default_cors_origins = [
    # Vite dev/preview servers
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]
_allowed_origins_env = os.getenv("BESSLAB_CORS_ORIGINS", "")
_allowed_origins = [
    origin.strip()
    for origin in _allowed_origins_env.split(",")
    if origin.strip()
] or _default_cors_origins

# Allow browser-based clients (e.g., Vite dev server) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    """Simple liveness probe for container orchestrators."""
    return {"status": "ok"}


@app.post("/uploads")
def create_upload(payload: UploadPayload) -> Dict[str, str]:
    """Accept PV or cycle tables as JSON and cache them for reuse."""

    if payload.kind == "pv":
        upload_id = uploads.store_pv(_pv_rows_to_df(payload.pv_rows or []), payload.name)
    else:
        upload_id = uploads.store_cycle(_cycle_rows_to_df(payload.cycle_rows or []), payload.name)
    return {"upload_id": upload_id}


@app.post("/simulate")
def simulate(request: SimulationRequest) -> Dict[str, Any]:
    """Run a single simulation and return the full output plus KPI summary."""

    pv_df, cycle_df, data_warnings = _resolve_inputs(request.data, uploads)
    cfg, cfg_warnings = _validate_and_hydrate_config(request.config, pv_df)

    sim_output = simulate_project(cfg, pv_df, cycle_df, request.dod_override, request.include_hourly_logs)
    summary = summarize_simulation(sim_output)

    return {
        "warnings": data_warnings + cfg_warnings,
        "summary": _serialize_summary(summary),
        "output": _serialize_simulation_output(sim_output),
    }


@app.post("/sweep")
def sweep(request: SweepRequest) -> Dict[str, Any]:
    """Run a BESS sizing sweep using shared simulation hooks."""

    pv_df, cycle_df, data_warnings = _resolve_inputs(request.data, uploads)
    cfg, cfg_warnings = _validate_and_hydrate_config(request.config, pv_df)
    results_df = sweep_bess_sizes(
        cfg,
        pv_df,
        cycle_df,
        request.dod_override,
        power_mw_values=request.power_values,
        duration_h_values=request.duration_values,
        energy_mwh_values=request.energy_values,
        fixed_power_mw=request.fixed_power_mw,
        min_soh=request.min_soh,
        min_compliance_pct=request.min_compliance_pct,
        max_shortfall_mwh=request.max_shortfall_mwh,
        max_candidates=request.max_candidates,
        on_exceed=request.on_exceed,
        batch_size=request.batch_size,
    )
    return {
        "warnings": data_warnings + cfg_warnings,
        "rows": results_df.to_dict(orient="records"),
    }


@app.post("/batch")
def batch(request: BatchRequest) -> Dict[str, Any]:
    """Run multiple configurations against the same dataset."""

    pv_df, cycle_df, data_warnings = _resolve_inputs(request.data, uploads)
    responses: List[Dict[str, Any]] = []
    warnings = list(data_warnings)

    for run in request.runs:
        cfg, cfg_warnings = _validate_and_hydrate_config(run.config, pv_df)
        warnings.extend(cfg_warnings)
        sim_output = simulate_project(cfg, pv_df, cycle_df, request.dod_override, False)
        summary = summarize_simulation(sim_output)
        responses.append(
            {
                "name": run.name or "scenario",
                "summary": _serialize_summary(summary),
                "output": _serialize_simulation_output(sim_output),
            }
        )

    return {"warnings": warnings, "runs": responses}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=False)
