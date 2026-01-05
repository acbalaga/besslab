export interface WindowPayload {
  start: number;
  end: number;
}

export interface AugmentationSchedulePayload {
  year: number;
  basis: string;
  value: number;
}

export interface SimConfigPayload {
  years?: number;
  step_hours?: number;
  pv_deg_rate?: number;
  pv_availability?: number;
  bess_availability?: number;
  rte_roundtrip?: number;
  use_split_rte?: boolean;
  charge_efficiency?: number | null;
  discharge_efficiency?: number | null;
  soc_floor?: number;
  soc_ceiling?: number;
  initial_power_mw?: number;
  initial_usable_mwh?: number;
  contracted_mw?: number;
  discharge_windows_text?: string | null;
  discharge_windows?: WindowPayload[] | null;
  charge_windows_text?: string;
  charge_windows?: WindowPayload[] | null;
  max_cycles_per_day_cap?: number;
  calendar_fade_rate?: number;
  use_calendar_exp_model?: boolean;
  augmentation?: string;
  aug_trigger_type?: string;
  aug_threshold_margin?: number;
  aug_topup_margin?: number;
  aug_soh_trigger_pct?: number;
  aug_soh_add_frac_initial?: number;
  aug_periodic_every_years?: number;
  aug_periodic_add_frac_of_bol?: number;
  aug_add_mode?: string;
  aug_fixed_energy_mwh?: number;
  aug_retire_old_cohort?: boolean;
  aug_retire_soh_pct?: number;
  augmentation_schedule?: AugmentationSchedulePayload[];
}

export interface PvRow {
  pv_mw: number;
  hour_index?: number;
  timestamp?: string;
}

export interface DataSource {
  pv_upload_id?: string;
  cycle_upload_id?: string;
  use_sample_pv?: boolean;
  use_sample_cycle?: boolean;
  pv_rows?: PvRow[];
  cycle_rows?: Record<string, any>[];
}

export interface SimulationRequest {
  config: SimConfigPayload;
  data: DataSource;
  dod_override?: string;
  include_hourly_logs?: boolean;
}

export interface SweepRequest {
  config: SimConfigPayload;
  data: DataSource;
  dod_override?: string;
  power_values?: number[];
  duration_values?: number[];
  energy_values?: number[];
  fixed_power_mw?: number | null;
  min_soh?: number;
  min_compliance_pct?: number | null;
  max_shortfall_mwh?: number | null;
  max_candidates?: number | null;
  on_exceed?: "raise" | "return" | "batch";
  batch_size?: number | null;
}

export interface BatchRun {
  name?: string;
  config: SimConfigPayload;
}

export interface BatchRequest {
  data: DataSource;
  dod_override?: string;
  runs: BatchRun[];
}

export interface SimulationSummary {
  compliance: number;
  bess_share_of_firm: number;
  charge_discharge_ratio: number;
  pv_capture_ratio: number;
  discharge_capacity_factor: number;
  total_project_generation_mwh: number;
  bess_generation_mwh: number;
  pv_generation_mwh: number;
  pv_excess_mwh: number;
  bess_losses_mwh: number;
  total_shortfall_mwh: number;
  avg_eq_cycles_per_year: number;
  cap_ratio_final: number;
}

export interface YearResult {
  year_index: number;
  expected_firm_mwh: number;
  delivered_firm_mwh: number;
  shortfall_mwh: number;
  breach_days: number;
  charge_mwh: number;
  discharge_mwh: number;
  available_pv_mwh: number;
  pv_to_contract_mwh: number;
  bess_to_contract_mwh: number;
  avg_rte: number;
  eq_cycles: number;
  cum_cycles: number;
  soh_cycle: number;
  soh_calendar: number;
  soh_total: number;
  eoy_usable_mwh: number;
  eoy_power_mw: number;
  pv_curtailed_mwh: number;
  flags: Record<string, number>;
}

export interface MonthResult extends YearResult {
  month_index: number;
  month_label: string;
  eom_usable_mwh: number;
  eom_power_mw: number;
}

export interface HourlyLog {
  hod: number[];
  pv_mw: number[];
  pv_to_contract_mw: number[];
  bess_to_contract_mw: number[];
  charge_mw: number[];
  discharge_mw: number[];
  soc_mwh: number[];
}

export interface SimulationOutput {
  config: SimConfigPayload;
  discharge_hours_per_day: number;
  results: YearResult[];
  monthly_results: MonthResult[];
  first_year_logs?: HourlyLog | null;
  final_year_logs?: HourlyLog | null;
  hod_count: number[];
  hod_sum_pv: number[];
  hod_sum_pv_resource: number[];
  hod_sum_bess: number[];
  hod_sum_charge: number[];
  augmentation_energy_added_mwh: number[];
  augmentation_retired_energy_mwh: number[];
  augmentation_events: number;
}

export interface SimulationResponse {
  warnings: string[];
  summary: SimulationSummary;
  output: SimulationOutput;
}

export interface SweepResponse {
  warnings: string[];
  rows: Record<string, any>[];
}

export interface BatchRunResponse {
  name?: string;
  summary: SimulationSummary;
  output: SimulationOutput;
}

export interface BatchResponse {
  warnings: string[];
  runs: BatchRunResponse[];
}

export interface UploadResponse {
  upload_id: string;
}
