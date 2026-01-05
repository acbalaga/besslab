import { useMemo, useState } from "react";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Field } from "../components/ui/Field";
import { Notice } from "../components/ui/Notice";
import { Tabs } from "../components/ui/Tabs";
import { KpiCard } from "../components/KpiCard";
import { TimeSeriesChart } from "../components/charts/TimeSeriesChart";
import { BarSummaryChart } from "../components/charts/BarSummaryChart";
import { DataTable } from "../components/DataTable";
import { useUploads } from "../state/UploadContext";
import { runSimulation } from "../api/client";
import { HourlyLog, MonthResult, SimulationRequest, SimulationResponse, YearResult } from "../api/types";
import { DownloadLink } from "../components/DownloadLinks";

const defaultConfig = {
  years: 20,
  contracted_mw: 30,
  initial_power_mw: 30,
  initial_usable_mwh: 120,
  soc_floor: 0.1,
  soc_ceiling: 0.9,
  rte_roundtrip: 0.88,
  pv_availability: 0.98,
  bess_availability: 0.99,
  discharge_windows_text: "10:00-14:00,18:00-22:00",
  charge_windows_text: "",
};

function formatPercent(value?: number): string {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "–";
  }
  return `${(value * 100).toFixed(1)}%`;
}

function formatNumber(value?: number, digits = 1): string {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return "–";
  }
  return value.toFixed(digits);
}

function buildLogRows(log?: HourlyLog | null) {
  if (!log) return [];
  return log.hod.map((hod, idx) => ({
    hod,
    pv_mw: log.pv_mw[idx],
    pv_to_contract_mw: log.pv_to_contract_mw[idx],
    bess_to_contract_mw: log.bess_to_contract_mw[idx],
    charge_mw: log.charge_mw[idx],
    discharge_mw: log.discharge_mw[idx],
    soc_mwh: log.soc_mwh[idx],
  }));
}

export function InputsResultsPage() {
  const { uploads } = useUploads();
  const [config, setConfig] = useState(defaultConfig);
  const [response, setResponse] = useState<SimulationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");

  const handleChange = (key: string, value: string) => {
    const numericValue = value === "" ? 0 : Number(value);
    setConfig((prev) => ({ ...prev, [key]: numericValue }));
  };

  const handleSimulate = async () => {
    setLoading(true);
    setError("");
    try {
      const payload: SimulationRequest = {
        config: {
          ...config,
        },
        data: {
          pv_upload_id: uploads.pvId,
          cycle_upload_id: uploads.cycleId,
          use_sample_pv: !uploads.pvId,
          use_sample_cycle: !uploads.cycleId,
        },
        include_hourly_logs: true,
      };

      const resp = await runSimulation(payload);
      setResponse(resp);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const summary = response?.summary;
  const yearly = response?.output.results ?? [];
  const monthly = response?.output.monthly_results ?? [];

  const firstYearLog = useMemo(() => buildLogRows(response?.output.first_year_logs), [response]);
  const finalYearLog = useMemo(() => buildLogRows(response?.output.final_year_logs), [response]);

  return (
    <div className="stack">
      <div className="section-title">
        <div className="badge">Inputs & Results</div>
        <h2>Run a single simulation, review KPIs, and export data</h2>
      </div>
      <Card title="Simulation inputs" description="Baseline assumptions. All values numeric.">
        <div className="grid grid-2">
          <Field
            label="Contracted MW"
            type="number"
            value={config.contracted_mw}
            onChange={(e) => handleChange("contracted_mw", e.target.value)}
          />
          <Field
            label="Usable energy (MWh)"
            type="number"
            value={config.initial_usable_mwh}
            onChange={(e) => handleChange("initial_usable_mwh", e.target.value)}
          />
          <Field
            label="Power rating (MW)"
            type="number"
            value={config.initial_power_mw}
            onChange={(e) => handleChange("initial_power_mw", e.target.value)}
          />
          <Field
            label="Round-trip efficiency"
            type="number"
            step="0.01"
            value={config.rte_roundtrip}
            onChange={(e) => handleChange("rte_roundtrip", e.target.value)}
          />
          <Field
            label="SOC floor"
            type="number"
            step="0.01"
            value={config.soc_floor}
            onChange={(e) => handleChange("soc_floor", e.target.value)}
          />
          <Field
            label="SOC ceiling"
            type="number"
            step="0.01"
            value={config.soc_ceiling}
            onChange={(e) => handleChange("soc_ceiling", e.target.value)}
          />
          <Field
            label="PV availability"
            type="number"
            step="0.01"
            value={config.pv_availability}
            onChange={(e) => handleChange("pv_availability", e.target.value)}
          />
          <Field
            label="BESS availability"
            type="number"
            step="0.01"
            value={config.bess_availability}
            onChange={(e) => handleChange("bess_availability", e.target.value)}
          />
          <Field
            label="Discharge windows"
            value={config.discharge_windows_text}
            onChange={(e) => setConfig((prev) => ({ ...prev, discharge_windows_text: e.target.value }))}
            helperText="Comma-separated HH:MM-HH:MM"
          />
          <Field
            label="Charge windows (optional)"
            value={config.charge_windows_text}
            onChange={(e) => setConfig((prev) => ({ ...prev, charge_windows_text: e.target.value }))}
            helperText="Leave blank to allow anytime charging"
          />
        </div>
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", marginTop: "1rem" }}>
          <Button onClick={() => void handleSimulate()} disabled={loading}>
            {loading ? "Running..." : "Run simulation"}
          </Button>
          {error && <Notice tone="error" message={error} />}
          {response?.warnings?.length ? <Notice message={response.warnings.join(" | ")} /> : null}
        </div>
      </Card>

      {summary && (
        <>
          <div className="section-title">
            <div className="badge">KPIs</div>
            <h2>Compliance, coverage, and economics snapshot</h2>
          </div>
          <div className="card-grid">
            <KpiCard label="Compliance" value={formatPercent(summary.compliance)} tone={summary.compliance >= 0.98 ? "good" : "warn"} helper="Share of firm delivered" />
            <KpiCard label="Coverage" value={formatPercent(summary.bess_share_of_firm)} helper="BESS share of firm energy" />
            <KpiCard label="Charge/Discharge ratio" value={formatNumber(summary.charge_discharge_ratio, 2)} />
            <KpiCard label="PV capture" value={formatPercent(summary.pv_capture_ratio)} />
            <KpiCard label="Discharge CF" value={formatPercent(summary.discharge_capacity_factor)} helper="Capacity factor on discharge window" />
            <KpiCard label="Losses" value={`${formatNumber(summary.bess_losses_mwh)} MWh`} helper="Energy lost to conversion" />
            <KpiCard label="Cycles / yr" value={formatNumber(summary.avg_eq_cycles_per_year, 2)} helper="Average equivalent cycles" />
            <KpiCard label="Final capability" value={formatPercent(summary.cap_ratio_final)} helper="End-of-life capability ratio" tone={summary.cap_ratio_final > 0.8 ? "good" : "warn"} />
          </div>

          <div className="section-title">
            <div className="badge">Charts</div>
            <h2>Dispatch traces and resource splits</h2>
          </div>
          <div className="grid grid-2">
            <TimeSeriesChart
              title="First year dispatch"
              description="Minute/hourly traces pulled from /simulate"
              data={firstYearLog}
              xKey="hod"
              series={[
                { key: "pv_mw", label: "PV MW", color: "#38bdf8" },
                { key: "pv_to_contract_mw", label: "PV to contract", color: "#10b981" },
                { key: "bess_to_contract_mw", label: "BESS to contract", color: "#4f46e5" },
                { key: "soc_mwh", label: "SOC (MWh)", color: "#f59e0b" },
              ]}
            />
            <TimeSeriesChart
              title="Final year dispatch"
              description="Captured from final-year hourly log"
              data={finalYearLog}
              xKey="hod"
              series={[
                { key: "pv_mw", label: "PV MW", color: "#38bdf8" },
                { key: "bess_to_contract_mw", label: "BESS to contract", color: "#4f46e5" },
                { key: "soc_mwh", label: "SOC (MWh)", color: "#f59e0b" },
              ]}
            />
          </div>

          <BarSummaryChart
            title="Energy balance"
            description="Total project generation split by resource"
            data={[{
              label: "Project totals",
              total_project_generation_mwh: summary.total_project_generation_mwh,
              bess_generation_mwh: summary.bess_generation_mwh,
              pv_generation_mwh: summary.pv_generation_mwh,
              pv_excess_mwh: summary.pv_excess_mwh,
            }]}
            xKey="label"
            series={[
              { key: "pv_generation_mwh", label: "PV to contract", color: "#10b981" },
              { key: "bess_generation_mwh", label: "BESS to contract", color: "#4f46e5" },
              { key: "pv_excess_mwh", label: "PV excess", color: "#f59e0b" },
            ]}
          />

          <Tabs
            tabs={[
              {
                id: "annual",
                label: "Annual KPIs",
                content: (
                  <DataTable<YearResult>
                    title="Annual results"
                    description="Compliance, RTE, SOH by year"
                    columns={[
                      { key: "year_index", header: "Year" },
                      { key: "delivered_firm_mwh", header: "Delivered (MWh)" },
                      { key: "shortfall_mwh", header: "Shortfall (MWh)" },
                      { key: "avg_rte", header: "Avg RTE", render: (row) => formatPercent(row.avg_rte) },
                      { key: "eq_cycles", header: "Eq cycles" },
                      { key: "soh_total", header: "SOH", render: (row) => formatPercent(row.soh_total) },
                      { key: "eoy_usable_mwh", header: "EOY usable" },
                      { key: "pv_curtailed_mwh", header: "PV curtailed" },
                    ]}
                    rows={yearly}
                  />
                ),
              },
              {
                id: "monthly",
                label: "Monthly detail",
                content: (
                  <DataTable<MonthResult>
                    title="Monthly breakdown"
                    description="Seasonality, augmentation events"
                    columns={[
                      { key: "month_label", header: "Month" },
                      { key: "delivered_firm_mwh", header: "Delivered (MWh)" },
                      { key: "shortfall_mwh", header: "Shortfall" },
                      { key: "avg_rte", header: "Avg RTE", render: (row) => formatPercent(row.avg_rte) },
                      { key: "eq_cycles", header: "Eq cycles" },
                      { key: "soh_total", header: "SOH", render: (row) => formatPercent(row.soh_total) },
                      { key: "eom_usable_mwh", header: "EOM usable" },
                      { key: "pv_curtailed_mwh", header: "PV curtailed" },
                    ]}
                    rows={monthly}
                  />
                ),
              },
            ]}
          />

          <Card title="Downloads" description="Export raw JSON for offline review">
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
              <DownloadLink label="Summary JSON" data={summary} fileName="summary.json" />
              <DownloadLink label="Annual results" data={yearly} fileName="annual_results.json" />
              <DownloadLink label="Monthly results" data={monthly} fileName="monthly_results.json" />
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
