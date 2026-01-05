import { useState } from "react";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Field } from "../components/ui/Field";
import { Notice } from "../components/ui/Notice";
import { useUploads } from "../state/UploadContext";
import { runBatch } from "../api/client";
import { BatchRequest, BatchRunResponse } from "../api/types";
import { KpiCard } from "../components/KpiCard";
import { DataTable } from "../components/DataTable";

interface ScenarioInput {
  name: string;
  config: Record<string, any>;
}

const baseConfig = {
  contracted_mw: 30,
  initial_power_mw: 30,
  initial_usable_mwh: 120,
  discharge_windows_text: "10:00-14:00,18:00-22:00",
};

export function BatchPage() {
  const { uploads } = useUploads();
  const [scenarios, setScenarios] = useState<ScenarioInput[]>([
    { name: "Base case", config: { ...baseConfig } },
    { name: "High availability", config: { ...baseConfig, bess_availability: 0.995 } },
  ]);
  const [results, setResults] = useState<BatchRunResponse[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");

  const updateScenario = (index: number, field: string, value: string) => {
    setScenarios((prev) => {
      const next = [...prev];
      const numericValue = value === "" ? undefined : Number(value);
      next[index] = {
        ...next[index],
        config: { ...next[index].config, [field]: numericValue },
      };
      return next;
    });
  };

  const handleRunBatch = async () => {
    setLoading(true);
    setError("");
    try {
      const payload: BatchRequest = {
        data: {
          pv_upload_id: uploads.pvId,
          cycle_upload_id: uploads.cycleId,
          use_sample_pv: !uploads.pvId,
          use_sample_cycle: !uploads.cycleId,
        },
        runs: scenarios.map((scenario) => ({ name: scenario.name, config: scenario.config })),
      };
      const resp = await runBatch(payload);
      setResults(resp.runs || []);
      setWarnings(resp.warnings || []);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="stack">
      <div className="section-title">
        <div className="badge">Batch</div>
        <h2>Run multiple scenarios with shared uploads</h2>
      </div>
      <Card title="Scenario inputs" description="Adjust power, energy, and availability per run.">
        <div className="stack">
          {scenarios.map((scenario, idx) => (
            <div key={scenario.name + idx} style={{ border: "1px solid #e2e8f0", borderRadius: "12px", padding: "0.75rem" }}>
              <div style={{ display: "flex", gap: "1rem", alignItems: "center", marginBottom: "0.75rem" }}>
                <strong>{scenario.name}</strong>
                <Field
                  label="Power MW"
                  type="number"
                  value={scenario.config.initial_power_mw ?? ""}
                  onChange={(e) => updateScenario(idx, "initial_power_mw", e.target.value)}
                />
                <Field
                  label="Usable MWh"
                  type="number"
                  value={scenario.config.initial_usable_mwh ?? ""}
                  onChange={(e) => updateScenario(idx, "initial_usable_mwh", e.target.value)}
                />
                <Field
                  label="Availability"
                  type="number"
                  step="0.001"
                  value={scenario.config.bess_availability ?? ""}
                  onChange={(e) => updateScenario(idx, "bess_availability", e.target.value)}
                />
              </div>
            </div>
          ))}
          <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
            <Button
              variant="secondary"
              onClick={() => setScenarios((prev) => [...prev, { name: `Scenario ${prev.length + 1}`, config: { ...baseConfig } }])}
            >
              Add scenario
            </Button>
            <Button onClick={() => void handleRunBatch()} disabled={loading}>
              {loading ? "Running..." : "Run batch"}
            </Button>
            {warnings.length > 0 && <Notice message={warnings.join(" | ")} />}
            {error && <Notice tone="error" message={error} />}
          </div>
        </div>
      </Card>

      {results.length > 0 && (
        <div className="stack">
          <div className="grid grid-3">
            {results.map((run) => (
              <KpiCard
                key={run.name}
                label={run.name || "Scenario"}
                value={`${(run.summary.compliance * 100).toFixed(1)}% compliance`}
                helper={`SOH final: ${(run.summary.cap_ratio_final * 100).toFixed(1)}%`}
                tone={run.summary.compliance >= 0.98 ? "good" : "warn"}
              />
            ))}
          </div>
          <DataTable
            title="Batch runs"
            description="One row per scenario"
            columns={[
              { key: "name", header: "Scenario" },
              { key: "compliance", header: "Compliance", render: (row) => `${(row.summary.compliance * 100).toFixed(1)}%` },
              { key: "bess_share_of_firm", header: "Coverage", render: (row) => `${(row.summary.bess_share_of_firm * 100).toFixed(1)}%` },
              { key: "avg_eq_cycles_per_year", header: "Cycles / yr", render: (row) => row.summary.avg_eq_cycles_per_year.toFixed(2) },
              { key: "cap_ratio_final", header: "Final cap", render: (row) => `${(row.summary.cap_ratio_final * 100).toFixed(1)}%` },
              { key: "total_shortfall_mwh", header: "Shortfall", render: (row) => row.summary.total_shortfall_mwh.toFixed(1) },
            ]}
            rows={results as any}
          />
        </div>
      )}
    </div>
  );
}
