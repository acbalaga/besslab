import { useMemo, useState } from "react";
import { Button } from "../components/ui/Button";
import { Card } from "../components/ui/Card";
import { Field } from "../components/ui/Field";
import { Notice } from "../components/ui/Notice";
import { DataTable } from "../components/DataTable";
import { useUploads } from "../state/UploadContext";
import { runSweep } from "../api/client";
import { SweepRequest } from "../api/types";

const baseConfig = {
  contracted_mw: 30,
  initial_power_mw: 30,
  initial_usable_mwh: 120,
  discharge_windows_text: "10:00-14:00,18:00-22:00",
};

function parseList(input: string): number[] {
  return input
    .split(",")
    .map((token) => Number(token.trim()))
    .filter((value) => Number.isFinite(value));
}

export function SweepPage() {
  const { uploads } = useUploads();
  const [powerValues, setPowerValues] = useState("20,30,40");
  const [durationValues, setDurationValues] = useState("2,4");
  const [minCompliance, setMinCompliance] = useState("0.9");
  const [response, setResponse] = useState<Record<string, any>[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>("");

  const handleSweep = async () => {
    setLoading(true);
    setError("");
    try {
      const payload: SweepRequest = {
        config: baseConfig,
        data: {
          pv_upload_id: uploads.pvId,
          cycle_upload_id: uploads.cycleId,
          use_sample_pv: !uploads.pvId,
          use_sample_cycle: !uploads.cycleId,
        },
        power_values: parseList(powerValues),
        duration_values: parseList(durationValues),
        min_compliance_pct: Number(minCompliance) || undefined,
      };
      const resp = await runSweep(payload);
      setResponse(resp.rows || []);
      setWarnings(resp.warnings || []);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const rows = useMemo(() => response.slice(0, 200), [response]);

  return (
    <div className="stack">
      <div className="section-title">
        <div className="badge">BESS sizing sweep</div>
        <h2>Scan multiple durations and power levels at once</h2>
      </div>
      <Card title="Sweep inputs" description="Provide comma-separated lists; cached uploads seed every run.">
        <div className="grid grid-2">
          <Field
            label="Power values (MW)"
            value={powerValues}
            onChange={(e) => setPowerValues(e.target.value)}
            helperText="Example: 20,30,40"
          />
          <Field
            label="Duration values (h)"
            value={durationValues}
            onChange={(e) => setDurationValues(e.target.value)}
            helperText="Example: 2,4"
          />
          <Field
            label="Min compliance (0-1)"
            value={minCompliance}
            onChange={(e) => setMinCompliance(e.target.value)}
            helperText="Optional filter"
          />
        </div>
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center", marginTop: "1rem" }}>
          <Button onClick={() => void handleSweep()} disabled={loading}>
            {loading ? "Sweeping..." : "Run sweep"}
          </Button>
          {warnings.length > 0 && <Notice message={warnings.join(" | ")} />}
          {error && <Notice tone="error" message={error} />}
        </div>
      </Card>

      <DataTable
        title="Sweep results"
        description="Compliance, feasibility, and economics markers"
        columns={[
          { key: "power_mw", header: "Power (MW)" },
          { key: "duration_h", header: "Duration (h)" },
          { key: "energy_mwh", header: "Energy (MWh)" },
          { key: "compliance_pct", header: "Compliance", render: (row) => `${(row.compliance_pct * 100).toFixed(1)}%` },
          { key: "total_shortfall_mwh", header: "Shortfall" },
          { key: "feasible", header: "Feasible?", render: (row) => (row.feasible ? "Yes" : "No") },
          { key: "irr_pct", header: "IRR %", render: (row) => (Number.isFinite(row.irr_pct) ? (row.irr_pct * 100).toFixed(1) : "–") },
          { key: "lcoe_usd_per_mwh", header: "LCOE ($/MWh)", render: (row) => (Number.isFinite(row.lcoe_usd_per_mwh) ? row.lcoe_usd_per_mwh.toFixed(1) : "–") },
          { key: "status", header: "Status" },
        ]}
        rows={rows}
        emptyMessage="Run a sweep to populate results."
      />
    </div>
  );
}
