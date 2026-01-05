import { useState } from "react";
import Papa from "papaparse";
import * as XLSX from "xlsx";
import { Card } from "../components/ui/Card";
import { Button } from "../components/ui/Button";
import { Notice } from "../components/ui/Notice";
import { useUploads } from "../state/UploadContext";
import { createUpload } from "../api/client";
import { PvRow } from "../api/types";

async function parseCsv(file: File): Promise<Record<string, any>[]> {
  return new Promise((resolve, reject) => {
    Papa.parse<Record<string, any>>(file, {
      header: true,
      skipEmptyLines: true,
      complete: (result) => resolve(result.data),
      error: (err) => reject(err),
    });
  });
}

async function parseSheet(file: File): Promise<Record<string, any>[]> {
  const buffer = await file.arrayBuffer();
  const workbook = XLSX.read(buffer, { type: "array" });
  const sheet = workbook.Sheets[workbook.SheetNames[0]];
  return XLSX.utils.sheet_to_json<Record<string, any>>(sheet, { defval: null });
}

function normalizePvRows(rows: Record<string, any>[]): PvRow[] {
  return rows
    .map((row, idx) => {
      const pvValue = Number(row.pv_mw ?? row.pv ?? row["pv MW"] ?? row["PV_MW"]);
      const hourIndexRaw = row.hour_index ?? row.hour ?? row["Hour Index"] ?? row["hour"];
      const hourIndex = hourIndexRaw !== undefined && hourIndexRaw !== null ? Number(hourIndexRaw) : idx;
      const timestamp = row.timestamp ? String(row.timestamp) : undefined;
      if (Number.isFinite(pvValue)) {
        return { pv_mw: pvValue, hour_index: Number.isFinite(hourIndex) ? hourIndex : idx, timestamp };
      }
      return undefined;
    })
    .filter(Boolean) as PvRow[];
}

function asRowsFromFile(file: File): Promise<Record<string, any>[]> {
  const lower = file.name.toLowerCase();
  if (lower.endsWith(".xlsx") || lower.endsWith(".xls")) {
    return parseSheet(file);
  }
  return parseCsv(file);
}

export function LandingPage() {
  const { uploads, update, clear } = useUploads();
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);

  const handlePvUpload = async (file: File) => {
    setLoading(true);
    setError("");
    try {
      const rows = normalizePvRows(await asRowsFromFile(file));
      if (!rows.length) {
        throw new Error("No usable PV rows found. Ensure a 'pv_mw' column is present.");
      }
      const response = await createUpload({ kind: "pv", pv_rows: rows, name: file.name });
      update({ pvId: response.upload_id });
      setStatus(`Uploaded PV profile (${rows.length} rows) and cached as ${response.upload_id}.`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const handleCycleUpload = async (file: File) => {
    setLoading(true);
    setError("");
    try {
      const rows = await asRowsFromFile(file);
      if (!rows.length) {
        throw new Error("Cycle table is empty; include at least one row.");
      }
      const response = await createUpload({ kind: "cycle", cycle_rows: rows, name: file.name });
      update({ cycleId: response.upload_id });
      setStatus(`Uploaded cycle model (${rows.length} rows) and cached as ${response.upload_id}.`);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="stack">
      <div className="section-title">
        <div className="badge">Uploads</div>
        <h2>Warm the cache and reuse inputs across flows</h2>
      </div>
      <Card
        title="Upload PV profile"
        description="Accepts CSV or Excel. Must include a pv_mw column; hour_index or timestamp is optional."
      >
        <div style={{ display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap" }}>
          <input
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) {
                void handlePvUpload(file);
              }
            }}
          />
          {uploads.pvId && <Notice tone="success" message={`Using cached PV upload: ${uploads.pvId}`} />}
        </div>
      </Card>

      <Card
        title="Upload cycle model (optional)"
        description="Upload your degradation table as CSV/Excel or fall back to the sample model."
      >
        <div style={{ display: "flex", gap: "1rem", alignItems: "center", flexWrap: "wrap" }}>
          <input
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) {
                void handleCycleUpload(file);
              }
            }}
          />
          {uploads.cycleId && <Notice tone="success" message={`Using cached cycle upload: ${uploads.cycleId}`} />}
        </div>
      </Card>

      <Card title="Cache status" description="Tokens are stored locally and reused across all pages.">
        <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
          <Button variant="secondary" onClick={() => clear()}>
            Clear cached tokens
          </Button>
          {status && <Notice tone="success" message={status} />}
          {error && <Notice tone="error" message={error} />}
          {loading && <Notice message="Uploading..." />}
        </div>
      </Card>
    </div>
  );
}
