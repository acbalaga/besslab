import { BatchRequest, BatchResponse, SimulationRequest, SimulationResponse, SweepRequest, SweepResponse, UploadResponse } from "./types";

// Prefer an explicit env override, otherwise fall back to the current origin so that
// co-hosted deployments (e.g., reverse-proxied API + frontend) avoid cross-origin fetches.
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  (typeof window !== "undefined" ? window.location.origin : "") ||
  "http://127.0.0.1:8000";

async function request<T>(path: string, options: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${path}`;

  try {
    const response = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });

    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`Request failed (${response.status}) for ${url}: ${detail}`);
    }

    return (await response.json()) as T;
  } catch (error) {
    // Surface the endpoint to make browser console debugging easier for connectivity/CORS issues.
    const reason = error instanceof Error ? error.message : String(error);
    throw new Error(`Failed to fetch ${url}: ${reason}`);
  }
}

export async function createUpload(payload: any): Promise<UploadResponse> {
  return request<UploadResponse>("/uploads", { method: "POST", body: JSON.stringify(payload) });
}

export async function runSimulation(payload: SimulationRequest): Promise<SimulationResponse> {
  return request<SimulationResponse>("/simulate", { method: "POST", body: JSON.stringify(payload) });
}

export async function runSweep(payload: SweepRequest): Promise<SweepResponse> {
  return request<SweepResponse>("/sweep", { method: "POST", body: JSON.stringify(payload) });
}

export async function runBatch(payload: BatchRequest): Promise<BatchResponse> {
  return request<BatchResponse>("/batch", { method: "POST", body: JSON.stringify(payload) });
}

export { API_BASE_URL };
