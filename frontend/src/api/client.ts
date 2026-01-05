import { BatchRequest, BatchResponse, SimulationRequest, SimulationResponse, SweepRequest, SweepResponse, UploadResponse } from "./types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function request<T>(path: string, options: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`Request failed (${response.status}): ${detail}`);
  }

  return (await response.json()) as T;
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
