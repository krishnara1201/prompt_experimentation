import type { ArmSummaryResponse, CalibrationResponse, PairedComparisonResponse, RunSummary } from './types';

const BASE = '/api';

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function fetchRuns(): Promise<RunSummary[]> {
  return getJson<RunSummary[]>('/runs');
}

export function fetchRunSummary(runId: number): Promise<ArmSummaryResponse[]> {
  return getJson<ArmSummaryResponse[]>(`/runs/${runId}/summary`);
}

export function fetchCompare(runId: number, metric: string): Promise<PairedComparisonResponse[]> {
  return getJson<PairedComparisonResponse[]>(`/runs/${runId}/compare?metric=${metric}`);
}

export function fetchCalibration(runId: number): Promise<CalibrationResponse> {
  return getJson<CalibrationResponse>(`/runs/${runId}/calibration`);
}
