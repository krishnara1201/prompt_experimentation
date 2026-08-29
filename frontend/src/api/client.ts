import type {
  ArmInfo,
  ArmSummaryResponse,
  CalibrationResponse,
  PairedComparisonResponse,
  RunCreateRequest,
  RunCreateResponse,
  RunSummary,
} from './types';

const BASE = '/api';

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

async function postJson<T>(path: string, payload: unknown): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function fetchRuns(): Promise<RunSummary[]> {
  return getJson<RunSummary[]>('/runs');
}

export function fetchArms(): Promise<ArmInfo[]> {
  return getJson<ArmInfo[]>('/arms');
}

export function createRun(body: RunCreateRequest): Promise<RunCreateResponse> {
  return postJson<RunCreateResponse>('/runs', body);
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
