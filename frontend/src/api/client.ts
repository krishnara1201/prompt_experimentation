import type {
  ArmInfo,
  ArmSummaryResponse,
  CalibrationResponse,
  EquivalenceResponse,
  PairedComparisonResponse,
  PowerResponse,
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

export function fetchEquivalence(
  runId: number,
  params: { armLocal: string; armApi: string; epsilon: number },
): Promise<EquivalenceResponse> {
  const query = new URLSearchParams({
    metric: 'judge_score',
    arm_local: params.armLocal,
    arm_api: params.armApi,
    epsilon: String(params.epsilon),
  });
  return getJson<EquivalenceResponse>(`/runs/${runId}/equivalence?${query}`);
}

export function fetchPower(
  runId: number,
  params: { armA: string; armB: string; power: number; alpha: number; effectSize?: number },
): Promise<PowerResponse> {
  const query = new URLSearchParams({
    metric: 'judge_score',
    arm_a: params.armA,
    arm_b: params.armB,
    power: String(params.power),
    alpha: String(params.alpha),
  });
  if (params.effectSize !== undefined) query.set('effect_size', String(params.effectSize));
  return getJson<PowerResponse>(`/runs/${runId}/power?${query}`);
}
