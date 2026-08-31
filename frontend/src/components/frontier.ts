import type { ArmSummaryResponse } from '../api/types';

export interface FrontierPoint {
  arm_name: string;
  cost: number;
  quality: number;
  latency: number;
  noCost: boolean;
  noQuality: boolean;
}

export function toFrontierPoints(rows: ArmSummaryResponse[]): FrontierPoint[] {
  return rows.map((row) => ({
    arm_name: row.arm_name,
    cost: row.mean_cost_estimate_usd ?? 0,
    quality: row.mean_judge_score ?? 0,
    latency: row.mean_latency_ms ?? 0,
    noCost: row.mean_cost_estimate_usd === null,
    noQuality: row.mean_judge_score === null,
  }));
}
