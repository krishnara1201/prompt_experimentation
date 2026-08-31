import { describe, expect, it } from 'vitest';
import { toFrontierPoints } from '../frontier';
import type { ArmSummaryResponse } from '../../api/types';

function summaryRow(overrides: Partial<ArmSummaryResponse>): ArmSummaryResponse {
  return {
    arm_name: 'arm',
    n: 10,
    mean_judge_score: 4,
    mean_latency_ms: 1000,
    mean_cost_estimate_usd: 0.001,
    mean_prompt_tokens: 50,
    mean_completion_tokens: 20,
    ...overrides,
  };
}

describe('toFrontierPoints', () => {
  it('maps a summary row to a plotted point', () => {
    const [point] = toFrontierPoints([
      summaryRow({ arm_name: 'gpt-4o-mini', mean_judge_score: 4.2, mean_latency_ms: 900, mean_cost_estimate_usd: 0.0003 }),
    ]);
    expect(point).toEqual({
      arm_name: 'gpt-4o-mini',
      quality: 4.2,
      latency: 900,
      cost: 0.0003,
      noCost: false,
      noQuality: false,
    });
  });

  it('flags a local arm with null cost and coerces it to 0', () => {
    const [point] = toFrontierPoints([summaryRow({ mean_cost_estimate_usd: null })]);
    expect(point.cost).toBe(0);
    expect(point.noCost).toBe(true);
  });

  it('flags an unjudged arm with null quality and coerces it to 0', () => {
    const [point] = toFrontierPoints([summaryRow({ mean_judge_score: null })]);
    expect(point.quality).toBe(0);
    expect(point.noQuality).toBe(true);
  });

  it('returns an empty array for no rows', () => {
    expect(toFrontierPoints([])).toEqual([]);
  });
});
