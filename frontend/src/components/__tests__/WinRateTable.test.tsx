import { describe, expect, it, vi, beforeEach } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithQuery } from '../../test/renderWithQuery';
import { WinRateTable } from '../WinRateTable';
import * as client from '../../api/client';

vi.mock('../../api/client');

const mocked = vi.mocked(client);

beforeEach(() => {
  vi.resetAllMocks();
});

describe('WinRateTable', () => {
  it('renders one summary row per arm with formatted quality and latency', async () => {
    mocked.fetchRunSummary.mockResolvedValue([
      {
        arm_name: 'qwen3-8b-local',
        n: 120,
        mean_judge_score: 4.4321,
        mean_latency_ms: 6743.7,
        mean_cost_estimate_usd: null,
        mean_prompt_tokens: 40,
        mean_completion_tokens: 300,
      },
    ]);
    mocked.fetchCompare.mockResolvedValue([]);

    renderWithQuery(<WinRateTable runId={705} />);

    const cell = await screen.findByText('qwen3-8b-local');
    const row = cell.closest('tr')!;
    expect(row).toHaveTextContent('120');
    expect(row).toHaveTextContent('4.43'); // toFixed(2)
    expect(row).toHaveTextContent('6744'); // toFixed(0)
    expect(row).toHaveTextContent('—'); // null cost
  });

  it('formats the comparison CI and corrected p-value and marks significance', async () => {
    mocked.fetchRunSummary.mockResolvedValue([]);
    mocked.fetchCompare.mockResolvedValue([
      {
        arm_a: 'ft-qwen3-8b-local',
        arm_b: 'qwen3-8b-local',
        metric: 'judge_score',
        n_examples: 50,
        n_excluded: 0,
        mean_diff: 0.3,
        ci_lower: 0.06,
        ci_upper: 0.6,
        wilcoxon_statistic: 12,
        p_value: 0.01,
        p_value_corrected: 0.031,
      },
    ]);

    renderWithQuery(<WinRateTable runId={705} />);

    const cell = await screen.findByText('ft-qwen3-8b-local');
    const row = cell.closest('tr')!;
    expect(row).toHaveTextContent('0.300'); // mean_diff toFixed(3)
    expect(row).toHaveTextContent('[0.060, 0.600]');
    expect(row).toHaveTextContent('0.0310');
    expect(row).toHaveTextContent('Yes'); // p < 0.05
  });

  it('marks a non-significant comparison as "No"', async () => {
    mocked.fetchRunSummary.mockResolvedValue([]);
    mocked.fetchCompare.mockResolvedValue([
      {
        arm_a: 'a',
        arm_b: 'b',
        metric: 'judge_score',
        n_examples: 50,
        n_excluded: 0,
        mean_diff: 0.05,
        ci_lower: -0.1,
        ci_upper: 0.2,
        wilcoxon_statistic: 1,
        p_value: 0.4,
        p_value_corrected: 0.37,
      },
    ]);

    renderWithQuery(<WinRateTable runId={705} />);

    const cell = await screen.findByText('a');
    expect(cell.closest('tr')!).toHaveTextContent('No');
  });
});
