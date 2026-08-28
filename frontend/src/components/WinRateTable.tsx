import { useQuery } from '@tanstack/react-query';
import { fetchCompare, fetchRunSummary } from '../api/client';
import { QueryState } from './QueryState';

export function WinRateTable({ runId }: { runId: number }) {
  const summaryQuery = useQuery({
    queryKey: ['run-summary', runId],
    queryFn: () => fetchRunSummary(runId),
  });
  const compareQuery = useQuery({
    queryKey: ['run-compare', runId, 'judge_score'],
    queryFn: () => fetchCompare(runId, 'judge_score'),
  });

  const isLoading = summaryQuery.isLoading || compareQuery.isLoading;
  const error = summaryQuery.error ?? compareQuery.error;
  const retry = () => {
    summaryQuery.refetch();
    compareQuery.refetch();
  };

  return (
    <QueryState isLoading={isLoading} error={error} onRetry={retry}>
      {summaryQuery.data && (
        <table className="mb-6 w-full text-left text-sm">
          <thead>
            <tr className="border-b text-xs uppercase text-gray-500">
              <th className="py-2">Arm</th>
              <th>n</th>
              <th>Mean quality</th>
              <th>Mean latency (ms)</th>
              <th>Mean cost ($)</th>
            </tr>
          </thead>
          <tbody>
            {summaryQuery.data.map((row) => (
              <tr key={row.arm_name} className="border-b">
                <td className="py-2">{row.arm_name}</td>
                <td>{row.n}</td>
                <td>{row.mean_judge_score?.toFixed(2) ?? '—'}</td>
                <td>{row.mean_latency_ms?.toFixed(0) ?? '—'}</td>
                <td>{row.mean_cost_estimate_usd?.toFixed(4) ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {compareQuery.data && (
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b text-xs uppercase text-gray-500">
              <th className="py-2">Arm A</th>
              <th>Arm B</th>
              <th>Mean diff</th>
              <th>95% CI</th>
              <th>p (corrected)</th>
              <th>Significant</th>
            </tr>
          </thead>
          <tbody>
            {compareQuery.data.map((row) => (
              <tr key={`${row.arm_a}-${row.arm_b}`} className="border-b">
                <td className="py-2">{row.arm_a}</td>
                <td>{row.arm_b}</td>
                <td>{row.mean_diff.toFixed(3)}</td>
                <td>
                  [{row.ci_lower.toFixed(3)}, {row.ci_upper.toFixed(3)}]
                </td>
                <td>{row.p_value_corrected?.toFixed(4) ?? '—'}</td>
                <td>{(row.p_value_corrected ?? 1) < 0.05 ? 'Yes' : 'No'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </QueryState>
  );
}
