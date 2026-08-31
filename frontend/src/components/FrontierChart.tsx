import { useQuery } from '@tanstack/react-query';
import {
  CartesianGrid,
  LabelList,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts';
import { fetchRunSummary } from '../api/client';
import { QueryState } from './QueryState';
import { toFrontierPoints, type FrontierPoint } from './frontier';

export function FrontierChart({ runId }: { runId: number }) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['run-summary', runId],
    queryFn: () => fetchRunSummary(runId),
  });

  const points: FrontierPoint[] = toFrontierPoints(data ?? []);

  return (
    <QueryState isLoading={isLoading} error={error} onRetry={refetch}>
      <ResponsiveContainer width="100%" height={400}>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" dataKey="cost" name="Mean cost ($)" />
          <YAxis type="number" dataKey="quality" name="Mean quality" />
          <ZAxis type="number" dataKey="latency" range={[100, 1000]} name="Mean latency (ms)" />
          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            content={({ payload }) => {
              if (!payload || payload.length === 0) return null;
              const point = payload[0].payload as FrontierPoint;
              return (
                <div className="rounded border bg-white p-2 text-xs shadow">
                  <p className="font-medium">{point.arm_name}</p>
                  <p>{point.noQuality ? 'Quality: not yet judged' : `Quality: ${point.quality.toFixed(2)}`}</p>
                  <p>Cost: {point.noCost ? 'no per-token cost — local compute' : `$${point.cost.toFixed(4)}`}</p>
                  <p>Latency: {point.latency.toFixed(0)} ms</p>
                </div>
              );
            }}
          />
          <Scatter data={points} fill="#2563eb">
            <LabelList dataKey="arm_name" position="top" />
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </QueryState>
  );
}
