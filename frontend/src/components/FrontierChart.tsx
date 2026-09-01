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

// Points can land on top of each other (e.g. two local arms both at cost 0 with
// near-identical quality). Stagger the labels vertically by index so the text
// never overlaps, and nudge alternate labels left/right.
function ArmLabel(props: {
  x?: number;
  y?: number;
  value?: string | number;
  index?: number;
}) {
  const { x = 0, y = 0, value, index = 0 } = props;
  const above = index % 2 === 0;
  return (
    <text
      x={x + 10}
      y={y + (above ? -10 : 18)}
      textAnchor="start"
      fontSize={11}
      fill="#4b5563"
    >
      {value}
    </text>
  );
}

export function FrontierChart({ runId }: { runId: number }) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['run-summary', runId],
    queryFn: () => fetchRunSummary(runId),
  });

  const points: FrontierPoint[] = toFrontierPoints(data ?? []);

  return (
    <QueryState isLoading={isLoading} error={error} onRetry={refetch}>
      <p className="mb-2 text-xs text-gray-500">
        Bubble size ∝ mean latency. Local arms sit at cost 0 and can overlap — hover a
        bubble for exact values.
      </p>
      <ResponsiveContainer width="100%" height={420}>
        <ScatterChart margin={{ top: 24, right: 56, bottom: 48, left: 16 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="cost"
            domain={[0, (max: number) => (max > 0 ? max * 1.25 : 0.001)]}
            tickFormatter={(v: number) => `$${v.toFixed(4)}`}
            label={{ value: 'Mean cost per call ($)', position: 'insideBottom', offset: -12 }}
          />
          <YAxis
            type="number"
            dataKey="quality"
            domain={[0, 5]}
            allowDecimals
            label={{
              value: 'Mean judge quality (1–5)',
              angle: -90,
              position: 'insideLeft',
              style: { textAnchor: 'middle' },
            }}
          />
          <ZAxis type="number" dataKey="latency" range={[80, 500]} name="Mean latency (ms)" />
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
          <Scatter data={points} fill="#2563eb" fillOpacity={0.65}>
            <LabelList dataKey="arm_name" content={<ArmLabel />} />
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </QueryState>
  );
}
