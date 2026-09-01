import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link, useParams } from 'react-router-dom';
import { fetchRunStatus } from '../api/client';
import { StatusBadge } from '../components/StatusBadge';
import { WinRateTable } from '../components/WinRateTable';
import { FrontierChart } from '../components/FrontierChart';
import { CalibrationReport } from '../components/CalibrationReport';
import { EquivalencePanel } from '../components/EquivalencePanel';
import { PowerPanel } from '../components/PowerPanel';
import type { RunStatusResponse } from '../api/types';

type TabKey = 'winrate' | 'frontier' | 'equivalence' | 'power' | 'calibration';

const TABS: { key: TabKey; label: string }[] = [
  { key: 'winrate', label: 'Win-rate' },
  { key: 'frontier', label: 'Frontier' },
  { key: 'equivalence', label: 'Equivalence' },
  { key: 'power', label: 'Power' },
  { key: 'calibration', label: 'Calibration' },
];

const POLL_MS = 3000;
const TERMINAL_STATUSES = new Set(['completed', 'completed_with_errors']);

export function RunDashboardPage() {
  const { runId } = useParams<{ runId: string }>();
  const [tab, setTab] = useState<TabKey>('winrate');
  const runIdNum = Number(runId);

  const statusQuery = useQuery({
    queryKey: ['run-status', runIdNum],
    queryFn: () => fetchRunStatus(runIdNum),
    enabled: Number.isFinite(runIdNum),
    refetchInterval: (query) => {
      const run = query.state.data as RunStatusResponse | undefined;
      return run && TERMINAL_STATUSES.has(run.status) ? false : POLL_MS;
    },
  });

  const run = statusQuery.data;

  return (
    <div className="p-6">
      <Link to="/" className="text-sm text-blue-600 hover:underline">
        ← Runs
      </Link>
      <div className="mb-4 mt-2">
        <h1 className="text-xl font-semibold">Run #{runIdNum}</h1>
        {run && (
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-sm text-gray-500">
            <StatusBadge status={run.status} />
            <span>{run.task}</span>
            <span>{run.arm_names.join(', ')}</span>
            <span>
              {run.completed + run.failed} / {run.total_calls} calls
              {run.failed > 0 && <span className="text-red-600"> ({run.failed} failed)</span>}
            </span>
            {/* Backend stores naive UTC (no offset), so append 'Z' to parse as UTC. */}
            <span>{new Date(run.created_at + 'Z').toLocaleString()}</span>
          </div>
        )}
        {statusQuery.error && (
          <p className="mt-1 text-sm text-red-600">
            {statusQuery.error instanceof Error ? statusQuery.error.message : 'Failed to load run.'}
          </p>
        )}
      </div>
      <div className="mb-4 flex gap-2 border-b">
        {TABS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={`px-3 py-2 text-sm ${
              tab === key ? 'border-b-2 border-blue-600 font-medium' : 'text-gray-500'
            }`}
          >
            {label}
          </button>
        ))}
      </div>
      {tab === 'winrate' && <WinRateTable runId={runIdNum} />}
      {tab === 'frontier' && <FrontierChart runId={runIdNum} />}
      {tab === 'equivalence' && <EquivalencePanel runId={runIdNum} />}
      {tab === 'power' && <PowerPanel runId={runIdNum} />}
      {tab === 'calibration' && <CalibrationReport runId={runIdNum} />}
    </div>
  );
}
