import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { useParams } from 'react-router-dom';
import { fetchRunStatus } from '../api/client';
import { WinRateTable } from '../components/WinRateTable';
import { FrontierChart } from '../components/FrontierChart';
import { CalibrationReport } from '../components/CalibrationReport';
import { EquivalencePanel } from '../components/EquivalencePanel';
import { PowerPanel } from '../components/PowerPanel';

type TabKey = 'winrate' | 'frontier' | 'equivalence' | 'power' | 'calibration';

const TABS: { key: TabKey; label: string }[] = [
  { key: 'winrate', label: 'Win-rate' },
  { key: 'frontier', label: 'Frontier' },
  { key: 'equivalence', label: 'Equivalence' },
  { key: 'power', label: 'Power' },
  { key: 'calibration', label: 'Calibration' },
];

export function RunDashboardPage() {
  const { runId } = useParams<{ runId: string }>();
  const [tab, setTab] = useState<TabKey>('winrate');
  const runIdNum = Number(runId);

  const statusQuery = useQuery({
    queryKey: ['run-status', runIdNum],
    queryFn: () => fetchRunStatus(runIdNum),
    enabled: Number.isFinite(runIdNum),
  });

  return (
    <div className="p-6">
      <div className="mb-4">
        <h1 className="text-xl font-semibold">Run #{runIdNum}</h1>
        {statusQuery.data && (
          <p className="mt-1 text-sm text-gray-500">Task: {statusQuery.data.task}</p>
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
