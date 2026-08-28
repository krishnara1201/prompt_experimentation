import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { WinRateTable } from '../components/WinRateTable';
import { FrontierChart } from '../components/FrontierChart';
import { CalibrationReport } from '../components/CalibrationReport';

type TabKey = 'winrate' | 'frontier' | 'calibration';

const TABS: { key: TabKey; label: string }[] = [
  { key: 'winrate', label: 'Win-rate' },
  { key: 'frontier', label: 'Frontier' },
  { key: 'calibration', label: 'Calibration' },
];

export function RunDashboardPage() {
  const { runId } = useParams<{ runId: string }>();
  const [tab, setTab] = useState<TabKey>('winrate');
  const runIdNum = Number(runId);

  return (
    <div className="p-6">
      <h1 className="mb-4 text-xl font-semibold">Run #{runIdNum}</h1>
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
      {tab === 'calibration' && <CalibrationReport runId={runIdNum} />}
    </div>
  );
}
