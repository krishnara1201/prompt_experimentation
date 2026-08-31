import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import { fetchRuns } from '../api/client';
import { NewRunForm } from '../components/NewRunForm';
import { QueryState } from '../components/QueryState';
import { StatusBadge } from '../components/StatusBadge';
import type { RunSummary } from '../api/types';

const POLL_MS = 3000;
const TERMINAL_STATUSES = new Set(['completed', 'completed_with_errors']);

export function RunListPage() {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['runs'],
    queryFn: fetchRuns,
    refetchInterval: (query) => {
      const runs = query.state.data as RunSummary[] | undefined;
      const hasActiveRun = runs ? runs.some((run) => !TERMINAL_STATUSES.has(run.status)) : true;
      return hasActiveRun ? POLL_MS : false;
    },
  });

  return (
    <div className="p-6">
      <h1 className="mb-4 text-xl font-semibold">Runs</h1>
      <NewRunForm />
      <QueryState isLoading={isLoading} error={error} onRetry={refetch}>
        {data && data.length === 0 && (
          <p className="text-sm text-gray-500">No runs yet — start one with the “New run” button.</p>
        )}
        {data && data.length > 0 && (
          <table className="w-full text-left text-sm">
            <thead>
              <tr className="border-b text-xs uppercase text-gray-500">
                <th className="py-2">Run</th>
                <th>Created</th>
                <th>Task</th>
                <th>Arms</th>
                <th>Status</th>
                <th>Progress</th>
              </tr>
            </thead>
            <tbody>
              {data.map((run) => (
                <tr key={run.run_id} className="border-b hover:bg-gray-50">
                  <td className="py-2">
                    <Link to={`/runs/${run.run_id}`} className="text-blue-600 hover:underline">
                      #{run.run_id}
                    </Link>
                  </td>
                  {/* Backend stores naive UTC (no offset in the string), so append 'Z'
                      to make Date parse it as UTC instead of local time. */}
                  <td>{new Date(run.created_at + 'Z').toLocaleString()}</td>
                  <td>{run.task}</td>
                  <td>{run.arm_names.join(', ')}</td>
                  <td>
                    <StatusBadge status={run.status} />
                  </td>
                  <td>
                    {run.completed + run.failed} / {run.total_calls}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </QueryState>
    </div>
  );
}
