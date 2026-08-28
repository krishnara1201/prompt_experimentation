import { useParams } from 'react-router-dom';

export function RunDashboardPage() {
  const { runId } = useParams<{ runId: string }>();
  return <div className="p-6">Run dashboard placeholder for run #{runId}</div>;
}
