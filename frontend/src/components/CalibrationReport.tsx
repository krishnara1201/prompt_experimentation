import { useQuery } from '@tanstack/react-query';
import { fetchCalibration } from '../api/client';

export function CalibrationReport({ runId }: { runId: number }) {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['run-calibration', runId],
    queryFn: () => fetchCalibration(runId),
    retry: false,
  });

  if (isLoading) {
    return <div className="p-4 text-sm text-gray-500">Loading…</div>;
  }

  if (error) {
    const message = error instanceof Error ? error.message : 'Something went wrong.';
    if (message.toLowerCase().includes('no calibration')) {
      return (
        <p className="text-sm text-gray-500">
          No calibration sample recorded for this run — see select_calibration_sample.py.
        </p>
      );
    }
    return (
      <div className="p-4 text-sm text-red-600">
        <p>{message}</p>
        <button onClick={() => refetch()} className="mt-2 rounded border px-2 py-1 text-xs">
          Retry
        </button>
      </div>
    );
  }

  if (!data) return null;

  return (
    <dl className="grid max-w-md grid-cols-2 gap-3 text-sm">
      <dt className="text-gray-500">n</dt>
      <dd>{data.n}</dd>
      <dt className="text-gray-500">Spearman r</dt>
      <dd>
        {data.spearman_r.toFixed(3)} (p = {data.spearman_p.toFixed(3)})
      </dd>
      <dt className="text-gray-500">Cohen&apos;s kappa</dt>
      <dd>{data.cohens_kappa.toFixed(3)}</dd>
      <dt className="text-gray-500">Mean abs diff</dt>
      <dd>{data.mean_abs_diff.toFixed(3)}</dd>
    </dl>
  );
}
