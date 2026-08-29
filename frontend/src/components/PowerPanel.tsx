import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchPower, fetchRunSummary } from '../api/client';
import { QueryState } from './QueryState';
import { ArmPairFields } from './ArmPairFields';

export function PowerPanel({ runId }: { runId: number }) {
  const summaryQuery = useQuery({
    queryKey: ['run-summary', runId],
    queryFn: () => fetchRunSummary(runId),
  });

  const [armA, setArmA] = useState('');
  const [armB, setArmB] = useState('');
  const [power, setPower] = useState('0.8');
  const [alpha, setAlpha] = useState('0.05');
  const [effectSize, setEffectSize] = useState('');

  const powerNum = Number(power);
  const alphaNum = Number(alpha);
  const effectSizeNum = effectSize.trim() === '' ? undefined : Number(effectSize);
  const canCompute =
    armA !== '' &&
    armB !== '' &&
    armA !== armB &&
    powerNum > 0 &&
    powerNum < 1 &&
    alphaNum > 0 &&
    alphaNum < 1 &&
    (effectSizeNum === undefined || effectSizeNum > 0);

  const result = useQuery({
    queryKey: ['run-power', runId, armA, armB, powerNum, alphaNum, effectSizeNum],
    queryFn: () =>
      fetchPower(runId, {
        armA,
        armB,
        power: powerNum,
        alpha: alphaNum,
        effectSize: effectSizeNum,
      }),
    enabled: false,
    retry: false,
  });

  return (
    <div className="max-w-lg text-sm">
      <p className="mb-4 text-gray-500">
        Sample-size / power: given this run's judge-score difference as a pilot, how many paired
        examples reach the target power? Metric is fixed to <code>judge_score</code>.
      </p>

      <QueryState
        isLoading={summaryQuery.isLoading}
        error={summaryQuery.error}
        onRetry={summaryQuery.refetch}
      >
        {summaryQuery.data && (
          <>
            <ArmPairFields
              arms={summaryQuery.data.map((row) => row.arm_name)}
              labelA="Arm A"
              labelB="Arm B"
              valueA={armA}
              valueB={armB}
              onChangeA={setArmA}
              onChangeB={setArmB}
            />
            <label className="mb-2 block">
              <span className="text-xs uppercase text-gray-500">Target power</span>
              <input
                type="number"
                min="0"
                max="1"
                step="0.05"
                value={power}
                onChange={(e) => setPower(e.target.value)}
                className="mt-1 w-full rounded border px-2 py-1"
              />
            </label>
            <label className="mb-2 block">
              <span className="text-xs uppercase text-gray-500">Alpha</span>
              <input
                type="number"
                min="0"
                max="1"
                step="0.01"
                value={alpha}
                onChange={(e) => setAlpha(e.target.value)}
                className="mt-1 w-full rounded border px-2 py-1"
              />
            </label>
            <label className="mb-3 block">
              <span className="text-xs uppercase text-gray-500">Effect size (optional)</span>
              <input
                type="number"
                min="0"
                step="0.05"
                value={effectSize}
                onChange={(e) => setEffectSize(e.target.value)}
                placeholder="from pilot data"
                className="mt-1 w-full rounded border px-2 py-1"
              />
            </label>

            <button
              onClick={() => result.refetch()}
              disabled={!canCompute || result.isFetching}
              className="rounded bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {result.isFetching ? 'Computing…' : 'Compute'}
            </button>

            {result.error && (
              <p className="mt-3 text-xs text-red-600">
                {result.error instanceof Error ? result.error.message : 'Computation failed.'}
              </p>
            )}

            {result.data && !result.isFetching && (
              <dl className="mt-4 grid grid-cols-2 gap-3">
                <dt className="text-gray-500">Required n (per arm)</dt>
                <dd className="font-medium">{result.data.required_n}</dd>
                <dt className="text-gray-500">Achieved power</dt>
                <dd>{result.data.achieved_power.toFixed(3)}</dd>
                <dt className="text-gray-500">Effect size</dt>
                <dd>{result.data.effect_size.toFixed(3)}</dd>
                <dt className="text-gray-500">Pilot mean diff</dt>
                <dd>{result.data.pilot_mean_diff.toFixed(3)}</dd>
                <dt className="text-gray-500">Pilot std diff</dt>
                <dd>{result.data.pilot_std_diff.toFixed(3)}</dd>
                <dt className="text-gray-500">Pilot n</dt>
                <dd>
                  {result.data.pilot_n}
                  {result.data.n_excluded > 0 && (
                    <span className="text-gray-400"> ({result.data.n_excluded} excluded)</span>
                  )}
                </dd>
              </dl>
            )}
          </>
        )}
      </QueryState>
    </div>
  );
}
