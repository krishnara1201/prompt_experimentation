import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchEquivalence, fetchRunSummary } from '../api/client';
import { QueryState } from './QueryState';
import { ArmPairFields } from './ArmPairFields';

export function EquivalencePanel({ runId }: { runId: number }) {
  const summaryQuery = useQuery({
    queryKey: ['run-summary', runId],
    queryFn: () => fetchRunSummary(runId),
  });

  const [armLocal, setArmLocal] = useState('');
  const [armApi, setArmApi] = useState('');
  const [epsilon, setEpsilon] = useState('0.05');

  const epsilonNum = Number(epsilon);
  const canCompute = armLocal !== '' && armApi !== '' && armLocal !== armApi && epsilonNum > 0;

  const result = useQuery({
    queryKey: ['run-equivalence', runId, armLocal, armApi, epsilonNum],
    queryFn: () => fetchEquivalence(runId, { armLocal, armApi, epsilon: epsilonNum }),
    enabled: false,
    retry: false,
  });

  return (
    <div className="max-w-lg text-sm">
      <p className="mb-4 text-gray-500">
        Bayesian equivalence: P(quality<sub>local</sub> ≥ quality<sub>api</sub> − ε) on the paired
        per-example judge-score difference. Metric is fixed to <code>judge_score</code>.
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
              labelA="Local arm"
              labelB="API arm"
              valueA={armLocal}
              valueB={armApi}
              onChangeA={setArmLocal}
              onChangeB={setArmApi}
            />
            <label className="mb-3 block">
              <span className="text-xs uppercase text-gray-500">Margin ε</span>
              <input
                type="number"
                min="0"
                step="0.01"
                value={epsilon}
                onChange={(e) => setEpsilon(e.target.value)}
                className="mt-1 w-full rounded border px-2 py-1"
              />
            </label>

            <button
              onClick={() => result.refetch()}
              disabled={!canCompute || result.isFetching}
              className="rounded bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {result.isFetching ? 'Sampling posterior… (~5s)' : 'Compute'}
            </button>

            {result.error && (
              <p className="mt-3 text-xs text-red-600">
                {result.error instanceof Error ? result.error.message : 'Computation failed.'}
              </p>
            )}

            {result.data && !result.isFetching && (
              <dl className="mt-4 grid grid-cols-2 gap-3">
                <dt className="text-gray-500">P(local ≥ api − ε)</dt>
                <dd className="font-medium">{result.data.p_equivalent.toFixed(3)}</dd>
                <dt className="text-gray-500">Posterior mean diff</dt>
                <dd>{result.data.posterior_mean.toFixed(3)}</dd>
                <dt className="text-gray-500">94% CI</dt>
                <dd>
                  [{result.data.ci_lower.toFixed(3)}, {result.data.ci_upper.toFixed(3)}]
                </dd>
                <dt className="text-gray-500">n examples</dt>
                <dd>
                  {result.data.n_examples}
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
