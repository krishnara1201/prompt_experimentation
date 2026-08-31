import { useEffect, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { createRun, fetchArms, fetchTasks } from '../api/client';
import type { RunCreateRequest } from '../api/types';

function parseOptionalInt(value: string): number | undefined {
  const trimmed = value.trim();
  if (trimmed === '') return undefined;
  const n = Number(trimmed);
  return Number.isFinite(n) ? n : undefined;
}

export function NewRunForm() {
  const [open, setOpen] = useState(false);
  const [sampleSize, setSampleSize] = useState('');
  const [repeats, setRepeats] = useState('1');
  const [seed, setSeed] = useState('');
  const [selectedArms, setSelectedArms] = useState<string[]>([]);
  const [task, setTask] = useState<string>('');

  const queryClient = useQueryClient();
  const navigate = useNavigate();

  const armsQuery = useQuery({ queryKey: ['arms'], queryFn: fetchArms, enabled: open });
  const tasksQuery = useQuery({ queryKey: ['tasks'], queryFn: fetchTasks, enabled: open });

  useEffect(() => {
    if (task !== '' || !tasksQuery.data) return;
    const active = tasksQuery.data.find((t) => t.active) ?? tasksQuery.data[0];
    if (active) setTask(active.name);
  }, [task, tasksQuery.data]);

  const selectedTask = tasksQuery.data?.find((t) => t.name === task);
  const taskNotSeeded = selectedTask !== undefined && selectedTask.seeded_count === 0;

  const mutation = useMutation({
    mutationFn: (body: RunCreateRequest) => createRun(body),
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['runs'] });
      navigate(`/runs/${data.run_id}`);
    },
  });

  function toggleArm(name: string) {
    setSelectedArms((prev) =>
      prev.includes(name) ? prev.filter((a) => a !== name) : [...prev, name],
    );
  }

  function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    const body: RunCreateRequest = { repeats: parseOptionalInt(repeats) ?? 1 };
    const sample = parseOptionalInt(sampleSize);
    if (sample !== undefined) body.sample_size = sample;
    const seedValue = parseOptionalInt(seed);
    if (seedValue !== undefined) body.seed = seedValue;
    if (selectedArms.length > 0) body.arms = selectedArms;
    if (task) body.task = task;
    mutation.mutate(body);
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="mb-4 rounded bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700"
      >
        New run
      </button>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="mb-6 max-w-md rounded border p-4 text-sm">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="font-semibold">New run</h2>
        <button type="button" onClick={() => setOpen(false)} className="text-xs text-gray-500 hover:underline">
          Cancel
        </button>
      </div>

      <label className="mb-2 block">
        <span className="text-xs uppercase text-gray-500">Task</span>
        {tasksQuery.isLoading && <p className="mt-1 text-xs text-gray-500">Loading tasks…</p>}
        {tasksQuery.error && (
          <p className="mt-1 text-xs text-red-600">
            {tasksQuery.error instanceof Error ? tasksQuery.error.message : 'Failed to load tasks.'}
          </p>
        )}
        {tasksQuery.data && (
          <select
            value={task}
            onChange={(e) => setTask(e.target.value)}
            className="mt-1 w-full rounded border px-2 py-1"
          >
            {tasksQuery.data.map((t) => (
              <option key={t.name} value={t.name}>
                {t.name} ({t.seeded_count} seeded)
              </option>
            ))}
          </select>
        )}
        {taskNotSeeded && (
          <p className="mt-1 text-xs text-red-600">
            Run <code>pe seed --task {task}</code> first.
          </p>
        )}
      </label>

      <label className="mb-2 block">
        <span className="text-xs uppercase text-gray-500">Sample size</span>
        <input
          type="number"
          min="1"
          value={sampleSize}
          onChange={(e) => setSampleSize(e.target.value)}
          placeholder="whole dataset"
          className="mt-1 w-full rounded border px-2 py-1"
        />
      </label>

      <label className="mb-2 block">
        <span className="text-xs uppercase text-gray-500">Repeats</span>
        <input
          type="number"
          min="1"
          value={repeats}
          onChange={(e) => setRepeats(e.target.value)}
          className="mt-1 w-full rounded border px-2 py-1"
        />
      </label>

      <label className="mb-3 block">
        <span className="text-xs uppercase text-gray-500">Seed</span>
        <input
          type="number"
          value={seed}
          onChange={(e) => setSeed(e.target.value)}
          placeholder="random"
          className="mt-1 w-full rounded border px-2 py-1"
        />
      </label>

      <fieldset className="mb-3">
        <legend className="text-xs uppercase text-gray-500">Arms</legend>
        {armsQuery.isLoading && <p className="mt-1 text-xs text-gray-500">Loading arms…</p>}
        {armsQuery.error && (
          <p className="mt-1 text-xs text-red-600">
            {armsQuery.error instanceof Error ? armsQuery.error.message : 'Failed to load arms.'}
          </p>
        )}
        {armsQuery.data?.map((arm) => (
          <label key={arm.name} className="mt-1 flex items-center gap-2">
            <input
              type="checkbox"
              checked={selectedArms.includes(arm.name)}
              onChange={() => toggleArm(arm.name)}
            />
            <span>
              {arm.name}
              {arm.model ? <span className="text-gray-400"> ({arm.model})</span> : null}
            </span>
          </label>
        ))}
        <p className="mt-1 text-xs text-gray-400">None selected = every configured arm.</p>
      </fieldset>

      {mutation.error && (
        <p className="mb-2 text-xs text-red-600">
          {mutation.error instanceof Error ? mutation.error.message : 'Failed to start run.'}
        </p>
      )}

      <button
        type="submit"
        disabled={mutation.isPending || taskNotSeeded}
        className="rounded bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
      >
        {mutation.isPending ? 'Starting…' : 'Start run'}
      </button>
    </form>
  );
}
