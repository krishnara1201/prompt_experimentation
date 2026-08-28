import type { ReactNode } from 'react';

interface QueryStateProps {
  isLoading: boolean;
  error: unknown;
  onRetry: () => void;
  children: ReactNode;
}

export function QueryState({ isLoading, error, onRetry, children }: QueryStateProps) {
  if (isLoading) {
    return <div className="p-4 text-sm text-gray-500">Loading…</div>;
  }
  if (error) {
    const message = error instanceof Error ? error.message : 'Something went wrong.';
    return (
      <div className="p-4 text-sm text-red-600">
        <p>{message}</p>
        <button onClick={onRetry} className="mt-2 rounded border px-2 py-1 text-xs">
          Retry
        </button>
      </div>
    );
  }
  return <>{children}</>;
}
