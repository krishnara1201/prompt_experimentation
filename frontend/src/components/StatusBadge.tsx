const STATUS_STYLES: Record<string, string> = {
  pending: 'bg-gray-100 text-gray-700',
  running: 'bg-blue-100 text-blue-700',
  completed: 'bg-green-100 text-green-700',
  completed_with_errors: 'bg-yellow-100 text-yellow-800',
};

export function StatusBadge({ status }: { status: string }) {
  const style = STATUS_STYLES[status] ?? 'bg-gray-100 text-gray-700';
  return <span className={`rounded px-2 py-0.5 text-xs font-medium ${style}`}>{status}</span>;
}
