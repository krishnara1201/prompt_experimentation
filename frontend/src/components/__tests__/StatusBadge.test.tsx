import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { StatusBadge } from '../StatusBadge';

describe('StatusBadge', () => {
  it('shows the status text', () => {
    render(<StatusBadge status="running" />);
    expect(screen.getByText('running')).toBeInTheDocument();
  });

  it('applies the per-status color style', () => {
    render(<StatusBadge status="completed" />);
    expect(screen.getByText('completed').className).toContain('bg-green-100');
  });

  it('applies a yellow style for completed_with_errors', () => {
    render(<StatusBadge status="completed_with_errors" />);
    expect(screen.getByText('completed_with_errors').className).toContain('bg-yellow-100');
  });

  it('falls back to the gray style for an unknown status', () => {
    render(<StatusBadge status="mystery" />);
    expect(screen.getByText('mystery').className).toContain('bg-gray-100');
  });
});
