import { describe, expect, it, vi, beforeEach } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithQuery } from '../../test/renderWithQuery';
import { NewRunForm } from '../NewRunForm';
import * as client from '../../api/client';

vi.mock('../../api/client');

const mocked = vi.mocked(client);

beforeEach(() => {
  vi.resetAllMocks();
  mocked.fetchArms.mockResolvedValue([]);
  mocked.fetchTasks.mockResolvedValue([
    { name: 'financial_sentiment', description: '', labels: [], active: true, seeded_count: 2264 },
    { name: 'ag_news', description: '', labels: [], active: false, seeded_count: 120 },
  ]);
});

describe('NewRunForm', () => {
  it('populates the task dropdown from GET /tasks after opening', async () => {
    const user = userEvent.setup();
    renderWithQuery(<NewRunForm />);

    await user.click(screen.getByRole('button', { name: 'New run' }));

    const select = await screen.findByRole('combobox');
    expect(select).toHaveValue('financial_sentiment'); // active task is the default
    expect(screen.getByRole('option', { name: /financial_sentiment \(2264 seeded\)/ })).toBeInTheDocument();
    expect(screen.getByRole('option', { name: /ag_news \(120 seeded\)/ })).toBeInTheDocument();
  });

  it('warns and disables submit when the chosen task has no seeded examples', async () => {
    mocked.fetchTasks.mockResolvedValue([
      { name: 'empty_task', description: '', labels: [], active: true, seeded_count: 0 },
    ]);
    const user = userEvent.setup();
    renderWithQuery(<NewRunForm />);

    await user.click(screen.getByRole('button', { name: 'New run' }));

    expect(await screen.findByText(/pe seed --task empty_task/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Start run' })).toBeDisabled();
  });
});
