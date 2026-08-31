import { describe, expect, it, vi, beforeEach } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithQuery } from '../../test/renderWithQuery';
import { CalibrationReport } from '../CalibrationReport';
import * as client from '../../api/client';

vi.mock('../../api/client');

const mocked = vi.mocked(client);

beforeEach(() => {
  vi.resetAllMocks();
});

describe('CalibrationReport', () => {
  it('renders the agreement stats when a calibration sample exists', async () => {
    mocked.fetchCalibration.mockResolvedValue({
      run_id: 705,
      n: 50,
      spearman_r: 0.812,
      spearman_p: 0.0001,
      cohens_kappa: 0.744,
      mean_abs_diff: 0.36,
    });

    renderWithQuery(<CalibrationReport runId={705} />);

    expect(await screen.findByText('50')).toBeInTheDocument();
    expect(screen.getByText('0.744')).toBeInTheDocument();
    expect(screen.getByText(/0\.812/)).toBeInTheDocument();
    expect(screen.getByText('0.360')).toBeInTheDocument();
  });

  it('shows a friendly message when no calibration sample is recorded', async () => {
    mocked.fetchCalibration.mockRejectedValue(new Error('No calibration sample for run 705'));

    renderWithQuery(<CalibrationReport runId={705} />);

    expect(
      await screen.findByText(/No calibration sample recorded for this run/i),
    ).toBeInTheDocument();
  });
});
