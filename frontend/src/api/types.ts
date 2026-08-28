export interface RunSummary {
  run_id: number;
  created_at: string;
  arm_names: string[];
  status: string;
  total_calls: number;
  completed: number;
  failed: number;
  pending: number;
}

export interface ArmSummaryResponse {
  arm_name: string;
  n: number;
  mean_judge_score: number | null;
  mean_latency_ms: number | null;
  mean_cost_estimate_usd: number | null;
  mean_prompt_tokens: number | null;
  mean_completion_tokens: number | null;
}

export interface PairedComparisonResponse {
  arm_a: string;
  arm_b: string;
  metric: string;
  n_examples: number;
  n_excluded: number;
  mean_diff: number;
  ci_lower: number;
  ci_upper: number;
  wilcoxon_statistic: number;
  p_value: number;
  p_value_corrected: number | null;
}

export interface CalibrationResponse {
  run_id: number;
  n: number;
  spearman_r: number;
  spearman_p: number;
  cohens_kappa: number;
  mean_abs_diff: number;
}
