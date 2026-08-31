export interface ArmInfo {
  name: string;
  adapter: string;
  model: string | null;
  prompt_template: string;
}

export interface TaskInfo {
  name: string;
  description: string;
  labels: string[];
  active: boolean;
  seeded_count: number;
}

export interface RunCreateRequest {
  arms?: string[];
  sample_size?: number;
  repeats?: number;
  seed?: number;
  task?: string;
}

export interface RunCreateResponse {
  run_id: number;
  status: string;
  total_calls: number;
}

export interface RunSummary {
  run_id: number;
  created_at: string;
  arm_names: string[];
  task: string;
  status: string;
  total_calls: number;
  completed: number;
  failed: number;
  pending: number;
}

export interface RunStatusResponse {
  run_id: number;
  status: string;
  task: string;
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

export interface EquivalenceResponse {
  arm_local: string;
  arm_api: string;
  metric: string;
  epsilon: number;
  n_examples: number;
  n_excluded: number;
  posterior_mean: number;
  ci_lower: number;
  ci_upper: number;
  p_equivalent: number;
}

export interface PowerResponse {
  arm_a: string;
  arm_b: string;
  metric: string;
  pilot_n: number;
  pilot_mean_diff: number;
  pilot_std_diff: number;
  effect_size: number;
  alpha: number;
  target_power: number;
  required_n: number;
  achieved_power: number;
  n_excluded: number;
}

export interface CalibrationResponse {
  run_id: number;
  n: number;
  spearman_r: number | null;
  spearman_p: number | null;
  cohens_kappa: number;
  mean_abs_diff: number;
}
