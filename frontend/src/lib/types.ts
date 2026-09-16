import type { ExerciseId, FaultCode } from "../pose/rules";

export type ModelStatus = "ok" | "uncertain" | "out_of_distribution" | "not_scored" | "unavailable";

export type Rep = {
  i: number; set: number;
  score: number | null;              // null = counted, form not scored (low tracking confidence)
  model_score: number | null;
  faults: FaultCode[];
  ecc_s: number; con_s: number; rom: number; t: number; features: Record<string, number>;
  confidence?: number | null;
  scored?: boolean;
  abstain_reason?: string | null;
  model_status?: ModelStatus;
  model_confidence?: number | null;
  model_ood?: string[];
};

export type Fatigue = {
  detected: boolean; onset_rep: number | null; shift: number; t_stat?: number; min_reps?: number;
  slopes: { con_s: number; rom: number; score: number }; index?: number[];
};

export type Summary = {
  total_reps: number; sets: number; avg_score: number | null; avg_model_score: number | null;
  best_set_reps: number; clean_reps: number; avg_ecc_s: number | null; avg_con_s: number | null;
  top_faults: { code: FaultCode; count: number }[];
  scored_reps?: number; unscored_reps?: number; avg_confidence?: number | null;
  model_status_counts?: Partial<Record<ModelStatus, number>>;
};

export type Session = {
  id: string; exercise: ExerciseId; source: "live" | "upload"; status: "active" | "done" | "abandoned";
  started_at: string; finished_at?: string; duration_s?: number; demo?: boolean;
  summary?: Summary; fatigue?: Fatigue;
  sets?: { set: number; reps: number; avg_score: number | null; scored_reps?: number }[];
  reps?: Rep[];
};

export type Page<T> = { items: T[]; next_cursor: string | null };

export type Overview = {
  weeks: number;
  totals: { sessions: number; reps: number; clean_reps: number; avg_score: number | null; fatigue_sessions: number; demo_sessions?: number };
  weekly: { week: string; reps: number; sessions: number; minutes: number }[];
  daily: { day: string; exercise: ExerciseId; avg_score: number; avg_model_score: number | null; reps: number }[];
  faults: { code: FaultCode; exercise: ExerciseId; label: string; count: number }[];
  bests: { exercise: ExerciseId; sessions: number; total_reps: number; max_session_reps: number;
    max_set_reps: number; best_avg_score: number | null; last: string; trend_per_week: number | null }[];
};

type Scores = { accuracy: number; f1_faulty: number; roc_auc: number | null; pr_auc_faulty?: number | null; ece?: number | null; brier?: number | null };

export type ModelCard = {
  name: string; model_version: string; trained_at: string; git_commit: string | null; task: string;
  dataset: { name: string; synthetic: boolean; simulator_version: string; sizes: Record<string, number> };
  preprocessing_version: string; seed: number; libraries: Record<string, string>;
  decision: { threshold: string; abstention: string };
  intended_use: string; limitations: string[];
};

export type ModelInfo = {
  ready: boolean; error: string | null; sklearn_version: string; synthetic?: boolean; version?: string | null;
  training_data?: string; card?: ModelCard | null;
  metrics: { n_train: number; n_val?: number; n_test: number; synthetic?: boolean; exercises: Record<string, {
    selected: string; model: Scores; rule_baseline: Scores; logreg?: Scores; majority?: Scores;
    selective?: { min_confidence: number; coverage: number; accuracy_on_covered: number | null } | null;
  }> } | null;
  real_clip_eval?: RealClipEval | null;
};

export type RepCountMetrics = {
  n_clips: number; gt_reps_total?: number; pred_reps_total?: number; mae?: number;
  exact_match_rate?: number; within_1_rate?: number; scored_share_of_predicted?: number;
};

export type RealClipEval = {
  run_id: string;
  dataset: { name: string; synthetic: boolean; n_clips_listed: number };
  overall: RepCountMetrics;
  per_exercise: Record<string, RepCountMetrics>;
  per_source_type?: Record<string, RepCountMetrics>;
  model_on_real_reps?: { reps: number; status_counts: Record<string, number>; ood_features: Record<string, number> };
  note: string;
};
