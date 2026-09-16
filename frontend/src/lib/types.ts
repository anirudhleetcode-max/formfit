import type { ExerciseId, FaultCode } from "../pose/rules";

export type Rep = {
  i: number; set: number; score: number; model_score: number | null; faults: FaultCode[];
  ecc_s: number; con_s: number; rom: number; t: number; features: Record<string, number>;
};

export type Fatigue = {
  detected: boolean; onset_rep: number | null; shift: number; t_stat?: number;
  slopes: { con_s: number; rom: number; score: number }; index?: number[];
};

export type Summary = {
  total_reps: number; sets: number; avg_score: number | null; avg_model_score: number | null;
  best_set_reps: number; clean_reps: number; avg_ecc_s: number | null; avg_con_s: number | null;
  top_faults: { code: FaultCode; count: number }[];
};

export type Session = {
  id: string; exercise: ExerciseId; source: "live" | "upload"; status: "active" | "done";
  started_at: string; finished_at?: string; duration_s?: number;
  summary?: Summary; fatigue?: Fatigue; sets?: { set: number; reps: number; avg_score: number }[];
  reps?: Rep[];
};

export type Page<T> = { items: T[]; next_cursor: string | null };

export type Overview = {
  weeks: number;
  totals: { sessions: number; reps: number; clean_reps: number; avg_score: number | null; fatigue_sessions: number };
  weekly: { week: string; reps: number; sessions: number; minutes: number }[];
  daily: { day: string; exercise: ExerciseId; avg_score: number; avg_model_score: number | null; reps: number }[];
  faults: { code: FaultCode; exercise: ExerciseId; label: string; count: number }[];
  bests: { exercise: ExerciseId; sessions: number; total_reps: number; max_session_reps: number;
    max_set_reps: number; best_avg_score: number | null; last: string; trend_per_week: number | null }[];
};

export type ModelInfo = {
  ready: boolean; error: string | null; sklearn_version: string;
  metrics: { n_train: number; n_test: number; exercises: Record<string, {
    selected: string; model: { accuracy: number; f1_faulty: number; roc_auc: number };
    rule_baseline: { accuracy: number; f1_faulty: number; roc_auc: number };
  }> } | null;
};
