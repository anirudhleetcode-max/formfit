import { EXERCISES, type ExerciseId } from "../pose/rules";

export const exLabel = (e: ExerciseId) => EXERCISES[e]?.label ?? e;

export function clock(totalS: number) {
  const s = Math.max(0, Math.floor(totalS));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

export function duration(totalS?: number) {
  if (!totalS) return "—";
  const m = Math.round(totalS / 60);
  return m < 1 ? `${Math.round(totalS)} s` : `${m} min`;
}

const dateFmt = new Intl.DateTimeFormat(undefined, { weekday: "short", day: "numeric", month: "short" });
const timeFmt = new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" });
export const fmtDate = (iso: string) => dateFmt.format(new Date(iso));
export const fmtTime = (iso: string) => timeFmt.format(new Date(iso));
export const shortDay = (iso: string) => new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short" }).format(new Date(iso));

export function scoreClass(score: number | null | undefined) {
  if (score === null || score === undefined) return "";
  return score >= 80 ? "s-good" : score >= 60 ? "s-mid" : "s-bad";
}

export const localTz = () => {
  try { return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"; } catch { return "UTC"; }
};
