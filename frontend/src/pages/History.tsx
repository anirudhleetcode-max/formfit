import { ChevronRight } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Skeleton } from "../components/Feedback";
import { api } from "../lib/api";
import { duration, exLabel, fmtDate, fmtTime, scoreClass } from "../lib/format";
import type { Page, Session } from "../lib/types";
import { EXERCISE_LIST, type ExerciseId } from "../pose/rules";

export default function History() {
  const [filter, setFilter] = useState<ExerciseId | "">("");
  const [items, setItems] = useState<Session[]>([]);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (reset: boolean, cur: string | null) => {
    setLoading(true);
    setError(null);
    try {
      const q = new URLSearchParams({ limit: "15" });
      if (filter) q.set("exercise", filter);
      if (!reset && cur) q.set("cursor", cur);
      const page = await api<Page<Session>>(`/api/sessions?${q}`);
      setItems((prev) => (reset ? page.items : [...prev, ...page.items]));
      setCursor(page.next_cursor);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => { load(true, null); }, [load]);

  return (
    <div className="page">
      <header className="page-head">
        <div>
          <h1>History</h1>
          <p className="sub">Every saved session, newest first. Open one for the rep-by-rep breakdown.</p>
        </div>
        <label className="select">
          <span className="sr-only">Exercise</span>
          <select value={filter} onChange={(e) => setFilter(e.target.value as ExerciseId | "")}>
            <option value="">All exercises</option>
            {EXERCISE_LIST.map((d) => <option key={d.id} value={d.id}>{d.label}</option>)}
          </select>
        </label>
      </header>
      {error && <p className="alert" role="alert">{error}</p>}

      <div className="table-wrap">
        <table className="table sessions">
          <thead>
            <tr>
              <th>Date</th><th>Exercise</th><th className="r">Reps</th><th className="r">Sets</th>
              <th className="r">Form</th><th className="r hide-sm" title="Average probability of a clean rep from a classifier trained on synthetic reps">Model %</th><th className="hide-sm">Fatigue</th><th className="hide-sm">Length</th><th aria-label="Open" />
            </tr>
          </thead>
          <tbody>
            {items.map((s) => (
              <tr key={s.id}>
                <td>
                  <Link to={`/history/${s.id}`} className="row-link">{fmtDate(s.started_at)}</Link>
                  <span className="muted tiny block">{fmtTime(s.started_at)}{s.source === "upload" ? " · video" : ""}
                    {s.demo && <span className="tag demo">demo</span>}</span>
                </td>
                <td>{exLabel(s.exercise)}</td>
                <td className="r num">{s.summary?.total_reps ?? 0}</td>
                <td className="r num">{s.summary?.sets ?? 0}</td>
                <td className={"r num " + scoreClass(s.summary?.avg_score)}>{s.summary?.avg_score ?? "—"}</td>
                <td className="r num hide-sm">{s.summary?.avg_model_score ?? "—"}</td>
                <td className="hide-sm">{s.fatigue?.detected ? <span className="tag warn">from rep {s.fatigue.onset_rep}</span> : <span className="muted">—</span>}</td>
                <td className="hide-sm muted">{duration(s.duration_s)}</td>
                <td className="r"><Link to={`/history/${s.id}`} aria-label="Open session" className="chev"><ChevronRight size={16} /></Link></td>
              </tr>
            ))}
          </tbody>
        </table>
        {!loading && items.length === 0 && (
          <div className="empty">
            <p>No sessions {filter ? `for ${exLabel(filter)} ` : ""}yet.</p>
            <p className="muted">Finish a set on <Link to="/">Train</Link> or <Link to="/upload">upload a video</Link> and it shows up here.</p>
          </div>
        )}
        {loading && <div className="pad"><Skeleton rows={4} /></div>}
      </div>
      {cursor && !loading && <button className="btn more" onClick={() => load(false, cursor)}>Load older sessions</button>}
    </div>
  );
}
