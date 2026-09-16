import { ArrowLeft, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { RepChart } from "../components/charts";
import { api, ApiError } from "../lib/api";
import { duration, exLabel, fmtDate, fmtTime, scoreClass } from "../lib/format";
import type { Session } from "../lib/types";
import { FAULT_INFO } from "../pose/rules";

export default function SessionDetail() {
  const { id } = useParams();
  const nav = useNavigate();
  const [s, setS] = useState<Session | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api<Session>(`/api/sessions/${id}`).then(setS).catch((e: ApiError) => setError(e.message));
  }, [id]);

  async function remove() {
    if (!confirm("Delete this session? This cannot be undone.")) return;
    try {
      await api(`/api/sessions/${id}`, { method: "DELETE" });
      nav("/history");
    } catch (e) {
      setError((e as Error).message);
    }
  }

  if (error) return <div className="page"><p className="alert">{error}</p><Link to="/history">Back to history</Link></div>;
  if (!s) return <div className="page"><p className="muted">Loading session…</p></div>;

  const sum = s.summary;
  const reps = s.reps ?? [];
  const f = s.fatigue;
  return (
    <div className="page detail">
      <Link to="/history" className="back"><ArrowLeft size={15} /> History</Link>
      <header className="page-head">
        <div>
          <h1>{exLabel(s.exercise)} <span className="h-date">{fmtDate(s.started_at)}, {fmtTime(s.started_at)}</span></h1>
          <p className="sub">{s.source === "upload" ? "From an uploaded video" : "Live session"} · {duration(s.duration_s)}</p>
        </div>
        <button className="btn ghost danger" onClick={remove}><Trash2 size={15} /> Delete</button>
      </header>

      <section className="kpis" aria-label="Summary">
        <div className="kpi"><span>Reps</span><b data-testid="detail-reps">{sum?.total_reps ?? 0}</b></div>
        <div className="kpi"><span>Sets</span><b>{sum?.sets ?? 0}</b></div>
        <div className="kpi"><span>Form score</span><b className={scoreClass(sum?.avg_score)}>{sum?.avg_score ?? "—"}</b></div>
        <div className="kpi"><span>Model: clean-rep chance</span><b>{sum?.avg_model_score ?? "—"}</b></div>
        <div className="kpi"><span>Clean reps</span><b>{sum ? `${sum.clean_reps}/${sum.total_reps}` : "—"}</b></div>
        <div className="kpi"><span>Tempo down / up</span><b>{sum?.avg_ecc_s?.toFixed(1) ?? "—"}<small> / </small>{sum?.avg_con_s?.toFixed(1) ?? "—"}<small>s</small></b></div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Rep by rep</h2>
          {f && reps.length > 0 && (
            <p className={"fatigue-note" + (f.detected ? " on" : "")} data-testid="fatigue-note">
              {f.detected
                ? `Fatigue from rep ${f.onset_rep}: lift time ${f.slopes.con_s >= 0 ? "+" : ""}${(f.slopes.con_s * 1000).toFixed(0)} ms/rep, range ${f.slopes.rom.toFixed(1)}°/rep.`
                : reps.length < (6) ? "Fatigue check needs at least 6 reps." : "No clear fatigue: tempo and range held steady."}
            </p>
          )}
        </div>
        {reps.length ? <RepChart reps={reps} onset={f?.detected ? f.onset_rep : null} /> : <p className="empty">No reps were counted in this session.</p>}
      </section>

      <div className="two-col">
        <section className="panel">
          <h2>Sets</h2>
          <table className="table compact">
            <thead><tr><th>Set</th><th className="r">Reps</th><th className="r">Avg form</th></tr></thead>
            <tbody>
              {(s.sets ?? []).map((x) => (
                <tr key={x.set}><td>{x.set}</td><td className="r num">{x.reps}</td><td className={"r num " + scoreClass(x.avg_score)}>{x.avg_score}</td></tr>
              ))}
            </tbody>
          </table>
          {sum && sum.top_faults.length > 0 && (
            <>
              <h3>Most frequent</h3>
              <ul className="plain">
                {sum.top_faults.map((t) => (
                  <li key={t.code}><span>{FAULT_INFO[t.code]?.label ?? t.code}</span><span className="num muted">{t.count}×</span></li>
                ))}
              </ul>
            </>
          )}
        </section>
        <section className="panel">
          <h2>Reps</h2>
          <div className="scroll-y">
            <table className="table compact">
              <thead><tr><th>#</th><th className="r">Form</th><th className="r" title="Classifier probability that the rep is clean (0-100)">Model %</th><th className="r">↓ s</th><th className="r">↑ s</th><th className="r">ROM</th><th>Faults</th></tr></thead>
              <tbody>
                {reps.map((r) => (
                  <tr key={r.i} className={f?.detected && f.onset_rep && r.i >= f.onset_rep ? "tired" : ""}>
                    <td className="num">{r.i}<span className="muted tiny"> s{r.set}</span></td>
                    <td className={"r num " + scoreClass(r.score)}>{r.score}</td>
                    <td className="r num">{r.model_score ?? "—"}</td>
                    <td className="r num">{r.ecc_s.toFixed(1)}</td>
                    <td className="r num">{r.con_s.toFixed(1)}</td>
                    <td className="r num">{Math.round(r.rom)}°</td>
                    <td className="faults-cell">{r.faults.length ? r.faults.map((c) => FAULT_INFO[c]?.label ?? c).join(", ") : <span className="muted">—</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}
