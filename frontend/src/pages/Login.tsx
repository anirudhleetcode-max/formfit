import { useState, type FormEvent } from "react";
import { Navigate } from "react-router-dom";
import { DISCLAIMER } from "../components/Feedback";
import { Logo } from "../components/Shell";
import { useAuth } from "../lib/auth";

export default function Login() {
  const { user, login, register } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to="/" replace />;

  async function submit(e: FormEvent) {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      if (mode === "login") await login(email, password);
      else await register(name, email, password);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="auth">
      <section className="auth-side">
        <Logo />
        <div className="auth-copy">
          <h1>Count every rep.<br />Fix the ones that don't count.</h1>
          <p>FormFit watches your squats, push-ups, curls and presses through the webcam, counts reps and calls out form faults as they happen. Video never leaves your computer.</p>
        </div>
        <ul className="auth-facts">
          <li><b>5</b><span>exercises</span></li>
          <li><b>17</b><span>form checks</span></li>
          <li><b>0</b><span>frames uploaded</span></li>
        </ul>
        <p className="auth-disclaimer">{DISCLAIMER}</p>
      </section>
      <section className="auth-form-wrap">
        <form className="auth-form" onSubmit={submit}>
          <h2>{mode === "login" ? "Sign in" : "Create account"}</h2>
          {mode === "register" && (
            <label className="field">
              <span>Name</span>
              <input value={name} onChange={(e) => setName(e.target.value)} required maxLength={60} autoComplete="name" />
            </label>
          )}
          <label className="field">
            <span>Email</span>
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" />
          </label>
          <label className="field">
            <span>Password</span>
            <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={6}
              autoComplete={mode === "login" ? "current-password" : "new-password"} />
          </label>
          {error && <p className="alert" role="alert">{error}</p>}
          <button className="btn primary wide" disabled={busy} type="submit">
            {busy ? "Please wait…" : mode === "login" ? "Sign in" : "Create account"}
          </button>
          <div className="auth-links">
            <button type="button" className="link" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(null); }}>
              {mode === "login" ? "New here? Create an account" : "Have an account? Sign in"}
            </button>
            {mode === "login" && (
              <button type="button" className="link muted" onClick={() => { setEmail("demo@formfit.app"); setPassword("demo1234"); }}>
                Use demo account
              </button>
            )}
          </div>
        </form>
      </section>
    </div>
  );
}
