import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";

/* ---------- toasts ---------- */
type Toast = { id: number; text: string; kind: "ok" | "error" };
const ToastCtx = createContext<(text: string, kind?: Toast["kind"]) => void>(() => {});

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);
  const push = useCallback((text: string, kind: Toast["kind"] = "ok") => {
    const id = Date.now() + Math.random();
    setItems((xs) => [...xs, { id, text, kind }]);
    setTimeout(() => setItems((xs) => xs.filter((x) => x.id !== id)), 4000);
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {items.map((t) => <div key={t.id} className={"toast " + t.kind}>{t.text}</div>)}
      </div>
    </ToastCtx.Provider>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
export const useToast = () => useContext(ToastCtx);

/* ---------- confirm dialog ---------- */
export function ConfirmDialog({ open, title, body, confirmLabel, onConfirm, onCancel, danger }: {
  open: boolean; title: string; body: string; confirmLabel: string;
  onConfirm: () => void; onCancel: () => void; danger?: boolean;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) {
      if (typeof d.showModal === "function") d.showModal(); else d.setAttribute("open", "");
    }
    if (!open && d.open) {
      if (typeof d.close === "function") d.close(); else d.removeAttribute("open");
    }
  }, [open]);
  return (
    <dialog ref={ref} className="dialog" aria-labelledby="dlg-title" onCancel={(e) => { e.preventDefault(); onCancel(); }}>
      <h2 id="dlg-title">{title}</h2>
      <p className="muted">{body}</p>
      <div className="dialog-actions">
        <button className="btn" onClick={onCancel} autoFocus>Cancel</button>
        <button className={"btn " + (danger ? "danger-solid" : "primary")} onClick={onConfirm}>{confirmLabel}</button>
      </div>
    </dialog>
  );
}

/* ---------- skeletons ---------- */
export function Skeleton({ rows = 3, height = 18 }: { rows?: number; height?: number }) {
  return (
    <div className="skeleton" aria-busy="true" aria-label="Loading">
      {Array.from({ length: rows }, (_, i) => <span key={i} style={{ height, width: `${90 - i * 12}%` }} />)}
    </div>
  );
}

/* ---------- confidence meter ---------- */
export function ConfidenceMeter({ value, label = "Tracking" }: { value: number | null | undefined; label?: string }) {
  const v = value ?? 0;
  const level = value === null || value === undefined ? "none" : v >= 0.7 ? "high" : v >= 0.5 ? "mid" : "low";
  const text = { none: "—", high: "good", mid: "fair", low: "poor" }[level];
  return (
    <div className={"conf conf-" + level} data-testid="confidence">
      <span className="conf-label">{label}</span>
      <span className="conf-bar" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(v * 100)} aria-label={`${label} confidence`}>
        <i style={{ width: `${Math.round(v * 100)}%` }} />
      </span>
      <span className="conf-text">{value === null || value === undefined ? text : `${Math.round(v * 100)}% ${text}`}</span>
    </div>
  );
}

export const DISCLAIMER =
  "FormFit gives general exercise-technique feedback from a webcam. It is not medical advice and cannot assess injury risk. Stop if something hurts and ask a qualified professional.";
