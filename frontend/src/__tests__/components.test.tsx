// @vitest-environment jsdom
import "./setup";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { createRef } from "react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { ConfidenceMeter, ConfirmDialog, ToastProvider } from "../components/Feedback";
import { EMPTY_HUD, Stage, type HudState } from "../components/Stage";
import { AuthProvider } from "../lib/auth";
import type { RepResult } from "../pose/engine";
import About from "../pages/About";
import Login from "../pages/Login";
import SessionDetail from "../pages/SessionDetail";

const feats = {
  minAngle: 60, maxAngle: 175, rom: 115, eccS: 1.2, conS: 0.9, torsoLeanMax: 20, torsoSway: 15,
  hipLineMin: 175, valgusMin: 1, elbowDriftMax: 10, depth: 0.1, heelRise: 0, hipOffsetAtWorst: 0,
};
const rep = (over: Partial<RepResult>): RepResult => ({
  n: 1, set: 1, score: 85, faults: ["heel_lift"], confidence: 0.9, scored: true, abstainReason: null,
  eccS: 1.2, conS: 0.9, rom: 115, t: 3, features: feats, ...over,
});

function renderStage(hud: Partial<HudState>) {
  return render(
    <Stage videoRef={createRef()} canvasRef={createRef()} mirrored={false} hud={{ ...EMPTY_HUD, tracking: "tracking", ...hud }} />,
  );
}

function jsonResponse(body: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }));
}

describe("Stage HUD", () => {
  it("shows the score and faults of a scored rep", () => {
    renderStage({ last: rep({}), confidence: 0.82, setReps: 3 });
    expect(screen.getByTestId("rep-count")).toHaveTextContent("3");
    expect(screen.getByTestId("last-score")).toHaveTextContent("85");
    expect(screen.getByText("Heels lifting")).toBeInTheDocument();
    expect(screen.getByRole("meter", { name: /tracking confidence/i })).toHaveAttribute("aria-valuenow", "82");
  });

  it("abstains visibly when a rep was not scored", () => {
    renderStage({ last: rep({ score: null, faults: [], scored: false, confidence: 0.3, abstainReason: "Can't see your knees clearly" }), confidence: 0.3 });
    expect(screen.getByTestId("last-score")).toHaveTextContent("—");
    expect(screen.getByTestId("last-rep-feedback")).toHaveTextContent("Rep counted, form not scored: Can't see your knees clearly");
    expect(screen.queryByText("Clean rep")).not.toBeInTheDocument();
    expect(screen.getByTestId("confidence")).toHaveClass("conf-low");
  });

  it("flags weak tracking in the status pill", () => {
    renderStage({ lowConfidence: true, message: "Can't see your ankles clearly", confidence: 0.2 });
    expect(screen.getByTestId("tracking")).toHaveTextContent("Can't see your ankles clearly");
    expect(screen.getByTestId("tracking")).toHaveClass("t-weak");
  });
});

describe("ConfidenceMeter", () => {
  it.each([[0.9, "conf-high", "90% good"], [0.55, "conf-mid", "55% fair"], [0.2, "conf-low", "20% poor"], [null, "conf-none", "—"]])(
    "renders %s as %s", (v, cls, text) => {
      render(<ConfidenceMeter value={v} />);
      expect(screen.getByTestId("confidence")).toHaveClass(cls);
      expect(screen.getByTestId("confidence")).toHaveTextContent(text);
    });
});

describe("ConfirmDialog", () => {
  it("calls confirm and cancel", () => {
    const ok = vi.fn(), no = vi.fn();
    render(<ConfirmDialog open title="Delete?" body="Gone for good" confirmLabel="Delete" onConfirm={ok} onCancel={no} danger />);
    fireEvent.click(screen.getByRole("button", { name: "Delete" }));
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(ok).toHaveBeenCalledOnce();
    expect(no).toHaveBeenCalledOnce();
  });
});

describe("Login", () => {
  function renderLogin() {
    return render(<AuthProvider><MemoryRouter><Login /></MemoryRouter></AuthProvider>);
  }

  it("fills the demo credentials and shows the disclaimer", () => {
    renderLogin();
    fireEvent.click(screen.getByRole("button", { name: "Use demo account" }));
    expect(screen.getByLabelText("Email")).toHaveValue("demo@formfit.app");
    expect(screen.getByLabelText("Password")).toHaveValue("demo1234");
    expect(screen.getByText(/not medical advice/i)).toBeInTheDocument();
  });

  it("shows the server error on a failed sign-in", async () => {
    vi.stubGlobal("fetch", vi.fn(() => jsonResponse({ detail: "Incorrect email or password" }, 401)));
    renderLogin();
    fireEvent.change(screen.getByLabelText("Email"), { target: { value: "a@b.co" } });
    fireEvent.change(screen.getByLabelText("Password"), { target: { value: "wrongpw" } });
    fireEvent.click(screen.getByRole("button", { name: "Sign in" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Incorrect email or password");
  });

  it("requires a name and a 6+ character password when registering", () => {
    renderLogin();
    fireEvent.click(screen.getByRole("button", { name: /create an account/i }));
    expect(screen.getByLabelText("Name")).toBeRequired();
    expect(screen.getByLabelText("Password")).toHaveAttribute("minLength", "6");
  });
});

const session = {
  id: "s1", exercise: "squat", source: "upload", status: "done", demo: true,
  started_at: "2026-09-10T08:00:00Z", duration_s: 60,
  summary: { total_reps: 3, sets: 1, avg_score: 85, avg_model_score: 70, best_set_reps: 3, clean_reps: 1,
    avg_ecc_s: 1.1, avg_con_s: 0.9, top_faults: [], scored_reps: 2, unscored_reps: 1, avg_confidence: 0.64 },
  fatigue: { detected: false, onset_rep: null, shift: 0, slopes: { con_s: 0, rom: 0, score: 0 }, min_reps: 6 },
  sets: [{ set: 1, reps: 3, avg_score: 85, scored_reps: 2 }],
  reps: [
    { i: 1, set: 1, score: 100, faults: [], ecc_s: 1, con_s: 1, rom: 120, t: 2, features: {}, confidence: 0.9, scored: true,
      model_score: 81, model_status: "ok", model_confidence: 0.81 },
    { i: 2, set: 1, score: 70, faults: ["shallow_depth"], ecc_s: 1, con_s: 1, rom: 90, t: 5, features: {}, confidence: 0.8, scored: true,
      model_score: null, model_status: "out_of_distribution", model_ood: ["ecc_s"] },
    { i: 3, set: 1, score: null, faults: [], ecc_s: 1, con_s: 1, rom: 60, t: 8, features: {}, confidence: 0.22, scored: false,
      abstain_reason: "Can't see your knees clearly", model_score: null, model_status: "not_scored" },
  ],
};

function renderDetail() {
  return render(
    <AuthProvider><ToastProvider><MemoryRouter initialEntries={["/history/s1"]}>
      <Routes><Route path="/history/:id" element={<SessionDetail />} /></Routes>
    </MemoryRouter></ToastProvider></AuthProvider>,
  );
}

describe("SessionDetail", () => {
  it("shows unscored reps, model abstentions, demo label and tracking quality", async () => {
    vi.stubGlobal("fetch", vi.fn(() => jsonResponse(session)));
    renderDetail();
    expect(await screen.findByTestId("detail-reps")).toHaveTextContent("3");
    expect(screen.getByText("demo data")).toBeInTheDocument();
    expect(screen.getByTestId("tracking-quality")).toHaveTextContent("1 of 3 reps were counted but not scored");
    const table = screen.getByTestId("rep-table");
    expect(table).toHaveTextContent("Not scored: Can't see your knees clearly");
    expect(table).toHaveTextContent("outside training range");
    expect(table).toHaveTextContent("22%");
    expect(screen.getByText("synthetic-trained")).toBeInTheDocument();
  });

  it("renders the API error instead of crashing", async () => {
    vi.stubGlobal("fetch", vi.fn(() => jsonResponse({ detail: "Session not found" }, 404)));
    renderDetail();
    expect(await screen.findByRole("alert")).toHaveTextContent("Session not found");
    expect(screen.getByRole("link", { name: "Back to history" })).toBeInTheDocument();
  });

  it("asks for confirmation before deleting", async () => {
    const fetchMock = vi.fn(() => jsonResponse(session));
    vi.stubGlobal("fetch", fetchMock);
    renderDetail();
    await screen.findByTestId("detail-reps");
    fireEvent.click(screen.getByRole("button", { name: /delete/i }));
    expect(screen.getByText("Delete this session?")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(fetchMock).toHaveBeenCalledTimes(1); // only the initial GET
  });
});

describe("About / model panel", () => {
  it("labels the model as synthetic-trained", async () => {
    vi.stubGlobal("fetch", vi.fn(() => jsonResponse({
      ready: true, error: null, sklearn_version: "1.8.0", synthetic: true, training_data: "SYNTHETIC reps from the simulator",
      card: null, metrics: { n_train: 6000, n_test: 2000, exercises: { squat: {
        selected: "gradient_boosting", model: { accuracy: 0.9, f1_faulty: 0.93, roc_auc: 0.95, ece: 0.03 },
        rule_baseline: { accuracy: 0.83, f1_faulty: 0.88, roc_auc: 0.87 }, selective: { min_confidence: 0.8, coverage: 0.75, accuracy_on_covered: 0.96 },
      } } },
    })));
    render(<MemoryRouter><About /></MemoryRouter>);
    await waitFor(() => expect(screen.getByTestId("model-panel")).toHaveTextContent("Trained on synthetic data."));
    expect(screen.getByTestId("model-panel")).toHaveTextContent("0.950");
    expect(screen.getByTestId("disclaimer")).toHaveTextContent("not medical advice");
  });

  it("explains the degraded state when the model is not loaded", async () => {
    vi.stubGlobal("fetch", vi.fn(() => jsonResponse({ ready: false, error: "FileNotFoundError", sklearn_version: "1.8.0", metrics: null })));
    render(<MemoryRouter><About /></MemoryRouter>);
    expect(await screen.findByRole("alert")).toHaveTextContent("not loaded (FileNotFoundError)");
  });

  it("says real-video accuracy is not measured when no evaluation was run", async () => {
    vi.stubGlobal("fetch", vi.fn(() => jsonResponse({ ready: true, error: null, sklearn_version: "1.8.0", metrics: null, real_clip_eval: null })));
    render(<MemoryRouter><About /></MemoryRouter>);
    await waitFor(() => expect(screen.getByTestId("real-eval")).toHaveTextContent("Not measured yet."));
  });

  it("shows measured rep-count numbers from the real-clip run", async () => {
    vi.stubGlobal("fetch", vi.fn(() => jsonResponse({
      ready: true, error: null, sklearn_version: "1.8.0", metrics: null,
      real_clip_eval: {
        run_id: "20260101-000000-real_clips_repcount", note: "tiny",
        dataset: { name: "formfit-real-clips", synthetic: false, n_clips_listed: 3 },
        overall: { n_clips: 3, gt_reps_total: 20, pred_reps_total: 19, mae: 0.333, exact_match_rate: 0.667, within_1_rate: 1, scored_share_of_predicted: 0.9 },
        per_exercise: {},
        model_on_real_reps: { reps: 19, status_counts: { ok: 10, out_of_distribution: 9 }, ood_features: {} },
      },
    })));
    render(<MemoryRouter><About /></MemoryRouter>);
    const panel = await screen.findByTestId("real-eval");
    await waitFor(() => expect(panel).toHaveTextContent("19 / 20"));
    expect(panel).toHaveTextContent("3 public video clips");
    expect(panel).toHaveTextContent("67% of clips");
    expect(panel).toHaveTextContent("9 out of distribution");
  });
});
