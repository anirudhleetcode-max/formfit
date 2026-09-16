import { lazy, Suspense } from "react";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { ToastProvider } from "./components/Feedback";
import Shell from "./components/Shell";
import { AuthProvider, useAuth } from "./lib/auth";
import History from "./pages/History";
import Login from "./pages/Login";

// pose runtime (MediaPipe) and charts (recharts) load only when their screens open
const Train = lazy(() => import("./pages/Train"));
const Upload = lazy(() => import("./pages/Upload"));
const Progress = lazy(() => import("./pages/Progress"));
const SessionDetail = lazy(() => import("./pages/SessionDetail"));
const About = lazy(() => import("./pages/About"));
const wrap = (el: React.ReactNode) => <Suspense fallback={<div className="page muted">Loading…</div>}>{el}</Suspense>;

function Protected() {
  const { user, loading } = useAuth();
  if (loading) return <div className="boot">Loading…</div>;
  if (!user) return <Navigate to="/login" replace />;
  return <Shell />;
}

export default function App() {
  return (
    <AuthProvider>
      <ToastProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route element={<Protected />}>
            <Route index element={wrap(<Train />)} />
            <Route path="upload" element={wrap(<Upload />)} />
            <Route path="history" element={<History />} />
            <Route path="history/:id" element={wrap(<SessionDetail />)} />
            <Route path="progress" element={wrap(<Progress />)} />
            <Route path="about" element={wrap(<About />)} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
      </ToastProvider>
    </AuthProvider>
  );
}
