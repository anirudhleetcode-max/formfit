import { History, Info, LineChart, LogOut, ScanLine, Upload } from "lucide-react";
import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../lib/auth";

const NAV = [
  { to: "/", label: "Train", icon: ScanLine, end: true },
  { to: "/upload", label: "Upload", icon: Upload },
  { to: "/history", label: "History", icon: History },
  { to: "/progress", label: "Progress", icon: LineChart },
  { to: "/about", label: "About", icon: Info },
];

export function Logo() {
  return (
    <span className="logo" aria-label="FormFit">
      <svg width="22" height="22" viewBox="0 0 24 24" aria-hidden="true">
        <rect x="1" y="1" width="22" height="22" rx="3" fill="var(--accent)" />
        <path d="M7 18V6h10M7 12h7" stroke="#0f1012" strokeWidth="2.6" fill="none" strokeLinecap="square" />
      </svg>
      <span>FORMFIT</span>
    </span>
  );
}

export default function Shell() {
  const { user, logout } = useAuth();
  return (
    <div className="shell">
      <aside className="side">
        <Logo />
        <nav className="nav" aria-label="Main">
          {NAV.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} aria-label={label} className={({ isActive }) => "nav-link" + (isActive ? " active" : "")}>
              <Icon size={18} strokeWidth={2} aria-hidden="true" />
              <span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="side-foot">
          <div className="who" title={user?.email}>
            <span className="who-name">{user?.name}</span>
            <span className="who-mail">{user?.email}</span>
          </div>
          <button className="icon-btn" onClick={logout} aria-label="Sign out" title="Sign out">
            <LogOut size={17} />
          </button>
        </div>
      </aside>
      <main className="main">
        <Outlet />
        <footer className="app-foot">
          Not medical advice. FormFit gives general technique feedback and cannot assess injury risk.{" "}
          <NavLink to="/about">How it works</NavLink>
        </footer>
      </main>
    </div>
  );
}
