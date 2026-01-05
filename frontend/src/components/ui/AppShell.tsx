import { ReactNode } from "react";
import { Link, NavLink } from "react-router-dom";
import { useUploads } from "../../state/UploadContext";
import { Pill } from "./Pill";
import { Button } from "./Button";
import { API_BASE_URL } from "../../api/client";

const navItems = [
  { to: "/", label: "Landing / Uploads" },
  { to: "/inputs", label: "Inputs & Results" },
  { to: "/sweep", label: "BESS sizing sweep" },
  { to: "/batch", label: "Multi-scenario batch" },
];

export function AppShell({ children }: { children: ReactNode }) {
  const { uploads } = useUploads();

  return (
    <div>
      <header
        style={{
          background: "white",
          borderBottom: "1px solid #e2e8f0",
          boxShadow: "0 4px 20px rgba(15,23,42,0.05)",
          position: "sticky",
          top: 0,
          zIndex: 10,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "0.75rem 1.25rem", maxWidth: "1280px", margin: "0 auto" }}>
          <Link to="/" style={{ fontWeight: 700, letterSpacing: 0.2, color: "#312e81" }}>
            BESSLab Web
          </Link>
          <nav style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center" }}>
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                style={({ isActive }) => ({
                  padding: "0.45rem 0.85rem",
                  borderRadius: "10px",
                  fontWeight: 600,
                  color: isActive ? "white" : "#0f172a",
                  background: isActive ? "#4f46e5" : "transparent",
                  textDecoration: "none",
                })}
              >
                {item.label}
              </NavLink>
            ))}
            <Button asChild variant="ghost" size="sm">
              <a href={`${API_BASE_URL}/docs`} rel="noreferrer" target="_blank">
                API docs
              </a>
            </Button>
          </nav>
        </div>
        {uploads && (uploads.pvId || uploads.cycleId) ? (
          <div style={{ padding: "0.35rem 1.25rem", background: "#eef2ff", borderTop: "1px solid #e0e7ff" }}>
            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", alignItems: "center", maxWidth: "1280px", margin: "0 auto" }}>
              <span style={{ fontWeight: 600, color: "#312e81" }}>Cached uploads:</span>
              {uploads.pvId && <Pill label={`PV: ${uploads.pvId}`} tone="info" />}
              {uploads.cycleId && <Pill label={`Cycle: ${uploads.cycleId}`} tone="success" />}
            </div>
          </div>
        ) : null}
      </header>
      <main>{children}</main>
    </div>
  );
}
