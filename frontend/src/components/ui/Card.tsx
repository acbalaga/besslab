import { ReactNode } from "react";

interface CardProps {
  title?: string;
  description?: string;
  children: ReactNode;
  actions?: ReactNode;
  padded?: boolean;
}

export function Card({ title, description, children, actions, padded = true }: CardProps) {
  return (
    <div
      style={{
        background: "white",
        borderRadius: "14px",
        padding: padded ? "1.1rem" : 0,
        boxShadow: "0 12px 32px rgba(15,23,42,0.08)",
        border: "1px solid #e2e8f0",
      }}
    >
      {(title || description || actions) && (
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: padded ? "0.75rem" : "1rem", padding: padded ? 0 : "1rem" }}>
          <div>
            {title && <div style={{ fontWeight: 700, marginBottom: description ? "0.25rem" : 0 }}>{title}</div>}
            {description && <div style={{ color: "#475569", fontSize: "0.95rem" }}>{description}</div>}
          </div>
          {actions}
        </div>
      )}
      {children}
    </div>
  );
}
