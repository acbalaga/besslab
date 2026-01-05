import { ReactNode } from "react";
import { Card } from "./ui/Card";

interface KpiCardProps {
  label: string;
  value: ReactNode;
  helper?: string;
  tone?: "neutral" | "good" | "warn";
}

export function KpiCard({ label, value, helper, tone = "neutral" }: KpiCardProps) {
  const toneColor = tone === "good" ? "#166534" : tone === "warn" ? "#92400e" : "#0f172a";
  const badgeBg = tone === "good" ? "#ecfdf3" : tone === "warn" ? "#fffbeb" : "#f8fafc";

  return (
    <Card padded>
      <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
        <span className="text-subtle" style={{ fontWeight: 600 }}>
          {label}
        </span>
        <div style={{ fontSize: "1.4rem", fontWeight: 700, color: toneColor }}>{value}</div>
        {helper && (
          <span
            style={{
              background: badgeBg,
              color: toneColor,
              borderRadius: "8px",
              padding: "0.4rem 0.55rem",
              fontSize: "0.9rem",
            }}
          >
            {helper}
          </span>
        )}
      </div>
    </Card>
  );
}
