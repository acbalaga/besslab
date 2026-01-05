interface PillProps {
  label: string;
  tone?: "info" | "success" | "neutral";
}

export function Pill({ label, tone = "neutral" }: PillProps) {
  const colors: Record<typeof tone, string> = {
    neutral: "#e2e8f0",
    info: "#c7d2fe",
    success: "#bbf7d0",
  };
  const text: Record<typeof tone, string> = {
    neutral: "#0f172a",
    info: "#312e81",
    success: "#166534",
  };

  return (
    <span
      style={{
        background: colors[tone],
        color: text[tone],
        borderRadius: "999px",
        padding: "0.2rem 0.75rem",
        fontSize: "0.85rem",
        border: "1px solid rgba(15,23,42,0.05)",
      }}
    >
      {label}
    </span>
  );
}
