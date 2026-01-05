import { ReactNode, useState } from "react";

interface TabDef {
  id: string;
  label: string;
  content: ReactNode;
}

interface TabsProps {
  tabs: TabDef[];
  defaultTab?: string;
}

export function Tabs({ tabs, defaultTab }: TabsProps) {
  const [active, setActive] = useState(defaultTab || tabs[0]?.id);
  const current = tabs.find((tab) => tab.id === active) || tabs[0];

  return (
    <div>
      <div style={{ display: "flex", gap: "0.35rem", flexWrap: "wrap", marginBottom: "0.75rem" }}>
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActive(tab.id)}
            style={{
              border: "1px solid #cbd5e1",
              borderRadius: "10px",
              padding: "0.45rem 0.9rem",
              background: tab.id === current?.id ? "#4f46e5" : "white",
              color: tab.id === current?.id ? "white" : "#0f172a",
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            {tab.label}
          </button>
        ))}
      </div>
      <div>{current?.content}</div>
    </div>
  );
}
