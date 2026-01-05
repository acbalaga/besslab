export function Spinner() {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", color: "#4f46e5" }}>
      <div
        style={{
          width: "16px",
          height: "16px",
          borderRadius: "50%",
          border: "3px solid #c7d2fe",
          borderTopColor: "#4f46e5",
          animation: "spin 1s linear infinite",
        }}
      />
      <style>
        {`@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }`}
      </style>
      <span style={{ fontWeight: 600 }}>Loading...</span>
    </div>
  );
}
