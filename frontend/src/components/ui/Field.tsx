import { InputHTMLAttributes, ReactNode } from "react";

interface FieldProps extends InputHTMLAttributes<HTMLInputElement> {
  label: string;
  helperText?: ReactNode;
}

export function Field({ label, helperText, ...rest }: FieldProps) {
  return (
    <label style={{ display: "flex", flexDirection: "column", gap: "0.35rem", fontWeight: 600 }}>
      <span>
        {label}
        {rest.required && <span style={{ color: "#ef4444" }}> *</span>}
      </span>
      <input
        {...rest}
        style={{
          padding: "0.65rem 0.85rem",
          borderRadius: "10px",
          border: "1px solid #cbd5e1",
          fontSize: "0.95rem",
          outline: "none",
          boxShadow: "0 2px 4px rgba(15,23,42,0.04)",
        }}
      />
      {helperText && <small style={{ color: "#475569", fontWeight: 400 }}>{helperText}</small>}
    </label>
  );
}
