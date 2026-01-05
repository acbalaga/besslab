import { ButtonHTMLAttributes, ReactNode } from "react";
import { LinkProps } from "react-router-dom";
import { clsx } from "clsx";

interface CommonProps {
  children: ReactNode;
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md" | "lg";
  asChild?: boolean;
}

type ButtonProps = CommonProps & ButtonHTMLAttributes<HTMLButtonElement>;

type AnchorProps = CommonProps & LinkProps & { href?: string };

type CombinedProps = ButtonProps | AnchorProps;

export function Button(props: CombinedProps) {
  const { children, variant = "primary", size = "md", asChild, ...rest } = props as ButtonProps;
  const classes = clsx("button", variant, size);

  if (asChild && "href" in (props as AnchorProps)) {
    const anchorProps = props as AnchorProps;
    return (
      <a className={classes} {...anchorProps}>
        {children}
      </a>
    );
  }

  return (
    <button className={classes} {...(rest as ButtonProps)}>
      {children}
    </button>
  );
}

if (typeof document !== "undefined" && !document.getElementById("besslab-button-styles")) {
  const styles = document.createElement("style");
  styles.id = "besslab-button-styles";
  styles.innerHTML = `
.button {
  border: none;
  border-radius: 10px;
  font-weight: 600;
  cursor: pointer;
  transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
  font-size: 0.95rem;
}
.button:disabled { opacity: 0.7; cursor: not-allowed; }
.button:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(79,70,229,0.2); }
.button.primary { background: linear-gradient(135deg,#4f46e5,#6366f1); color: white; padding: 0.6rem 1rem; }
.button.secondary { background: white; color: #312e81; border: 1px solid #cbd5e1; padding: 0.55rem 1rem; }
.button.ghost { background: transparent; color: #312e81; padding: 0.55rem 0.9rem; }
.button.sm { font-size: 0.9rem; padding: 0.45rem 0.75rem; }
.button.lg { font-size: 1rem; padding: 0.85rem 1.25rem; }
`;
  document.head.appendChild(styles);
}
