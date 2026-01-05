interface NoticeProps {
  title?: string;
  message: string;
  tone?: "info" | "error" | "success";
}

export function Notice({ title, message, tone = "info" }: NoticeProps) {
  const classes: Record<typeof tone, string> = {
    info: "alert info",
    error: "alert error",
    success: "alert success",
  };

  return (
    <div className={classes[tone]}>
      {title && <strong style={{ display: "block", marginBottom: "0.35rem" }}>{title}</strong>}
      <span>{message}</span>
    </div>
  );
}
