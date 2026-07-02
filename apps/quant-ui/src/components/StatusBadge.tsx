interface StatusBadgeProps {
  status: string;
  label?: string;
}

export function StatusBadge({ status, label }: StatusBadgeProps): JSX.Element {
  const normalized = status.toLowerCase();
  const tone =
    normalized.includes("ready") || normalized.includes("normal") || normalized.includes("success")
      ? "success"
      : normalized.includes("warn") || normalized.includes("partial")
        ? "warning"
        : normalized.includes("error") || normalized.includes("fail")
          ? "danger"
          : "muted";
  return (
    <span className={`status-badge status-${tone}`}>
      <i aria-hidden="true" />
      {label ?? status}
    </span>
  );
}
