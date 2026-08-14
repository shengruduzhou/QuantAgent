import { verdictTone } from "../utils/verdictTone";

interface StatusBadgeProps {
  status: string;
  label?: string;
}

export function StatusBadge({ status, label }: StatusBadgeProps): JSX.Element {
  // Negation-aware: plain substring matching rendered every U0 "..._NOT_READY_..."
  // state as success, because "not_ready".includes("ready") is true.
  const resolved = verdictTone(status);
  const tone = resolved === "unknown" ? "muted" : resolved;
  return (
    <span className={`status-badge status-${tone}`}>
      <i aria-hidden="true" />
      {label ?? status}
    </span>
  );
}
