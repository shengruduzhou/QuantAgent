import type { Icon } from "@phosphor-icons/react";
import { valueTone } from "../utils/format";

interface MetricCardProps {
  label: string;
  value: string;
  delta?: number | null;
  detail?: string;
  icon?: Icon;
  tone?: "positive" | "negative" | "warning" | "neutral";
}

export function MetricCard({
  label,
  value,
  delta,
  detail,
  icon: MetricIcon,
  tone,
}: MetricCardProps): JSX.Element {
  const resolvedTone = tone ?? valueTone(delta);
  return (
    <article className={`metric-card metric-${resolvedTone}`}>
      <div className="metric-card-top">
        <span>{label}</span>
        {MetricIcon ? <MetricIcon size={17} weight="duotone" aria-hidden="true" /> : null}
      </div>
      <strong>{value}</strong>
      <div className="metric-card-foot">
        {delta !== null && delta !== undefined ? (
          <span className={`tone-${valueTone(delta)}`}>
            {delta > 0 ? "+" : ""}
            {(delta * 100).toFixed(2)}%
          </span>
        ) : null}
        <small>{detail ?? "真实 runtime"}</small>
      </div>
    </article>
  );
}
