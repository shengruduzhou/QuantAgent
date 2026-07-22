import type { Icon } from "@phosphor-icons/react";
import {
  ArrowRight,
  Database,
  WarningCircle,
} from "@phosphor-icons/react";
import type { ReactNode } from "react";

export type WorkbenchTone = "neutral" | "info" | "positive" | "warning" | "danger" | "ai";

export interface WorkbenchMetric {
  label: string;
  value: string;
  detail?: string;
  tone?: WorkbenchTone;
  icon?: Icon;
}

export function WorkbenchHeader({
  eyebrow,
  title,
  description,
  asOf,
  context,
  actions,
}: {
  eyebrow: string;
  title: string;
  description: string;
  asOf?: string;
  context?: string;
  actions?: ReactNode;
}): JSX.Element {
  return (
    <header className="iw-header">
      <div className="iw-header-copy">
        <span>{eyebrow}</span>
        <h1>{title}</h1>
        <p>{description}</p>
      </div>
      <div className="iw-header-context">
        {actions ? <div className="iw-header-actions">{actions}</div> : null}
        {asOf || context ? <div><strong>{asOf ?? "runtime"}</strong><span>{context ?? "source-backed context"}</span></div> : null}
      </div>
    </header>
  );
}

export function WorkbenchMetricStrip({ metrics }: { metrics: WorkbenchMetric[] }): JSX.Element {
  return (
    <section className={`iw-metric-strip iw-metrics-${Math.min(6, metrics.length)}`} aria-label="决策指标">
      {metrics.map((metric) => {
        const MetricIcon = metric.icon;
        return (
          <article className={`iw-metric tone-${metric.tone ?? "neutral"}`} key={metric.label}>
            <header><span>{metric.label}</span>{MetricIcon ? <MetricIcon size={15} weight="duotone" /> : null}</header>
            <strong>{metric.value}</strong>
            <small>{metric.detail ?? "persisted runtime"}</small>
          </article>
        );
      })}
    </section>
  );
}

export function WorkbenchPanel({
  eyebrow,
  title,
  meta,
  actions,
  className = "",
  children,
}: {
  eyebrow?: string;
  title: string;
  meta?: string;
  actions?: ReactNode;
  className?: string;
  children: ReactNode;
}): JSX.Element {
  return (
    <section className={`iw-panel ${className}`.trim()}>
      <header className="iw-panel-header">
        <div>{eyebrow ? <span>{eyebrow}</span> : null}<h2>{title}</h2>{meta ? <small>{meta}</small> : null}</div>
        {actions ? <div className="iw-panel-actions">{actions}</div> : null}
      </header>
      <div className="iw-panel-body">{children}</div>
    </section>
  );
}

export function SegmentedTabs<T extends string>({
  items,
  active,
  onChange,
  label,
}: {
  items: Array<{ id: T; label: string; count?: number }>;
  active: T;
  onChange: (value: T) => void;
  label: string;
}): JSX.Element {
  return (
    <nav className="iw-segmented-tabs" aria-label={label}>
      {items.map((item) => (
        <button type="button" key={item.id} className={active === item.id ? "active" : ""} aria-pressed={active === item.id} onClick={() => onChange(item.id)}>
          {item.label}{item.count !== undefined ? <span>{item.count}</span> : null}
        </button>
      ))}
    </nav>
  );
}

export function ActionableState({
  title,
  detail,
  icon: StateIcon = Database,
  tone = "neutral",
  primary,
  secondary,
  compact = false,
}: {
  title: string;
  detail: string;
  icon?: Icon;
  tone?: WorkbenchTone;
  primary?: { label: string; onClick: () => void };
  secondary?: { label: string; onClick: () => void };
  compact?: boolean;
}): JSX.Element {
  return (
    <div className={`iw-actionable-state tone-${tone} ${compact ? "compact" : ""}`}>
      <StateIcon size={compact ? 19 : 25} weight="duotone" />
      <div><strong>{title}</strong><p>{detail}</p></div>
      {primary || secondary ? <div className="iw-state-actions">
        {primary ? <button type="button" className="iw-primary-action" onClick={primary.onClick}>{primary.label}<ArrowRight size={13} /></button> : null}
        {secondary ? <button type="button" onClick={secondary.onClick}>{secondary.label}</button> : null}
      </div> : null}
    </div>
  );
}

export function TruthNotice({ children, tone = "info" }: { children: ReactNode; tone?: WorkbenchTone }): JSX.Element {
  return <div className={`iw-truth-notice tone-${tone}`}><WarningCircle size={15} weight="duotone" /><span>{children}</span></div>;
}
