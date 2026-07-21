import { ClockCounterClockwise, CloudSlash, WarningCircle, Database, SpinnerGap } from "@phosphor-icons/react";

type StateViewState = "loading" | "empty" | "error" | "stale" | "unavailable";

interface StateViewProps {
  state: StateViewState;
  title?: string;
  detail?: string;
}

export function StateView({ state, title, detail }: StateViewProps): JSX.Element {
  const Icon = state === "loading" ? SpinnerGap
    : state === "error" ? WarningCircle
      : state === "stale" ? ClockCounterClockwise
        : state === "unavailable" ? CloudSlash
          : Database;
  const defaultTitle = state === "loading" ? "正在读取 runtime"
    : state === "error" ? "数据读取失败"
      : state === "stale" ? "数据已过期"
        : state === "unavailable" ? "数据当前不可用"
          : "暂无可用数据";
  return (
    <div className={`state-view state-${state}`}>
      <Icon size={24} weight="duotone" className={state === "loading" ? "spin" : ""} />
      <strong>{title ?? defaultTitle}</strong>
      <span>{detail ?? "该区域只展示已持久化的真实 QuantAgent artifact。"}</span>
    </div>
  );
}
