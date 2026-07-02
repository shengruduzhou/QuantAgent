import { WarningCircle, Database, SpinnerGap } from "@phosphor-icons/react";

interface StateViewProps {
  state: "loading" | "empty" | "error";
  title?: string;
  detail?: string;
}

export function StateView({ state, title, detail }: StateViewProps): JSX.Element {
  const Icon = state === "loading" ? SpinnerGap : state === "error" ? WarningCircle : Database;
  const defaultTitle =
    state === "loading" ? "正在读取 runtime" : state === "error" ? "数据读取失败" : "暂无可用数据";
  return (
    <div className={`state-view state-${state}`}>
      <Icon size={24} weight="duotone" className={state === "loading" ? "spin" : ""} />
      <strong>{title ?? defaultTitle}</strong>
      <span>{detail ?? "该区域只展示已持久化的真实 QuantAgent artifact。"}</span>
    </div>
  );
}
