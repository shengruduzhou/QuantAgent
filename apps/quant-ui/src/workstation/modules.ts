import type { Icon } from "@phosphor-icons/react";
import {
  ArrowsClockwise,
  Atom,
  Brain,
  ChartLine,
  ChartLineUp,
  Database,
  FileText,
  Flask,
  Funnel,
  Gear,
  ListMagnifyingGlass,
  ShieldCheck,
} from "@phosphor-icons/react";

export type ModuleGroup = "monitor" | "research" | "execution" | "system";

export interface WorkstationModule {
  path: string;
  label: string;
  caption: string;
  group: ModuleGroup;
  icon: Icon;
  keywords: string;
}

export const moduleGroups: Array<{ id: ModuleGroup; label: string }> = [
  { id: "monitor", label: "MONITOR" },
  { id: "research", label: "RESEARCH" },
  { id: "execution", label: "EXECUTION" },
  { id: "system", label: "SYSTEM" },
];

export const workstationModules: WorkstationModule[] = [
  { path: "/", label: "总览", caption: "Command Center", group: "monitor", icon: ChartLineUp, keywords: "dashboard overview 总览" },
  { path: "/stock-replay", label: "市场复盘", caption: "Stock Replay", group: "monitor", icon: ChartLine, keywords: "stock replay 股票 kline" },
  { path: "/selection", label: "选股研究", caption: "Selection", group: "research", icon: Funnel, keywords: "selection ranking 选股" },
  { path: "/factors", label: "因子研究", caption: "Factor Lab", group: "research", icon: Atom, keywords: "factor 因子 ic" },
  { path: "/models", label: "模型实验", caption: "Model Lab", group: "research", icon: Brain, keywords: "model training prediction 模型" },
  { path: "/backtests", label: "回测实验", caption: "Backtest Lab", group: "research", icon: Flask, keywords: "backtest 回测 experiment" },
  { path: "/reports", label: "研究报告", caption: "Reports", group: "research", icon: FileText, keywords: "report evidence 报告" },
  { path: "/t-plus-one", label: "T+1 做 T", caption: "Compliant Overlay", group: "execution", icon: ArrowsClockwise, keywords: "t+1 do t analysis 做T" },
  { path: "/risk", label: "风险监控", caption: "Risk Center", group: "execution", icon: ShieldCheck, keywords: "risk 风控 exposure kill switch" },
  { path: "/runtime", label: "Runtime / Data", caption: "Artifact Manager", group: "system", icon: Database, keywords: "runtime data artifact lineage catalog" },
  { path: "/parity", label: "VN.PY 对齐", caption: "Capability Registry", group: "system", icon: ListMagnifyingGlass, keywords: "vnpy parity capability registry 对齐 能力" },
  { path: "/settings", label: "系统控制", caption: "Jobs & Settings", group: "system", icon: Gear, keywords: "settings control job system" },
];

export function moduleForPath(pathname: string): WorkstationModule {
  return workstationModules.find((item) => item.path === pathname) ?? workstationModules[0];
}
