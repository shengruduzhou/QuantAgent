import type { Icon } from "@phosphor-icons/react";
import {
  Atom,
  Brain,
  ChartLine,
  ChartLineUp,
  Database,
  FileText,
  Flask,
  Gear,
  Graph,
  HardDrives,
  ListBullets,
  ListMagnifyingGlass,
  Pulse,
  Question,
  ShieldCheck,
  SlidersHorizontal,
  Strategy,
} from "@phosphor-icons/react";

export type VNextModuleGroup = "research" | "trading" | "control";

export interface VNextModule {
  id: string;
  path: string;
  label: string;
  caption: string;
  group: VNextModuleGroup;
  icon: Icon;
  keywords: string;
}

export const vnextModuleGroups: Array<{ id: VNextModuleGroup; label: string }> = [
  { id: "research", label: "RESEARCH" },
  { id: "trading", label: "TRADING" },
  { id: "control", label: "CONTROL" },
];

export const vnextModules: VNextModule[] = [
  { id: "dashboard", path: "/", label: "决策总览", caption: "Decision Dashboard", group: "research", icon: ChartLineUp, keywords: "dashboard overview portfolio model risk operations 总览" },
  { id: "data", path: "/runtime?view=data", label: "数据实验室", caption: "Data Lab", group: "research", icon: Database, keywords: "data dataset provider tickflow coverage quarantine 数据" },
  { id: "factor", path: "/factors", label: "因子实验室", caption: "Factor Lab", group: "research", icon: Atom, keywords: "factor alpha ic 因子" },
  { id: "training", path: "/training", label: "训练实验室", caption: "Training Lab", group: "research", icon: HardDrives, keywords: "training experiment run gpu checkpoint 训练" },
  { id: "model", path: "/models", label: "模型注册表", caption: "Model Registry", group: "research", icon: Brain, keywords: "model registry prediction 模型" },
  { id: "backtest", path: "/backtests", label: "回测工作站", caption: "Backtester", group: "research", icon: Flask, keywords: "backtest strategy experiment 回测" },
  { id: "chart", path: "/stock-replay", label: "图表工作站", caption: "Chart Workstation", group: "trading", icon: ChartLine, keywords: "market chart kline replay stock k线 行情" },
  { id: "strategy", path: "/selection", label: "策略与选股", caption: "Strategy Research", group: "trading", icon: Strategy, keywords: "selection prediction target weights strategy 选股 策略" },
  { id: "t1", path: "/t-plus-one", label: "T+1 分析", caption: "Compliant Overlay", group: "trading", icon: SlidersHorizontal, keywords: "t+1 analysis position trading" },
  { id: "risk", path: "/risk", label: "风险管理", caption: "Risk Manager", group: "control", icon: ShieldCheck, keywords: "risk gate kill switch exposure 风控" },
  { id: "pipeline", path: "/runtime?view=lineage", label: "Pipeline", caption: "Lineage & Runs", group: "control", icon: Graph, keywords: "pipeline graph lineage task artifact" },
  { id: "tasks", path: "/settings?view=jobs", label: "任务中心", caption: "Task Center", group: "control", icon: ListBullets, keywords: "jobs task logs events 任务" },
  { id: "runtime", path: "/runtime", label: "Runtime", caption: "Artifact Inspector", group: "control", icon: Pulse, keywords: "runtime artifact catalog cleanup" },
  { id: "evidence", path: "/reports", label: "证据与报告", caption: "Evidence Center", group: "control", icon: FileText, keywords: "evidence report audit 证据 报告" },
  { id: "parity", path: "/parity", label: "VN.PY 对齐", caption: "Capability Registry", group: "control", icon: ListMagnifyingGlass, keywords: "vnpy veighna parity capability" },
  { id: "settings", path: "/settings", label: "系统设置", caption: "System Settings", group: "control", icon: Gear, keywords: "settings system configuration 设置" },
  { id: "help", path: "/help", label: "帮助中心", caption: "Product Help", group: "control", icon: Question, keywords: "help keyboard guide 帮助" },
];

export function moduleForVNextPath(path: string): VNextModule {
  const target = new URL(path, window.location.origin);
  const candidates = vnextModules
    .map((module) => {
      const registered = new URL(module.path, window.location.origin);
      if (registered.pathname !== target.pathname) return null;
      const requiredParameters = [...registered.searchParams.entries()];
      if (requiredParameters.some(([key, value]) => target.searchParams.get(key) !== value)) return null;
      return { module, score: requiredParameters.length };
    })
    .filter((item): item is { module: VNextModule; score: number } => item !== null)
    .sort((left, right) => right.score - left.score);
  return candidates[0]?.module ?? vnextModules[0];
}

export function contextForPath(path: string) {
  const url = new URL(path, window.location.origin);
  const read = (key: string): string | undefined => url.searchParams.get(key) ?? undefined;
  return {
    symbol: read("symbol"),
    experiment: read("experiment"),
    run: read("run") ?? read("runId"),
    model: read("model") ?? read("modelId"),
    portfolio: read("portfolio"),
    account: read("account"),
    artifact: read("artifact") ?? read("query"),
  };
}
