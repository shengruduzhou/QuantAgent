import type { Icon } from "@phosphor-icons/react";
import {
  Atom,
  Brain,
  ChartLine,
  ChartLineUp,
  Database,
  FileText,
  Fire,
  Flask,
  Gauge,
  Gavel,
  Gear,
  Graph,
  HardDrives,
  ListBullets,
  ListMagnifyingGlass,
  Pulse,
  Question,
  ShieldCheck,
  ShieldStar,
  SlidersHorizontal,
  Sparkle,
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
  { id: "market-intelligence", path: "/market-intelligence", label: "市场情报台", caption: "Market Intelligence", group: "research", icon: Fire, keywords: "fuyao financial api market intelligence hot stock dragon tiger industry concept financial 行情 热榜 龙虎榜 行业 概念 财务" },
  { id: "market-playbooks", path: "/market-playbooks", label: "市场研究剧本", caption: "Fuyao Research Playbooks", group: "research", icon: Flask, keywords: "fuyao playbook research report breakout momentum reversal cashflow attention limitup dragon tiger 研究 剧本 报告 突破 动量 反转" },
  { id: "fund-research", path: "/funds", label: "基金研究", caption: "Fund / ETF / REITs", group: "research", icon: ChartLine, keywords: "fund etf reit nav holdings holders return 基金 净值 持仓 持有人 收益" },
  { id: "factor", path: "/factors", label: "因子实验室", caption: "Factor Lab", group: "research", icon: Atom, keywords: "factor alpha ic 因子" },
  { id: "fusion", path: "/fusion", label: "因子融合工场", caption: "Alpha Foundry", group: "research", icon: Sparkle, keywords: "fusion blend pareto frontier robustness pbo dsr 融合 前沿 稳健" },
  { id: "strategy-studio", path: "/strategy", label: "策略实验室", caption: "Strategy Studio", group: "research", icon: Strategy, keywords: "strategy authoring factor model portfolio backtest risk 策略 编写 回测 风控" },
  { id: "runs", path: "/runs", label: "研究运行", caption: "Runs & Conclusions", group: "research", icon: Gauge, keywords: "run job conclusion gate diagnosis retry 运行 结论 诊断 重试 闸门" },
  { id: "training", path: "/training", label: "训练实验室", caption: "Training Lab", group: "research", icon: HardDrives, keywords: "training experiment run gpu checkpoint 训练" },
  { id: "model", path: "/models", label: "模型注册表", caption: "Model Registry", group: "research", icon: Brain, keywords: "model registry prediction 模型" },
  { id: "backtest", path: "/backtests", label: "回测工作站", caption: "Backtester", group: "research", icon: Flask, keywords: "backtest strategy experiment 回测" },
  { id: "chart", path: "/stock-replay", label: "图表工作站", caption: "Chart Workstation", group: "trading", icon: ChartLine, keywords: "market chart kline replay stock k线 行情" },
  { id: "selection", path: "/selection", label: "选股决策", caption: "Selection Decisions", group: "trading", icon: SlidersHorizontal, keywords: "selection prediction target weights ranking 选股 决策" },
  { id: "t1", path: "/t-plus-one", label: "T+1 分析", caption: "Compliant Overlay", group: "trading", icon: SlidersHorizontal, keywords: "t+1 analysis position trading" },
  { id: "council", path: "/council", label: "决策议事会", caption: "Decision Council", group: "control", icon: Gavel, keywords: "council agent veto override audit evidence 议事会 否决 推翻 审计" },
  { id: "risk", path: "/risk", label: "风险管理", caption: "Risk Manager", group: "control", icon: ShieldCheck, keywords: "risk gate kill switch exposure 风控" },
  { id: "pipeline", path: "/runtime?view=lineage", label: "Pipeline", caption: "Lineage & Runs", group: "control", icon: Graph, keywords: "pipeline graph lineage task artifact" },
  { id: "tasks", path: "/settings?view=jobs", label: "任务中心", caption: "Task Center", group: "control", icon: ListBullets, keywords: "jobs task logs events 任务" },
  { id: "runtime", path: "/runtime", label: "Runtime", caption: "Artifact Inspector", group: "control", icon: Pulse, keywords: "runtime artifact catalog cleanup" },
  { id: "evidence", path: "/reports", label: "证据与报告", caption: "Evidence Center", group: "control", icon: FileText, keywords: "evidence report audit 证据 报告" },
  { id: "governance", path: "/governance", label: "运营治理", caption: "Operations Governance", group: "control", icon: ShieldStar, keywords: "governance shadow s4 u0 lineage readiness operations 治理 影子 就绪" },
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
