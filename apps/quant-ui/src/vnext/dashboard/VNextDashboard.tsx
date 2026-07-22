import { useMemo, useState } from "react";
import type { BacktestSummary, EquityPoint, JobSummary, Page, RiskOverview, SystemOverview, Trade } from "../../api/types";
import { useApi } from "../../hooks/useApi";
import { StateView } from "../../components/StateView";
import { formatDate } from "../../utils/format";
import { ActionQueue } from "./ActionQueue";
import { DecisionStateStrip } from "./DecisionStateStrip";
import { PrimaryDecisionCanvas } from "./PrimaryDecisionCanvas";
import type { ActionQueueItem, DecisionView, RiskRuleView } from "./types";

export function VNextDashboard(): JSX.Element {
  const [view, setView] = useState<DecisionView>("portfolio");
  const overview = useApi<SystemOverview>(["vnext-dashboard-overview"], "/system/overview", undefined, { refetchInterval: 15_000, staleTime: 10_000 });
  const jobs = useApi<JobSummary[]>(["global-activity-jobs"], "/jobs", undefined, { refetchInterval: 5_000, staleTime: 2_000 });
  const backtests = useApi<BacktestSummary[]>(["vnext-dashboard-backtests"], "/backtests");
  const data = overview.data?.data;
  const latestBacktest = data?.latestBacktest ?? backtests.data?.data[0];
  const equity = useApi<EquityPoint[]>(["vnext-dashboard-equity", latestBacktest?.id], latestBacktest ? `/backtests/${latestBacktest.id}/equity` : null);
  const trades = useApi<Page<Trade>>(["vnext-dashboard-trades", latestBacktest?.id], latestBacktest ? `/backtests/${latestBacktest.id}/trades` : null, { pageSize: 80 });
  const risk = useApi<RiskOverview>(["vnext-dashboard-risk", latestBacktest?.id], "/risk/overview", { backtestId: latestBacktest?.id });

  const jobItems = jobs.data?.data ?? [];
  const riskData = risk.data?.data ?? data?.risk;
  const latestPoint = equity.data?.data.at(-1);
  const riskEventCount = Object.values(riskData?.eventCounts ?? {}).reduce((sum, count) => sum + count, 0);

  const riskRules = useMemo<RiskRuleView[]>(() => {
    if (!riskData) return [];
    return (riskData.rules ?? []).map((rule) => {
    const id = String(rule.id ?? rule.name ?? "rule");
    const currentMap: Record<string, number | null | undefined> = {
      max_drawdown: riskData.maxDrawdown,
      max_daily_loss: riskData.maxDailyLoss,
      max_name_weight: riskData.concentration,
      max_sector_weight: riskData.sectorConcentration,
      max_turnover: latestBacktest?.turnover,
    };
    const current = currentMap[id] ?? null;
    const threshold = typeof rule.threshold === "number" || typeof rule.threshold === "string" ? rule.threshold : null;
    const warning = current !== null && typeof threshold === "number" && Math.abs(current) >= Math.abs(threshold) * 0.8;
    return {
      id,
      name: String(rule.name ?? id),
      description: String(rule.description ?? ""),
      current,
      threshold,
      enabled: rule.enabled !== false,
      state: current === null ? "unavailable" : warning ? "warning" : "normal",
    };
    });
  }, [latestBacktest?.turnover, riskData]);

  const queueItems = useMemo<ActionQueueItem[]>(() => {
    if (!data) return [];
    const items: ActionQueueItem[] = [];
    for (const job of jobItems.filter((item) => item.status === "failed").slice(0, 3)) {
      items.push({ id: `job-${job.id}`, severity: "critical", entity: job.commandId, reason: job.error ?? job.message ?? "任务失败", timestamp: formatDate(job.finishedAt ?? job.createdAt), source: "Task Center", action: "检查日志", path: `/training?job=${job.id}` });
    }
    if (riskEventCount) items.push({ id: "risk", severity: "warning", entity: "RiskGate / Backtest", reason: `${riskEventCount} 个持久化风险事件需要逐条检查。`, timestamp: formatDate(data.runtime.indexedAt), source: "RiskAdapter", action: "处理风险", path: "/risk" });
    const stale = data.runtime.byFreshness?.stale ?? 0;
    if (stale) items.push({ id: "stale-data", severity: "warning", entity: "Runtime Catalog", reason: `${stale} 个 artifact 已标记 stale。`, timestamp: formatDate(data.runtime.indexedAt), source: "RuntimeIndexer", action: "检查数据", path: "/runtime?freshnessStatus=stale" });
    if (data.latestModel && !data.latestModel.productionReady) items.push({ id: "model-gate", severity: "info", entity: data.latestModel.version ?? data.latestModel.id, reason: data.latestModel.verdict ?? "最新模型尚未通过 production/paper acceptance gate。", timestamp: formatDate(data.latestModel.createdAt), source: "Model Registry", action: "检查 Gate", path: `/training?modelId=${data.latestModel.id}` });
    if (data.runtime.manifestCoverage !== undefined && data.runtime.manifestCoverage < 1) items.push({ id: "manifest", severity: "info", entity: "Runtime governance", reason: `Manifest coverage ${Math.round(data.runtime.manifestCoverage * 100)}%，仍有未声明 artifact。`, timestamp: formatDate(data.runtime.indexedAt), source: "RuntimeIndexer", action: "查看缺口", path: "/runtime?validationStatus=unverified" });
    if (!items.length) items.push({ id: "clear", severity: "success", entity: "Decision state", reason: "没有失败任务、已记录风险事件或 stale artifact。", timestamp: formatDate(data.runtime.indexedAt), source: "Quant API", action: "查看任务", path: "/settings?view=jobs" });
    return items.slice(0, 7);
  }, [data, jobItems, riskEventCount]);

  if (overview.isLoading) return <StateView state="loading" detail="正在读取 Portfolio、Model、Risk、Task 与 Runtime 状态。" />;
  if (overview.isError || !data || !riskData) return <StateView state="error" detail={overview.error?.message ?? "System decision state unavailable"} />;

  return (
    <div className="vnext-dashboard">
      <header className="vnext-dashboard-title">
        <div><span>INSTITUTIONAL DECISION DASHBOARD</span><h1>今日决策总览</h1><p>发现异常、判断可信状态并进入对应工作站；复杂操作不在 Dashboard 内展开。</p></div>
        <div><strong>{formatDate(data.runtime.indexedAt)}</strong><span>Runtime decision as-of</span></div>
      </header>
      <DecisionStateStrip overview={data} latestPoint={latestPoint} jobs={jobItems} />
      <section className="vnext-dashboard-main">
        <PrimaryDecisionCanvas
          view={view}
          setView={setView}
          equity={equity.data?.data ?? []}
          backtest={latestBacktest}
          model={data.latestModel}
          risk={riskData}
          riskRules={riskRules}
          trades={trades.data?.data}
          jobs={jobItems}
        />
        <ActionQueue items={queueItems} />
      </section>
    </div>
  );
}
