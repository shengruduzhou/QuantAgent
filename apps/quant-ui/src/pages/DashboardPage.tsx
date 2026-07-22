import { useMemo, useState } from "react";
import {
  ArrowSquareOut,
  ChartLineUp,
  Coins,
  Database,
  Gauge,
  Pulse,
  ShieldWarning,
  TrendDown,
} from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import type {
  BacktestSummary,
  EquityPoint,
  Page,
  RiskOverview,
  SelectionRun,
  SystemOverview,
  Trade,
} from "../api/types";
import { useApi } from "../hooks/useApi";
import { EquityChart } from "../components/EquityChart";
import { MetricCard } from "../components/MetricCard";
import { Panel } from "../components/Panel";
import { RiskRadar } from "../components/RiskRadar";
import { SelectionFunnel } from "../components/SelectionFunnel";
import { StateView } from "../components/StateView";
import { StatusBadge } from "../components/StatusBadge";
import { TradeTable } from "../components/TradeTable";
import { formatBytes, formatCompact, formatDate, formatNumber, formatPercent } from "../utils/format";

interface RiskStock {
  symbol: string;
  netPnl?: number | null;
  winRate?: number | null;
  tradeCount?: number | null;
  riskScore?: number | null;
}

interface FunnelStage {
  stage: string;
  count: number | null;
  reason?: string | null;
}

type EquityRange = "3m" | "6m" | "1y" | "all";

const EQUITY_RANGE_POINTS: Record<EquityRange, number | null> = {
  "3m": 63,
  "6m": 126,
  "1y": 252,
  all: null,
};

export function DashboardPage(): JSX.Element {
  const [equityRange, setEquityRange] = useState<EquityRange>("all");
  const overview = useApi<SystemOverview>(["dashboard-overview"], "/system/overview");
  const backtests = useApi<BacktestSummary[]>(["dashboard-backtests"], "/backtests");
  const data = overview.data?.data;
  const latestBacktest = data?.latestBacktest ?? backtests.data?.data[0];
  const latestSelection = data?.latestSelection as SelectionRun | null | undefined;
  const equity = useApi<EquityPoint[]>(
    ["dashboard-equity", latestBacktest?.id],
    latestBacktest ? `/backtests/${latestBacktest.id}/equity` : null,
  );
  const trades = useApi<Page<Trade>>(
    ["dashboard-trades", latestBacktest?.id],
    latestBacktest ? `/backtests/${latestBacktest.id}/trades` : null,
    { pageSize: 8 },
  );
  const risk = useApi<RiskOverview>(
    ["dashboard-risk", latestBacktest?.id],
    "/risk/overview",
    { backtestId: latestBacktest?.id },
  );
  const riskStocks = useApi<RiskStock[]>(
    ["dashboard-risk-stocks", latestBacktest?.id],
    "/risk/stocks",
    { backtestId: latestBacktest?.id },
  );
  const funnel = useApi<FunnelStage[]>(
    ["dashboard-funnel", latestSelection?.id],
    latestSelection ? `/selection/runs/${latestSelection.id}/funnel` : null,
  );
  const equityPoints = equity.data?.data ?? [];
  const visibleEquity = useMemo(() => {
    const limit = EQUITY_RANGE_POINTS[equityRange];
    return limit ? equityPoints.slice(-limit) : equityPoints;
  }, [equityPoints, equityRange]);

  if (overview.isLoading) return <StateView state="loading" />;
  if (overview.isError || !data) return <StateView state="error" detail={overview.error?.message} />;

  const topContributors = [...(riskStocks.data?.data ?? [])]
    .filter((item) => item.netPnl !== null && item.netPnl !== undefined)
    .sort((left, right) => (right.netPnl ?? 0) - (left.netPnl ?? 0))
    .slice(0, 6);
  const contributorScale = Math.max(
    1,
    ...topContributors.map((item) => Math.abs(item.netPnl ?? 0)),
  );
  const riskData = risk.data?.data ?? data.risk;
  const riskEventCount = Object.values(data.risk.eventCounts ?? {}).reduce((sum, value) => sum + value, 0);
  const healthRows = [
    { label: "Runtime Index", status: "ready", latency: "snapshot", quality: `${data.runtime.artifactCount.toLocaleString()} items` },
    { label: "Model Registry", status: data.modelStatus, latency: "metadata", quality: data.latestModel?.version ?? "暂无版本" },
    { label: "Backtest Adapter", status: latestBacktest ? "ready" : "partial", latency: "on-demand", quality: `${backtests.data?.data.length ?? 0} runs` },
    { label: "Selection Pipeline", status: latestSelection ? "ready" : "partial", latency: "on-demand", quality: `${data.stockPoolCount ?? 0} names` },
    { label: "Risk Engine", status: data.riskStatus, latency: "streamed", quality: `${riskEventCount} events` },
  ];
  const healthyCount = healthRows.filter((row) => row.status === "ready" || row.status === "normal").length;
  const manifestCoverage = data.runtime.manifestCoverage;

  return (
    <div className="page dashboard-page dashboard-v5">
      <header className="dashboard-command-head">
        <div className="dashboard-command-title">
          <span className="dashboard-command-kicker"><Pulse size={14} weight="fill" /> PORTFOLIO COMMAND</span>
          <div>
            <h1>研究驾驶舱</h1>
            <p>{latestBacktest?.name ?? "等待已持久化回测"} · {latestBacktest?.horizon ?? "multi-horizon"}</p>
          </div>
        </div>
        <div className="dashboard-command-meta">
          <StatusBadge
            status={data.riskStatus}
            label={data.riskStatus === "normal" ? "风险受控" : "风险需关注"}
          />
          <span><Database size={14} /> {data.runtime.artifactCount.toLocaleString()} artifacts</span>
          <span>INDEXED <time>{formatDate(data.runtime.indexedAt)}</time></span>
          <Link className="dashboard-primary-link" to="/backtests">打开实验台 <ArrowSquareOut size={14} /></Link>
        </div>
      </header>

      <section className="dashboard-kpi-deck" aria-label="组合关键指标">
        <article className="portfolio-pulse-card">
          <div className="portfolio-pulse-top">
            <div>
              <span>组合总收益</span>
              <small>Portfolio return · persisted result</small>
            </div>
            <ChartLineUp size={21} weight="duotone" />
          </div>
          <div className="portfolio-pulse-value">
            <strong>{formatPercent(latestBacktest?.totalReturn)}</strong>
            <span className={(latestBacktest?.totalReturn ?? 0) >= 0 ? "tone-positive" : "tone-negative"}>
              {(latestBacktest?.totalReturn ?? 0) > 0 ? "+" : ""}{formatPercent(latestBacktest?.totalReturn)}
            </span>
          </div>
          <div className="portfolio-pulse-foot">
            <span><small>年化收益</small><b>{formatPercent(latestBacktest?.annualReturn)}</b></span>
            <span><small>区间</small><b>{latestBacktest?.startDate ?? "—"} → {latestBacktest?.endDate ?? "—"}</b></span>
          </div>
        </article>
        <MetricCard label="最大回撤" value={formatPercent(latestBacktest?.maxDrawdown)} delta={latestBacktest?.maxDrawdown ? -Math.abs(latestBacktest.maxDrawdown) : null} icon={TrendDown} />
        <MetricCard label="Sharpe" value={formatNumber(latestBacktest?.sharpe)} detail={`Calmar ${formatNumber(latestBacktest?.calmar)}`} icon={Gauge} />
        <MetricCard label="年化换手" value={formatPercent(latestBacktest?.turnover)} tone="warning" detail={`${formatCompact(latestBacktest?.tradeCount)} trades`} icon={Coins} />
        <MetricCard label="当前候选池" value={formatCompact(data.stockPoolCount)} detail={`初始 ${formatCompact(data.candidateCount)}`} icon={Pulse} />
        <MetricCard label="风险状态" value={data.riskStatus === "normal" ? "受控" : "关注"} tone={data.riskStatus === "normal" ? "neutral" : "warning"} detail={`${riskEventCount} 个事件`} icon={ShieldWarning} />
      </section>

      <section className="dashboard-command-grid">
        <Panel
          title="组合净值"
          eyebrow={`${latestBacktest?.strategyVersion ?? latestBacktest?.name ?? "暂无策略"} · 净值 / 回撤`}
          className="dashboard-equity"
          actions={(
            <div className="dashboard-chart-actions">
              <div className="dashboard-range-control" aria-label="净值时间范围">
                {(["3m", "6m", "1y", "all"] as EquityRange[]).map((range) => (
                  <button
                    key={range}
                    type="button"
                    className={equityRange === range ? "active" : ""}
                    aria-pressed={equityRange === range}
                    onClick={() => setEquityRange(range)}
                  >
                    {{ "3m": "近3月", "6m": "近6月", "1y": "近1年", all: "全部" }[range]}
                  </button>
                ))}
              </div>
              <Link className="text-link" to="/backtests" aria-label="进入回测实验"><ArrowSquareOut size={15} /></Link>
            </div>
          )}
        >
          {visibleEquity.length ? (
            <EquityChart points={visibleEquity} height={326} />
          ) : <StateView state="empty" detail="当前实验没有可验证的净值序列。" />}
        </Panel>

        <Panel
          title="风险控制台"
          eyebrow="Risk envelope · relative thresholds"
          className="dashboard-risk-console"
          actions={<StatusBadge status={data.riskStatus} label={data.riskStatus === "normal" ? "NORMAL" : "ATTENTION"} />}
        >
          <div className="dashboard-risk-stack">
            <RiskRadar risk={riskData} />
            <div className="dashboard-risk-meters">
              <ExposureRow label="组合回撤" value={riskData.maxDrawdown} />
              <ExposureRow label="流动性风险" value={riskData.liquidityRisk} />
              <ExposureRow label="跌停风险" value={riskData.limitDownRisk} />
              <ExposureRow label="波动率暴露" value={riskData.volatilityExposure} />
            </div>
          </div>
        </Panel>

        <Panel
          title="透明选股漏斗"
          eyebrow="Universe → liquidity → risk → factor → model → portfolio"
          className="dashboard-funnel"
          actions={<Link className="text-link" to="/selection">查看决策链 <ArrowSquareOut size={14} /></Link>}
        >
          {funnel.data?.data.length ? <SelectionFunnel stages={funnel.data.data} /> : <StateView state="empty" detail="当前 selection run 没有 persisted funnel。" />}
        </Panel>

        <Panel title="系统健康" eyebrow={`${healthyCount}/${healthRows.length} services nominal`} className="dashboard-health">
          <div className="health-list">
            {healthRows.map((row) => <HealthRow key={row.label} {...row} />)}
          </div>
          <div className="health-summary health-summary-v5">
            <span><small>Runtime 体积</small><strong>{formatBytes(data.runtime.totalSizeBytes)}</strong></span>
            <span>
              <small>Manifest 覆盖率</small>
              <strong>{manifestCoverage === undefined ? "暂无" : formatPercent(manifestCoverage, 0)}</strong>
            </span>
          </div>
        </Panel>

        <Panel
          title="最新交易明细"
          eyebrow="Execution blotter · persisted fills only"
          className="dashboard-trades"
          actions={<Link className="text-link" to="/stock-replay">进入股票复盘 <ArrowSquareOut size={14} /></Link>}
        >
          {trades.data?.data.items.length ? (
            <TradeTable trades={trades.data.data.items} compact />
          ) : <StateView state="empty" detail={trades.data?.issues[0]?.message ?? "当前实验没有 persisted fills。"} />}
        </Panel>

        <Panel title="Top 贡献标的" eyebrow="Realized PnL attribution" className="dashboard-contributors">
          {topContributors.length ? (
            <div className="contributor-list">
              {topContributors.map((stock, index) => (
                <div key={stock.symbol} className="contributor-row">
                  <span className="rank-number">{String(index + 1).padStart(2, "0")}</span>
                  <strong>{stock.symbol}</strong>
                  <div className="contributor-bar"><i style={{ width: `${Math.max(6, (Math.abs(stock.netPnl ?? 0) / contributorScale) * 100)}%` }} /></div>
                  <span className={(stock.netPnl ?? 0) >= 0 ? "tone-positive mono" : "tone-negative mono"}>{formatNumber(stock.netPnl)}</span>
                </div>
              ))}
            </div>
          ) : <StateView state="empty" detail="profit_by_stock.csv 不存在或没有 realized PnL。" />}
        </Panel>
      </section>
    </div>
  );
}

function ExposureRow({ label, value }: { label: string; value?: number | null }): JSX.Element {
  const intensity = Math.min(100, Math.abs(value ?? 0) * 100);
  const tone = intensity >= 65 ? "critical" : intensity >= 35 ? "warning" : "normal";
  return (
    <div className={`exposure-row exposure-${tone}`}>
      <span>{label}</span>
      <div><i style={{ width: `${Math.max(value === null || value === undefined ? 0 : 2, intensity)}%` }} /></div>
      <strong className="mono">{formatPercent(value)}</strong>
    </div>
  );
}

interface HealthRowProps {
  label: string;
  status: string;
  latency: string;
  quality: string;
}

function HealthRow({ label, status, latency, quality }: HealthRowProps): JSX.Element {
  return (
    <div className="health-row">
      <div><strong>{label}</strong><small>{latency}</small></div>
      <StatusBadge status={status} />
      <span className="mono">{quality}</span>
    </div>
  );
}
