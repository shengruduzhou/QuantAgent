import { useMemo } from "react";
import { ShieldWarning } from "@phosphor-icons/react";
import type { EChartsOption } from "echarts";
import type { Page, RiskOverview } from "../api/types";
import { useApi } from "../hooks/useApi";
import { EChart } from "../components/EChart";
import { MetricCard } from "../components/MetricCard";
import { MonitorTable, type MonitorColumn } from "../components/MonitorTable";
import { Panel } from "../components/Panel";
import { RiskRadar } from "../components/RiskRadar";
import { StateView } from "../components/StateView";
import { StatusBadge } from "../components/StatusBadge";
import { formatCompact, formatNumber, formatPercent } from "../utils/format";
import { marketPalette } from "../theme/marketPalette";

interface RiskEvent {
  id: string;
  datetime?: string | null;
  symbol?: string | null;
  type: string;
  severity: string;
  reason?: string | null;
  rule?: string | null;
  blocked?: boolean | null;
  sourcePath: string;
}

interface RiskStock {
  symbol: string;
  netPnl?: number | null;
  winRate?: number | null;
  tradeCount?: number | null;
  riskScore?: number | null;
}

interface RiskRule {
  id: string;
  name: string;
  description: string;
  threshold: unknown;
  enabled: boolean;
  codeLocation: string;
}

export function RiskCenterPage(): JSX.Element {
  const overview = useApi<RiskOverview>(["risk-overview"], "/risk/overview");
  const events = useApi<Page<RiskEvent>>(["risk-events"], "/risk/events", { pageSize: 200 });
  const stocks = useApi<RiskStock[]>(["risk-stocks"], "/risk/stocks");
  const rules = useApi<RiskRule[]>(["risk-rules"], "/risk/rules");
  const risk = overview.data?.data;

  const eventOption = useMemo<EChartsOption>(() => {
    const entries = Object.entries(risk?.eventCounts ?? {}).sort((left, right) => right[1] - left[1]).slice(0, 12);
    return {
      animation: false,
      grid: { left: 106, right: 18, top: 16, bottom: 24 },
      tooltip: { trigger: "axis", backgroundColor: marketPalette.panel, borderColor: marketPalette.border, textStyle: { color: marketPalette.text } },
      xAxis: { type: "value", axisLabel: { color: marketPalette.axis }, splitLine: { lineStyle: { color: marketPalette.grid } } },
      yAxis: { type: "category", inverse: true, data: entries.map(([name]) => name), axisLabel: { color: "#9cb1c3", fontSize: 10 } },
      series: [{ type: "bar", data: entries.map(([, value]) => value), itemStyle: { color: marketPalette.risk }, barMaxWidth: 14 }],
    };
  }, [risk?.eventCounts]);

  const riskStockColumns = useMemo<MonitorColumn<RiskStock>[]>(() => [
    {
      id: "symbol",
      header: "股票",
      value: (stock) => stock.symbol,
      render: (stock) => <strong>{stock.symbol}</strong>,
      width: 116,
    },
    {
      id: "netPnl",
      header: "净 PnL",
      value: (stock) => stock.netPnl ?? Number.NEGATIVE_INFINITY,
      csvValue: (stock) => stock.netPnl,
      render: (stock) => (
        <span className={`mono ${(stock.netPnl ?? 0) >= 0 ? "tone-positive" : "tone-negative"}`}>
          {formatNumber(stock.netPnl)}
        </span>
      ),
      align: "right",
      width: 112,
    },
    {
      id: "winRate",
      header: "胜率",
      value: (stock) => stock.winRate ?? Number.NEGATIVE_INFINITY,
      csvValue: (stock) => stock.winRate,
      render: (stock) => formatPercent(stock.winRate),
      align: "right",
      width: 92,
    },
    {
      id: "tradeCount",
      header: "交易数",
      value: (stock) => stock.tradeCount ?? 0,
      render: (stock) => <span className="mono">{formatCompact(stock.tradeCount)}</span>,
      align: "right",
      width: 86,
    },
    {
      id: "riskScore",
      header: "风险分",
      value: (stock) => stock.riskScore ?? Number.NEGATIVE_INFINITY,
      csvValue: (stock) => stock.riskScore,
      render: (stock) => <span className="mono tone-warning">{formatNumber(stock.riskScore)}</span>,
      align: "right",
      width: 92,
    },
  ], []);

  if (overview.isLoading) return <StateView state="loading" />;
  if (!risk) return <StateView state="empty" />;

  return (
    <div className="page risk-page">
      <section className="metric-grid metric-grid-7">
        <MetricCard label="最大回撤" value={formatPercent(risk.maxDrawdown)} tone="negative" />
        <MetricCard label="单票最大亏损" value={formatNumber(risk.maxSingleStockLoss)} tone="negative" />
        <MetricCard label="单日最大亏损" value={formatPercent(risk.maxDailyLoss)} tone="negative" />
        <MetricCard label="连续亏损天数" value={formatCompact(risk.consecutiveLossDays)} tone="warning" />
        <MetricCard label="流动性风险" value={formatPercent(risk.liquidityRisk)} />
        <MetricCard label="跌停风险" value={formatPercent(risk.limitDownRisk)} />
        <MetricCard label="停牌风险" value={formatPercent(risk.suspensionRisk)} />
      </section>

      <section className="risk-grid">
        <Panel title="风险雷达" eyebrow="Risk Radar · relative thresholds" className="risk-radar-panel">
          <RiskRadar risk={risk} />
        </Panel>
        <Panel title="风控事件分布" eyebrow="Persisted risk_events.json" className="risk-events-chart">
          {Object.keys(risk.eventCounts).length ? <EChart option={eventOption} className="chart chart-medium" /> : <StateView state="empty" />}
        </Panel>
        <Panel title="风险规则" eyebrow="Thresholds read from code defaults" className="risk-rules-panel">
          <div className="risk-rule-list">
            {(rules.data?.data ?? []).map((rule) => (
              <div key={rule.id}>
                <ShieldWarning size={17} />
                <span><strong>{rule.name}</strong><small>{rule.description}</small></span>
                <code>{String(rule.threshold ?? "event gate")}</code>
                <StatusBadge status={rule.enabled ? "ready" : "partial"} label={rule.enabled ? "启用" : "关闭"} />
              </div>
            ))}
          </div>
        </Panel>
        <Panel title="单票风险排名" eyebrow="Negative realized PnL first" className="risk-stock-panel">
          <MonitorTable
            monitorId="risk-stocks"
            ariaLabel="单票风险排名"
            rows={stocks.data?.data ?? []}
            columns={riskStockColumns}
            rowKey={(stock) => stock.symbol}
            maxRows={80}
            exportFilename="quantagent-risk-stocks.csv"
            emptyDetail="profit_by_stock.csv 不存在。"
          />
        </Panel>
        <Panel title="风控事件时间线" eyebrow={`${events.data?.data.total ?? 0}+ indexed events`} className="risk-timeline-panel">
          {(events.data?.data.items ?? []).length ? (
            <div className="risk-timeline">
              {events.data?.data.items.slice(0, 60).map((event) => (
                <div key={event.id} className={`risk-timeline-item severity-${event.severity}`}>
                  <i />
                  <div><strong>{event.type}</strong><span>{event.symbol ?? "portfolio"} · {event.reason ?? "reason unavailable"}</span></div>
                  <time>{event.datetime?.slice(0, 10) ?? "—"}</time>
                </div>
              ))}
            </div>
          ) : <StateView state="empty" />}
        </Panel>
      </section>
    </div>
  );
}
