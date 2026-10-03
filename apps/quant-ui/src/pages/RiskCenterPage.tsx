import { useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import { ChartLineDown, Drop, ShieldWarning, TrendDown, WarningCircle } from "@phosphor-icons/react";
import type { EChartsOption } from "echarts";
import type { BacktestSummary, Page, RiskOverview, SystemOverview } from "../api/types";
import { useApi } from "../hooks/useApi";
import { EChart } from "../components/EChart";
import { MonitorTable, type MonitorColumn } from "../components/MonitorTable";
import { Panel } from "../components/Panel";
import { RiskRadar } from "../components/RiskRadar";
import { StateView } from "../components/StateView";
import { UNMEASURED_TITLE, formatCompact, formatNumber, formatPercent, toneClass } from "../utils/format";
import { useVNextChartPalette } from "../vnext/theme";
import { usePaperAccount } from "../hooks/usePaperAccount";
import { buildVenueRuleRows } from "../vnext/paper/venueRules";
import { VenueRuleTable } from "../vnext/paper/VenueRuleTable";
import { ActionableState, WorkbenchHeader, WorkbenchMetricStrip } from "../vnext/workbench/InstitutionalWorkbench";
import { PaperKillSwitchChip } from "../vnext/paper/PaperAccountRiskCard";

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

type RiskRule = Record<string, unknown>;

export function RiskCenterPage(): JSX.Element {
  const palette = useVNextChartPalette();
  const [searchParams, setSearchParams] = useSearchParams();
  // One named subject per screen. Without an explicit ?backtestId= the page
  // audits the same backtest as the command bar's RISK chip and the dashboard
  // (the system overview's risk subject); it used to fall back to the
  // server's runs[0] - a different run, never named on the page.
  const system = useApi<SystemOverview>(["system-overview-shell"], "/system/overview", undefined, { refetchInterval: 15_000, staleTime: 10_000 });
  const backtests = useApi<BacktestSummary[]>(["backtest-lab"], "/backtests");
  const requestedId = searchParams.get("backtestId");
  const subjectId = requestedId ?? system.data?.data?.risk?.backtestId ?? null;
  const subjectReady = requestedId !== null || system.isFetched;
  const overview = useApi<RiskOverview>(["risk-overview"], subjectReady ? "/risk/overview" : null, { backtestId: subjectId });
  const risk = overview.data?.data;
  // Events and per-stock rows are fetched for the id the overview *answered*,
  // so counts from one run can never sit beside another run's headline.
  const auditedId = risk?.backtestId ?? null;
  const events = useApi<Page<RiskEvent>>(["risk-events"], auditedId ? "/risk/events" : null, { backtestId: auditedId, pageSize: 200 });
  const stocks = useApi<RiskStock[]>(["risk-stocks"], auditedId ? "/risk/stocks" : null, { backtestId: auditedId });
  const rules = useApi<RiskRule[]>(["risk-rules"], "/risk/rules");
  const paperAccount = usePaperAccount();
  const ruleRows = useMemo(
    () => buildVenueRuleRows(rules.data?.data ?? risk?.rules, paperAccount.data?.data?.riskState ? paperAccount.data.data : undefined),
    [paperAccount.data, risk?.rules, rules.data],
  );
  const subject = (backtests.data?.data ?? []).find((item) => item.id === auditedId);
  const subjectLabel = risk?.backtestName ?? subject?.name ?? auditedId ?? "no backtest";
  const selectSubject = (id: string): void => {
    const next = new URLSearchParams(searchParams);
    next.set("backtestId", id);
    setSearchParams(next, { replace: true });
  };
  const eventPage = events.data?.data;
  const eventItems = eventPage?.items ?? [];
  // The server reports `total` only once a page comes back short. While more
  // pages remain it is null, and we must not print a number that tracks the page
  // size rather than the data -- the old `total` evaluated to pageSize + 1 and
  // was rendered as an exact "201 alerts".
  const eventTotal = eventPage?.total ?? null;
  const eventLoaded = eventPage?.loadedCount ?? eventPage?.items.length ?? 0;
  const eventCountLabel = eventTotal !== null ? String(eventTotal) : `≥${eventLoaded}`;
  const eventCountValue = eventTotal ?? eventLoaded;

  const eventOption = useMemo<EChartsOption>(() => {
    const entries = Object.entries(risk?.eventCounts ?? {}).sort((left, right) => right[1] - left[1]).slice(0, 12);
    return {
      animation: false,
      grid: { left: 106, right: 18, top: 16, bottom: 24 },
      tooltip: { trigger: "axis", backgroundColor: palette.tooltip, borderColor: palette.tooltipBorder, textStyle: { color: palette.tooltipText } },
      xAxis: { type: "value", axisLabel: { color: palette.text }, splitLine: { lineStyle: { color: palette.grid } } },
      yAxis: { type: "category", inverse: true, data: entries.map(([name]) => name), axisLabel: { color: palette.text, fontSize: 10 } },
      series: [{ type: "bar", data: entries.map(([, value]) => value), itemStyle: { color: palette.warning }, barMaxWidth: 14 }],
    };
  }, [palette, risk?.eventCounts]);

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
        <span className={`mono ${toneClass(stock.netPnl)}`} title={stock.netPnl == null ? UNMEASURED_TITLE : undefined}>
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

  const venuePanel = (
    <Panel
      title="Paper venue limits · 纸面账户"
      eyebrow="/api/risk/rules + /api/paper/account · not this backtest"
      className="risk-venue-panel"
    >
      <VenueRuleTable rows={ruleRows} />
      <p className="risk-venue-note">
        限额来自 paper venue 的 RiskLimits；“Paper account now” 只放同一单位的 paper 账户读数（riskState）。
        回测的回撤、单日亏损属于另一个主体，见下方回测审计，不与这些限额比较。
      </p>
    </Panel>
  );

  const header = (
    <WorkbenchHeader
      eyebrow="RISK CONTROL / FAIL CLOSED"
      title="风险管理工作站"
      description="上：paper venue 的限额与账户读数。下：一个具名回测的持久化风险事件与回撤审计。两个主体不混合比较。"
      asOf={subject?.endDate ?? (eventItems[0]?.datetime?.slice(0, 10) ?? "as-of unavailable")}
      context={auditedId ? `audits ${subjectLabel} · ${auditedId}` : "no backtest audited"}
      actions={<PaperKillSwitchChip />}
    />
  );

  if (!subjectReady || overview.isLoading) return <StateView state="loading" detail="正在确定审计对象（与命令栏 RISK 同一回测）。" />;
  if (!risk) {
    return (
      <div className="page institutional-workbench risk-page">
        {header}
        {venuePanel}
        <ActionableState
          title={overview.isError ? `无法读取回测风险概览${subjectId ? `（${subjectId}）` : ""}` : "没有可审计的回测"}
          detail={overview.isError ? `${overview.error?.message ?? "risk overview unavailable"}。不显示任何回测风险数字；请从下拉框选择一个存在的回测。` : "没有回测产物时不显示任何回测风险数字；paper venue 限额仍在上方。"}
          icon={ShieldWarning}
          tone="danger"
        />
      </div>
    );
  }

  const trustCaveats = [
    subject?.trustClass ? `trust: ${subject.trustClass}` : null,
    subject?.validationStatus ? `validation: ${subject.validationStatus}` : null,
    subject && subject.timingCanonical === false ? "pre-fix clock" : null,
    subject?.quarantineOverlap?.length ? `overlaps quarantined ${subject.quarantineOverlap.join(", ")}` : null,
  ].filter((item): item is string => Boolean(item));
  const noEventArtifact = subject?.capabilities?.riskEvents === false;

  return (
    <div className="page institutional-workbench risk-page">
      {header}
      {venuePanel}

      <section className="atlas-surface risk-subject" data-rail="warning" aria-label="回测审计对象">
        <div className="risk-subject-copy">
          <span className="atlas-eyebrow">BACKTEST AUDIT · RESEARCH ARTIFACT · NOT THE PAPER ACCOUNT</span>
          <h2>{subjectLabel}</h2>
          <p className="mono">{auditedId} · {subject?.startDate ?? "—"} → {subject?.endDate ?? "—"}</p>
          <div className="atlas-row">
            {trustCaveats.length ? trustCaveats.map((item) => <span className="atlas-chip" data-tone="warning" key={item}>{item}</span>) : <span className="atlas-chip">trust class not reported</span>}
          </div>
        </div>
        <label className="atlas-field risk-subject-select">
          <span>审计回测 / audited backtest</span>
          <select value={auditedId ?? ""} onChange={(event) => selectSubject(event.target.value)} aria-label="选择审计回测">
            {auditedId && !subject ? <option value={auditedId}>{subjectLabel}</option> : null}
            {(backtests.data?.data ?? []).map((item) => <option key={item.id} value={item.id}>{item.name ?? item.id} · {item.endDate ?? "?"}</option>)}
          </select>
          <small>默认 = 命令栏 RISK 与决策总览使用的同一回测</small>
        </label>
      </section>

      <WorkbenchMetricStrip metrics={[
        { label: "最大回撤", value: formatPercent(risk.maxDrawdown), detail: `backtest NAV · ${subjectLabel}`, tone: "danger", icon: ChartLineDown },
        { label: "单票最大亏损", value: formatNumber(risk.maxSingleStockLoss), detail: "backtest realized PnL · CNY", tone: "danger", icon: TrendDown },
        { label: "最差单日收益", value: formatPercent(risk.maxDailyLoss), detail: "backtest daily return", tone: "danger", icon: TrendDown },
        { label: "连续亏损", value: risk.consecutiveLossDays == null ? "未测量" : formatCompact(risk.consecutiveLossDays), detail: risk.consecutiveLossDays == null ? "无日收益序列" : "trading days", tone: risk.consecutiveLossDays == null ? "neutral" : "warning", icon: WarningCircle },
        { label: "流动性事件占比", value: formatPercent(risk.liquidityRisk), detail: `${risk.eventCountsExact === false ? "first 1,000 events only · " : ""}跌停 ${formatPercent(risk.limitDownRisk)} · 停牌 ${formatPercent(risk.suspensionRisk)}`, tone: "warning", icon: Drop },
        {
          label: "风险事件",
          value: noEventArtifact ? "无产物" : eventTotal !== null ? formatCompact(eventTotal) : `≥${formatCompact(eventLoaded)}`,
          detail: noEventArtifact ? "risk_events 未落盘 · 不等于零违规" : "persisted backtest events",
          tone: noEventArtifact ? "neutral" : eventCountValue ? "warning" : "neutral",
          icon: ShieldWarning,
        },
      ]} />

      <section className="risk-grid risk-backtest-grid">
        <Panel title="风险雷达" eyebrow={`Backtest · ${subjectLabel}`} className="risk-radar-panel">
          <RiskRadar risk={risk} />
        </Panel>
        <Panel title="风控事件分布" eyebrow={`Persisted risk_events · ${subjectLabel}${risk.eventCountsExact === false ? " · first 1,000" : ""}`} className="risk-events-chart">
          {risk.eventCounts && Object.keys(risk.eventCounts).length ? <EChart option={eventOption} className="chart chart-medium" /> : <StateView state="empty" title="没有 risk_events 产物" detail="该回测没有写出 risk_events.json，或写出的事件计数为空。图表不会把「没有记录」画成「零违规」。下一步：确认回测启用了风控评估，或在 Runtime 工作站检查该运行的产物清单。" />}
        </Panel>
        <Panel title="单票风险排名" eyebrow={`Negative realized PnL first · ${subjectLabel}`} className="risk-stock-panel">
          <MonitorTable
            monitorId="risk-stocks"
            ariaLabel="单票风险排名"
            rows={stocks.data?.data ?? []}
            columns={riskStockColumns}
            rowKey={(stock) => stock.symbol}
            maxRows={80}
            exportFilename="quantagent-risk-stocks.csv"
            emptyDetail="该回测没有 profit_by_stock.csv。"
          />
        </Panel>
        <Panel title="风控事件时间线" eyebrow={`${eventCountLabel} indexed events · ${subjectLabel}`} className="risk-timeline-panel">
          {eventItems.length ? (
            <div className="risk-timeline">
              {eventItems.slice(0, 60).map((event) => (
                <div key={event.id} className={`risk-timeline-item severity-${event.severity}`}>
                  <i />
                  <div><strong>{event.type}</strong><span>{event.symbol ?? "portfolio"} · {event.reason ?? "reason unavailable"}</span></div>
                  <time>{event.datetime?.slice(0, 10) ?? "—"}</time>
                </div>
              ))}
            </div>
          ) : <StateView state="empty" title="没有逐条风控事件" detail="风控事件流为空。这表示这次运行没有落盘事件记录，不等于运行期间没有触发限额。下一步：在 Runtime 工作站核对该运行的 risk artifact。" />}
        </Panel>
      </section>
    </div>
  );
}
