import { useEffect, useMemo, useState } from "react";
import { CheckSquare, DownloadSimple, Square } from "@phosphor-icons/react";
import type { EChartsOption } from "echarts";
import type { BacktestSummary, EquityPoint } from "../api/types";
import { downloadJson } from "../api/client";
import { useApi } from "../hooks/useApi";
import { EChart } from "../components/EChart";
import { MetricCard } from "../components/MetricCard";
import { Panel } from "../components/Panel";
import { StateView } from "../components/StateView";
import { StatusBadge } from "../components/StatusBadge";
import { formatCompact, formatNumber, formatPercent } from "../utils/format";

export function BacktestLabPage(): JSX.Element {
  const backtests = useApi<BacktestSummary[]>(["backtest-lab"], "/backtests");
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const runs = backtests.data?.data ?? [];

  useEffect(() => {
    if (!selectedIds.length && runs.length) {
      setSelectedIds(runs.slice(0, 4).map((run) => run.id));
    }
  }, [runs, selectedIds.length]);

  const primary = runs.find((run) => run.id === selectedIds[0]) ?? runs[0];
  const equity = useApi<EquityPoint[]>(
    ["backtest-lab-equity", primary?.id],
    primary ? `/backtests/${primary.id}/equity` : null,
  );

  const comparisonOption = useMemo<EChartsOption>(() => ({
    animation: false,
    grid: { left: 54, right: 18, top: 20, bottom: 34 },
    tooltip: { trigger: "axis", backgroundColor: "#0b1824", borderColor: "#27425a", textStyle: { color: "#d7e4ef" } },
    xAxis: {
      type: "category",
      data: equity.data?.data.map((point) => point.datetime) ?? [],
      axisLabel: { color: "#71879a", fontSize: 10 },
      axisLine: { lineStyle: { color: "#20364a" } },
    },
    yAxis: {
      type: "value",
      scale: true,
      axisLabel: { color: "#71879a", fontSize: 10 },
      splitLine: { lineStyle: { color: "#14283a" } },
    },
    series: [{
      name: primary?.name ?? "NAV",
      type: "line",
      data: equity.data?.data.map((point) => point.nav) ?? [],
      showSymbol: false,
      lineStyle: { color: "#3f8cff", width: 1.8 },
      areaStyle: { color: "rgba(63,140,255,.08)" },
    }],
  }), [equity.data?.data, primary?.name]);

  if (backtests.isLoading) return <StateView state="loading" />;
  if (!runs.length) return <StateView state="empty" />;

  const selectedRuns = runs.filter((run) => selectedIds.includes(run.id));

  return (
    <div className="page backtest-page">
      <section className="metric-grid metric-grid-6">
        <MetricCard label="总收益" value={formatPercent(primary?.totalReturn)} delta={primary?.totalReturn} />
        <MetricCard label="年化收益" value={formatPercent(primary?.annualReturn)} delta={primary?.annualReturn} />
        <MetricCard label="最大回撤" value={formatPercent(primary?.maxDrawdown)} tone="negative" />
        <MetricCard label="Sharpe" value={formatNumber(primary?.sharpe)} detail={`Calmar ${formatNumber(primary?.calmar)}`} />
        <MetricCard label="换手率" value={formatPercent(primary?.turnover)} tone="warning" />
        <MetricCard label="成交数量" value={formatCompact(primary?.tradeCount)} detail={`${formatCompact(primary?.fillCount)} fills`} />
      </section>

      <section className="backtest-grid">
        <Panel title="实验净值" eyebrow={`${primary?.name} · ${primary?.startDate ?? "未知"} → ${primary?.endDate ?? "未知"}`} className="backtest-equity-panel">
          {equity.data?.data.length ? <EChart option={comparisonOption} className="chart" /> : <StateView state="empty" detail="该研究实验没有 NAV artifact。" />}
        </Panel>

        <Panel title="实验能力" eyebrow="Artifact Capability Matrix" className="capability-panel">
          <div className="capability-grid">
            {Object.entries(primary?.capabilities ?? {}).map(([name, enabled]) => (
              <div key={name}>
                <span>{name}</span>
                {typeof enabled === "boolean"
                  ? <StatusBadge status={enabled ? "ready" : "partial"} label={enabled ? "可用" : "暂无"} />
                  : <strong>{enabled ?? "—"}</strong>}
              </div>
            ))}
          </div>
          <div className="experiment-path"><span>Artifact path</span><code>{primary?.path}</code></div>
        </Panel>

        <Panel
          title="实验对比"
          eyebrow={`${selectedRuns.length} selected · 真实 runtime metrics`}
          className="backtest-table-panel"
          actions={<button className="secondary-button" onClick={() => downloadJson("backtest-comparison.json", selectedRuns)}><DownloadSimple size={15} /> 导出</button>}
        >
          <div className="table-scroll">
            <table className="data-table comparison-table">
              <thead>
                <tr>
                  <th>对比</th>
                  <th>实验 / Horizon</th>
                  <th>区间</th>
                  <th className="numeric">总收益</th>
                  <th className="numeric">年化</th>
                  <th className="numeric">Sharpe</th>
                  <th className="numeric">最大回撤</th>
                  <th className="numeric">Calmar</th>
                  <th className="numeric">换手</th>
                  <th className="numeric">交易数</th>
                  <th>能力</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((run) => {
                  const selected = selectedIds.includes(run.id);
                  return (
                    <tr
                      key={run.id}
                      className={run.id === primary?.id ? "row-selected" : ""}
                      tabIndex={0}
                      onClick={() => {
                        setSelectedIds((current) => selected
                          ? current.filter((id) => id !== run.id)
                          : [...current.slice(0, 5), run.id]);
                      }}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          setSelectedIds((current) => selected
                            ? current.filter((id) => id !== run.id)
                            : [...current.slice(0, 5), run.id]);
                        }
                      }}
                    >
                      <td>{selected ? <CheckSquare size={17} weight="fill" className="tone-positive" /> : <Square size={17} />}</td>
                      <td><strong>{run.name}</strong><span>{run.horizon ?? "research"}</span></td>
                      <td className="mono">{run.startDate?.slice(0, 10) ?? "—"} → {run.endDate?.slice(0, 10) ?? "—"}</td>
                      <td className={`numeric ${tone(run.totalReturn)}`}>{formatPercent(run.totalReturn)}</td>
                      <td className={`numeric ${tone(run.annualReturn)}`}>{formatPercent(run.annualReturn)}</td>
                      <td className="numeric mono">{formatNumber(run.sharpe)}</td>
                      <td className="numeric tone-negative">{formatPercent(run.maxDrawdown)}</td>
                      <td className="numeric mono">{formatNumber(run.calmar)}</td>
                      <td className="numeric">{formatPercent(run.turnover)}</td>
                      <td className="numeric mono">{formatCompact(run.tradeCount)}</td>
                      <td><StatusBadge status={run.status} /></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      </section>
    </div>
  );
}

function tone(value: number | null | undefined): string {
  if (value === null || value === undefined) return "";
  return value >= 0 ? "tone-positive" : "tone-negative";
}
