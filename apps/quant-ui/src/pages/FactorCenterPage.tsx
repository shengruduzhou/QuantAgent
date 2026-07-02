import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { Code, MagnifyingGlass, ShieldCheck } from "@phosphor-icons/react";
import type { EChartsOption } from "echarts";
import type { Factor } from "../api/types";
import { useApi } from "../hooks/useApi";
import { EChart } from "../components/EChart";
import { MetricCard } from "../components/MetricCard";
import { Panel } from "../components/Panel";
import { StateView } from "../components/StateView";
import { StatusBadge } from "../components/StatusBadge";
import { formatNumber, formatPercent } from "../utils/format";

interface FactorBacktest {
  factorName: string;
  totalReturn?: number | null;
  annualReturn?: number | null;
  maxDrawdown?: number | null;
  sharpe?: number | null;
  calmar?: number | null;
  winRate?: number | null;
  turnover?: number | null;
  ic?: number | null;
  rankIc?: number | null;
  icir?: number | null;
  rankIcir?: number | null;
  coverage?: number | null;
  stability?: number | null;
  crowding?: number | null;
  verdict?: string | null;
  bestHorizon?: string | null;
  regimeIc: Record<string, number | null>;
  icSeries: Array<{ datetime: string; value: number }>;
  rankIcSeries: Array<{ datetime: string; value: number }>;
  quantileReturns: Array<Record<string, unknown>>;
  longShortEquity: Array<Record<string, unknown>>;
  decay: Array<{ horizonDays: number; ic?: number | null; rankIc?: number | null }>;
  availability: Record<string, boolean>;
}

export function FactorCenterPage(): JSX.Element {
  const [query, setQuery] = useState("");
  const [selectedName, setSelectedName] = useState("");
  const deferredQuery = useDeferredValue(query);
  const factors = useApi<Factor[]>(["factors", deferredQuery], "/factors", { query: deferredQuery });
  const list = factors.data?.data ?? [];

  useEffect(() => {
    if ((!selectedName || !list.some((factor) => factor.name === selectedName)) && list[0]) {
      setSelectedName(list[0].name);
    }
  }, [list, selectedName]);

  const detail = useApi<Factor>(["factor-detail", selectedName], selectedName ? `/factors/${selectedName}` : null);
  const backtest = useApi<FactorBacktest>(["factor-backtest", selectedName], selectedName ? `/factors/${selectedName}/backtest` : null);
  const factor = detail.data?.data;
  const metrics = backtest.data?.data;

  const icOption = useMemo<EChartsOption>(() => ({
    animation: false,
    grid: { left: 48, right: 16, top: 20, bottom: 32 },
    tooltip: { trigger: "axis", backgroundColor: "#0b1824", borderColor: "#27425a", textStyle: { color: "#d7e4ef" } },
    xAxis: { type: "category", data: metrics?.icSeries.map((point) => point.datetime) ?? [], axisLabel: { color: "#71879a", fontSize: 10 } },
    yAxis: { type: "value", axisLabel: { color: "#71879a", fontSize: 10 }, splitLine: { lineStyle: { color: "#14283a" } } },
    series: [
      { name: "IC", type: "line", showSymbol: true, data: metrics?.icSeries.map((point) => point.value) ?? [], lineStyle: { color: "#3f8cff" } },
      { name: "Rank IC", type: "line", showSymbol: true, data: metrics?.rankIcSeries.map((point) => point.value) ?? [], lineStyle: { color: "#2ac8a0" } },
    ],
  }), [metrics?.icSeries, metrics?.rankIcSeries]);

  const decayOption = useMemo<EChartsOption>(() => ({
    animation: false,
    grid: { left: 48, right: 16, top: 20, bottom: 32 },
    tooltip: { trigger: "axis", backgroundColor: "#0b1824", borderColor: "#27425a", textStyle: { color: "#d7e4ef" } },
    xAxis: { type: "category", data: metrics?.decay.map((point) => `${point.horizonDays}D`) ?? [], axisLabel: { color: "#71879a" } },
    yAxis: { type: "value", axisLabel: { color: "#71879a" }, splitLine: { lineStyle: { color: "#14283a" } } },
    series: [{ type: "bar", data: metrics?.decay.map((point) => point.ic ?? 0) ?? [], itemStyle: { color: "#3f8cff" }, barMaxWidth: 34 }],
  }), [metrics?.decay]);

  return (
    <div className="page factor-page">
      <aside className="factor-sidebar panel">
        <div className="factor-search">
          <MagnifyingGlass size={16} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索因子名称 / 类别" />
        </div>
        <div className="factor-count">{list.length} factors</div>
        <div className="factor-list">
          {list.map((item) => (
            <button key={item.name} className={item.name === selectedName ? "active" : ""} onClick={() => setSelectedName(item.name)}>
              <div><strong>{item.displayName ?? item.name}</strong><span>{item.category ?? item.sourceKind}</span></div>
              <StatusBadge status={item.lifecycle ?? (item.usedInTraining ? "ready" : "partial")} label={item.usedInTraining ? "训练中" : item.lifecycle ?? "研究"} />
            </button>
          ))}
          {!list.length ? <StateView state={factors.isLoading ? "loading" : "empty"} /> : null}
        </div>
      </aside>

      <section className="factor-content">
        <Panel title={factor?.displayName ?? selectedName ?? "因子"} eyebrow={`${factor?.category ?? "unknown"} · ${factor?.frequency ?? "unknown frequency"}`} className="factor-definition">
          {factor ? (
            <div className="factor-definition-grid">
              <div className="factor-summary">
                <p>{factor.description ?? "暂无人工解释。"}</p>
                <div className="formula-block">
                  <Code size={18} />
                  <code>{factor.formula ?? "暂无可提取公式；请查看代码位置。"}</code>
                </div>
              </div>
              <div className="factor-metadata">
                <MetaRow label="方向" value={factor.direction} />
                <MetaRow label="代码位置" value={factor.codeLocation ?? "暂无"} />
                <MetaRow label="输入数据" value={factor.requiredColumns.join(", ") || "暂无 metadata"} />
                <MetaRow label="参数" value={Object.keys(factor.parameters).length ? JSON.stringify(factor.parameters) : "无显式参数"} />
                <MetaRow label="PIT 安全" value={factor.pitSafe === true ? "通过" : factor.pitSafe === false ? "未通过" : "未声明"} />
              </div>
              <div className="factor-usage">
                <UsageFlag label="参与训练" active={factor.usedInTraining} />
                <UsageFlag label="参与选股" active={factor.usedInSelection} />
                <UsageFlag label="参与择时" active={factor.usedInTiming} />
                <UsageFlag label="参与风控" active={factor.usedInRisk} />
              </div>
            </div>
          ) : <StateView state={detail.isLoading ? "loading" : "empty"} />}
        </Panel>

        <section className="metric-grid metric-grid-6 factor-metrics">
          <MetricCard label="IC" value={formatNumber(metrics?.ic, 4)} />
          <MetricCard label="Rank IC" value={formatNumber(metrics?.rankIc, 4)} />
          <MetricCard label="ICIR" value={formatNumber(metrics?.icir)} />
          <MetricCard label="覆盖率" value={formatPercent(metrics?.coverage)} />
          <MetricCard label="稳定性" value={formatNumber(metrics?.stability)} />
          <MetricCard label="最佳 Horizon" value={metrics?.bestHorizon ?? "暂无"} detail={metrics?.verdict ?? "no verdict"} />
        </section>

        <div className="factor-chart-grid">
          <Panel title="IC 时间序列" eyebrow="Source-backed annual/periodic metrics">
            {metrics?.icSeries.length ? <EChart option={icOption} className="chart chart-medium" /> : <StateView state="empty" detail="没有独立 IC series artifact。" />}
          </Panel>
          <Panel title="因子衰减" eyebrow="IC by horizon">
            {metrics?.decay.length ? <EChart option={decayOption} className="chart chart-medium" /> : <StateView state="empty" />}
          </Panel>
          <Panel title="市场环境适用性" eyebrow="Regime IC">
            <div className="regime-cards">
              {Object.entries(metrics?.regimeIc ?? {}).map(([regime, value]) => (
                <div key={regime}><span>{regime}</span><strong className={(value ?? 0) >= 0 ? "tone-positive" : "tone-negative"}>{formatNumber(value, 4)}</strong></div>
              ))}
            </div>
          </Panel>
          <Panel title="独立回测可用性" eyebrow="No fabricated single-factor trades">
            <div className="availability-list">
              {Object.entries(metrics?.availability ?? {}).map(([name, available]) => (
                <div key={name}><span>{name}</span><StatusBadge status={available ? "ready" : "partial"} label={available ? "可用" : "暂无 artifact"} /></div>
              ))}
            </div>
            {!metrics?.availability.trades ? (
              <div className="truth-note"><ShieldCheck size={18} /><span>未发现该因子的独立成交回报，因此不复用 multi-factor 买卖点。</span></div>
            ) : null}
          </Panel>
        </div>
      </section>
    </div>
  );
}

function MetaRow({ label, value }: { label: string; value: string }): JSX.Element {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}

function UsageFlag({ label, active }: { label: string; active?: boolean | null }): JSX.Element {
  return <div className={active ? "usage-active" : ""}><i /><span>{label}</span><strong>{active ? "YES" : "NO"}</strong></div>;
}
