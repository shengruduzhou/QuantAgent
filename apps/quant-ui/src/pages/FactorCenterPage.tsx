import { useDeferredValue, useEffect, useMemo, useState } from "react";
import { Broom, Code, MagnifyingGlass, Play, ShieldCheck } from "@phosphor-icons/react";
import type { EChartsOption } from "echarts";
import { useNavigate } from "react-router-dom";
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

type UtilityFilter = "all" | "active" | "useful" | "excluded" | "unevaluated";
type UtilityClass = Exclude<UtilityFilter, "all" | "active">;

const UTILITY_LABELS: Record<UtilityFilter, string> = {
  all: "全部",
  active: "已启用",
  useful: "有效候选",
  excluded: "已剔除",
  unevaluated: "待评估",
};

function isActiveFactor(factor: Factor): boolean {
  return Boolean(factor.usedInTraining || factor.usedInSelection || factor.usedInTiming || factor.usedInRisk);
}

function utilityClass(factor: Factor): UtilityClass {
  const lifecycle = (factor.lifecycle ?? "").toLowerCase();
  if (["rejected", "disabled", "deprecated", "excluded", "useless", "noise", "invalid"].some((value) => lifecycle.includes(value))) {
    return "excluded";
  }
  if (isActiveFactor(factor) || ["approved", "accepted", "selected", "production", "ready", "useful"].some((value) => lifecycle.includes(value))) {
    return "useful";
  }
  return "unevaluated";
}

function utilityStatus(factor: Factor): { label: string; status: string } {
  const classification = utilityClass(factor);
  if (classification === "excluded") return { label: "已剔除", status: "error" };
  if (isActiveFactor(factor)) return { label: "已启用", status: "ready" };
  if (classification === "useful") return { label: "有效候选", status: "ready" };
  return { label: "待评估", status: "partial" };
}

export function FactorCenterPage(): JSX.Element {
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [selectedName, setSelectedName] = useState("");
  const [utilityFilter, setUtilityFilter] = useState<UtilityFilter>("all");
  const deferredQuery = useDeferredValue(query);
  const factors = useApi<Factor[]>(["factors", deferredQuery], "/factors", { query: deferredQuery });
  const list = factors.data?.data ?? [];

  const counts = useMemo(() => ({
    all: list.length,
    active: list.filter(isActiveFactor).length,
    useful: list.filter((factor) => utilityClass(factor) === "useful").length,
    excluded: list.filter((factor) => utilityClass(factor) === "excluded").length,
    unevaluated: list.filter((factor) => utilityClass(factor) === "unevaluated").length,
  }), [list]);

  const visibleList = useMemo(() => {
    if (utilityFilter === "all") return list;
    if (utilityFilter === "active") return list.filter(isActiveFactor);
    return list.filter((factor) => utilityClass(factor) === utilityFilter);
  }, [list, utilityFilter]);

  useEffect(() => {
    if ((!selectedName || !visibleList.some((factor) => factor.name === selectedName)) && visibleList[0]) {
      setSelectedName(visibleList[0].name);
    } else if (!visibleList.length) {
      setSelectedName("");
    }
  }, [selectedName, visibleList]);

  const detail = useApi<Factor>(["factor-detail", selectedName], selectedName ? `/factors/${selectedName}` : null);
  const backtest = useApi<FactorBacktest>(["factor-backtest", selectedName], selectedName ? `/factors/${selectedName}/backtest` : null);
  const factor = detail.data?.data;
  const metrics = backtest.data?.data;
  const selectedUtility = factor ? utilityStatus(factor) : null;

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
    <div className="page factor-page factor-page-v2">
      <aside className="factor-sidebar panel">
        <div className="factor-search">
          <MagnifyingGlass size={16} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索因子名称 / 类别" />
        </div>
        <div className="factor-utility-filters" aria-label="因子有效性筛选">
          {(Object.keys(UTILITY_LABELS) as UtilityFilter[]).map((item) => (
            <button key={item} type="button" className={utilityFilter === item ? "active" : ""} onClick={() => setUtilityFilter(item)}>
              <span>{UTILITY_LABELS[item]}</span><strong>{counts[item]}</strong>
            </button>
          ))}
        </div>
        <div className="factor-count"><span>{visibleList.length} visible</span><small>{list.length} total factors</small></div>
        <div className="factor-list">
          {visibleList.map((item) => {
            const utility = utilityStatus(item);
            return (
              <button key={item.name} className={item.name === selectedName ? "active" : ""} onClick={() => setSelectedName(item.name)}>
                <div><strong>{item.displayName ?? item.name}</strong><span>{item.category ?? item.sourceKind}</span></div>
                <StatusBadge status={utility.status} label={utility.label} />
              </button>
            );
          })}
          {!visibleList.length ? <StateView state={factors.isLoading ? "loading" : "empty"} detail="当前有效性筛选没有匹配因子。" /> : null}
        </div>
        <div className="factor-sidebar-summary">
          <span>已剔除因子不会因为页面存在而自动进入训练；最终训练集仍以 dataset、feature policy 和运行配置为准。</span>
        </div>
      </aside>

      <section className="factor-content">
        <section className="factor-commandbar">
          <div>
            <span className="page-kicker">FACTOR GOVERNANCE</span>
            <strong>{selectedUtility?.label ?? "未选择因子"}</strong>
            <small>{metrics?.verdict ?? "只使用显式 lifecycle、pipeline usage 和真实评估 artifact，不凭名称判断因子有效性。"}</small>
          </div>
          <button type="button" className="secondary-button" onClick={() => navigate("/runtime?view=cleanup")}><Broom size={15} /> 清理无用产物</button>
          <button type="button" className="primary-button" onClick={() => navigate("/settings?job=train&universe=all")}><Play size={15} weight="fill" /> 全宇宙训练</button>
        </section>

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
                <MetaRow label="有效性" value={selectedUtility?.label ?? "待评估"} />
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
            {metrics?.decay.length ? <EChart option={decayOption} className="chart chart-medium" /> : <StateView state="empty" detail="没有衰减评估 artifact。" />}
          </Panel>
          <Panel title="市场环境适用性" eyebrow="Regime IC">
            {Object.keys(metrics?.regimeIc ?? {}).length ? (
              <div className="regime-cards">
                {Object.entries(metrics?.regimeIc ?? {}).map(([regime, value]) => (
                  <div key={regime}><span>{regime}</span><strong className={(value ?? 0) >= 0 ? "tone-positive" : "tone-negative"}>{formatNumber(value, 4)}</strong></div>
                ))}
              </div>
            ) : <StateView state="empty" detail="没有市场环境分层评估。" />}
          </Panel>
          <Panel title="独立回测可用性" eyebrow="No fabricated single-factor trades">
            {Object.keys(metrics?.availability ?? {}).length ? (
              <div className="availability-list">
                {Object.entries(metrics?.availability ?? {}).map(([name, available]) => (
                  <div key={name}><span>{name}</span><StatusBadge status={available ? "ready" : "partial"} label={available ? "可用" : "暂无 artifact"} /></div>
                ))}
              </div>
            ) : <StateView state="empty" detail="没有独立因子回测能力声明。" />}
            {!metrics?.availability?.trades ? (
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
