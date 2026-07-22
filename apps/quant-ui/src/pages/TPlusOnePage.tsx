import { useEffect, useMemo, useState } from "react";
import { ArrowRight, ArrowsClockwise, CheckCircle, ClockCounterClockwise, Database, ShieldCheck, TrendDown, WarningCircle } from "@phosphor-icons/react";
import type { EChartsOption } from "echarts";
import { useNavigate } from "react-router-dom";
import { useApi } from "../hooks/useApi";
import { EChart } from "../components/EChart";
import { Panel } from "../components/Panel";
import { StateView } from "../components/StateView";
import { StatusBadge } from "../components/StatusBadge";
import { formatCompact, formatNumber, formatPercent } from "../utils/format";
import { ActionableState, TruthNotice, WorkbenchHeader, WorkbenchMetricStrip, WorkbenchPanel } from "../vnext/workbench/InstitutionalWorkbench";

interface DoTSource {
  id: string;
  name: string;
  path: string;
  verdict?: string | null;
  reason?: string | null;
  metrics: Record<string, number | null>;
  modifiedAt: number;
}

interface DoTPair {
  id: string;
  symbol: string;
  tradeDate: string;
  mode: string;
  buyTime?: string | null;
  sellTime?: string | null;
  buyPrice?: number | null;
  sellPrice?: number | null;
  quantity?: number | null;
  grossPnl?: number | null;
  cost?: number | null;
  netPnl?: number | null;
  edgePct?: number | null;
  state?: string | null;
  highSellFailed?: boolean | null;
  lowBuyFailed?: boolean | null;
  missedUpsidePct?: number | null;
  adverseMovePct?: number | null;
  success?: boolean | null;
  issues?: Array<{ message: string }>;
}

interface DoTAnalysis {
  sourceId: string;
  symbol?: string | null;
  summary: {
    pairCount: number;
    successRate?: number | null;
    failureRate?: number | null;
    highSellFailureRate?: number | null;
    lowBuyFailureRate?: number | null;
    totalNetPnl?: number | null;
    returnContribution?: number | null;
    drawdownContribution?: number | null;
    qualityScore?: number | null;
  };
  pairs: DoTPair[];
  byRegime: Record<string, unknown>;
  verdict?: string | null;
  reason?: string | null;
}

export function TPlusOnePage(): JSX.Element {
  const navigate = useNavigate();
  const sources = useApi<DoTSource[]>(["t1-sources"], "/do-t/sources");
  const [sourceId, setSourceId] = useState("");
  const [symbol, setSymbol] = useState("");
  const [pairPage, setPairPage] = useState(1);

  useEffect(() => {
    if (!sourceId && sources.data?.data[0]) setSourceId(sources.data.data[0].id);
  }, [sourceId, sources.data?.data]);

  const analysis = useApi<DoTAnalysis>(
    ["t1-analysis", sourceId, symbol],
    sourceId ? "/do-t/analysis" : null,
    { sourceId, symbol: symbol || undefined, limit: 1000 },
  );
  const data = analysis.data?.data;
  const symbols = useMemo(
    () => [...new Set((data?.pairs ?? []).map((pair) => pair.symbol))].sort(),
    [data?.pairs],
  );
  const pairPageSize = 100;
  const pairPageCount = Math.max(1, Math.ceil((data?.pairs.length ?? 0) / pairPageSize));
  const visiblePairs = (data?.pairs ?? []).slice(
    (pairPage - 1) * pairPageSize,
    pairPage * pairPageSize,
  );

  const waterfall = useMemo<EChartsOption>(() => ({
    animation: false,
    grid: { left: 52, right: 16, top: 20, bottom: 50 },
    tooltip: { trigger: "axis", backgroundColor: "#0b1824", borderColor: "#27425a", textStyle: { color: "#d7e4ef" } },
    xAxis: {
      type: "category",
      data: (data?.pairs ?? []).slice(0, 80).map((pair) => `${pair.tradeDate?.slice(5, 10)} ${pair.symbol.slice(0, 6)}`),
      axisLabel: { color: "#71879a", fontSize: 9, hideOverlap: true, formatter: (value: string) => value.slice(0, 5) },
      axisLine: { lineStyle: { color: "#20364a" } },
    },
    yAxis: {
      type: "value",
      axisLabel: { color: "#71879a", fontSize: 10 },
      splitLine: { lineStyle: { color: "#14283a" } },
    },
    dataZoom: [
      { type: "inside", zoomOnMouseWheel: true, moveOnMouseMove: true, moveOnMouseWheel: false, start: (data?.pairs.length ?? 0) > 40 ? 50 : 0, end: 100 },
      { type: "slider", bottom: 3, height: 14, showDetail: false, brushSelect: false, start: (data?.pairs.length ?? 0) > 40 ? 50 : 0, end: 100 },
    ],
    series: [{
      type: "bar",
      data: (data?.pairs ?? []).slice(0, 80).map((pair) => ({
        value: pair.netPnl ?? 0,
        itemStyle: { color: (pair.netPnl ?? 0) >= 0 ? "#28c79a" : "#ef5c63" },
      })),
      barMaxWidth: 12,
    }],
  }), [data?.pairs]);

  if (sources.isLoading) return <StateView state="loading" />;
  if (!sources.data?.data.length) return <EmptyTPlusOneWorkspace navigate={navigate} />;

  return (
    <div className="page institutional-workbench t1-page">
      <WorkbenchHeader eyebrow="T+1 COMPLIANT OVERLAY / FAILURE CONTROL" title="T+1 分析工作站" description="卖出只使用昨仓 sellable inventory；逐笔证据、失败类型与风险阈值共享同一运行上下文。" asOf={data?.pairs[0]?.tradeDate?.slice(0, 10) ?? "as-of unavailable"} context={data?.verdict ?? "research verdict"} />
      <section className="workbench-toolbar">
        <label>
          <span>研究数据源</span>
          <select value={sourceId} onChange={(event) => { setSourceId(event.target.value); setSymbol(""); setPairPage(1); }}>
            {sources.data.data.map((source) => <option value={source.id} key={source.id}>{source.name}</option>)}
          </select>
        </label>
        <label>
          <span>股票过滤</span>
          <select value={symbol} onChange={(event) => { setSymbol(event.target.value); setPairPage(1); }}>
            <option value="">全部标的</option>
            {symbols.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
        </label>
        <div className="t1-safety-note">
          <ArrowsClockwise size={18} />
          <span><strong>T+1 合规约束</strong>：卖出只使用昨仓 sellable inventory；当日买入不得当日卖出。</span>
        </div>
        <StatusBadge status={data?.verdict ?? "partial"} label={data?.verdict ?? "读取中"} />
      </section>

      <WorkbenchMetricStrip metrics={[
        { label: "做 T 对数", value: formatCompact(data?.summary.pairCount), detail: "persisted pairs", tone: "info", icon: ArrowsClockwise },
        { label: "成功率", value: formatPercent(data?.summary.successRate), detail: "pair-level", tone: "positive", icon: CheckCircle },
        { label: "失败率", value: formatPercent(data?.summary.failureRate), detail: "failure control", tone: "danger", icon: WarningCircle },
        { label: "高抛失败", value: formatPercent(data?.summary.highSellFailureRate), detail: "threshold 15%", tone: "warning", icon: TrendDown },
        { label: "低吸失败", value: formatPercent(data?.summary.lowBuyFailureRate), detail: "threshold 15%", tone: "warning", icon: TrendDown },
        { label: "信号质量", value: formatNumber(data?.summary.qualityScore), detail: `return ${formatPercent(data?.summary.returnContribution)}`, tone: "info", icon: ShieldCheck },
      ]} />

      <section className="t1-grid">
        <Panel title="每笔 T+1 做 T 收益" eyebrow="Pair-level waterfall · only persisted fills" className="t1-waterfall">
          {data?.pairs.length ? <EChart option={waterfall} className="chart" /> : <StateView state={analysis.isLoading ? "loading" : "empty"} />}
        </Panel>

        <Panel title="失败控制" eyebrow="T-Pair Failure Control" className="t1-control">
          <div className="failure-control-list">
            <FailureControl label="高抛后继续上涨" value={data?.summary.highSellFailureRate} threshold={0.15} />
            <FailureControl label="低吸后继续下跌" value={data?.summary.lowBuyFailureRate} threshold={0.15} />
            <FailureControl label="整体失败率" value={data?.summary.failureRate} threshold={0.5} />
            <FailureControl label="回撤控制贡献" value={data?.summary.drawdownContribution} threshold={0} />
          </div>
          <div className="t1-verdict">
            {data?.verdict?.includes("ENABLE") ? <CheckCircle size={23} className="tone-positive" /> : <WarningCircle size={23} className="tone-warning" />}
            <div><strong>{data?.verdict ?? "暂无 verdict"}</strong><span>{data?.reason ?? "严格遵循 runtime report，不从收益猜测结论。"}</span></div>
          </div>
        </Panel>

        <Panel title="T+1 交易对明细" eyebrow="Entry / Exit / Quantity / Failure State" className="t1-table-panel">
          {data?.pairs.length ? (
            <>
              <div className="table-scroll">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>日期 / 标的</th><th>模式</th><th>买入</th><th>卖出</th>
                      <th className="numeric">数量</th><th className="numeric">净收益</th>
                      <th>状态</th><th>失败识别</th><th>数据质量</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visiblePairs.map((pair) => (
                      <tr key={pair.id}>
                        <td><strong>{pair.symbol}</strong><span>{pair.tradeDate?.slice(0, 10)}</span></td>
                        <td>{pair.mode}</td>
                        <td><strong className="tone-positive mono">{formatNumber(pair.buyPrice)}</strong><span>{pair.buyTime?.slice(11, 19) ?? "暂无时间"}</span></td>
                        <td><strong className="tone-negative mono">{formatNumber(pair.sellPrice)}</strong><span>{pair.sellTime?.slice(11, 19) ?? "暂无时间"}</span></td>
                        <td className="numeric mono">{formatCompact(pair.quantity)}</td>
                        <td className={`numeric mono ${(pair.netPnl ?? 0) >= 0 ? "tone-positive" : "tone-negative"}`}>{formatNumber(pair.netPnl)}</td>
                        <td><StatusBadge status={pair.success ? "success" : pair.success === false ? "failed" : "partial"} label={pair.state ?? "unknown"} /></td>
                        <td>{pair.highSellFailed ? "高抛失败" : pair.lowBuyFailed ? "低吸失败" : pair.success ? "成功" : "暂无判定"}</td>
                        <td>{pair.issues?.[0]?.message ?? "minute fill available"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="pagination">
                <button disabled={pairPage <= 1} onClick={() => setPairPage((value) => Math.max(1, value - 1))}>上一页</button>
                <span>{pairPage} / {pairPageCount}</span>
                <button disabled={pairPage >= pairPageCount} onClick={() => setPairPage((value) => Math.min(pairPageCount, value + 1))}>下一页</button>
              </div>
            </>
          ) : <StateView state="empty" />}
        </Panel>
      </section>
    </div>
  );
}

function EmptyTPlusOneWorkspace({ navigate }: { navigate: ReturnType<typeof useNavigate> }): JSX.Element {
  return <div className="institutional-workbench t1-page t1-empty-page">
    <WorkbenchHeader eyebrow="T+1 COMPLIANT OVERLAY / FAILURE CONTROL" title="T+1 分析工作站" description="卖出只使用昨仓 sellable inventory；缺失成交证据时展示准备路径，不从日线收益伪造分钟成交。" context="paper / research only" />
    <WorkbenchMetricStrip metrics={[
      { label: "交易对", value: "0", detail: "persisted fills only", tone: "neutral", icon: ArrowsClockwise },
      { label: "昨仓库存", value: "—", detail: "sellable inventory required", tone: "warning", icon: Database },
      { label: "分钟成交", value: "—", detail: "entry / exit required", tone: "warning", icon: ClockCounterClockwise },
      { label: "失败控制", value: "LOCKED", detail: "thresholds unavailable", tone: "warning", icon: WarningCircle },
      { label: "收益贡献", value: "—", detail: "never inferred", tone: "neutral", icon: TrendDown },
      { label: "实盘", value: "DISABLED", detail: "research evidence only", tone: "positive", icon: ShieldCheck },
    ]} />
    <section className="t1-empty-grid">
      <WorkbenchPanel eyebrow="EVIDENCE CONTRACT" title="T+1 成交证据链" meta="fail closed">
        <div className="t1-empty-timeline"><article><span>01</span><strong>昨仓快照</strong><small>sellable inventory · as-of open</small></article><ArrowRight /><article><span>02</span><strong>低吸成交</strong><small>minute/tick fill + quantity</small></article><ArrowRight /><article><span>03</span><strong>高抛成交</strong><small>prior inventory only</small></article><ArrowRight /><article><span>04</span><strong>失败归因</strong><small>adverse move / missed upside</small></article></div>
        <TruthNotice tone="warning">缺少任一成交、时间或库存字段时不计算 pair PnL，也不显示可启用结论。</TruthNotice>
      </WorkbenchPanel>
      <WorkbenchPanel eyebrow="PAIR FAILURE CONTROL" title="风险阈值" meta="awaiting artifacts">
        <div className="t1-empty-thresholds"><span>高抛后继续上涨<i>≤ 15%</i></span><span>低吸后继续下跌<i>≤ 15%</i></span><span>整体失败率<i>≤ 50%</i></span><span>回撤控制贡献<i>source-backed</i></span></div>
      </WorkbenchPanel>
      <WorkbenchPanel eyebrow="NEXT ACTION" title="恢复工作上下文" meta="runtime source">
        <ActionableState compact title="没有 T+1 做 T artifact" detail="先检查 do_t 成交与库存产物，再从受治理 overlay 生成逐对证据。" icon={ArrowsClockwise} primary={{ label: "检查 Runtime", onClick: () => navigate("/runtime?kind=do_t") }} secondary={{ label: "检查任务", onClick: () => navigate("/settings?view=jobs") }} />
      </WorkbenchPanel>
    </section>
  </div>;
}

function FailureControl({ label, value, threshold }: { label: string; value: number | null | undefined; threshold: number }): JSX.Element {
  const safe = value === null || value === undefined ? null : value <= threshold;
  return (
    <div className="failure-control-row">
      <span>{label}</span>
      <div><i style={{ width: `${Math.min(100, Math.max(0, (value ?? 0) * 100))}%` }} /></div>
      <strong className={safe === null ? "" : safe ? "tone-positive" : "tone-negative"}>{formatPercent(value)}</strong>
      <small>阈值 {formatPercent(threshold)}</small>
    </div>
  );
}
