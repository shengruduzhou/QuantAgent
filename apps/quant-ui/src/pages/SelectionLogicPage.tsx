import { useEffect, useMemo, useState } from "react";
import { ArrowRight, Brain, CheckCircle, Database, FunnelSimple, ShieldCheck, Target, XCircle } from "@phosphor-icons/react";
import type { EChartsOption } from "echarts";
import type { SelectionRun } from "../api/types";
import { useNavigate } from "react-router-dom";
import { useApi } from "../hooks/useApi";
import { EChart } from "../components/EChart";
import { Panel } from "../components/Panel";
import { SelectionFunnel } from "../components/SelectionFunnel";
import { StateView } from "../components/StateView";
import { StatusBadge } from "../components/StatusBadge";
import { formatNumber } from "../utils/format";
import { ActionableState, TruthNotice, WorkbenchHeader, WorkbenchMetricStrip, WorkbenchPanel } from "../vnext/workbench/InstitutionalWorkbench";

interface RankingRow {
  symbol: string;
  name?: string | null;
  sector?: string | null;
  modelRank?: number | null;
  modelScore?: number | null;
  factorScore?: number | null;
  llmScore?: number | null;
  confidence?: number | null;
  riskScore?: number | null;
  doTSuitability?: number | null;
  finalScore?: number | null;
  finalRank?: number | null;
  actionBucket?: string | null;
  included: boolean;
  exclusionReason?: string | null;
  factorContributions: Record<string, number>;
  noOrdersGenerated: boolean;
}

interface FunnelStage {
  stage: string;
  count: number | null;
  reason?: string | null;
}

interface DecisionChain {
  symbol: string;
  datetime?: string | null;
  finalDecision?: string | null;
  traceType: string;
  gates: Array<{ order: number; name: string; passed: boolean; reason?: string | null; detail: Record<string, unknown> }>;
  issues?: Array<{ message: string }>;
}

export function SelectionLogicPage(): JSX.Element {
  const navigate = useNavigate();
  const runs = useApi<SelectionRun[]>(["selection-runs"], "/selection/runs");
  const [runId, setRunId] = useState("");
  const [symbol, setSymbol] = useState("");

  useEffect(() => {
    if (!runId && runs.data?.data[0]) setRunId(runs.data.data[0].id);
  }, [runId, runs.data?.data]);

  const funnel = useApi<FunnelStage[]>(["selection-funnel", runId], runId ? `/selection/runs/${runId}/funnel` : null);
  const ranking = useApi<RankingRow[]>(["selection-ranking", runId], runId ? `/selection/runs/${runId}/ranking` : null, { limit: 1000 });

  useEffect(() => {
    if ((!symbol || !ranking.data?.data.some((row) => row.symbol === symbol)) && ranking.data?.data[0]) {
      setSymbol(ranking.data.data[0].symbol);
    }
  }, [ranking.data?.data, symbol]);

  const chain = useApi<DecisionChain>(
    ["selection-chain", runId, symbol],
    runId && symbol ? `/selection/runs/${runId}/stocks/${symbol}/decision-chain` : null,
  );
  const selected = ranking.data?.data.find((row) => row.symbol === symbol);

  const sectorOption = useMemo<EChartsOption>(() => {
    const counts = new Map<string, number>();
    for (const row of ranking.data?.data ?? []) {
      const sector = row.sector ?? "未知行业";
      counts.set(sector, (counts.get(sector) ?? 0) + 1);
    }
    const data = [...counts.entries()].sort((left, right) => right[1] - left[1]).slice(0, 12);
    return {
      animation: false,
      grid: { left: 76, right: 18, top: 16, bottom: 24 },
      tooltip: { trigger: "axis", backgroundColor: "#0b1824", borderColor: "#27425a", textStyle: { color: "#d7e4ef" } },
      xAxis: { type: "value", axisLabel: { color: "#71879a" }, splitLine: { lineStyle: { color: "#14283a" } } },
      yAxis: { type: "category", inverse: true, data: data.map(([name]) => name), axisLabel: { color: "#9cb1c3", fontSize: 10 } },
      series: [{ type: "bar", data: data.map(([, value]) => value), itemStyle: { color: "#3f8cff" }, barMaxWidth: 15 }],
    };
  }, [ranking.data?.data]);

  if (runs.isLoading) return <StateView state="loading" />;
  if (!runs.data?.data.length) return <EmptySelectionWorkspace navigate={navigate} />;

  return (
    <div className="page institutional-workbench selection-page">
      <WorkbenchHeader eyebrow="STRATEGY RESEARCH / SELECTION" title="策略与选股工作站" description="Universe → liquidity → risk → factor → model → portfolio；输出研究排名与 target hint，不直接生成订单。" asOf={runs.data.data.find((run) => run.id === runId)?.asOfDate ?? "as-of unavailable"} context="PIT / T+1 / human gated" />
      <WorkbenchMetricStrip metrics={[
        { label: "运行目录", value: String(runs.data.data.length), detail: "persisted selection runs", tone: "info" },
        { label: "最终候选", value: String(ranking.data?.data.filter((row) => row.included).length ?? 0), detail: `${ranking.data?.data.length ?? 0} ranked names`, tone: "positive" },
        { label: "漏斗关卡", value: String(funnel.data?.data.length ?? 0), detail: "source-backed stages", tone: "info" },
        { label: "Fallback", value: runs.data.data.find((run) => run.id === runId)?.usedFallback ? "已使用" : "未使用", detail: "explicit metadata", tone: runs.data.data.find((run) => run.id === runId)?.usedFallback ? "warning" : "positive" },
        { label: "选中排名", value: selected?.finalRank ? `#${selected.finalRank}` : "—", detail: selected?.symbol ?? "no selection", tone: "info" },
        { label: "决策链", value: String(chain.data?.data.gates.length ?? 0), detail: chain.data?.data.finalDecision ?? "trace unavailable", tone: chain.data?.data ? "positive" : "neutral" },
      ]} />
      <section className="workbench-toolbar">
        <label>
          <span>选股运行</span>
          <select value={runId} onChange={(event) => { setRunId(event.target.value); setSymbol(""); }}>
            {runs.data.data.map((run) => <option key={run.id} value={run.id}>{run.asOfDate ?? "unknown date"} · {run.finalCount ?? 0} stocks</option>)}
          </select>
        </label>
        <StatusBadge status={runs.data.data.find((run) => run.id === runId)?.usedFallback ? "warning" : "ready"} label={runs.data.data.find((run) => run.id === runId)?.usedFallback ? "Fallback used" : "Source-backed"} />
        <div className="truth-note"><CheckCircle size={17} /><span>Selection 只输出 research ranking / target hint，不生成订单。</span></div>
      </section>

      <section className="selection-top-grid">
        <Panel title="透明选股漏斗" eyebrow="Universe → liquidity → risk → factor → model → portfolio">
          {funnel.data?.data.length ? <SelectionFunnel stages={funnel.data.data} /> : <StateView state="empty" />}
        </Panel>
        <Panel title="行业分布" eyebrow="Final research pool">
          <EChart option={sectorOption} className="chart chart-medium" />
        </Panel>
        <Panel title="选中标的" eyebrow={symbol || "选择 ranking row"}>
          {selected ? (
            <div className="selected-stock-card">
              <div><strong>{selected.symbol}</strong><span>{selected.sector ?? "行业暂无"}</span></div>
              <div className="score-orbit">
                <ScorePill label="模型" value={selected.modelScore} />
                <ScorePill label="因子" value={selected.factorScore} />
                <ScorePill label="风险" value={selected.riskScore} reverse />
                <ScorePill label="T+1" value={selected.doTSuitability} />
              </div>
              <div className="final-rank"><span>最终排名</span><strong>#{selected.finalRank ?? "—"}</strong><em>{selected.actionBucket ?? "research"}</em></div>
            </div>
          ) : <StateView state="empty" />}
        </Panel>
      </section>

      <section className="selection-main-grid">
        <Panel title="股票排名" eyebrow={`${ranking.data?.data.length ?? 0} names · 点击查看决策链`} className="selection-ranking-panel">
          {ranking.data?.data.length ? (
            <div className="table-scroll">
              <table className="data-table">
                <thead><tr><th>排名</th><th>股票</th><th>行业</th><th className="numeric">模型</th><th className="numeric">因子</th><th className="numeric">风险</th><th className="numeric">T+1</th><th className="numeric">综合分</th><th>动作桶</th></tr></thead>
                <tbody>
                  {ranking.data.data.map((row) => (
                    <tr
                      key={row.symbol}
                      className={row.symbol === symbol ? "row-selected" : ""}
                      onClick={() => setSymbol(row.symbol)}
                      tabIndex={0}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") setSymbol(row.symbol);
                      }}
                    >
                      <td className="mono">#{row.finalRank ?? row.modelRank ?? "—"}</td>
                      <td><strong>{row.symbol}</strong><span>{row.name ?? "名称暂无"}</span></td>
                      <td>{row.sector ?? "未知"}</td>
                      <td className="numeric mono">{formatNumber(row.modelScore)}</td>
                      <td className="numeric mono">{formatNumber(row.factorScore)}</td>
                      <td className="numeric mono tone-warning">{formatNumber(row.riskScore)}</td>
                      <td className="numeric mono">{formatNumber(row.doTSuitability)}</td>
                      <td className="numeric mono tone-positive">{formatNumber(row.finalScore)}</td>
                      <td><StatusBadge status={row.included ? "ready" : "partial"} label={row.actionBucket ?? "research"} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <StateView state={ranking.isLoading ? "loading" : "empty"} />}
        </Panel>

        <Panel title="完整决策链" eyebrow={`${chain.data?.data.traceType ?? "no trace"} · ${chain.data?.data.datetime ?? ""}`} className="selection-chain-panel">
          {chain.data?.data ? (
            <div className="vertical-chain">
              {chain.data.data.gates.map((gate, index) => (
                <div className={`vertical-chain-step ${gate.passed ? "chain-pass" : "chain-fail"}`} key={`${gate.order}-${gate.name}`}>
                  <div className="chain-icon">{gate.passed ? <CheckCircle size={18} weight="fill" /> : <XCircle size={18} />}</div>
                  <div className="chain-copy">
                    <span>STEP {gate.order}</span>
                    <strong>{gate.name}</strong>
                    <small>{gate.reason ?? JSON.stringify(gate.detail)}</small>
                  </div>
                  {index < chain.data.data.gates.length - 1 ? <ArrowRight size={15} className="chain-arrow" /> : null}
                </div>
              ))}
              <div className="chain-result"><span>最终决策</span><strong>{chain.data.data.finalDecision ?? "research only"}</strong></div>
              {chain.data.data.issues?.map((issue) => <p className="muted-copy" key={issue.message}>{issue.message}</p>)}
            </div>
          ) : <StateView state={chain.isLoading ? "loading" : "empty"} />}
        </Panel>
      </section>
    </div>
  );
}

function EmptySelectionWorkspace({ navigate }: { navigate: ReturnType<typeof useNavigate> }): JSX.Element {
  const gates = [
    { icon: Database, order: "01", title: "Universe / PIT", detail: "等待 hybrid stock pool 与 as-of 契约" },
    { icon: ShieldCheck, order: "02", title: "Liquidity / Risk", detail: "等待流动性、ST、涨跌停与风险事件" },
    { icon: FunnelSimple, order: "03", title: "Factor scoring", detail: "等待已审核因子和 feature policy" },
    { icon: Brain, order: "04", title: "Model ranking", detail: "等待注册模型、预测与校准证据" },
    { icon: Target, order: "05", title: "Research portfolio", detail: "只输出 ranking / target hint，不生成订单" },
  ];
  return <div className="institutional-workbench selection-page selection-empty-page">
    <WorkbenchHeader eyebrow="STRATEGY RESEARCH / SELECTION" title="策略与选股工作站" description="Universe → liquidity → risk → factor → model → portfolio；缺少运行时展示可执行的研究准备路径。" context="research ranking only" />
    <WorkbenchMetricStrip metrics={[
      { label: "Selection runs", value: "0", detail: "persisted runs", tone: "neutral", icon: FunnelSimple },
      { label: "Universe", value: "—", detail: "artifact required", tone: "warning", icon: Database },
      { label: "Risk gates", value: "—", detail: "evidence required", tone: "warning", icon: ShieldCheck },
      { label: "Factor policy", value: "—", detail: "reviewed factors only", tone: "ai", icon: FunnelSimple },
      { label: "Model rank", value: "—", detail: "registered model only", tone: "ai", icon: Brain },
      { label: "Orders", value: "LOCKED", detail: "research output only", tone: "positive", icon: Target },
    ]} />
    <section className="selection-empty-grid">
      <WorkbenchPanel eyebrow="DECISION PIPELINE" title="研究决策链" meta="waiting for persisted run" className="selection-empty-chain">
        <div className="selection-empty-gates">{gates.map(({ icon: GateIcon, order, title, detail }) => <article key={order}><span>{order}</span><GateIcon size={18} weight="duotone" /><div><strong>{title}</strong><small>{detail}</small></div><i>PENDING</i></article>)}</div>
      </WorkbenchPanel>
      <WorkbenchPanel eyebrow="RUN READINESS" title="启动前证据" meta="fail closed">
        <div className="selection-readiness-list"><span><CheckCircle size={15} />PIT universe contract</span><span><CheckCircle size={15} />Tradability and T+1 guards</span><span><CheckCircle size={15} />Reviewed feature policy</span><span><CheckCircle size={15} />Model registry acceptance</span><span><CheckCircle size={15} />Persisted ranking artifact</span></div>
        <TruthNotice tone="warning">任何一项不可用时保持研究态，不补造排名、评分或订单。</TruthNotice>
      </WorkbenchPanel>
      <WorkbenchPanel eyebrow="NEXT ACTION" title="恢复工作上下文" meta="two source-backed paths">
        <ActionableState compact title="没有 Selection run" detail="检查数据与 lineage，或从受治理任务配置创建研究运行。" icon={FunnelSimple} primary={{ label: "检查 Pipeline", onClick: () => navigate("/runtime?view=lineage") }} secondary={{ label: "检查 Runtime", onClick: () => navigate("/runtime?kind=selection") }} />
      </WorkbenchPanel>
    </section>
  </div>;
}

function ScorePill({ label, value, reverse = false }: { label: string; value?: number | null; reverse?: boolean }): JSX.Element {
  const width = Math.min(100, Math.max(0, (value ?? 0) * 100));
  return <div><span>{label}</span><i><b style={{ width: `${width}%` }} /></i><strong className={reverse ? "tone-warning" : "tone-positive"}>{formatNumber(value)}</strong></div>;
}
