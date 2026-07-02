import { useEffect, useMemo, useState } from "react";
import { ArrowRight, CheckCircle, XCircle } from "@phosphor-icons/react";
import type { EChartsOption } from "echarts";
import type { SelectionRun } from "../api/types";
import { useApi } from "../hooks/useApi";
import { EChart } from "../components/EChart";
import { Panel } from "../components/Panel";
import { SelectionFunnel } from "../components/SelectionFunnel";
import { StateView } from "../components/StateView";
import { StatusBadge } from "../components/StatusBadge";
import { formatNumber } from "../utils/format";

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
  if (!runs.data?.data.length) return <StateView state="empty" detail="没有 hybrid_stock_pool selection run。" />;

  return (
    <div className="page selection-page">
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

function ScorePill({ label, value, reverse = false }: { label: string; value?: number | null; reverse?: boolean }): JSX.Element {
  const width = Math.min(100, Math.max(0, (value ?? 0) * 100));
  return <div><span>{label}</span><i><b style={{ width: `${width}%` }} /></i><strong className={reverse ? "tone-warning" : "tone-positive"}>{formatNumber(value)}</strong></div>;
}
