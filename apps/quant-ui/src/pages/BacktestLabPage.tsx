import { useEffect, useState } from "react";
import { ChartLineUp, DownloadSimple, Flask, Play, ShieldCheck, WarningCircle } from "@phosphor-icons/react";
import { useNavigate, useSearchParams } from "react-router-dom";
import type { BacktestSummary, EquityPoint } from "../api/types";
import { downloadJson } from "../api/client";
import { useApi } from "../hooks/useApi";
import { EquityChart } from "../components/EquityChart";
import { Panel } from "../components/Panel";
import { StateView } from "../components/StateView";
import { StatusBadge } from "../components/StatusBadge";
import { formatCompact, formatNumber, formatPercent } from "../utils/format";
import { ActionableState, WorkbenchHeader, WorkbenchMetricStrip } from "../vnext/workbench/InstitutionalWorkbench";

export function BacktestLabPage(): JSX.Element {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const backtests = useApi<BacktestSummary[]>(["backtest-lab"], "/backtests");
  const [selectedId, setSelectedId] = useState(searchParams.get("run") ?? "");
  const runs = backtests.data?.data ?? [];

  useEffect(() => {
    if ((!selectedId || !runs.some((run) => run.id === selectedId)) && runs[0]) {
      setSelectedId(runs[0].id);
    }
  }, [runs, selectedId]);

  const primary = runs.find((run) => run.id === selectedId) ?? runs[0];
  const selectRun = (id: string): void => {
    setSelectedId(id);
    const next = new URLSearchParams(searchParams);
    next.set("run", id);
    setSearchParams(next, { replace: true });
  };
  const equity = useApi<EquityPoint[]>(
    ["backtest-lab-equity", primary?.id],
    primary ? `/backtests/${primary.id}/equity` : null,
  );

  if (backtests.isLoading) return <StateView state="loading" />;
  if (!runs.length) return <div className="institutional-workbench"><WorkbenchHeader eyebrow="BACKTEST WORKSTATION / STRICT A-SHARE" title="回测工作站" description="单一活动实验上下文、成本与 T+1 约束、可追踪 artifact。" context="no fabricated metrics" /><ActionableState title="没有可识别回测实验" detail="从经过 allowlist 与路径校验的严格 A 股回测任务开始。" icon={Flask} primary={{ label: "配置回测任务", onClick: () => navigate("/settings?job=backtest") }} secondary={{ label: "检查 Runtime", onClick: () => navigate("/runtime?kind=backtest") }} /></div>;

  return (
    <div className="page institutional-workbench backtest-page backtest-page-v2">
      <WorkbenchHeader eyebrow="BACKTEST WORKSTATION / STRICT A-SHARE" title="回测工作站" description="单一活动实验驱动主图、指标和详情；多实验只进入独立 Compare，不再混合上下文。" asOf={primary?.endDate?.slice(0, 10)} context={primary?.trustClass ?? "research experiment"} actions={<><button type="button" onClick={() => primary && downloadJson("backtest-experiment.json", primary)}><DownloadSimple size={14} />导出当前</button><button type="button" className="primary" onClick={() => navigate("/settings?job=backtest")}><Play size={14} weight="fill" />新建回测</button></>} />
      <WorkbenchMetricStrip metrics={[
        { label: "总收益", value: formatPercent(primary?.totalReturn), detail: primary?.name ?? "active run", tone: toneName(primary?.totalReturn), icon: ChartLineUp },
        { label: "年化收益", value: formatPercent(primary?.annualReturn), detail: primary?.horizon ?? "horizon unknown", tone: toneName(primary?.annualReturn), icon: ChartLineUp },
        { label: "最大回撤", value: formatPercent(primary?.maxDrawdown), detail: "strict NAV", tone: "danger", icon: WarningCircle },
        { label: "Sharpe", value: formatNumber(primary?.sharpe), detail: `Calmar ${formatNumber(primary?.calmar)}`, tone: "info", icon: ChartLineUp },
        { label: "换手率", value: formatPercent(primary?.turnover), detail: "cost-sensitive", tone: "warning", icon: ShieldCheck },
        { label: "成交数量", value: formatCompact(primary?.tradeCount), detail: `${formatCompact(primary?.fillCount)} fills`, tone: primary?.capabilities?.trades ? "positive" : "neutral", icon: Flask },
      ]} />

      <section className="backtest-grid">
        <Panel title="实验净值" eyebrow={`${primary?.name} · ${primary?.startDate?.slice(0, 10) ?? "未知"} → ${primary?.endDate?.slice(0, 10) ?? "未知"}`} className="backtest-equity-panel">
          {equity.data?.data.length ? <EquityChart points={equity.data.data} height={286} showDrawdown /> : <ActionableState title="该实验没有 NAV artifact" detail="主指标保持暂无，不从成交或收益摘要反推净值。可切换其他实验，或检查该 run 的产物契约。" icon={ChartLineUp} primary={{ label: "检查 Runtime", onClick: () => navigate(`/runtime?runId=${encodeURIComponent(primary?.id ?? "")}`) }} />}
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
          title="实验浏览"
          eyebrow={`单一活动实验 · ${runs.length} runtime runs`}
          className="backtest-table-panel"
          actions={<button className="secondary-button" onClick={() => primary && downloadJson("backtest-experiment.json", primary)}><DownloadSimple size={15} /> 导出当前</button>}
        >
          <div className="backtest-context-note">选择一行会替换当前实验上下文；不再通过复选框同时激活多个实验。多实验统计对比应使用独立 Compare 工作区，避免主图、指标卡和详情来源不一致。</div>
          <div className="table-scroll" role="radiogroup" aria-label="当前回测实验">
            <table className="data-table comparison-table single-select-table">
              <thead>
                <tr>
                  <th>当前</th>
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
                  const selected = run.id === primary?.id;
                  return (
                    <tr
                      key={run.id}
                      className={selected ? "row-selected" : ""}
                      aria-selected={selected}
                      tabIndex={0}
                      onClick={() => selectRun(run.id)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" || event.key === " ") {
                          event.preventDefault();
                          selectRun(run.id);
                        }
                      }}
                    >
                      <td><input type="radio" name="active-backtest" checked={selected} readOnly aria-label={`选择实验 ${run.name ?? run.id}`} /></td>
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

function toneName(value: number | null | undefined): "positive" | "danger" | "neutral" {
  if (value === null || value === undefined) return "neutral";
  return value >= 0 ? "positive" : "danger";
}
