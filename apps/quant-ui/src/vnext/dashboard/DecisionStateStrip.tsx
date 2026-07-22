import { ArrowRight, Brain, Briefcase, HardDrives, ShieldWarning } from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import type { EquityPoint, JobSummary, SystemOverview } from "../../api/types";
import { formatBytes, formatDate, formatPercent } from "../../utils/format";

interface DecisionStateStripProps {
  overview: SystemOverview;
  latestPoint?: EquityPoint;
  jobs: JobSummary[];
}

export function DecisionStateStrip({ overview, latestPoint, jobs }: DecisionStateStripProps): JSX.Element {
  const backtest = overview.latestBacktest;
  const model = overview.latestModel;
  const riskEventCount = Object.values(overview.risk.eventCounts ?? {}).reduce((sum, count) => sum + count, 0);
  const activeJobs = jobs.filter((job) => ["queued", "running", "cancelling"].includes(job.status));
  const failedJobs = jobs.filter((job) => job.status === "failed");
  const staleArtifacts = overview.runtime.byFreshness?.stale ?? 0;

  return (
    <section className="vnext-decision-strip" aria-label="系统决策状态">
      <article className="vnext-decision-state state-portfolio">
        <header><span><Briefcase size={17} /> Portfolio State</span><em>{backtest ? "PERSISTED" : "UNAVAILABLE"}</em></header>
        <div className="vnext-state-primary"><strong>{latestPoint?.nav?.toLocaleString(undefined, { maximumFractionDigits: 0 }) ?? "—"}</strong><span className={(latestPoint?.dailyReturn ?? 0) >= 0 ? "tone-positive" : "tone-negative"}>{formatPercent(latestPoint?.dailyReturn)}</span></div>
        <dl><div><dt>区间收益</dt><dd>{formatPercent(backtest?.totalReturn)}</dd></div><div><dt>Benchmark excess</dt><dd>{formatPercent(latestPoint?.excessNav == null ? null : latestPoint.excessNav - 1)}</dd></div><div><dt>当前回撤</dt><dd>{formatPercent(latestPoint?.drawdown ?? backtest?.maxDrawdown)}</dd></div></dl>
        <p>{backtest ? `${backtest.name ?? backtest.id} · ${backtest.endDate ?? "unknown as-of"}` : "没有可验证回测，无法判断当前组合。"}</p>
        <Link to="/backtests">检查组合与回测 <ArrowRight size={14} /></Link>
      </article>

      <article className="vnext-decision-state state-model">
        <header><span><Brain size={17} /> Model State</span><em>{model?.productionReady ? "ACCEPTED" : model ? "RESEARCH" : "UNAVAILABLE"}</em></header>
        <div className="vnext-state-primary"><strong>{model?.version ?? "—"}</strong><span>{model?.modelFamily ?? model?.modelType ?? "no model"}</span></div>
        <dl><div><dt>最新预测</dt><dd>{model?.testEnd ?? "暂无 artifact"}</dd></div><div><dt>Acceptance</dt><dd>{model?.verdict ?? (model?.productionReady ? "pass" : "not passed")}</dd></div><div><dt>Drift / confidence</dt><dd>UNAVAILABLE</dd></div></dl>
        <p>{model?.issues[0]?.message ?? "模型状态来自 persisted registry；不推断缺失置信度。"}</p>
        <Link to="/training">检查训练与 Gate <ArrowRight size={14} /></Link>
      </article>

      <article className={`vnext-decision-state state-risk ${riskEventCount ? "attention" : ""}`}>
        <header><span><ShieldWarning size={17} /> Risk State</span><em>{riskEventCount ? "ATTENTION" : "CLEAR"}</em></header>
        <div className="vnext-state-primary"><strong>{riskEventCount}</strong><span>active violations</span></div>
        <dl><div><dt>最大回撤</dt><dd>{formatPercent(overview.risk.maxDrawdown)}</dd></div><div><dt>流动性风险</dt><dd>{formatPercent(overview.risk.liquidityRisk)}</dd></div><div><dt>Stale data</dt><dd>{staleArtifacts}</dd></div></dl>
        <p>{riskEventCount ? "存在持久化风险事件，需要检查具体规则与拦截原因。" : "没有已记录违规；缺失指标仍保持 unavailable。"}</p>
        <Link to="/risk">打开 Risk Manager <ArrowRight size={14} /></Link>
      </article>

      <article className={`vnext-decision-state state-operations ${failedJobs.length ? "attention" : ""}`}>
        <header><span><HardDrives size={17} /> Operations State</span><em>{failedJobs.length ? "DEGRADED" : "NOMINAL"}</em></header>
        <div className="vnext-state-primary"><strong>{activeJobs.length}</strong><span>active jobs</span></div>
        <dl><div><dt>失败任务</dt><dd>{failedJobs.length}</dd></div><div><dt>Runtime</dt><dd>{formatBytes(overview.runtime.totalSizeBytes)}</dd></div><div><dt>Manifest</dt><dd>{overview.runtime.manifestCoverage === undefined ? "UNAVAILABLE" : formatPercent(overview.runtime.manifestCoverage, 0)}</dd></div></dl>
        <p>Index as-of {formatDate(overview.runtime.indexedAt)} · API/WS 状态显示在全局命令栏。</p>
        <Link to="/settings?view=jobs">处理任务与系统问题 <ArrowRight size={14} /></Link>
      </article>
    </section>
  );
}
