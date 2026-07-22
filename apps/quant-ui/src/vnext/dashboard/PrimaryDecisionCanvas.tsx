import { ArrowRight, ChartLine, CircleNotch, WarningCircle } from "@phosphor-icons/react";
import { Link } from "react-router-dom";
import type { BacktestSummary, EquityPoint, JobSummary, ModelSummary, Page, RiskOverview, Trade } from "../../api/types";
import { EquityChart } from "../../components/EquityChart";
import { StateView } from "../../components/StateView";
import { formatDate, formatNumber, formatPercent } from "../../utils/format";
import type { DecisionView, RiskRuleView } from "./types";

const viewLabels: Array<{ id: DecisionView; label: string }> = [
  { id: "portfolio", label: "Portfolio" },
  { id: "model", label: "Model" },
  { id: "backtest", label: "Backtest" },
  { id: "market", label: "Market" },
  { id: "risk", label: "Risk" },
  { id: "training", label: "Training" },
];

interface PrimaryDecisionCanvasProps {
  view: DecisionView;
  setView: (view: DecisionView) => void;
  equity: EquityPoint[];
  backtest?: BacktestSummary | null;
  model?: ModelSummary | null;
  risk: RiskOverview;
  riskRules: RiskRuleView[];
  trades?: Page<Trade>;
  jobs: JobSummary[];
}

export function PrimaryDecisionCanvas(props: PrimaryDecisionCanvasProps): JSX.Element {
  return (
    <section className="vnext-primary-canvas">
      <header>
        <div><span>PRIMARY DECISION CANVAS</span><h2>{viewLabels.find((item) => item.id === props.view)?.label} Analysis</h2></div>
        <nav aria-label="主分析视图">
          {viewLabels.map((item) => <button type="button" key={item.id} className={props.view === item.id ? "active" : ""} aria-pressed={props.view === item.id} onClick={() => props.setView(item.id)}>{item.label}</button>)}
        </nav>
      </header>
      <div className="vnext-primary-body">
        {props.view === "portfolio" || props.view === "backtest" ? <PortfolioView {...props} /> : null}
        {props.view === "model" ? <ModelView model={props.model} /> : null}
        {props.view === "risk" ? <RiskView rules={props.riskRules} /> : null}
        {props.view === "training" ? <TrainingView jobs={props.jobs} /> : null}
        {props.view === "market" ? <StateView state="unavailable" title="市场实时状态未接入" detail="当前 API 没有可信市场时钟或实时行情摘要；请进入 Chart Workstation 查看持久化 K 线，不在 Dashboard 伪造行情。" /> : null}
      </div>
    </section>
  );
}

function PortfolioView(props: PrimaryDecisionCanvasProps): JSX.Element {
  const successfulTrades = (props.trades?.items ?? []).filter((trade) => trade.success !== false);
  const largestTrades = [...successfulTrades].sort((a, b) => Math.abs(b.amount ?? 0) - Math.abs(a.amount ?? 0)).slice(0, 4);
  return (
    <div className="vnext-portfolio-canvas">
      <div className="vnext-canvas-chart">
        <div className="vnext-chart-caption"><span>{props.backtest?.name ?? "Persisted portfolio"}</span><small>{props.backtest?.startDate ?? "—"} → {props.backtest?.endDate ?? "—"}</small><Link to="/backtests">Open Backtester <ArrowRight size={13} /></Link></div>
        {props.equity.length ? <EquityChart points={props.equity} height={390} /> : <StateView state="empty" detail="当前组合没有 NAV/benchmark/drawdown artifact。" />}
      </div>
      <aside className="vnext-canvas-context">
        <section><h3>Major trades</h3>{largestTrades.length ? largestTrades.map((trade) => <Link key={trade.id} to={`/stock-replay?symbol=${trade.symbol}&backtestId=${props.backtest?.id ?? ""}`}><span><strong>{trade.symbol}</strong><small>{trade.action} · {formatDate(trade.datetime)}</small></span><em>{formatNumber(trade.amount)}</em></Link>) : <p>暂无 persisted fills。</p>}</section>
        <section><h3>Active risk</h3><div className="vnext-inline-risk"><span>Drawdown <strong>{formatPercent(props.risk.maxDrawdown)}</strong></span><span>Liquidity <strong>{formatPercent(props.risk.liquidityRisk)}</strong></span><span>Limit-down <strong>{formatPercent(props.risk.limitDownRisk)}</strong></span></div><Link to="/risk">Inspect exact limits <ArrowRight size={13} /></Link></section>
      </aside>
    </div>
  );
}

function ModelView({ model }: { model?: ModelSummary | null }): JSX.Element {
  if (!model) return <StateView state="unavailable" title="没有可信模型" detail="Model Registry 没有可识别 artifact。" />;
  return (
    <div className="vnext-model-decision">
      <header><span><strong>{model.version ?? model.id}</strong><small>{model.modelFamily ?? model.modelType ?? "model"}</small></span><em>{model.productionReady ? "ACCEPTED" : "RESEARCH ONLY"}</em></header>
      <dl><div><dt>Train window</dt><dd>{model.trainStart ?? "—"} → {model.trainEnd ?? "—"}</dd></div><div><dt>Test end</dt><dd>{model.testEnd ?? "—"}</dd></div><div><dt>Horizons</dt><dd>{model.horizons.join(" / ") || "—"}</dd></div><div><dt>Samples / features</dt><dd>{model.sampleCount?.toLocaleString() ?? "—"} / {model.featureCount ?? "—"}</dd></div><div><dt>Device</dt><dd>{model.device ?? "UNAVAILABLE"}</dd></div><div><dt>Verdict</dt><dd>{model.verdict ?? "not declared"}</dd></div></dl>
      {model.issues.length ? <div className="vnext-model-issues">{model.issues.map((issue) => <p key={issue.code}><WarningCircle size={15} />{issue.message}</p>)}</div> : null}
      <Link to={`/training?modelId=${model.id}`}>Open Training Lab <ArrowRight size={14} /></Link>
    </div>
  );
}

function RiskView({ rules }: { rules: RiskRuleView[] }): JSX.Element {
  return (
    <div className="vnext-risk-table" role="table" aria-label="风险规则与阈值">
      <div role="row" className="vnext-risk-head"><span>Rule</span><span>Current</span><span>Hard threshold</span><span>State</span><span>Action</span></div>
      {rules.map((rule) => {
        const current = rule.current;
        const threshold = typeof rule.threshold === "number" ? rule.threshold : null;
        const ratio = current !== null && threshold ? Math.min(100, Math.abs(current / threshold) * 100) : 0;
        return (
          <div role="row" key={rule.id} className={`state-${rule.state}`}>
            <span><strong>{rule.name}</strong><small>{rule.description ?? rule.id}</small></span>
            <span className="mono">{current === null ? "UNAVAILABLE" : formatPercent(current)}</span>
            <span className="mono">{typeof rule.threshold === "number" ? formatPercent(rule.threshold) : rule.threshold ?? "POLICY"}</span>
            <span><i><b style={{ width: `${ratio}%` }} /></i><em>{rule.state}</em></span>
            <Link to="/risk">Inspect <ArrowRight size={12} /></Link>
          </div>
        );
      })}
    </div>
  );
}

function TrainingView({ jobs }: { jobs: JobSummary[] }): JSX.Element {
  const trainingJobs = jobs.filter((job) => job.type === "train").slice(0, 12);
  return (
    <div className="vnext-training-summary">
      {trainingJobs.length ? trainingJobs.map((job) => (
        <Link key={job.id} to={`/training?job=${job.id}`}>
          {job.status === "running" ? <CircleNotch size={17} className="spin" /> : <ChartLine size={17} />}
          <span><strong>{job.commandId}</strong><small>{job.id} · {formatDate(job.createdAt)}</small></span>
          <i><b style={{ width: `${Math.max(0, Math.min(100, (job.progress ?? 0) * 100))}%` }} /></i>
          <em>{job.progress === null || job.progress === undefined ? job.status : `${Math.round(job.progress * 100)}%`}</em>
        </Link>
      )) : <StateView state="empty" detail="没有 persisted training jobs；可在 Training Lab 验证配置后启动。" />}
    </div>
  );
}
