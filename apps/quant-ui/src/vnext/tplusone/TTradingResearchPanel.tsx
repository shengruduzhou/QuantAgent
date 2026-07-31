import { useState } from "react";
import { Play, Stop, Warning } from "@phosphor-icons/react";
import { apiPost } from "../../api/client";
import type { JobSummary } from "../../api/types";
import { useJobStream } from "../../hooks/useJobStream";
import { TruthNotice, WorkbenchPanel } from "../workbench/InstitutionalWorkbench";

/**
 * Governed launcher for the T+1-compatible intraday research loop.
 *
 * A-share has no true T+0. Every round trip this study evaluates must be funded
 * by inventory that was already held at yesterday's close, so the holdings file
 * is a required input rather than an option — the form cannot express a study
 * that quietly assumes same-day sellability.
 */
interface TTradingDraft {
  minuteDir: string;
  holdingsCsv: string;
  marketPanel: string;
  outputDir: string;
  start: string;
  end: string;
  trainEnd: string;
  validationEnd: string;
  horizonMinutes: number;
  orderNotionalYuan: number;
  backend: "lightgbm" | "xgboost" | "catboost" | "sklearn";
  makerOnly: boolean;
  slippageBps: number;
  spreadBps: number;
  commissionRate: number;
  edgeCostMultiple: number;
  minRoundTripsEnable: number;
  maxSymbols: number;
}

const DEFAULT_DRAFT: TTradingDraft = {
  minuteDir: "runtime/data/v7/silver/minute_bars",
  holdingsCsv: "runtime/paper/replay_2026/holdings_daily.csv",
  marketPanel: "runtime/data/v7/silver/market_panel/market_panel.parquet",
  outputDir: "runtime/reports/t_plus_one/ev_study_01",
  start: "2025-09-01",
  end: "2026-06-12",
  trainEnd: "2026-02-27",
  validationEnd: "2026-04-15",
  horizonMinutes: 60,
  orderNotionalYuan: 100_000,
  backend: "lightgbm",
  makerOnly: false,
  slippageBps: 8,
  spreadBps: 6,
  commissionRate: 0.0003,
  edgeCostMultiple: 2,
  minRoundTripsEnable: 300,
  maxSymbols: 0,
};

export function TTradingResearchPanel(): JSX.Element {
  const [draft, setDraft] = useState<TTradingDraft>(DEFAULT_DRAFT);
  const [activeJob, setActiveJob] = useState<JobSummary | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const stream = useJobStream(activeJob?.id ?? null);
  const job = stream.job ?? activeJob;
  const running = Boolean(job && ["queued", "starting", "running", "paused", "cancelling"].includes(job.status));

  const update = <K extends keyof TTradingDraft>(key: K, value: TTradingDraft[K]): void => {
    setDraft((current) => ({ ...current, [key]: value }));
  };

  const launch = async (): Promise<void> => {
    setBusy(true);
    setError("");
    try {
      const result = await apiPost<JobSummary>("/jobs/t-plus-one-research", {
        commandId: "research-intraday-t-trading",
        parameters: {
          minute_dir: draft.minuteDir,
          holdings_csv: draft.holdingsCsv,
          market_panel: draft.marketPanel,
          output_dir: draft.outputDir,
          start: draft.start,
          end: draft.end,
          train_end: draft.trainEnd,
          validation_end: draft.validationEnd,
          horizon_minutes: draft.horizonMinutes,
          order_notional_yuan: draft.orderNotionalYuan,
          backend: draft.backend,
          maker_only: draft.makerOnly,
          slippage_bps: draft.slippageBps,
          spread_bps: draft.spreadBps,
          commission_rate: draft.commissionRate,
          edge_cost_multiple: draft.edgeCostMultiple,
          min_round_trips_enable: draft.minRoundTripsEnable,
          max_symbols: draft.maxSymbols,
        },
      });
      setActiveJob(result.data);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "T+1 研究任务提交失败");
    } finally {
      setBusy(false);
    }
  };

  const cancel = async (): Promise<void> => {
    if (!job) return;
    try {
      const result = await apiPost<JobSummary>(`/jobs/${job.id}/cancel`, {});
      setActiveJob(result.data);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "取消失败");
    }
  };

  return (
    <WorkbenchPanel
      eyebrow="T+1 RESEARCH LOOP"
      title="做 T 研究与验证"
      meta="governed command · research-intraday-t-trading"
      className="tplus-research-panel"
    >
      <TruthNotice tone="warning">
        A 股没有真正的 T+0。本研究把每一次日内回转都限制在<strong>昨收可卖库存</strong>之内，
        因此持仓文件是必填输入。收益必须相对“持有底仓不动”的基线计算，否则会把底仓的
        beta 记成做 T 的 alpha。引擎默认输出 NO_TRADE：只有成本后 EV 显著为正才产生动作。
      </TruthNotice>

      <div className="tplus-form-grid">
        <label className="atlas-field wide">
          <span>可卖库存（昨收持仓）</span>
          <input className="mono" value={draft.holdingsCsv} onChange={(event) => update("holdingsCsv", event.target.value)} />
          <small>必填。没有它就无法界定当日可卖数量。</small>
        </label>
        <label className="atlas-field wide">
          <span>分钟行情目录</span>
          <input className="mono" value={draft.minuteDir} onChange={(event) => update("minuteDir", event.target.value)} />
        </label>
        <label className="atlas-field wide">
          <span>日线面板（涨跌停 / 停牌 / ST）</span>
          <input className="mono" value={draft.marketPanel} onChange={(event) => update("marketPanel", event.target.value)} />
        </label>
        <label className="atlas-field wide">
          <span>输出目录</span>
          <input className="mono" value={draft.outputDir} onChange={(event) => update("outputDir", event.target.value)} />
        </label>

        <label className="atlas-field"><span>起始日</span>
          <input type="date" value={draft.start} onChange={(event) => update("start", event.target.value)} /></label>
        <label className="atlas-field"><span>结束日</span>
          <input type="date" value={draft.end} onChange={(event) => update("end", event.target.value)} /></label>
        <label className="atlas-field"><span>训练截止</span>
          <input type="date" value={draft.trainEnd} onChange={(event) => update("trainEnd", event.target.value)} /></label>
        <label className="atlas-field"><span>验证截止</span>
          <input type="date" value={draft.validationEnd} onChange={(event) => update("validationEnd", event.target.value)} /></label>
        <label className="atlas-field"><span>持有分钟数</span>
          <input type="number" min={1} max={240} value={draft.horizonMinutes} onChange={(event) => update("horizonMinutes", Number(event.target.value))} /></label>
        <label className="atlas-field"><span>单笔名义金额</span>
          <input type="number" min={1000} step={10000} value={draft.orderNotionalYuan} onChange={(event) => update("orderNotionalYuan", Number(event.target.value))} /></label>
        <label className="atlas-field"><span>模型后端</span>
          <select value={draft.backend} onChange={(event) => update("backend", event.target.value as TTradingDraft["backend"])}>
            <option value="lightgbm">LightGBM</option>
            <option value="xgboost">XGBoost</option>
            <option value="catboost">CatBoost</option>
            <option value="sklearn">scikit-learn</option>
          </select></label>
        <label className="atlas-field"><span>最少回合数（启用门槛）</span>
          <input type="number" min={1} value={draft.minRoundTripsEnable} onChange={(event) => update("minRoundTripsEnable", Number(event.target.value))} /></label>
      </div>

      <div className="tplus-cost-block">
        <span className="atlas-eyebrow">COST SURFACE · 单点成本不构成证据</span>
        <div className="tplus-form-grid">
          <label className="atlas-field foundry-check">
            <span>Maker 通道</span>
            <input type="checkbox" checked={draft.makerOnly} onChange={(event) => update("makerOnly", event.target.checked)} />
            <small>限价撮合（约 10bps 往返），否则按散户 taker 计</small>
          </label>
          <label className="atlas-field"><span>滑点 (bps)</span>
            <input type="number" min={0} step={0.5} value={draft.slippageBps} onChange={(event) => update("slippageBps", Number(event.target.value))} /></label>
          <label className="atlas-field"><span>价差 (bps)</span>
            <input type="number" min={0} step={0.5} value={draft.spreadBps} onChange={(event) => update("spreadBps", Number(event.target.value))} /></label>
          <label className="atlas-field"><span>佣金率</span>
            <input type="number" min={0} step={0.0001} value={draft.commissionRate} onChange={(event) => update("commissionRate", Number(event.target.value))} /></label>
          <label className="atlas-field"><span>边际/成本倍数</span>
            <input type="number" min={0} step={0.5} value={draft.edgeCostMultiple} onChange={(event) => update("edgeCostMultiple", Number(event.target.value))} /></label>
        </div>
      </div>

      <div className="tplus-actions">
        <button type="button" className="atlas-action" data-variant="primary" onClick={launch} disabled={busy || running}>
          <Play size={13} weight="fill" />{busy ? "提交中" : "启动 T+1 研究"}
        </button>
        {running ? (
          <button type="button" className="atlas-action" data-variant="danger" onClick={cancel}>
            <Stop size={13} weight="fill" />取消
          </button>
        ) : null}
      </div>

      {error ? (
        <div className="foundry-error" role="alert"><Warning size={14} /><span>{error}</span></div>
      ) : null}

      {job ? (
        <div className="foundry-job">
          <div className="atlas-row">
            <span className="atlas-eyebrow">JOB</span>
            <code>{job.id}</code>
            <span className="atlas-chip" data-tone={job.status === "failed" ? "danger" : running ? "live" : "success"}>{job.status}</span>
          </div>
          <i className="atlas-meter" data-tone="agent">
            <i style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }} />
          </i>
          <div className="foundry-console" aria-live="polite">
            {stream.lines.length
              ? stream.lines.slice(-30).map((line, index) => <code key={`${index}-${line.slice(0, 12)}`}>{line}</code>)
              : <span>等待研究循环输出…</span>}
          </div>
        </div>
      ) : null}
    </WorkbenchPanel>
  );
}
