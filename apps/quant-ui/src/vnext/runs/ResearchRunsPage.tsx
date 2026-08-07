import { useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  ArrowClockwise,
  ArrowSquareOut,
  CheckCircle,
  Cpu,
  FileText,
  Gauge,
  Pause,
  Play,
  Prohibit,
  Rows,
  Scales,
  Stop,
  Trash,
  WarningCircle,
  XCircle,
} from "@phosphor-icons/react";
import type { EChartsOption } from "echarts";
import { apiDelete, apiPost } from "../../api/client";
import type {
  CouncilRunReview,
  JobSummary,
  RunComparison,
  RunConclusion,
  StrategyManifestSummary,
  StrategyRun,
  StrategyRunDetail,
} from "../../api/types";
import { EChart } from "../../components/EChart";
import { StateView } from "../../components/StateView";
import { StatusBadge } from "../../components/StatusBadge";
import { useApi } from "../../hooks/useApi";
import { useJobStream } from "../../hooks/useJobStream";
import { formatBytes, formatDate } from "../../utils/format";
import {
  TruthNotice,
  WorkbenchHeader,
  WorkbenchMetricStrip,
  WorkbenchPanel,
} from "../workbench/InstitutionalWorkbench";
import { useVNextChartPalette } from "../theme";

const ACTIVE = new Set(["queued", "starting", "running", "paused", "cancelling"]);

/** How each terminal outcome is presented. A rejected hypothesis is a result. */
const OUTCOME_PRESENTATION: Record<
  RunConclusion["outcome"],
  { label: string; tone: "ready" | "warning" | "error" | "partial"; description: string }
> = {
  accepted: {
    label: "通过全部闸门",
    tone: "ready",
    description: "流程完整、闸门全过。仍是 research/paper 结论，晋级需要人工复核。",
  },
  not_accepted: {
    label: "未通过验收",
    tone: "warning",
    description: "流程跑通、证据齐全，但至少一个预先声明的闸门未达标。",
  },
  rejected: {
    label: "研究闸门否决",
    tone: "warning",
    description: "运行完成并被过拟合/协议闸门否决。这是结论，不是故障。",
  },
  incomplete: {
    label: "未走完流程",
    tone: "error",
    description: "运行中断，结论不完整。先看诊断再决定重试还是改设计。",
  },
  no_evidence: {
    label: "尚无证据",
    tone: "partial",
    description: "运行仍在进行，或在写出任何产物前就结束了。",
  },
};

function statusTone(status: string): "ready" | "warning" | "error" | "partial" | "running" {
  if (status === "succeeded") return "ready";
  if (status === "rejected") return "warning";
  // Blocked means nothing ran: no evidence to read, only a config to fix.
  if (status === "blocked") return "warning";
  if (status === "failed") return "error";
  if (status === "cancelled") return "partial";
  return "running";
}

function percent(value?: number | null, digits = 1): string {
  return value === null || value === undefined ? "—" : `${(value * 100).toFixed(digits)}%`;
}

function decimal(value?: number | null, digits = 3): string {
  return value === null || value === undefined ? "—" : value.toFixed(digits);
}

function secondsSince(timestamp?: string | null): number | null {
  if (!timestamp) return null;
  const parsed = Date.parse(timestamp);
  return Number.isNaN(parsed) ? null : Math.max(0, (Date.now() - parsed) / 1000);
}

function duration(seconds?: number | null): string {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${seconds.toFixed(0)} 秒`;
  if (seconds < 3_600) return `${Math.floor(seconds / 60)} 分 ${Math.round(seconds % 60)} 秒`;
  return `${Math.floor(seconds / 3_600)} 小时 ${Math.round((seconds % 3_600) / 60)} 分`;
}

export function ResearchRunsPage(): JSX.Element {
  const queryClient = useQueryClient();
  const palette = useVNextChartPalette();
  const [selectedRunId, setSelectedRunId] = useState("");
  const [actionError, setActionError] = useState("");
  const [busy, setBusy] = useState("");
  const [confirmDelete, setConfirmDelete] = useState("");
  // Bounded to four by the API: a wider table invites picking a winner out of
  // noise rather than reading the evidence.
  const [comparisonIds, setComparisonIds] = useState<string[]>([]);

  const runs = useApi<StrategyRun[]>(["strategy-runs"], "/strategies/runs", undefined, {
    refetchInterval: 5_000,
  });
  const strategies = useApi<StrategyManifestSummary[]>(["strategy-index"], "/strategies");

  const runList = useMemo(() => runs.data?.data ?? [], [runs.data]);
  const activeRunId = selectedRunId || runList[0]?.runId || "";
  const detail = useApi<StrategyRunDetail>(
    ["strategy-run-detail", activeRunId],
    activeRunId ? `/strategies/runs/${activeRunId}` : null,
    undefined,
    { refetchInterval: 5_000 },
  );

  // The council only has something to adjudicate once a run has written
  // evidence, so it is fetched for terminal runs only.
  const council = useApi<CouncilRunReview>(
    ["council-run-review", activeRunId],
    activeRunId ? `/council/review/run/${activeRunId}` : null,
  );

  const comparison = useApi<RunComparison>(
    ["strategy-run-compare", comparisonIds.join(",")],
    comparisonIds.length > 1 ? "/strategies/runs/compare" : null,
    { runs: comparisonIds.join(",") },
  );

  const toggleComparison = (runId: string): void => {
    setComparisonIds((current) =>
      current.includes(runId)
        ? current.filter((item) => item !== runId)
        : current.length >= 4
          ? current
          : [...current, runId],
    );
  };

  const councilReview = Array.isArray(council.data?.data?.findings) ? council.data.data : null;
  const comparisonData = Array.isArray(comparison.data?.data?.runs) ? comparison.data.data : null;
  const run = detail.data?.data;
  const job = run?.job ?? null;
  const live = Boolean(job && ACTIVE.has(job.status));
  const stream = useJobStream(live ? job?.id ?? null : null);
  const liveJob = stream.job ?? job;

  // Once a run finishes its evidence changes shape; refresh the list so the
  // conclusion column stops showing a stale "running".
  useEffect(() => {
    if (liveJob?.terminal) void runs.refetch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [liveJob?.status]);

  const control = async (action: "cancel" | "pause" | "resume" | "retry"): Promise<void> => {
    if (!job) return;
    setBusy(action);
    setActionError("");
    try {
      await apiPost(`/jobs/${job.id}/${action}`, {});
      await Promise.all([runs.refetch(), detail.refetch()]);
      await queryClient.invalidateQueries({ queryKey: ["global-activity-jobs"] });
    } catch (error) {
      setActionError(error instanceof Error ? error.message : `${action} 失败`);
    } finally {
      setBusy("");
    }
  };

  const removeStrategy = async (strategyId: string, deleteOutputs: boolean): Promise<void> => {
    setBusy("delete");
    setActionError("");
    try {
      await apiDelete(
        `/strategies/${strategyId}${deleteOutputs ? "?deleteOutputs=true" : ""}`,
      );
      setConfirmDelete("");
      setSelectedRunId("");
      await Promise.all([runs.refetch(), strategies.refetch()]);
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "删除失败");
    } finally {
      setBusy("");
    }
  };

  const quietSeconds = secondsSince(liveJob?.lastOutputAt);
  const result = run?.result;
  const conclusion = result?.conclusion;
  const presentation = conclusion ? OUTCOME_PRESENTATION[conclusion.outcome] : null;

  const navOption = useMemo<EChartsOption>(() => {
    const points = result?.backtest?.nav ?? [];
    return {
      animation: false,
      grid: { left: 54, right: 18, top: 18, bottom: 30 },
      tooltip: {
        trigger: "axis",
        backgroundColor: palette.tooltip,
        borderColor: palette.tooltipBorder,
        textStyle: { color: palette.tooltipText },
      },
      xAxis: {
        type: "category",
        data: points.map((point) => point.date),
        axisLine: { lineStyle: { color: palette.axis } },
        axisLabel: { color: palette.text, fontSize: 10 },
      },
      yAxis: {
        type: "value",
        scale: true,
        axisLine: { lineStyle: { color: palette.axis } },
        splitLine: { lineStyle: { color: palette.grid } },
        axisLabel: { color: palette.text, fontSize: 10 },
      },
      series: [{
        type: "line",
        showSymbol: false,
        data: points.map((point) => point.nav),
        lineStyle: { color: palette.primary, width: 2 },
      }],
    };
  }, [palette, result?.backtest?.nav]);

  if (runs.isLoading && !runList.length) {
    return <StateView state="loading" detail="正在读取已登记的研究运行。" />;
  }

  if (!runList.length) {
    return (
      <div className="page institutional-workbench">
        <WorkbenchHeader
          eyebrow="RESEARCH OPERATIONS / RUN → CONCLUSION"
          title="研究运行"
          description="每一次策略启动都会在这里留下运行记录、实时状态、失败诊断和最终结论。"
          asOf="NO RUNS"
          context="research / paper only"
        />
        <StateView
          state="empty"
          title="尚无研究运行"
          detail="在策略实验室配置并启动一次闭环后，这里会出现它的阶段进度、资源占用、日志与结论。运行记录来自 Runtime，不由前端模拟。"
        />
      </div>
    );
  }

  return (
    <div className="page institutional-workbench research-runs-page">
      <WorkbenchHeader
        eyebrow="RESEARCH OPERATIONS / RUN → CONCLUSION"
        title="研究运行"
        description="策略启动之后的全部过程与结论：阶段进度、资源占用、异常诊断、闸门判定与产物。"
        asOf={liveJob ? `${liveJob.status.toUpperCase()} · ${Math.round((liveJob.progress ?? 0) * 100)}%` : "IDLE"}
        context={`${runList.length} 次运行 · ${strategies.data?.data.length ?? 0} 个策略`}
        actions={
          <StatusBadge
            status={liveJob ? statusTone(liveJob.status) : "partial"}
            label={liveJob?.status.toUpperCase() ?? "SELECT RUN"}
          />
        }
      />

      {liveJob ? (
        <WorkbenchMetricStrip
          metrics={[
            {
              label: "运行状态",
              value: liveJob.status.toUpperCase(),
              detail: liveJob.adopted ? "服务重启后已重新接管" : liveJob.message?.slice(0, 46) ?? "—",
              tone: liveJob.status === "failed" ? "danger" : liveJob.status === "succeeded" ? "positive" : "ai",
              icon: Gauge,
            },
            {
              label: "阶段",
              value: liveJob.stage ?? "—",
              detail: `${Math.round((liveJob.progress ?? 0) * 100)}% · ${liveJob.stages?.length ?? 0} 个阶段`,
              tone: "info",
              icon: Play,
            },
            {
              label: "耗时",
              value: duration(liveJob.elapsedSeconds),
              detail: liveJob.finishedAt ? `结束 ${formatDate(liveJob.finishedAt)}` : "进行中",
              tone: "neutral",
              icon: ArrowClockwise,
            },
            {
              label: "最近输出",
              value: liveJob.terminal
                ? "已结束"
                : quietSeconds === null ? "—" : `${duration(quietSeconds)}前`,
              // Long stages are legitimately quiet, so this is reported, never
              // judged: the operator decides whether the silence is expected.
              detail: liveJob.terminal
                ? "任务已进入终态"
                : quietSeconds !== null && quietSeconds > 600
                  ? "长时间没有输出，确认是否仍在推进"
                  : "进程仍在写出日志",
              tone: !liveJob.terminal && quietSeconds !== null && quietSeconds > 600 ? "warning" : "neutral",
              icon: Gauge,
            },
            {
              label: "资源占用",
              value: liveJob.resources?.cpuPercent == null
                ? "—"
                : `${liveJob.resources.cpuPercent.toFixed(0)}% · ${formatBytes(liveJob.resources.rssBytes ?? undefined)}`,
              detail: [
                `${liveJob.resources?.threads ?? "—"} 线程`,
                `${liveJob.resources?.childProcesses ?? 0} 子进程`,
                liveJob.resources?.gpuMemoryMiB
                  ? `GPU ${liveJob.resources.gpuMemoryMiB.toFixed(0)} MiB`
                  : "未占用显存",
              ].join(" · "),
              tone: "info",
              icon: Cpu,
            },
            {
              label: "结论",
              value: presentation?.label ?? "待定",
              detail: conclusion?.headline.slice(0, 40) ?? "运行结束后给出",
              tone: conclusion?.outcome === "accepted" ? "positive" : conclusion ? "warning" : "neutral",
              icon: Scales,
            },
          ]}
        />
      ) : null}

      <section className="research-runs-grid">
        <aside className="research-runs-list">
          <WorkbenchPanel eyebrow="RUN LEDGER" title="运行记录" meta="newest first · runtime backed">
            <div className="research-run-rows">
              {runList.map((item) => {
                const itemJob = item.job;
                const status = itemJob?.status ?? "unknown";
                return (
                  <div className="research-run-row" key={item.runId}>
                  <input
                    type="checkbox"
                    aria-label={`将 ${item.strategyName} 加入对比`}
                    checked={comparisonIds.includes(item.runId)}
                    disabled={!comparisonIds.includes(item.runId) && comparisonIds.length >= 4}
                    onChange={() => toggleComparison(item.runId)}
                  />
                  <button
                    type="button"
                    className={item.runId === activeRunId ? "active" : ""}
                    onClick={() => setSelectedRunId(item.runId)}
                  >
                    <span className={`research-run-state state-${status}`} />
                    <span className="research-run-label">
                      <strong>{item.strategyName}</strong>
                      <small>
                        {item.strategyId} · {formatDate(item.createdAt)}
                      </small>
                    </span>
                    <em>{itemJob?.progress == null ? "—" : `${Math.round(itemJob.progress * 100)}%`}</em>
                    <b className={`state-${status}`}>{status}</b>
                  </button>
                  </div>
                );
              })}
            </div>
          </WorkbenchPanel>

          <WorkbenchPanel eyebrow="STRATEGY REGISTRY" title="策略管理" meta="versions · runs · archive">
            <div className="research-strategy-rows">
              {(strategies.data?.data ?? []).map((item) => (
                <article key={item.id}>
                  <div>
                    <strong>{item.name}</strong>
                    <small>
                      {item.id} · {item.versionCount ?? 1} 个版本 · {item.runCount ?? 0} 次运行
                    </small>
                  </div>
                  {confirmDelete === item.id ? (
                    <div className="research-delete-confirm">
                      <span>归档该策略的全部版本？</span>
                      <button type="button" onClick={() => void removeStrategy(item.id, false)} disabled={busy === "delete"}>
                        仅归档记录
                      </button>
                      <button
                        type="button"
                        className="danger"
                        onClick={() => void removeStrategy(item.id, true)}
                        disabled={busy === "delete"}
                      >
                        归档并删除运行产物
                      </button>
                      <button type="button" onClick={() => setConfirmDelete("")}>取消</button>
                    </div>
                  ) : (
                    <button type="button" className="research-delete" onClick={() => setConfirmDelete(item.id)}>
                      <Trash size={14} />删除
                    </button>
                  )}
                </article>
              ))}
            </div>
            <TruthNotice>
              删除会把策略清单移入 <code>runtime/archives</code>，研究记录不会被静默销毁；运行产物只有在明确要求时才一并删除。
            </TruthNotice>
          </WorkbenchPanel>
        </aside>

        <div className="research-run-detail">
          {comparisonIds.length > 1 ? (
            <WorkbenchPanel
              eyebrow="RUN COMPARISON"
              title={`对比 ${comparisonIds.length} 次运行`}
              meta="最多 4 项 · 数值取自各自产物"
              className="research-compare-panel"
              actions={
                <button type="button" className="research-clear-compare" onClick={() => setComparisonIds([])}>
                  <Rows size={14} />清空对比
                </button>
              }
            >
              {comparisonData?.runs.length ? (
                <>
                  <div className="research-compare-scroll">
                    <table className="research-compare-table">
                      <thead>
                        <tr>
                          <th>指标</th>
                          {comparisonData.runs.map((item) => (
                            <th key={item.runId}>
                              <strong>{item.strategyName}</strong>
                              <small>{OUTCOME_PRESENTATION[item.outcome ?? "no_evidence"].label}</small>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {comparisonData.metrics.map((metric) => (
                          <tr key={metric.key}>
                            <th scope="row">
                              <span>{metric.label}</span>
                              <small>{metric.group}{metric.direction ? ` · ${metric.direction === "higher" ? "越高越好" : "越低越好"}` : ""}</small>
                            </th>
                            {metric.values.map((value, index) => (
                              <td
                                key={`${metric.key}-${index}`}
                                className={metric.bestIndex === index ? "best" : ""}
                              >
                                {value === null || value === undefined
                                  ? <em title="该运行没有产出该字段">未产出</em>
                                  : typeof value === "number"
                                    ? value.toFixed(Math.abs(value) < 1 ? 4 : 2)
                                    : String(value)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <details className="research-compare-gates">
                    <summary>逐条闸门对照（{comparisonData.gates.length} 项）</summary>
                    <div className="research-compare-scroll">
                      <table className="research-compare-table">
                        <tbody>
                          {comparisonData.gates.map((gate) => (
                            <tr key={gate.name}>
                              <th scope="row"><span>{gate.name}</span></th>
                              {gate.values.map((value, index) => (
                                <td key={`${gate.name}-${index}`} className={value?.passed === false ? "failed" : ""}>
                                  {value === null
                                    ? <em>未评估</em>
                                    : `${value.passed ? "通过" : "未通过"} · ${
                                        typeof value.actual === "number" ? value.actual.toFixed(4) : String(value.actual)
                                      }`}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </details>
                  {comparisonData.note ? <TruthNotice>{comparisonData.note}</TruthNotice> : null}
                </>
              ) : (
                <StateView state="loading" detail="正在对齐所选运行的产物。" />
              )}
            </WorkbenchPanel>
          ) : null}
          {!run ? (
            <StateView state="loading" detail="正在读取该运行的产物。" />
          ) : (
            <>
              <WorkbenchPanel
                eyebrow="RUN CONCLUSION"
                title={conclusion?.headline ?? "运行进行中"}
                meta={`${run.strategyName} · ${run.strategyVersion}`}
                className="research-conclusion-panel"
              >
                {presentation ? (
                  <div className={`research-conclusion tone-${presentation.tone}`}>
                    <div className="research-conclusion-head">
                      {conclusion?.outcome === "accepted" ? (
                        <CheckCircle weight="fill" size={22} />
                      ) : conclusion?.outcome === "rejected" || conclusion?.outcome === "not_accepted" ? (
                        <Scales weight="fill" size={22} />
                      ) : (
                        <WarningCircle weight="fill" size={22} />
                      )}
                      <span>
                        <strong>{presentation.label}</strong>
                        <small>{presentation.description}</small>
                      </span>
                    </div>
                    {conclusion?.reasons?.length ? (
                      <ul className="research-reason-list">
                        {(conclusion.reasons ?? []).map((reason) => (
                          <li key={reason}>{reason}</li>
                        ))}
                      </ul>
                    ) : null}
                    {conclusion?.remediation ? (
                      <p className="research-remediation">{conclusion.remediation}</p>
                    ) : null}
                  </div>
                ) : (
                  <TruthNotice>运行结束后，这里会给出通过/未通过的判定以及每一条判定依据。</TruthNotice>
                )}

                <div className="research-run-controls">
                  {liveJob?.status === "running" ? (
                    <button type="button" onClick={() => void control("pause")} disabled={Boolean(busy)}>
                      <Pause size={14} />暂停
                    </button>
                  ) : null}
                  {liveJob?.status === "paused" ? (
                    <button type="button" onClick={() => void control("resume")} disabled={Boolean(busy)}>
                      <Play size={14} />继续
                    </button>
                  ) : null}
                  {liveJob && ACTIVE.has(liveJob.status) ? (
                    <button type="button" className="danger" onClick={() => void control("cancel")} disabled={Boolean(busy)}>
                      <Stop size={14} />停止
                    </button>
                  ) : null}
                  {liveJob?.canRetry ? (
                    <button type="button" className="primary" onClick={() => void control("retry")} disabled={Boolean(busy)}>
                      <ArrowClockwise size={14} />以相同参数重试
                    </button>
                  ) : null}
                  <a className="research-output-link" href={`/runtime?query=${encodeURIComponent(run.outputDir)}`}>
                    <ArrowSquareOut size={14} />在 Runtime 中打开产物
                  </a>
                </div>
                {actionError ? (
                  <p className="research-action-error" role="alert">
                    <XCircle size={14} />{actionError}
                  </p>
                ) : null}
                {liveJob?.adopted ? (
                  <TruthNotice tone="warning">
                    该任务在 API 重启后被重新接管。进程一直在运行，阶段进度来自它自己写出的日志。
                  </TruthNotice>
                ) : null}
              </WorkbenchPanel>

              {liveJob?.failure ? (
                <WorkbenchPanel
                  eyebrow="FAILURE DIAGNOSIS"
                  title={liveJob.failure.title}
                  meta={`${liveJob.failure.code} · exit ${liveJob.failure.exitCode ?? "unknown"}`}
                  className="research-failure-panel"
                >
                  <p className="research-failure-detail">{liveJob.failure.detail}</p>
                  <div className="research-failure-remediation">
                    <strong>下一步</strong>
                    <span>{liveJob.failure.remediation}</span>
                  </div>
                  {liveJob.failure.logTail?.length ? (
                    <details className="research-log-tail">
                      <summary>查看日志尾部 {liveJob.failure.logTail?.length ?? 0} 行</summary>
                      <pre>{(liveJob.failure.logTail ?? []).join("\n")}</pre>
                    </details>
                  ) : null}
                  {liveJob.exitStatusObserved === false ? (
                    <TruthNotice tone="warning">
                      退出码未被观测到，因此无法断定这次运行是完成还是中断；上面的判断只基于它写出的日志。
                    </TruthNotice>
                  ) : null}
                </WorkbenchPanel>
              ) : null}

              {liveJob?.verdict ? (
                <WorkbenchPanel
                  eyebrow="RESEARCH VERDICT"
                  title={liveJob.verdict.title}
                  meta={`${liveJob.verdict.code} · ${liveJob.verdict.stage ?? "gate"}`}
                  className="research-verdict-panel"
                >
                  <ul className="research-reason-list">
                    {(liveJob.verdict.reasons ?? []).map((reason) => (
                      <li key={reason}>{reason}</li>
                    ))}
                  </ul>
                  {liveJob.verdict.remediation ? <p className="research-remediation">{liveJob.verdict.remediation}</p> : null}
                  <TruthNotice>
                    这是预先声明的闸门给出的结论，不能通过事后放宽阈值来"通过"。运行产物已完整保留，可用于复盘。
                  </TruthNotice>
                </WorkbenchPanel>
              ) : null}

              <WorkbenchPanel
                eyebrow="STAGE TIMELINE"
                title="阶段进度与证据"
                meta={liveJob?.lastOutputAt ? `最后输出 ${formatDate(liveJob.lastOutputAt)}` : "等待输出"}
                className="research-stage-panel"
              >
                <div className="research-stage-list">
                  {(result?.stages ?? []).map((stage) => {
                    const runtimeStage = liveJob?.stages?.find((item) => item.id === stage.id);
                    const state = stage.present
                      ? "complete"
                      : runtimeStage?.status === "running"
                        ? "active"
                        : runtimeStage
                          ? "stopped"
                          : "pending";
                    return (
                      <article key={stage.id} className={state}>
                        <span>{stage.label}</span>
                        <small>
                          {stage.present
                            ? `产物已写入 · ${formatBytes(stage.sizeBytes ?? undefined)}`
                            : state === "active"
                              ? runtimeStage?.message || "进行中"
                              : state === "stopped"
                                ? "在此阶段停止"
                                : "尚未产出"}
                        </small>
                        <b>{stage.present ? <CheckCircle weight="fill" /> : state === "active" ? <ArrowClockwise /> : <Prohibit />}</b>
                      </article>
                    );
                  })}
                </div>
              </WorkbenchPanel>

              {result?.acceptance ? (
                <WorkbenchPanel
                  eyebrow="ACCEPTANCE GATES"
                  title={`验收闸门 ${result.acceptance.passedCount}/${result.acceptance.totalCount}`}
                  meta={result.acceptance.sourcePath}
                  className="research-gates-panel"
                >
                  <table className="research-gate-table">
                    <thead>
                      <tr><th>闸门</th><th>实测</th><th>阈值</th><th>判定</th></tr>
                    </thead>
                    <tbody>
                      {(result.acceptance.gates ?? []).map((gate) => (
                        <tr key={gate.name} className={gate.passed ? "" : "failed"}>
                          <td>{gate.name}</td>
                          <td className="mono">{typeof gate.actual === "number" ? gate.actual.toFixed(4) : String(gate.actual)}</td>
                          <td className="mono">{String(gate.threshold)}</td>
                          <td>{gate.passed ? <span className="pass">通过</span> : <span className="fail">{gate.reason ?? "未通过"}</span>}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </WorkbenchPanel>
              ) : null}

              {councilReview?.findings.length ? (
                <WorkbenchPanel
                  eyebrow="DECISION COUNCIL"
                  title="多 Agent 审查"
                  meta={`${councilReview.decision.state} · ${councilReview.decision.summary}`}
                  className="research-council-panel"
                >
                  <div className="research-council-list">
                    {councilReview.findings.map((finding) => {
                      const role = councilReview.roles?.find((item) => item.id === finding.roleId);
                      return (
                        <article key={finding.roleId} className={`verdict-${finding.verdict}`}>
                          <header>
                            <strong>{role?.label ?? finding.roleId}</strong>
                            <span className={`research-verdict-badge verdict-${finding.verdict}`}>
                              {finding.verdict === "pass" ? "通过"
                                : finding.verdict === "blocked" ? "否决"
                                  : finding.verdict === "warn" ? "警告" : "证据缺失"}
                            </span>
                          </header>
                          <p className="research-council-headline">{finding.headline}</p>
                          <p className="research-council-detail">{finding.detail}</p>
                          <dl className="research-council-evidence">
                            {Object.entries(finding.evidence ?? {}).slice(0, 5).map(([key, value]) => (
                              <div key={key}>
                                <dt>{key}</dt>
                                <dd>{value === null || value === undefined ? "—" : String(value)}</dd>
                              </div>
                            ))}
                          </dl>
                          {finding.nextAction && finding.nextAction !== "无" ? (
                            <p className="research-council-next">下一步：{finding.nextAction}</p>
                          ) : null}
                        </article>
                      );
                    })}
                  </div>
                  <TruthNotice>
                    每个角色只在自己的职责域内否决，裁决必须附带它实际读取的字段。
                    证据缺失记为「证据缺失」，既不阻塞研究，也不算通过。
                  </TruthNotice>
                </WorkbenchPanel>
              ) : null}

              {result?.candidates?.length ? (
                <WorkbenchPanel
                  eyebrow="CANDIDATE SEARCH"
                  title={`组合候选 ${result.candidates.length} 组`}
                  meta="同一成本模型 · 冠军由早期 OOS 选出，holdout 只验收一次"
                  className="research-candidates-panel"
                >
                  <div className="research-compare-scroll">
                    <table className="research-gate-table research-candidate-table">
                      <thead>
                        <tr>
                          <th>候选</th><th>年化</th><th>成本后净收益</th><th>最大回撤</th>
                          <th>Sharpe</th><th>成交</th><th>验收</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(result.candidates ?? []).map((candidate) => (
                          <tr key={candidate.id} className={candidate.selected ? "selected" : ""}>
                            <td>
                              {candidate.id}
                              {candidate.selected ? <em className="research-champion">冠军</em> : null}
                            </td>
                            <td className="mono">{percent(candidate.annualisedReturn, 2)}</td>
                            <td className="mono">{percent(candidate.netReturnAfterCosts, 2)}</td>
                            <td className="mono">{percent(candidate.maxDrawdown, 2)}</td>
                            <td className="mono">{decimal(candidate.sharpe, 2)}</td>
                            <td className="mono">{candidate.tradeCount ?? "—"}</td>
                            <td>{candidate.acceptanceStatus ?? "—"}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <TruthNotice>
                    冠军是在早期 OOS 段上按研究偏好选出的，因此它在冻结 holdout 上不一定是数字最高的一个——
                    这正是这张表要暴露的事实。每个候选都计入过拟合治理的试验次数。
                  </TruthNotice>
                </WorkbenchPanel>
              ) : null}

              {result?.governance ? (
                <WorkbenchPanel
                  eyebrow="OVERFITTING GOVERNANCE"
                  title="过拟合治理"
                  meta={result.governance.sourcePath}
                  className="research-governance-panel"
                >
                  <div className="research-metric-grid">
                    <span><small>PBO</small><strong>{decimal(result.governance.pbo)}</strong></span>
                    <span><small>DSR 概率</small><strong>{decimal(result.governance.dsrProbability)}</strong></span>
                    <span><small>SPA p-value</small><strong>{decimal(result.governance.spaPValue)}</strong></span>
                    <span><small>观测天数</small><strong>{result.governance.observedDays ?? "—"}</strong></span>
                    <span><small>累计试验</small><strong>{result.governance.cumulativeTrials ?? "—"}</strong></span>
                    <span><small>入选候选</small><strong>{result.governance.selectedCandidate ?? "—"}</strong></span>
                  </div>
                  {result.governance.rejectionReasons?.length ? (
                    <ul className="research-reason-list">
                      {(result.governance.rejectionReasons ?? []).map((reason) => <li key={reason}>{reason}</li>)}
                    </ul>
                  ) : null}
                </WorkbenchPanel>
              ) : null}

              {result?.training ? (
                <WorkbenchPanel
                  eyebrow="TRAINING EVIDENCE"
                  title="训练与样本外证据"
                  meta={result.training.sourcePath}
                  className="research-training-panel"
                >
                  <div className="research-metric-grid">
                    <span><small>Rank IC</small><strong>{decimal(result.training.rankIcMean, 4)}</strong></span>
                    <span><small>ICIR</small><strong>{decimal(result.training.icir)}</strong></span>
                    <span><small>命中率</small><strong>{percent(result.training.hitRate)}</strong></span>
                    <span><small>折数</small><strong>{result.training.foldCount ?? "—"}</strong></span>
                    <span><small>特征数</small><strong>{result.training.featureCount ?? "—"}</strong></span>
                    <span><small>评估交易日</small><strong>{result.training.evaluatedDays ?? "—"}</strong></span>
                  </div>
                  {result.training.annualisationWarning ? (
                    <TruthNotice tone="warning">{result.training.annualisationWarning}</TruthNotice>
                  ) : null}
                </WorkbenchPanel>
              ) : null}

              {result?.backtest && result.backtest.navPoints > 0 ? (
                <WorkbenchPanel
                  eyebrow="A-SHARE BACKTEST"
                  title="成本后净值"
                  meta={`${result.backtest.navPoints} 个交易日 · ${result.backtest.sourcePath}`}
                  className="research-backtest-panel"
                >
                  <div className="research-metric-grid">
                    <span><small>区间收益</small><strong>{percent(result.backtest.totalReturn, 2)}</strong></span>
                    <span><small>最大回撤</small><strong>{percent(result.backtest.maxDrawdown, 2)}</strong></span>
                    <span><small>成交笔数</small><strong>{result.backtest.orderCount ?? "—"}</strong></span>
                    <span><small>被约束跳过</small><strong>{result.backtest.skippedOrderCount ?? "—"}</strong></span>
                  </div>
                  <EChart option={navOption} className="research-nav-chart" />
                  <TruthNotice>
                    净值来自 A 股撮合模拟（T+1、涨跌停、停牌、整手、成交量上限与成本）。被跳过的委托数量说明了约束的强度。
                  </TruthNotice>
                </WorkbenchPanel>
              ) : null}

              {live ? (
                <WorkbenchPanel
                  eyebrow="LIVE OUTPUT"
                  title="实时输出"
                  meta={stream.connected ? "SSE connected" : "stream reconnectable"}
                  className="research-console-panel"
                >
                  <div className="research-console" aria-live="polite">
                    {stream.lines.length
                      ? stream.lines.slice(-200).map((line, index) => <code key={`${index}-${line.slice(0, 12)}`}>{line}</code>)
                      : <span>等待进程输出。</span>}
                  </div>
                </WorkbenchPanel>
              ) : null}

              {result?.artifacts?.length ? (
                <WorkbenchPanel
                  eyebrow="ARTIFACTS"
                  title={`产物 ${result.artifacts.length} 个`}
                  meta={result.outputDir}
                  className="research-artifacts-panel"
                >
                  <div className="research-artifact-list">
                    {result.artifacts.slice(0, 60).map((artifact) => (
                      <a key={artifact.path} href={`/runtime?query=${encodeURIComponent(artifact.path)}`}>
                        <FileText size={13} />
                        <span>{artifact.relative}</span>
                        <em>{formatBytes(artifact.sizeBytes)}</em>
                      </a>
                    ))}
                  </div>
                </WorkbenchPanel>
              ) : null}
            </>
          )}
        </div>
      </section>
    </div>
  );
}
