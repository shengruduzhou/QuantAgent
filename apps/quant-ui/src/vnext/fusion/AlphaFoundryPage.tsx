import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowsClockwise,
  Atom,
  ChartScatter,
  CheckCircle,
  Play,
  Scales,
  ShieldWarning,
  Stop,
  Target,
  Warning,
} from "@phosphor-icons/react";
import type { EChartsOption } from "echarts";
import { apiPost } from "../../api/client";
import type {
  FusionCandidate,
  FusionNavRow,
  FusionRunDetail,
  FusionRunSummary,
  JobSummary,
  StrategyDefaults,
} from "../../api/types";
import { EChart } from "../../components/EChart";
import { useApi } from "../../hooks/useApi";
import { useJobStream } from "../../hooks/useJobStream";
import { useVNextChartPalette } from "../theme";
import {
  ActionableState,
  TruthNotice,
  WorkbenchHeader,
  WorkbenchMetricStrip,
  WorkbenchPanel,
} from "../workbench/InstitutionalWorkbench";
import { FusionSearchForm, type FusionSearchDraft, DEFAULT_SEARCH_DRAFT } from "./FusionSearchForm";
import { CandidateInspector } from "./CandidateInspector";

const MAX_COMPARE = 4;

function percent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

function ratio(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

/** CSS direction class for a return cell: A-share red up, green down. */
function objectiveTone(candidate: FusionCandidate): "positive" | "negative" | "neutral" {
  if (candidate.metrics.excessReturn > 0) return "positive";
  if (candidate.metrics.excessReturn < 0) return "negative";
  return "neutral";
}

/** Metric-strip tone, which uses status semantics rather than market direction. */
function excessStripTone(candidate: FusionCandidate | undefined): "positive" | "danger" | "neutral" {
  if (!candidate) return "neutral";
  if (candidate.metrics.excessReturn > 0) return "positive";
  if (candidate.metrics.excessReturn < 0) return "danger";
  return "neutral";
}

export function AlphaFoundryPage(): JSX.Element {
  const palette = useVNextChartPalette();
  const [draft, setDraft] = useState<FusionSearchDraft>(DEFAULT_SEARCH_DRAFT);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [comparisonIds, setComparisonIds] = useState<string[]>([]);
  const [inspectedId, setInspectedId] = useState<string>("");
  const [activeJob, setActiveJob] = useState<JobSummary | null>(null);
  const [busy, setBusy] = useState<"launch" | "cancel" | "">("");
  const [error, setError] = useState("");
  const [showControls, setShowControls] = useState(true);

  const runs = useApi<FusionRunSummary[]>(["fusion-runs"], "/fusion/runs", undefined, {
    refetchInterval: 30_000,
  });
  const defaults = useApi<StrategyDefaults>(["strategy-runtime-defaults"], "/strategies/defaults");
  const runList = useMemo(
    () => (Array.isArray(runs.data?.data) ? runs.data.data : []),
    [runs.data],
  );
  const effectiveRunId = selectedRunId || runList[0]?.id || "";
  const detail = useApi<FusionRunDetail>(
    ["fusion-run", effectiveRunId],
    effectiveRunId ? `/fusion/runs/${effectiveRunId}` : null,
  );
  const navs = useApi<FusionNavRow[]>(
    ["fusion-nav", effectiveRunId],
    effectiveRunId ? `/fusion/runs/${effectiveRunId}/nav` : null,
  );

  const stream = useJobStream(activeJob?.id ?? null);
  const job = stream.job ?? activeJob;
  const running = Boolean(job && ["queued", "running", "cancelling"].includes(job.status));

  const candidates = useMemo(
    () => detail.data?.data.candidates ?? [],
    [detail.data],
  );
  const summary = detail.data?.data.summary ?? {};
  const visibleCandidates = useMemo(
    () => (showControls ? candidates : candidates.filter((item) => !item.isControl)),
    [candidates, showControls],
  );
  const frontier = useMemo(
    () => candidates.filter((item) => item.onFrontier),
    [candidates],
  );
  const inspected = useMemo(
    () => candidates.find((item) => item.id === inspectedId) ?? frontier[0] ?? candidates[0],
    [candidates, frontier, inspectedId],
  );

  // A run change invalidates any selection made against the previous run.
  useEffect(() => {
    setComparisonIds([]);
    setInspectedId("");
  }, [effectiveRunId]);

  // Newly finished searches should appear without a manual refresh.
  useEffect(() => {
    if (job?.status === "succeeded") {
      void runs.refetch();
    }
  }, [job?.status, runs]);

  const toggleComparison = useCallback((candidateId: string): void => {
    setComparisonIds((current) => {
      if (current.includes(candidateId)) {
        return current.filter((item) => item !== candidateId);
      }
      if (current.length >= MAX_COMPARE) return current;
      return [...current, candidateId];
    });
  }, []);

  const launch = async (): Promise<void> => {
    setBusy("launch");
    setError("");
    try {
      const result = await apiPost<JobSummary>("/jobs/fusion-search", {
        commandId: "search-factor-fusion",
        parameters: {
          factor_panel_path: draft.factorPanelPath,
          forward_returns_path: draft.forwardReturnsPath,
          factor_names: draft.factorNames,
          output_dir: draft.outputDir,
          forward_column: draft.forwardColumn,
          horizon_days: draft.horizonDays,
          top_k: draft.topK,
          n_folds: draft.nFolds,
          embargo_days: draft.embargoDays,
          min_train_days: draft.minTrainDays,
          min_test_days: draft.minTestDays,
          transaction_cost_bps: draft.transactionCostBps,
          include_genetic: draft.includeGenetic,
          random_controls: draft.randomControls,
          single_factor_baselines: draft.singleFactorBaselines,
          seed: draft.seed,
          benchmark_symbol: draft.benchmarkSymbol,
          preference_excess_return: draft.preference.excessReturn,
          preference_annual_return: draft.preference.annualReturn,
          preference_drawdown_control: draft.preference.drawdownControl,
          preference_robustness: draft.preference.robustness,
        },
      });
      setActiveJob(result.data);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "融合搜索提交失败");
    } finally {
      setBusy("");
    }
  };

  const cancel = async (): Promise<void> => {
    if (!job) return;
    setBusy("cancel");
    try {
      const result = await apiPost<JobSummary>(`/jobs/${job.id}/cancel`, {});
      setActiveJob(result.data);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "取消失败");
    } finally {
      setBusy("");
    }
  };

  /* ------------------------------------------------------------ frontier -- */

  const frontierOption = useMemo<EChartsOption>(() => {
    const point = (item: FusionCandidate) => ({
      name: item.id,
      value: [
        item.metrics.maxDrawdown,
        item.metrics.excessReturn,
        item.metrics.robustness,
        item.label,
      ],
    });
    const onFrontier = visibleCandidates.filter((item) => item.onFrontier);
    const dominated = visibleCandidates.filter((item) => !item.onFrontier && !item.isControl);
    const controls = visibleCandidates.filter((item) => item.isControl && !item.onFrontier);
    return {
      animationDuration: 240,
      grid: { left: 56, right: 20, top: 28, bottom: 44 },
      tooltip: {
        trigger: "item",
        formatter: (params: unknown) => {
          const item = params as { value: [number, number, number, string]; name: string };
          return [
            `<strong>${item.value[3]}</strong>`,
            `超额 ${percent(item.value[1])}`,
            `最大回撤 ${percent(item.value[0])}`,
            `稳健性 ${ratio(item.value[2])}`,
          ].join("<br/>");
        },
      },
      legend: { top: 0, right: 0, itemWidth: 10, itemHeight: 10, textStyle: { fontSize: 10 } },
      xAxis: {
        name: "最大回撤 →",
        nameLocation: "middle",
        nameGap: 28,
        nameTextStyle: { fontSize: 10 },
        axisLabel: { formatter: (value: number) => percent(value, 0), fontSize: 10 },
      },
      yAxis: {
        name: "超额收益 ↑",
        nameTextStyle: { fontSize: 10 },
        axisLabel: { formatter: (value: number) => percent(value, 0), fontSize: 10 },
      },
      series: [
        {
          name: "Pareto 前沿",
          type: "scatter",
          symbolSize: (value: number[]) => 12 + value[2] * 22,
          itemStyle: { color: palette.primary, borderColor: palette.tooltipText, borderWidth: 1 },
          data: onFrontier.map(point),
        },
        {
          name: "被支配候选",
          type: "scatter",
          symbolSize: (value: number[]) => 8 + value[2] * 14,
          itemStyle: { color: palette.muted, opacity: 0.65 },
          data: dominated.map(point),
        },
        {
          name: "对照组",
          type: "scatter",
          symbol: "diamond",
          symbolSize: 9,
          itemStyle: { color: "transparent", borderColor: palette.warning, borderWidth: 1.4 },
          data: controls.map(point),
        },
      ],
    };
  }, [palette, visibleCandidates]);

  /* ----------------------------------------------------------------- nav -- */

  const navOption = useMemo<EChartsOption>(() => {
    const rows = navs.data?.data ?? [];
    if (!rows.length) return {};
    const shown = comparisonIds.length
      ? comparisonIds
      : frontier.slice(0, MAX_COMPARE).map((item) => item.id);
    const dates = rows.map((row) => String(row.trade_date ?? ""));
    const labelFor = (id: string): string =>
      candidates.find((item) => item.id === id)?.label ?? id;
    return {
      animationDuration: 240,
      grid: { left: 52, right: 16, top: 30, bottom: 32 },
      tooltip: { trigger: "axis" },
      legend: { top: 0, itemWidth: 12, itemHeight: 8, textStyle: { fontSize: 10 } },
      xAxis: { type: "category", data: dates, axisLabel: { fontSize: 10 } },
      yAxis: { type: "value", scale: true, axisLabel: { fontSize: 10 } },
      series: shown.map((id, index) => ({
        name: labelFor(id),
        type: "line",
        showSymbol: false,
        lineStyle: { width: 1.6, color: palette.series[index % palette.series.length] },
        itemStyle: { color: palette.series[index % palette.series.length] },
        data: rows.map((row) => {
          const value = row[id];
          return typeof value === "number" ? value : null;
        }),
      })),
    };
  }, [candidates, comparisonIds, frontier, navs.data, palette]);

  /* -------------------------------------------------------------- render -- */

  const trials = summary.nTrials ?? null;
  const pbo = summary.pbo ?? null;
  const best = frontier[0];

  return (
    <div className="page institutional-workbench alpha-foundry-page">
      <WorkbenchHeader
        eyebrow="ATLAS L2 / FACTOR FUSION FOUNDRY"
        title="因子融合工场"
        description="枚举融合方案、在清洗过的滚动样本外折上评估，并按最大超额、最小回撤、最大年化与样本外稳健性输出 Pareto 前沿。试验次数由搜索空间决定，不可人工声明。"
        asOf={job?.status ? `${job.status.toUpperCase()} · ${Math.round((job.progress ?? 0) * 100)}%` : "RESEARCH ONLY"}
        context="无实盘订单 · 试验计数进入 DSR"
        actions={
          <>
            <button type="button" onClick={() => void runs.refetch()} disabled={runs.isFetching}>
              <ArrowsClockwise size={14} />刷新搜索产物
            </button>
            {running ? (
              <button type="button" className="danger" onClick={cancel} disabled={busy === "cancel"}>
                <Stop size={14} weight="fill" />取消搜索
              </button>
            ) : null}
          </>
        }
      />

      <WorkbenchMetricStrip
        metrics={[
          {
            label: "搜索产物",
            value: String(runList.length),
            detail: effectiveRunId ? `当前 ${detail.data?.data.path ?? "—"}` : "尚无搜索",
            tone: runList.length ? "info" : "neutral",
            icon: Atom,
          },
          {
            label: "试验次数",
            value: trials === null ? "—" : String(trials),
            detail: "枚举得出 · 用于 Sharpe 收缩",
            tone: "ai",
            icon: Scales,
          },
          {
            label: "前沿规模",
            value: String(frontier.length),
            detail: `${candidates.length} 个候选中非受支配`,
            tone: frontier.length ? "positive" : "neutral",
            icon: ChartScatter,
          },
          {
            label: "过拟合概率",
            value: pbo === null ? "无估计" : ratio(pbo),
            detail: pbo === null ? "时间切片不足以估计 PBO" : "CSCV · 越低越好",
            tone: pbo === null ? "warning" : pbo > 0.5 ? "danger" : "positive",
            icon: ShieldWarning,
          },
          {
            label: "首选超额",
            value: best ? percent(best.metrics.excessReturn) : "—",
            detail: best ? `${best.label} · 回撤 ${percent(best.metrics.maxDrawdown)}` : "无前沿候选",
            tone: excessStripTone(best),
            icon: Target,
          },
          {
            label: "基准口径",
            value: (summary.benchmarkMode ?? "—").startsWith("index") ? "指数" : "宇宙等权",
            detail: summary.benchmarkMode ?? "运行后写入产物",
            tone: (summary.benchmarkMode ?? "").startsWith("index") ? "info" : "warning",
            icon: CheckCircle,
          },
        ]}
      />

      <section className="atlas-split">
        <div className="atlas-stack">
          <WorkbenchPanel
            eyebrow="OBJECTIVE FRONTIER"
            title="四目标 Pareto 前沿"
            meta={
              effectiveRunId
                ? `${visibleCandidates.length} 个候选 · 气泡大小=稳健性`
                : "等待搜索产物"
            }
            actions={
              <label className="foundry-toggle">
                <input
                  type="checkbox"
                  checked={showControls}
                  onChange={(event) => setShowControls(event.target.checked)}
                />
                <span>显示对照组</span>
              </label>
            }
          >
            {candidates.length ? (
              <>
                <EChart
                  option={frontierOption}
                  className="foundry-frontier-chart"
                  ariaLabel="因子融合候选的超额收益与最大回撤散点"
                  onClick={(params) => {
                    const point = params as { name?: string };
                    if (point.name) setInspectedId(point.name);
                  }}
                />
                <TruthNotice>
                  横轴回撤按调仓频率净值序列计算（每 {summary.horizonDays ?? "?"} 个交易日一期），
                  因此低于日频标记的回撤。菱形为对照组：它们不读取训练段，只用于给拟合方案设立可被击败的下限。
                </TruthNotice>
              </>
            ) : (
              <ActionableState
                title={runs.isLoading ? "正在读取搜索产物" : "尚无因子融合搜索产物"}
                detail="右侧配置一次搜索并启动后，前沿、候选明细与净值曲线会在此显示。搜索只写入 Runtime 产物，不产生任何订单。"
                icon={ChartScatter}
                tone={runs.isError ? "danger" : "neutral"}
              />
            )}
          </WorkbenchPanel>

          <WorkbenchPanel
            eyebrow="CANDIDATE LEDGER"
            title="候选账本"
            meta={
              comparisonIds.length
                ? `已选 ${comparisonIds.length}/${MAX_COMPARE} 项对比`
                : "勾选最多 4 项进入净值对比"
            }
          >
            {visibleCandidates.length ? (
              <div className="atlas-scroll-x">
                <table className="atlas-grid foundry-ledger">
                  <thead>
                    <tr>
                      <th scope="col">对比</th>
                      <th scope="col">候选</th>
                      <th scope="col">类型</th>
                      <th scope="col" className="num">超额</th>
                      <th scope="col" className="num">年化</th>
                      <th scope="col" className="num">最大回撤</th>
                      <th scope="col" className="num">稳健性</th>
                      <th scope="col" className="num">换手</th>
                      <th scope="col" className="num">样本</th>
                    </tr>
                  </thead>
                  <tbody>
                    {visibleCandidates.map((candidate) => (
                      <tr
                        key={candidate.id}
                        aria-selected={inspected?.id === candidate.id}
                        data-control={candidate.isControl}
                        onClick={() => setInspectedId(candidate.id)}
                      >
                        <td>
                          <input
                            type="checkbox"
                            aria-label={`对比 ${candidate.label}`}
                            checked={comparisonIds.includes(candidate.id)}
                            disabled={
                              !comparisonIds.includes(candidate.id)
                              && comparisonIds.length >= MAX_COMPARE
                            }
                            onChange={() => toggleComparison(candidate.id)}
                            onClick={(event) => event.stopPropagation()}
                          />
                        </td>
                        <td>
                          <strong>{candidate.label}</strong>
                          {candidate.onFrontier ? (
                            <span className="atlas-chip" data-tone="primary">前沿</span>
                          ) : null}
                        </td>
                        <td>
                          <span
                            className="atlas-chip"
                            data-tone={candidate.isControl ? "control" : "agent"}
                          >
                            {candidate.isControl ? "对照" : "拟合"}
                          </span>
                        </td>
                        <td className={`num tone-${objectiveTone(candidate)}`}>
                          {percent(candidate.metrics.excessReturn)}
                        </td>
                        <td className="num">{percent(candidate.metrics.annualReturn)}</td>
                        <td className="num">{percent(candidate.metrics.maxDrawdown)}</td>
                        <td className="num">
                          <span className="foundry-robustness">
                            {ratio(candidate.metrics.robustness)}
                            <i
                              className="atlas-meter"
                              data-tone={candidate.metrics.robustness > 0.6 ? "success" : "warning"}
                            >
                              <i style={{ width: `${Math.round(candidate.metrics.robustness * 100)}%` }} />
                            </i>
                          </span>
                        </td>
                        <td className="num">{ratio(candidate.metrics.averageTurnover)}</td>
                        <td className="num">{candidate.metrics.observations}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <ActionableState
                title="当前没有可显示的候选"
                detail={
                  candidates.length
                    ? "所有候选都是对照组，已被过滤。打开“显示对照组”查看它们。"
                    : "选择一个搜索产物，或先启动一次融合搜索。"
                }
                compact
                primary={
                  candidates.length
                    ? { label: "显示对照组", onClick: () => setShowControls(true) }
                    : undefined
                }
              />
            )}
          </WorkbenchPanel>

          <WorkbenchPanel
            eyebrow="OUT-OF-SAMPLE NAV"
            title="样本外净值对比"
            meta={
              comparisonIds.length
                ? `${comparisonIds.length} 个已选候选`
                : `默认展示前沿前 ${MAX_COMPARE} 项`
            }
          >
            {navs.data?.data?.length ? (
              <EChart
                option={navOption}
                className="foundry-nav-chart"
                ariaLabel="候选样本外净值曲线"
              />
            ) : (
              <ActionableState
                title="没有净值产物"
                detail="净值曲线来自搜索写入的 fusion_nav.csv；若搜索没有产生可用的样本外观测，该文件不会生成。"
                compact
              />
            )}
          </WorkbenchPanel>
        </div>

        <aside className="atlas-stack">
          <WorkbenchPanel
            eyebrow="SEARCH CONTROL"
            title="配置与启动"
            meta={running ? "运行中 · 可取消" : "governed command · search-factor-fusion"}
          >
            <FusionSearchForm
              draft={draft}
              onChange={setDraft}
              defaults={defaults.data?.data}
              disabled={running || busy === "launch"}
            />
            <div className="foundry-actions">
              <button
                type="button"
                className="primary"
                onClick={launch}
                disabled={running || busy === "launch"}
              >
                <Play size={14} weight="fill" />
                {busy === "launch" ? "提交中" : "启动融合搜索"}
              </button>
              {running ? (
                <button type="button" className="danger" onClick={cancel} disabled={busy === "cancel"}>
                  <Stop size={14} weight="fill" />取消
                </button>
              ) : null}
            </div>
            {error ? (
              <div className="foundry-error" role="alert">
                <Warning size={14} />
                <span>{error}</span>
              </div>
            ) : null}
            {job ? (
              <div className="foundry-job">
                <div className="atlas-row">
                  <span className="atlas-eyebrow">JOB</span>
                  <code>{job.id}</code>
                  <span className="atlas-chip" data-tone={job.status === "failed" ? "danger" : running ? "live" : "success"}>
                    {job.status}
                  </span>
                </div>
                <i className="atlas-meter" data-tone="agent">
                  <i style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }} />
                </i>
                <div className="foundry-console" aria-live="polite">
                  {stream.lines.length
                    ? stream.lines.slice(-40).map((line, index) => (
                        <code key={`${index}-${line.slice(0, 12)}`}>{line}</code>
                      ))
                    : <span>等待搜索输出…</span>}
                </div>
              </div>
            ) : null}
          </WorkbenchPanel>

          <WorkbenchPanel
            eyebrow="RUN SELECTOR"
            title="搜索产物"
            meta={`${runList.length} 个已索引`}
          >
            {runList.length ? (
              <ul className="foundry-run-list">
                {runList.map((run) => (
                  <li key={run.id}>
                    <button
                      type="button"
                      className={run.id === effectiveRunId ? "selected" : ""}
                      onClick={() => setSelectedRunId(run.id)}
                      aria-pressed={run.id === effectiveRunId}
                    >
                      <strong>{run.name}</strong>
                      <small>
                        {run.generatedAt?.slice(0, 16) ?? "未知时间"} · {run.nTrials ?? "?"} 次试验 ·
                        前沿 {run.frontierSize}
                      </small>
                      <small className="mono">{run.contentHash ?? "no hash"}</small>
                    </button>
                  </li>
                ))}
              </ul>
            ) : (
              <ActionableState
                title="Runtime 中没有融合搜索产物"
                detail="搜索完成后会写入 runtime/reports/fusion/<name>/，并带 manifest 与内容哈希。"
                compact
              />
            )}
          </WorkbenchPanel>

          {inspected ? <CandidateInspector candidate={inspected} summary={summary} /> : null}
        </aside>
      </section>
    </div>
  );
}
