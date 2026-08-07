import { useMemo, useState, type ReactNode } from "react";
import {
  ArrowRight,
  ArrowsClockwise,
  Brain,
  CheckCircle,
  Database,
  Flask,
  FloppyDisk,
  FolderOpen,
  Graph,
  Info,
  Lightning,
  Play,
  ShieldCheck,
  SlidersHorizontal,
  Stop,
  Target,
  WarningCircle,
  Wrench,
} from "@phosphor-icons/react";
import type { EChartsOption } from "echarts";
import { apiPost } from "../../api/client";
import type {
  DecisionCouncilMember,
  StrategyRun,
  JobSummary,
  StrategyDefaults,
  StrategyDraft,
  StrategyInputOption,
  StrategyLaunchResult,
  StrategyManifestSummary,
  StrategyValidation,
  StrategyValidationIssue,
} from "../../api/types";
import { EChart } from "../../components/EChart";
import { StatusBadge } from "../../components/StatusBadge";
import { useApi } from "../../hooks/useApi";
import { useJobStream } from "../../hooks/useJobStream";
import {
  TruthNotice,
  WorkbenchHeader,
  WorkbenchMetricStrip,
  WorkbenchPanel,
} from "../workbench/InstitutionalWorkbench";
import { useVNextChartPalette, useVNextTheme } from "../theme";

export const DEFAULT_DRAFT: StrategyDraft = {
  name: "A股多因子 · 滚动 OOS 策略",
  hypothesis: "经人工审核的价量与质量因子在滚动样本外窗口中提供稳定横截面超额，且在A股交易约束和成本后仍然成立。",
  invalidationCriteria: "任一 OOS Gate 失败、最大回撤超过 15%、Sharpe 低于 1.0，或因子相关性/漂移证据失效时，停止晋级并重新研究。",
  marketPanelPath: "runtime/data/gold/full_universe/adjusted_market_panel.parquet",
  labelsPath: "runtime/data/gold/full_universe/labels.parquet",
  fundamentalsRoot: "",
  valuationPath: "",
  disclosuresPath: "",
  sectorMapPath: "",
  trainingDatasetPath: "",
  synthesizedFactorsPath: "",
  outputDir: "runtime/reports/strategy_studio/a_share_multi_factor",
  factorLibrary: "all_reviewed",
  model: "ft_transformer",
  researchPreset: "stable_alpha",
  horizons: "1,5,20,60,120",
  primaryHorizon: 5,
  horizonBlendMethod: "adaptive_oos",
  splitMode: "rolling",
  // 80 selection + 20 holdout days, plus the day NAV differencing consumes,
  // needs 101 OOS days. At 20 days per fold that is 6 folds; 5 produced exactly
  // 100 and aborted in governance after the whole portfolio search had run.
  nSplits: 6,
  requireGpu: true,
  topK: 30,
  topKCandidates: [20, 30, 50],
  stockSelectionModes: ["none", "fundamental"],
  fundamentalSelectionMode: "auto",
  fundamentalSelectionThreshold: 0.5,
  fundamentalBlendWeight: 0.4,
  fundamentalThresholdCandidates: [0.35, 0.5],
  fundamentalBlendCandidates: [0.25, 0.5],
  selectionMaxCandidates: 64,
  selectionMinOosDays: 80,
  selectionMinHoldoutDays: 20,
  maxPbo: 0.25,
  minDsrProbability: 0.95,
  maxSpaPValue: 0.05,
  factorScreeningMode: "pretrain",
  doTMode: "daily_swing",
  minutePanelPath: "",
  maxWeightPerName: 0.06,
  maxSectorWeight: 0.3,
  maxTurnover: 0.45,
  objective: "max_expected_alpha",
  weighting: "rank",
  initialCash: 1_000_000,
  benchmarkSymbol: "",
  objectiveWeights: { excessReturn: 0, annualReturn: 0.45, drawdownControl: 0.55 },
  universeScope: "full",
  universeSymbols: "",
  universeSymbolsFile: "",
  riskLimits: { maxDrawdown: 0.15, maxTurnover: 0.45, minSharpe: 1.0 },
  humanApproved: false,
};

const PIPELINE = [
  { id: "universe", label: "Universe / PIT", icon: Database },
  { id: "factor", label: "因子融合", icon: Graph },
  { id: "training", label: "滚动训练", icon: Brain },
  { id: "portfolio", label: "目标权重", icon: Target },
  { id: "backtest", label: "A股回测", icon: Flask },
  { id: "risk", label: "风控 / Paper", icon: ShieldCheck },
];

const RESEARCH_PRESETS: Array<{
  id: StrategyDraft["researchPreset"];
  label: string;
  description: string;
}> = [
  {
    id: "stable_alpha",
    label: "稳健超额（以最大超额为主，回撤次之）",
    description: "自适应周期融合、适中集中度、成本后 Pareto 选优。",
  },
  {
    id: "drawdown_first",
    label: "防守稳健（以最小回撤为主）",
    description: "更低单票与换手上限，优先保留回撤更小的前沿候选。",
  },
  {
    id: "annual_growth",
    label: "进取年化（以年化收益为主）",
    description: "提高年化偏好与换手预算，但仍受回撤和过拟合闸门约束。",
  },
  {
    id: "transparent_baseline",
    label: "可解释基线（以复现与审计为主）",
    description: "Ridge、主周期、无基本面预筛选，用作必要的消融基线。",
  },
  {
    id: "custom",
    label: "自定义（以人工研究判断为主）",
    description: "保留当前参数；高级设置的责任由研究员承担。",
  },
];

const PRESET_MANAGED_FIELDS = new Set<keyof StrategyDraft>([
  "factorLibrary",
  "model",
  "horizons",
  "primaryHorizon",
  "horizonBlendMethod",
  "topK",
  "topKCandidates",
  "stockSelectionModes",
  "fundamentalSelectionMode",
  "fundamentalSelectionThreshold",
  "fundamentalBlendWeight",
  "fundamentalThresholdCandidates",
  "fundamentalBlendCandidates",
  "selectionMinOosDays",
  "selectionMinHoldoutDays",
  "maxPbo",
  "minDsrProbability",
  "maxSpaPValue",
  "factorScreeningMode",
  "maxWeightPerName",
  "maxSectorWeight",
  "maxTurnover",
  "objective",
  "weighting",
  "riskLimits",
]);

function parseHorizons(value: string): number[] {
  return Array.from(new Set(
    value
      .split(",")
      .map((item) => Number(item.trim()))
      .filter((item) => Number.isInteger(item) && item >= 1 && item <= 252),
  )).sort((left, right) => left - right);
}

function normalizeDraft(draft: StrategyDraft): StrategyDraft {
  const merged = { ...DEFAULT_DRAFT, ...draft };
  const horizons = parseHorizons(merged.horizons);
  return {
    ...merged,
    primaryHorizon: horizons.includes(merged.primaryHorizon)
      ? merged.primaryHorizon
      : (horizons[0] ?? DEFAULT_DRAFT.primaryHorizon),
  };
}


/** Objectives that cannot be measured must not carry preference weight.
 *
 * Excess return is defined against a benchmark. With no benchmark selected the
 * pipeline refuses to optimise for it, so a preset that asks for it produces a
 * blocked launch rather than a preference. Redistribute it instead of shipping
 * a configuration the operator has to repair by hand.
 */
function withMeasurableObjectives(
  weights: StrategyDraft["objectiveWeights"],
  benchmarkSymbol: string | null | undefined,
): StrategyDraft["objectiveWeights"] {
  if (benchmarkSymbol) return weights;
  const spare = weights.excessReturn;
  if (spare <= 0) return weights;
  const remainder = weights.annualReturn + weights.drawdownControl;
  if (remainder <= 0) {
    return { excessReturn: 0, annualReturn: 0.5, drawdownControl: 0.5 };
  }
  const annual = Number((weights.annualReturn + spare * (weights.annualReturn / remainder)).toFixed(3));
  return {
    excessReturn: 0,
    annualReturn: annual,
    drawdownControl: Number((1 - annual).toFixed(3)),
  };
}

function withPreset(
  current: StrategyDraft,
  preset: StrategyDraft["researchPreset"],
): StrategyDraft {
  if (preset === "custom") return { ...current, researchPreset: preset };
  const shared = {
    ...current,
    researchPreset: preset,
    factorScreeningMode: "pretrain" as const,
    horizonBlendMethod: "adaptive_oos" as const,
    topK: 30,
    topKCandidates: [20, 30, 50],
    stockSelectionModes: ["none", "fundamental"] as Array<"none" | "fundamental">,
    fundamentalSelectionMode: "auto" as const,
    fundamentalThresholdCandidates: [0.35, 0.5],
    fundamentalBlendCandidates: [0.25, 0.5],
  };
  const measurable = (weights: StrategyDraft["objectiveWeights"]) =>
    withMeasurableObjectives(weights, current.benchmarkSymbol);
  if (preset === "drawdown_first") {
    return {
      ...shared,
      objective: "min_variance",
      weighting: "rank",
      maxWeightPerName: 0.05,
      maxTurnover: 0.35,
      objectiveWeights: measurable({ excessReturn: 0.25, annualReturn: 0.15, drawdownControl: 0.6 }),
      riskLimits: { ...current.riskLimits, maxDrawdown: 0.1, maxTurnover: 0.35, minSharpe: 0.8 },
    };
  }
  if (preset === "annual_growth") {
    return {
      ...shared,
      objective: "max_expected_alpha",
      weighting: "softmax",
      maxWeightPerName: 0.08,
      maxTurnover: 0.65,
      objectiveWeights: measurable({ excessReturn: 0.35, annualReturn: 0.5, drawdownControl: 0.15 }),
      riskLimits: { ...current.riskLimits, maxDrawdown: 0.2, maxTurnover: 0.65 },
    };
  }
  if (preset === "transparent_baseline") {
    return {
      ...shared,
      model: "ridge",
      requireGpu: false,
      factorLibrary: "basic",
      factorScreeningMode: "evaluate_only",
      horizonBlendMethod: "primary_only",
      topKCandidates: [20, 30],
      stockSelectionModes: ["none"],
      fundamentalSelectionMode: "off",
      objective: "mean_variance",
      weighting: "rank",
      maxWeightPerName: 0.06,
      maxTurnover: 0.4,
      objectiveWeights: measurable({ excessReturn: 0.4, annualReturn: 0.25, drawdownControl: 0.35 }),
      riskLimits: { ...current.riskLimits, maxDrawdown: 0.15, maxTurnover: 0.4 },
    };
  }
  return {
    ...shared,
    objective: "max_expected_alpha",
    weighting: "rank",
    maxWeightPerName: 0.06,
    maxTurnover: 0.45,
    objectiveWeights: measurable({ excessReturn: 0.5, annualReturn: 0.2, drawdownControl: 0.3 }),
    riskLimits: { ...current.riskLimits, maxDrawdown: 0.15, maxTurnover: 0.45, minSharpe: 1.0 },
  };
}

export function StrategyStudioPage(): JSX.Element {
  const theme = useVNextTheme();
  const chartPalette = useVNextChartPalette();
  const [draft, setDraft] = useState<StrategyDraft>(DEFAULT_DRAFT);
  const [validation, setValidation] = useState<StrategyValidation | null>(null);
  const [saved, setSaved] = useState<StrategyManifestSummary | null>(null);
  const [activeJob, setActiveJob] = useState<JobSummary | null>(null);
  const [activeRun, setActiveRun] = useState<StrategyRun | null>(null);
  const [selectedCouncilId, setSelectedCouncilId] = useState("data_quality");
  const [busy, setBusy] = useState<"validate" | "save" | "launch" | "cancel" | "repair" | "">("");
  const [error, setError] = useState("");
  const defaults = useApi<StrategyDefaults>(["strategy-runtime-defaults"], "/strategies/defaults");
  const manifests = useApi<StrategyManifestSummary[]>(["strategy-manifests"], "/strategies");
  const stream = useJobStream(activeJob?.id ?? null);
  const job = stream.job ?? activeJob;
  const running = Boolean(job && ["queued", "running", "cancelling"].includes(job.status));
  const progress = job?.progress ?? 0;
  const activeStage = Math.min(PIPELINE.length - 1, Math.floor(progress * PIPELINE.length));
  const horizonOptions = useMemo(() => parseHorizons(draft.horizons), [draft.horizons]);
  const portfolioCandidateCount = useMemo(() => {
    const topKCount = new Set(draft.topKCandidates).size;
    const baseline = draft.stockSelectionModes.includes("none") ? 1 : 0;
    const fundamental = draft.stockSelectionModes.includes("fundamental")
      ? draft.fundamentalSelectionMode === "auto"
        ? new Set(draft.fundamentalThresholdCandidates).size
          * new Set(draft.fundamentalBlendCandidates).size
        : draft.fundamentalSelectionMode === "off" ? 0 : 1
      : 0;
    return topKCount * (baseline + fundamental);
  }, [draft]);

  const validationIssues = useMemo<StrategyValidationIssue[]>(() => {
    if (!validation) return [];
    if (validation.issues?.length) return validation.issues;
    return [
      ...validation.errors.map((detail, index) => ({
        code: `legacy_error_${index}`,
        severity: "blocking" as const,
        title: "策略契约阻塞",
        detail,
      })),
      ...validation.warnings.map((detail, index) => ({
        code: `legacy_warning_${index}`,
        severity: "warning" as const,
        title: "研究提醒",
        detail,
      })),
    ];
  }, [validation]);
  const blockingIssues = validationIssues.filter((item) => item.severity === "blocking");
  const warningIssues = validationIssues.filter((item) => item.severity === "warning");
  const infoIssues = validationIssues.filter((item) => item.severity === "info");
  const council = validation?.decisionCouncil ?? [];
  const selectedCouncil = council.find((member) => member.id === selectedCouncilId) ?? council[0];

  const resetEvidence = (): void => {
    setValidation(null);
    setSaved(null);
  };

  const update = <K extends keyof StrategyDraft>(key: K, value: StrategyDraft[K]): void => {
    setDraft((current) => ({
      ...current,
      [key]: value,
      ...(PRESET_MANAGED_FIELDS.has(key) ? { researchPreset: "custom" as const } : {}),
    }));
    resetEvidence();
  };

  const updateMany = (values: Partial<StrategyDraft>): void => {
    setDraft((current) => ({ ...current, ...values }));
    resetEvidence();
  };

  const applyPreset = (preset: StrategyDraft["researchPreset"]): void => {
    setDraft((current) => withPreset(current, preset));
    resetEvidence();
  };

  const updateObjective = (key: keyof StrategyDraft["objectiveWeights"], value: number): void => {
    const current = draft.objectiveWeights;
    const otherKeys = (Object.keys(current) as Array<keyof typeof current>).filter((item) => item !== key);
    const remainder = Math.max(0, 1 - value);
    const otherTotal = otherKeys.reduce((sum, item) => sum + current[item], 0) || 1;
    updateMany({
      researchPreset: "custom",
      objectiveWeights: {
        ...current,
        [key]: value,
        [otherKeys[0]]: Number((remainder * current[otherKeys[0]] / otherTotal).toFixed(3)),
        [otherKeys[1]]: Number((remainder * current[otherKeys[1]] / otherTotal).toFixed(3)),
      },
    });
  };

  const updateHorizons = (value: string): void => {
    const horizons = parseHorizons(value);
    setDraft((current) => ({
      ...current,
      horizons: value.replace(/\s+/g, ""),
      researchPreset: "custom",
      primaryHorizon: horizons.includes(current.primaryHorizon)
        ? current.primaryHorizon
        : (horizons[0] ?? current.primaryHorizon),
    }));
    resetEvidence();
  };

  const validate = async (): Promise<StrategyValidation | null> => {
    setBusy("validate");
    setError("");
    try {
      const result = await apiPost<StrategyValidation>("/strategies/validate", draft);
      setValidation(result.data);
      return result.data;
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "策略校验失败");
      return null;
    } finally {
      setBusy("");
    }
  };

  const save = async (): Promise<void> => {
    setBusy("save");
    setError("");
    try {
      const result = await apiPost<StrategyManifestSummary>("/strategies", draft);
      setSaved(result.data);
      const checked = await apiPost<StrategyValidation>("/strategies/validate", draft);
      setValidation(checked.data);
      await manifests.refetch();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "策略保存失败");
    } finally {
      setBusy("");
    }
  };

  const launch = async (): Promise<void> => {
    setBusy("launch");
    setError("");
    try {
      const result = await apiPost<StrategyLaunchResult>("/strategies/launch", draft);
      setSaved(result.data.strategy);
      setActiveJob(result.data.job);
      setActiveRun(result.data.run ?? null);
      const checked = await apiPost<StrategyValidation>("/strategies/validate", draft);
      setValidation(checked.data);
      await manifests.refetch();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "策略启动失败");
    } finally {
      setBusy("");
    }
  };

  const repairLabels = async (): Promise<void> => {
    setBusy("repair");
    setError("");
    try {
      const result = await apiPost<JobSummary>("/jobs/data", {
        commandId: "build-labels-v7",
        parameters: {
          market_panel_path: draft.marketPanelPath,
          output_path: draft.labelsPath,
          horizons: draft.horizons,
        },
      });
      setActiveJob(result.data);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Labels 重建任务提交失败");
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

  const applyRuntimeDefaults = (): void => {
    const selected = defaults.data?.data.selected;
    if (!selected) return;
    updateMany({
      marketPanelPath: selected.marketPanelPath ?? draft.marketPanelPath,
      labelsPath: selected.labelsPath ?? draft.labelsPath,
      trainingDatasetPath: selected.trainingDatasetPath ?? draft.trainingDatasetPath,
      sectorMapPath: selected.sectorMapPath ?? draft.sectorMapPath,
      fundamentalsRoot: selected.fundamentalsRoot ?? draft.fundamentalsRoot,
      valuationPath: selected.valuationPath ?? draft.valuationPath,
      disclosuresPath: selected.disclosuresPath ?? draft.disclosuresPath,
    });
  };

  const adoptAvailableHorizons = (issue: StrategyValidationIssue): void => {
    const available = issue.evidence?.availableHorizons ?? validation?.availableHorizons ?? [];
    if (!available.length) return;
    updateHorizons(available.join(","));
  };

  const applyFundamentalDefault = (): void => {
    const path = defaults.data?.data.selected?.fundamentalsRoot
      ?? defaults.data?.data.options?.fundamentalsRoot?.find((item) => item.exists)?.path;
    if (path) update("fundamentalsRoot", path);
  };

  const resolveBenchmark = (): void => {
    // The old remedy set 000300.SH, which is absent from a full-universe stock
    // panel and made the run abort at the portfolio stage. Clearing the
    // benchmark and its excess-return weight is the configuration that runs.
    updateMany({
      benchmarkSymbol: "",
      researchPreset: "custom",
      objectiveWeights: { excessReturn: 0, annualReturn: 0.5, drawdownControl: 0.5 },
    });
  };

  const resolveOosDays = (issue: StrategyValidationIssue): void => {
    const minimum = Number(issue.evidence?.minimumSplits ?? 0);
    if (minimum > 0) update("nSplits", Math.min(20, minimum));
  };

  const disableFundamentalCandidate = (): void => {
    updateMany({
      fundamentalSelectionMode: "off",
      stockSelectionModes: ["none"],
      researchPreset: "custom",
    });
  };

  const inputOptions = (field: string): StrategyInputOption[] => (
    defaults.data?.data.options?.[field] ?? []
  );

  const objectiveOption = useMemo<EChartsOption>(() => ({
    animationDuration: 240,
    radar: {
      indicator: [
        { name: "最大超额", max: 1 },
        { name: "最大年化", max: 1 },
        { name: "回撤控制", max: 1 },
      ],
      splitNumber: 4,
      radius: "66%",
      axisName: { color: chartPalette.text, fontSize: 10 },
      axisLine: { lineStyle: { color: chartPalette.axis } },
      splitLine: { lineStyle: { color: chartPalette.grid } },
      splitArea: { areaStyle: { color: [chartPalette.slider, chartPalette.sliderData] } },
    },
    tooltip: {
      trigger: "item",
      backgroundColor: chartPalette.tooltip,
      borderColor: chartPalette.tooltipBorder,
      textStyle: { color: chartPalette.tooltipText },
    },
    series: [{
      type: "radar",
      data: [{
        value: [
          draft.objectiveWeights.excessReturn,
          draft.objectiveWeights.annualReturn,
          draft.objectiveWeights.drawdownControl,
        ],
        name: "研究目标权重",
        areaStyle: { color: theme === "day" ? "rgba(23, 107, 194, .16)" : "rgba(64, 180, 255, .22)" },
        lineStyle: { color: theme === "day" ? "#176bc2" : theme === "dawn" ? "#67b5ff" : "#6ea6ff", width: 2 },
        itemStyle: { color: theme === "day" ? "#087e88" : theme === "dawn" ? "#4fd0c8" : "#43c1ce" },
      }],
    }],
  }), [chartPalette, draft.objectiveWeights, theme]);

  return (
    <div className="page institutional-workbench strategy-studio-page">
      <WorkbenchHeader
        eyebrow="STRATEGY OPERATING SYSTEM / RESEARCH → PAPER"
        title="策略实验室"
        description="把研究假设、因子融合、滚动训练、组合优化、A股约束回测和风控验收固化为同一份可版本化策略契约。"
        asOf={job?.status ? `${job.status.toUpperCase()} · ${Math.round(progress * 100)}%` : "DRAFT / RESEARCH ONLY"}
        context="无自由 Shell · 无实盘订单 · 人工 Gate"
        actions={<>
          <button type="button" onClick={applyRuntimeDefaults} disabled={!defaults.data?.data.selected?.marketPanelPath}><Database size={14} />载入 Runtime 数据</button>
          <StatusBadge status={running ? "running" : validation?.valid ? "ready" : "warning"} label={running ? "LIVE STREAM" : validation?.valid ? "VALIDATED" : "NOT VALIDATED"} />
        </>}
      />
      <WorkbenchMetricStrip metrics={[
        { label: "因子库", value: draft.factorLibrary.toUpperCase(), detail: draft.synthesizedFactorsPath ? "含审核后的合成因子" : "native reviewed library", tone: "ai", icon: Graph },
        { label: "融合", value: draft.horizonBlendMethod.toUpperCase(), detail: `${draft.horizons} · primary ${draft.primaryHorizon}D`, tone: "info", icon: ArrowsClockwise },
        { label: "模型", value: draft.model === "ft_transformer" ? "V8 DEEP GPU" : "V7 CLASSICAL", detail: `${draft.nSplits} folds · ${draft.splitMode}`, tone: "info", icon: Brain },
        { label: "候选赛", value: `${portfolioCandidateCount} 组`, detail: `上限 ${draft.selectionMaxCandidates} · frozen holdout`, tone: portfolioCandidateCount > draft.selectionMaxCandidates ? "danger" : "positive", icon: Target },
        { label: "回撤 Gate", value: `≤ ${(draft.riskLimits.maxDrawdown * 100).toFixed(0)}%`, detail: `Sharpe ≥ ${draft.riskLimits.minSharpe.toFixed(1)}`, tone: "warning", icon: ShieldCheck },
        { label: "运行状态", value: job?.status.toUpperCase() ?? "DRAFT", detail: job?.id ?? saved?.version ?? "not persisted", tone: job?.status === "failed" ? "danger" : running ? "ai" : "neutral", icon: Lightning },
      ]} />

      <section className="strategy-studio-grid">
        <div className="strategy-main-column">
          <WorkbenchPanel eyebrow="STRATEGY CONTRACT" title="研究方案与真实输入" meta="schema validated · versioned" className="strategy-contract-panel">
            <TruthNotice>
              默认方案只暴露关键研究选择；高级参数保留在折叠区。Runtime 已发现 {defaults.data?.data.evidence?.length ?? 0} 个真实输入候选，路径仍需人工确认。
            </TruthNotice>
            <div className="strategy-form-grid strategy-core-form">
              <Field label="策略名称" wide><input value={draft.name} onChange={(event) => update("name", event.target.value)} /></Field>
              <Field label="研究方案" wide hint={RESEARCH_PRESETS.find((item) => item.id === draft.researchPreset)?.description}>
                <select value={draft.researchPreset} onChange={(event) => applyPreset(event.target.value as StrategyDraft["researchPreset"])}>
                  {RESEARCH_PRESETS.map((preset) => <option key={preset.id} value={preset.id}>{preset.label}</option>)}
                </select>
              </Field>
              <Field label="因子融合">
                <select value={draft.factorLibrary} onChange={(event) => update("factorLibrary", event.target.value as StrategyDraft["factorLibrary"])}>
                  <option value="all_reviewed">审查后自适应（以跨库去冗余因子为主）</option>
                  <option value="cicc_ashare80">A股基本面（以盈利质量与估值为主）</option>
                  <option value="alpha181">A股价量（以日频价量信号为主）</option>
                  <option value="alpha101">通用价量（以横截面信号为主）</option>
                  <option value="basic">可解释基线（以少量基础因子为主）</option>
                </select>
              </Field>
              <Field label="周期混合">
                <select value={draft.horizonBlendMethod} onChange={(event) => update("horizonBlendMethod", event.target.value as StrategyDraft["horizonBlendMethod"])}>
                  <option value="adaptive_oos">自适应 OOS（以早期 OOS 稳定 RankIC 为主）</option>
                  <option value="balanced">均衡周期（以 20D / 60D 中周期为主）</option>
                  <option value="short_tactical">短线战术（以 1D / 5D 响应为主）</option>
                  <option value="long_fundamental">中长基本面（以 20D / 60D / 120D 为主）</option>
                  <option value="primary_only">主周期基线（以单一主周期为主）</option>
                </select>
              </Field>
              <Field label="模型">
                <select value={draft.model} onChange={(event) => {
                  const model = event.target.value as StrategyDraft["model"];
                  updateMany({ model, requireGpu: model === "ft_transformer", researchPreset: "custom" });
                }}>
                  <option value="ft_transformer">FT-Transformer（以非线性交互为主，GPU fail closed）</option>
                  <option value="ridge">Ridge（以可解释稳定基线为主）</option>
                </select>
              </Field>
              <Field label="T+1 执行模式">
                <select value={draft.doTMode} onChange={(event) => update("doTMode", event.target.value as StrategyDraft["doTMode"])}>
                  <option value="daily_swing">日线波段（以 ATR timing gate 为主）</option>
                  <option value="intraday">分钟做T（以 T+1 可售库存约束为主）</option>
                  <option value="both">联合对照（分别报告，不混淆能力）</option>
                  <option value="off">关闭（以纯策略基线为主）</option>
                </select>
              </Field>
              <Field label="研究范围" hint={draft.universeScope === "pilot" ? "试点结论不可外推到全宇宙" : "全部可用标的，耗时最长"}>
                <select value={draft.universeScope} onChange={(event) => {
                  const scope = event.target.value as StrategyDraft["universeScope"];
                  updateMany({ universeScope: scope, researchPreset: "custom" });
                }}>
                  <option value="full">全宇宙（以完整结论为主）</option>
                  <option value="pilot">试点宇宙（先验证配置与链路）</option>
                </select>
              </Field>
              {draft.universeScope === "pilot"
                ? <Field label="试点标的清单" wide hint="项目内一行一个 symbol 的文件路径，或在下方直接填写逗号分隔的 symbol">
                    <input className="mono" value={draft.universeSymbolsFile ?? ""} onChange={(event) => update("universeSymbolsFile", event.target.value)} placeholder="runtime/data/u0/pilot_symbols.txt" />
                  </Field>
                : null}
              {draft.universeScope === "pilot"
                ? <Field label="或直接指定 symbol" wide>
                    <input className="mono" value={draft.universeSymbols ?? ""} onChange={(event) => update("universeSymbols", event.target.value)} placeholder="000001.SZ,600519.SH" />
                  </Field>
                : null}
              <Field label="基准指数" hint="留空表示不做超额比较；个股全宇宙面板通常不含指数标的">
                <input value={draft.benchmarkSymbol ?? ""} onChange={(event) => updateMany({
                  benchmarkSymbol: event.target.value,
                  objectiveWeights: withMeasurableObjectives(draft.objectiveWeights, event.target.value),
                })} />
              </Field>
              <Field label="Horizon 集合"><input aria-describedby="strategy-horizon-help" value={draft.horizons} onChange={(event) => updateHorizons(event.target.value)} /></Field>
              <Field label="主周期">
                <select value={draft.primaryHorizon} onChange={(event) => update("primaryHorizon", Number(event.target.value))} disabled={!horizonOptions.length}>
                  {horizonOptions.map((horizon) => <option key={horizon} value={horizon}>{horizon}D · forward_return_{horizon}d</option>)}
                </select>
              </Field>
              <div id="strategy-horizon-help" className="strategy-field-help">自适应权重只读取早期 OOS，按 Horizon 清除 forward-label 重叠区，并在最终 holdout 之前冻结；Labels 必须包含所选周期列。</div>
            </div>

            <div className="strategy-section-title"><FolderOpen size={15} /><span><strong>数据契约</strong><small>从真实候选中选择，也可手动输入项目内路径</small></span></div>
            <div className="strategy-form-grid">
              <PathField label="Market panel" value={draft.marketPanelPath} options={inputOptions("marketPanelPath")} onChange={(value) => update("marketPanelPath", value)} />
              <PathField label="Labels" value={draft.labelsPath} options={inputOptions("labelsPath")} onChange={(value) => update("labelsPath", value)} />
              <PathField label="基本面 PIT 根目录" value={draft.fundamentalsRoot ?? ""} options={inputOptions("fundamentalsRoot")} onChange={(value) => update("fundamentalsRoot", value)} />
              <Field label="基本面学习">
                <select value={draft.fundamentalSelectionMode} onChange={(event) => {
                  const mode = event.target.value as StrategyDraft["fundamentalSelectionMode"];
                  updateMany({
                    fundamentalSelectionMode: mode,
                    stockSelectionModes: mode === "off" ? ["none"] : ["none", "fundamental"],
                    researchPreset: "custom",
                  });
                }}>
                  <option value="auto">早期 OOS 学习（以无预筛选基线作对照）</option>
                  <option value="fixed">固定复现（以人工声明阈值为主）</option>
                  <option value="off">关闭（以无预筛选基线为主）</option>
                </select>
              </Field>
              {draft.doTMode === "intraday" || draft.doTMode === "both"
                ? <Field label="分钟数据路径" wide><input className="mono" value={draft.minutePanelPath ?? ""} onChange={(event) => update("minutePanelPath", event.target.value)} /></Field>
                : null}
            </div>

            <details className="strategy-advanced">
              <summary><SlidersHorizontal size={16} /><span><strong>高级研究契约</strong><small>候选网格、统计闸门、风险限额与可选 PIT 输入</small></span></summary>
              <div className="strategy-form-grid">
                <Field label="研究假设" wide><textarea value={draft.hypothesis} onChange={(event) => update("hypothesis", event.target.value)} /></Field>
                <Field label="失效条件" wide><textarea value={draft.invalidationCriteria} onChange={(event) => update("invalidationCriteria", event.target.value)} /></Field>
                <PathField label="估值 PIT" value={draft.valuationPath ?? ""} options={inputOptions("valuationPath")} onChange={(value) => update("valuationPath", value)} />
                <PathField label="披露日 PIT" value={draft.disclosuresPath ?? ""} options={inputOptions("disclosuresPath")} onChange={(value) => update("disclosuresPath", value)} />
                <PathField label="训练数据集" value={draft.trainingDatasetPath ?? ""} options={inputOptions("trainingDatasetPath")} onChange={(value) => update("trainingDatasetPath", value)} />
                <PathField label="行业映射" value={draft.sectorMapPath ?? ""} options={inputOptions("sectorMapPath")} onChange={(value) => update("sectorMapPath", value)} />
                <Field label="Top K"><input type="number" min={5} max={500} value={draft.topK} onChange={(event) => update("topK", Number(event.target.value))} /></Field>
                <Field label="Top K 候选"><input value={draft.topKCandidates.join(",")} onChange={(event) => update("topKCandidates", event.target.value.split(",").map(Number).filter((value) => Number.isFinite(value) && value > 0))} /></Field>
                {draft.fundamentalSelectionMode === "auto" ? <>
                  <Field label="基本面入围分位候选"><input value={draft.fundamentalThresholdCandidates.join(",")} onChange={(event) => update("fundamentalThresholdCandidates", event.target.value.split(",").map(Number).filter(Number.isFinite))} /></Field>
                  <Field label="基本面混合权重候选"><input value={draft.fundamentalBlendCandidates.join(",")} onChange={(event) => update("fundamentalBlendCandidates", event.target.value.split(",").map(Number).filter(Number.isFinite))} /></Field>
                </> : draft.fundamentalSelectionMode === "fixed" ? <>
                  <Field label="固定入围分位"><input type="number" step="0.05" min="0.05" max="0.95" value={draft.fundamentalSelectionThreshold} onChange={(event) => update("fundamentalSelectionThreshold", Number(event.target.value))} /></Field>
                  <Field label="固定混合权重"><input type="number" step="0.05" min="0" max="1" value={draft.fundamentalBlendWeight} onChange={(event) => update("fundamentalBlendWeight", Number(event.target.value))} /></Field>
                </> : null}
                <Field label="早期 OOS 最少交易日"><input type="number" min="20" max="1260" value={draft.selectionMinOosDays} onChange={(event) => update("selectionMinOosDays", Number(event.target.value))} /></Field>
                <Field label="最终 Holdout 最少交易日"><input type="number" min="10" max="504" value={draft.selectionMinHoldoutDays} onChange={(event) => update("selectionMinHoldoutDays", Number(event.target.value))} /></Field>
                <Field label="因子筛选">
                  <select value={draft.factorScreeningMode} onChange={(event) => update("factorScreeningMode", event.target.value as StrategyDraft["factorScreeningMode"])}>
                    <option value="pretrain">预训练筛选（以去冗余和稳定性为主）</option>
                    <option value="evaluate_only">只评估（以消融对照为主）</option>
                    <option value="off">关闭（以原始基线为主）</option>
                  </select>
                </Field>
                <Field label="组合目标">
                  <select value={draft.objective} onChange={(event) => update("objective", event.target.value as StrategyDraft["objective"])}>
                    <option value="max_expected_alpha">最大预期 Alpha</option>
                    <option value="mean_variance">均值-方差</option>
                    <option value="min_variance">最小方差</option>
                  </select>
                </Field>
                <Field label="单票上限"><input type="number" step="0.01" min="0.01" max="0.25" value={draft.maxWeightPerName} onChange={(event) => update("maxWeightPerName", Number(event.target.value))} /></Field>
                <Field label="行业上限"><input type="number" step="0.01" min="0.05" max="1" value={draft.maxSectorWeight} onChange={(event) => update("maxSectorWeight", Number(event.target.value))} /></Field>
                <Field label="换手上限"><input type="number" step="0.05" min="0.05" max="2" value={draft.maxTurnover} onChange={(event) => update("maxTurnover", Number(event.target.value))} /></Field>
                <Field label="最大回撤 Gate"><input type="number" step="0.01" min="0.01" max="0.8" value={draft.riskLimits.maxDrawdown} onChange={(event) => update("riskLimits", { ...draft.riskLimits, maxDrawdown: Number(event.target.value) })} /></Field>
                <Field label="风险换手 Gate"><input type="number" step="0.05" min="0.05" max="2" value={draft.riskLimits.maxTurnover} onChange={(event) => update("riskLimits", { ...draft.riskLimits, maxTurnover: Number(event.target.value) })} /></Field>
                <Field label="最低 Sharpe"><input type="number" step="0.1" min="-5" max="10" value={draft.riskLimits.minSharpe} onChange={(event) => update("riskLimits", { ...draft.riskLimits, minSharpe: Number(event.target.value) })} /></Field>
                <Field label="最大 PBO"><input type="number" step="0.05" min="0" max="1" value={draft.maxPbo} onChange={(event) => update("maxPbo", Number(event.target.value))} /></Field>
                <Field label="最低 DSR 概率"><input type="number" step="0.05" min="0" max="1" value={draft.minDsrProbability} onChange={(event) => update("minDsrProbability", Number(event.target.value))} /></Field>
                <Field label="最大 SPA p-value"><input type="number" step="0.01" min="0" max="1" value={draft.maxSpaPValue} onChange={(event) => update("maxSpaPValue", Number(event.target.value))} /></Field>
                <Field label="输出目录" wide><input className="mono" value={draft.outputDir} onChange={(event) => update("outputDir", event.target.value)} /></Field>
                <div className={`strategy-search-budget ${portfolioCandidateCount > draft.selectionMaxCandidates ? "invalid" : ""}`}>
                  <strong>{portfolioCandidateCount} / {draft.selectionMaxCandidates} 组搜索预算</strong>
                  <span>所有候选使用同一成本模型；参数仅在早期 OOS 学习，冻结后最终 Holdout 只验收一次。</span>
                </div>
              </div>
            </details>
          </WorkbenchPanel>

          <WorkbenchPanel eyebrow="PIPELINE CONTROL" title="训练—回测—风控闭环" meta={stream.connected ? "SSE connected" : job ? "stream reconnectable" : "waiting for launch"} className="strategy-pipeline-panel">
            <div className="strategy-pipeline">
              {PIPELINE.map(({ id, label, icon: StageIcon }, index) => {
                const complete = job?.status === "succeeded" || (running && index < activeStage);
                const active = running && index === activeStage;
                const failed = job?.status === "failed" && index === activeStage;
                return <article key={id} className={complete ? "complete" : active ? "active" : failed ? "failed" : ""}>
                  <span>{String(index + 1).padStart(2, "0")}</span><StageIcon size={19} weight="duotone" />
                  <div><strong>{label}</strong><small>{complete ? "evidence persisted" : active ? "running · live" : failed ? "blocked" : "pending"}</small></div>
                  <i>{complete ? <CheckCircle weight="fill" /> : active ? <ArrowsClockwise /> : failed ? <WarningCircle /> : null}</i>
                </article>;
              })}
            </div>
            <div className="strategy-progress"><i style={{ width: `${Math.round(progress * 100)}%` }} /><span>{Math.round(progress * 100)}%</span></div>
          </WorkbenchPanel>

          <WorkbenchPanel eyebrow="DECISION COUNCIL" title="多 Agent 审查" meta="click to inspect · role-scoped veto" className="strategy-council-panel">
            {council.length ? <>
              <div className="strategy-council" aria-label="多 Agent 审查角色">
                {council.map((member) => (
                  <button
                    type="button"
                    key={member.id}
                    className={`status-${member.status} ${selectedCouncil?.id === member.id ? "selected" : ""}`}
                    onClick={() => setSelectedCouncilId(member.id)}
                    aria-pressed={selectedCouncil?.id === member.id}
                  >
                    <span>{member.veto ? "VETO" : "REVIEW"} · {member.issueCount ?? 0}</span>
                    <strong>{member.label}</strong>
                    <small>{member.finding ?? member.responsibility}</small>
                    <StatusBadge status={member.status === "approved" || member.status === "ready" ? "ready" : member.status === "blocked" ? "error" : "partial"} label={member.status} />
                  </button>
                ))}
              </div>
              {selectedCouncil ? <CouncilInspector member={selectedCouncil} issues={validationIssues} /> : null}
            </> : <TruthNotice>先校验策略，系统再基于真实输入生成角色级审查队列；点击角色可检查发现、否决范围和下一步动作。</TruthNotice>}
          </WorkbenchPanel>
        </div>

        <aside className="strategy-side-column">
          <WorkbenchPanel eyebrow="OBJECTIVE FRONTIER" title="优化偏好" meta="operator input · not a performance claim" className="strategy-objective-panel">
            <label className="strategy-preset-compact">
              <span>当前方案</span>
              <select value={draft.researchPreset} onChange={(event) => applyPreset(event.target.value as StrategyDraft["researchPreset"])}>
                {RESEARCH_PRESETS.map((preset) => <option key={preset.id} value={preset.id}>{preset.label}</option>)}
              </select>
            </label>
            <EChart option={objectiveOption} className="strategy-objective-chart" />
            <ObjectiveSlider label="最大超额" value={draft.objectiveWeights.excessReturn} onChange={(value) => updateObjective("excessReturn", value)} />
            <ObjectiveSlider label="最大年化" value={draft.objectiveWeights.annualReturn} onChange={(value) => updateObjective("annualReturn", value)} />
            <ObjectiveSlider label="最小回撤" value={draft.objectiveWeights.drawdownControl} onChange={(value) => updateObjective("drawdownControl", value)} />
            <TruthNotice>目标通常冲突。权重只在早期 OOS 的 Pareto 前沿内排序，最终 holdout 不参与选择。</TruthNotice>
          </WorkbenchPanel>

          <WorkbenchPanel eyebrow="RUN INSPECTOR" title="验证、修复与启动" meta="human gated · cancellable" className="strategy-run-panel">
            <label className="strategy-version-picker">
              <span>活动策略版本</span>
              <select value={saved?.path ?? ""} onChange={(event) => {
                const items = Array.isArray(manifests.data?.data) ? manifests.data.data : [];
                const selected = items.find((item) => item.path === event.target.value);
                if (!selected) return;
                setDraft(normalizeDraft(selected.draft));
                setSaved(selected);
                setValidation(null);
                setError("");
              }}>
                <option value="">当前未保存草稿</option>
                {(Array.isArray(manifests.data?.data) ? manifests.data.data : []).slice(0, 30).map((item) => <option key={item.path} value={item.path}>{item.name} · {item.version}</option>)}
              </select>
            </label>
            <div className="strategy-run-actions">
              <button type="button" onClick={validate} disabled={Boolean(busy) || running}><ShieldCheck />{busy === "validate" ? "校验中" : "校验"}</button>
              <button type="button" onClick={save} disabled={Boolean(busy) || running}><FloppyDisk />{busy === "save" ? "保存中" : "保存版本"}</button>
              <button type="button" className="primary" onClick={launch} disabled={Boolean(busy) || running || !draft.humanApproved}><Play weight="fill" />{busy === "launch" ? "提交中" : "启动闭环"}</button>
              <button type="button" className="danger" onClick={cancel} disabled={!running || busy === "cancel"}><Stop weight="fill" />取消</button>
            </div>
            <label className="strategy-arm">
              <input type="checkbox" checked={draft.humanApproved} onChange={(event) => update("humanApproved", event.target.checked)} />
              <span><strong>Human Gate：授权研究 / paper 任务</strong><small>不启用实盘，不绕过风险否决；每次配置变更都需要重新校验。</small></span>
            </label>

            {validation ? <div className={`strategy-validation ${validation.valid ? "valid" : "invalid"}`}>
              <div className="strategy-validation-head">
                {validation.valid ? <CheckCircle weight="fill" /> : <WarningCircle weight="fill" />}
                <span><strong>{validation.valid ? "Schema 与路径校验通过" : `${blockingIssues.length} 项阻塞`}</strong><small>{warningIssues.length} 项待确认 · {infoIssues.length} 项研究协议</small></span>
              </div>
              {blockingIssues.map((issue) => (
                <IssueCard
                  key={issue.code}
                  issue={issue}
                  canApplyDefaults={Boolean(defaults.data?.data.selected?.marketPanelPath)}
                  canApplyFundamentals={Boolean(defaults.data?.data.selected?.fundamentalsRoot)}
                  running={running}
                  repairing={busy === "repair"}
                  onApplyDefaults={applyRuntimeDefaults}
                  onAdoptHorizons={() => adoptAvailableHorizons(issue)}
                  onRepairLabels={repairLabels}
                  onApplyFundamentals={applyFundamentalDefault}
                  onDisableFundamentals={disableFundamentalCandidate}
                  onResolveBenchmark={resolveBenchmark}
                  onResolveOosDays={() => resolveOosDays(issue)}
                />
              ))}
              {warningIssues.map((issue) => (
                <IssueCard
                  key={issue.code}
                  issue={issue}
                  canApplyDefaults={Boolean(defaults.data?.data.selected?.marketPanelPath)}
                  canApplyFundamentals={Boolean(defaults.data?.data.selected?.fundamentalsRoot)}
                  running={running}
                  repairing={busy === "repair"}
                  onApplyDefaults={applyRuntimeDefaults}
                  onAdoptHorizons={() => adoptAvailableHorizons(issue)}
                  onRepairLabels={repairLabels}
                  onApplyFundamentals={applyFundamentalDefault}
                  onDisableFundamentals={disableFundamentalCandidate}
                  onResolveBenchmark={resolveBenchmark}
                  onResolveOosDays={() => resolveOosDays(issue)}
                />
              ))}
              {infoIssues.length ? <details className="strategy-protocols">
                <summary><Info size={14} />查看 {infoIssues.length} 项研究协议</summary>
                {infoIssues.map((issue) => <div key={issue.code}><strong>{issue.title}</strong><span>{issue.detail}</span></div>)}
              </details> : null}
            </div> : null}
            {activeRun ? <div className="strategy-run-handoff">
              <span>
                <strong>{running ? "运行进行中" : job?.status === "succeeded" ? "运行完成，结论已就绪" : job?.status === "rejected" ? "运行完成并被研究闸门否决" : job?.status === "blocked" ? "配置无法执行该协议，运行未开始训练" : job?.status === "failed" ? "运行中止，已生成诊断" : "运行已登记"}</strong>
                <small>{activeRun.runId} · {activeRun.outputDir}</small>
              </span>
              <a href={`/runs?run=${encodeURIComponent(activeRun.runId)}`}>
                查看阶段、诊断与结论<ArrowRight size={14} />
              </a>
            </div> : null}
            {saved ? <div className="strategy-saved"><CheckCircle weight="fill" /><span><strong>{saved.version}</strong><small>{saved.path} · {saved.contentHash?.slice(0, 12)}</small></span></div> : null}
            {error ? <div className="strategy-error" role="alert"><WarningCircle /><span><strong>无法提交当前策略</strong>{error.split("；").map((item) => <small key={item}>{item}</small>)}</span></div> : null}
          </WorkbenchPanel>

          <WorkbenchPanel eyebrow="LIVE TELEMETRY" title="实时任务流" meta={`${stream.lines.length} bounded log lines`} className="strategy-console-panel">
            <div className="strategy-console" aria-live="polite">
              {stream.lines.length ? stream.lines.map((line, index) => <code key={`${index}-${line.slice(0, 16)}`}>{line}</code>) : <span>启动后在此显示 Labels 修复、数据构建、训练、预测、组合、回测和 paper report 的实时输出。</span>}
            </div>
            {stream.error ? <TruthNotice tone="warning">{stream.error}</TruthNotice> : null}
          </WorkbenchPanel>
        </aside>
      </section>
    </div>
  );
}

function Field({
  label,
  wide = false,
  hint,
  children,
}: {
  label: string;
  wide?: boolean;
  hint?: string;
  children: ReactNode;
}): JSX.Element {
  return <label className={wide ? "wide" : ""}><span>{label}</span>{children}{hint ? <small className="strategy-input-hint">{hint}</small> : null}</label>;
}

function PathField({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: StrategyInputOption[];
  onChange: (value: string) => void;
}): JSX.Element {
  const listId = `strategy-path-${Array.from(label).map((character) => character.codePointAt(0)?.toString(36)).join("-")}`;
  const selected = options.find((item) => item.path === value);
  return <Field label={label} wide hint={options.length ? `${options.filter((item) => item.exists).length} 个可用 / ${options.length} 个 canonical 候选${selected ? ` · ${selected.exists ? "已发现" : "尚不存在"}` : " · 当前为手动路径"}` : "当前未返回 canonical 候选"}>
    <input className="mono" list={listId} value={value} onChange={(event) => onChange(event.target.value)} />
    <datalist id={listId}>
      {options.map((option) => <option key={option.path} value={option.path}>{option.exists ? `可用${option.availableHorizons.length ? ` · ${option.availableHorizons.join("/")}D` : ""}` : "尚未构建"}</option>)}
    </datalist>
  </Field>;
}

function ObjectiveSlider({
  label,
  value,
  onChange,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}): JSX.Element {
  return <label className="strategy-objective-slider"><span>{label}</span><input type="range" min="0.05" max="0.9" step="0.05" value={value} onChange={(event) => onChange(Number(event.target.value))} /><strong>{Math.round(value * 100)}%</strong></label>;
}

function IssueCard({
  issue,
  canApplyDefaults,
  canApplyFundamentals,
  running,
  repairing,
  onApplyDefaults,
  onAdoptHorizons,
  onRepairLabels,
  onApplyFundamentals,
  onDisableFundamentals,
  onResolveBenchmark,
  onResolveOosDays,
}: {
  issue: StrategyValidationIssue;
  canApplyDefaults: boolean;
  canApplyFundamentals: boolean;
  running: boolean;
  repairing: boolean;
  onApplyDefaults: () => void;
  onAdoptHorizons: () => void;
  onRepairLabels: () => void;
  onApplyFundamentals: () => void;
  onDisableFundamentals: () => void;
  onResolveBenchmark: () => void;
  onResolveOosDays: () => void;
}): JSX.Element {
  const missingHorizons = issue.evidence?.missingHorizons ?? [];
  const availableHorizons = issue.evidence?.availableHorizons ?? [];
  return <article className={`strategy-issue strategy-issue-${issue.severity}`}>
    <div><Wrench size={14} /><span><strong>{issue.title}</strong><small>{issue.detail}</small></span></div>
    {issue.code === "missing_horizon_columns" ? <div className="strategy-issue-evidence">
      <span>缺少 {missingHorizons.length ? missingHorizons.map((item) => `${item}D`).join(" / ") : "所选周期"}</span>
      <span>现有 {availableHorizons.length ? availableHorizons.map((item) => `${item}D`).join(" / ") : "无可用周期"}</span>
    </div> : null}
    <div className="strategy-issue-actions">
      {issue.code === "missing_horizon_columns" && availableHorizons.length
        ? <button type="button" onClick={onAdoptHorizons}>采用现有周期</button>
        : null}
      {["missing_horizon_columns", "labels_unreadable"].includes(issue.code)
        ? <button type="button" className="primary" onClick={onRepairLabels} disabled={running || repairing}>{repairing ? "提交中" : "补齐 Labels"}</button>
        : null}
      {issue.code === "fundamentals_missing" && canApplyFundamentals
        ? <button type="button" onClick={onApplyFundamentals}>使用可用目录</button>
        : null}
      {issue.code === "fundamentals_missing"
        ? <button type="button" onClick={onDisableFundamentals}>关闭基本面候选</button>
        : null}
      {["benchmark_missing", "benchmark_absent_from_panel"].includes(issue.code)
        ? <button type="button" className="primary" onClick={onResolveBenchmark}>清空基准并把超额权重设为 0</button>
        : null}
      {issue.code === "insufficient_projected_oos_days"
        ? <button type="button" className="primary" onClick={onResolveOosDays}>
            提高到 {issue.evidence?.minimumSplits ?? "所需"} 折
          </button>
        : null}
      {issue.code.endsWith("_missing") && !["fundamentals_missing", "missing_horizon_columns"].includes(issue.code) && canApplyDefaults
        ? <button type="button" onClick={onApplyDefaults}>载入可用输入</button>
        : null}
    </div>
  </article>;
}

function CouncilInspector({
  member,
  issues,
}: {
  member: DecisionCouncilMember;
  issues: StrategyValidationIssue[];
}): JSX.Element {
  const related = issues.filter((issue) => member.issueCodes?.includes(issue.code));
  return <div className="strategy-council-inspector">
    <div><span>{member.veto ? "VETO AUTHORITY" : "REVIEW AUTHORITY"}</span><strong>{member.label}</strong><small>{member.responsibility}</small></div>
    <div><span>当前判断</span><strong>{member.finding ?? "等待结构化审查"}</strong><small>{related.find((item) => item.severity !== "info")?.detail ?? "当前无角色级阻塞；运行产物生成后仍需复核证据。"}</small></div>
    <div><span>下一步</span><strong>{member.nextAction ?? "复核运行证据"}</strong><small>{member.status === "blocked" ? "修复后重新校验，角色状态才会更新。" : "点击其他角色可切换审查视角。"}</small></div>
  </div>;
}
