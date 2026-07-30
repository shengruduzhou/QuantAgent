import { useMemo, useState, type ReactNode } from "react";
import {
  ArrowsClockwise,
  Brain,
  CheckCircle,
  Database,
  Flask,
  FloppyDisk,
  Graph,
  Lightning,
  Play,
  ShieldCheck,
  Stop,
  Target,
  WarningCircle,
} from "@phosphor-icons/react";
import type { EChartsOption } from "echarts";
import { apiPost } from "../../api/client";
import type {
  JobSummary,
  StrategyDefaults,
  StrategyDraft,
  StrategyLaunchResult,
  StrategyManifestSummary,
  StrategyValidation,
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

const DEFAULT_DRAFT: StrategyDraft = {
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
  horizons: "1,5,20,60,120",
  primaryHorizon: 5,
  splitMode: "rolling",
  nSplits: 4,
  requireGpu: true,
  topK: 30,
  topKCandidates: [10, 20, 30, 50, 80],
  stockSelectionModes: ["none", "fundamental"],
  fundamentalSelectionThreshold: 0.5,
  factorScreeningMode: "pretrain",
  doTMode: "daily_swing",
  minutePanelPath: "",
  maxWeightPerName: 0.08,
  maxSectorWeight: 0.30,
  maxTurnover: 0.50,
  objective: "max_expected_alpha",
  weighting: "rank",
  initialCash: 1_000_000,
  benchmarkSymbol: "000300.SH",
  objectiveWeights: { excessReturn: 0.45, annualReturn: 0.30, drawdownControl: 0.25 },
  riskLimits: { maxDrawdown: 0.15, maxTurnover: 0.50, minSharpe: 1.0 },
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

export function StrategyStudioPage(): JSX.Element {
  const theme = useVNextTheme();
  const chartPalette = useVNextChartPalette();
  const [draft, setDraft] = useState<StrategyDraft>(DEFAULT_DRAFT);
  const [validation, setValidation] = useState<StrategyValidation | null>(null);
  const [saved, setSaved] = useState<StrategyManifestSummary | null>(null);
  const [activeJob, setActiveJob] = useState<JobSummary | null>(null);
  const [busy, setBusy] = useState<"validate" | "save" | "launch" | "cancel" | "">("");
  const [error, setError] = useState("");
  const defaults = useApi<StrategyDefaults>(["strategy-runtime-defaults"], "/strategies/defaults");
  const manifests = useApi<StrategyManifestSummary[]>(["strategy-manifests"], "/strategies");
  const stream = useJobStream(activeJob?.id ?? null);
  const job = stream.job ?? activeJob;
  const running = Boolean(job && ["queued", "running", "cancelling"].includes(job.status));
  const progress = job?.progress ?? 0;
  const activeStage = Math.min(PIPELINE.length - 1, Math.floor(progress * PIPELINE.length));

  const update = <K extends keyof StrategyDraft>(key: K, value: StrategyDraft[K]): void => {
    setDraft((current) => ({ ...current, [key]: value }));
    setValidation(null);
    setSaved(null);
  };

  const updateObjective = (key: keyof StrategyDraft["objectiveWeights"], value: number): void => {
    const current = draft.objectiveWeights;
    const otherKeys = (Object.keys(current) as Array<keyof typeof current>).filter((item) => item !== key);
    const remainder = Math.max(0, 1 - value);
    const otherTotal = otherKeys.reduce((sum, item) => sum + current[item], 0) || 1;
    update("objectiveWeights", {
      ...current,
      [key]: value,
      [otherKeys[0]]: Number((remainder * current[otherKeys[0]] / otherTotal).toFixed(3)),
      [otherKeys[1]]: Number((remainder * current[otherKeys[1]] / otherTotal).toFixed(3)),
    });
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
      const checked = await apiPost<StrategyValidation>("/strategies/validate", draft);
      setValidation(checked.data);
      await manifests.refetch();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "策略启动失败");
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
    setDraft((current) => ({
      ...current,
      marketPanelPath: selected.marketPanelPath ?? current.marketPanelPath,
      labelsPath: selected.labelsPath ?? current.labelsPath,
      trainingDatasetPath: selected.trainingDatasetPath ?? current.trainingDatasetPath,
      sectorMapPath: selected.sectorMapPath ?? current.sectorMapPath,
      fundamentalsRoot: selected.fundamentalsRoot ?? current.fundamentalsRoot,
      valuationPath: selected.valuationPath ?? current.valuationPath,
      disclosuresPath: selected.disclosuresPath ?? current.disclosuresPath,
    }));
    setValidation(null);
    setSaved(null);
  };

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
        { label: "模型", value: draft.model === "ft_transformer" ? "V8 DEEP GPU" : "V7 CLASSICAL", detail: `${draft.nSplits} folds · ${draft.splitMode}`, tone: "info", icon: Brain },
        { label: "主周期", value: `${draft.primaryHorizon}D`, detail: draft.horizons, tone: "info", icon: ArrowsClockwise },
        { label: "持仓搜索", value: `${draft.topKCandidates.length} 组`, detail: draft.topKCandidates.join(" / "), tone: "positive", icon: Target },
        { label: "回撤 Gate", value: `≤ ${(draft.riskLimits.maxDrawdown * 100).toFixed(0)}%`, detail: `Sharpe ≥ ${draft.riskLimits.minSharpe.toFixed(1)}`, tone: "warning", icon: ShieldCheck },
        { label: "运行状态", value: job?.status.toUpperCase() ?? "DRAFT", detail: job?.id ?? saved?.version ?? "not persisted", tone: job?.status === "failed" ? "danger" : running ? "ai" : "neutral", icon: Lightning },
      ]} />

      <section className="strategy-studio-grid">
        <WorkbenchPanel eyebrow="STRATEGY CONTRACT" title="研究假设与输入" meta="schema validated · versioned" className="strategy-contract-panel">
          <TruthNotice>
            Runtime 只检查已知 canonical 路径，不递归扫描大文件；已发现 {defaults.data?.data.evidence?.length ?? 0} 个可用输入候选。点击页首按钮后再人工确认。
          </TruthNotice>
          <div className="strategy-form-grid">
            <Field label="策略名称" wide><input value={draft.name} onChange={(event) => update("name", event.target.value)} /></Field>
            <Field label="研究假设" wide><textarea value={draft.hypothesis} onChange={(event) => update("hypothesis", event.target.value)} /></Field>
            <Field label="失效条件" wide><textarea value={draft.invalidationCriteria} onChange={(event) => update("invalidationCriteria", event.target.value)} /></Field>
            <Field label="Market panel" wide><input className="mono" value={draft.marketPanelPath} onChange={(event) => update("marketPanelPath", event.target.value)} /></Field>
            <Field label="Labels" wide><input className="mono" value={draft.labelsPath} onChange={(event) => update("labelsPath", event.target.value)} /></Field>
            <Field label="基本面 PIT 根目录" wide><input className="mono" value={draft.fundamentalsRoot ?? ""} onChange={(event) => update("fundamentalsRoot", event.target.value)} /></Field>
            <Field label="估值 PIT" wide><input className="mono" value={draft.valuationPath ?? ""} onChange={(event) => update("valuationPath", event.target.value)} /></Field>
            <Field label="披露日 PIT" wide><input className="mono" value={draft.disclosuresPath ?? ""} onChange={(event) => update("disclosuresPath", event.target.value)} /></Field>
            <Field label="模型">
              <select value={draft.model} onChange={(event) => update("model", event.target.value as StrategyDraft["model"])}>
                <option value="ft_transformer">V8 Deep GPU Model · FT-Transformer</option><option value="ridge">V7 Classical Baseline · Ridge</option>
              </select>
            </Field>
            <Field label="因子库">
              <select value={draft.factorLibrary} onChange={(event) => update("factorLibrary", event.target.value as StrategyDraft["factorLibrary"])}>
                <option value="all_reviewed">全混合 · 全部已审查因子</option><option value="alpha181">Alpha181</option><option value="alpha101">Alpha101</option><option value="cicc_ashare80">CICC A-share 80</option><option value="basic">Basic</option>
              </select>
            </Field>
            <Field label="Horizon"><input value={draft.horizons} onChange={(event) => update("horizons", event.target.value)} /></Field>
            <Field label="主周期"><input type="number" min={1} max={252} value={draft.primaryHorizon} onChange={(event) => update("primaryHorizon", Number(event.target.value))} /></Field>
            <Field label="Top K"><input type="number" min={5} max={500} value={draft.topK} onChange={(event) => update("topK", Number(event.target.value))} /></Field>
            <Field label="Top K 候选（逗号分隔）" wide><input value={draft.topKCandidates.join(",")} onChange={(event) => update("topKCandidates", event.target.value.split(",").map(Number).filter((value) => Number.isFinite(value) && value > 0))} /></Field>
            <Field label="选股实验">
              <select value={draft.stockSelectionModes.includes("fundamental") ? "both" : "none"} onChange={(event) => update("stockSelectionModes", event.target.value === "both" ? ["none", "fundamental"] : ["none"])}>
                <option value="both">无预筛选 vs 基本面选股</option><option value="none">仅无预筛选基线</option>
              </select>
            </Field>
            <Field label="基本面入围分位"><input type="number" step="0.05" min="0.05" max="1" value={draft.fundamentalSelectionThreshold} onChange={(event) => update("fundamentalSelectionThreshold", Number(event.target.value))} /></Field>
            <Field label="因子筛选">
              <select value={draft.factorScreeningMode} onChange={(event) => update("factorScreeningMode", event.target.value as StrategyDraft["factorScreeningMode"])}><option value="pretrain">训练前实验筛选</option><option value="off">关闭（仅作对照）</option></select>
            </Field>
            <Field label="T+1 做T模式">
              <select value={draft.doTMode} onChange={(event) => update("doTMode", event.target.value as StrategyDraft["doTMode"])}><option value="daily_swing">日线波段做T</option><option value="intraday">分钟级做T（需要分钟数据）</option><option value="off">关闭对照</option></select>
            </Field>
            <Field label="分钟数据路径" wide><input className="mono" value={draft.minutePanelPath ?? ""} onChange={(event) => update("minutePanelPath", event.target.value)} disabled={draft.doTMode !== "intraday"} /></Field>
            <Field label="训练设备"><select value={draft.requireGpu ? "required" : "optional"} onChange={(event) => update("requireGpu", event.target.value === "required")}><option value="required">强制 GPU（缺失即失败）</option><option value="optional">允许 CPU（仅对照）</option></select></Field>
            <Field label="基准指数"><input value={draft.benchmarkSymbol ?? ""} onChange={(event) => update("benchmarkSymbol", event.target.value)} /></Field>
            <Field label="组合目标">
              <select value={draft.objective} onChange={(event) => update("objective", event.target.value as StrategyDraft["objective"])}>
                <option value="max_expected_alpha">最大预期 Alpha</option><option value="mean_variance">均值-方差</option><option value="min_variance">最小方差</option>
              </select>
            </Field>
            <Field label="单票上限"><input type="number" step="0.01" min="0.01" max="0.25" value={draft.maxWeightPerName} onChange={(event) => update("maxWeightPerName", Number(event.target.value))} /></Field>
            <Field label="行业上限"><input type="number" step="0.01" min="0.05" max="1" value={draft.maxSectorWeight} onChange={(event) => update("maxSectorWeight", Number(event.target.value))} /></Field>
            <Field label="换手上限"><input type="number" step="0.05" min="0.05" max="2" value={draft.maxTurnover} onChange={(event) => update("maxTurnover", Number(event.target.value))} /></Field>
            <Field label="最大回撤 Gate"><input type="number" step="0.01" min="0.01" max="0.8" value={draft.riskLimits.maxDrawdown} onChange={(event) => update("riskLimits", { ...draft.riskLimits, maxDrawdown: Number(event.target.value) })} /></Field>
            <Field label="风险换手 Gate"><input type="number" step="0.05" min="0.05" max="2" value={draft.riskLimits.maxTurnover} onChange={(event) => update("riskLimits", { ...draft.riskLimits, maxTurnover: Number(event.target.value) })} /></Field>
            <Field label="最低 Sharpe"><input type="number" step="0.1" min="-5" max="10" value={draft.riskLimits.minSharpe} onChange={(event) => update("riskLimits", { ...draft.riskLimits, minSharpe: Number(event.target.value) })} /></Field>
            <Field label="输出目录" wide><input className="mono" value={draft.outputDir} onChange={(event) => update("outputDir", event.target.value)} /></Field>
          </div>
        </WorkbenchPanel>

        <WorkbenchPanel eyebrow="OBJECTIVE FRONTIER" title="优化偏好" meta="operator input · not a performance claim" className="strategy-objective-panel">
          <EChart option={objectiveOption} className="strategy-objective-chart" />
          <ObjectiveSlider label="最大超额" value={draft.objectiveWeights.excessReturn} onChange={(value) => updateObjective("excessReturn", value)} />
          <ObjectiveSlider label="最大年化" value={draft.objectiveWeights.annualReturn} onChange={(value) => updateObjective("annualReturn", value)} />
          <ObjectiveSlider label="最小回撤" value={draft.objectiveWeights.drawdownControl} onChange={(value) => updateObjective("drawdownControl", value)} />
          <TruthNotice>三项目标通常冲突，不能承诺全部“拉满”。系统会在早期 OOS 上生成 Pareto 前沿，再用完全隔离的后段 holdout 验证候选；权重仅用于前沿内排序。</TruthNotice>
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

        <WorkbenchPanel eyebrow="RUN INSPECTOR" title="验证与启动" meta="human gated · cancellable" className="strategy-run-panel">
          <label className="strategy-version-picker">
            <span>活动策略版本</span>
            <select value={saved?.path ?? ""} onChange={(event) => {
              const items = Array.isArray(manifests.data?.data) ? manifests.data.data : [];
              const selected = items.find((item) => item.path === event.target.value);
              if (!selected) return;
              setDraft(selected.draft);
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
            <strong>{validation.valid ? "Schema 与路径校验通过" : `${validation.errors.length} 项阻塞`}</strong>
            {validation.errors.map((item) => <span key={item}>{item}</span>)}
            {validation.warnings.map((item) => <small key={item}>{item}</small>)}
          </div> : null}
          {saved ? <div className="strategy-saved"><CheckCircle weight="fill" /><span><strong>{saved.version}</strong><small>{saved.path} · {saved.contentHash?.slice(0, 12)}</small></span></div> : null}
          {error ? <div className="strategy-error" role="alert"><WarningCircle /><span>{error}</span></div> : null}
        </WorkbenchPanel>

        <WorkbenchPanel eyebrow="DECISION COUNCIL" title="多 Agent 审查" meta="structured roles · veto visible" className="strategy-council-panel">
          <div className="strategy-council">
            {(validation?.decisionCouncil ?? []).length ? validation?.decisionCouncil.map((member) => (
              <article key={member.id} className={`status-${member.status}`}>
                <span>{member.veto ? "VETO" : "REVIEW"}</span><strong>{member.label}</strong><small>{member.responsibility}</small><StatusBadge status={member.status === "approved" || member.status === "ready" ? "ready" : member.status === "blocked" ? "error" : "partial"} label={member.status} />
              </article>
            )) : <TruthNotice>先校验策略，系统再基于真实输入状态生成 Agent 审查队列；Agent 只能提交结构化建议，不能生成订单。</TruthNotice>}
          </div>
        </WorkbenchPanel>

        <WorkbenchPanel eyebrow="LIVE TELEMETRY" title="实时任务流" meta={`${stream.lines.length} bounded log lines`} className="strategy-console-panel">
          <div className="strategy-console" aria-live="polite">
            {stream.lines.length ? stream.lines.map((line, index) => <code key={`${index}-${line.slice(0, 16)}`}>{line}</code>) : <span>启动后在此显示数据构建、训练、预测、组合、回测和 paper report 的实时输出。</span>}
          </div>
          {stream.error ? <TruthNotice tone="warning">{stream.error}</TruthNotice> : null}
        </WorkbenchPanel>
      </section>
    </div>
  );
}

function Field({ label, wide = false, children }: { label: string; wide?: boolean; children: ReactNode }): JSX.Element {
  return <label className={wide ? "wide" : ""}><span>{label}</span>{children}</label>;
}

function ObjectiveSlider({ label, value, onChange }: { label: string; value: number; onChange: (value: number) => void }): JSX.Element {
  return <label className="strategy-objective-slider"><span>{label}</span><input type="range" min="0.05" max="0.90" step="0.05" value={value} onChange={(event) => onChange(Number(event.target.value))} /><strong>{Math.round(value * 100)}%</strong></label>;
}
