import { useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  ArrowClockwise,
  Atom,
  Brain,
  Broom,
  ChartLineUp,
  Check,
  CheckCircle,
  Code,
  Database,
  FlowArrow,
  FunnelSimple,
  GitBranch,
  MagnifyingGlass,
  Play,
  Scales,
  ShieldCheck,
  Sparkle,
  Stack,
  WarningCircle,
  X,
} from "@phosphor-icons/react";
import type { EChartsOption } from "echarts";
import { useNavigate, useSearchParams } from "react-router-dom";
import { apiPost } from "../api/client";
import type { Factor, JobSummary, JobValidation } from "../api/types";
import { EChart } from "../components/EChart";
import { StateView } from "../components/StateView";
import { StatusBadge } from "../components/StatusBadge";
import { mutableTemplate, type JobLaunchPayload } from "../domain/jobTemplates";
import { useApi } from "../hooks/useApi";
import { formatNumber, formatPercent } from "../utils/format";
import {
  ActionableState,
  SegmentedTabs,
  TruthNotice,
  WorkbenchHeader,
  WorkbenchMetricStrip,
  WorkbenchPanel,
  type WorkbenchTone,
} from "../vnext/workbench/InstitutionalWorkbench";

interface FactorBacktest {
  factorName: string;
  ic?: number | null;
  rankIc?: number | null;
  icir?: number | null;
  coverage?: number | null;
  stability?: number | null;
  verdict?: string | null;
  bestHorizon?: string | null;
  regimeIc: Record<string, number | null>;
  icSeries: Array<{ datetime: string; value: number }>;
  rankIcSeries: Array<{ datetime: string; value: number }>;
  decay: Array<{ horizonDays: number; ic?: number | null; rankIc?: number | null }>;
  availability: Record<string, boolean>;
}

interface FactorReview {
  id: string;
  factorName: string;
  action: "approve" | "revise" | "reject" | "deprecate";
  note: string;
  createdAt: string;
  scope: "research_review";
  registryMutation: false;
  liveExecution: false;
}

type UtilityFilter = "all" | "active" | "candidate" | "rejected" | "unevaluated";
type UtilityClass = Exclude<UtilityFilter, "all">;
type FactorView = "hypothesis" | "formula" | "validation" | "pipeline" | "correlation" | "lineage";
type PipelineState = "passed" | "failed" | "pending" | "unavailable";

const UTILITY_LABELS: Record<UtilityFilter, string> = {
  all: "全部",
  active: "已启用",
  candidate: "有效候选",
  rejected: "已剔除",
  unevaluated: "待评估",
};

const FACTOR_TABS: Array<{ id: FactorView; label: string }> = [
  { id: "hypothesis", label: "研究假设" },
  { id: "formula", label: "公式 / AST" },
  { id: "validation", label: "验证证据" },
  { id: "pipeline", label: "AI 发现链" },
  { id: "correlation", label: "正交与比较" },
  { id: "lineage", label: "血缘与使用" },
];

const PIPELINE_COPY: Record<string, { title: string; detail: string; evidence: string }> = {
  evidence: { title: "研究问题与证据", detail: "明确市场微观结构、行为或风险补偿假设，并绑定可追溯证据。", evidence: "description / source metadata" },
  formula: { title: "公式与受限 DSL", detail: "将假设压缩成可审计表达式；禁止任意代码和未来字段。", evidence: "formula / expression" },
  schema: { title: "字段契约", detail: "解析必需字段、频率、回看窗口与缺失值策略。", evidence: "requiredColumns / frequency" },
  pit: { title: "PIT 与泄漏关卡", detail: "验证字段在交易时点可获得，并保留训练边界与 embargo。", evidence: "pitSafe / train_end" },
  compute: { title: "计算与覆盖率", detail: "在真实 market panel 上计算，记录失败、缺失和有限值覆盖率。", evidence: "codeLocation / coverage" },
  ic: { title: "IC / RankIC", detail: "评估截面排序能力与稳定性，结果仅代表研究证据。", evidence: "IC / RankIC / ICIR" },
  decay: { title: "衰减与 Horizon", detail: "比较多持有期衰减，不从单一窗口推断普适结论。", evidence: "decay artifact" },
  orthogonal: { title: "相关性与正交", detail: "对现有因子库进行重复与拥挤检查，避免同质候选。", evidence: "correlation artifact" },
  regime: { title: "市场环境稳健性", detail: "检查牛市、震荡、熊市和流动性分层表现。", evidence: "regime IC" },
  backtest: { title: "组合级回测", detail: "只使用独立持久化的回测产物；不借用多因子成交冒充。", evidence: "summary / trades / signals" },
  review: { title: "人工复核", detail: "复核经济含义、证据、差异和风险；记录不自动修改注册表。", evidence: "factor review log" },
  registry: { title: "注册与下游使用", detail: "只有通过训练/风险 Gate 后才可进入 dataset、模型或选股。", evidence: "registry / feature policy" },
};

function isActiveFactor(factor: Factor): boolean {
  return Boolean(factor.usedInTraining || factor.usedInSelection || factor.usedInTiming || factor.usedInRisk);
}

function utilityClass(factor: Factor): UtilityClass {
  const lifecycle = (factor.lifecycle ?? "").toLowerCase();
  if (["reject", "drop", "remove", "deprecated", "invalid", "useless", "fail"].some((value) => lifecycle.includes(value))) return "rejected";
  if (isActiveFactor(factor) || ["approved", "accepted", "selected", "production", "ready", "useful"].some((value) => lifecycle.includes(value))) return "active";
  if (factor.sourceKind === "synthesized" || ["candidate", "promising", "pass"].some((value) => lifecycle.includes(value))) return "candidate";
  return "unevaluated";
}

function utilityStatus(factor: Factor): { label: string; status: string } {
  const classification = utilityClass(factor);
  if (isActiveFactor(factor)) return { label: "已启用", status: "ready" };
  if (classification === "candidate") return { label: "有效候选", status: "partial" };
  if (classification === "rejected") return { label: "已剔除", status: "error" };
  if (classification === "active") return { label: "已通过", status: "ready" };
  return { label: "待评估", status: "unavailable" };
}

export function FactorCenterPage(): JSX.Element {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const factors = useApi<Factor[]>(["factors"], "/factors");
  const jobs = useApi<JobSummary[]>(["global-activity-jobs"], "/jobs", undefined, { refetchInterval: 5_000, staleTime: 2_000 });
  const list = Array.isArray(factors.data?.data) ? factors.data.data : [];
  const jobList = Array.isArray(jobs.data?.data) ? jobs.data.data : [];
  const [selectedName, setSelectedName] = useState(searchParams.get("factor") ?? "");
  const [query, setQuery] = useState("");
  const [utilityFilter, setUtilityFilter] = useState<UtilityFilter>("all");
  const [activeView, setActiveView] = useState<FactorView>("pipeline");
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [selectedStage, setSelectedStage] = useState("evidence");
  const [discoveryOpen, setDiscoveryOpen] = useState(false);
  const [discoveryConfig, setDiscoveryConfig] = useState<JobLaunchPayload>(() => mutableTemplate("factor-discovery"));
  const [discoveryValidation, setDiscoveryValidation] = useState<JobValidation | null>(null);
  const [discoveryError, setDiscoveryError] = useState("");
  const [discoveryArmed, setDiscoveryArmed] = useState(false);
  const [discoveryBusy, setDiscoveryBusy] = useState(false);
  const [reviewNote, setReviewNote] = useState("");
  const [reviewBusy, setReviewBusy] = useState(false);
  const [reviewError, setReviewError] = useState("");

  useEffect(() => {
    if ((!selectedName || !list.some((item) => item.name === selectedName)) && list[0]) setSelectedName(list[0].name);
  }, [list, selectedName]);

  const detail = useApi<Factor>(["factor", selectedName], selectedName ? `/factors/${encodeURIComponent(selectedName)}` : null);
  const metricsQuery = useApi<FactorBacktest>(["factor-backtest", selectedName], selectedName ? `/factors/${encodeURIComponent(selectedName)}/backtest` : null);
  const reviewsQuery = useApi<FactorReview[]>(["factor-reviews", selectedName], selectedName ? `/factors/${encodeURIComponent(selectedName)}/reviews` : null);
  const factor = detail.data?.data;
  const metrics = metricsQuery.data?.data;
  const reviews = Array.isArray(reviewsQuery.data?.data) ? reviewsQuery.data.data : [];
  const latestReview = reviews[0];

  const counts = useMemo(() => {
    const result: Record<UtilityFilter, number> = { all: list.length, active: 0, candidate: 0, rejected: 0, unevaluated: 0 };
    for (const item of list) result[utilityClass(item)] += 1;
    return result;
  }, [list]);

  const visibleList = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return list.filter((item) => {
      if (utilityFilter !== "all" && utilityClass(item) !== utilityFilter) return false;
      return !needle || [item.name, item.displayName, item.category, item.description, item.sourceKind]
        .some((value) => String(value ?? "").toLowerCase().includes(needle));
    });
  }, [list, query, utilityFilter]);

  const selectFactor = (name: string): void => {
    setSelectedName(name);
    const next = new URLSearchParams(searchParams);
    next.set("factor", name);
    setSearchParams(next, { replace: true });
  };

  const toggleCompare = (name: string): void => {
    setCompareIds((current) => current.includes(name) ? current.filter((item) => item !== name) : current.length >= 4 ? current : [...current, name]);
  };

  const workflow = useMemo(() => buildWorkflow(factor, metrics, latestReview), [factor, latestReview, metrics]);
  const activeStage = workflow.find((stage) => stage.id === selectedStage) ?? workflow[0];
  const activeDiscoveryJobs = jobList.filter((job) => job.type === "factor-discovery" && ["queued", "running"].includes(job.status));
  const latestDiscoveryJob = jobList.find((job) => job.type === "factor-discovery");

  const icOption = useMemo<EChartsOption>(() => ({
    animationDuration: 240,
    grid: { left: 48, right: 16, top: 22, bottom: 32 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: metrics?.icSeries.map((point) => point.datetime) ?? [] },
    yAxis: { type: "value", scale: true },
    series: [
      { name: "IC", type: "line", showSymbol: true, smooth: .16, data: metrics?.icSeries.map((point) => point.value) ?? [], lineStyle: { color: "#5b9cff", width: 2 }, itemStyle: { color: "#5b9cff" } },
      { name: "Rank IC", type: "line", showSymbol: true, smooth: .16, data: metrics?.rankIcSeries.map((point) => point.value) ?? [], lineStyle: { color: "#35c39e", width: 2 }, itemStyle: { color: "#35c39e" } },
    ],
  }), [metrics?.icSeries, metrics?.rankIcSeries]);

  const decayOption = useMemo<EChartsOption>(() => ({
    animationDuration: 240,
    grid: { left: 48, right: 16, top: 22, bottom: 32 },
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: metrics?.decay.map((point) => `${point.horizonDays}D`) ?? [] },
    yAxis: { type: "value", scale: true },
    series: [{ type: "bar", data: metrics?.decay.map((point) => point.ic ?? 0) ?? [], itemStyle: { color: "#7b8cff", borderRadius: [2, 2, 0, 0] }, barMaxWidth: 28 }],
  }), [metrics?.decay]);

  const setDiscoveryParameter = (key: string, value: string | number | boolean | null): void => {
    setDiscoveryConfig((current) => ({ ...current, parameters: { ...current.parameters, [key]: value } }));
    setDiscoveryValidation(null);
    setDiscoveryArmed(false);
  };

  const validateDiscovery = async (): Promise<void> => {
    setDiscoveryBusy(true);
    setDiscoveryError("");
    try {
      const result = await apiPost<JobValidation>("/jobs/factor-discovery/validate", discoveryConfig);
      setDiscoveryValidation(result.data);
    } catch (error) {
      setDiscoveryValidation(null);
      setDiscoveryError(error instanceof Error ? error.message : "Factor discovery validation failed");
    } finally {
      setDiscoveryBusy(false);
    }
  };

  const launchDiscovery = async (): Promise<void> => {
    if (!discoveryValidation?.valid || !discoveryArmed) return;
    setDiscoveryBusy(true);
    setDiscoveryError("");
    try {
      await apiPost<JobSummary>("/jobs/factor-discovery", discoveryConfig);
      setDiscoveryArmed(false);
      await queryClient.invalidateQueries({ queryKey: ["global-activity-jobs"] });
    } catch (error) {
      setDiscoveryError(error instanceof Error ? error.message : "Factor discovery launch failed");
    } finally {
      setDiscoveryBusy(false);
    }
  };

  const submitReview = async (action: FactorReview["action"]): Promise<void> => {
    if (!selectedName || reviewNote.trim().length < 3) {
      setReviewError("请先填写至少 3 个字符的复核说明。");
      return;
    }
    setReviewBusy(true);
    setReviewError("");
    try {
      await apiPost<FactorReview>(`/factors/${encodeURIComponent(selectedName)}/reviews`, { action, note: reviewNote.trim() });
      setReviewNote("");
      await queryClient.invalidateQueries({ queryKey: ["factor-reviews", selectedName] });
    } catch (error) {
      setReviewError(error instanceof Error ? error.message : "Review write failed");
    } finally {
      setReviewBusy(false);
    }
  };

  const metricStrip = [
    { label: "注册因子", value: String(list.length), detail: `${counts.active} downstream active`, tone: "info" as WorkbenchTone, icon: Database },
    { label: "AI / GP 候选", value: String(list.filter((item) => item.sourceKind === "synthesized").length), detail: "synthesized definitions", tone: "ai" as WorkbenchTone, icon: Sparkle },
    { label: "PIT 已声明", value: String(list.filter((item) => item.pitSafe === true).length), detail: `${list.filter((item) => item.pitSafe !== true).length} need review`, tone: "positive" as WorkbenchTone, icon: ShieldCheck },
    { label: "有效候选", value: String(counts.candidate), detail: "not auto-promoted", tone: "warning" as WorkbenchTone, icon: FunnelSimple },
    { label: "已剔除", value: String(counts.rejected), detail: "excluded by lifecycle", tone: counts.rejected ? "danger" as WorkbenchTone : "neutral" as WorkbenchTone, icon: X },
    { label: "发现任务", value: String(activeDiscoveryJobs.length), detail: latestDiscoveryJob?.status ?? "no active run", tone: activeDiscoveryJobs.length ? "ai" as WorkbenchTone : "neutral" as WorkbenchTone, icon: Brain },
  ];

  return (
    <div className="institutional-workbench factor-intelligence-page">
      <WorkbenchHeader
        eyebrow="FACTOR INTELLIGENCE STUDIO / GOVERNED DISCOVERY"
        title="因子智能实验室"
        description="证据 → 假设 → 公式 → PIT → 评估 → 人工复核 → 注册；训练与实盘始终由独立 Gate 控制。"
        asOf={new Intl.DateTimeFormat("en-CA", { timeZone: "Asia/Shanghai" }).format(new Date())}
        context="research / paper only"
        actions={<>
          <button type="button" onClick={() => navigate("/runtime?view=cleanup")}><Broom size={14} />产物治理</button>
          <button type="button" className="primary" onClick={() => setDiscoveryOpen(true)}><Sparkle size={14} />新建发现实验</button>
        </>}
      />
      <WorkbenchMetricStrip metrics={metricStrip} />

      <div className="factor-studio-layout">
        <WorkbenchPanel eyebrow="EXPERIMENT NAVIGATOR" title="因子目录" meta={`${visibleList.length} / ${list.length}`} className="factor-navigator">
          <div className="factor-nav-search"><MagnifyingGlass size={15} /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="名称、类别、来源、假设" aria-label="搜索因子名称 / 类别" /></div>
          <div className="factor-utility-filters" aria-label="因子有效性筛选">
            {(Object.keys(UTILITY_LABELS) as UtilityFilter[]).map((item) => (
              <button key={item} type="button" className={utilityFilter === item ? "active" : ""} onClick={() => setUtilityFilter(item)}>
                <span>{UTILITY_LABELS[item]}</span><strong>{counts[item]}</strong>
              </button>
            ))}
          </div>
          <div className="factor-nav-legend"><span><i className="legend-ai" />AI candidate</span><span><i className="legend-active" />downstream active</span></div>
          <div className="factor-nav-list">
            {visibleList.map((item) => {
              const utility = utilityStatus(item);
              const compared = compareIds.includes(item.name);
              return (
                <div className={`factor-nav-row ${item.name === selectedName ? "active" : ""}`} key={item.name}>
                  <button type="button" className="factor-nav-main" onClick={() => selectFactor(item.name)}>
                    <span className={`factor-source source-${item.sourceKind}`}><Atom size={15} /></span>
                    <span><strong>{item.displayName ?? item.name}</strong><small>{item.category ?? item.sourceKind} · {item.frequency ?? "unknown"}</small></span>
                    <StatusBadge status={utility.status} label={utility.label} />
                  </button>
                  <button type="button" className={`factor-compare-toggle ${compared ? "active" : ""}`} aria-label={`${compared ? "移出" : "加入"}比较 ${item.name}`} aria-pressed={compared} disabled={!compared && compareIds.length >= 4} onClick={() => toggleCompare(item.name)}>{compared ? <Check size={12} weight="bold" /> : <Scales size={12} />}</button>
                </div>
              );
            })}
            {!visibleList.length ? <ActionableState compact title={factors.isLoading ? "正在恢复目录" : "当前筛选没有因子"} detail={factors.isError ? "Factor API 读取失败，请检查统一服务日志。" : "可以调整筛选，或从真实 market panel 启动受治理的发现实验。"} icon={factors.isError ? WarningCircle : Atom} tone={factors.isError ? "danger" : "neutral"} primary={{ label: "新建发现实验", onClick: () => setDiscoveryOpen(true) }} secondary={{ label: "查看数据", onClick: () => navigate("/runtime?view=data") }} /> : null}
          </div>
          <TruthNotice tone="warning">已剔除因子不会因为页面存在而自动进入训练；dataset、feature policy、PIT 和风险 Gate 仍是唯一入口。</TruthNotice>
        </WorkbenchPanel>

        <main className="factor-primary-column">
          <WorkbenchPanel
            eyebrow={factor?.sourceKind === "synthesized" ? "AI / GP CANDIDATE" : "REGISTERED FACTOR"}
            title={factor?.displayName ?? factor?.name ?? "Discovery cockpit"}
            meta={factor ? `${factor.category ?? "unclassified"} · ${factor.frequency ?? "frequency undeclared"}` : "select a factor or start a governed discovery run"}
            className="factor-primary-canvas"
            actions={factor ? <><StatusBadge status={utilityStatus(factor).status} label={utilityStatus(factor).label} /><button type="button" onClick={() => toggleCompare(factor.name)}><Scales size={13} />{compareIds.includes(factor.name) ? "移出比较" : "加入比较"}</button></> : undefined}
          >
            <div className="factor-canvas-tabs"><SegmentedTabs items={FACTOR_TABS} active={activeView} onChange={setActiveView} label="因子研究视图" /></div>
            {!factor && !detail.isLoading ? <EmptyDiscoveryCanvas onOpen={() => setDiscoveryOpen(true)} onData={() => navigate("/runtime?view=data")} workflow={workflow} selectedStage={selectedStage} setSelectedStage={setSelectedStage} /> : null}
            {detail.isLoading ? <StateView state="loading" detail="正在恢复因子定义与来源。" /> : null}
            {factor && activeView === "hypothesis" ? <HypothesisView factor={factor} metrics={metrics} /> : null}
            {factor && activeView === "formula" ? <FormulaView factor={factor} /> : null}
            {factor && activeView === "validation" ? <ValidationView metrics={metrics} loading={metricsQuery.isLoading} icOption={icOption} decayOption={decayOption} /> : null}
            {factor && activeView === "pipeline" ? <PipelineView workflow={workflow} selectedStage={selectedStage} setSelectedStage={setSelectedStage} /> : null}
            {factor && activeView === "correlation" ? <ComparisonView factor={factor} factors={list.filter((item) => compareIds.includes(item.name))} onOpenDiscovery={() => setDiscoveryOpen(true)} /> : null}
            {factor && activeView === "lineage" ? <LineageView factor={factor} /> : null}
          </WorkbenchPanel>

          {compareIds.length ? <CompareBasket factors={list.filter((item) => compareIds.includes(item.name))} remove={toggleCompare} clear={() => setCompareIds([])} /> : null}
        </main>

        <WorkbenchPanel eyebrow="EVIDENCE INSPECTOR" title="研究复核" meta={factor?.name ?? "no selection"} className="factor-inspector">
          {factor ? <>
            <section className="factor-inspector-stage">
              <div className={`pipeline-state state-${activeStage.state}`}>{pipelineStateIcon(activeStage.state)}</div>
              <div><span>SELECTED GATE · {activeStage.group}</span><strong>{activeStage.title}</strong><p>{activeStage.detail}</p><code>{activeStage.evidence}</code></div>
            </section>
            <dl className="factor-inspector-kv">
              <div><dt>Lifecycle</dt><dd>{factor.lifecycle ?? "unclassified"}</dd></div>
              <div><dt>PIT declaration</dt><dd>{factor.pitSafe === true ? "passed" : factor.pitSafe === false ? "failed" : "not declared"}</dd></div>
              <div><dt>Source</dt><dd>{factor.sourceKind}</dd></div>
              <div><dt>Code</dt><dd>{factor.codeLocation ?? "unavailable"}</dd></div>
              <div><dt>Review log</dt><dd>{reviews.length} events</dd></div>
              <div><dt>Registry mutation</dt><dd>separate gate</dd></div>
            </dl>
            <section className="factor-usage-matrix">
              <h3>DOWNSTREAM USAGE</h3>
              <UsageFlag label="训练" active={factor.usedInTraining} />
              <UsageFlag label="选股" active={factor.usedInSelection} />
              <UsageFlag label="择时" active={factor.usedInTiming} />
              <UsageFlag label="风控" active={factor.usedInRisk} />
            </section>
            <section className="factor-review-form">
              <h3>HUMAN REVIEW</h3>
              {latestReview ? <div className={`latest-review review-${latestReview.action}`}><span>{latestReview.action.toUpperCase()} · {latestReview.createdAt.slice(0, 16)}</span><p>{latestReview.note}</p></div> : null}
              <textarea value={reviewNote} onChange={(event) => setReviewNote(event.target.value)} placeholder="记录证据、差异、风险与复核结论…" aria-label="人工复核说明" />
              <div className="factor-review-actions">
                <button type="button" disabled={reviewBusy} onClick={() => void submitReview("approve")}><CheckCircle size={13} />研究通过</button>
                <button type="button" disabled={reviewBusy} onClick={() => void submitReview("revise")}><ArrowClockwise size={13} />要求修订</button>
                <button type="button" disabled={reviewBusy} onClick={() => void submitReview("reject")}><X size={13} />驳回</button>
              </div>
              {reviewError ? <p className="factor-form-error">{reviewError}</p> : null}
              <TruthNotice>复核记录追加写入 runtime governance；不会自动注册、训练、下单或开启实盘。</TruthNotice>
            </section>
          </> : <ActionableState title="选择一个因子" detail="检查假设、公式、PIT、验证证据与下游使用，再进行人工复核。" icon={ShieldCheck} />}
        </WorkbenchPanel>
      </div>

      {discoveryOpen ? <DiscoveryDrawer
        config={discoveryConfig}
        setParameter={setDiscoveryParameter}
        validation={discoveryValidation}
        error={discoveryError}
        busy={discoveryBusy}
        armed={discoveryArmed}
        setArmed={setDiscoveryArmed}
        validate={() => void validateDiscovery()}
        launch={() => void launchDiscovery()}
        close={() => setDiscoveryOpen(false)}
        activeJob={latestDiscoveryJob}
      /> : null}
    </div>
  );
}

function EmptyDiscoveryCanvas({ onOpen, onData, workflow, selectedStage, setSelectedStage }: { onOpen: () => void; onData: () => void; workflow: PipelineStage[]; selectedStage: string; setSelectedStage: (value: string) => void }): JSX.Element {
  return <div className="empty-discovery-canvas"><ActionableState title="暂无可用数据" detail="只展示已持久化的真实 QuantAgent artifact。因子发现工作台仍可用：选择真实 market panel，先做受限 DSL / PIT / 相关性验证；只有显式开启网络并二次确认时才调用 LLM。" icon={Sparkle} tone="ai" primary={{ label: "配置发现实验", onClick: onOpen }} secondary={{ label: "检查数据覆盖", onClick: onData }} /><PipelineView workflow={workflow} selectedStage={selectedStage} setSelectedStage={setSelectedStage} compact /></div>;
}

function HypothesisView({ factor, metrics }: { factor: Factor; metrics?: FactorBacktest }): JSX.Element {
  return <div className="factor-hypothesis-view">
    <section className="factor-hypothesis-copy"><span>ECONOMIC HYPOTHESIS</span><h3>{factor.description ?? "尚未持久化人工研究假设。"}</h3><p>方向：{factor.direction} · Horizon：{factor.horizonDays ? `${factor.horizonDays}D` : "未声明"} · 频率：{factor.frequency ?? "未声明"}</p></section>
    <section className="factor-equation"><Code size={18} /><code>{factor.formula ?? "公式不可提取；请从代码位置与输入字段继续审计。"}</code></section>
    <div className="factor-evidence-grid">
      <EvidenceCard label="输入字段" value={factor.requiredColumns.join(", ") || "未声明"} state={factor.requiredColumns.length ? "passed" : "pending"} />
      <EvidenceCard label="PIT" value={factor.pitSafe === true ? "通过声明" : factor.pitSafe === false ? "未通过" : "待审计"} state={factor.pitSafe === true ? "passed" : factor.pitSafe === false ? "failed" : "pending"} />
      <EvidenceCard label="验证结论" value={metrics?.verdict ?? factor.lifecycle ?? "待评估"} state={metrics?.verdict ? "passed" : "pending"} />
      <EvidenceCard label="来源" value={factor.sourceKind} state={factor.sourceKind === "synthesized" ? "pending" : "passed"} />
    </div>
  </div>;
}

function FormulaView({ factor }: { factor: Factor }): JSX.Element {
  const tokens = factor.formula?.match(/[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[-+*/(),]/g) ?? [];
  return <div className="factor-formula-view">
    <section className="factor-formula-code"><header><span>SAFE EXPRESSION / SOURCE</span><small>{factor.codeLocation ?? "source unavailable"}</small></header><pre>{factor.formula ?? "UNAVAILABLE"}</pre></section>
    <section className="factor-ast"><header><span>PARSED TOKENS</span><small>{tokens.length} tokens · presentation only</small></header>{tokens.length ? <div>{tokens.slice(0, 90).map((token, index) => <span className={/^[A-Za-z_]/.test(token) ? "token-name" : /^\d/.test(token) ? "token-number" : "token-operator"} key={`${token}-${index}`}>{token}</span>)}</div> : <ActionableState compact title="没有可提取公式" detail="查看代码位置，或让发现任务输出 synthesized_definitions.json。" icon={Code} />}</section>
    <TruthNotice tone="ai">AST 视图只解释持久化表达式；浏览器不会执行任意因子代码。</TruthNotice>
  </div>;
}

function ValidationView({ metrics, loading, icOption, decayOption }: { metrics?: FactorBacktest; loading: boolean; icOption: EChartsOption; decayOption: EChartsOption }): JSX.Element {
  if (loading) return <StateView state="loading" detail="正在读取独立因子评估 artifact。" />;
  return <div className="factor-validation-view">
    <WorkbenchMetricStrip metrics={[
      { label: "IC", value: formatNumber(metrics?.ic, 4), detail: "cross-sectional", tone: toneFromMetric(metrics?.ic) },
      { label: "Rank IC", value: formatNumber(metrics?.rankIc, 4), detail: "rank correlation", tone: toneFromMetric(metrics?.rankIc) },
      { label: "ICIR", value: formatNumber(metrics?.icir), detail: "stability", tone: toneFromMetric(metrics?.icir) },
      { label: "覆盖率", value: formatPercent(metrics?.coverage), detail: "finite / tradable", tone: metrics?.coverage ? "positive" : "neutral" },
      { label: "稳定性", value: formatNumber(metrics?.stability), detail: "periods passed", tone: metrics?.stability ? "positive" : "neutral" },
      { label: "最佳 Horizon", value: metrics?.bestHorizon ?? "暂无", detail: metrics?.verdict ?? "no verdict", tone: metrics?.verdict ? "info" : "neutral" },
    ]} />
    <div className="factor-validation-grid">
      <WorkbenchPanel eyebrow="SOURCE-BACKED SERIES" title="IC / RankIC 时间序列">{metrics?.icSeries.length || metrics?.rankIcSeries.length ? <EChart option={icOption} className="factor-chart" /> : <ActionableState title="没有独立 IC 序列" detail="先运行 factor judgment / discovery evaluation，再从对应 artifact 读取。" icon={ChartLineUp} />}</WorkbenchPanel>
      <WorkbenchPanel eyebrow="HORIZON DECAY" title="衰减曲线">{metrics?.decay.length ? <EChart option={decayOption} className="factor-chart" /> : <ActionableState title="没有衰减评估" detail="至少需要两个持有期的 source-backed IC 指标。" icon={ChartLineUp} />}</WorkbenchPanel>
      <WorkbenchPanel eyebrow="REGIME ROBUSTNESS" title="市场环境稳健性"><div className="regime-evidence">{Object.entries(metrics?.regimeIc ?? {}).map(([name, value]) => <EvidenceCard key={name} label={name} value={formatNumber(value, 4)} state={value === null ? "unavailable" : (value ?? 0) >= 0 ? "passed" : "failed"} />)}</div></WorkbenchPanel>
      <WorkbenchPanel eyebrow="ARTIFACT CONTRACT" title="独立验证能力"><div className="factor-availability">{Object.entries(metrics?.availability ?? {}).map(([name, available]) => <div key={name}><span>{name}</span><StatusBadge status={available ? "ready" : "unavailable"} label={available ? "可用" : "缺失"} /></div>)}</div><TruthNotice tone="warning">没有独立 trade / signal artifact 时，不复用多因子买卖点。</TruthNotice></WorkbenchPanel>
    </div>
  </div>;
}

interface PipelineStage { id: string; title: string; detail: string; evidence: string; group: string; state: PipelineState }

function buildWorkflow(factor?: Factor, metrics?: FactorBacktest, review?: FactorReview): PipelineStage[] {
  const hasRegime = Object.values(metrics?.regimeIc ?? {}).some((value) => value !== null && value !== undefined);
  const definitions: Array<[string, string, PipelineState]> = [
    ["evidence", "PROPOSE", factor?.description ? "passed" : "pending"],
    ["formula", "IMPLEMENT", factor?.formula ? "passed" : "pending"],
    ["schema", "IMPLEMENT", factor?.requiredColumns.length ? "passed" : "pending"],
    ["pit", "VALIDATE", factor?.pitSafe === true ? "passed" : factor?.pitSafe === false ? "failed" : "pending"],
    ["compute", "VALIDATE", factor?.codeLocation || factor?.sourceKind === "synthesized" ? "passed" : "pending"],
    ["ic", "EVALUATE", metrics?.ic !== null && metrics?.ic !== undefined ? "passed" : "unavailable"],
    ["decay", "EVALUATE", metrics?.decay.length ? "passed" : "unavailable"],
    ["orthogonal", "EVALUATE", "unavailable"],
    ["regime", "EVALUATE", hasRegime ? "passed" : "unavailable"],
    ["backtest", "IMPACT", metrics?.availability?.summaryMetrics ? "passed" : "unavailable"],
    ["review", "HUMAN GATE", review ? (review.action === "approve" ? "passed" : review.action === "reject" ? "failed" : "pending") : "pending"],
    ["registry", "PROMOTE", factor && isActiveFactor(factor) ? "passed" : "pending"],
  ];
  return definitions.map(([id, group, state]) => ({ id, group, state, ...PIPELINE_COPY[id] }));
}

function PipelineView({ workflow, selectedStage, setSelectedStage, compact = false }: { workflow: PipelineStage[]; selectedStage: string; setSelectedStage: (value: string) => void; compact?: boolean }): JSX.Element {
  const selected = workflow.find((stage) => stage.id === selectedStage) ?? workflow[0];
  return <div className={`factor-pipeline-view ${compact ? "compact" : ""}`}>
    <div className="factor-pipeline-grid" role="list" aria-label="因子发现与治理流程">
      {workflow.map((stage, index) => <button type="button" key={stage.id} role="listitem" className={`${selectedStage === stage.id ? "active" : ""} state-${stage.state}`} onClick={() => setSelectedStage(stage.id)}><span>{String(index + 1).padStart(2, "0")} · {stage.group}</span><strong>{stage.title}</strong><small>{stage.evidence}</small><i>{pipelineStateLabel(stage.state)}</i></button>)}
    </div>
    {!compact ? <section className="factor-pipeline-detail"><div className={`pipeline-state state-${selected.state}`}>{pipelineStateIcon(selected.state)}</div><div><span>{selected.group} · {pipelineStateLabel(selected.state)}</span><h3>{selected.title}</h3><p>{selected.detail}</p><code>Evidence: {selected.evidence}</code></div></section> : null}
  </div>;
}

function ComparisonView({ factor, factors, onOpenDiscovery }: { factor: Factor; factors: Factor[]; onOpenDiscovery: () => void }): JSX.Element {
  const rows = factors.length ? factors : [factor];
  return <div className="factor-comparison-view">
    <TruthNotice tone="ai">比较篮最多 4 个候选。当前 API 未持久化 pairwise correlation 时，界面明确显示缺口，不生成估算值。</TruthNotice>
    <div className="factor-compare-table"><table><thead><tr><th>因子</th><th>来源</th><th>PIT</th><th>频率</th><th>Horizon</th><th>下游使用</th><th>相关性</th></tr></thead><tbody>{rows.map((item) => <tr key={item.name}><td><strong>{item.displayName ?? item.name}</strong><small>{item.category ?? "unclassified"}</small></td><td>{item.sourceKind}</td><td>{item.pitSafe === true ? "通过" : item.pitSafe === false ? "失败" : "待审计"}</td><td>{item.frequency ?? "—"}</td><td>{item.horizonDays ? `${item.horizonDays}D` : "—"}</td><td>{[item.usedInTraining && "train", item.usedInSelection && "selection", item.usedInRisk && "risk"].filter(Boolean).join(" · ") || "none"}</td><td>artifact 缺失</td></tr>)}</tbody></table></div>
    <ActionableState compact title="相关性矩阵尚未持久化" detail="运行发现评估会检查 max_reference_correlation / max_sota_correlation；结果写入 artifact 后再可视化。" icon={Scales} primary={{ label: "配置发现评估", onClick: onOpenDiscovery }} />
  </div>;
}

function LineageView({ factor }: { factor: Factor }): JSX.Element {
  const outputs = [factor.usedInTraining && "Training dataset", factor.usedInSelection && "Selection", factor.usedInTiming && "Timing", factor.usedInRisk && "Risk"].filter(Boolean) as string[];
  return <div className="factor-lineage-view">
    <section><header>UPSTREAM FIELDS</header>{factor.requiredColumns.length ? factor.requiredColumns.map((item) => <div key={item}><Database size={14} /><span>{item}</span></div>) : <p>没有声明输入字段</p>}</section>
    <FlowArrow size={24} />
    <section className="lineage-focus"><header>FACTOR</header><div><Atom size={18} /><span><strong>{factor.name}</strong><small>{factor.codeLocation ?? factor.sourceKind}</small></span></div></section>
    <FlowArrow size={24} />
    <section><header>DOWNSTREAM</header>{outputs.length ? outputs.map((item) => <div key={item}><Stack size={14} /><span>{item}</span></div>) : <p>尚未进入下游 pipeline</p>}</section>
  </div>;
}

function CompareBasket({ factors, remove, clear }: { factors: Factor[]; remove: (name: string) => void; clear: () => void }): JSX.Element {
  return <section className="factor-compare-basket"><header><div><Scales size={16} /><span><strong>候选比较篮</strong><small>{factors.length}/4 · 只比较，不改变活动因子</small></span></div><button type="button" onClick={clear}>清空</button></header><div>{factors.map((item) => <button type="button" key={item.name} onClick={() => remove(item.name)}>{item.displayName ?? item.name}<X size={11} /></button>)}{factors.length < 2 ? <span>再加入一个因子即可比较</span> : <strong>已就绪：打开“正交与比较”</strong>}</div></section>;
}

function DiscoveryDrawer({ config, setParameter, validation, error, busy, armed, setArmed, validate, launch, close, activeJob }: { config: JobLaunchPayload; setParameter: (key: string, value: string | number | boolean | null) => void; validation: JobValidation | null; error: string; busy: boolean; armed: boolean; setArmed: (value: boolean) => void; validate: () => void; launch: () => void; close: () => void; activeJob?: JobSummary }): JSX.Element {
  const params = config.parameters;
  const useLlm = Boolean(params.use_llm);
  return <div className="factor-discovery-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) close(); }}><aside className="factor-discovery-drawer" role="dialog" aria-modal="true" aria-label="因子发现实验配置">
    <header><div><span>GOVERNED AI FACTOR DISCOVERY</span><h2>新建因子发现实验</h2><p>调用现有 synthesize-factors-v7 / RD-Agent 流程，不创建第二套训练系统。</p></div><button type="button" aria-label="关闭因子发现配置" onClick={close}><X size={16} /></button></header>
    {activeJob && ["queued", "running"].includes(activeJob.status) ? <div className="factor-active-job"><Brain size={17} /><span><strong>{activeJob.commandId}</strong><small>{activeJob.id} · {activeJob.status} · {activeJob.progress == null ? "progress unavailable" : `${Math.round(activeJob.progress * 100)}%`}</small></span></div> : null}
    <section className="factor-discovery-form">
      <label className="wide"><span>Market panel</span><input value={String(params.market_panel_path ?? "")} onChange={(event) => setParameter("market_panel_path", event.target.value)} /></label>
      <label className="wide"><span>Labels（可选）</span><input value={String(params.labels_path ?? "")} onChange={(event) => setParameter("labels_path", event.target.value || null)} /></label>
      <label className="wide"><span>Output directory</span><input value={String(params.output_dir ?? "")} onChange={(event) => setParameter("output_dir", event.target.value)} /></label>
      <label><span>Label</span><input value={String(params.label_column ?? "forward_return_5d")} onChange={(event) => setParameter("label_column", event.target.value)} /></label>
      <label><span>Rounds</span><input type="number" min={1} max={20} value={Number(params.rounds ?? 4)} onChange={(event) => setParameter("rounds", Number(event.target.value))} /></label>
      <label><span>Factors / round</span><input type="number" min={1} max={5} value={Number(params.factors_per_round ?? 3)} onChange={(event) => setParameter("factors_per_round", Number(event.target.value))} /></label>
      <label><span>Top K</span><input type="number" min={1} max={100} value={Number(params.top_k ?? 20)} onChange={(event) => setParameter("top_k", Number(event.target.value))} /></label>
      <label><span>Reference corr.</span><input type="number" step="0.05" min={0} max={1} value={Number(params.max_reference_correlation ?? .7)} onChange={(event) => setParameter("max_reference_correlation", Number(event.target.value))} /></label>
      <label><span>Train cutoff</span><input value={String(params.train_end ?? "")} placeholder="YYYY-MM-DD" onChange={(event) => setParameter("train_end", event.target.value)} /></label>
    </section>
    <section className="factor-discovery-controls">
      <label><input type="checkbox" checked={Boolean(params.exclude_st)} onChange={(event) => setParameter("exclude_st", event.target.checked)} /><span><strong>排除 ST</strong><small>tradability guard</small></span></label>
      <label className={useLlm ? "active" : ""}><input type="checkbox" checked={useLlm} onChange={(event) => { setParameter("use_llm", event.target.checked); if (!event.target.checked) setParameter("allow_network", false); }} /><span><strong>启用 LLM 提案</strong><small>仍受 DSL / PIT / correlation gate</small></span></label>
      <label className={Boolean(params.allow_network) ? "danger" : ""}><input type="checkbox" checked={Boolean(params.allow_network)} disabled={!useLlm} onChange={(event) => setParameter("allow_network", event.target.checked)} /><span><strong>允许网络模型调用</strong><small>use_llm=true 时必须显式确认</small></span></label>
    </section>
    <section className="factor-discovery-flow"><span><Database size={14} />Market panel</span><FlowArrow size={14} /><span><Brain size={14} />{useLlm ? "LLM + blueprints" : "Blueprint + GP"}</span><FlowArrow size={14} /><span><ShieldCheck size={14} />PIT / IC / corr.</span><FlowArrow size={14} /><span><GitBranch size={14} />Candidate artifact</span></section>
    {validation ? <div className="factor-validation-result"><CheckCircle size={17} /><span><strong>后端验证通过</strong><small>{validation.entrypoint} · {validation.outputPaths.join(", ")}</small>{validation.warnings.map((warning) => <em key={warning}>{warning}</em>)}</span></div> : null}
    {error ? <div className="factor-discovery-error"><WarningCircle size={17} /><span>{error}</span></div> : null}
    <label className="factor-arm"><input type="checkbox" checked={armed} disabled={!validation?.valid} onChange={(event) => setArmed(event.target.checked)} /><span><strong>Arm research launch</strong><small>仅生成研究候选；不会注册、训练或下单。</small></span></label>
    <footer><button type="button" onClick={close}>取消</button><button type="button" onClick={validate} disabled={busy}><ShieldCheck size={14} />{busy ? "校验中…" : "验证配置"}</button><button type="button" className="primary" onClick={launch} disabled={busy || !validation?.valid || !armed}><Play size={14} weight="fill" />启动发现任务</button></footer>
  </aside></div>;
}

function EvidenceCard({ label, value, state }: { label: string; value: string; state: PipelineState }): JSX.Element {
  return <div className={`factor-evidence-card state-${state}`}><span>{label}</span><strong>{value}</strong><small>{pipelineStateLabel(state)}</small></div>;
}

function UsageFlag({ label, active }: { label: string; active?: boolean | null }): JSX.Element {
  return <div className={active ? "active" : ""}><i /> <span>{label}</span><strong>{active ? "YES" : "NO"}</strong></div>;
}

function pipelineStateLabel(state: PipelineState): string {
  return state === "passed" ? "PASSED" : state === "failed" ? "FAILED" : state === "pending" ? "PENDING" : "NO ARTIFACT";
}

function pipelineStateIcon(state: PipelineState): JSX.Element {
  return state === "passed" ? <CheckCircle size={18} weight="fill" /> : state === "failed" ? <X size={18} /> : state === "pending" ? <ArrowClockwise size={18} /> : <WarningCircle size={18} />;
}

function toneFromMetric(value?: number | null): WorkbenchTone {
  if (value === null || value === undefined) return "neutral";
  return value > 0 ? "positive" : value < 0 ? "danger" : "warning";
}
