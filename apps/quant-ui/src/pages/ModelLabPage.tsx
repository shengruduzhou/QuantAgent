import { useDeferredValue, useEffect, useMemo, useState } from "react";
import {
  ArrowsLeftRight,
  Brain,
  ChartScatter,
  Check,
  Cpu,
  Database,
  FileCode,
  FunnelSimple,
  HardDrives,
  MagnifyingGlass,
  ShieldCheck,
  Stack,
  WarningCircle,
} from "@phosphor-icons/react";
import type { Icon } from "@phosphor-icons/react";
import type { EChartsOption } from "echarts";
import { useNavigate, useSearchParams } from "react-router-dom";
import type {
  ModelComparison,
  ModelMetric,
  ModelObservability,
  ModelSummary,
} from "../api/types";
import { useApi } from "../hooks/useApi";
import { EChart } from "../components/EChart";
import { MetricCard } from "../components/MetricCard";
import { Panel } from "../components/Panel";
import { StateView } from "../components/StateView";
import { StatusBadge } from "../components/StatusBadge";
import { formatBytes, formatCompact, formatDate, formatNumber, formatPercent } from "../utils/format";
import { ActionableState, WorkbenchHeader, WorkbenchMetricStrip } from "../vnext/workbench/InstitutionalWorkbench";

interface TrainingPoint {
  epoch: number;
  loss?: number | null;
  validationLoss?: number | null;
  metrics: Record<string, number>;
}

interface FeatureImportance {
  feature: string;
  importance: number;
  method: string;
}

interface Prediction {
  datetime: string;
  symbol: string;
  score: number;
  horizon?: string | null;
  actualReturn?: number | null;
  rank?: number | null;
}

const familyLabels: Record<string, string> = {
  all: "全部模型",
  deep_alpha: "Deep Alpha",
  registered_alpha: "Registered Alpha",
  reinforcement_learning: "RL Policy",
  intraday_t_plus_one: "T+1 Models",
  generic_artifact: "Other Artifacts",
};

type ModelTab = "overview" | "metrics" | "artifacts" | "config";

const modelTabs: Array<{ key: ModelTab; label: string; icon: Icon }> = [
  { key: "overview", label: "可观测总览", icon: ChartScatter },
  { key: "metrics", label: "全部指标", icon: ShieldCheck },
  { key: "artifacts", label: "关联产物", icon: Database },
  { key: "config", label: "配置与 Schema", icon: FileCode },
];

export function ModelLabPage(): JSX.Element {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const models = useApi<ModelSummary[]>(["models"], "/models");
  const [modelId, setModelId] = useState(searchParams.get("modelId") ?? "");
  const [family, setFamily] = useState("all");
  const [query, setQuery] = useState("");
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<ModelTab>("overview");
  const deferredQuery = useDeferredValue(query);
  const list = models.data?.data ?? [];

  const visibleModels = useMemo(() => {
    const needle = deferredQuery.trim().toLowerCase();
    return list.filter((model) => {
      const familyMatches = family === "all" || model.modelFamily === family;
      const queryMatches = !needle || [
        model.version,
        model.modelType,
        model.modelFamily,
        model.verdict,
        model.path,
      ].some((value) => String(value ?? "").toLowerCase().includes(needle));
      return familyMatches && queryMatches;
    });
  }, [deferredQuery, family, list]);

  useEffect(() => {
    const preferred = list.find((model) => model.modelFamily === "deep_alpha")
      ?? list.find((model) => model.modelFamily !== "generic_artifact")
      ?? list[0];
    if ((!modelId || !list.some((model) => model.id === modelId)) && preferred) {
      setModelId(preferred.id);
    }
  }, [list, modelId]);

  useEffect(() => {
    if (modelId && visibleModels.length && !visibleModels.some((model) => model.id === modelId)) {
      setModelId(visibleModels[0].id);
    }
  }, [modelId, visibleModels]);

  const detail = useApi<ModelObservability>(
    ["model-observability", modelId],
    modelId ? `/models/${modelId}/observability` : null,
  );
  const training = useApi<TrainingPoint[]>(
    ["model-training", modelId],
    modelId ? `/models/${modelId}/training-metrics` : null,
  );
  const importance = useApi<FeatureImportance[]>(
    ["model-importance", modelId],
    modelId ? `/models/${modelId}/feature-importance` : null,
  );
  const predictions = useApi<Prediction[]>(
    ["model-predictions", modelId],
    modelId ? `/models/${modelId}/predictions` : null,
    { limit: 3_000 },
  );
  const comparison = useApi<ModelComparison>(
    ["model-comparison", compareIds],
    compareIds.length >= 2 ? "/models/compare" : null,
    { ids: compareIds.join(",") },
  );
  const selected = list.find((model) => model.id === modelId);
  const observed = detail.data?.data;
  const selectModel = (id: string): void => {
    setModelId(id);
    const next = new URLSearchParams(searchParams);
    next.set("modelId", id);
    setSearchParams(next, { replace: true });
  };

  const familyCounts = useMemo(() => {
    const counts: Record<string, number> = { all: list.length };
    for (const model of list) {
      const key = model.modelFamily ?? "generic_artifact";
      counts[key] = (counts[key] ?? 0) + 1;
    }
    return counts;
  }, [list]);

  const lossOption = useMemo<EChartsOption>(() => ({
    animationDuration: 280,
    grid: { left: 52, right: 18, top: 32, bottom: 34 },
    tooltip: { trigger: "axis", backgroundColor: "#071521", borderColor: "#24506b", textStyle: { color: "#e2edf5" } },
    legend: { data: ["Train Loss", "Validation Loss"], textStyle: { color: "#8da3b7" }, top: 2 },
    xAxis: { type: "category", data: training.data?.data.map((point) => point.epoch) ?? [], axisLabel: { color: "#71879a" } },
    yAxis: { type: "value", axisLabel: { color: "#71879a" }, splitLine: { lineStyle: { color: "#14283a" } } },
    series: [
      { name: "Train Loss", type: "line", smooth: true, data: training.data?.data.map((point) => point.loss) ?? [], showSymbol: false, lineStyle: { color: "#4f91ff", width: 2 } },
      { name: "Validation Loss", type: "line", smooth: true, data: training.data?.data.map((point) => point.validationLoss) ?? [], showSymbol: false, lineStyle: { color: "#27d3ad", width: 2 } },
    ],
  }), [training.data?.data]);

  const importanceOption = useMemo<EChartsOption>(() => {
    const rows = (importance.data?.data ?? []).slice(0, 18).reverse();
    return {
      animationDuration: 280,
      grid: { left: 128, right: 18, top: 16, bottom: 24 },
      tooltip: { trigger: "axis", backgroundColor: "#071521", borderColor: "#24506b", textStyle: { color: "#e2edf5" } },
      xAxis: { type: "value", axisLabel: { color: "#71879a" }, splitLine: { lineStyle: { color: "#14283a" } } },
      yAxis: { type: "category", data: rows.map((row) => row.feature), axisLabel: { color: "#a8bdcc", fontSize: 10 } },
      series: [{ type: "bar", data: rows.map((row) => row.importance), itemStyle: { color: "#4389ef", borderRadius: [0, 3, 3, 0] }, barMaxWidth: 12 }],
    };
  }, [importance.data?.data]);

  const scatterOption = useMemo<EChartsOption>(() => ({
    animationDuration: 280,
    grid: { left: 54, right: 20, top: 22, bottom: 38 },
    tooltip: {
      trigger: "item",
      backgroundColor: "#071521",
      borderColor: "#24506b",
      textStyle: { color: "#e2edf5" },
      formatter: (params: unknown) => {
        const value = (params as { value?: [number, number, string] }).value;
        return value ? `${value[2]}<br/>score ${formatNumber(value[0], 4)}<br/>return ${formatPercent(value[1])}` : "";
      },
    },
    xAxis: { type: "value", name: "Prediction", nameTextStyle: { color: "#71879a" }, axisLabel: { color: "#71879a" }, splitLine: { lineStyle: { color: "#14283a" } } },
    yAxis: { type: "value", name: "Actual Return", nameTextStyle: { color: "#71879a" }, axisLabel: { color: "#71879a" }, splitLine: { lineStyle: { color: "#14283a" } } },
    series: [{
      type: "scatter",
      symbolSize: 6,
      data: (predictions.data?.data ?? [])
        .filter((row) => row.actualReturn !== null && row.actualReturn !== undefined)
        .map((row) => [row.score, row.actualReturn, row.symbol]),
      itemStyle: { color: "#4b92ff", opacity: 0.62 },
    }],
  }), [predictions.data?.data]);

  const metricOption = useMemo<EChartsOption>(() => {
    const rows = selectVisualMetrics(observed?.metrics ?? []).slice(0, 14).reverse();
    return {
      animationDuration: 280,
      grid: { left: 160, right: 24, top: 16, bottom: 24 },
      tooltip: { trigger: "axis", backgroundColor: "#071521", borderColor: "#24506b", textStyle: { color: "#e2edf5" } },
      xAxis: { type: "value", axisLabel: { color: "#71879a" }, splitLine: { lineStyle: { color: "#14283a" } } },
      yAxis: { type: "category", data: rows.map((row) => row.label), axisLabel: { color: "#a8bdcc", fontSize: 10, width: 145, overflow: "truncate" } },
      series: [{
        type: "bar",
        data: rows.map((row) => ({
          value: row.value,
          itemStyle: { color: row.value >= 0 ? groupColor(row.group) : "#ef6870", borderRadius: [0, 3, 3, 0] },
        })),
        barMaxWidth: 12,
      }],
    };
  }, [observed?.metrics]);

  if (models.isLoading) return <StateView state="loading" />;
  if (!list.length) return <div className="institutional-workbench"><WorkbenchHeader eyebrow="MODEL REGISTRY / OBSERVABILITY" title="模型注册表" description="版本、训练证据、评估、血缘与人工 Gate 的统一模型资产视图。" context="source-backed only" /><ActionableState title="没有可识别模型" detail="先从训练实验室验证并运行研究任务；成功产物会由现有 RuntimeIndexer 自动进入注册表。" icon={Brain} primary={{ label: "打开训练实验室", onClick: () => navigate("/training") }} secondary={{ label: "检查 Runtime", onClick: () => navigate("/runtime") }} /></div>;

  const keyMetrics = selectKeyMetrics(observed?.metrics ?? []);

  return (
    <div className="page institutional-workbench model-observatory-page">
      <WorkbenchHeader eyebrow="MODEL REGISTRY / OBSERVABILITY" title="模型注册表" description="比较版本、检查训练与验证证据、追踪 lineage；promotion 与实盘始终由独立 Gate 控制。" asOf={formatDate(selected?.createdAt)} context="research registry" actions={<button type="button" className="primary" onClick={() => navigate("/training")}><HardDrives size={14} />打开训练实验室</button>} />
      <WorkbenchMetricStrip metrics={[
        { label: "模型资产", value: String(list.length), detail: `${visibleModels.length} visible`, tone: "info", icon: Database },
        { label: "Deep Alpha", value: String(familyCounts.deep_alpha ?? 0), detail: "registered family", tone: "ai", icon: Brain },
        { label: "Production ready", value: String(list.filter((item) => item.productionReady).length), detail: "explicit metadata", tone: "positive", icon: ShieldCheck },
        { label: "待评估", value: String(list.filter((item) => !item.verdict).length), detail: "missing verdict", tone: "warning", icon: WarningCircle },
        { label: "比较篮", value: `${compareIds.length}/4`, detail: "normalized metrics", tone: compareIds.length >= 2 ? "info" : "neutral", icon: ArrowsLeftRight },
        { label: "当前能力", value: `${Object.values(observed?.availability ?? {}).filter(Boolean).length}/${Object.keys(observed?.availability ?? {}).length || 6}`, detail: "visual evidence", tone: "info", icon: ChartScatter },
      ]} />
      <section className="model-commandbar">
        <div className="model-search">
          <MagnifyingGlass size={17} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索模型版本、family、verdict、artifact path" />
        </div>
        <div className="model-family-tabs">
          {Object.entries(familyLabels).map(([key, label]) => (
            <button key={key} className={family === key ? "active" : ""} onClick={() => setFamily(key)}>
              {label}<span>{familyCounts[key] ?? 0}</span>
            </button>
          ))}
        </div>
        <div className="compare-status">
          <ArrowsLeftRight size={17} />
          <span>{compareIds.length}/4 对比</span>
        </div>
      </section>

      <section className="model-observatory-layout">
        <aside className="model-catalog panel">
          <div className="catalog-header">
            <div><strong>模型资产目录</strong><span>{visibleModels.length} visible / {list.length} total</span></div>
            <Database size={18} />
          </div>
          <div className="model-catalog-list">
            {visibleModels.map((model) => {
              const selectedModel = model.id === modelId;
              const compared = compareIds.includes(model.id);
              return (
                <button key={model.id} className={selectedModel ? "active" : ""} onClick={() => selectModel(model.id)}>
                  <span className={`model-family-icon family-${model.modelFamily ?? "generic_artifact"}`}>
                    {model.modelFamily === "reinforcement_learning" ? <Brain size={18} /> :
                      model.modelFamily === "intraday_t_plus_one" ? <FunnelSimple size={18} /> :
                        model.modelFamily === "deep_alpha" ? <Cpu size={18} /> : <Stack size={18} />}
                  </span>
                  <span className="model-catalog-copy">
                    <strong>{model.version ?? model.modelType}</strong>
                    <small>{familyLabels[model.modelFamily ?? "generic_artifact"] ?? model.modelFamily} · {formatDate(model.createdAt)}</small>
                    <em>{model.verdict ?? model.status}</em>
                  </span>
                  <span
                    className={`compare-check ${compared ? "checked" : ""}`}
                    role="checkbox"
                    aria-checked={compared}
                    tabIndex={0}
                    onClick={(event) => {
                      event.stopPropagation();
                      setCompareIds((current) =>
                        compared ? current.filter((id) => id !== model.id) : current.length >= 4 ? current : [...current, model.id],
                      );
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault();
                        event.stopPropagation();
                        setCompareIds((current) =>
                          compared ? current.filter((id) => id !== model.id) : current.length >= 4 ? current : [...current, model.id],
                        );
                      }
                    }}
                  >
                    {compared ? <Check size={12} weight="bold" /> : null}
                  </span>
                </button>
              );
            })}
            {!visibleModels.length ? <StateView state="empty" detail="当前过滤条件没有模型。" /> : null}
          </div>
        </aside>

        <main className="model-observatory-main">
          <section className="model-hero panel">
            <div className="model-hero-primary">
              <span className="model-kicker">{selected?.modelFamily ?? "model"} / {selected?.sourceKind ?? "artifact"}</span>
              <div className="model-title-row">
                <h2>{selected?.version ?? selected?.modelType}</h2>
                <StatusBadge status={selected?.status ?? "partial"} label={selected?.verdict ?? selected?.status} />
              </div>
              <p>{selected?.path}</p>
              <div className="model-hero-tags">
                <span><Cpu size={14} /> {selected?.modelType ?? "unknown"}</span>
                <span><HardDrives size={14} /> {selected?.device ?? "device unknown"}</span>
                <span><ShieldCheck size={14} /> checkpoint metadata only</span>
              </div>
            </div>
            <div className="model-coverage">
              <strong>{Object.values(observed?.availability ?? {}).filter(Boolean).length}</strong>
              <span>/ {Object.keys(observed?.availability ?? {}).length || 6} 可视化能力</span>
              <div className="coverage-meter">
                <i style={{ width: `${coveragePercent(observed?.availability)}%` }} />
              </div>
              <small>
                {observed?.artifacts.length ?? 0} linked artifacts ·{" "}
                {formatBytes(observed?.artifacts.reduce((total, artifact) => total + artifact.sizeBytes, 0))}
              </small>
            </div>
          </section>

          {selected?.issues.length ? (
            <div className="model-issue-strip">
              <WarningCircle size={18} />
              <div><strong>模型可观测性不完整</strong><span>{selected.issues.map((issue) => issue.message).join("；")}</span></div>
            </div>
          ) : null}

          <section className="metric-grid metric-grid-6 model-key-metrics">
            <MetricCard label="模型类型" value={selected?.modelType ?? "暂无"} icon={Cpu} />
            <MetricCard label="特征数量" value={formatCompact(selected?.featureCount)} detail={`${selected?.horizons.map((item) => `${item}D`).join(" / ") || "policy model"}`} />
            <MetricCard label={keyMetrics[0]?.label ?? "样本数量"} value={keyMetrics[0] ? formatMetric(keyMetrics[0]) : formatCompact(selected?.sampleCount)} />
            <MetricCard label={keyMetrics[1]?.label ?? "训练设备"} value={keyMetrics[1] ? formatMetric(keyMetrics[1]) : selected?.device ?? "暂无"} />
            <MetricCard label={keyMetrics[2]?.label ?? "训练截止"} value={keyMetrics[2] ? formatMetric(keyMetrics[2]) : formatDate(selected?.trainEnd)} />
            <MetricCard label="Artifact 覆盖" value={`${observed?.artifacts.length ?? 0}`} detail={`${observed?.metrics.length ?? 0} normalized metrics`} icon={FileCode} />
          </section>

          {comparison.data?.data.models.length ? (
            <Panel title="模型横向对比" eyebrow={`${comparison.data.data.models.length} selected · normalized persisted metrics`} className="model-comparison-panel">
              <div className="comparison-matrix">
                <div className="comparison-row comparison-head">
                  <span>Metric</span>
                  {comparison.data.data.models.map((model) => <strong key={model.id}>{model.version}</strong>)}
                </div>
                {comparison.data.data.metricKeys.slice(0, 12).map((key) => (
                  <div className="comparison-row" key={key}>
                    <span>{key.replaceAll("_", " ")}</span>
                    {comparison.data.data.models.map((model) => (
                      <strong key={model.id} className={(model.metrics[key] ?? 0) >= 0 ? "tone-positive" : "tone-negative"}>
                        {formatNumber(model.metrics[key], 4)}
                      </strong>
                    ))}
                  </div>
                ))}
              </div>
            </Panel>
          ) : null}

          <nav className="model-tabs" aria-label="模型详情">
            {modelTabs.map(({ key, label, icon: IconComponent }) => (
              <button key={key} className={activeTab === key ? "active" : ""} onClick={() => setActiveTab(key)}>
                <IconComponent size={16} /> {label}
              </button>
            ))}
          </nav>

          {activeTab === "overview" ? (
            <section className="model-visual-grid">
              <Panel title="训练与验证曲线" eyebrow="Persisted history only">
                {training.data?.data.length && training.data.data.some((point) => point.loss !== null)
                  ? <EChart option={lossOption} className="chart chart-medium" />
                  : <StateView state="empty" detail="该模型没有逐 epoch loss；训练规模指标已显示在 Metrics。" />}
              </Panel>
              <Panel title="特征重要性" eyebrow="Native / permutation / persisted gain">
                {importance.data?.data.length
                  ? <EChart option={importanceOption} className="chart chart-medium" />
                  : <StateView state="empty" detail="未发现 feature_importance artifact。" />}
              </Panel>
              <Panel title="预测分数 vs 实际收益" eyebrow={`${predictions.data?.data.length ?? 0} rows · source-backed`}>
                {predictions.data?.data.some((row) => row.actualReturn !== null && row.actualReturn !== undefined)
                  ? <EChart option={scatterOption} className="chart chart-medium" />
                  : predictions.data?.data.length
                    ? <PredictionWeightView predictions={predictions.data.data} />
                    : <StateView state="empty" detail="该模型没有可映射 predictions；artifact 仍可在关联产物中查看。" />}
              </Panel>
              <Panel title="核心评估指标" eyebrow={`${observed?.metrics.length ?? 0} normalized metrics`}>
                {observed?.metrics.length
                  ? <EChart option={metricOption} className="chart chart-medium" />
                  : <StateView state="empty" detail="没有 persisted numeric evaluation。" />}
              </Panel>
              <Panel title="能力覆盖矩阵" eyebrow="What can actually be visualized" className="model-capability-panel">
                <div className="capability-observability-grid">
                  {Object.entries(observed?.availability ?? {}).map(([name, available]) => (
                    <div key={name} className={available ? "available" : ""}>
                      <span>{available ? <Check size={13} /> : <WarningCircle size={13} />}</span>
                      <strong>{name}</strong>
                      <small>{available ? "已接入" : "artifact 缺失"}</small>
                    </div>
                  ))}
                </div>
              </Panel>
              <Panel title="最新模型输出" eyebrow="Prediction / policy weight sample" className="model-predictions-panel">
                <PredictionTable predictions={predictions.data?.data ?? []} />
              </Panel>
            </section>
          ) : null}

          {activeTab === "metrics" ? (
            <section className="model-detail-grid">
              <Panel title="标准化指标库" eyebrow="Return · quality · risk · scale" className="model-metric-table-panel">
                <MetricTable metrics={observed?.metrics ?? []} />
              </Panel>
              <Panel title="评估原文" eyebrow="Persisted verdict / strict evaluation">
                {observed?.evaluations.length ? (
                  <div className="evaluation-stack">
                    {observed.evaluations.map((evaluation) => (
                      <details key={evaluation.path} open>
                        <summary>{evaluation.name}<span>{evaluation.path}</span></summary>
                        <pre className="json-view">{JSON.stringify(evaluation.data, null, 2)}</pre>
                      </details>
                    ))}
                  </div>
                ) : <StateView state="empty" />}
              </Panel>
            </section>
          ) : null}

          {activeTab === "artifacts" ? (
            <Panel title="模型关联产物" eyebrow="Checkpoint content is never exposed" className="model-artifact-panel">
              {observed?.artifacts.length ? (
                <div className="table-scroll">
                  <table className="data-table">
                    <thead><tr><th>角色</th><th>文件</th><th>路径</th><th className="numeric">大小</th><th>更新时间</th><th>浏览器预览</th></tr></thead>
                    <tbody>{observed.artifacts.map((artifact) => (
                      <tr key={artifact.path}>
                        <td><span className={`artifact-kind kind-${artifact.role}`}>{artifact.role}</span></td>
                        <td><strong>{artifact.name}</strong><span>{artifact.extension}</span></td>
                        <td className="mono">{artifact.path}</td>
                        <td className="numeric mono">{formatBytes(artifact.sizeBytes)}</td>
                        <td className="mono">{formatDate(artifact.modifiedAt)}</td>
                        <td><StatusBadge status={artifact.previewable ? "ready" : "partial"} label={artifact.previewable ? "metadata/data" : "metadata only"} /></td>
                      </tr>
                    ))}</tbody>
                  </table>
                </div>
              ) : <StateView state="empty" />}
            </Panel>
          ) : null}

          {activeTab === "config" ? (
            <Panel title="模型配置与 Schema" eyebrow="Repository-relative metadata">
              {observed?.config && Object.keys(observed.config).length
                ? <pre className="json-view model-config-view">{JSON.stringify(observed.config, null, 2)}</pre>
                : <StateView state="empty" detail="没有可读取配置；binary checkpoint 未反序列化。" />}
            </Panel>
          ) : null}
        </main>
      </section>
    </div>
  );
}

function selectKeyMetrics(metrics: ModelMetric[]): ModelMetric[] {
  const preferred = [
    "strict_annualized",
    "annualised_return",
    "annualized_return",
    "strict_sharpe",
    "annualised_sharpe",
    "rank_ic_mean",
    "ICIR",
    "excess_return_after_costs",
    "max_drawdown",
    "hit_rate",
  ];
  const selected = preferred
    .map((key) => metrics.find((metric) => metric.key.endsWith(key)))
    .filter((metric): metric is ModelMetric => Boolean(metric));
  return selected.slice(0, 3);
}

function selectVisualMetrics(metrics: ModelMetric[]): ModelMetric[] {
  return metrics
    .filter((metric) => metric.group !== "scale" && Number.isFinite(metric.value))
    .sort((left, right) => Math.abs(right.value) - Math.abs(left.value));
}

function formatMetric(metric: ModelMetric): string {
  if (metric.unit === "ratio") return formatPercent(metric.value);
  if (metric.unit === "count") return formatCompact(metric.value);
  if (metric.unit === "bps") return `${formatNumber(metric.value)} bps`;
  return formatNumber(metric.value, 4);
}

function groupColor(group: ModelMetric["group"]): string {
  if (group === "return") return "#25c79f";
  if (group === "risk") return "#e9a740";
  if (group === "quality") return "#4b92ff";
  return "#7f8fa1";
}

function coveragePercent(availability?: Record<string, boolean>): number {
  const values = Object.values(availability ?? {});
  if (!values.length) return 0;
  return (values.filter(Boolean).length / values.length) * 100;
}

function PredictionWeightView({ predictions }: { predictions: Prediction[] }): JSX.Element {
  const option = useMemo<EChartsOption>(() => {
    const rows = predictions.slice(0, 50);
    return {
      animationDuration: 280,
      grid: { left: 92, right: 20, top: 18, bottom: 28 },
      xAxis: { type: "value", axisLabel: { color: "#71879a" }, splitLine: { lineStyle: { color: "#14283a" } } },
      yAxis: { type: "category", data: rows.map((row) => row.symbol), axisLabel: { color: "#9cb1c3", fontSize: 9 } },
      series: [{ type: "bar", data: rows.map((row) => row.score), itemStyle: { color: "#27caa2", borderRadius: [0, 3, 3, 0] }, barMaxWidth: 10 }],
    };
  }, [predictions]);
  return <EChart option={option} className="chart chart-medium" />;
}

function PredictionTable({ predictions }: { predictions: Prediction[] }): JSX.Element {
  if (!predictions.length) return <StateView state="empty" />;
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead><tr><th>日期</th><th>股票</th><th>类型</th><th className="numeric">预测/权重</th><th className="numeric">实际收益</th><th className="numeric">排名</th></tr></thead>
        <tbody>{predictions.slice(0, 120).map((row, index) => (
          <tr key={`${row.symbol}-${row.datetime}-${index}`}>
            <td className="mono">{row.datetime.slice(0, 10) || "—"}</td>
            <td><strong>{row.symbol || "portfolio"}</strong></td>
            <td>{row.horizon ?? "—"}</td>
            <td className="numeric mono">{formatNumber(row.score, 5)}</td>
            <td className={`numeric mono ${(row.actualReturn ?? 0) >= 0 ? "tone-positive" : "tone-negative"}`}>{formatNumber(row.actualReturn, 4)}</td>
            <td className="numeric mono">{row.rank ?? "—"}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

function MetricTable({ metrics }: { metrics: ModelMetric[] }): JSX.Element {
  if (!metrics.length) return <StateView state="empty" />;
  return (
    <div className="table-scroll">
      <table className="data-table">
        <thead><tr><th>分组</th><th>指标</th><th className="numeric">数值</th><th>单位</th><th>来源</th></tr></thead>
        <tbody>{metrics.map((metric) => (
          <tr key={`${metric.source}-${metric.key}`}>
            <td><span className={`metric-group metric-group-${metric.group}`}>{metric.group}</span></td>
            <td><strong>{metric.label}</strong><span className="mono">{metric.key}</span></td>
            <td className={`numeric mono ${metric.value >= 0 ? "tone-positive" : "tone-negative"}`}>{formatMetric(metric)}</td>
            <td>{metric.unit}</td>
            <td>{metric.source}</td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}
