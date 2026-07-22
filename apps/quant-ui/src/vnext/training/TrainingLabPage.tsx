import { useEffect, useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useSearchParams } from "react-router-dom";
import { CaretDown, CaretUp, SlidersHorizontal } from "@phosphor-icons/react";
import type { JobSummary, JobValidation, ModelComparison, ModelObservability, ModelSummary, TrainingMetricPoint } from "../../api/types";
import { apiPost } from "../../api/client";
import { mutableTemplate, type JobLaunchPayload } from "../../domain/jobTemplates";
import { useApi } from "../../hooks/useApi";
import { StateView } from "../../components/StateView";
import { TrainingCanvas } from "./TrainingCanvas";
import { TrainingComparison } from "./TrainingComparison";
import { TrainingConsole } from "./TrainingConsole";
import { TrainingNavigator } from "./TrainingNavigator";
import { TrainingRunInspector } from "./TrainingRunInspector";

type ConfigView = "form" | "yaml" | "diff" | "validation";

const configFields: Array<{ key: string; label: string; type: "text" | "number" | "boolean"; group: string }> = [
  { key: "dataset_path", label: "Dataset", type: "text", group: "Data" },
  { key: "silver_panel_path", label: "Market panel", type: "text", group: "Data" },
  { key: "horizon_class", label: "Horizon", type: "text", group: "Target" },
  { key: "feature_policy", label: "Feature policy", type: "text", group: "Features" },
  { key: "max_epochs", label: "Epochs", type: "number", group: "Optimization" },
  { key: "batch_size", label: "Batch size", type: "number", group: "Optimization" },
  { key: "learning_rate", label: "Learning rate", type: "number", group: "Optimization" },
  { key: "early_stopping_patience", label: "Early stopping", type: "number", group: "Optimization" },
  { key: "require_gpu", label: "Require GPU", type: "boolean", group: "Runtime" },
  { key: "output_dir", label: "Output directory", type: "text", group: "Runtime" },
];

export function TrainingLabPage(): JSX.Element {
  const queryClient = useQueryClient();
  const [searchParams, setSearchParams] = useSearchParams();
  const models = useApi<ModelSummary[]>(["vnext-training-models"], "/models");
  const jobs = useApi<JobSummary[]>(["global-activity-jobs"], "/jobs", undefined, { refetchInterval: 5_000, staleTime: 2_000 });
  const modelList = models.data?.data ?? [];
  const jobList = jobs.data?.data ?? [];
  const [selectedModelId, setSelectedModelId] = useState(searchParams.get("modelId") ?? "");
  const [selectedJobId, setSelectedJobId] = useState(searchParams.get("job") ?? "");
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [config, setConfig] = useState<JobLaunchPayload>(() => mutableTemplate("train"));
  const [configView, setConfigView] = useState<ConfigView>("form");
  const [configExpanded, setConfigExpanded] = useState(false);
  const [validation, setValidation] = useState<JobValidation | undefined>();
  const [validationError, setValidationError] = useState("");
  const [busy, setBusy] = useState(false);
  const [armed, setArmed] = useState(false);

  useEffect(() => {
    const preferred = modelList.find((model) => model.modelFamily === "deep_alpha") ?? modelList[0];
    if (!selectedModelId && preferred) setSelectedModelId(preferred.id);
  }, [modelList, selectedModelId]);
  useEffect(() => {
    const preferred = jobList.find((job) => job.type === "train" && ["queued", "running"].includes(job.status)) ?? jobList.find((job) => job.type === "train");
    if (!selectedJobId && preferred) setSelectedJobId(preferred.id);
  }, [jobList, selectedJobId]);

  const detail = useApi<ModelObservability>(["vnext-training-model", selectedModelId], selectedModelId ? `/models/${selectedModelId}/observability` : null);
  const metrics = useApi<TrainingMetricPoint[]>(["vnext-training-metrics", selectedModelId], selectedModelId ? `/models/${selectedModelId}/training-metrics` : null, undefined, { refetchInterval: jobList.some((job) => job.id === selectedJobId && job.status === "running") ? 3_000 : false });
  const comparison = useApi<ModelComparison>(["vnext-training-comparison", ...compareIds], compareIds.length >= 2 ? "/models/compare" : null, { ids: compareIds.join(",") });
  const logs = useApi<string[]>(["vnext-training-logs", selectedJobId], selectedJobId ? `/jobs/${selectedJobId}/logs` : null, { limit: 2_000 }, { refetchInterval: jobList.some((job) => job.id === selectedJobId && job.status === "running") ? 2_000 : false });
  const selectedJob = jobList.find((job) => job.id === selectedJobId);

  const setModel = (id: string): void => {
    setSelectedModelId(id);
    const next = new URLSearchParams(searchParams);
    next.set("modelId", id);
    setSearchParams(next);
  };
  const setJob = (id: string): void => {
    setSelectedJobId(id);
    const next = new URLSearchParams(searchParams);
    next.set("job", id);
    setSearchParams(next);
  };
  const toggleCompare = (id: string): void => setCompareIds((current) => current.includes(id) ? current.filter((item) => item !== id) : [...current, id].slice(-6));

  const validate = async (): Promise<void> => {
    setBusy(true);
    setValidation(undefined);
    setValidationError("");
    try {
      const result = await apiPost<JobValidation>("/jobs/train/validate", config);
      setValidation(result.data);
      setConfigView("validation");
      setConfigExpanded(true);
    } catch (error) {
      setValidationError(error instanceof Error ? error.message : "Validation failed");
      setConfigView("validation");
      setConfigExpanded(true);
    } finally {
      setBusy(false);
    }
  };

  const start = async (): Promise<void> => {
    if (!validation?.valid || !armed) return;
    setBusy(true);
    setValidationError("");
    try {
      const result = await apiPost<JobSummary>("/jobs/train", config);
      setJob(result.data.id);
      setArmed(false);
      await queryClient.invalidateQueries({ queryKey: ["global-activity-jobs"] });
    } catch (error) {
      setValidationError(error instanceof Error ? error.message : "Training launch failed");
    } finally {
      setBusy(false);
    }
  };

  const cancel = async (): Promise<void> => {
    if (!selectedJob) return;
    setBusy(true);
    try {
      await apiPost(`/jobs/${selectedJob.id}/cancel`, {});
      await queryClient.invalidateQueries({ queryKey: ["global-activity-jobs"] });
    } catch (error) {
      setValidationError(error instanceof Error ? error.message : "Cancel failed");
    } finally {
      setBusy(false);
    }
  };

  const setParameter = (key: string, value: string | number | boolean): void => {
    setConfig((current) => ({ ...current, parameters: { ...current.parameters, [key]: value } }));
    setValidation(undefined);
  };
  const yaml = useMemo(() => toYaml(config), [config]);

  if (models.isLoading || jobs.isLoading) return <StateView state="loading" detail="正在恢复 experiments、runs 和 persisted task state。" />;

  return (
    <div className="vnext-training-lab">
      <header className="vnext-training-title"><div><span>TRAINING WORKSTATION</span><h1>Training Lab</h1><p>Schema-driven configuration · persisted Jobs · live metrics · logs · lineage</p></div><div><strong>{jobList.filter((job) => job.type === "train" && ["queued", "running"].includes(job.status)).length}</strong><span>active training runs</span></div></header>
      <div className="vnext-training-layout">
        <TrainingNavigator models={modelList} jobs={jobList} selectedModelId={selectedModelId} selectedJobId={selectedJobId} compareIds={compareIds} query={query} setQuery={setQuery} selectModel={setModel} selectJob={setJob} toggleCompare={toggleCompare} />
        <main>
          <TrainingComparison ids={compareIds} comparison={comparison.data?.data} loading={comparison.isLoading} clear={() => setCompareIds([])} />
          <section className={`vnext-config-inspector ${configExpanded ? "expanded" : "collapsed"}`}>
            <header>
              <div><span>CONFIGURATION INSPECTOR</span><h2>train-v8-deep</h2></div>
              <div className="vnext-config-header-actions">
                {configExpanded ? <nav>{(["form", "yaml", "diff", "validation"] as ConfigView[]).map((item) => <button type="button" key={item} className={configView === item ? "active" : ""} onClick={() => setConfigView(item)}>{item}</button>)}</nav> : null}
                <button type="button" className="vnext-config-toggle" aria-expanded={configExpanded} onClick={() => setConfigExpanded((current) => !current)}>
                  <SlidersHorizontal size={14} />{configExpanded ? "收起配置" : "编辑配置"}{configExpanded ? <CaretUp size={12} /> : <CaretDown size={12} />}
                </button>
              </div>
            </header>
            {!configExpanded ? <div className="vnext-config-summary"><span><small>Dataset</small><strong>{shortPath(String(config.parameters.dataset_path ?? "UNAVAILABLE"))}</strong></span><span><small>Horizon</small><strong>{String(config.parameters.horizon_class ?? "UNAVAILABLE")}</strong></span><span><small>Epochs / batch</small><strong>{String(config.parameters.max_epochs ?? "—")} / {String(config.parameters.batch_size ?? "—")}</strong></span><span><small>Runtime</small><strong>{config.parameters.require_gpu ? "GPU REQUIRED" : "CPU ALLOWED"}</strong></span></div> : null}
            {configExpanded && configView === "form" ? <div className="vnext-config-form">{configFields.map((field) => <label key={field.key}><span><small>{field.group}</small>{field.label}</span>{field.type === "boolean" ? <input type="checkbox" checked={Boolean(config.parameters[field.key])} onChange={(event) => setParameter(field.key, event.target.checked)} /> : <input type={field.type} value={String(config.parameters[field.key] ?? "")} onChange={(event) => setParameter(field.key, field.type === "number" ? Number(event.target.value) : event.target.value)} />}</label>)}</div> : null}
            {configExpanded && configView === "yaml" ? <pre className="vnext-config-code">{yaml}</pre> : null}
            {configExpanded && configView === "diff" ? <div className="vnext-config-diff"><p>Current draft is compared with the canonical `train-v8-deep` template.</p><pre>{diffTemplate(config)}</pre></div> : null}
            {configExpanded && configView === "validation" ? <div className={validation ? "vnext-config-validation passed" : "vnext-config-validation"}><strong>{validation ? "Backend validation passed" : validationError ? "Validation failed" : "Not validated"}</strong><p>{validation ? `${validation.entrypoint} · outputs: ${validation.outputPaths.join(", ") || "none"}` : validationError || "Validate checks command allowlist, required parameters, input existence and runtime output boundaries."}</p>{validation?.warnings.map((warning) => <small key={warning}>{warning}</small>)}</div> : null}
          </section>
          <TrainingCanvas points={metrics.data?.data ?? []} job={selectedJob} />
        </main>
        <TrainingRunInspector model={detail.data?.data} job={selectedJob} validation={validation} busy={busy} armed={armed} setArmed={setArmed} validate={() => void validate()} start={() => void start()} cancel={() => void cancel()} clone={() => { setConfig(mutableTemplate("train")); setValidation(undefined); setArmed(false); }} />
      </div>
      <TrainingConsole lines={logs.data?.data ?? []} jobId={selectedJob?.id} />
    </div>
  );
}

function toYaml(value: unknown, indent = 0): string {
  if (!value || typeof value !== "object") return String(value);
  return Object.entries(value as Record<string, unknown>).map(([key, item]) => {
    const prefix = `${" ".repeat(indent)}${key}:`;
    if (item && typeof item === "object" && !Array.isArray(item)) return `${prefix}\n${toYaml(item, indent + 2)}`;
    if (Array.isArray(item)) return `${prefix} [${item.join(", ")}]`;
    return `${prefix} ${typeof item === "string" ? JSON.stringify(item) : String(item)}`;
  }).join("\n");
}

function diffTemplate(config: JobLaunchPayload): string {
  const baseline = mutableTemplate("train");
  return Object.keys({ ...baseline.parameters, ...config.parameters }).map((key) => {
    const before = baseline.parameters[key];
    const after = config.parameters[key];
    return before === after ? `  ${key}: ${String(after)}` : `- ${key}: ${String(before)}\n+ ${key}: ${String(after)}`;
  }).join("\n");
}

function shortPath(value: string): string {
  const parts = value.split("/").filter(Boolean);
  return parts.slice(-2).join("/") || value;
}
