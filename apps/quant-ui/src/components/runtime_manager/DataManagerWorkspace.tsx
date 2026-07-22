import { useMemo, useState } from "react";
import {
  ArrowClockwise,
  CheckCircle,
  Database,
  GlobeHemisphereEast,
  Play,
  ShieldCheck,
  Stop,
  WarningCircle,
} from "@phosphor-icons/react";
import { useNavigate } from "react-router-dom";
import { apiPost } from "../../api/client";
import type { DataManagerOverview, DataProvider, JobSummary } from "../../api/types";
import { useApi } from "../../hooks/useApi";
import { formatDate } from "../../utils/format";
import { Panel } from "../Panel";
import { StateView } from "../StateView";
import { StatusBadge } from "../StatusBadge";

type ExecutableProvider = "akshare_market" | "qlib_local" | "tushare_fundamentals";

const OUTPUTS: Record<ExecutableProvider, string> = {
  akshare_market: "runtime/data/v7/silver/market_panel/web_akshare_market_panel.parquet",
  qlib_local: "runtime/data/v7",
  tushare_fundamentals: "runtime/data/v7/raw/tushare/fundamentals",
};

function isExecutableProvider(value: string): value is ExecutableProvider {
  return value === "akshare_market" || value === "qlib_local" || value === "tushare_fundamentals";
}

export function DataManagerWorkspace(): JSX.Element {
  const navigate = useNavigate();
  const registry = useApi<DataManagerOverview>(["data-manager-providers"], "/data/providers");
  const jobs = useApi<JobSummary[]>(["data-manager-jobs"], "/jobs", undefined, { refetchInterval: 3_000, staleTime: 1_000 });
  const providers = registry.data?.data.providers ?? [];
  const [providerId, setProviderId] = useState<ExecutableProvider>("akshare_market");
  const [symbols, setSymbols] = useState("000001.SZ,600519.SH");
  const [startDate, setStartDate] = useState("2025-01-01");
  const [endDate, setEndDate] = useState("2026-07-22");
  const [outputPath, setOutputPath] = useState(OUTPUTS.akshare_market);
  const [providerUri, setProviderUri] = useState("runtime/data/v7/raw/qlib/cn_data");
  const [networkApproved, setNetworkApproved] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [actionError, setActionError] = useState("");
  const [submitted, setSubmitted] = useState<JobSummary | null>(null);
  const currentProvider = providers.find((provider) => provider.id === providerId);
  const dataJobs = useMemo(
    () => (jobs.data?.data ?? []).filter((job) => job.type === "data"),
    [jobs.data?.data],
  );
  const needsNetwork = providerId !== "qlib_local";
  const canSubmit = Boolean(
    currentProvider?.installed
    && symbols.trim()
    && startDate
    && endDate
    && outputPath.trim()
    && (providerId !== "qlib_local" || providerUri.trim())
    && (providerId !== "tushare_fundamentals" || currentProvider.configured)
    && (!needsNetwork || networkApproved),
  );

  const selectProvider = (provider: DataProvider): void => {
    if (!isExecutableProvider(provider.id)) return;
    setProviderId(provider.id);
    setOutputPath(OUTPUTS[provider.id]);
    setNetworkApproved(false);
    setActionError("");
    setSubmitted(null);
  };

  const submit = async (): Promise<void> => {
    if (!canSubmit) return;
    setSubmitting(true);
    setActionError("");
    try {
      const common = { symbols: symbols.trim(), start_date: startDate, end_date: endDate };
      const payload = providerId === "akshare_market"
        ? {
          commandId: "build-akshare-market-panel-v7",
          parameters: { ...common, output: outputPath.trim(), adjust: "qfq", allow_network: true },
        }
        : providerId === "qlib_local"
          ? {
            commandId: "build-market-panel-v7",
            parameters: { ...common, provider_uri: providerUri.trim(), output_root: outputPath.trim(), region: "cn" },
          }
          : {
            commandId: "build-fundamentals-v7",
            parameters: {
              ...common,
              provider: "tushare",
              fundamentals_root: outputPath.trim(),
              allow_network: true,
              token_env: "TUSHARE_TOKEN",
            },
          };
      const result = await apiPost<JobSummary>("/jobs/data", payload);
      setSubmitted(result.data);
      await jobs.refetch();
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "data job launch failed");
    } finally {
      setSubmitting(false);
    }
  };

  const cancel = async (jobId: string): Promise<void> => {
    setActionError("");
    try {
      await apiPost<JobSummary>(`/jobs/${jobId}/cancel`, {});
      await jobs.refetch();
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "cancel failed");
    }
  };

  if (registry.isLoading) return <StateView state="loading" />;
  if (registry.isError) return <StateView state="error" detail={registry.error.message} />;

  return (
    <section className="data-manager-workspace">
      <div className="data-manager-toolbar">
        <div><Database size={18} /><span><strong>DATA OPERATIONS</strong><small>provider → allowlisted job → PIT gate → manifest → Runtime catalog</small></span></div>
        <div>
          <button className="secondary-button" onClick={() => navigate("/settings?job=train&universe=all")}><GlobeHemisphereEast size={15} /> 全宇宙训练</button>
          <button className="secondary-button" onClick={() => navigate("/runtime?view=cleanup")}><ShieldCheck size={15} /> 受保护清理</button>
        </div>
      </div>

      <div className="data-manager-grid">
        <Panel title="数据服务" eyebrow="Detected backend capability · no browser credentials" className="data-provider-panel">
          <div className="data-provider-list">
            {providers.map((provider) => (
              <button
                type="button"
                key={provider.id}
                className={provider.id === providerId ? "selected" : ""}
                disabled={!isExecutableProvider(provider.id)}
                onClick={() => selectProvider(provider)}
              >
                <span><strong>{provider.label}</strong><small>{provider.assetClasses.join(" · ")} · {provider.intervals.join(" · ")}</small></span>
                <StatusBadge status={provider.status === "needs_configuration" ? "warning" : provider.status} label={provider.status === "ready" ? "可用" : provider.status === "needs_configuration" ? "需配置" : "未安装"} />
                <p>{provider.note}</p>
              </button>
            ))}
          </div>
        </Panel>

        <Panel title="查询 / 下载 / 更新" eyebrow="Explicit fields · validated backend contract" className="data-job-form-panel">
          <div className="data-job-form">
            <label className="data-symbols-field"><span>股票范围</span><input value={symbols} onChange={(event) => setSymbols(event.target.value)} placeholder="000001.SZ,600519.SH" /><small>网络下载必须显式给出股票；全宇宙训练使用已验证数据集，不在这里隐式抓取全市场。</small></label>
            <label><span>开始日期</span><input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} /></label>
            <label><span>结束日期</span><input type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} /></label>
            {providerId === "qlib_local" ? <label className="data-wide-field"><span>Qlib provider URI</span><input value={providerUri} onChange={(event) => setProviderUri(event.target.value)} /></label> : null}
            <label className="data-wide-field"><span>Runtime 输出</span><input value={outputPath} onChange={(event) => setOutputPath(event.target.value)} /><small>后端拒绝 Runtime 之外的输出路径。</small></label>
          </div>
          {needsNetwork ? (
            <label className="network-confirmation">
              <input type="checkbox" checked={networkApproved} onChange={(event) => setNetworkApproved(event.target.checked)} />
              <span><strong>确认允许本次数据任务访问网络</strong><small>只运行选中的 allowlisted provider command；不会接受 shell、URL 或凭据文本。</small></span>
            </label>
          ) : <div className="local-provider-notice"><Database size={16} /><span>本地查询不会开启网络访问；provider URI 必须位于项目或 Runtime 范围内且真实存在。</span></div>}
          <div className="data-job-actions">
            <span><ShieldCheck size={16} />PIT/schema 失败会使任务失败，不会回退为 mock。</span>
            <button className="primary-button" disabled={!canSubmit || submitting} onClick={submit}><Play size={15} weight="fill" />{submitting ? "提交中…" : "提交数据任务"}</button>
          </div>
          {actionError ? <div className="launch-message launch-error"><WarningCircle size={17} />{actionError}</div> : null}
          {submitted ? <div className="launch-message launch-success"><CheckCircle size={17} />已提交 {submitted.commandId} · {submitted.id}</div> : null}
        </Panel>
      </div>

      <Panel title="数据任务" eyebrow={`${dataJobs.length} governed jobs · refresh 3s`} className="data-job-queue-panel">
        {dataJobs.length ? (
          <div className="table-scroll"><table className="data-table"><thead><tr><th>任务</th><th>状态</th><th>进度</th><th>创建 / 完成</th><th>输出</th><th>操作</th></tr></thead><tbody>
            {dataJobs.map((job) => <tr key={job.id}><td><strong>{job.commandId}</strong><span className="mono">{job.id}</span></td><td><StatusBadge status={job.status} /></td><td className="mono">{job.progress == null ? "indeterminate" : `${Math.round(job.progress * 100)}%`}</td><td><strong>{formatDate(job.createdAt)}</strong><span>{job.finishedAt ? formatDate(job.finishedAt) : job.message ?? "—"}</span></td><td>{job.outputPaths.join(" · ") || "等待产物"}</td><td>{["queued", "running", "cancelling"].includes(job.status) ? <button className="secondary-button" onClick={() => cancel(job.id)}><Stop size={14} />取消</button> : "—"}</td></tr>)}
          </tbody></table></div>
        ) : <StateView state={jobs.isError ? "error" : "empty"} detail={jobs.error?.message ?? "尚未提交数据下载或更新任务。"} />}
        <div className="data-job-queue-footer"><ArrowClockwise size={15} /><span>成功后自动使 RuntimeIndexer 缓存失效；Catalog 的下一次读取会扫描新 manifest/artifact。</span></div>
      </Panel>
    </section>
  );
}
