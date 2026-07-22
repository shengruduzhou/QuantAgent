import { useEffect, useMemo, useState } from "react";
import {
  ArrowClockwise,
  CheckCircle,
  Database,
  DownloadSimple,
  MagnifyingGlass,
  Play,
  Record,
  ShieldCheck,
  Stop,
  UploadSimple,
  WarningCircle,
} from "@phosphor-icons/react";
import { useNavigate } from "react-router-dom";
import { apiGet, apiPost } from "../../api/client";
import type {
  DataCoverage,
  DataManagerOverview,
  DataProvider,
  JobSummary,
  QuarantineFile,
} from "../../api/types";
import { useApi } from "../../hooks/useApi";
import { formatCompact, formatDate } from "../../utils/format";
import { Panel } from "../Panel";
import { StateView } from "../StateView";
import { StatusBadge } from "../StatusBadge";

type WorkspaceTab = "acquire" | "coverage" | "transfer" | "recorder";
type TickflowMode = "daily" | "minute" | "tick" | "depth";
type RecorderMode = "tick" | "depth";
type TransferOperation = "import" | "export";

const DEFAULT_OUTPUTS: Record<string, string> = {
  tickflow: "runtime/data/v7/silver/market_panel/tickflow_daily.parquet",
  akshare_market: "runtime/data/v7/silver/market_panel/web_akshare_market_panel.parquet",
  qlib_local: "runtime/data/v7",
  tushare_fundamentals: "runtime/data/v7/raw/tushare/fundamentals",
};

const TABS: Array<{ id: WorkspaceTab; label: string; detail: string }> = [
  { id: "acquire", label: "获取 / 更新", detail: "provider-specific" },
  { id: "coverage", label: "覆盖与重复", detail: "server scan" },
  { id: "transfer", label: "导入 / 导出", detail: "quarantine" },
  { id: "recorder", label: "实时录制", detail: "TickFlow L2" },
];

function providerLabel(provider: DataProvider): string {
  if (provider.status === "ready") return "可用";
  if (provider.status === "partial") return "部分可用";
  if (provider.status === "needs_configuration") return "需配置";
  return "未安装";
}

export function DataManagerWorkspace(): JSX.Element {
  const navigate = useNavigate();
  const registry = useApi<DataManagerOverview>(["data-manager-providers"], "/data/providers");
  const quarantine = useApi<QuarantineFile[]>(["data-manager-quarantine"], "/data/quarantine", undefined, { refetchInterval: 5_000 });
  const jobs = useApi<JobSummary[]>(["data-manager-jobs"], "/jobs", undefined, { refetchInterval: 2_000, staleTime: 500 });
  const providers = registry.data?.data.providers ?? [];
  const [tab, setTab] = useState<WorkspaceTab>("acquire");
  const [providerId, setProviderId] = useState("tickflow");
  const [tickflowMode, setTickflowMode] = useState<TickflowMode>("daily");
  const [symbols, setSymbols] = useState("000001.SZ,600519.SH");
  const [startDate, setStartDate] = useState("2025-01-01");
  const [endDate, setEndDate] = useState("2026-07-22");
  const [outputPath, setOutputPath] = useState(DEFAULT_OUTPUTS.tickflow);
  const [providerUri, setProviderUri] = useState("runtime/data/v7/raw/qlib/cn_data");
  const [networkApproved, setNetworkApproved] = useState(false);
  const [coveragePath, setCoveragePath] = useState("runtime/data/v7/silver/market_panel/market_panel.parquet");
  const [dateColumn, setDateColumn] = useState("trade_date");
  const [symbolColumn, setSymbolColumn] = useState("symbol");
  const [deepCoverage, setDeepCoverage] = useState(false);
  const [coverageResult, setCoverageResult] = useState<DataCoverage | null>(null);
  const [transferOperation, setTransferOperation] = useState<TransferOperation>("import");
  const [transferSource, setTransferSource] = useState("");
  const [transferOutput, setTransferOutput] = useState("runtime/data/imported/validated_import.parquet");
  const [loopSeconds, setLoopSeconds] = useState(30);
  const [maxIterations, setMaxIterations] = useState(120);
  const [recorderMode, setRecorderMode] = useState<RecorderMode>("tick");
  const [submitting, setSubmitting] = useState(false);
  const [actionError, setActionError] = useState("");
  const [submitted, setSubmitted] = useState<JobSummary | null>(null);

  useEffect(() => {
    const first = quarantine.data?.data[0]?.path;
    if (!transferSource && first) setTransferSource(first);
  }, [quarantine.data?.data, transferSource]);

  const currentProvider = providers.find((provider) => provider.id === providerId);
  const tickflowProvider = providers.find((provider) => provider.id === "tickflow");
  const dataJobs = useMemo(() => (jobs.data?.data ?? []).filter((job) => job.type === "data"), [jobs.data?.data]);
  const needsNetwork = providerId !== "qlib_local";
  const tickflowCredentialMissing = tickflowProvider?.missingOptionalRequirements?.includes("TICKFLOW_API_KEY") ?? false;
  const selectedModeNeedsKey = providerId === "tickflow" && tickflowMode !== "daily";
  const canAcquire = Boolean(
    currentProvider?.installed
    && symbols.trim()
    && startDate
    && endDate
    && (!selectedModeNeedsKey || !tickflowCredentialMissing)
    && (!needsNetwork || networkApproved)
    && (providerId !== "tushare_fundamentals" || currentProvider.configured)
    && (providerId === "tickflow" && tickflowMode !== "daily" ? true : outputPath.trim()),
  );

  const selectProvider = (provider: DataProvider): void => {
    if (!provider.commandId) {
      setTab("coverage");
      return;
    }
    setProviderId(provider.id);
    setOutputPath(DEFAULT_OUTPUTS[provider.id] ?? outputPath);
    setNetworkApproved(false);
    setActionError("");
    setSubmitted(null);
  };

  const launch = async (commandId: string, parameters: Record<string, string | number | boolean>): Promise<void> => {
    setSubmitting(true);
    setActionError("");
    try {
      const result = await apiPost<JobSummary>("/jobs/data", { commandId, parameters });
      setSubmitted(result.data);
      await jobs.refetch();
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "data operation failed");
    } finally {
      setSubmitting(false);
    }
  };

  const submitAcquire = async (): Promise<void> => {
    if (!canAcquire) return;
    const common = { symbols: symbols.trim(), allow_network: true };
    if (providerId === "tickflow") {
      if (tickflowMode === "daily") {
        await launch("fetch-tickflow-daily", { ...common, start_date: startDate, end_date: endDate, batch_size: 80, output: outputPath.trim() });
      } else if (tickflowMode === "minute") {
        await launch("fetch-tickflow-minute", { ...common, start: startDate, end: endDate });
      } else if (tickflowMode === "tick") {
        await launch("record-tickflow-quotes", { ...common, loop_seconds: 0, max_iterations: 1 });
      } else {
        await launch("record-tickflow-depth", { ...common, loop_seconds: 0, max_iterations: 1 });
      }
    } else if (providerId === "akshare_market") {
      await launch("build-akshare-market-panel-v7", { ...common, start_date: startDate, end_date: endDate, output: outputPath.trim(), adjust: "qfq" });
    } else if (providerId === "qlib_local") {
      await launch("build-market-panel-v7", { symbols: symbols.trim(), start_date: startDate, end_date: endDate, provider_uri: providerUri.trim(), output_root: outputPath.trim(), region: "cn" });
    } else {
      await launch("build-fundamentals-v7", { ...common, start_date: startDate, end_date: endDate, provider: "tushare", fundamentals_root: outputPath.trim(), token_env: "TUSHARE_TOKEN" });
    }
  };

  const inspectCoverage = async (): Promise<void> => {
    setSubmitting(true);
    setActionError("");
    try {
      const result = await apiGet<DataCoverage>("/data/coverage", { path: coveragePath, dateColumn, symbolColumn, deep: deepCoverage });
      setCoverageResult(result.data);
    } catch (error) {
      setCoverageResult(null);
      setActionError(error instanceof Error ? error.message : "coverage inspection failed");
    } finally {
      setSubmitting(false);
    }
  };

  const submitTransfer = async (): Promise<void> => {
    if (!transferSource || !transferOutput) return;
    await launch("data-manager-transfer", {
      operation: transferOperation,
      source: transferSource,
      output: transferOutput,
      date_column: dateColumn,
      symbol_column: symbolColumn,
      start_date: startDate,
      end_date: endDate,
      symbols: symbols.trim(),
    });
  };

  const submitRecorder = async (): Promise<void> => {
    if (!networkApproved || tickflowCredentialMissing || !symbols.trim()) return;
    await launch(recorderMode === "tick" ? "record-tickflow-quotes" : "record-tickflow-depth", {
      symbols: symbols.trim(),
      loop_seconds: loopSeconds,
      max_iterations: maxIterations,
      sleep: 0.05,
      allow_network: true,
    });
  };

  const cancel = async (jobId: string): Promise<void> => {
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
    <section className="data-manager-workspace data-manager-v4">
      <div className="data-manager-toolbar">
        <div><Database size={20} weight="duotone" /><span><strong>DATA MANAGER</strong><small>TickFlow-first · server-side streaming · audited Runtime paths</small></span></div>
        <div>
          <button className="secondary-button" onClick={() => navigate("/settings?job=train&universe=all")}><Play size={15} /> 全宇宙训练</button>
          <button className="secondary-button" onClick={() => navigate("/runtime?view=cleanup")}><ShieldCheck size={15} /> 审计删除</button>
        </div>
      </div>

      <nav className="data-manager-tabs" aria-label="数据管理工作流">
        {TABS.map((item) => <button key={item.id} className={tab === item.id ? "active" : ""} aria-current={tab === item.id ? "page" : undefined} onClick={() => setTab(item.id)}><strong>{item.label}</strong><small>{item.detail}</small></button>)}
      </nav>

      <div className="data-manager-grid">
        <Panel title="数据服务" eyebrow="Runtime-detected capability" className="data-provider-panel">
          <div className="data-provider-list">
            {providers.map((provider) => (
              <button type="button" key={provider.id} className={provider.id === providerId ? "selected" : ""} onClick={() => selectProvider(provider)}>
                <span><strong>{provider.label}</strong><small>{provider.assetClasses.join(" · ")} · {provider.intervals.join(" / ")}</small></span>
                <StatusBadge status={provider.status === "needs_configuration" ? "warning" : provider.status} label={providerLabel(provider)} />
                <p>{provider.note}</p>
              </button>
            ))}
          </div>
        </Panel>

        {tab === "acquire" ? (
          <Panel title="获取与增量更新" eyebrow="Explicit provider contract · no browser upload" className="data-operation-panel">
            {providerId === "tickflow" ? <div className="segmented-control" role="radiogroup" aria-label="TickFlow 数据粒度">{(["daily", "minute", "tick", "depth"] as TickflowMode[]).map((mode) => <label key={mode} className={tickflowMode === mode ? "active" : ""}><input type="radio" name="tickflow-mode" value={mode} checked={tickflowMode === mode} onChange={() => setTickflowMode(mode)} /><span>{mode === "daily" ? "日线（免费）" : mode === "minute" ? "分钟线" : mode === "tick" ? "Tick 行情" : "Level-2 盘口"}</span></label>)}</div> : null}
            <DataFields symbols={symbols} setSymbols={setSymbols} startDate={startDate} setStartDate={setStartDate} endDate={endDate} setEndDate={setEndDate} />
            {providerId === "qlib_local" ? <label className="field-row"><span>Qlib provider URI</span><input value={providerUri} onChange={(event) => setProviderUri(event.target.value)} /></label> : null}
            {!(providerId === "tickflow" && tickflowMode !== "daily") ? <label className="field-row"><span>Runtime 输出</span><input value={outputPath} onChange={(event) => setOutputPath(event.target.value)} /><small>只接受 Runtime 内路径；分钟线和盘口沿用项目已有 canonical 目录。</small></label> : null}
            {needsNetwork ? <NetworkApproval checked={networkApproved} setChecked={setNetworkApproved} /> : <div className="local-provider-notice"><Database size={17} />本地 provider 不访问网络。</div>}
            {selectedModeNeedsKey && tickflowCredentialMissing ? <Notice tone="warning" text="分钟线与 Level-2 需要后端 TICKFLOW_API_KEY；凭据不会进入浏览器。" /> : null}
            <ActionRow text="真实 provider 失败时任务失败，不会回退为 mock。"><button className="primary-button" disabled={!canAcquire || submitting} onClick={submitAcquire}><DownloadSimple size={16} />{submitting ? "提交中…" : "启动任务"}</button></ActionRow>
          </Panel>
        ) : null}

        {tab === "coverage" ? (
          <Panel title="日期覆盖与重复键" eyebrow="Chunked scan · optional disk-bounded exact mode" className="data-operation-panel">
            <label className="field-row"><span>Runtime 数据文件</span><input value={coveragePath} onChange={(event) => setCoveragePath(event.target.value)} placeholder="runtime/.../*.parquet" /><small>直接扫描服务器文件，适用于浏览器无法上传的大型 TickFlow 数据。</small></label>
            <div className="field-grid"><label><span>日期列</span><input value={dateColumn} onChange={(event) => setDateColumn(event.target.value)} /></label><label><span>股票列</span><input value={symbolColumn} onChange={(event) => setSymbolColumn(event.target.value)} /></label></div>
            <label className="check-row"><input type="checkbox" checked={deepCoverage} onChange={(event) => setDeepCoverage(event.target.checked)} /><span><strong>精确跨分片重复检查</strong><small>使用 Runtime 临时 SQLite，内存有界；大文件耗时更长。</small></span></label>
            <ActionRow text="缺口先按工作日标记；交易所节假日需再与 canonical 交易日历核对。"><button className="primary-button" disabled={!coveragePath || submitting} onClick={inspectCoverage}><MagnifyingGlass size={16} />开始检查</button></ActionRow>
            {coverageResult ? <CoverageResult value={coverageResult} /> : null}
          </Panel>
        ) : null}

        {tab === "transfer" ? (
          <Panel title="隔离导入与过滤导出" eyebrow="Quarantine → validate → manifest → Runtime" className="data-operation-panel">
            <div className="segmented-control" role="radiogroup" aria-label="传输操作">{(["import", "export"] as TransferOperation[]).map((operation) => <label key={operation} className={transferOperation === operation ? "active" : ""}><input type="radio" name="transfer-operation" checked={transferOperation === operation} onChange={() => { setTransferOperation(operation); setTransferOutput(operation === "import" ? "runtime/data/imported/validated_import.parquet" : "runtime/exports/filtered_export.parquet"); }} /><span>{operation === "import" ? "隔离导入" : "过滤导出"}</span></label>)}</div>
            <label className="field-row"><span>服务器源文件</span><input list="quarantine-files" value={transferSource} onChange={(event) => setTransferSource(event.target.value)} /><datalist id="quarantine-files">{(quarantine.data?.data ?? []).map((file) => <option key={file.path} value={file.path}>{file.name}</option>)}</datalist><small>{transferOperation === "import" ? `必须先由运维放入 ${registry.data?.data.serverPaths.quarantine ?? "runtime/import_quarantine"}` : "导出源必须位于 Runtime。"}</small></label>
            <label className="field-row"><span>输出文件</span><input value={transferOutput} onChange={(event) => setTransferOutput(event.target.value)} /></label>
            <DataFields symbols={symbols} setSymbols={setSymbols} startDate={startDate} setStartDate={setStartDate} endDate={endDate} setEndDate={setEndDate} compact />
            <ActionRow text="导入会精确去重；导出按股票和日期过滤；均写入 SHA-256 manifest。"><button className="primary-button" disabled={!transferSource || !transferOutput || submitting} onClick={submitTransfer}>{transferOperation === "import" ? <UploadSimple size={16} /> : <DownloadSimple size={16} />}{transferOperation === "import" ? "验证并导入" : "生成导出"}</button></ActionRow>
          </Panel>
        ) : null}

        {tab === "recorder" ? (
          <Panel title="DataRecorder · TickFlow Tick / Level-2" eyebrow="Forward-only streaming recorder" className="data-operation-panel">
            <Notice tone="info" text="Tick 与盘口均为真实前向快照：按日期和时间分片写入 Runtime，避免反复加载整日大文件；不伪造历史成交。" />
            <div className="segmented-control" role="radiogroup" aria-label="DataRecorder 数据类型">{(["tick", "depth"] as RecorderMode[]).map((mode) => <label key={mode} className={recorderMode === mode ? "active" : ""}><input type="radio" name="recorder-mode" checked={recorderMode === mode} onChange={() => setRecorderMode(mode)} /><span>{mode === "tick" ? "Tick 行情快照" : "Level-2 五档盘口"}</span></label>)}</div>
            <label className="field-row"><span>股票范围</span><input value={symbols} onChange={(event) => setSymbols(event.target.value)} /></label>
            <div className="field-grid"><label><span>轮询间隔（秒）</span><input type="number" min={5} value={loopSeconds} onChange={(event) => setLoopSeconds(Number(event.target.value))} /></label><label><span>最大轮次</span><input type="number" min={1} value={maxIterations} onChange={(event) => setMaxIterations(Number(event.target.value))} /></label></div>
            <NetworkApproval checked={networkApproved} setChecked={setNetworkApproved} />
            {tickflowCredentialMissing ? <Notice tone="warning" text="后端未检测到 TICKFLOW_API_KEY，录制按钮已锁定。" /> : null}
            <ActionRow text="任务可取消；权限不足会快速失败并给出明确原因。"><button className="primary-button recorder-button" disabled={!networkApproved || tickflowCredentialMissing || !symbols.trim() || submitting} onClick={submitRecorder}><Record size={16} weight="fill" />开始录制</button></ActionRow>
          </Panel>
        ) : null}
      </div>

      {actionError ? <div className="launch-message launch-error"><WarningCircle size={18} />{actionError}</div> : null}
      {submitted ? <div className="launch-message launch-success"><CheckCircle size={18} />已提交 {submitted.commandId} · {submitted.id}</div> : null}

      <Panel title="数据任务" eyebrow={`${dataJobs.length} governed jobs · refresh 2s`} className="data-job-queue-panel">
        {dataJobs.length ? <div className="table-scroll"><table className="data-table"><thead><tr><th>任务</th><th>状态</th><th>进度</th><th>创建 / 完成</th><th>输出</th><th>操作</th></tr></thead><tbody>{dataJobs.map((job) => <tr key={job.id}><td><strong>{job.commandId}</strong><span className="mono">{job.id}</span></td><td><StatusBadge status={job.status} /></td><td><div className="job-progress"><span style={{ width: `${Math.round((job.progress ?? 0) * 100)}%` }} /><b>{job.progress == null ? "等待 provider" : `${Math.round(job.progress * 100)}%`}</b></div></td><td><strong>{formatDate(job.createdAt)}</strong><span>{job.finishedAt ? formatDate(job.finishedAt) : job.message ?? "—"}</span></td><td>{job.outputPaths.join(" · ") || "等待产物"}</td><td>{["queued", "running", "cancelling"].includes(job.status) ? <button className="secondary-button" onClick={() => cancel(job.id)}><Stop size={14} />取消</button> : "—"}</td></tr>)}</tbody></table></div> : <StateView state={jobs.isError ? "error" : "empty"} detail={jobs.error?.message ?? "尚未提交数据任务。"} />}
        <div className="data-job-queue-footer"><ArrowClockwise size={16} /><span>Provider 日志实时解析为进度；成功后 Runtime Catalog 自动失效并重扫。</span></div>
      </Panel>
    </section>
  );
}

function DataFields({ symbols, setSymbols, startDate, setStartDate, endDate, setEndDate, compact = false }: { symbols: string; setSymbols: (value: string) => void; startDate: string; setStartDate: (value: string) => void; endDate: string; setEndDate: (value: string) => void; compact?: boolean }): JSX.Element {
  return <div className={`field-grid data-fields ${compact ? "compact" : ""}`}><label className="symbol-field"><span>股票范围</span><input value={symbols} onChange={(event) => setSymbols(event.target.value)} placeholder="000001.SZ,600519.SH" /></label><label><span>开始日期</span><input type="date" value={startDate} onChange={(event) => setStartDate(event.target.value)} /></label><label><span>结束日期</span><input type="date" value={endDate} onChange={(event) => setEndDate(event.target.value)} /></label></div>;
}

function NetworkApproval({ checked, setChecked }: { checked: boolean; setChecked: (value: boolean) => void }): JSX.Element {
  return <label className="check-row network-confirmation"><input type="checkbox" checked={checked} onChange={(event) => setChecked(event.target.checked)} /><span><strong>确认允许本次真实 provider 任务访问网络</strong><small>后端只运行白名单命令；不接收 shell、URL 或浏览器凭据。</small></span></label>;
}

function ActionRow({ text, children }: { text: string; children: JSX.Element }): JSX.Element {
  return <div className="data-job-actions"><span><ShieldCheck size={16} />{text}</span>{children}</div>;
}

function Notice({ tone, text }: { tone: "info" | "warning"; text: string }): JSX.Element {
  return <div className={`data-notice ${tone}`}><WarningCircle size={17} /><span>{text}</span></div>;
}

function CoverageResult({ value }: { value: DataCoverage }): JSX.Element {
  const metrics = [["有效行", formatCompact(value.scannedKeyRows)], ["股票", formatCompact(value.symbolCount)], ["日期", `${value.dateStart ?? "—"} → ${value.dateEnd ?? "—"}`], ["重复键", formatCompact(value.duplicateKeys)], ["疑似缺口", formatCompact(value.missingBusinessDayCount)]];
  return <section className="coverage-result" aria-label="覆盖检查结果"><div className="coverage-metrics">{metrics.map(([label, metric]) => <div key={label}><span>{label}</span><strong>{metric}</strong></div>)}</div><p>{value.duplicateMode === "exact" ? "已完成跨分片精确去重检查" : "当前仅统计分片内重复；可启用精确模式"} · {value.columns.length} columns · {(value.sizeBytes / 1024 / 1024).toFixed(1)} MB</p>{value.missingBusinessDayCandidates.length ? <details><summary>查看前 {value.missingBusinessDayCandidates.length} 个疑似缺口</summary><code>{value.missingBusinessDayCandidates.join(" · ")}</code></details> : null}</section>;
}
