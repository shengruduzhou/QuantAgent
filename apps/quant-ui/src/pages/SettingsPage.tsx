import { useEffect, useMemo, useState } from "react";
import {
  CheckCircle,
  Atom,
  Database,
  DesktopTower,
  Flask,
  GlobeHemisphereEast,
  HardDrives,
  Lock,
  Play,
  ShieldCheck,
  TerminalWindow,
  WarningCircle,
} from "@phosphor-icons/react";
import { useSearchParams } from "react-router-dom";
import { apiPost } from "../api/client";
import type { JobSummary, SystemOverview } from "../api/types";
import { useApi } from "../hooks/useApi";
import { Panel } from "../components/Panel";
import { StateView } from "../components/StateView";
import { StatusBadge } from "../components/StatusBadge";
import { formatBytes, formatDate } from "../utils/format";
import { isJobType, templateJson, type JobType } from "../domain/jobTemplates";

export function SettingsPage(): JSX.Element {
  const [searchParams] = useSearchParams();
  const requestedJob = searchParams.get("job");
  const requestedUniverse = searchParams.get("universe");
  const initialType: JobType = isJobType(requestedJob) ? requestedJob : "backtest";
  const overview = useApi<SystemOverview>(["settings-overview"], "/system/overview");
  const jobs = useApi<JobSummary[]>(["settings-jobs"], "/jobs");
  const [jobType, setJobType] = useState<JobType>(initialType);
  const [jobJson, setJobJson] = useState(() => templateJson(initialType));
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState("");
  const [launchedJob, setLaunchedJob] = useState<JobSummary | null>(null);
  const data = overview.data?.data;

  const activeJobs = useMemo(
    () => (jobs.data?.data ?? []).filter((job) => ["queued", "running"].includes(job.status)).length,
    [jobs.data?.data],
  );

  const selectTemplate = (type: JobType): void => {
    setJobType(type);
    setJobJson(templateJson(type));
    setLaunchError("");
    setLaunchedJob(null);
  };

  useEffect(() => {
    if (!isJobType(requestedJob)) return;
    selectTemplate(requestedJob);
  }, [requestedJob, requestedUniverse]);

  if (overview.isLoading) return <StateView state="loading" />;

  const launch = async (): Promise<void> => {
    setLaunching(true);
    setLaunchError("");
    try {
      const payload = JSON.parse(jobJson) as { commandId: string; parameters: Record<string, unknown> };
      const result = await apiPost<JobSummary>(`/jobs/${jobType}`, payload);
      setLaunchedJob(result.data);
      await jobs.refetch();
    } catch (error) {
      setLaunchError(error instanceof Error ? error.message : "job launch failed");
    } finally {
      setLaunching(false);
    }
  };

  return (
    <div className="page institutional-workbench control-center-page control-center-v2">
      <section className="control-hero">
        <div>
          <span className="page-kicker">QUANTAGENT CONTROL PLANE</span>
          <h2>控制中心</h2>
          <p>从网页提交经过 allowlist、参数和路径校验的研究任务。训练、推理和回测都由后台任务执行，不在浏览器线程中运行。</p>
        </div>
        <div className="control-health">
          <span><i className="health-dot" /> API ready</span>
          <strong>{activeJobs}</strong>
          <small>active jobs</small>
        </div>
      </section>

      <section className="settings-grid control-grid">
        <Panel title="一键启动" eyebrow="Single process · integrated static UI" className="startup-panel">
          <div className="startup-command">
            <TerminalWindow size={24} weight="duotone" />
            <div><span>项目根目录执行</span><code>./scripts/run_quant_ui.sh --runtime /path/to/runtime</code></div>
            <StatusBadge status="ready" label="localhost:8000" />
          </div>
          <div className="settings-list compact-settings">
            <SettingRow icon={DesktopTower} label="Unified Server" value="/api + React SPA on 127.0.0.1:8000" status={overview.isError ? "error" : "ready"} />
            <SettingRow icon={Database} label="Runtime Index" value={`${data?.runtime.artifactCount ?? 0} artifacts · ${formatBytes(data?.runtime.totalSizeBytes)}`} status={data ? "ready" : "partial"} />
            <SettingRow icon={ShieldCheck} label="Safety Mode" value="Research only · no live order route" status="ready" />
            <SettingRow icon={Lock} label="Path Policy" value="Inputs project-only · outputs runtime-only" status="ready" />
          </div>
        </Panel>

        <Panel title="网页启动研究任务" eyebrow="Allowlisted CLI adapter · validate before launch" className="job-launcher-panel">
          <div className="job-type-tabs">
            <button className={jobType === "backtest" ? "active" : ""} onClick={() => selectTemplate("backtest")}><Flask size={16} /> 回测</button>
            <button className={jobType === "train" ? "active" : ""} onClick={() => selectTemplate("train")}><HardDrives size={16} /> 训练</button>
            <button className={jobType === "infer" ? "active" : ""} onClick={() => selectTemplate("infer")}><Play size={16} /> 推理</button>
            <button className={jobType === "factor-discovery" ? "active" : ""} onClick={() => selectTemplate("factor-discovery")}><Atom size={16} /> 因子发现</button>
          </div>

          {jobType === "train" ? (
            <div className="training-scope-banner" role="status">
              <GlobeHemisphereEast size={21} weight="duotone" />
              <span>
                <strong>全宇宙训练 · ALL SYMBOLS IN DATASET</strong>
                <small>模板不传 `symbols` 或 `symbols_file`，因此训练数据集中的全部股票都会参与。`feature_policy: judgment` 只采用已生成判断清单中的接受因子；清单缺失时任务会失败关闭，而不是回退并伪装成功。</small>
              </span>
              <StatusBadge status="warning" label="GPU / HIGH COST" />
            </div>
          ) : null}

          {jobType === "factor-discovery" ? (
            <div className="training-scope-banner" role="status">
              <Atom size={21} weight="duotone" />
              <span>
                <strong>受治理的因子发现 · RESEARCH CANDIDATES ONLY</strong>
                <small>先运行 PIT/表达式/相关性验证，再由人工复核。默认不调用网络 LLM；只有同时开启 use_llm 与 allow_network 才会允许模型提案。</small>
              </span>
              <StatusBadge status="warning" label="HUMAN GATE" />
            </div>
          ) : null}

          <textarea className="job-json-editor mono" value={jobJson} onChange={(event) => setJobJson(event.target.value)} spellCheck={false} />
          <div className="job-launch-footer">
            <div>
              <ShieldCheck size={17} />
              <span>不接受 shell command；backend 会验证 command ID、参数、输入存在性和 runtime 输出范围。</span>
            </div>
            <button className="primary-button" disabled={launching} onClick={launch}><Play size={16} weight="fill" /> {launching ? "提交中…" : jobType === "train" ? "确认并启动全宇宙训练" : "启动任务"}</button>
          </div>
          {launchError ? <div className="launch-message launch-error"><WarningCircle size={18} /><span>{launchError}</span></div> : null}
          {launchedJob ? <div className="launch-message launch-success"><CheckCircle size={18} /><span>已提交 {launchedJob.commandId} · {launchedJob.id}</span></div> : null}
        </Panel>

        <Panel title="研究任务队列" eyebrow="Queued · running · completed · failed" className="settings-jobs">
          {(jobs.data?.data ?? []).length ? (
            <div className="table-scroll">
              <table className="data-table">
                <thead><tr><th>任务</th><th>类型</th><th>状态</th><th>创建时间</th><th>输出</th><th>消息</th></tr></thead>
                <tbody>{jobs.data?.data.map((job) => (
                  <tr key={job.id}>
                    <td><strong>{job.commandId}</strong><span className="mono">{job.id}</span></td>
                    <td>{job.type}</td>
                    <td><StatusBadge status={job.status} /></td>
                    <td className="mono">{formatDate(job.createdAt)}</td>
                    <td>{job.outputPaths.join(", ") || "暂无"}</td>
                    <td>{job.error ?? job.message ?? "—"}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          ) : <StateView state="empty" detail="尚未从 UI/API 启动 research job。" />}
        </Panel>
      </section>
    </div>
  );
}

function SettingRow({
  icon: Icon,
  label,
  value,
  status,
}: {
  icon: typeof DesktopTower;
  label: string;
  value: string;
  status: string;
}): JSX.Element {
  return (
    <div>
      <Icon size={20} weight="duotone" />
      <span><strong>{label}</strong><small>{value}</small></span>
      <StatusBadge status={status} />
    </div>
  );
}
