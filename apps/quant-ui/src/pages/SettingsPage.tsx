import { useMemo, useState } from "react";
import {
  CheckCircle,
  Database,
  DesktopTower,
  Flask,
  HardDrives,
  Lock,
  Play,
  ShieldCheck,
  TerminalWindow,
  WarningCircle,
} from "@phosphor-icons/react";
import { apiPost } from "../api/client";
import type { SystemOverview } from "../api/types";
import { useApi } from "../hooks/useApi";
import { Panel } from "../components/Panel";
import { StateView } from "../components/StateView";
import { StatusBadge } from "../components/StatusBadge";
import { formatBytes, formatDate } from "../utils/format";

interface Job {
  id: string;
  type: string;
  status: string;
  commandId: string;
  createdAt: string;
  startedAt?: string | null;
  finishedAt?: string | null;
  progress?: number | null;
  message?: string | null;
  outputPaths: string[];
  error?: string | null;
}

const jobTemplates = {
  backtest: {
    commandId: "run-strict-a-share-backtest-v8",
    parameters: {
      target_weights_path: "runtime/reports/v8/deep/v89_rankfix_20260613_1044/short_5d/target_weights.parquet",
      market_panel_path: "runtime/data/v7/silver/market_panel/market_panel.parquet",
      output_dir: "runtime/reports/quant_ui_jobs/web_backtest",
      initial_cash: 1000000,
      slippage_bps: 8,
    },
  },
  train: {
    commandId: "train-v8-deep",
    parameters: {
      horizon_class: "short",
      dataset_path: "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus8.parquet",
      silver_panel_path: "runtime/data/v7/silver/market_panel/market_panel.parquet",
      output_dir: "runtime/reports/quant_ui_jobs/web_train",
      max_epochs: 20,
      require_gpu: true,
    },
  },
  infer: {
    commandId: "predict-alpha-v7",
    parameters: {
      model_dir: "runtime/reports/v8/deep/v89_rankfix_20260613_1044/short_5d/ft",
      feature_dataset: "runtime/data/v7/gold/training_dataset/training_dataset_alpha181_exec_v89_plus8.parquet",
      output: "runtime/predictions/quant_ui_web_predictions.parquet",
      primary_horizon: 5,
    },
  },
} as const;

type JobType = keyof typeof jobTemplates;

export function SettingsPage(): JSX.Element {
  const overview = useApi<SystemOverview>(["settings-overview"], "/system/overview");
  const jobs = useApi<Job[]>(["settings-jobs"], "/jobs");
  const [jobType, setJobType] = useState<JobType>("backtest");
  const [jobJson, setJobJson] = useState(() => JSON.stringify(jobTemplates.backtest, null, 2));
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState("");
  const [launchedJob, setLaunchedJob] = useState<Job | null>(null);
  const data = overview.data?.data;

  const activeJobs = useMemo(
    () => (jobs.data?.data ?? []).filter((job) => ["queued", "running"].includes(job.status)).length,
    [jobs.data?.data],
  );

  if (overview.isLoading) return <StateView state="loading" />;

  const selectTemplate = (type: JobType): void => {
    setJobType(type);
    setJobJson(JSON.stringify(jobTemplates[type], null, 2));
    setLaunchError("");
  };

  const launch = async (): Promise<void> => {
    setLaunching(true);
    setLaunchError("");
    try {
      const payload = JSON.parse(jobJson) as { commandId: string; parameters: Record<string, unknown> };
      const result = await apiPost<Job>(`/jobs/${jobType}`, payload);
      setLaunchedJob(result.data);
      await jobs.refetch();
    } catch (error) {
      setLaunchError(error instanceof Error ? error.message : "job launch failed");
    } finally {
      setLaunching(false);
    }
  };

  return (
    <div className="page control-center-page">
      <section className="control-hero">
        <div>
          <span className="page-kicker">QUANTAGENT CONTROL PLANE</span>
          <h2>控制中心</h2>
          <p>一个进程启动 Web 与 API；在页面内启动 allowlisted research jobs、查看状态与输出。</p>
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
            <div><span>项目根目录执行</span><code>./scripts/run_quant_ui.sh</code></div>
            <StatusBadge status="ready" label="localhost:8000" />
          </div>
          <div className="settings-list compact-settings">
            <SettingRow icon={DesktopTower} label="Unified Server" value="/api + React SPA on 127.0.0.1:8000" status={overview.isError ? "error" : "ready"} />
            <SettingRow icon={Database} label="Runtime Index" value={`${data?.runtime.artifactCount ?? 0} artifacts · ${formatBytes(data?.runtime.totalSizeBytes)}`} status={data ? "ready" : "partial"} />
            <SettingRow icon={ShieldCheck} label="Safety Mode" value="Research only · no live order route" status="ready" />
            <SettingRow icon={Lock} label="Path Policy" value="Inputs project-only · outputs runtime-only" status="ready" />
          </div>
        </Panel>

        <Panel title="网页启动研究任务" eyebrow="Allowlisted CLI adapter · edit before launch" className="job-launcher-panel">
          <div className="job-type-tabs">
            <button className={jobType === "backtest" ? "active" : ""} onClick={() => selectTemplate("backtest")}><Flask size={16} /> 回测</button>
            <button className={jobType === "train" ? "active" : ""} onClick={() => selectTemplate("train")}><HardDrives size={16} /> 训练</button>
            <button className={jobType === "infer" ? "active" : ""} onClick={() => selectTemplate("infer")}><Play size={16} /> 推理</button>
          </div>
          <textarea className="job-json-editor mono" value={jobJson} onChange={(event) => setJobJson(event.target.value)} spellCheck={false} />
          <div className="job-launch-footer">
            <div>
              <ShieldCheck size={17} />
              <span>不接受 shell command；backend 会验证 command ID、参数和路径。</span>
            </div>
            <button className="primary-button" disabled={launching} onClick={launch}><Play size={16} weight="fill" /> {launching ? "提交中…" : "启动任务"}</button>
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
