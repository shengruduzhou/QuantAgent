import { useMemo, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  Bell,
  CaretDown,
  CaretUp,
  ChartBar,
  ListMagnifyingGlass,
  Pulse,
  TerminalWindow,
  WarningCircle,
  XCircle,
} from "@phosphor-icons/react";
import { apiPost } from "../../api/client";
import type { JobSummary, SystemOverview } from "../../api/types";
import { useApi } from "../../hooks/useApi";
import type { JobEventStreamState } from "../../hooks/useJobEvents";
import { formatBytes, formatDate } from "../../utils/format";
import type { WorkspaceDockTab } from "../workspace/types";

const dockTabs: Array<{ id: WorkspaceDockTab; label: string; icon: typeof Bell }> = [
  { id: "tasks", label: "Tasks", icon: ListMagnifyingGlass },
  { id: "logs", label: "Logs", icon: TerminalWindow },
  { id: "alerts", label: "Alerts", icon: WarningCircle },
  { id: "events", label: "Events", icon: Pulse },
  { id: "resources", label: "Resources", icon: ChartBar },
];

interface OperationsDockProps {
  open: boolean;
  tab: WorkspaceDockTab;
  size: number;
  jobs: JobSummary[];
  overview?: SystemOverview;
  realtime: JobEventStreamState;
  setTab: (tab: WorkspaceDockTab) => void;
  toggle: () => void;
  setSize: (size: number) => void;
  openPath: (path: string) => void;
}

export function OperationsDock({ open, tab, size, jobs, overview, realtime, setTab, toggle, setSize, openPath }: OperationsDockProps): JSX.Element {
  const queryClient = useQueryClient();
  const [selectedJobId, setSelectedJobId] = useState(jobs[0]?.id ?? "");
  const [cancelError, setCancelError] = useState("");
  const selectedJob = jobs.find((job) => job.id === selectedJobId) ?? jobs[0];
  const logs = useApi<string[]>(["vnext-job-logs", selectedJob?.id], open && tab === "logs" && selectedJob ? `/jobs/${selectedJob.id}/logs` : null, { limit: 600 }, { refetchInterval: selectedJob && ["queued", "running"].includes(selectedJob.status) ? 2_000 : false });
  const activeCount = jobs.filter((job) => ["queued", "running", "cancelling"].includes(job.status)).length;
  const failedJobs = jobs.filter((job) => job.status === "failed");
  const riskEventCount = Object.values(overview?.risk.eventCounts ?? {}).reduce((sum, count) => sum + count, 0);
  const alertCount = failedJobs.length + riskEventCount;

  const alerts = useMemo(() => [
    ...failedJobs.map((job) => ({ id: job.id, severity: "error", title: job.commandId, detail: job.error ?? job.message ?? "Job failed", path: `/training?job=${job.id}` })),
    ...(riskEventCount ? [{ id: "risk-events", severity: "warning", title: `${riskEventCount} active risk events`, detail: "Open the Risk Manager for exact rules and thresholds.", path: "/risk" }] : []),
  ], [failedJobs, riskEventCount]);

  const cancelJob = async (job: JobSummary): Promise<void> => {
    setCancelError("");
    try {
      await apiPost(`/jobs/${job.id}/cancel`, {});
      await queryClient.invalidateQueries({ queryKey: ["global-activity-jobs"] });
    } catch (error) {
      setCancelError(error instanceof Error ? error.message : "Cancel failed");
    }
  };

  const beginResize = (event: React.PointerEvent<HTMLDivElement>): void => {
    event.currentTarget.setPointerCapture(event.pointerId);
    const move = (pointer: PointerEvent): void => setSize(window.innerHeight - pointer.clientY - 28);
    const end = (): void => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", end);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", end);
  };

  return (
    <section className={`vnext-operations-dock ${open ? "open" : "collapsed"}`} style={{ height: open ? `${size}px` : "32px" }} aria-label="Operations Dock">
      {open ? <div className="vnext-dock-resizer" onPointerDown={beginResize} role="separator" aria-orientation="horizontal" aria-label="调整 Operations Dock 高度" /> : null}
      <header>
        <div className="vnext-dock-tabs">
          {dockTabs.map(({ id, label, icon: Icon }) => (
            <button type="button" key={id} className={tab === id && open ? "active" : ""} onClick={() => setTab(id)}>
              <Icon size={14} /> {label}
              {id === "tasks" && activeCount ? <em>{activeCount}</em> : null}
              {id === "alerts" && alertCount ? <em className="warning">{alertCount}</em> : null}
            </button>
          ))}
        </div>
        <div className="vnext-dock-summary">
          <span className={`state-${realtime.status === "live" ? "ready" : "warning"}`}>WS {realtime.status}</span>
          <span>Runtime {overview?.runtime.artifactCount?.toLocaleString() ?? "—"}</span>
          <span>Safety <strong>NO LIVE ORDERS</strong></span>
          <button type="button" onClick={toggle} aria-label={open ? "收起 Operations Dock" : "展开 Operations Dock"}>{open ? <CaretDown size={15} /> : <CaretUp size={15} />}</button>
        </div>
      </header>

      {open ? (
        <div className="vnext-dock-body">
          {tab === "tasks" ? (
            <div className="vnext-task-list">
              {jobs.length ? jobs.slice(0, 80).map((job) => (
                <div key={job.id} className={`vnext-task-row ${job.id === selectedJob?.id ? "active" : ""}`}>
                  <button type="button" className="vnext-task-main" onClick={() => setSelectedJobId(job.id)} onDoubleClick={() => openPath(`/training?job=${job.id}`)}>
                    <span className={`vnext-job-state state-${job.status}`} />
                    <span><strong>{job.commandId}</strong><small>{job.id} · {formatDate(job.createdAt)}</small></span>
                    <em>{job.progress === null || job.progress === undefined ? "—" : `${Math.round(job.progress * 100)}%`}</em>
                    <b>{job.status}</b>
                  </button>
                  {["queued", "running", "cancelling"].includes(job.status) ? <button type="button" className="vnext-cancel-job" onClick={() => void cancelJob(job)}><XCircle size={14} /> Cancel</button> : <span />}
                </div>
              )) : <p className="vnext-dock-empty">没有持久化任务。任务状态不会由前端模拟。</p>}
              {cancelError ? <p className="vnext-dock-error">{cancelError}</p> : null}
            </div>
          ) : null}

          {tab === "logs" ? (
            <div className="vnext-log-viewer">
              <aside>{jobs.slice(0, 40).map((job) => <button type="button" key={job.id} className={job.id === selectedJob?.id ? "active" : ""} onClick={() => setSelectedJobId(job.id)}>{job.commandId}<small>{job.status}</small></button>)}</aside>
              <pre>{logs.data?.data.length ? logs.data.data.join("\n") : selectedJob ? "No persisted log lines for this job." : "Select a persisted job."}</pre>
            </div>
          ) : null}

          {tab === "alerts" ? (
            <div className="vnext-alert-list">
              {alerts.length ? alerts.map((alert) => <button type="button" key={alert.id} onClick={() => openPath(alert.path)}><WarningCircle size={16} /><span><strong>{alert.title}</strong><small>{alert.detail}</small></span><em>{alert.severity}</em></button>) : <p className="vnext-dock-empty">当前没有失败任务或已记录风险事件。</p>}
            </div>
          ) : null}

          {tab === "events" ? (
            <div className="vnext-event-state"><Pulse size={26} /><span><strong>Typed Event Stream</strong><small>Status: {realtime.status}</small><small>Last event: {realtime.lastEventAt ?? "not received"}</small></span></div>
          ) : null}

          {tab === "resources" ? (
            <div className="vnext-resource-grid">
              <span><small>Runtime size</small><strong>{formatBytes(overview?.runtime.totalSizeBytes)}</strong></span>
              <span><small>Artifacts</small><strong>{overview?.runtime.artifactCount?.toLocaleString() ?? "—"}</strong></span>
              <span><small>Runs</small><strong>{overview?.runtime.runCount ?? "—"}</strong></span>
              <span><small>Manifest coverage</small><strong>{overview?.runtime.manifestCoverage === undefined ? "UNAVAILABLE" : `${Math.round(overview.runtime.manifestCoverage * 100)}%`}</strong></span>
              <p>GPU/CPU/RAM telemetry is not exposed by the current API and is intentionally marked unavailable.</p>
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
