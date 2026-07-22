import {
  ArrowsInLineHorizontal,
  CaretDown,
  CircleNotch,
  Database,
  Gear,
  HardDrives,
  MagnifyingGlass,
  ShieldCheck,
  SidebarSimple,
  UserCircle,
  WarningCircle,
  WifiHigh,
  WifiSlash,
} from "@phosphor-icons/react";
import type { JobSummary, SystemOverview } from "../../api/types";
import type { JobEventStreamState } from "../../hooks/useJobEvents";
import type { WorkspaceTab } from "../workspace/types";
import type { WorkspaceTheme } from "../workspace/types";
import { ThemeSwitcher } from "./ThemeSwitcher";

interface GlobalCommandBarProps {
  activeTab: WorkspaceTab;
  overview?: SystemOverview;
  apiState: "loading" | "ready" | "error";
  jobs: JobSummary[];
  realtime: JobEventStreamState;
  railExpanded: boolean;
  density: "compact" | "comfortable";
  theme: WorkspaceTheme;
  onToggleRail: () => void;
  onOpenCommand: () => void;
  onToggleDensity: () => void;
  onSetTheme: (theme: WorkspaceTheme) => void;
  openPath: (path: string) => void;
}

export function GlobalCommandBar({
  activeTab,
  overview,
  apiState,
  jobs,
  realtime,
  railExpanded,
  density,
  theme,
  onToggleRail,
  onOpenCommand,
  onToggleDensity,
  onSetTheme,
  openPath,
}: GlobalCommandBarProps): JSX.Element {
  const activeJobs = jobs.filter((job) => ["queued", "running", "cancelling"].includes(job.status)).length;
  const riskEvents = Object.values(overview?.risk.eventCounts ?? {}).reduce((sum, count) => sum + count, 0);
  const contexts = Object.entries(activeTab.context).filter((entry): entry is [string, string] => Boolean(entry[1]));

  return (
    <header className="vnext-commandbar">
      <div className="vnext-brand-cluster">
        <button type="button" className="vnext-icon-button" onClick={onToggleRail} aria-label={railExpanded ? "收起模块栏" : "展开模块栏"}>
          <SidebarSimple size={19} weight="duotone" />
        </button>
        <button type="button" className="vnext-workspace-name" onClick={onOpenCommand}>
          <span>QUANTAGENT</span>
          <strong>{activeTab.title}</strong>
          <CaretDown size={12} />
        </button>
      </div>

      <button type="button" className="vnext-global-search" onClick={onOpenCommand} aria-label="打开全局实体与命令搜索">
        <MagnifyingGlass size={17} />
        <span>搜索股票、因子、模型、Experiment、Run、Artifact 或命令</span>
        <kbd>⌘K</kbd>
      </button>

      <div className="vnext-context-strip" aria-label="当前工作区上下文">
        {contexts.length ? contexts.slice(0, 3).map(([key, value]) => (
          <span key={key}><small>{key}</small><strong>{value}</strong></span>
        )) : <span><small>context</small><strong>GLOBAL</strong></span>}
        <span><small>as-of</small><strong>{overview?.runtime.indexedAt?.slice(0, 10) ?? "UNAVAILABLE"}</strong></span>
      </div>

      <div className="vnext-system-strip" aria-label="系统与安全状态">
        <span className={`vnext-status-chip state-${apiState}`} title="Quant API">
          {apiState === "loading" ? <CircleNotch size={14} className="spin" /> : apiState === "ready" ? <Database size={14} /> : <WarningCircle size={14} />}
          API {apiState.toUpperCase()}
        </span>
        <span className={`vnext-status-chip state-${realtime.status === "live" ? "ready" : "warning"}`} title={`WebSocket ${realtime.status}`}>
          {realtime.status === "live" ? <WifiHigh size={14} /> : <WifiSlash size={14} />}
          WS {realtime.status.toUpperCase()}
        </span>
        <button type="button" className="vnext-status-button" onClick={() => openPath("/settings?view=jobs")} title="打开任务中心">
          <HardDrives size={14} /> {activeJobs} JOBS
        </button>
        <button type="button" className={`vnext-status-button ${riskEvents ? "warning" : "safe"}`} onClick={() => openPath("/risk")} title="打开风险管理">
          {riskEvents ? <WarningCircle size={14} /> : <ShieldCheck size={14} />}
          RISK {riskEvents || "CLEAR"}
        </button>
        <span className="vnext-status-chip state-ready" title="Live trading is disabled by policy"><ShieldCheck size={14} /> KILL LOCKED</span>
        <ThemeSwitcher theme={theme} onChange={onSetTheme} />
        <button type="button" className="vnext-icon-button" onClick={onToggleDensity} aria-label={`切换为${density === "compact" ? "舒适" : "紧凑"}密度`} title={`Density: ${density}`}>
          <ArrowsInLineHorizontal size={17} />
        </button>
        <button type="button" className="vnext-icon-button" onClick={() => openPath("/settings")} aria-label="打开系统设置"><Gear size={17} /></button>
        <button type="button" className="vnext-icon-button" onClick={() => openPath("/help")} aria-label="打开用户与帮助"><UserCircle size={18} /></button>
      </div>
    </header>
  );
}
