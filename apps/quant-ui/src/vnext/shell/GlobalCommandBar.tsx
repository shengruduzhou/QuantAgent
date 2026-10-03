import {
  ArrowsInLineHorizontal,
  CaretDown,
  CircleNotch,
  Database,
  Gear,
  HardDrives,
  LockKey,
  MagnifyingGlass,
  ShieldCheck,
  ShieldWarning,
  SidebarSimple,
  UserCircle,
  WarningCircle,
  WifiHigh,
  WifiSlash,
} from "@phosphor-icons/react";
import type { PaperAccount } from "../../api/paperAccount";
import type { JobSummary, SystemOverview } from "../../api/types";
import type { JobEventStreamState } from "../../hooks/useJobEvents";
import type { WorkspaceTab } from "../workspace/types";
import type { WorkspaceTheme } from "../workspace/types";
import { ThemeSwitcher } from "./ThemeSwitcher";

interface GlobalCommandBarProps {
  activeTab: WorkspaceTab;
  overview?: SystemOverview;
  apiState: "loading" | "ready" | "error";
  /** `/api/paper/account`; undefined while loading or when it failed. */
  paperAccount?: PaperAccount;
  paperAccountState: "loading" | "ready" | "error";
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
  paperAccount,
  paperAccountState,
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
        <span className={`vnext-status-chip vnext-collapsible-chip state-${realtime.status === "live" ? "ready" : "warning"}`} title={`WebSocket ${realtime.status}`}>
          {realtime.status === "live" ? <WifiHigh size={14} /> : <WifiSlash size={14} />}
          WS {realtime.status.toUpperCase()}
        </span>
        <button type="button" className="vnext-status-button vnext-collapsible-chip" onClick={() => openPath("/settings?view=jobs")} title="打开任务中心">
          <HardDrives size={14} /> {activeJobs} JOBS
        </button>
        <button type="button" className={`vnext-status-button ${riskEvents ? "warning" : "safe"}`} onClick={() => openPath("/risk")} title={`回测风险事件（非纸面账户）：${overview?.risk.backtestName ?? overview?.risk.backtestId ?? "unknown backtest"}`}>
          {riskEvents ? <WarningCircle size={14} /> : <ShieldCheck size={14} />}
          {/* No overview yet means risk is unmeasured, not clear. */}
          {/* Persisted events of one research backtest (e.g. skipped orders), not
              the paper account's risk - labelled so it is never read as a live alarm. */}
          BT EVENTS {!overview ? "UNKNOWN" : riskEvents ? (overview.risk.eventCountsExact === false ? `≥${riskEvents}` : riskEvents) : "NONE"}
        </button>
        <LiveChip account={paperAccount} state={paperAccountState} />
        <KillSwitchChip account={paperAccount} state={paperAccountState} onOpen={() => openPath("/t-plus-one")} />
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

/**
 * Both safety chips are read from `/api/paper/account`. They used to be one
 * string literal, "KILL LOCKED", painted green whatever the venue said — so a
 * tripped (or never-attached) kill switch read exactly like a healthy one.
 */
function LiveChip({ account, state }: { account?: PaperAccount; state: "loading" | "ready" | "error" }): JSX.Element {
  const live = account?.mode?.live_trading_available;
  if (state !== "ready" || live === undefined) {
    return <span className="vnext-status-chip state-warning vnext-safety-chip" title="Paper policy unavailable: live-trading state unknown">{state === "loading" ? <CircleNotch size={14} className="spin" /> : <WarningCircle size={14} />} LIVE UNKNOWN</span>;
  }
  return live
    ? <span className="vnext-status-chip state-error vnext-safety-chip" title={account?.mode?.banner}><WarningCircle size={14} /> LIVE AVAILABLE</span>
    : <span className="vnext-status-chip state-ready vnext-safety-chip" title={account?.mode?.banner ?? "Live trading is disabled by policy"}><LockKey size={14} /> LIVE DISABLED</span>;
}

function KillSwitchChip({ account, state, onOpen }: { account?: PaperAccount; state: "loading" | "ready" | "error"; onOpen: () => void }): JSX.Element {
  const risk = account?.riskState;
  let label: string;
  let tone: "ready" | "warning" | "error";
  let title: string;
  // Only an explicit boolean is a measurement: a payload without
  // killSwitch.active (or riskEngineAttached) is unknown, never "armed".
  if (state !== "ready" || !risk || typeof risk.killSwitch?.active !== "boolean"
      || typeof risk.riskEngineAttached !== "boolean") {
    label = "KILL UNKNOWN";
    tone = "warning";
    title = "Paper account unavailable: kill-switch state unknown";
  } else if (risk.killSwitch.active) {
    label = `KILL ACTIVE · ${risk.killSwitch.scope ?? "?"}`;
    tone = "error";
    title = `${risk.killSwitch.reason ?? "no reason recorded"}${risk.killSwitch.reduceOnly ? " (reduce-only)" : ""}`;
  } else if (!risk.riskEngineAttached) {
    label = "KILL UNARMED";
    tone = "error";
    title = risk.reasons.riskEngine ?? "no risk engine attached";
  } else {
    label = "KILL ARMED";
    tone = "ready";
    title = "Risk engine attached; kill switch not tripped";
  }
  return (
    <button type="button" className={`vnext-status-chip vnext-status-button state-${tone} vnext-safety-chip`} onClick={onOpen} title={title}>
      {tone === "ready" ? <ShieldCheck size={14} /> : tone === "error" ? <ShieldWarning size={14} /> : <WarningCircle size={14} />} {label}
    </button>
  );
}
