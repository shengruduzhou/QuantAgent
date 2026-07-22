import { useCallback, useEffect, useMemo, useState, type FormEvent } from "react";
import {
  Bell,
  CaretDown,
  Database,
  ListMagnifyingGlass,
  MagnifyingGlass,
  Pulse,
  ShieldCheck,
  SidebarSimple,
  TrendUp,
  Warning,
  X,
} from "@phosphor-icons/react";
import { NavLink, Outlet, useLocation, useNavigate } from "react-router-dom";
import { useApi } from "../hooks/useApi";
import { useJobEvents } from "../hooks/useJobEvents";
import type { JobSummary, SystemOverview } from "../api/types";
import { moduleForPath, moduleGroups, workstationModules } from "../workstation/modules";
import { useWorkspaceLayout } from "../workstation/useWorkspaceLayout";
import { formatDate } from "../utils/format";
import { CommandPalette } from "./CommandPalette";
import { StateView } from "./StateView";
import { StatusBadge } from "./StatusBadge";

export function AppShell(): JSX.Element {
  const [search, setSearch] = useState("");
  const [paletteOpen, setPaletteOpen] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const page = moduleForPath(location.pathname);
  const layout = useWorkspaceLayout();
  const jobEvents = useJobEvents(layout.activityOpen);
  const overview = useApi<SystemOverview>(
    ["system-overview-shell"],
    "/system/overview",
    undefined,
    { refetchInterval: 15_000, staleTime: 10_000 },
  );
  const jobs = useApi<JobSummary[]>(
    ["global-activity-jobs"],
    layout.activityOpen ? "/jobs" : null,
    undefined,
    { refetchInterval: jobEvents.status === "live" ? false : 5_000, staleTime: 2_000 },
  );
  const data = overview.data?.data;
  const activeJobs = useMemo(
    () => (jobs.data?.data ?? []).filter((job) => ["queued", "running"].includes(job.status)).length,
    [jobs.data?.data],
  );
  const closePalette = useCallback(() => setPaletteOpen(false), []);

  useEffect(() => {
    const openPalette = (event: KeyboardEvent): void => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setPaletteOpen(true);
      }
    };
    window.addEventListener("keydown", openPalette);
    return () => window.removeEventListener("keydown", openPalette);
  }, []);

  const submitSearch = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    const value = search.trim();
    if (!value) {
      setPaletteOpen(true);
      return;
    }
    const stock = value.toUpperCase().match(/^(\d{6})(?:\.(SZ|SH|BJ))?$/);
    if (stock) {
      const suffix = stock[2] ?? (stock[1].startsWith("6") ? "SH" : "SZ");
      navigate(`/stock-replay?symbol=${stock[1]}.${suffix}`);
    } else {
      navigate(`/runtime?query=${encodeURIComponent(value)}`);
    }
  };

  const openVnpyDocs = (): void => {
    window.open("https://www.vnpy.com/docs/cn/index.html", "_blank", "noopener,noreferrer");
  };

  return (
    <div className={`app-frame terminal-frame ${layout.launcherCollapsed ? "sidebar-collapsed" : ""} ${layout.activityOpen ? "activity-open" : ""}`}>
      <aside className="sidebar terminal-launcher">
        <div className="brand">
          <div className="brand-mark"><TrendUp size={22} weight="bold" /></div>
          <div><strong>QuantAgent</strong><span>QUANT WORKSTATION</span></div>
          <button className="icon-button sidebar-toggle" onClick={layout.toggleLauncher} aria-label="切换模块启动器">
            <SidebarSimple size={18} />
          </button>
        </div>
        <nav className="main-nav module-nav" aria-label="模块启动器">
          {moduleGroups.map((group) => (
            <section key={group.id} className="module-group">
              <span className="module-group-label">{group.label}</span>
              {workstationModules.filter((item) => item.group === group.id).map(({ path, label, caption, icon: NavIcon }) => (
                <NavLink key={path} to={path} end={path === "/"} title={`${label} · ${caption}`}>
                  <NavIcon size={19} weight="duotone" />
                  <span><strong>{label}</strong><small>{caption}</small></span>
                </NavLink>
              ))}
            </section>
          ))}
        </nav>
        <div className="launcher-security">
          <ShieldCheck size={17} />
          <span><strong>PAPER / READ ONLY</strong><small>Live trading disabled</small></span>
        </div>
      </aside>

      <header className="topbar terminal-topbar">
        <div className="terminal-menu" aria-label="终端菜单">
          <button onClick={() => setPaletteOpen(true)} title="打开模块启动器">模块 <CaretDown size={11} /></button>
          <button onClick={layout.resetLayout} title="恢复默认工作区布局">还原布局</button>
          <button onClick={() => navigate("/runtime")} title="打开 Runtime / DataManager">数据</button>
          <button onClick={() => navigate("/factors")} title="打开因子研究工作区">研究</button>
          <button onClick={openVnpyDocs} title="打开 VeighNa 官方文档">帮助</button>
        </div>
        <form className="global-search" onSubmit={submitSearch}>
          <MagnifyingGlass size={16} />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            onFocus={() => setPaletteOpen(true)}
            placeholder="股票 / 因子 / 模型 / Run ID / 命令"
            aria-label="全局搜索"
          />
          <kbd>⌘K</kbd>
        </form>
        <div className="topbar-status">
          <StatusBadge status={overview.isError ? "error" : overview.isLoading ? "loading" : "ready"} label={overview.isError ? "API OFFLINE" : "API CONNECTED"} />
          <span className="topbar-stat"><Database size={14} /> {data?.runtime.artifactCount?.toLocaleString() ?? "—"}</span>
          <span className="topbar-stat"><Pulse size={14} /> PIT</span>
          {(data?.riskStatus ?? "") === "warning" ? <Warning size={17} className="tone-warning" /> : <ShieldCheck size={17} className="tone-positive" />}
        </div>
      </header>

      <nav className="workspace-tabs" aria-label="已打开工作区">
        {layout.tabs.map((tab) => {
          const Icon = tab.module.icon;
          return (
            <div key={tab.path} className={tab.path === layout.activePath ? "active" : ""}>
              <button onClick={() => navigate(tab.path)} title={tab.path}>
                <Icon size={14} />
                <span>{tab.module.label}{tab.context ? ` · ${tab.context}` : ""}</span>
              </button>
              <button className="tab-close" onClick={() => layout.closeTab(tab.path)} aria-label={`关闭 ${tab.module.label}`}><X size={11} /></button>
            </div>
          );
        })}
      </nav>

      <main className="workspace terminal-workspace">
        <div className="workspace-context">
          <span><strong>{page.label}</strong> / {page.caption}</span>
          <span>Context: {location.search || "global"}</span>
          <span>As-of: runtime snapshot</span>
          <span className="context-trust">Research results ≠ executable orders</span>
        </div>
        {overview.isError ? (
          <div className="api-banner">
            <Warning size={18} />
            <span>Quant API 未连接；页面保持真实空数据态，不使用模拟结果。</span>
            <code>在仓库根目录执行：python -m services.quant_api</code>
          </div>
        ) : null}
        <Outlet />
      </main>

      {layout.activityOpen ? (
        <aside className="activity-drawer" aria-label="全局任务与事件">
          <header><span><Bell size={15} /> ACTIVITY / JOBS</span><small className={`activity-stream-status ${jobEvents.status}`}>{jobEvents.status === "live" ? "WebSocket live · typed events" : `${jobEvents.status} · 5s REST fallback`}</small><button onClick={layout.toggleActivity}><X size={14} /></button></header>
          {jobs.isLoading ? <StateView state="loading" /> : jobs.isError ? <StateView state="error" detail={jobs.error.message} /> : jobs.data?.data.length ? (
            <div className="activity-table-wrap">
              <table className="data-table activity-table">
                <thead><tr><th>Status</th><th>Job</th><th>Command</th><th>Created</th><th>Message</th></tr></thead>
                <tbody>{jobs.data.data.slice(0, 30).map((job) => (
                  <tr key={job.id}><td><StatusBadge status={job.status} /></td><td className="mono">{job.id}</td><td>{job.commandId}</td><td>{formatDate(job.createdAt)}</td><td>{job.message ?? job.error ?? "—"}</td></tr>
                ))}</tbody>
              </table>
            </div>
          ) : <StateView state="empty" detail="当前没有持久化任务事件。" />}
        </aside>
      ) : null}

      <footer className="statusbar terminal-statusbar">
        <span><i className={`health-dot ${overview.isError ? "offline" : ""}`} /> API {overview.isError ? "offline" : "connected"}</span>
        <span>Runtime {data?.runtime.indexedAt ? formatDate(data.runtime.indexedAt) : "unavailable"}</span>
        <span>Artifacts {data?.runtime.artifactCount ?? 0}</span>
        <span>Safety <strong>NO LIVE ORDERS</strong></span>
        <button className={layout.activityOpen ? "active" : ""} onClick={layout.toggleActivity}><ListMagnifyingGlass size={14} /> Activity {activeJobs ? `(${activeJobs})` : ""}</button>
      </footer>
      <CommandPalette open={paletteOpen} initialQuery={search} onClose={closePalette} />
    </div>
  );
}
