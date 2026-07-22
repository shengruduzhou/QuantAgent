import { useCallback, useEffect, useState } from "react";
import { WarningCircle, X } from "@phosphor-icons/react";
import type { JobSummary, SystemOverview } from "../../api/types";
import { useApi } from "../../hooks/useApi";
import { useJobEvents } from "../../hooks/useJobEvents";
import { moduleForVNextPath } from "../workspace/modules";
import { useWorkspaceStore } from "../workspace/useWorkspaceStore";
import { EntityCommandPalette } from "./EntityCommandPalette";
import { GlobalCommandBar } from "./GlobalCommandBar";
import { ModuleRail } from "./ModuleRail";
import { OperationsDock } from "./OperationsDock";
import { WorkspaceRoutes } from "./WorkspaceRoutes";
import { WorkspaceTabs } from "./WorkspaceTabs";
import { VNextThemeContext } from "../theme";

export function InstitutionalShell(): JSX.Element {
  const [commandOpen, setCommandOpen] = useState(false);
  const workspace = useWorkspaceStore();
  const overview = useApi<SystemOverview>(["system-overview-shell"], "/system/overview", undefined, { refetchInterval: 15_000, staleTime: 10_000 });
  const jobs = useApi<JobSummary[]>(["global-activity-jobs"], "/jobs", undefined, { refetchInterval: 5_000, staleTime: 2_000 });
  const realtime = useJobEvents(true);
  const data = overview.data?.data;
  const jobItems = jobs.data?.data ?? [];
  const riskEventCount = Object.values(data?.risk.eventCounts ?? {}).reduce((sum, count) => sum + count, 0);
  const activeModule = moduleForVNextPath(workspace.activeTab?.path ?? "/");
  const openCommand = useCallback(() => setCommandOpen(true), []);

  useEffect(() => {
    const shortcut = (event: KeyboardEvent): void => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
        event.preventDefault();
        setCommandOpen(true);
      }
      if ((event.metaKey || event.ctrlKey) && event.shiftKey && event.key.toLowerCase() === "t") {
        event.preventDefault();
        workspace.reopenLastTab();
      }
    };
    window.addEventListener("keydown", shortcut);
    return () => window.removeEventListener("keydown", shortcut);
  }, [workspace.reopenLastTab]);

  if (!workspace.activeTab) return <div className="vnext-shell-fatal">Workspace state unavailable.</div>;

  return (
    <VNextThemeContext.Provider value={workspace.state.theme}>
    <div
      className={`vnext-shell theme-${workspace.state.theme} rail-${workspace.state.railExpanded ? "expanded" : "collapsed"} density-${workspace.state.density} dock-${workspace.state.dockOpen ? "open" : "closed"}`}
      data-theme={workspace.state.theme}
      style={{ "--vnext-dock-height": workspace.state.dockOpen ? `${workspace.state.dockSize}px` : "32px" } as React.CSSProperties}
    >
      <ModuleRail
        expanded={workspace.state.railExpanded}
        activeModuleId={activeModule.id}
        jobs={jobItems}
        riskEventCount={riskEventCount}
        onToggle={workspace.toggleRail}
        openPath={workspace.openPath}
      />
      <GlobalCommandBar
        activeTab={workspace.activeTab}
        overview={data}
        apiState={overview.isLoading ? "loading" : overview.isError ? "error" : "ready"}
        jobs={jobItems}
        realtime={realtime}
        railExpanded={workspace.state.railExpanded}
        density={workspace.state.density}
        theme={workspace.state.theme}
        onToggleRail={workspace.toggleRail}
        onOpenCommand={openCommand}
        onToggleDensity={workspace.toggleDensity}
        onSetTheme={workspace.setTheme}
        openPath={workspace.openPath}
      />
      <WorkspaceTabs
        tabs={workspace.state.tabs}
        activeTabId={workspace.state.activeTabId}
        splitTabId={workspace.splitTab?.id}
        canReopen={Boolean(workspace.state.closedTabs.length)}
        activateTab={workspace.activateTab}
        closeTab={workspace.closeTab}
        togglePin={workspace.togglePin}
        duplicateTab={workspace.duplicateTab}
        closeOtherTabs={workspace.closeOtherTabs}
        reorderTabs={workspace.reorderTabs}
        reopenLastTab={workspace.reopenLastTab}
        setSplit={workspace.setSplit}
        openLauncher={openCommand}
      />
      <main className={`vnext-workspace ${workspace.state.split ? `split-${workspace.state.split.direction}` : "single"}`}>
        {overview.isError ? (
          <div className="vnext-api-banner" role="alert"><WarningCircle size={16} /><span>Quant API 未连接；工作站保持真实 unavailable 状态，不使用模拟 Dashboard 数据。</span></div>
        ) : null}
        <section className="vnext-workspace-pane primary" aria-label={`${workspace.activeTab.title} 主工作区`}>
          <WorkspaceRoutes location={workspace.activeTab.path} />
        </section>
        {workspace.splitTab && workspace.state.split ? (
          <section className="vnext-workspace-pane secondary" aria-label={`${workspace.splitTab.title} 分屏工作区`}>
            <header className="vnext-split-header"><span>{workspace.splitTab.title}</span><code>{workspace.splitTab.path}</code><button type="button" onClick={workspace.clearSplit} aria-label="关闭分屏"><X size={14} /></button></header>
            <WorkspaceRoutes location={workspace.splitTab.path} />
          </section>
        ) : null}
      </main>
      <OperationsDock
        open={workspace.state.dockOpen}
        tab={workspace.state.dockTab}
        size={workspace.state.dockSize}
        jobs={jobItems}
        overview={data}
        realtime={realtime}
        setTab={workspace.setDockTab}
        toggle={workspace.toggleDock}
        setSize={workspace.setDockSize}
        openPath={workspace.openPath}
      />
      <EntityCommandPalette open={commandOpen} onClose={() => setCommandOpen(false)} openPath={workspace.openPath} />
    </div>
    </VNextThemeContext.Provider>
  );
}
