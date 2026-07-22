import { useCallback, useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { contextForPath, moduleForVNextPath } from "./modules";
import type {
  SavedWorkspaceState,
  WorkspaceDockTab,
  WorkspaceSplitDirection,
  WorkspaceTab,
  WorkspaceTheme,
} from "./types";

const STORAGE_KEY = "quantagent.workstation.vnext.v2";
const MAX_TABS = 16;
const MAX_CLOSED_TABS = 12;

function createId(): string {
  return typeof window.crypto?.randomUUID === "function"
    ? window.crypto.randomUUID()
    : `workspace-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function canonicalPath(path: string): string {
  const url = new URL(path, window.location.origin);
  url.searchParams.delete("ui");
  const search = url.searchParams.toString();
  return `${url.pathname}${search ? `?${search}` : ""}`;
}

function createTab(path: string, pinned = false): WorkspaceTab {
  const normalizedPath = canonicalPath(path);
  const module = moduleForVNextPath(normalizedPath);
  return {
    id: createId(),
    moduleId: module.id,
    path: normalizedPath,
    title: module.label,
    context: contextForPath(normalizedPath),
    pinned,
    dirty: false,
    status: "idle",
    createdAt: Date.now(),
  };
}

function defaultState(path: string): SavedWorkspaceState {
  const tab = createTab(path, path === "/");
  return {
    version: 3,
    railExpanded: true,
    dockOpen: false,
    dockTab: "tasks",
    dockSize: 230,
    density: "comfortable",
    theme: "night",
    tabs: [tab],
    activeTabId: tab.id,
    closedTabs: [],
    split: null,
  };
}

function isTab(value: unknown): value is WorkspaceTab {
  if (!value || typeof value !== "object") return false;
  const tab = value as Partial<WorkspaceTab>;
  return typeof tab.id === "string" && typeof tab.path === "string" && typeof tab.moduleId === "string";
}

function normalizeTab(tab: WorkspaceTab): WorkspaceTab {
  const path = canonicalPath(tab.path);
  const module = moduleForVNextPath(path);
  return {
    ...tab,
    path,
    moduleId: module.id,
    title: module.label,
    context: contextForPath(path),
    pinned: Boolean(tab.pinned),
    dirty: Boolean(tab.dirty),
    status: ["idle", "loading", "error"].includes(tab.status) ? tab.status : "idle",
    createdAt: Number(tab.createdAt) || Date.now(),
  };
}

function limitTabs(tabs: WorkspaceTab[]): WorkspaceTab[] {
  if (tabs.length <= MAX_TABS) return tabs;
  const pinned = tabs.filter((tab) => tab.pinned).slice(0, MAX_TABS);
  const available = Math.max(1, MAX_TABS - pinned.length);
  const recentUnpinned = tabs.filter((tab) => !tab.pinned).slice(-available);
  const retained = new Set([
    ...pinned.map((tab) => tab.id),
    ...recentUnpinned.map((tab) => tab.id),
  ]);
  return tabs.filter((tab) => retained.has(tab.id));
}

function loadState(path: string): SavedWorkspaceState {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return defaultState(path);
    const parsed = JSON.parse(raw) as Partial<SavedWorkspaceState>;
    const tabs = Array.isArray(parsed.tabs) ? limitTabs(parsed.tabs.filter(isTab).map(normalizeTab)) : [];
    if (!tabs.length) return defaultState(path);
    const activeTabId = tabs.some((tab) => tab.id === parsed.activeTabId) ? parsed.activeTabId as string : tabs[0].id;
    return {
      version: 3,
      railExpanded: parsed.railExpanded !== false,
      dockOpen: Boolean(parsed.dockOpen),
      dockTab: ["tasks", "logs", "alerts", "events", "resources"].includes(String(parsed.dockTab))
        ? parsed.dockTab as WorkspaceDockTab
        : "tasks",
      dockSize: Math.min(420, Math.max(160, Number(parsed.dockSize) || 230)),
      density: parsed.density === "compact" ? "compact" : "comfortable",
      theme: ["night", "dawn", "day"].includes(String(parsed.theme)) ? parsed.theme as WorkspaceTheme : "night",
      tabs,
      activeTabId,
      closedTabs: Array.isArray(parsed.closedTabs) ? parsed.closedTabs.filter(isTab).map(normalizeTab).slice(0, MAX_CLOSED_TABS) : [],
      split: parsed.split
        && tabs.some((tab) => tab.id === parsed.split?.tabId)
        && ["right", "bottom"].includes(String(parsed.split.direction))
        ? { tabId: parsed.split.tabId, direction: parsed.split.direction as WorkspaceSplitDirection }
        : null,
    };
  } catch {
    return defaultState(path);
  }
}

function withInstance(path: string): string {
  const url = new URL(canonicalPath(path), window.location.origin);
  url.searchParams.set("workspaceInstance", createId());
  return `${url.pathname}${url.search}`;
}

export function useWorkspaceStore() {
  const location = useLocation();
  const navigate = useNavigate();
  const routePath = canonicalPath(`${location.pathname}${location.search}`);
  const [state, setState] = useState<SavedWorkspaceState>(() => loadState(routePath));

  useEffect(() => {
    setState((current) => {
      const existing = current.tabs.find((tab) => tab.path === routePath);
      if (existing) return existing.id === current.activeTabId ? current : { ...current, activeTabId: existing.id };
      const active = current.tabs.find((tab) => tab.id === current.activeTabId);
      const routeModule = moduleForVNextPath(routePath);
      if (active?.moduleId === routeModule.id) {
        return {
          ...current,
          tabs: current.tabs.map((tab) => tab.id === active.id ? {
            ...tab,
            path: routePath,
            context: contextForPath(routePath),
          } : tab),
        };
      }
      const tab = createTab(routePath, routePath === "/");
      return { ...current, tabs: limitTabs([...current.tabs, tab]), activeTabId: tab.id };
    });
  }, [routePath]);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch {
      // The workspace remains usable without persistence.
    }
  }, [state]);

  const activeTab = useMemo(
    () => state.tabs.find((tab) => tab.id === state.activeTabId) ?? state.tabs[0],
    [state.activeTabId, state.tabs],
  );
  const splitTab = useMemo(
    () => state.split ? state.tabs.find((tab) => tab.id === state.split?.tabId) ?? null : null,
    [state.split, state.tabs],
  );

  const activateTab = useCallback((tabId: string) => {
    const tab = state.tabs.find((item) => item.id === tabId);
    if (!tab) return;
    setState((current) => ({ ...current, activeTabId: tabId }));
    navigate(tab.path);
  }, [navigate, state.tabs]);

  const openPath = useCallback((path: string, newInstance = false) => {
    const resolved = newInstance ? withInstance(path) : path;
    const existing = !newInstance ? state.tabs.find((tab) => tab.path === resolved) : undefined;
    if (existing) {
      activateTab(existing.id);
      return;
    }
    const tab = createTab(resolved);
    setState((current) => ({ ...current, tabs: limitTabs([...current.tabs, tab]), activeTabId: tab.id }));
    navigate(resolved);
  }, [activateTab, navigate, state.tabs]);

  const closeTab = useCallback((tabId: string) => {
    setState((current) => {
      const target = current.tabs.find((tab) => tab.id === tabId);
      if (!target || target.pinned) return current;
      let tabs = current.tabs.filter((tab) => tab.id !== tabId);
      if (!tabs.length) tabs = [createTab("/", true)];
      const fallback = tabs[Math.max(0, Math.min(current.tabs.findIndex((tab) => tab.id === tabId) - 1, tabs.length - 1))];
      const activeTabId = current.activeTabId === tabId ? fallback.id : current.activeTabId;
      if (current.activeTabId === tabId) window.setTimeout(() => navigate(fallback.path), 0);
      return {
        ...current,
        tabs,
        activeTabId,
        closedTabs: [target, ...current.closedTabs].slice(0, MAX_CLOSED_TABS),
        split: current.split?.tabId === tabId ? null : current.split,
      };
    });
  }, [navigate]);

  const togglePin = useCallback((tabId: string) => {
    setState((current) => ({
      ...current,
      tabs: current.tabs.map((tab) => tab.id === tabId ? { ...tab, pinned: !tab.pinned } : tab),
    }));
  }, []);

  const duplicateTab = useCallback((tabId: string) => {
    const source = state.tabs.find((tab) => tab.id === tabId);
    if (source) openPath(source.path, true);
  }, [openPath, state.tabs]);

  const closeOtherTabs = useCallback((tabId: string) => {
    setState((current) => {
      const keep = current.tabs.filter((tab) => tab.id === tabId || tab.pinned);
      const removed = current.tabs.filter((tab) => !keep.includes(tab));
      const target = keep.find((tab) => tab.id === tabId) ?? keep[0];
      window.setTimeout(() => navigate(target.path), 0);
      return {
        ...current,
        tabs: keep,
        activeTabId: target.id,
        closedTabs: [...removed.reverse(), ...current.closedTabs].slice(0, MAX_CLOSED_TABS),
        split: current.split && keep.some((tab) => tab.id === current.split?.tabId) ? current.split : null,
      };
    });
  }, [navigate]);

  const reorderTabs = useCallback((sourceId: string, targetId: string) => {
    if (sourceId === targetId) return;
    setState((current) => {
      const sourceIndex = current.tabs.findIndex((tab) => tab.id === sourceId);
      const targetIndex = current.tabs.findIndex((tab) => tab.id === targetId);
      if (sourceIndex < 0 || targetIndex < 0) return current;
      const tabs = [...current.tabs];
      const [source] = tabs.splice(sourceIndex, 1);
      tabs.splice(targetIndex, 0, source);
      return { ...current, tabs };
    });
  }, []);

  const reopenLastTab = useCallback(() => {
    setState((current) => {
      const [closed, ...rest] = current.closedTabs;
      if (!closed) return current;
      const tab = { ...closed, id: createId(), pinned: false, status: "idle" as const };
      window.setTimeout(() => navigate(tab.path), 0);
      return { ...current, tabs: limitTabs([...current.tabs, tab]), activeTabId: tab.id, closedTabs: rest };
    });
  }, [navigate]);

  const setSplit = useCallback((tabId: string, direction: WorkspaceSplitDirection) => {
    setState((current) => ({ ...current, split: { tabId, direction } }));
  }, []);

  const resetWorkspace = useCallback(() => {
    const reset = defaultState("/");
    setState(reset);
    navigate("/");
  }, [navigate]);

  return {
    state,
    activeTab,
    splitTab,
    activateTab,
    openPath,
    closeTab,
    togglePin,
    duplicateTab,
    closeOtherTabs,
    reorderTabs,
    reopenLastTab,
    setSplit,
    clearSplit: () => setState((current) => ({ ...current, split: null })),
    toggleRail: () => setState((current) => ({ ...current, railExpanded: !current.railExpanded })),
    toggleDock: () => setState((current) => ({ ...current, dockOpen: !current.dockOpen })),
    setDockTab: (dockTab: WorkspaceDockTab) => setState((current) => ({ ...current, dockTab, dockOpen: true })),
    setDockSize: (dockSize: number) => setState((current) => ({ ...current, dockSize: Math.min(420, Math.max(160, dockSize)) })),
    toggleDensity: () => setState((current) => ({ ...current, density: current.density === "compact" ? "comfortable" : "compact" })),
    setTheme: (theme: WorkspaceTheme) => setState((current) => ({ ...current, theme })),
    resetWorkspace,
  };
}
