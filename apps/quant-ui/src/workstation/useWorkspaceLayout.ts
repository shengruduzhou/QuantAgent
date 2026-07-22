import { useCallback, useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { moduleForPath } from "./modules";

const STORAGE_KEY = "quantagent.workstation.layout.v1";
const MAX_TABS = 8;

interface SavedLayout {
  launcherCollapsed: boolean;
  activityOpen: boolean;
  openTabs: string[];
}

const defaultLayout: SavedLayout = {
  launcherCollapsed: false,
  activityOpen: false,
  openTabs: ["/"],
};

function loadLayout(): SavedLayout {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    if (!value) return defaultLayout;
    const parsed = JSON.parse(value) as Partial<SavedLayout>;
    return {
      launcherCollapsed: Boolean(parsed.launcherCollapsed),
      activityOpen: Boolean(parsed.activityOpen),
      openTabs: Array.isArray(parsed.openTabs) && parsed.openTabs.length
        ? parsed.openTabs.filter((item): item is string => typeof item === "string").slice(-MAX_TABS)
        : ["/"],
    };
  } catch {
    return defaultLayout;
  }
}

export function useWorkspaceLayout() {
  const [layout, setLayout] = useState<SavedLayout>(loadLayout);
  const location = useLocation();
  const navigate = useNavigate();
  const activePath = `${location.pathname}${location.search}`;

  useEffect(() => {
    setLayout((current) => current.openTabs.includes(activePath)
      ? current
      : { ...current, openTabs: [...current.openTabs, activePath].slice(-MAX_TABS) });
  }, [activePath]);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(layout));
    } catch {
      // The terminal remains usable when browser storage is unavailable.
    }
  }, [layout]);

  const tabs = useMemo(() => layout.openTabs.map((path) => {
    const url = new URL(path, window.location.origin);
    return { path, module: moduleForPath(url.pathname), context: url.searchParams.get("symbol") ?? url.searchParams.get("query") };
  }), [layout.openTabs]);

  const closeTab = useCallback((path: string) => {
    setLayout((current) => {
      const nextTabs = current.openTabs.filter((item) => item !== path);
      const resolved = nextTabs.length ? nextTabs : ["/"];
      if (path === activePath) {
        window.setTimeout(() => navigate(resolved[resolved.length - 1]), 0);
      }
      return { ...current, openTabs: resolved };
    });
  }, [activePath, navigate]);

  const resetLayout = useCallback(() => {
    setLayout(defaultLayout);
    if (activePath !== "/") {
      window.setTimeout(() => navigate("/"), 0);
    }
  }, [activePath, navigate]);

  return {
    ...layout,
    activePath,
    tabs,
    closeTab,
    resetLayout,
    toggleLauncher: () => setLayout((current) => ({ ...current, launcherCollapsed: !current.launcherCollapsed })),
    toggleActivity: () => setLayout((current) => ({ ...current, activityOpen: !current.activityOpen })),
  };
}
