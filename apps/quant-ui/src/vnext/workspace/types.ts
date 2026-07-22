export type WorkspaceDensity = "compact" | "comfortable";
export type WorkspaceTheme = "night" | "dawn" | "day";
export type WorkspaceDockTab = "tasks" | "logs" | "alerts" | "events" | "resources";
export type WorkspaceSplitDirection = "right" | "bottom";
export type WorkspaceTabStatus = "idle" | "loading" | "error";

export interface WorkspaceEntityContext {
  symbol?: string;
  experiment?: string;
  run?: string;
  model?: string;
  portfolio?: string;
  account?: string;
  artifact?: string;
}

export interface WorkspaceTab {
  id: string;
  moduleId: string;
  path: string;
  title: string;
  context: WorkspaceEntityContext;
  pinned: boolean;
  dirty: boolean;
  status: WorkspaceTabStatus;
  createdAt: number;
}

export interface WorkspaceSplit {
  direction: WorkspaceSplitDirection;
  tabId: string;
}

export interface SavedWorkspaceState {
  version: 3;
  railExpanded: boolean;
  dockOpen: boolean;
  dockTab: WorkspaceDockTab;
  dockSize: number;
  density: WorkspaceDensity;
  theme: WorkspaceTheme;
  tabs: WorkspaceTab[];
  activeTabId: string;
  closedTabs: WorkspaceTab[];
  split: WorkspaceSplit | null;
}
