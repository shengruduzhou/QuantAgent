import { useState } from "react";
import {
  ArrowBendDownRight,
  ArrowBendRightDown,
  Copy,
  DotsThree,
  LockSimple,
  LockSimpleOpen,
  Plus,
  X,
} from "@phosphor-icons/react";
import { moduleForVNextPath } from "../workspace/modules";
import type { WorkspaceTab } from "../workspace/types";

interface WorkspaceTabsProps {
  tabs: WorkspaceTab[];
  activeTabId: string;
  splitTabId?: string;
  canReopen: boolean;
  activateTab: (tabId: string) => void;
  closeTab: (tabId: string) => void;
  togglePin: (tabId: string) => void;
  duplicateTab: (tabId: string) => void;
  closeOtherTabs: (tabId: string) => void;
  reorderTabs: (sourceId: string, targetId: string) => void;
  reopenLastTab: () => void;
  setSplit: (tabId: string, direction: "right" | "bottom") => void;
  openLauncher: () => void;
}

export function WorkspaceTabs(props: WorkspaceTabsProps): JSX.Element {
  const [menuTabId, setMenuTabId] = useState<string | null>(null);
  const [draggedTabId, setDraggedTabId] = useState<string | null>(null);

  return (
    <nav className="vnext-workspace-tabs" aria-label="工作区标签">
      <div className="vnext-tab-strip">
        {props.tabs.map((tab) => {
          const module = moduleForVNextPath(tab.path);
          const Icon = module.icon;
          const context = Object.values(tab.context).find(Boolean);
          return (
            <div
              key={tab.id}
              className={`vnext-workspace-tab ${tab.id === props.activeTabId ? "active" : ""} ${tab.id === props.splitTabId ? "split" : ""}`}
              draggable
              onDragStart={() => setDraggedTabId(tab.id)}
              onDragOver={(event) => event.preventDefault()}
              onDrop={() => {
                if (draggedTabId) props.reorderTabs(draggedTabId, tab.id);
                setDraggedTabId(null);
              }}
              onContextMenu={(event) => {
                event.preventDefault();
                setMenuTabId(tab.id);
              }}
            >
              <button type="button" className="vnext-tab-main" onClick={() => props.activateTab(tab.id)} title={tab.path}>
                {tab.pinned ? <LockSimple size={12} /> : <Icon size={14} />}
                <span><strong>{tab.title}</strong>{context ? <small>{context}</small> : null}</span>
                {tab.dirty ? <i title="未保存更改" /> : null}
                {tab.status !== "idle" ? <em className={`state-${tab.status}`}>{tab.status}</em> : null}
              </button>
              <button type="button" className="vnext-tab-menu-button" onClick={() => setMenuTabId(menuTabId === tab.id ? null : tab.id)} aria-label={`管理 ${tab.title} 标签`}><DotsThree size={15} /></button>
              {!tab.pinned ? <button type="button" className="vnext-tab-close" onClick={() => props.closeTab(tab.id)} aria-label={`关闭 ${tab.title}`}><X size={12} /></button> : null}
              {menuTabId === tab.id ? (
                <div className="vnext-tab-menu" role="menu">
                  <button type="button" onClick={() => { props.togglePin(tab.id); setMenuTabId(null); }}>{tab.pinned ? <LockSimpleOpen size={14} /> : <LockSimple size={14} />}{tab.pinned ? "取消固定" : "固定标签"}</button>
                  <button type="button" onClick={() => { props.duplicateTab(tab.id); setMenuTabId(null); }}><Copy size={14} />复制实例</button>
                  <button type="button" onClick={() => { props.setSplit(tab.id, "right"); setMenuTabId(null); }}><ArrowBendDownRight size={14} />在右侧打开</button>
                  <button type="button" onClick={() => { props.setSplit(tab.id, "bottom"); setMenuTabId(null); }}><ArrowBendRightDown size={14} />在下方打开</button>
                  <button type="button" onClick={() => { props.closeOtherTabs(tab.id); setMenuTabId(null); }}><X size={14} />关闭其他</button>
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
      <div className="vnext-tab-actions">
        <button type="button" onClick={props.openLauncher} aria-label="打开新工作区"><Plus size={15} /></button>
        <button type="button" onClick={props.reopenLastTab} disabled={!props.canReopen} title="恢复最近关闭的标签">↶</button>
      </div>
    </nav>
  );
}
