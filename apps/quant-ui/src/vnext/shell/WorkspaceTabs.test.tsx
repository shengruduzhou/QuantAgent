import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import type { WorkspaceTab } from "../workspace/types";
import { WorkspaceTabs } from "./WorkspaceTabs";

afterEach(cleanup);

const tab: WorkspaceTab = {
  id: "strategy",
  moduleId: "strategy",
  path: "/strategy",
  title: "策略实验室",
  context: {},
  pinned: false,
  dirty: false,
  status: "idle",
  createdAt: 1,
};

test("ellipsis exposes visible, actionable tab operations", () => {
  const duplicateTab = vi.fn();
  const closeTab = vi.fn();
  render(<WorkspaceTabs
    tabs={[tab]}
    activeTabId={tab.id}
    canReopen={false}
    activateTab={vi.fn()}
    closeTab={closeTab}
    togglePin={vi.fn()}
    duplicateTab={duplicateTab}
    closeOtherTabs={vi.fn()}
    reorderTabs={vi.fn()}
    reopenLastTab={vi.fn()}
    setSplit={vi.fn()}
    openLauncher={vi.fn()}
  />);

  const menuButton = screen.getByRole("button", { name: "管理 策略实验室 标签" });
  fireEvent.click(menuButton);
  expect(menuButton).toHaveAttribute("aria-expanded", "true");
  expect(screen.getByRole("menu", { name: "策略实验室 标签操作" })).toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "复制实例" }));
  expect(duplicateTab).toHaveBeenCalledWith("strategy");

  fireEvent.click(menuButton);
  fireEvent.click(screen.getByRole("button", { name: "关闭标签" }));
  expect(closeTab).toHaveBeenCalledWith("strategy");
});
