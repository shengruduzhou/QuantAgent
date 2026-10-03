import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { App } from "./App";
import { ErrorBoundary } from "./components/ErrorBoundary";

vi.mock("./components/EChart", () => ({ EChart: () => <div data-testid="chart" /> }));
// A page whose artifact has an unexpected shape: it throws during render.
vi.mock("./pages/RiskCenterPage", () => ({
  RiskCenterPage: () => {
    throw new TypeError("Cannot read properties of undefined (reading 'eventCounts')");
  },
}));

beforeEach(() => {
  vi.spyOn(console, "error").mockImplementation(() => undefined);
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const data = url.includes("/system/overview")
      ? { modelStatus: "ready", riskStatus: "normal", risk: { eventCounts: {}, rules: [] }, runtime: { artifactCount: 0, totalSizeBytes: 0, byKind: {}, indexedAt: "2026-10-01T00:00:00+00:00" } }
      : [];
    return new Response(JSON.stringify({ status: "ready", data, issues: [] }), { status: 200 });
  }));
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
  window.localStorage.clear();
  window.history.replaceState({}, "", "/");
});

function renderApp(path: string): void {
  window.history.replaceState({}, "", path);
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter initialEntries={[path]}><App /></MemoryRouter>
    </QueryClientProvider>,
  );
}

test("a crashing page shows a contained diagnosis while the shell and safety chrome stay up", async () => {
  renderApp("/risk");
  const panel = await screen.findByRole("alert", { name: "工作区渲染失败" });
  expect(within(panel).getByText(/TypeError: Cannot read properties of undefined/)).toBeInTheDocument();
  expect(within(panel).getByText("/risk")).toBeInTheDocument();
  // The rest of the workstation is still rendered.
  expect(screen.getByLabelText("QuantAgent 模块栏")).toBeInTheDocument();
  expect(screen.getByLabelText("系统与安全状态")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "切换界面主题" })).toBeInTheDocument();

  // Moving to another workspace clears the contained error.
  fireEvent.click(within(screen.getByLabelText("QuantAgent 模块栏")).getByTitle("训练实验室 · Training Lab"));
  expect(await screen.findByRole("heading", { name: "训练实验室" })).toBeInTheDocument();
  expect(screen.queryByRole("alert", { name: "工作区渲染失败" })).not.toBeInTheDocument();
});

test("the root boundary replaces a blank page with a diagnosis", () => {
  function Broken(): JSX.Element {
    throw new Error("shell exploded");
  }
  render(<ErrorBoundary scope="root" location="/"><Broken /></ErrorBoundary>);
  const panel = screen.getByRole("alert", { name: "工作站渲染失败" });
  expect(within(panel).getByText("Error: shell exploded")).toBeInTheDocument();
  expect(within(panel).getByRole("button", { name: /重新加载工作站/ })).toBeInTheDocument();
  expect(panel.closest(".vnext-shell")).toHaveAttribute("data-theme", "night");
});

test("the workspace boundary retries in place", () => {
  let fail = true;
  function Flaky(): JSX.Element {
    if (fail) throw new Error("transient");
    return <p>recovered</p>;
  }
  render(<ErrorBoundary scope="workspace" location="/x"><Flaky /></ErrorBoundary>);
  expect(screen.getByRole("alert", { name: "工作区渲染失败" })).toBeInTheDocument();
  fail = false;
  fireEvent.click(screen.getByRole("button", { name: /重试渲染/ }));
  expect(screen.getByText("recovered")).toBeInTheDocument();
});
