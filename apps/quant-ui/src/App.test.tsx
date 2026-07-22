import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { App } from "./App";

vi.mock("./components/EChart", () => ({
  EChart: () => <div data-testid="chart" />,
}));

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  window.localStorage.clear();
  window.history.replaceState({}, "", "/");
});

const overview = {
  modelStatus: "ready",
  latestModel: null,
  latestBacktest: null,
  latestSelection: null,
  stockPoolCount: 0,
  candidateCount: 0,
  signalCount: 0,
  buySignalCount: 0,
  sellSignalCount: 0,
  doTSignalCount: 0,
  riskStatus: "normal",
  risk: { eventCounts: {}, rules: [] },
  runtime: {
    artifactCount: 0,
    totalSizeBytes: 0,
    byKind: {},
    indexedAt: "2026-06-19T00:00:00+00:00",
  },
};

function installFetchMock(): void {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const data = url.includes("/system/overview") ? overview : [];
    return new Response(JSON.stringify({
      status: Array.isArray(data) && data.length === 0 ? "empty" : "ready",
      data,
      issues: [],
    }), { status: 200, headers: { "Content-Type": "application/json" } });
  }));
}

function renderApp(path = "/"): void {
  window.history.replaceState({}, "", path);
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[path]}>
        <App />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

test("renders the VNext institutional module rail with T+1 terminology", async () => {
  installFetchMock();
  renderApp("/");

  expect(await screen.findByLabelText("QuantAgent 模块栏")).toBeInTheDocument();
  expect(screen.getByText("T+1 分析")).toBeInTheDocument();
  expect(screen.queryByText("T+0 Analysis")).not.toBeInTheDocument();
});

test("VNext dashboard separates decision states from the primary canvas", async () => {
  installFetchMock();
  renderApp("/");

  expect(await screen.findByRole("heading", { name: "今日决策总览" })).toBeInTheDocument();
  expect(screen.getByText("Portfolio State")).toBeInTheDocument();
  expect(screen.getByText("Model State")).toBeInTheDocument();
  expect(screen.getByText("Risk State")).toBeInTheDocument();
  expect(screen.getByText("Operations State")).toBeInTheDocument();
  expect(screen.getByText("PRIMARY DECISION CANVAS")).toBeInTheDocument();

  const portfolio = screen.getByRole("button", { name: "Portfolio" });
  const risk = screen.getByRole("button", { name: "Risk" });
  expect(portfolio).toHaveAttribute("aria-pressed", "true");
  fireEvent.click(risk);
  expect(risk).toHaveAttribute("aria-pressed", "true");
  expect(portfolio).toHaveAttribute("aria-pressed", "false");
});

test("keeps the institutional workstation active when a retired legacy query is present", async () => {
  installFetchMock();
  renderApp("/?ui=legacy");

  expect(await screen.findByLabelText("QuantAgent 模块栏")).toBeInTheDocument();
  expect(await screen.findByRole("heading", { name: "今日决策总览" })).toBeInTheDocument();
});

test("opens QuantAgent help internally from the VNext command bar", async () => {
  installFetchMock();
  renderApp("/");

  fireEvent.click(await screen.findByRole("button", { name: "打开用户与帮助" }));
  expect(await screen.findByRole("heading", { name: "帮助中心" })).toBeInTheDocument();
  expect(document.querySelector('a[href^="http"]')).toBeNull();
});

test("switches and persists the institutional workstation visual theme", async () => {
  installFetchMock();
  renderApp("/");

  const trigger = await screen.findByRole("button", { name: "切换界面主题" });
  expect(document.querySelector(".vnext-shell")).toHaveAttribute("data-theme", "night");
  fireEvent.click(trigger);
  fireEvent.click(screen.getByRole("menuitemradio", { name: /日间/ }));

  expect(document.querySelector(".vnext-shell")).toHaveAttribute("data-theme", "day");
  await waitFor(() => {
    const saved = JSON.parse(window.localStorage.getItem("quantagent.workstation.vnext.v2") ?? "{}") as { theme?: string };
    expect(saved.theme).toBe("day");
  });
});

test("supports duplicate workspace instances and a real split pane", async () => {
  installFetchMock();
  renderApp("/");

  const rail = await screen.findByLabelText("QuantAgent 模块栏");
  fireEvent.click(within(rail).getByTitle("训练实验室 · Training Lab"));
  expect(await screen.findByRole("heading", { name: "Training Lab" })).toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: "管理 训练实验室 标签" }));
  fireEvent.click(screen.getByRole("button", { name: /复制实例/ }));
  await waitFor(() => expect(screen.getAllByRole("button", { name: "管理 训练实验室 标签" })).toHaveLength(2));

  fireEvent.click(screen.getAllByRole("button", { name: "管理 训练实验室 标签" })[0]);
  fireEvent.click(screen.getByRole("button", { name: /在右侧打开/ }));
  expect(await screen.findByLabelText(/训练实验室 分屏工作区/)).toBeInTheDocument();
});

test("renders explicit empty state without fabricated data", async () => {
  installFetchMock();
  renderApp("/factors");

  expect((await screen.findAllByText("暂无可用数据")).length).toBeGreaterThan(0);
  expect(screen.getAllByText(/只展示已持久化的真实 QuantAgent artifact/).length).toBeGreaterThan(0);
});

test("VNext command palette routes an exact stock code to Chart Workstation", async () => {
  installFetchMock();
  renderApp("/");

  fireEvent.click(await screen.findByRole("button", { name: "打开全局实体与命令搜索" }));
  const input = screen.getByRole("textbox", { name: "搜索 QuantAgent 实体和命令" });
  fireEvent.change(input, { target: { value: "000001.SZ" } });
  fireEvent.keyDown(input, { key: "Enter" });

  expect(await screen.findByTitle("/stock-replay?symbol=000001.SZ")).toBeInTheDocument();
});

test("runtime explorer distinguishes verified production artifacts", async () => {
  const artifact = {
    id: "artifact_metrics",
    kind: "backtest",
    name: "metrics.json",
    path: "runtime/reports/run-1/backtest/metrics.json",
    extension: ".json",
    sizeBytes: 128,
    modifiedAt: "2026-01-06T00:00:00+00:00",
    status: "ready",
    parser: "json",
    runId: "run-1",
    horizon: "short_5d",
    tags: [],
    schemaVersion: "quantagent.backtest.metrics.1",
    trustClass: "production_ready",
    validationStatus: "verified",
    freshnessStatus: "unknown",
    sourceTime: "2026-01-06T00:00:00+00:00",
    manifestPath: "runtime/reports/run-1/backtest/metrics.json.manifest.json",
    contentHash: "a".repeat(64),
    upstreamPaths: ["runtime/data/v7/gold/dataset.parquet"],
    capabilities: ["metadata", "preview", "research_display", "production_display"],
    issues: [],
  };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const data = url.includes("/system/overview") ? overview
      : url.includes("/system/runtime-catalog") ? {
        summary: {
          artifactCount: 1, totalSizeBytes: 128, byKind: { backtest: 1 },
          byTrust: { production_ready: 1 }, byValidation: { verified: 1 },
          byFreshness: { unknown: 1 }, byCapability: { production_display: 1 },
          byStatus: { ready: 1 }, runCount: 1, manifestCoverage: 1,
          indexedAt: "2026-01-06T00:00:00+00:00",
        },
        runs: [{
          id: "run-1", artifactCount: 1, totalSizeBytes: 128, kinds: ["backtest"],
          trustClasses: ["production_ready"], validationStatuses: ["verified"],
          capabilities: ["production_display"], issueCount: 0,
          latestModifiedAt: "2026-01-06T00:00:00+00:00",
        }],
        roots: ["runtime"],
      }
        : url.includes("/lineage") ? {
          artifact, upstream: [{ reference: artifact.upstreamPaths[0], artifact: null }],
          downstream: [], status: "partial", issues: [],
        }
      : url.includes("/system/runtime-index/artifact_metrics/preview") ? { total_return: 0.12 }
        : url.includes("/system/runtime-index") ? {
          items: [artifact], total: 1, page: 1, pageSize: 100, hasNext: false,
        }
          : [];
    return new Response(JSON.stringify({ status: "ready", data, issues: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }));

  renderApp("/runtime");

  expect(await screen.findByText("metrics.json")).toBeInTheDocument();
  expect(screen.getAllByText("production_ready").length).toBeGreaterThan(0);
  expect(screen.getAllByText("verified").length).toBeGreaterThan(0);
  fireEvent.click(screen.getByText("metrics.json"));
  expect(await screen.findByText("quantagent.backtest.metrics.1")).toBeInTheDocument();
  expect(screen.getAllByText(/production_display/).length).toBeGreaterThan(0);
  fireEvent.click(screen.getByRole("button", { name: "Runs" }));
  expect(await screen.findByText("run-1")).toBeInTheDocument();
});
