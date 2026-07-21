import { fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, useLocation } from "react-router-dom";
import { expect, test, vi } from "vitest";
import { App } from "./App";
import { CommandPalette } from "./components/CommandPalette";

vi.mock("./components/EChart", () => ({
  EChart: () => <div data-testid="chart" />,
}));

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

test("renders the research terminal navigation with T+1 terminology", async () => {
  installFetchMock();
  renderApp("/");

  expect(await screen.findByText("QuantAgent")).toBeInTheDocument();
  expect(screen.getByText("T+1 做 T")).toBeInTheDocument();
  expect(screen.queryByText("T+0 Analysis")).not.toBeInTheDocument();
});

test("renders explicit empty state without fabricated data", async () => {
  installFetchMock();
  renderApp("/factors");

  expect((await screen.findAllByText("暂无可用数据")).length).toBeGreaterThan(0);
  expect(screen.getAllByText(/只展示已持久化的真实 QuantAgent artifact/).length).toBeGreaterThan(0);
});

test("command palette routes a stock code to stock replay", () => {
  function LocationProbe(): JSX.Element {
    return <div data-testid="location">{useLocation().pathname}{useLocation().search}</div>;
  }

  render(
    <MemoryRouter>
      <CommandPalette open onClose={() => undefined} />
      <LocationProbe />
    </MemoryRouter>,
  );

  const input = screen.getByPlaceholderText("输入页面、股票代码、因子、模型或功能");
  fireEvent.change(input, { target: { value: "000001.SZ" } });
  fireEvent.keyDown(input, { key: "Enter" });

  expect(screen.getByTestId("location")).toHaveTextContent("/stock-replay?symbol=000001.SZ");
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
