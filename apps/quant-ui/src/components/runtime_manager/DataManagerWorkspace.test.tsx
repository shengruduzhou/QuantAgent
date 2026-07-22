import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { DataManagerWorkspace } from "./DataManagerWorkspace";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("submits the existing TickFlow daily provider contract by default", async () => {
  let submittedBody: Record<string, unknown> | undefined;
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith("/data/providers")) {
      return jsonResponse({
        providers: [
          { id: "tickflow", label: "TickFlow A股主数据源", module: "tickflow", commandId: "fetch-tickflow-daily", assetClasses: ["A股", "分钟线", "Level-2"], intervals: ["1d", "1m", "depth"], operations: ["download", "record"], requires: [], note: "PIT", installed: true, configured: true, status: "partial", missingRequirements: [], missingOptionalRequirements: ["TICKFLOW_API_KEY"] },
          { id: "runtime_catalog", label: "QuantAgent Runtime", module: null, commandId: null, assetClasses: ["PIT artifact"], intervals: ["manifest"], operations: ["preview"], requires: [], note: "catalog", installed: true, configured: true, status: "ready", missingRequirements: [] },
        ],
        constraints: [], jobEndpoint: "/api/jobs/data", coverageEndpoint: "/api/data/coverage", quarantineEndpoint: "/api/data/quarantine", supportsCancellation: true, runtimeRoot: "runtime", serverPaths: { quarantine: "runtime/import_quarantine", imports: "runtime/data/imported", exports: "runtime/exports" },
      });
    }
    if (url.endsWith("/data/quarantine")) return jsonResponse([]);
    if (url.endsWith("/jobs/data")) {
      submittedBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return jsonResponse({ id: "job_data", type: "data", status: "queued", commandId: "fetch-tickflow-daily", createdAt: "2026-07-22T00:00:00Z", outputPaths: [] });
    }
    return jsonResponse([]);
  }));

  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<MemoryRouter><QueryClientProvider client={queryClient}><DataManagerWorkspace /></QueryClientProvider></MemoryRouter>);

  expect(await screen.findByText("TickFlow A股主数据源")).toBeInTheDocument();
  const launch = screen.getByRole("button", { name: /启动任务/ });
  expect(launch).toBeDisabled();
  fireEvent.click(screen.getByRole("checkbox", { name: /确认允许本次真实 provider 任务访问网络/ }));
  fireEvent.click(launch);

  await waitFor(() => expect(submittedBody).toBeDefined());
  expect(submittedBody).toMatchObject({
    commandId: "fetch-tickflow-daily",
    parameters: {
      symbols: "000001.SZ,600519.SH",
      allow_network: true,
      output: "runtime/data/v7/silver/market_panel/tickflow_daily.parquet",
    },
  });
  expect(JSON.stringify(submittedBody)).not.toContain("shell");
  expect(await screen.findByText(/已提交 fetch-tickflow-daily/)).toBeInTheDocument();
});

test("keeps TickFlow acquisition and recorder modes mutually exclusive", async () => {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith("/data/providers")) {
      return jsonResponse({
        providers: [
          { id: "tickflow", label: "TickFlow A股主数据源", module: "tickflow", commandId: "fetch-tickflow-daily", assetClasses: ["A股", "分钟线", "Level-2"], intervals: ["1d", "1m", "tick", "depth"], operations: ["download", "update", "record"], requires: [], optionalRequirements: [], note: "PIT", installed: true, configured: true, status: "ready", missingRequirements: [], missingOptionalRequirements: [] },
        ],
        constraints: [], jobEndpoint: "/api/jobs/data", coverageEndpoint: "/api/data/coverage", quarantineEndpoint: "/api/data/quarantine", supportsCancellation: true, runtimeRoot: "runtime", serverPaths: { quarantine: "runtime/import_quarantine", imports: "runtime/data/imported", exports: "runtime/exports" },
      });
    }
    return jsonResponse([]);
  }));

  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<MemoryRouter><QueryClientProvider client={queryClient}><DataManagerWorkspace /></QueryClientProvider></MemoryRouter>);

  const granularity = await screen.findByRole("radiogroup", { name: "TickFlow 数据粒度" });
  const acquisitionModes = within(granularity).getAllByRole("radio") as HTMLInputElement[];
  expect(acquisitionModes).toHaveLength(4);
  expect(acquisitionModes.filter((item) => item.checked)).toHaveLength(1);
  fireEvent.click(within(granularity).getByRole("radio", { name: "分钟线" }));
  expect(acquisitionModes.filter((item) => item.checked)).toHaveLength(1);
  expect(within(granularity).getByRole("radio", { name: "分钟线" })).toBeChecked();

  fireEvent.click(screen.getByRole("button", { name: /实时录制/ }));
  const recorder = screen.getByRole("radiogroup", { name: "DataRecorder 数据类型" });
  const recorderModes = within(recorder).getAllByRole("radio") as HTMLInputElement[];
  expect(recorderModes).toHaveLength(2);
  expect(recorderModes.filter((item) => item.checked)).toHaveLength(1);
  fireEvent.click(within(recorder).getByRole("radio", { name: "Level-2 五档盘口" }));
  expect(recorderModes.filter((item) => item.checked)).toHaveLength(1);
  expect(within(recorder).getByRole("radio", { name: "Level-2 五档盘口" })).toBeChecked();
});

function jsonResponse(data: unknown): Response {
  return new Response(JSON.stringify({ status: "ready", data, issues: [] }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}
