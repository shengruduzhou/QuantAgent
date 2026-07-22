import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { DataManagerWorkspace } from "./DataManagerWorkspace";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("submits only the explicit governed AkShare data contract", async () => {
  let submittedBody: Record<string, unknown> | undefined;
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith("/data/providers")) {
      return jsonResponse({
        providers: [
          { id: "akshare_market", label: "AkShare A股行情", module: "akshare", commandId: "build-akshare-market-panel-v7", assetClasses: ["A股"], intervals: ["1d"], operations: ["download"], requires: [], note: "PIT", installed: true, configured: true, status: "ready", missingRequirements: [] },
          { id: "runtime_catalog", label: "QuantAgent Runtime", module: null, commandId: null, assetClasses: ["PIT artifact"], intervals: ["manifest"], operations: ["preview"], requires: [], note: "catalog", installed: true, configured: true, status: "ready", missingRequirements: [] },
        ],
        constraints: [], jobEndpoint: "/api/jobs/data", supportsCancellation: true, runtimeRoot: "runtime",
      });
    }
    if (url.endsWith("/jobs/data")) {
      submittedBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
      return jsonResponse({ id: "job_data", type: "data", status: "queued", commandId: "build-akshare-market-panel-v7", createdAt: "2026-07-22T00:00:00Z", outputPaths: [] });
    }
    return jsonResponse([]);
  }));

  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<MemoryRouter><QueryClientProvider client={queryClient}><DataManagerWorkspace /></QueryClientProvider></MemoryRouter>);

  expect(await screen.findByText("AkShare A股行情")).toBeInTheDocument();
  const launch = screen.getByRole("button", { name: /提交数据任务/ });
  expect(launch).toBeDisabled();
  fireEvent.click(screen.getByRole("checkbox", { name: /确认允许本次数据任务访问网络/ }));
  fireEvent.click(launch);

  await waitFor(() => expect(submittedBody).toBeDefined());
  expect(submittedBody).toMatchObject({
    commandId: "build-akshare-market-panel-v7",
    parameters: {
      symbols: "000001.SZ,600519.SH",
      allow_network: true,
      output: "runtime/data/v7/silver/market_panel/web_akshare_market_panel.parquet",
    },
  });
  expect(JSON.stringify(submittedBody)).not.toContain("shell");
  expect(await screen.findByText(/已提交 build-akshare-market-panel-v7/)).toBeInTheDocument();
});

function jsonResponse(data: unknown): Response {
  return new Response(JSON.stringify({ status: "ready", data, issues: [] }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}
