import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { TPlusOnePage } from "./TPlusOnePage";

vi.mock("../components/EChart", () => ({ EChart: () => <div data-testid="chart" /> }));

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("a take-profit exit that lost money is labelled as a loss, and counts cover the whole source", async () => {
  const analysis = {
    sourceId: "dot_1",
    summary: { pairCount: 1200, pairsShown: 1000, unmeasuredPairs: 3, successRate: 0.49, failureRate: 0.51 },
    pairs: [{ id: "p1", symbol: "002282.SZ", tradeDate: "2026-05-08T00:00:00", mode: "BUY_LOW_SELL_OLD_HIGH", buyPrice: 6.77, sellPrice: 6.75, quantity: 100, netPnl: -0.0038, state: "closed_profit", exitReason: "take_profit_trigger", success: false, issues: [] }],
    byRegime: {},
    verdict: "DO_NOT_ENABLE",
  };
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const data = url.includes("/do-t/sources")
      ? [{ id: "dot_1", name: "intraday_dot_factor_combo", path: "runtime/x", verdict: "DO_NOT_ENABLE", metrics: {} }]
      : url.includes("/do-t/analysis") ? analysis : [];
    return new Response(JSON.stringify({ status: "ready", data, issues: [] }), { status: 200, headers: { "Content-Type": "application/json" } });
  }));

  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter>
        <TPlusOnePage />
      </MemoryRouter>
    </QueryClientProvider>,
  );

  expect(await screen.findByText("止盈触发 · 净亏损")).toBeInTheDocument();
  expect(screen.queryByText("closed_profit")).toBeNull();
  expect(screen.getByText(/明细显示最近 1000 对，统计覆盖全部 · 3 未测量/)).toBeInTheDocument();
});
