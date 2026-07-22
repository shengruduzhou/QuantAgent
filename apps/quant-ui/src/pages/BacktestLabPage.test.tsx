import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, test, vi } from "vitest";
import { BacktestLabPage } from "./BacktestLabPage";

vi.mock("../components/EChart", () => ({
  EChart: () => <div role="img" aria-label="实验净值图" />,
}));

const runs = [
  {
    id: "run-a",
    name: "experiment-a",
    horizon: "short_5d",
    startDate: "2025-01-01",
    endDate: "2025-12-31",
    totalReturn: 0.12,
    annualReturn: 0.15,
    maxDrawdown: 0.08,
    sharpe: 1.1,
    calmar: 1.8,
    turnover: 0.4,
    tradeCount: 100,
    fillCount: 90,
    status: "ready",
    path: "runtime/reports/run-a",
    tags: [],
    capabilities: { equity: true },
  },
  {
    id: "run-b",
    name: "experiment-b",
    horizon: "mid_20d",
    startDate: "2025-02-01",
    endDate: "2025-12-31",
    totalReturn: 0.2,
    annualReturn: 0.23,
    maxDrawdown: 0.1,
    sharpe: 1.4,
    calmar: 2.1,
    turnover: 0.3,
    tradeCount: 80,
    fillCount: 75,
    status: "ready",
    path: "runtime/reports/run-b",
    tags: [],
    capabilities: { equity: true },
  },
];

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("keeps exactly one backtest experiment active", async () => {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const data = url.includes("/equity")
      ? [{ datetime: "2025-01-01", nav: 1 }, { datetime: "2025-01-02", nav: 1.01 }]
      : runs;
    return new Response(JSON.stringify({ status: "ready", data, issues: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }));

  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <BacktestLabPage />
    </QueryClientProvider>,
  );

  const radios = await screen.findAllByRole("radio");
  expect(radios).toHaveLength(2);
  expect(radios[0]).toBeChecked();
  expect(radios[1]).not.toBeChecked();
  expect(screen.getByText(/不再通过复选框同时激活多个实验/)).toBeInTheDocument();

  fireEvent.click(screen.getByText("experiment-b"));
  await waitFor(() => {
    expect(radios[0]).not.toBeChecked();
    expect(radios[1]).toBeChecked();
  });
});
