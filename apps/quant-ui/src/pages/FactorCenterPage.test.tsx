import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import type { Factor } from "../api/types";
import { FactorCenterPage } from "./FactorCenterPage";

vi.mock("../components/EChart", () => ({
  EChart: () => <div role="img" aria-label="因子图表" />,
}));

const factors: Factor[] = [
  {
    name: "alpha_active",
    displayName: "Active Alpha",
    category: "alpha101",
    description: "active factor",
    direction: "positive",
    parameters: {},
    dataSource: ["market"],
    requiredColumns: ["close"],
    pitSafe: true,
    usedInTraining: true,
    usedInSelection: false,
    usedInTiming: false,
    usedInRisk: false,
    lifecycle: "accepted",
    sourceKind: "registry",
  },
  {
    name: "alpha_bad",
    displayName: "Rejected Alpha",
    category: "alpha101",
    description: "rejected factor",
    direction: "positive",
    parameters: {},
    dataSource: ["market"],
    requiredColumns: ["close"],
    pitSafe: true,
    usedInTraining: false,
    usedInSelection: false,
    usedInTiming: false,
    usedInRisk: false,
    lifecycle: "rejected",
    sourceKind: "registry",
  },
  {
    name: "alpha_pending",
    displayName: "Pending Alpha",
    category: "alpha101",
    description: "pending factor",
    direction: "unknown",
    parameters: {},
    dataSource: ["market"],
    requiredColumns: ["close"],
    pitSafe: null,
    usedInTraining: false,
    usedInSelection: false,
    usedInTiming: false,
    usedInRisk: false,
    lifecycle: "research",
    sourceKind: "registry",
  },
];

const metrics = {
  factorName: "alpha_active",
  ic: 0.03,
  rankIc: 0.02,
  icir: 0.8,
  coverage: 0.9,
  stability: 0.7,
  verdict: "accepted",
  bestHorizon: "20D",
  regimeIc: {},
  icSeries: [],
  rankIcSeries: [],
  quantileReturns: [],
  longShortEquity: [],
  decay: [],
  availability: { trades: false },
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("separates active, rejected and unevaluated factors", async () => {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    let data: unknown;
    if (url.includes("/backtest")) {
      data = metrics;
    } else {
      const detailName = factors.find((factor) => url.includes(`/factors/${factor.name}`));
      data = detailName ?? factors;
    }
    return new Response(JSON.stringify({ status: "ready", data, issues: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }));

  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>
        <FactorCenterPage />
      </QueryClientProvider>
    </MemoryRouter>,
  );

  expect(await screen.findByText("Active Alpha")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /已启用 1/ })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /已剔除 1/ })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /待评估 1/ })).toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: /已剔除 1/ }));
  await waitFor(() => {
    expect(screen.getByText("Rejected Alpha")).toBeInTheDocument();
    expect(screen.queryByText("Active Alpha")).not.toBeInTheDocument();
    expect(screen.queryByText("Pending Alpha")).not.toBeInTheDocument();
  });

  expect(screen.getByText(/已剔除因子不会因为页面存在而自动进入训练/)).toBeInTheDocument();
});
