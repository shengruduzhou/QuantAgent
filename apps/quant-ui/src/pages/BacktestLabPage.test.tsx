import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, useLocation } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { BacktestLabPage } from "./BacktestLabPage";

vi.mock("../components/EChart", () => ({ EChart: () => <div data-testid="backtest-chart" /> }));

const runs = [
  { id: "run-a", name: "Baseline", horizon: "short_5d", status: "ready", path: "runtime/a", tags: [], totalReturn: 0.1 },
  { id: "run-b", name: "Candidate", horizon: "long_30d", status: "ready", path: "runtime/b", tags: [], totalReturn: 0.2 },
];

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("keeps exactly one active backtest while changing the URL context", async () => {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const data = /\/backtests\/[^/]+\/equity/.test(url)
      ? [{ datetime: "2026-01-01", nav: 1, drawdown: 0, dailyReturn: 0 }]
      : runs;
    return new Response(JSON.stringify({ status: "ready", data, issues: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }));

  function LocationProbe(): JSX.Element {
    return <output data-testid="location">{useLocation().search}</output>;
  }

  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter>
        <BacktestLabPage />
        <LocationProbe />
      </MemoryRouter>
    </QueryClientProvider>,
  );

  const radios = await screen.findAllByRole("radio");
  expect(radios).toHaveLength(2);
  expect(radios[0]).toBeChecked();
  expect(radios[1]).not.toBeChecked();

  fireEvent.click(radios[1]);
  await waitFor(() => expect(radios[1]).toBeChecked());
  expect(radios[0]).not.toBeChecked();
  expect(screen.getByTestId("location")).toHaveTextContent("run=run-b");
});

test("defaults to a run with NAV, never marks an empty run ready, and states cost/benchmark/clock caveats", async () => {
  const caveatRuns = [
    { id: "run-empty", name: "board_chase", status: "ready", path: "runtime/e", tags: [], endDate: "2026-06-11", capabilities: { equity: false }, trustClass: "unclassified", validationStatus: "unverified", timingCanonical: false, quarantineOverlap: ["2025-09-01..2026-05-18", "2026-05-19..2027-12-31"], totalCost: null },
    { id: "run-nav", name: "retrain_plus7", status: "ready", path: "runtime/n", tags: [], endDate: "2026-05-07", totalReturn: 0.36, capabilities: { equity: true }, trustClass: "unclassified", validationStatus: "unverified", timingCanonical: false, quarantineOverlap: ["2025-09-01..2026-05-18"], totalCost: 64823.97, totalCostBasis: "matched_round_trip_fees_only_pre_fix", benchmark: { mode: "universe_equal_weight", source: "baseline_protocol_export", caveat: "x" } },
  ];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const data = /\/backtests\/[^/]+\/equity/.test(url) ? [{ datetime: "2026-01-01", nav: 1, drawdown: 0, dailyReturn: 0 }] : caveatRuns;
    return new Response(JSON.stringify({ status: "ready", data, issues: [] }), { status: 200 });
  }));
  render(
    <QueryClientProvider client={new QueryClient({ defaultOptions: { queries: { retry: false } } })}>
      <MemoryRouter><BacktestLabPage /></MemoryRouter>
    </QueryClientProvider>,
  );
  const radios = await screen.findAllByRole("radio");
  // The NAV-backed run is the default subject, not runs[0].
  await waitFor(() => expect(radios[1]).toBeChecked());
  expect(radios[0]).not.toBeChecked();
  const emptyRow = radios[0].closest("tr") as HTMLElement;
  expect(within(emptyRow).getByText("NO NAV")).toBeInTheDocument();
  expect(within(emptyRow).queryByText(/^ready$/i)).not.toBeInTheDocument();
  expect(within(emptyRow).getByText("未记录")).toBeInTheDocument();
  expect(within(emptyRow).getByText("overlaps quarantined ×2")).toBeInTheDocument();
  const navRow = radios[1].closest("tr") as HTMLElement;
  expect(within(navRow).getByText("excl. slippage")).toBeInTheDocument();
  expect(within(navRow).getByText("pre-fix clock")).toBeInTheDocument();
  expect(within(navRow).getByText("宇宙等权 (universe_equal_weight)")).toBeInTheDocument();
  expect(within(navRow).getByText("超额高估 · 含不可交易")).toBeInTheDocument();
  expect(within(navRow).getByText("unclassified")).toBeInTheDocument();
  expect(screen.getByText(/不可引用 · trust: unclassified/)).toHaveTextContent("cost excludes slippage");
  expect(screen.getByText(/不可引用/)).toHaveTextContent("含不可交易标的");
});
