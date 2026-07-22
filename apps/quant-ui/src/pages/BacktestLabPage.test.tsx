import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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
