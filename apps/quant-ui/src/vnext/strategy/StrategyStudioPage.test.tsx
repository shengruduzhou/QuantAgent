import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, test, vi } from "vitest";
import { StrategyStudioPage } from "./StrategyStudioPage";

vi.mock("../../components/EChart", () => ({
  EChart: () => <div data-testid="objective-radar" />,
}));

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

test("validates, arms and exposes the governed decision council", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
    status: "ready",
    issues: [],
    data: {
      valid: true,
      errors: [],
      warnings: ["research only"],
      resolvedInputs: {
        marketPanelPath: "runtime/data/v7/silver/market_panel/market_panel.parquet",
        labelsPath: "runtime/data/v7/gold/labels/labels.parquet",
      },
      launch: {
        jobType: "strategy-pipeline",
        commandId: "run-full-real-training-v7",
        parameters: {},
        armed: true,
      },
      decisionCouncil: [
        { id: "data_quality", label: "Data Quality", responsibility: "PIT and coverage", status: "ready", veto: true },
        { id: "risk", label: "Risk", responsibility: "Drawdown and kill switch", status: "ready", veto: true },
        { id: "human_gate", label: "Human Gate", responsibility: "Operator approval", status: "approved", veto: true },
      ],
    },
  }), { status: 200, headers: { "Content-Type": "application/json" } })));

  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={queryClient}><StrategyStudioPage /></QueryClientProvider>);

  expect(screen.getByRole("heading", { name: "策略实验室" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "启动闭环" })).toBeDisabled();

  fireEvent.click(screen.getByRole("checkbox"));
  expect(screen.getByRole("button", { name: "启动闭环" })).toBeEnabled();

  fireEvent.click(screen.getByRole("button", { name: "校验" }));
  expect(await screen.findByText("Schema 与路径校验通过")).toBeInTheDocument();
  expect(screen.getByText("Data Quality")).toBeInTheDocument();
  expect(screen.getByText("Risk")).toBeInTheDocument();
  expect(screen.getByText("Human Gate")).toBeInTheDocument();
});
