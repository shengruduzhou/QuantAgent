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

test("constrains primary horizon to the declared label horizons", () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
    status: "empty",
    issues: [],
    data: [],
  }), { status: 200, headers: { "Content-Type": "application/json" } })));
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={queryClient}><StrategyStudioPage /></QueryClientProvider>);

  const horizonSet = screen.getByLabelText("Horizon 集合");
  const primary = screen.getByLabelText("主周期") as HTMLSelectElement;
  expect(primary.value).toBe("5");
  expect(screen.getByRole("option", { name: "5D · forward_return_5d" })).toBeInTheDocument();

  fireEvent.change(horizonSet, { target: { value: "1,20,60" } });
  expect(primary.value).toBe("1");
  expect(screen.queryByRole("option", { name: /5D/ })).not.toBeInTheDocument();
});

test("shows auto-learning budget and compacts FastAPI validation errors", async () => {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.includes("/strategies/validate") && init?.method === "POST") {
      return new Response(JSON.stringify({
        detail: [{
          type: "value_error",
          loc: ["body"],
          msg: "Value error, primaryHorizon must be included in horizons",
          input: { marketPanelPath: "must-not-be-rendered", labelsPath: "secretly-large-payload" },
        }],
      }), { status: 422, headers: { "Content-Type": "application/json" } });
    }
    return new Response(JSON.stringify({ status: "empty", issues: [], data: [] }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }));
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={queryClient}><StrategyStudioPage /></QueryClientProvider>);

  expect(screen.getByText("50 / 64 组搜索预算")).toBeInTheDocument();
  expect(screen.getByLabelText("基本面学习")).toHaveValue("auto");
  fireEvent.click(screen.getByRole("button", { name: "校验" }));

  expect(await screen.findByText("无法提交当前策略")).toBeInTheDocument();
  expect(screen.getByText("primaryHorizon must be included in horizons")).toBeInTheDocument();
  expect(screen.queryByText(/must-not-be-rendered/)).not.toBeInTheDocument();
  expect(screen.queryByText(/marketPanelPath/)).not.toBeInTheDocument();
});
