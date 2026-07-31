import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
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
  expect(screen.getAllByText("Data Quality").length).toBeGreaterThan(0);
  expect(screen.getAllByText("Risk").length).toBeGreaterThan(0);
  expect(screen.getAllByText("Human Gate").length).toBeGreaterThan(0);
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
  expect(within(primary).queryByRole("option", { name: /5D/ })).not.toBeInTheDocument();
});

test("applies named research presets and exposes declared blend policies", () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({
    status: "empty",
    issues: [],
    data: [],
  }), { status: 200, headers: { "Content-Type": "application/json" } })));
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={queryClient}><StrategyStudioPage /></QueryClientProvider>);

  expect(screen.getByRole("option", { name: "自适应 OOS（以早期 OOS 稳定 RankIC 为主）" })).toBeInTheDocument();
  fireEvent.change(screen.getByLabelText(/研究方案/), { target: { value: "drawdown_first" } });

  expect(screen.getByLabelText(/最小回撤/)).toHaveValue("0.6");
  expect(screen.getByLabelText("周期混合")).toHaveValue("adaptive_oos");
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

  expect(screen.getByText("15 / 64 组搜索预算")).toBeInTheDocument();
  expect(screen.getByLabelText("基本面学习")).toHaveValue("auto");
  fireEvent.click(screen.getByRole("button", { name: "校验" }));

  expect(await screen.findByText("无法提交当前策略")).toBeInTheDocument();
  expect(screen.getByText("primaryHorizon must be included in horizons")).toBeInTheDocument();
  expect(screen.queryByText(/must-not-be-rendered/)).not.toBeInTheDocument();
  expect(screen.queryByText(/marketPanelPath/)).not.toBeInTheDocument();
});

test("turns a missing horizon block into executable repair choices", async () => {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.includes("/strategies/validate") && init?.method === "POST") {
      return new Response(JSON.stringify({
        status: "ready",
        issues: [],
        data: {
          valid: false,
          errors: ["labelsPath: missing requested horizon columns forward_return_60d, forward_return_120d"],
          warnings: [],
          requestedHorizons: [1, 5, 20, 60, 120],
          availableHorizons: [1, 5, 20],
          resolvedInputs: {},
          launch: {
            jobType: "strategy-pipeline",
            commandId: "run-full-real-training-v7",
            parameters: {},
            armed: false,
          },
          issues: [{
            code: "missing_horizon_columns",
            severity: "blocking",
            title: "Labels 缺少研究周期",
            detail: "缺少 60D / 120D",
            evidence: {
              requestedHorizons: [1, 5, 20, 60, 120],
              availableHorizons: [1, 5, 20],
              missingHorizons: [60, 120],
            },
          }],
          decisionCouncil: [{
            id: "data_quality",
            label: "Data Quality",
            responsibility: "PIT and coverage",
            status: "blocked",
            veto: true,
            finding: "Labels 缺少研究周期",
            nextAction: "选择修复方式",
            issueCount: 1,
            issueCodes: ["missing_horizon_columns"],
          }],
        },
      }), { status: 200, headers: { "Content-Type": "application/json" } });
    }
    return new Response(JSON.stringify({
      status: "ready",
      issues: [],
      data: {
        selected: {
          marketPanelPath: "runtime/data/gold/full_universe/adjusted_market_panel.parquet",
          labelsPath: "runtime/data/gold/full_universe/labels.parquet",
          fundamentalsRoot: "runtime/data/v7/silver/fundamentals",
        },
        options: {
          fundamentalsRoot: [{
            field: "fundamentalsRoot",
            path: "runtime/data/v7/silver/fundamentals",
            exists: true,
            isDirectory: true,
            sizeBytes: 0,
            modifiedAt: "2026-01-01T00:00:00Z",
            availableHorizons: [],
          }],
        },
        evidence: [],
        selectionRule: "fixture",
      },
    }), { status: 200, headers: { "Content-Type": "application/json" } });
  }));
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={queryClient}><StrategyStudioPage /></QueryClientProvider>);

  await screen.findByText(/1 个可用 \/ 1 个 canonical 候选/);
  const fundamentals = await screen.findByLabelText(/基本面 PIT 根目录/);
  const listId = fundamentals.getAttribute("list");
  expect(listId).toBeTruthy();
  expect(document.getElementById(listId ?? "")?.querySelector("option")?.getAttribute("value"))
    .toBe("runtime/data/v7/silver/fundamentals");

  fireEvent.click(screen.getByRole("button", { name: "校验" }));
  expect((await screen.findAllByText("Labels 缺少研究周期")).length).toBeGreaterThan(0);
  expect(screen.getByRole("button", { name: "补齐 Labels" })).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: "采用现有周期" }));

  expect(screen.getByLabelText("Horizon 集合")).toHaveValue("1,5,20");
  expect(screen.getByLabelText("主周期")).toHaveValue("5");
});
