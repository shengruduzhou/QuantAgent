import { cleanup, render, screen, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, expect, test, vi } from "vitest";
import { ResearchRunsPage } from "./ResearchRunsPage";

vi.mock("../../components/EChart", () => ({
  EChart: () => <div data-testid="nav-chart" />,
}));

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const RUN = {
  runId: "run_abc123",
  strategyId: "alpha-one",
  strategyVersion: "20260801T000000Z",
  strategyName: "A股多因子",
  jobId: "job_1",
  outputDir: "runtime/reports/strategy_studio/alpha-one/runs/run_1",
  createdAt: "2026-08-01T00:00:00+00:00",
};

function jsonResponse(data: unknown, status = "ready"): Response {
  return new Response(JSON.stringify({ status, data, issues: [] }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function stubApi(routes: Record<string, unknown>): void {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input instanceof Request ? input.url : input);
    const match = Object.keys(routes).find((key) => url.includes(key));
    return jsonResponse(match ? routes[match] : []);
  }));
  // The runs page opens a log stream for live jobs only; stub it so jsdom
  // does not reject the constructor.
  vi.stubGlobal("EventSource", class {
    close(): void {}
    addEventListener(): void {}
  });
}

function renderPage(): void {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={queryClient}><ResearchRunsPage /></QueryClientProvider>);
}

test("explains what is missing when no run has been launched yet", async () => {
  stubApi({ "/strategies/runs": [], "/strategies": [] });

  renderPage();

  expect(await screen.findByText("尚无研究运行")).toBeInTheDocument();
  // An empty state must name the next action, not just say "no data".
  expect(screen.getByText(/在策略实验室配置并启动一次闭环/)).toBeInTheDocument();
});

test("presents a failed acceptance gate as a conclusion with its reasons", async () => {
  stubApi({
    "/strategies/runs/run_abc123": {
      ...RUN,
      job: { id: "job_1", type: "strategy-pipeline", status: "succeeded", commandId: "run-full-real-training-v7", createdAt: RUN.createdAt, outputPaths: [], terminal: true, progress: 1 },
      result: {
        status: "complete",
        outputDir: RUN.outputDir,
        conclusion: {
          outcome: "not_accepted",
          headline: "流程完整跑通，但验收闸门未通过",
          reasons: ["single_factor_dominance: 实测 0.7154，阈值 <= 0.6"],
          remediation: "这是研究结论而不是故障。",
          promotable: false,
        },
        acceptance: {
          failures: ["single_factor_dominance_too_high"],
          gates: [
            { name: "rank_ic_mean", passed: true, actual: 0.112, threshold: "> 0.0" },
            { name: "single_factor_dominance", passed: false, actual: 0.7154, threshold: "<= 0.6", reason: "single_factor_dominance_too_high" },
          ],
          passedCount: 1,
          totalCount: 2,
          sourcePath: "runtime/.../acceptance_report.json",
        },
        stages: [{ id: "training", label: "滚动样本外训练", present: true, sizeBytes: 1024 }],
        artifacts: [],
      },
    },
    "/strategies/runs": [RUN],
    "/strategies": [{ id: "alpha-one", name: "A股多因子", version: "v1", versionCount: 2, runCount: 1 }],
  });

  renderPage();

  expect((await screen.findAllByText("未通过验收")).length).toBeGreaterThan(0);
  expect(screen.getByText(/single_factor_dominance: 实测 0.7154/)).toBeInTheDocument();
  // The gate table must show the measured value next to the threshold it broke.
  const table = screen.getByRole("table");
  expect(within(table).getByText("0.7154")).toBeInTheDocument();
  expect(within(table).getByText("<= 0.6")).toBeInTheDocument();
});

test("shows a research rejection as a verdict, never as a crash", async () => {
  stubApi({
    "/strategies/runs/run_abc123": {
      ...RUN,
      job: {
        id: "job_1", type: "strategy-pipeline", status: "rejected", commandId: "run-full-real-training-v7",
        createdAt: RUN.createdAt, outputPaths: [], terminal: true, progress: 1, exitCode: 3,
        verdict: {
          verdict: "rejected", code: "overfitting_governance_rejected",
          title: "候选组合被过拟合治理闸门否决", stage: "portfolio_selection",
          reasons: ["pbo=0.5429 exceeds 0.2500"],
          remediation: "闸门不允许事后放宽。",
        },
        failure: null,
      },
      result: {
        status: "rejected",
        outputDir: RUN.outputDir,
        conclusion: {
          outcome: "rejected", headline: "候选组合被过拟合治理闸门否决",
          reasons: ["pbo=0.5429 exceeds 0.2500"], remediation: "闸门不允许事后放宽。", promotable: false,
        },
        stages: [], artifacts: [],
      },
    },
    "/strategies/runs": [RUN],
    "/strategies": [],
  });

  renderPage();

  expect((await screen.findAllByText("研究闸门否决")).length).toBeGreaterThan(0);
  expect(screen.getAllByText(/pbo=0.5429 exceeds 0.2500/).length).toBeGreaterThan(0);
  expect(screen.getByText(/这是结论，不是故障/)).toBeInTheDocument();
});

test("a failed run offers its diagnosis, remediation and a retry", async () => {
  stubApi({
    "/strategies/runs/run_abc123": {
      ...RUN,
      job: {
        id: "job_1", type: "strategy-pipeline", status: "failed", commandId: "run-full-real-training-v7",
        createdAt: RUN.createdAt, outputPaths: [], terminal: true, canRetry: true, exitCode: 1,
        failure: {
          code: "out_of_memory", title: "进程内存或显存耗尽",
          detail: "torch.cuda.OutOfMemoryError: CUDA out of memory",
          remediation: "缩小研究范围或等待显存释放后重试。",
          retryable: true, logTail: ["line one", "line two"], exitCode: 1,
        },
      },
      result: { status: "partial", outputDir: RUN.outputDir, conclusion: { outcome: "incomplete", headline: "运行未走完全部阶段，结论不完整", reasons: [], remediation: "查看诊断。", promotable: false }, stages: [], artifacts: [] },
    },
    "/strategies/runs": [RUN],
    "/strategies": [],
  });

  renderPage();

  expect(await screen.findByText("进程内存或显存耗尽")).toBeInTheDocument();
  expect(screen.getByText("缩小研究范围或等待显存释放后重试。")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /以相同参数重试/ })).toBeEnabled();
});

test("a job adopted after a restart says so rather than claiming it failed", async () => {
  stubApi({
    "/strategies/runs/run_abc123": {
      ...RUN,
      job: {
        id: "job_1", type: "strategy-pipeline", status: "running", commandId: "run-full-real-training-v7",
        createdAt: RUN.createdAt, outputPaths: [], adopted: true, progress: 0.48, stage: "training",
        message: "walk-forward training completed",
      },
      result: { status: "partial", outputDir: RUN.outputDir, stages: [], artifacts: [] },
    },
    "/strategies/runs": [RUN],
    "/strategies": [],
  });

  renderPage();

  expect(await screen.findByText(/该任务在 API 重启后被重新接管/)).toBeInTheDocument();
});

test("council findings are shown with their evidence, and absence is not a pass", async () => {
  stubApi({
    "/council/review/run/run_abc123": {
      subject: { type: "strategy_run", id: "run_abc123", path: RUN.outputDir, strategyName: "A股多因子", outcome: "not_accepted" },
      roles: [
        { id: "execution_realism", label: "执行可实现性", domain: "成本、T+1", vetoScope: "回测主张", veto: true },
        { id: "data_quality", label: "数据质量", domain: "PIT", vetoScope: "整条链", veto: true },
      ],
      findings: [
        {
          roleId: "execution_realism", verdict: "warn", headline: "过半委托被交易约束拒绝",
          detail: "644 / 672 笔委托被跳过。",
          evidence: { orderCount: 28, skippedOrderCount: 644 },
          nextAction: "降低换手或调整标的池",
        },
        {
          roleId: "data_quality", verdict: "unknown", headline: "缺少 PIT / 合成数据判定",
          detail: "验收报告没有写入该字段。", evidence: { noPitViolations: null }, nextAction: "确认验收阶段",
        },
      ],
      decision: { state: "BLOCKED", summary: "1 个角色警告", blockedRoles: [], unknownRoles: ["data_quality"] },
      overrides: [],
    },
    "/strategies/runs/run_abc123": {
      ...RUN,
      job: { id: "job_1", type: "strategy-pipeline", status: "succeeded", commandId: "run-full-real-training-v7", createdAt: RUN.createdAt, outputPaths: [], terminal: true },
      result: { status: "complete", outputDir: RUN.outputDir, stages: [], artifacts: [] },
    },
    "/strategies/runs": [RUN],
    "/strategies": [],
  });

  renderPage();

  expect(await screen.findByText("过半委托被交易约束拒绝")).toBeInTheDocument();
  // The numbers behind the verdict must be visible, not just the badge.
  expect(screen.getByText("644")).toBeInTheDocument();
  // Missing evidence reads as "证据缺失", never as a pass.
  expect(screen.getByText("证据缺失")).toBeInTheDocument();
  expect(screen.queryByText("通过")).not.toBeInTheDocument();
});

test("a panel survives an endpoint returning an unexpected shape", async () => {
  // One malformed response must degrade its own panel, not blank the page.
  stubApi({
    "/strategies/runs/run_abc123": {
      ...RUN,
      job: { id: "job_1", type: "strategy-pipeline", status: "succeeded", commandId: "x", createdAt: RUN.createdAt, outputPaths: [], terminal: true },
      result: { status: "complete", outputDir: RUN.outputDir },
    },
    "/strategies/runs": [RUN],
  });

  renderPage();

  expect(await screen.findByRole("heading", { name: "研究运行" })).toBeInTheDocument();
});
