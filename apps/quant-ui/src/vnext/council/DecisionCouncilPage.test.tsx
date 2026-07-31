import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { DecisionCouncilPage } from "./DecisionCouncilPage";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const ROSTER = {
  roles: [
    { id: "data_quality", label: "数据质量", domain: "PIT 完整性", vetoScope: "输入数据不可信时阻塞整条链", veto: true },
    { id: "fusion_search", label: "搜索统计", domain: "试验计数、PBO", vetoScope: "融合候选晋级", veto: true },
    { id: "execution_realism", label: "执行可实现性", domain: "成本、T+1", vetoScope: "回测可实现性主张", veto: true },
  ],
  thresholds: { maxPbo: 0.5, minObservations: 60 },
  protocol: "证据缺失记为 unknown，不记为通过。",
};

const RUN = {
  id: "run-1",
  name: "fixture_search",
  path: "runtime/reports/fusion/fixture_search",
  generatedAt: "2026-01-06T00:00:00+00:00",
  contentHash: "abcdef0123456789",
  nTrials: 4,
  pbo: 0.18,
  benchmarkMode: "index:000300.SH",
  horizonDays: 5,
  topK: 30,
  transactionCostBps: 8,
  factorNames: ["alpha001"],
  frontierSize: 1,
  candidateCount: 4,
  evaluatedCandidateCount: 4,
  foldCount: 3,
};

function review(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    subject: {
      type: "fusion_run", id: "run-1", path: RUN.path,
      candidateId: "ic_weighted", candidateLabel: "IC 加权",
    },
    roles: ROSTER.roles,
    thresholds: ROSTER.thresholds,
    findings: [
      {
        roleId: "data_quality", verdict: "pass", headline: "输入口径可追溯",
        detail: "基准 index:000300.SH。",
        evidence: { benchmarkMode: "index:000300.SH", observations: 120 },
        nextAction: "无",
      },
      {
        roleId: "fusion_search", verdict: "unknown", headline: "试验次数未记录",
        detail: "没有 nTrials 就无法收缩 Sharpe。",
        evidence: { nTrials: null, pbo: 0.18 },
        nextAction: "重新运行搜索",
      },
      {
        roleId: "execution_realism", verdict: "blocked", headline: "成本假设不成立",
        detail: "成本 0 bps 低于最低要求。",
        evidence: { transactionCostBps: 0 },
        nextAction: "使用真实成本重跑",
      },
    ],
    decision: {
      state: "BLOCKED",
      summary: "1 个角色否决：execution_realism",
      blockedRoles: ["execution_realism"],
      unknownRoles: ["fusion_search"],
      warnedRoles: [],
      overriddenRoles: [],
    },
    overrides: [],
    ...overrides,
  };
}

function stubApi(payload = review()): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const body = (data: unknown, status = "ready"): Response => new Response(
      JSON.stringify({ status, data, issues: [] }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
    if (url.includes("/council/roster")) return body(ROSTER);
    if (url.includes("/council/review")) return body(payload);
    if (url.includes("/council/overrides")) return body({ recordedAt: "2026-01-06T01:00:00+00:00" });
    if (url.includes("/fusion/runs")) return body([RUN]);
    return body({});
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function renderPage(): void {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>
        <DecisionCouncilPage />
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

test("renders one verdict per role with its veto scope", async () => {
  stubApi();
  renderPage();
  expect(await screen.findByText("多 Agent 决策议事会")).toBeInTheDocument();
  expect(await screen.findByText("数据质量")).toBeInTheDocument();
  expect(screen.getByText("输入数据不可信时阻塞整条链")).toBeInTheDocument();
  expect(screen.getByText("融合候选晋级")).toBeInTheDocument();
});

test("an unknown verdict is shown as insufficient evidence, never as a pass", async () => {
  stubApi();
  renderPage();
  await screen.findByText("试验次数未记录");
  const card = screen.getByText("试验次数未记录").closest("article");
  expect(within(card as HTMLElement).getByText("证据不足")).toBeInTheDocument();
  expect(within(card as HTMLElement).queryByText("通过")).toBeNull();
});

test("each verdict exposes the evidence it was computed from", async () => {
  stubApi();
  renderPage();
  await screen.findByText("输入口径可追溯");
  const card = screen.getByText("输入口径可追溯").closest("article");
  expect(within(card as HTMLElement).getByText("benchmarkMode")).toBeInTheDocument();
  expect(within(card as HTMLElement).getByText("index:000300.SH")).toBeInTheDocument();
});

test("an override keeps the original verdict visible next to it", async () => {
  stubApi(review({
    findings: [{
      roleId: "execution_realism", verdict: "blocked", headline: "成本假设不成立",
      detail: "成本 0 bps 低于最低要求。",
      evidence: { transactionCostBps: 0 },
      nextAction: "使用真实成本重跑",
      override: {
        verdict: "warn",
        reason: "成本将在下游 A 股回测中重新施加。",
        author: "研究员甲",
        recordedAt: "2026-01-06T01:00:00+00:00",
        replacedVerdict: "blocked",
      },
    }],
    decision: {
      state: "PROMOTABLE_WITH_WARNINGS", summary: "1 个角色提出保留意见",
      blockedRoles: [], unknownRoles: [], warnedRoles: ["execution_realism"],
      overriddenRoles: ["execution_realism"],
    },
    overrides: [{
      subjectType: "fusion_run", subjectId: "run-1", roleId: "execution_realism",
      verdict: "warn", reason: "成本将在下游 A 股回测中重新施加。",
      author: "研究员甲", recordedAt: "2026-01-06T01:00:00+00:00",
    }],
  }));
  renderPage();
  await screen.findByText("成本假设不成立");
  expect(screen.getByText("否决 → 保留意见")).toBeInTheDocument();
  expect(screen.getAllByText(/研究员甲/).length).toBeGreaterThan(0);
});

test("an override cannot be submitted without an author and a substantive reason", async () => {
  stubApi();
  renderPage();
  await screen.findByText("成本假设不成立");
  const card = screen.getByText("成本假设不成立").closest("article") as HTMLElement;
  fireEvent.click(within(card).getByRole("button", { name: /人工推翻/ }));
  const submit = within(card).getByRole("button", { name: "记录推翻" });
  expect(submit).toBeDisabled();
  fireEvent.change(within(card).getByLabelText("决策人"), { target: { value: "研究员甲" } });
  fireEvent.change(within(card).getByLabelText(/理由/), { target: { value: "太短" } });
  expect(submit).toBeDisabled();
  fireEvent.change(within(card).getByLabelText(/理由/), {
    target: { value: "成本将在下游 A 股回测中重新施加，此处只做因子排序。" },
  });
  await waitFor(() => expect(submit).not.toBeDisabled());
});

test("submitting an override posts the subject, role, reason and author", async () => {
  const fetchMock = stubApi();
  renderPage();
  await screen.findByText("成本假设不成立");
  const card = screen.getByText("成本假设不成立").closest("article") as HTMLElement;
  fireEvent.click(within(card).getByRole("button", { name: /人工推翻/ }));
  fireEvent.change(within(card).getByLabelText("决策人"), { target: { value: "研究员甲" } });
  fireEvent.change(within(card).getByLabelText(/理由/), {
    target: { value: "成本将在下游 A 股回测中重新施加，此处只做因子排序。" },
  });
  fireEvent.click(within(card).getByRole("button", { name: "记录推翻" }));
  await waitFor(() => {
    const call = fetchMock.mock.calls.find(
      ([url, init]) => String(url).includes("/council/overrides") && (init as RequestInit)?.method === "POST",
    );
    expect(call).toBeDefined();
    const body = JSON.parse(String((call?.[1] as RequestInit).body));
    expect(body.roleId).toBe("execution_realism");
    expect(body.subjectType).toBe("fusion_run");
    expect(body.author).toBe("研究员甲");
    expect(body.reason.length).toBeGreaterThanOrEqual(8);
  });
});

test("shows an actionable empty state when there is nothing to review", async () => {
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.includes("/council/roster")) {
      return new Response(JSON.stringify({ status: "ready", data: ROSTER, issues: [] }), { status: 200 });
    }
    return new Response(JSON.stringify({ status: "empty", data: [], issues: [] }), { status: 200 });
  }));
  renderPage();
  expect(await screen.findByText("没有可审查的研究产物")).toBeInTheDocument();
});
