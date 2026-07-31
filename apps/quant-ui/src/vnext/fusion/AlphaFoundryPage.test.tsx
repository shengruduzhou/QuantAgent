import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { AlphaFoundryPage } from "./AlphaFoundryPage";
import { trialCount, DEFAULT_SEARCH_DRAFT } from "./FusionSearchForm";

// ECharts needs a real canvas; the assertions here are about the data contract,
// not the pixels, so the chart is stubbed the same way other page tests do it.
vi.mock("../../components/EChart", () => ({
  EChart: ({ ariaLabel }: { ariaLabel?: string }) => (
    <div role="img" aria-label={ariaLabel ?? "chart"} />
  ),
}));

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

function candidate(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    id: "ic_weighted",
    label: "IC 加权",
    scheme: "ic_weighted",
    isControl: false,
    weights: { alpha001: 0.7, alpha002: 0.3 },
    metrics: {
      observations: 120,
      annualReturn: 0.19,
      excessReturn: 0.14,
      benchmarkAnnualReturn: 0.05,
      maxDrawdown: 0.11,
      sharpe: 1.1,
      calmar: 1.4,
      averageTurnover: 0.22,
      costDrag: 0.004,
      winRate: 0.55,
      robustness: 0.68,
    },
    robustnessBreakdown: {
      foldConsistency: 0.7,
      overfittingResistance: 0.82,
      deflatedSharpeProbability: 0.61,
      regimeConsistency: 0.66,
      pbo: 0.18,
    },
    folds: [{
      foldIndex: 0,
      trainStart: "2024-01-02",
      trainEnd: "2025-06-30",
      testStart: "2025-07-10",
      testEnd: "2025-12-31",
      weights: { alpha001: 0.7, alpha002: 0.3 },
      metrics: { observations: 120, excessReturn: 0.14 },
    }],
    onFrontier: true,
    preferenceRank: 0,
    preferenceScore: 0.71,
    ...overrides,
  };
}

const SUMMARY = {
  generatedAt: "2026-01-06T00:00:00+00:00",
  nTrials: 3,
  pbo: 0.18,
  benchmarkMode: "index:000300.SH",
  horizonDays: 5,
  topK: 30,
  transactionCostBps: 8,
  factorNames: ["alpha001", "alpha002"],
  foldWindows: [{
    foldIndex: "0",
    trainStart: "2024-01-02",
    trainEnd: "2025-06-30",
    testStart: "2025-07-10",
    testEnd: "2025-12-31",
  }],
  frontier: ["ic_weighted"],
  preferenceWeights: { excessReturn: 0.4, annualReturn: 0.2, maxDrawdown: 0.25, robustness: 0.15 },
  candidateCount: 3,
  evaluatedCandidateCount: 3,
};

const RUN = {
  id: "run-1",
  name: "fixture_search",
  path: "runtime/reports/fusion/fixture_search",
  generatedAt: "2026-01-06T00:00:00+00:00",
  contentHash: "abcdef0123456789",
  nTrials: 3,
  pbo: 0.18,
  benchmarkMode: "index:000300.SH",
  horizonDays: 5,
  topK: 30,
  transactionCostBps: 8,
  factorNames: ["alpha001", "alpha002"],
  frontierSize: 1,
  candidateCount: 3,
  evaluatedCandidateCount: 3,
  foldCount: 1,
};

const CANDIDATES = [
  candidate(),
  candidate({
    id: "equal",
    label: "等权基线",
    scheme: "equal",
    isControl: true,
    onFrontier: false,
    preferenceRank: null,
    preferenceScore: null,
    metrics: {
      observations: 120, annualReturn: 0.09, excessReturn: 0.04, benchmarkAnnualReturn: 0.05,
      maxDrawdown: 0.06, sharpe: 0.7, calmar: 1.1, averageTurnover: 0.18, costDrag: 0.003,
      winRate: 0.52, robustness: 0.55,
    },
  }),
];

function stubApi(options: {
  runs?: unknown[];
  candidates?: unknown[];
  summary?: unknown;
  nav?: unknown[];
} = {}): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    const body = (data: unknown, status = "ready"): Response => new Response(
      JSON.stringify({ status, data, issues: [] }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
    if (url.includes("/fusion/runs") && url.includes("/nav")) {
      return body(options.nav ?? []);
    }
    if (url.match(/\/fusion\/runs\/[^/]+$/)) {
      return body({
        id: "run-1",
        path: RUN.path,
        summary: options.summary ?? SUMMARY,
        manifest: {},
        ranking: [{ id: "ic_weighted", preferenceScore: 0.71, contributions: {} }],
        candidates: options.candidates ?? CANDIDATES,
      });
    }
    if (url.includes("/fusion/runs")) {
      const runs = options.runs ?? [RUN];
      return body(runs, runs.length ? "ready" : "empty");
    }
    if (url.includes("/strategies/defaults")) {
      return body({ selected: {}, options: {}, evidence: [] });
    }
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
        <AlphaFoundryPage />
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

test("renders the four declared objectives from persisted search artifacts", async () => {
  stubApi();
  renderPage();
  expect(await screen.findByText("因子融合工场")).toBeInTheDocument();
  // Trial count comes from the run, not from the form.
  await waitFor(() => expect(screen.getAllByText("3").length).toBeGreaterThan(0));
  expect(screen.getAllByText("14.0%").length).toBeGreaterThan(0);
  expect(screen.getAllByText("11.0%").length).toBeGreaterThan(0);
  // PBO is reported both in the metric strip and in the candidate evidence.
  expect(screen.getAllByText("0.18").length).toBeGreaterThan(0);
});

test("marks control candidates so they cannot be read as a result", async () => {
  stubApi();
  renderPage();
  await screen.findByText("等权基线");
  const row = screen.getByText("等权基线").closest("tr");
  expect(row).not.toBeNull();
  expect(row).toHaveAttribute("data-control", "true");
  expect(within(row as HTMLElement).getByText("对照")).toBeInTheDocument();
});

test("hiding controls removes them from the ledger", async () => {
  stubApi();
  renderPage();
  await screen.findByText("等权基线");
  const ledger = () => document.querySelector(".foundry-ledger") as HTMLElement;
  expect(within(ledger()).getByText("等权基线")).toBeInTheDocument();
  fireEvent.click(screen.getByLabelText("显示对照组", { exact: false }));
  await waitFor(() => expect(within(ledger()).queryByText("等权基线")).toBeNull());
  expect(within(ledger()).getByText("IC 加权")).toBeInTheDocument();
});

test("comparison selection is capped at four candidates", async () => {
  const many = Array.from({ length: 6 }, (_, index) => candidate({
    id: `cand_${index}`,
    label: `候选 ${index}`,
    onFrontier: false,
  }));
  stubApi({ candidates: many });
  renderPage();
  await screen.findAllByText("候选 0");
  for (let index = 0; index < 4; index += 1) {
    fireEvent.click(screen.getByLabelText(`对比 候选 ${index}`));
  }
  expect(screen.getByLabelText("对比 候选 4")).toBeDisabled();
  expect(screen.getByLabelText("对比 候选 0")).not.toBeDisabled();
});

test("shows an actionable empty state instead of fabricated metrics", async () => {
  stubApi({ runs: [] });
  renderPage();
  expect(await screen.findByText("尚无因子融合搜索产物")).toBeInTheDocument();
  expect(screen.getByText(/不产生任何订单/)).toBeInTheDocument();
});

test("a missing PBO estimate is reported as no evidence, never as a pass", async () => {
  stubApi({
    summary: { ...SUMMARY, pbo: null },
    candidates: [candidate({
      robustnessBreakdown: {
        foldConsistency: 0.7,
        overfittingResistance: 0.5,
        deflatedSharpeProbability: 0.61,
        regimeConsistency: 0.66,
        pbo: null,
      },
    })],
  });
  renderPage();
  await screen.findByText("因子融合工场");
  await waitFor(() => expect(screen.getAllByText("无估计").length).toBeGreaterThan(0));
  expect(screen.getByText(/不按通过计入/)).toBeInTheDocument();
});

test("launching a search posts the governed command without a trial count", async () => {
  const fetchMock = stubApi();
  renderPage();
  await screen.findByText("因子融合工场");
  fireEvent.click(screen.getByRole("button", { name: /启动融合搜索/ }));
  await waitFor(() => {
    const launch = fetchMock.mock.calls.find(
      ([url]) => String(url).includes("/jobs/fusion-search"),
    );
    expect(launch).toBeDefined();
    const body = JSON.parse(String((launch?.[1] as RequestInit).body));
    expect(body.commandId).toBe("search-factor-fusion");
    expect(Object.keys(body.parameters)).not.toContain("n_trials");
    expect(body.parameters.factor_panel_path).toBeTruthy();
  });
});

test("trial count is derived from the enumerated search space", () => {
  // 1 equal + 3 fitted + genetic + controls + baselines
  expect(trialCount(DEFAULT_SEARCH_DRAFT)).toBe(1 + 4 + 8 + 6);
  expect(trialCount({ ...DEFAULT_SEARCH_DRAFT, includeGenetic: false })).toBe(1 + 3 + 8 + 6);
  expect(
    trialCount({ ...DEFAULT_SEARCH_DRAFT, randomControls: 0, singleFactorBaselines: 0 }),
  ).toBe(1 + 4);
});

test("adding controls raises the declared trial count shown before launch", async () => {
  stubApi();
  renderPage();
  await screen.findByText("因子融合工场");
  const before = screen.getByText(String(trialCount(DEFAULT_SEARCH_DRAFT)));
  expect(before).toBeInTheDocument();
  const controls = screen.getByLabelText(/随机对照数/);
  fireEvent.change(controls, { target: { value: "12" } });
  await waitFor(() =>
    expect(
      screen.getByText(String(trialCount({ ...DEFAULT_SEARCH_DRAFT, randomControls: 12 }))),
    ).toBeInTheDocument());
});
