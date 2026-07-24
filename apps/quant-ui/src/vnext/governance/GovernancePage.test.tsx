import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { GovernancePage } from "./GovernancePage";

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

const NOT_READY: unknown = {
  shadow: {
    status: "ready",
    decision: "SHADOW_TEST_ACCUMULATING",
    validDays: 2,
    requiredDays: 7,
    validDates: ["2026-07-21", "2026-07-22"],
    excludedDates: [{ date: "2026-07-17", reason: "data_status=FAILED;failed_job_count=1" }],
    nextExpectedValidDate: "2026-07-23",
    ledgerChainValid: true,
    ledgerRecordsTotal: 11,
    fidelityCertificatePasses: true,
    fidelityCertificateHash: "37193bb82a477",
    unblindOrNonRoutineAccesses: 0,
    certificateWritten: false,
  },
  s4: {
    status: "ready",
    decision: "S4_BATCH_REPLAY_READY",
    exactReproduction: true,
    deterministic: true,
    archivedInputsComplete: true,
    refitCutoffsReplayed: 26,
    semanticsChanged: false,
    freshAccess: false,
    reverified: true,
    codeOrTraceHashChanged: false,
  },
  u0: {
    status: "ready",
    dataReadinessState: "FULL_UNIVERSE_DATA_NOT_READY_COVERAGE",
    trainingPermitted: false,
    gatePass: { integration: true, provider: true, coverage: false, pit: false },
    coverageByBoard: { SH_Main: 1562, SZ_Main: 1589, ChiNext: 877 },
    boardsAbsent: ["STAR", "BSE"],
    blockedByData: 920,
    coverageBacklogFetchable: 938,
    retryClassCounts: { OK: 4030, FETCHABLE_NOT_PROBED: 938, NOT_PROBED: 847, NO_RELIABLE_HISTORY: 73 },
    providerFailures: 73,
    pitGate: { st_history: "BLOCKED_BY_DATA", suspension_history: "BLOCKED_BY_DATA", delisting_status: "BLOCKED_BY_DATA", board_price_limits: "PARTIAL(current-snapshot)", ipo_special_limit: "PRESENT", corporate_actions: "BLOCKED_BY_DATA" },
    pitFieldAvailability: {},
    survivorshipBias: { delisted_total: 358, delisted_with_bar_history: 226, delisted_with_delisting_date: 0, delisted_fraction_of_master: 0.0608 },
    starBseProbe: { STAR: "FETCHABLE_NOT_PROBED", BSE: "BSE_920x_FETCHABLE_NOT_PROBED" },
    coveredBarHistory: 4030,
    backfill: { masterSecurities: 5888, panelSymbols: 4030, missingSymbols: 1858, stagedBackfillFiles: 159 },
  },
  u0BarPit: {
    status: "ready",
    barReadiness: { decision: "U0_BAR_NOT_READY_COVERAGE", gatePass: { identity: true, provider: true, coverage: false, quality: false }, coveredByBoard: { BSE: 54, STAR: 24 }, boardsAbsent: [], fetchableBacklog: 936, panelSha256: "abc123" },
    strictPitReadiness: { decision: "FULL_UNIVERSE_DATA_NOT_READY_PIT", trainingPermitted: false, blockedPitFields: ["st_intervals", "suspension_intervals", "delisting_intervals", "corporate_action_identity"] },
    pitSourceAudit: { st_intervals: "ALTERNATIVE_SOURCE_REQUIRED", corporate_action_identity: "ALTERNATIVE_SOURCE_REQUIRED" },
    tickflowBenchmark: { sdkVersion: "0.1.22", count10000Works: true, batchEntitled: false, measuredRatePerMin: 10, recommendedPath: "single get(count=10000)", old100BarCause: "MISSING count PARAMETER" },
    bseIdentity: { decision: "BSE_IDENTITY_CURRENT_RESOLVED", authoritativeCount: 329, masterCount: 327, truePlaceholders: [], missingFromMaster: ["920079", "920117"] },
    pitMetadataSourcing: { closedFields: ["price_limit_regimes", "ipo_special_limit_intervals", "delisting_intervals"], blockedFields: ["st_intervals", "suspension_intervals", "corporate_action_identity"], delistingDatesSourced: 367 },
    reconciliation: { supplementalAdditions: 2, supplementalSymbols: ["920079.BJ", "920117.BJ"], dualIdentityCollisions: 0, starCovered: 24, starTotal: 613 },
  },
  lineage: {
    status: "ready",
    headCommit: "731e61172121b5338a6f7e7d655d59432ccac6d0",
    originMainCommit: "731e61172121b5338a6f7e7d655d59432ccac6d0",
    headEqualsOriginMain: true,
    h030RemotelyRecoverable: true,
    overlappingFiles: ["tests/test_h030_operational_gates.py"],
    expectedConflictAreas: [],
    integrationBranch: "agent/h031-vnext-integration",
  },
  governedCommands: [
    { commandId: "validate-shadow-days", type: "governance", requiresNetwork: false, parameters: ["quiet"] },
    { commandId: "backfill-u0-market-panel", type: "data", requiresNetwork: true, parameters: ["allow_network", "max_minutes"] },
  ],
  blinding: "existence + gate level only",
};

function renderWith(payload: unknown): void {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(
    JSON.stringify({ status: "ready", data: payload, issues: [] }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  )));
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <MemoryRouter>
      <QueryClientProvider client={queryClient}>
        <GovernancePage />
      </QueryClientProvider>
    </MemoryRouter>,
  );
}

test("renders governed operational state without any candidate performance", async () => {
  renderWith(NOT_READY);
  expect(await screen.findByText("运营治理总览")).toBeInTheDocument();
  // shadow accumulating count is shown (metric strip + panel fact)
  expect(screen.getAllByText("2 / 7").length).toBeGreaterThan(0);
  // U0 state and absent boards are surfaced honestly
  expect(screen.getAllByText("FULL_UNIVERSE_DATA_NOT_READY_COVERAGE").length).toBeGreaterThan(0);
  expect(screen.getByText("STAR, BSE")).toBeInTheDocument();
  // H-032A: survivorship + STAR/BSE probe diagnosis are surfaced
  expect(screen.getByText(/226\/358 有行情/)).toBeInTheDocument();
  expect(screen.getByText(/STAR:FETCHABLE_NOT_PROBED/)).toBeInTheDocument();
  // H-032B: bar vs strict-PIT decisions shown separately, benchmark + BSE identity
  expect(screen.getByText("U0_BAR_NOT_READY_COVERAGE")).toBeInTheDocument();
  expect(screen.getAllByText("FULL_UNIVERSE_DATA_NOT_READY_PIT").length).toBeGreaterThan(0);
  expect(screen.getByText("BSE_IDENTITY_CURRENT_RESOLVED")).toBeInTheDocument();
  // H-032C: PIT delisting closure + reconciliation surfaced
  expect(screen.getByText("367")).toBeInTheDocument();
  expect(screen.getByText(/price_limit_regimes/)).toBeInTheDocument();

  // NO performance token may appear as a standalone word in the DOM
  // (word boundaries avoid false positives such as "nav" inside "unavailable").
  const text = document.body.textContent?.toLowerCase() ?? "";
  for (const banned of ["nav", "sharpe", "cagr", "drawdown", "calmar", "sortino", "pnl"]) {
    expect(new RegExp(`\\b${banned}\\b`).test(text)).toBe(false);
  }
});

test("full-universe training control stays disabled until FULL_UNIVERSE_DATA_READY", async () => {
  renderWith(NOT_READY);
  const button = await screen.findByRole("button", { name: "全宇宙训练" });
  expect(button).toBeDisabled();
  expect(screen.getByText(/该控制保持禁用/)).toBeInTheDocument();
});

test("training control enables only when readiness is verified", async () => {
  const ready = JSON.parse(JSON.stringify(NOT_READY));
  ready.u0.dataReadinessState = "FULL_UNIVERSE_DATA_READY";
  ready.u0.trainingPermitted = true;
  ready.u0.gatePass = { integration: true, provider: true, coverage: true, pit: true };
  ready.u0.boardsAbsent = [];
  renderWith(ready);
  const button = await screen.findByRole("button", { name: "全宇宙训练" });
  await waitFor(() => expect(button).not.toBeDisabled());
});

test("exposes no free-form shell or credential input field", async () => {
  renderWith(NOT_READY);
  await screen.findByText("运营治理总览");
  // governed contract: the surface submits allowlisted commands, never raw shell
  expect(screen.queryByRole("textbox")).toBeNull();
  expect(document.querySelector("input[type=password]")).toBeNull();
});

test("shows an explicit unavailable state when a manifest is missing", async () => {
  const partial = JSON.parse(JSON.stringify(NOT_READY));
  partial.shadow = { status: "unavailable", reason: "shadow_day_registry.json not found; run validate-shadow-days" };
  renderWith(partial);
  expect(await screen.findByText("影子注册表尚未生成")).toBeInTheDocument();
});
