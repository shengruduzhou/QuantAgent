import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, test, vi } from "vitest";
import { RiskCenterPage } from "./RiskCenterPage";
import { buildVenueRuleRows, formatThreshold } from "../vnext/paper/venueRules";
import type { PaperAccount } from "../api/paperAccount";

vi.mock("../components/EChart", () => ({ EChart: () => <div data-testid="chart" /> }));
vi.mock("../components/RiskRadar", () => ({ RiskRadar: () => <div data-testid="radar" /> }));

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const RULES = [
  { id: "max_drawdown", name: "Drawdown kill switch", description: "dd", threshold: 0.2, unit: "fraction_from_all_time_peak", enforcedAt: "portfolio_after_fill_and_session_open", enabled: true, codeLocation: "src/quantagent/paper/risk.py" },
  { id: "max_order_notional", name: "Order notional cap", description: "n", threshold: 200_000, unit: "cny", enforcedAt: "pre_trade_order", enabled: true, codeLocation: "src/quantagent/paper/risk.py" },
  { id: "max_daily_turnover", name: "Daily turnover cap", description: "t", threshold: 2, unit: "fraction_of_equity_per_session", enforcedAt: "pre_trade_order", enabled: true, codeLocation: "src/quantagent/paper/risk.py" },
];

function overviewFor(id: string, name: string, events: Record<string, number>) {
  return { backtestId: id, backtestName: name, eventCountsExact: true, maxDrawdown: 0.061, maxSingleStockLoss: -5443.2, maxDailyLoss: -0.046, consecutiveLossDays: 8, liquidityRisk: 0, limitDownRisk: 0, suspensionRisk: 0, eventCounts: events, rules: RULES };
}

const PAPER: PaperAccount = {
  cash: 1_000_000, realisedPnl: 0, totalFees: 0, positions: {}, nav: 1_000_000, initialCash: 1_000_000,
  mode: { mode: "PAPER", live_trading_available: false }, writable: true,
  accountIdentity: { portfolioId: "paper_api", initialCash: 1_000_000, accountInstanceId: null, identitySha256: null, identityPath: "/r/runtime/paper_orders/x.json", reasons: {} },
  riskState: {
    riskEngineAttached: true,
    limits: { max_drawdown: 0.15, max_order_notional: 200_000, max_daily_turnover: 2 },
    killSwitch: { active: false, scope: null, reason: null, triggeredAt: null, reduceOnly: false, switches: [] },
    peakEquity: 1_000_000, drawdownFromPeak: 0.03, nav: 1_000_000,
    sessionTurnover: null, sessionStartEquity: null, unpriceableSymbols: [], industryLimitEnforced: true,
    industryLimit: { limit: 0.3, mode: "refuse_unmeasured", mapLoaded: false, mapSymbolCount: 0, source: null, error: null },
    lastPortfolioCheck: null, valuationMarks: { source: "venue_last_observed", count: 0 },
    reasons: { sessionTurnover: "no_session_observed_by_this_process" },
  },
};

function installApi(): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = new URL(String(input));
    const id = url.searchParams.get("backtestId");
    const body = (data: unknown) => new Response(JSON.stringify({ status: "ready", data, issues: [] }), { status: 200 });
    if (url.pathname.endsWith("/system/overview")) {
      return body({ risk: overviewFor("bt_dashboard", "retrain_plus7", { order_skipped: 1000 }), runtime: { artifactCount: 0, totalSizeBytes: 0, byKind: {}, indexedAt: "2026-10-01" }, modelStatus: "ready", riskStatus: "normal" });
    }
    if (url.pathname.endsWith("/backtests")) {
      return body([
        { id: "bt_board", name: "board_chase", status: "ready", path: "x", tags: [], endDate: "2026-06-11", capabilities: { riskEvents: false } },
        { id: "bt_dashboard", name: "retrain_plus7", status: "ready", path: "y", tags: [], startDate: "2024-08-09", endDate: "2026-05-07", trustClass: "unclassified", validationStatus: "unverified", timingCanonical: false, quarantineOverlap: ["2025-09-01..2026-05-18"], capabilities: { riskEvents: true } },
      ]);
    }
    if (url.pathname.endsWith("/risk/overview")) {
      // The server's own default (no id) is a *different* run: the page must never rely on it.
      return id === "bt_dashboard" ? body(overviewFor("bt_dashboard", "retrain_plus7", { order_skipped: 1000 })) : body(overviewFor("bt_board", "board_chase", {}));
    }
    if (url.pathname.endsWith("/risk/events")) return body({ items: [{ id: `e-${id}`, datetime: "2024-08-09", type: `order_skipped_${id}`, severity: "warning", sourcePath: "x" }], total: 1, page: 1, pageSize: 200, hasNext: false });
    if (url.pathname.endsWith("/risk/stocks")) return body([{ symbol: `SYM_${id}`, netPnl: -10, tradeCount: 1, riskScore: 10 }]);
    if (url.pathname.endsWith("/risk/rules")) return body(RULES);
    if (url.pathname.endsWith("/paper/account")) return body(PAPER);
    return new Response("not found", { status: 404 });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function renderPage(path = "/risk"): void {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[path]}><div className="vnext-shell"><RiskCenterPage /></div></MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("RiskCenterPage audits one named subject", () => {
  test("defaults to the command bar's backtest and fetches events/stocks for that same id", async () => {
    const fetchMock = installApi();
    renderPage();
    const subject = await screen.findByRole("region", { name: "回测审计对象" });
    expect(within(subject).getByRole("heading", { name: "retrain_plus7" })).toBeInTheDocument();
    expect(within(subject).getByText(/bt_dashboard · 2024-08-09 → 2026-05-07/)).toBeInTheDocument();
    expect(within(subject).getByText("trust: unclassified")).toBeInTheDocument();
    expect(within(subject).getByText("pre-fix clock")).toBeInTheDocument();
    expect(await screen.findByText("order_skipped_bt_dashboard")).toBeInTheDocument();
    expect(await screen.findByText("SYM_bt_dashboard")).toBeInTheDocument();
    const urls = fetchMock.mock.calls.map(([input]) => new URL(String(input)));
    for (const route of ["/risk/overview", "/risk/events", "/risk/stocks"]) {
      const calls = urls.filter((url) => url.pathname.endsWith(route));
      expect(calls.length, route).toBeGreaterThan(0);
      expect(calls.every((url) => url.searchParams.get("backtestId") === "bt_dashboard"), route).toBe(true);
    }
    expect(screen.queryByText("order_skipped_bt_board")).not.toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "board_chase" })).not.toBeInTheDocument();
  });

  test("an explicit ?backtestId= is honoured and named", async () => {
    installApi();
    renderPage("/risk?backtestId=bt_board");
    const subject = await screen.findByRole("region", { name: "回测审计对象" });
    expect(within(subject).getByRole("heading", { name: "board_chase" })).toBeInTheDocument();
    // No risk_events artifact: the count is "no artifact", not a green zero.
    expect(await screen.findByText("无产物")).toBeInTheDocument();
  });

  test("rule rows carry unit and enforcement point; current values come from the paper account only", async () => {
    installApi();
    renderPage();
    const table = await screen.findByRole("table", { name: "Paper venue 风险规则" });
    const drawdown = within(table).getByText("Drawdown kill switch").closest("tr") as HTMLElement;
    // The venue's configured limit (15%) wins over the code default (20%).
    expect(within(drawdown).getByText("15%")).toBeInTheDocument();
    expect(within(drawdown).getByText("venue config")).toBeInTheDocument();
    expect(within(drawdown).getByText("% below all-time peak")).toBeInTheDocument();
    expect(within(drawdown).getByText("portfolio · after fill & at session open")).toBeInTheDocument();
    expect(within(drawdown).getByText(/^3\.00%/)).toBeInTheDocument();
    // The backtest's 6.10% drawdown is never placed beside the venue limit.
    expect(within(table).queryByText(/6\.10%/)).not.toBeInTheDocument();
    const notional = within(table).getByText("Order notional cap").closest("tr") as HTMLElement;
    expect(within(notional).getByText("200,000 CNY")).toBeInTheDocument();
    expect(within(notional).getByText("checked on each order · no standing value")).toBeInTheDocument();
    const turnover = within(table).getByText("Daily turnover cap").closest("tr") as HTMLElement;
    expect(within(turnover).getByText("not measured for the paper account")).toBeInTheDocument();
    await waitFor(() => expect(screen.getAllByText("KILL ARMED").length).toBeGreaterThan(0));
  });
});

describe("buildVenueRuleRows", () => {
  test("without a paper account every current value is unmeasured, never 0", () => {
    const rows = buildVenueRuleRows(RULES, undefined);
    expect(rows.map((row) => row.current.kind)).toEqual(["unmeasured", "unmeasured", "unmeasured"]);
    expect(rows.every((row) => row.enforcement === "unknown")).toBe(true);
    expect(rows[0].threshold).toBe("20%");
    expect(rows[0].thresholdSource).toBe("code default");
  });

  test("no risk engine: limits are reported as not enforced", () => {
    const account = { ...PAPER, riskState: { ...PAPER.riskState, riskEngineAttached: false, limits: null, reasons: { riskEngine: "no_risk_engine_attached: no portfolio limit is applied" } } };
    const rows = buildVenueRuleRows(RULES, account);
    expect(rows.every((row) => row.enforcement === "not_attached")).toBe(true);
    expect(rows[0].current).toMatchObject({ kind: "unmeasured", reason: "no_risk_engine_attached: no portfolio limit is applied" });
  });

  test("formats thresholds by unit", () => {
    expect(formatThreshold(0.3, "fraction_of_equity")).toBe("30%");
    expect(formatThreshold(0.125, "fraction_of_equity")).toBe("12.5%");
    expect(formatThreshold(200_000, "cny")).toBe("200,000 CNY");
    expect(formatThreshold(null, null)).toBe("rule");
  });
});
