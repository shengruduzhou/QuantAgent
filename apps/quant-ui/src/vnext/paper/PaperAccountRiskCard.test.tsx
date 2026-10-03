import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, test, vi } from "vitest";
import type { PaperAccount, PaperRiskState } from "../../api/paperAccount";
import { PaperAccountRiskCard, PaperAccountRiskView } from "./PaperAccountRiskCard";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const LIMITS = {
  max_order_notional: 200_000,
  max_order_shares: 1_000_000,
  max_single_name_weight: 0.1,
  max_industry_weight: 0.3,
  max_gross_exposure: 1,
  max_daily_turnover: 2,
  max_daily_loss_fraction: 0.05,
  max_daily_loss: null,
  max_drawdown: 0.2,
  max_participation: 0.1,
  max_quote_age_seconds: 300,
  max_price_deviation: 0.1,
  min_cash_buffer: 0,
};

function riskState(overrides: Partial<PaperRiskState> = {}): PaperRiskState {
  return {
    riskEngineAttached: true,
    limits: LIMITS,
    killSwitch: { active: false, scope: null, reason: null, triggeredAt: null, reduceOnly: false, switches: [] },
    peakEquity: 1_050_000,
    drawdownFromPeak: 0.0476,
    nav: 1_000_000,
    sessionTurnover: { session: "2026-09-30", notional: 150_000, fractionOfNav: 0.15, limit: 2 },
    sessionStartEquity: 1_020_000,
    unpriceableSymbols: [],
    industryLimitEnforced: true,
    industryLimit: { limit: 0.3, mode: "measured_with_map", mapLoaded: true, mapSymbolCount: 5_123, source: "runtime/sector_map.parquet", error: null },
    lastPortfolioCheck: { verdict: "APPROVED", approved: true, failed_checks: [], decided_at: "2026-09-30T07:00:00+00:00", checks: [] },
    valuationMarks: { source: "venue_last_observed", count: 3 },
    reasons: { killSwitch: "no_active_kill_switch" },
    ...overrides,
  };
}

function account(risk: PaperRiskState, overrides: Partial<PaperAccount> = {}): PaperAccount {
  return {
    cash: 400_000,
    realisedPnl: 0,
    totalFees: 12.5,
    positions: { "600519.SH": {}, "000001.SZ": {} },
    nav: null,
    initialCash: 1_000_000,
    mode: { mode: "PAPER", live_trading_available: false, banner: "LIVE TRADING: DISABLED BY POLICY" },
    writable: true,
    writerLockError: null,
    accountIdentity: {
      portfolioId: "paper_api",
      initialCash: 1_000_000,
      accountInstanceId: "acct-7f3a9c21-55d0-4c1e-9b7e-2a8f0d6e4b11",
      identitySha256: "9b2c4e6f8a0d1c3e5f7a9b1d3f5e7c9a",
      identityPath: "/home/u/QuantAgent/runtime/paper_orders/canonical.account_identity.json",
      reasons: {},
    },
    riskState: risk,
    ...overrides,
  };
}

function renderView(value: PaperAccount): void {
  render(<MemoryRouter><div className="vnext-shell"><PaperAccountRiskView account={value} /></div></MemoryRouter>);
}

describe("PaperAccountRiskView", () => {
  test("shows identity, NAV and every limit read from the API response", () => {
    renderView(account(riskState()));
    const card = screen.getByRole("region", { name: "Paper Account & Risk" });
    expect(within(card).getByText("paper_api")).toBeInTheDocument();
    // Identity hash is shortened but the full value is kept for audit.
    expect(within(card).getByTitle("9b2c4e6f8a0d1c3e5f7a9b1d3f5e7c9a")).toHaveTextContent("9b2c4e6f8a0d…");
    expect(within(card).getByText("runtime/paper_orders/canonical.account_identity.json")).toBeInTheDocument();
    expect(within(card).getByText("4.76%")).toBeInTheDocument();
    expect(within(card).getByText("/ limit 20%")).toBeInTheDocument();
    expect(within(card).getByText("/ limit 200%")).toBeInTheDocument();
    expect(within(card).getAllByText("KILL ARMED").length).toBeGreaterThan(0);
    expect(within(card).getByText("LIVE DISABLED")).toBeInTheDocument();
    expect(within(card).getByText("MEASURED")).toBeInTheDocument();
    expect(within(card).queryByText("未测量")).not.toBeInTheDocument();
  });

  test("no risk engine: every engine measurement is unmeasured with the producer's reason, never 0 or armed", () => {
    const reason = "no_risk_engine_attached: no portfolio limit is applied";
    renderView(account(riskState({
      riskEngineAttached: false,
      limits: null,
      peakEquity: null,
      drawdownFromPeak: null,
      sessionTurnover: null,
      sessionStartEquity: null,
      industryLimitEnforced: false,
      industryLimit: null,
      lastPortfolioCheck: null,
      reasons: { killSwitch: "no_active_kill_switch", riskEngine: reason },
    })));
    const card = screen.getByRole("region", { name: "Paper Account & Risk" });
    expect(card).toHaveAttribute("data-rail", "danger");
    expect(within(card).getAllByText("KILL UNARMED").length).toBeGreaterThan(0);
    expect(within(card).queryByText("KILL ARMED")).not.toBeInTheDocument();
    // Drawdown, session loss, turnover: unmeasured, each carrying the reason code.
    expect(within(card).getAllByText("未测量").length).toBeGreaterThanOrEqual(3);
    expect(within(card).getAllByText("no_risk_engine_attached").length).toBeGreaterThanOrEqual(3);
    expect(within(card).getByText("未施加")).toBeInTheDocument();
    expect(within(card).queryByText("0.00%")).not.toBeInTheDocument();
    expect(within(card).queryByText(/limit 20%/)).not.toBeInTheDocument();
  });

  test("active PORTFOLIO kill switch shows scope, reason, since and reduce-only", () => {
    renderView(account(riskState({
      drawdownFromPeak: 0.231,
      killSwitch: {
        active: true,
        scope: "PORTFOLIO",
        reason: "drawdown 23.10% exceeds 20.00%",
        triggeredAt: "2026-09-30T06:58:12+00:00",
        reduceOnly: true,
        switches: [{ scope: "PORTFOLIO", key: null, reason: "drawdown 23.10% exceeds 20.00%", triggeredAt: "2026-09-30T06:58:12+00:00" }],
      },
      reasons: {},
    })));
    const card = screen.getByRole("region", { name: "Paper Account & Risk" });
    expect(card).toHaveAttribute("data-rail", "danger");
    expect(within(card).getAllByText("KILL ACTIVE · PORTFOLIO").length).toBeGreaterThan(0);
    expect(within(card).getByText("drawdown 23.10% exceeds 20.00%")).toBeInTheDocument();
    expect(within(card).getByText("2026-09-30 06:58:12")).toBeInTheDocument();
    const reduceOnly = within(card).getByText("Reduce-only").closest("div");
    expect(reduceOnly).toHaveTextContent("YES");
    expect(within(card).getAllByText(/reduce-only：买入被拒，卖出放行/).length).toBeGreaterThan(0);
    // The breached drawdown meter is full and danger-toned.
    const meter = within(card).getAllByRole("meter")[0];
    expect(meter).toHaveAttribute("data-tone", "danger");
  });

  test("unmeasured drawdown and missing identity: reasons shown, not 0% / not verified", () => {
    renderView(account(riskState({
      nav: null,
      drawdownFromPeak: null,
      unpriceableSymbols: ["600519.SH"],
      sessionTurnover: { session: "2026-09-30", notional: 150_000, fractionOfNav: null, limit: 2 },
      lastPortfolioCheck: null,
      reasons: {
        nav: "unpriceable_positions: 600519.SH",
        drawdownFromPeak: "nav_unavailable",
        lastPortfolioCheck: "no_portfolio_check_run_in_this_process",
        killSwitch: "no_active_kill_switch",
      },
    }), {
      accountIdentity: {
        portfolioId: "paper_api",
        initialCash: 1_000_000,
        accountInstanceId: null,
        identitySha256: null,
        identityPath: "/x/runtime/paper_orders/canonical.account_identity.json",
        reasons: {
          accountInstanceId: "no_identity_record: this HTTP paper account has no immutable identity file",
          identitySha256: "no_identity_record: this HTTP paper account has no immutable identity file",
        },
      },
    }));
    const card = screen.getByRole("region", { name: "Paper Account & Risk" });
    expect(card).toHaveAttribute("data-rail", "danger");
    const drawdown = within(card).getByText("Drawdown from all-time peak").closest("article") as HTMLElement;
    expect(within(drawdown).getByText("未测量")).toBeInTheDocument();
    expect(within(drawdown).getByText("nav_unavailable")).toBeInTheDocument();
    expect(within(drawdown).queryByText(/%/)).not.toBeInTheDocument();
    expect(within(card).getAllByText("未记录").length).toBe(2);
    expect(within(card).getAllByText("no_identity_record").length).toBe(2);
    expect(within(card).getByText("process configuration")).toBeInTheDocument();
    expect(within(card).getByText(/600519\.SH — NAV 与组合限额不可测/)).toBeInTheDocument();
    expect(within(card).getByText("未运行")).toBeInTheDocument();
  });
});

describe("PaperAccountRiskCard (typed client + react-query hook)", () => {
  function renderCard(): void {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter><div className="vnext-shell"><PaperAccountRiskCard /></div></MemoryRouter>
      </QueryClientProvider>,
    );
  }

  test("reads /api/paper/account and renders the venue's state", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL) => new Response(JSON.stringify({ status: "ready", data: account(riskState()), issues: [] }), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);
    renderCard();
    expect(await screen.findByText("paper_api")).toBeInTheDocument();
    expect(String(fetchMock.mock.calls[0][0])).toContain("/api/paper/account");
  });

  test("an API failure leaves kill-switch state unknown rather than clear", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("boom", { status: 503 })));
    renderCard();
    expect(await screen.findByText("KILL UNKNOWN")).toBeInTheDocument();
    expect(screen.queryByText("KILL ARMED")).not.toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent("不按“未触发”处理");
  });
});
