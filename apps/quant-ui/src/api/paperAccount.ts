import { apiGet } from "./client";
import type { ApiResponse } from "./types";

/**
 * `GET /api/paper/account` — the HTTP paper venue's account, replayed from its
 * canonical ledger, plus the venue risk engine's own view of that account.
 *
 * Contract (services/quant_api/services/paper_orders.py): every field the venue
 * has not measured is `null` and carries a machine reason under `reasons`.
 * A null is never a zero — the UI must render the reason, not "0%" or "pass".
 */

export interface PaperAccountIdentity {
  portfolioId: string;
  initialCash: number;
  accountInstanceId: string | null;
  identitySha256: string | null;
  identityPath: string;
  reasons: Record<string, string>;
}

export interface PaperKillSwitchRecord {
  scope: string;
  key?: string | null;
  reason: string;
  triggeredAt?: string | null;
}

export interface PaperKillSwitchState {
  active: boolean;
  scope: string | null;
  reason: string | null;
  triggeredAt: string | null;
  reduceOnly: boolean;
  switches: PaperKillSwitchRecord[];
}

export interface PaperSessionTurnover {
  session: string;
  notional: number;
  fractionOfNav: number | null;
  limit: number;
}

export type PaperIndustryLimitMode = "measured_with_map" | "refuse_unmeasured" | "opt_out";

export interface PaperIndustryLimit {
  limit: number;
  mode: PaperIndustryLimitMode | string;
  mapLoaded: boolean;
  mapSymbolCount: number;
  source: string | null;
  error: string | null;
}

export interface PaperRiskCheck {
  name: string;
  passed: boolean;
  detail?: string;
  limit?: unknown;
  measured?: unknown;
}

export interface PaperPortfolioDecision {
  verdict: string;
  approved: boolean;
  failed_checks: string[];
  decided_at: string;
  checks: PaperRiskCheck[];
  override_available?: boolean;
}

/** `RiskLimits.to_dict()` — snake_case because it is the dataclass verbatim. */
export interface PaperRiskLimits {
  max_order_notional?: number;
  max_order_shares?: number;
  max_single_name_weight?: number;
  max_industry_weight?: number;
  max_gross_exposure?: number;
  max_daily_turnover?: number;
  max_daily_loss_fraction?: number;
  max_daily_loss?: number | null;
  max_drawdown?: number;
  max_participation?: number;
  max_quote_age_seconds?: number;
  max_price_deviation?: number;
  min_cash_buffer?: number;
  [key: string]: number | null | undefined;
}

export interface PaperRiskState {
  riskEngineAttached: boolean;
  limits: PaperRiskLimits | null;
  killSwitch: PaperKillSwitchState;
  peakEquity: number | null;
  drawdownFromPeak: number | null;
  nav: number | null;
  sessionTurnover: PaperSessionTurnover | null;
  sessionStartEquity: number | null;
  unpriceableSymbols: string[];
  industryLimitEnforced: boolean;
  /** Null when no risk engine is attached (no limit is applied at all). */
  industryLimit: PaperIndustryLimit | null;
  lastPortfolioCheck: PaperPortfolioDecision | null;
  valuationMarks: { source: string; count: number };
  reasons: Record<string, string>;
}

export interface PaperOperatingMode {
  mode: string;
  declared_at?: string | null;
  live_trading_available: boolean;
  banner?: string;
  executable?: boolean;
  simulatesOrders?: boolean;
  paperBannerLines?: string[];
}

export interface PaperAccount {
  cash: number;
  realisedPnl: number;
  totalFees: number;
  totalSlippage?: number;
  positions: Record<string, unknown>;
  contentHash?: string;
  /** Ledger valuation at the prices this route was given (none in production). */
  nav: number | null;
  marketValue?: number | null;
  unrealisedPnl?: number | null;
  unpriceableSymbols?: string[];
  reason?: string | null;
  initialCash: number;
  mode: PaperOperatingMode;
  writable: boolean;
  writerLockError?: string | null;
  accountIdentity: PaperAccountIdentity;
  riskState: PaperRiskState;
}

export const PAPER_ACCOUNT_PATH = "/paper/account";

export function fetchPaperAccount(signal?: AbortSignal): Promise<ApiResponse<PaperAccount>> {
  return apiGet<PaperAccount>(PAPER_ACCOUNT_PATH, undefined, signal);
}

/**
 * Plain-language text for the venue's machine reasons. The machine code stays
 * visible next to it (an operator must be able to grep the producer for it);
 * an unknown code is shown verbatim rather than guessed at.
 */
const REASON_TEXT: Array<[prefix: string, text: string]> = [
  ["no_identity_record", "没有不可变账户身份文件；portfolioId 与初始资金只是进程配置"],
  ["identity_record_unreadable", "账户身份文件无法读取或校验失败"],
  ["no_risk_engine_attached", "风险引擎未接入：不施加任何组合限额"],
  ["no_equity_observation", "风险引擎尚未观察到任何权益"],
  ["nav_unavailable", "净值不可得，无法计算"],
  ["peak_equity_unavailable", "没有历史峰值权益，无法计算回撤"],
  ["no_session_observed_by_this_process", "本进程尚未观察到交易会话"],
  ["no_session_start_valuation", "本会话没有开盘估值"],
  ["no_portfolio_check_run_in_this_process", "本进程尚未运行组合级检查"],
  ["no_active_kill_switch", "没有已触发的 kill switch"],
  ["unpriceable_positions", "持仓缺少行情标记，净值不可估"],
  ["no industry map", "没有行业映射：所有买入以 industry_unmeasured 拒绝"],
];

export function describeReason(reason: string | null | undefined): string | null {
  if (!reason) return null;
  for (const [prefix, text] of REASON_TEXT) {
    if (reason.startsWith(prefix)) return text;
  }
  return null;
}
