import { describeReason, type PaperAccount, type PaperRiskLimits } from "../../api/paperAccount";
import { formatNumber, formatPercent } from "../../utils/format";

/**
 * Risk rule rows for the paper venue (`/api/risk/rules`), each with its unit,
 * its enforcement point, and — only where the quantity is the same thing in
 * the same unit — the paper account's current value from `riskState`.
 *
 * A backtest's drawdown is never put beside a venue limit: it is another
 * subject (a research run), measured from another peak. Rules that are checked
 * per order have no standing value; that is said, not rendered as 0.
 */

export interface RawRiskRule {
  id?: unknown;
  name?: unknown;
  description?: unknown;
  threshold?: unknown;
  unit?: unknown;
  enforcedAt?: unknown;
  enabled?: unknown;
  codeLocation?: unknown;
}

export type VenueRuleCurrent =
  | { kind: "measured"; value: number; text: string; ratio: number | null; basis: string }
  | { kind: "per_order"; text: string }
  | { kind: "unmeasured"; text: string; reason: string | null };

export interface VenueRuleRow {
  id: string;
  name: string;
  description: string;
  unit: string | null;
  unitLabel: string;
  enforcedAt: string | null;
  enforcedAtLabel: string;
  threshold: string;
  thresholdValue: number | null;
  /** "venue" = the running engine's configured limit; "code default" = RiskLimits() default. */
  thresholdSource: "venue" | "code default" | "none";
  codeLocation: string | null;
  enforcement: "enforced" | "not_attached" | "unknown";
  current: VenueRuleCurrent;
  state: "normal" | "warning" | "breach" | "unmeasured" | "per_order";
}

const UNIT_LABEL: Record<string, string> = {
  fraction_of_equity: "% of equity",
  fraction_from_all_time_peak: "% below all-time peak",
  fraction_of_session_opening_equity: "% of session-open equity",
  fraction_of_equity_per_session: "% of equity per session",
  fraction_of_reference_price: "% from reference price",
  cny: "CNY per order",
};

const ENFORCED_AT_LABEL: Record<string, string> = {
  pre_trade_order: "pre-trade · every order",
  pre_trade_order_with_sector_map: "pre-trade · every BUY (needs sector map)",
  portfolio_after_fill_and_session_open: "portfolio · after fill & at session open",
  venue_instrument_rules: "venue · instrument rules",
};

/** `/api/risk/rules` id → key in `riskState.limits` (RiskLimits.to_dict()). */
const LIMIT_KEY: Record<string, keyof PaperRiskLimits> = {
  max_single_name_weight: "max_single_name_weight",
  max_industry_weight: "max_industry_weight",
  max_drawdown: "max_drawdown",
  max_daily_loss: "max_daily_loss_fraction",
  max_gross_exposure: "max_gross_exposure",
  max_daily_turnover: "max_daily_turnover",
  max_order_notional: "max_order_notional",
  max_price_deviation: "max_price_deviation",
};

export function formatThreshold(value: number | null, unit: string | null): string {
  if (value === null) return "rule";
  if (unit === "cny") return `${formatNumber(value, 0)} CNY`;
  if (unit?.startsWith("fraction")) {
    const pct = Math.round(value * 10_000) / 100;
    return formatPercent(value, Number.isInteger(pct) ? 0 : 1);
  }
  return String(value);
}

function asString(value: unknown): string | null {
  return typeof value === "string" && value ? value : null;
}

function largestCheck(account: PaperAccount, prefix: string): { value: number; at: string } | null {
  const decision = account.riskState.lastPortfolioCheck;
  if (!decision) return null;
  const measured = decision.checks
    .filter((check) => check.name === prefix || check.name.startsWith(`${prefix}:`))
    .map((check) => (typeof check.measured === "number" ? check.measured : null))
    .filter((value): value is number => value !== null);
  if (!measured.length) return null;
  return { value: Math.max(...measured), at: decision.decided_at.replace("T", " ").slice(0, 19) };
}

function unmeasured(reason: string | null | undefined, fallback = "not measured for the paper account"): VenueRuleCurrent {
  const code = reason ?? null;
  return { kind: "unmeasured", text: describeReason(code) ?? fallback, reason: code };
}

function currentFor(id: string, account: PaperAccount | undefined): VenueRuleCurrent {
  if (!account) return unmeasured(null, "paper account unavailable — not measured");
  const risk = account.riskState;
  if (!risk.riskEngineAttached && id !== "t_plus_one_and_price_limits") return unmeasured(risk.reasons.riskEngine ?? "no_risk_engine_attached");
  switch (id) {
    case "max_drawdown":
      return risk.drawdownFromPeak !== null
        ? { kind: "measured", value: risk.drawdownFromPeak, text: formatPercent(risk.drawdownFromPeak), ratio: null, basis: "riskState.drawdownFromPeak" }
        : unmeasured(risk.reasons.drawdownFromPeak);
    case "max_daily_loss": {
      if (risk.nav === null) return unmeasured(risk.reasons.nav ?? "nav_unavailable");
      if (risk.sessionStartEquity === null || !(risk.sessionStartEquity > 0)) return unmeasured(risk.reasons.sessionStartEquity ?? "no_session_start_valuation");
      const loss = Math.max(0, 1 - risk.nav / risk.sessionStartEquity);
      return { kind: "measured", value: loss, text: formatPercent(loss), ratio: null, basis: "1 − nav ÷ sessionStartEquity" };
    }
    case "max_daily_turnover": {
      const turnover = risk.sessionTurnover;
      if (!turnover) return unmeasured(risk.reasons.sessionTurnover);
      if (turnover.fractionOfNav === null) return unmeasured(risk.reasons.nav ?? "nav_unavailable");
      return { kind: "measured", value: turnover.fractionOfNav, text: formatPercent(turnover.fractionOfNav), ratio: null, basis: `sessionTurnover · ${turnover.session}` };
    }
    case "max_gross_exposure": {
      const check = largestCheck(account, "gross_exposure");
      return check
        ? { kind: "measured", value: check.value, text: formatPercent(check.value), ratio: null, basis: `last portfolio check ${check.at}` }
        : unmeasured(risk.reasons.lastPortfolioCheck);
    }
    case "max_single_name_weight": {
      const check = largestCheck(account, "concentration");
      return check
        ? { kind: "measured", value: check.value, text: formatPercent(check.value), ratio: null, basis: `largest name · last portfolio check ${check.at}` }
        : risk.lastPortfolioCheck ? { kind: "per_order", text: "no open position at last check · also checked per order" } : unmeasured(risk.reasons.lastPortfolioCheck);
    }
    case "max_industry_weight": {
      if (risk.industryLimit?.mode === "refuse_unmeasured") return unmeasured(risk.reasons.industryLimit ?? "no industry map", "no industry map — every BUY refused");
      if (risk.industryLimit?.mode === "opt_out") return { kind: "per_order", text: "opt-out: limit ≥ 100%" };
      const check = largestCheck(account, "industry");
      return check
        ? { kind: "measured", value: check.value, text: formatPercent(check.value), ratio: null, basis: `largest industry · last portfolio check ${check.at}` }
        : unmeasured(risk.reasons.lastPortfolioCheck);
    }
    case "max_order_notional":
    case "max_price_deviation":
      return { kind: "per_order", text: "checked on each order · no standing value" };
    case "t_plus_one_and_price_limits":
      return { kind: "per_order", text: "instrument rule on each order · no standing value" };
    default:
      return unmeasured(null);
  }
}

export function buildVenueRuleRows(rules: RawRiskRule[] | undefined, account: PaperAccount | undefined): VenueRuleRow[] {
  return (rules ?? []).map((rule) => {
    const id = asString(rule.id) ?? "rule";
    const unit = asString(rule.unit);
    const enforcedAt = asString(rule.enforcedAt);
    const limitKey = LIMIT_KEY[id];
    const venueLimit = limitKey && account?.riskState.limits ? account.riskState.limits[limitKey] : undefined;
    const codeDefault = typeof rule.threshold === "number" ? rule.threshold : null;
    const thresholdValue = typeof venueLimit === "number" ? venueLimit : codeDefault;
    const thresholdSource: VenueRuleRow["thresholdSource"] = typeof venueLimit === "number" ? "venue" : codeDefault !== null ? "code default" : "none";
    const current = currentFor(id, account);
    let state: VenueRuleRow["state"] = current.kind === "unmeasured" ? "unmeasured" : current.kind === "per_order" ? "per_order" : "normal";
    if (current.kind === "measured" && thresholdValue !== null && thresholdValue > 0) {
      current.ratio = current.value / thresholdValue;
      state = current.ratio > 1 ? "breach" : current.ratio >= 0.8 ? "warning" : "normal";
    }
    const enforcement: VenueRuleRow["enforcement"] = !account
      ? "unknown"
      : id === "t_plus_one_and_price_limits" || account.riskState.riskEngineAttached ? "enforced" : "not_attached";
    return {
      id,
      name: asString(rule.name) ?? id,
      description: asString(rule.description) ?? "",
      unit,
      unitLabel: unit ? UNIT_LABEL[unit] ?? unit : "rule",
      enforcedAt,
      enforcedAtLabel: enforcedAt ? ENFORCED_AT_LABEL[enforcedAt] ?? enforcedAt : "unspecified",
      threshold: formatThreshold(thresholdValue, unit),
      thresholdValue,
      thresholdSource,
      codeLocation: asString(rule.codeLocation),
      enforcement,
      current,
      state,
    };
  });
}
