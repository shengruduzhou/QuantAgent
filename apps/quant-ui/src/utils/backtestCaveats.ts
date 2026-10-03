import type { BacktestSummary } from "../api/types";
import { formatNumber } from "./format";

/**
 * What a reader must know before citing a backtest row: trust, clock,
 * quarantined-window overlap, cost basis and benchmark identity. Every absent
 * field reads as "not recorded", never as a clean value.
 */

export const COST_BASIS_ALL_FILLS = "all_fills_explicit_fees_plus_impact_plus_slippage";
export const COST_BASIS_PRE_FIX = "matched_round_trip_fees_only_pre_fix";

export interface CostView {
  value: string;
  basis: string;
  /** True when the recorded cost is known to exclude slippage. */
  understated: boolean;
  recorded: boolean;
}

export function costView(run: BacktestSummary | undefined): CostView {
  if (!run || run.totalCost === null || run.totalCost === undefined) {
    return { value: "未记录", basis: "cost not recorded by this run", understated: false, recorded: false };
  }
  const value = `${formatNumber(run.totalCost, 0)} CNY`;
  if (run.totalCostBasis === COST_BASIS_ALL_FILLS) {
    const slip = run.slippageCost != null ? ` · slippage ${formatNumber(run.slippageCost, 0)}` : "";
    return { value, basis: `fees + impact + slippage${slip}`, understated: false, recorded: true };
  }
  if (run.totalCostBasis === COST_BASIS_PRE_FIX) {
    return { value, basis: "fees only · pre-fix, excludes slippage (understated)", understated: true, recorded: true };
  }
  return { value, basis: run.totalCostBasis ? run.totalCostBasis : "basis not recorded", understated: false, recorded: true };
}

export interface BenchmarkView {
  label: string;
  caveat: string | null;
}

export function benchmarkView(run: BacktestSummary | undefined): BenchmarkView {
  const bench = run?.benchmark;
  if (!bench || !bench.mode) return { label: "无基准", caveat: null };
  if (bench.mode.startsWith("universe_equal_weight")) {
    return {
      label: "宇宙等权 (universe_equal_weight)",
      caveat: "含不可交易标的（涨停/停牌/ST）且不计成本，超额被高估",
    };
  }
  if (bench.mode === "unlabelled") return { label: "未标注基准", caveat: bench.caveat ?? "基准身份未记录" };
  return { label: bench.mode, caveat: bench.caveat ?? null };
}

export function timingView(run: BacktestSummary | undefined): { label: string; canonical: boolean | null } {
  if (!run || run.timingCanonical === undefined) return { label: "clock unknown", canonical: null };
  return run.timingCanonical ? { label: "canonical clock", canonical: true } : { label: "pre-fix clock", canonical: false };
}

/** Every reason this run's headline cannot be cited, in reading order. */
export function citationCaveats(run: BacktestSummary | undefined): string[] {
  if (!run) return [];
  const cost = costView(run);
  const bench = benchmarkView(run);
  return [
    run.trustClass && run.trustClass !== "production_ready" ? `trust: ${run.trustClass}` : null,
    run.validationStatus && run.validationStatus !== "verified" ? `validation: ${run.validationStatus}` : null,
    run.timingCanonical === false ? "pre-fix execution clock" : null,
    run.quarantineOverlap?.length ? `overlaps quarantined ${run.quarantineOverlap.join(", ")}` : null,
    cost.understated ? "cost excludes slippage" : !cost.recorded ? "cost not recorded" : null,
    bench.caveat ? `benchmark: ${bench.caveat}` : null,
  ].filter((item): item is string => Boolean(item));
}

/** A run with no NAV artifact is not a usable default subject. */
export function hasNav(run: BacktestSummary): boolean {
  return run.capabilities?.equity === true;
}
