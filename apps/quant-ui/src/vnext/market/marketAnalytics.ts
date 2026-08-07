import type { KlineBar } from "../../api/types";

export interface MarketAnalytics {
  latest: KlineBar | null;
  previousClose: number | null;
  dailyReturn: number | null;
  return20: number | null;
  return60: number | null;
  return120: number | null;
  ma20: number | null;
  ma60: number | null;
  ma120: number | null;
  maxDrawdown60: number | null;
  averageAmount20: number | null;
  high250: number | null;
  low250: number | null;
  annualizedVolatility20: number | null;
}

function validClose(bar: KlineBar | undefined): number | null {
  const value = bar?.close;
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

export function sessionReturn(bars: KlineBar[], sessions: number): number | null {
  if (sessions <= 0 || bars.length <= sessions) return null;
  const end = validClose(bars.at(-1));
  const start = validClose(bars[bars.length - 1 - sessions]);
  if (end === null || start === null) return null;
  return end / start - 1;
}

export function latestMovingAverage(bars: KlineBar[], sessions: number): number | null {
  if (sessions <= 0 || bars.length < sessions) return null;
  const values = bars.slice(-sessions).map((bar) => validClose(bar));
  if (values.some((value) => value === null)) return null;
  return (values as number[]).reduce((sum, value) => sum + value, 0) / sessions;
}

export function maxDrawdown(bars: KlineBar[], sessions: number): number | null {
  if (sessions <= 1 || bars.length < sessions) return null;
  const closes = bars.slice(-sessions).map((bar) => validClose(bar));
  if (closes.some((value) => value === null)) return null;
  let peak = closes[0] as number;
  let worst = 0;
  for (const close of closes as number[]) {
    peak = Math.max(peak, close);
    worst = Math.min(worst, close / peak - 1);
  }
  return worst;
}

export function averageAmount(bars: KlineBar[], sessions: number): number | null {
  if (sessions <= 0 || bars.length < sessions) return null;
  const values = bars.slice(-sessions).map((bar) => bar.amount);
  if (values.some((value) => typeof value !== "number" || !Number.isFinite(value))) return null;
  return (values as number[]).reduce((sum, value) => sum + value, 0) / sessions;
}

function trailingRange(bars: KlineBar[], sessions: number): { high: number | null; low: number | null } {
  if (!bars.length) return { high: null, low: null };
  const tail = bars.slice(-Math.min(sessions, bars.length));
  const highs = tail.map((bar) => bar.high).filter((value) => Number.isFinite(value));
  const lows = tail.map((bar) => bar.low).filter((value) => Number.isFinite(value));
  return {
    high: highs.length ? Math.max(...highs) : null,
    low: lows.length ? Math.min(...lows) : null,
  };
}

function annualizedVolatility(bars: KlineBar[], sessions: number): number | null {
  if (sessions <= 1 || bars.length <= sessions) return null;
  const closes = bars.slice(-(sessions + 1)).map((bar) => validClose(bar));
  if (closes.some((value) => value === null)) return null;
  const returns: number[] = [];
  for (let index = 1; index < closes.length; index += 1) {
    const current = closes[index] as number;
    const previous = closes[index - 1] as number;
    returns.push(Math.log(current / previous));
  }
  if (returns.length < 2) return null;
  const mean = returns.reduce((sum, value) => sum + value, 0) / returns.length;
  const variance = returns.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (returns.length - 1);
  return Math.sqrt(variance) * Math.sqrt(252);
}

export function buildMarketAnalytics(bars: KlineBar[]): MarketAnalytics {
  const latest = bars.at(-1) ?? null;
  const previousClose = validClose(bars.at(-2));
  const latestClose = validClose(latest ?? undefined);
  const range = trailingRange(bars, 250);

  return {
    latest,
    previousClose,
    dailyReturn: latestClose !== null && previousClose !== null ? latestClose / previousClose - 1 : null,
    return20: sessionReturn(bars, 20),
    return60: sessionReturn(bars, 60),
    return120: sessionReturn(bars, 120),
    ma20: latestMovingAverage(bars, 20),
    ma60: latestMovingAverage(bars, 60),
    ma120: latestMovingAverage(bars, 120),
    maxDrawdown60: maxDrawdown(bars, 60),
    averageAmount20: averageAmount(bars, 20),
    high250: range.high,
    low250: range.low,
    annualizedVolatility20: annualizedVolatility(bars, 20),
  };
}

export function describeTrend(analytics: MarketAnalytics): string {
  const close = analytics.latest?.close;
  if (typeof close !== "number" || !Number.isFinite(close)) return "历史行情不足，无法判定趋势结构。";
  const averages = [analytics.ma20, analytics.ma60, analytics.ma120];
  if (averages.some((value) => value === null)) return "均线样本不足，暂不输出完整趋势判断。";
  const [ma20, ma60, ma120] = averages as number[];
  if (close > ma20 && ma20 > ma60 && ma60 > ma120) return "价格位于 MA20/60/120 上方且均线多头排列。";
  if (close < ma20 && ma20 < ma60 && ma60 < ma120) return "价格位于 MA20/60/120 下方且均线空头排列。";
  if (close > ma60) return "价格仍在中期均线之上，但均线结构尚未形成完整多头排列。";
  return "价格处于中期均线下方，趋势结构偏弱或仍在修复。";
}
