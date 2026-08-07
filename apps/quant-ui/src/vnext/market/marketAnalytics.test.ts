import { describe, expect, test } from "vitest";
import type { KlineBar } from "../../api/types";
import { buildMarketAnalytics, describeTrend, latestMovingAverage, maxDrawdown, sessionReturn } from "./marketAnalytics";

function bars(closes: number[]): KlineBar[] {
  return closes.map((close, index) => ({
    datetime: new Date(Date.UTC(2025, 0, index + 1)).toISOString(),
    symbol: "600519.SH",
    open: close,
    high: close + 1,
    low: close - 1,
    close,
    volume: 1_000_000,
    amount: 100_000_000 + index,
  }));
}

describe("market analytics", () => {
  test("uses full-session anchors instead of partial lookbacks", () => {
    const values = bars(Array.from({ length: 121 }, (_, index) => 100 + index));
    expect(sessionReturn(values, 20)).toBeCloseTo(220 / 200 - 1, 8);
    expect(sessionReturn(values.slice(-20), 20)).toBeNull();
  });

  test("computes moving averages and drawdown without future rows", () => {
    const values = bars([10, 12, 11, 9, 8]);
    expect(latestMovingAverage(values, 3)).toBeCloseTo((11 + 9 + 8) / 3, 8);
    expect(maxDrawdown(values, 5)).toBeCloseTo(8 / 12 - 1, 8);
  });

  test("builds the Financial-API-inspired research snapshot", () => {
    const values = bars(Array.from({ length: 140 }, (_, index) => 100 + index * 0.5));
    const analytics = buildMarketAnalytics(values);
    expect(analytics.return20).not.toBeNull();
    expect(analytics.return60).not.toBeNull();
    expect(analytics.return120).not.toBeNull();
    expect(analytics.ma120).not.toBeNull();
    expect(analytics.averageAmount20).not.toBeNull();
    expect(describeTrend(analytics)).toMatch(/多头/);
  });
});
