import { describe, expect, test } from "vitest";
import type { KlineBar, Trade } from "../api/types";
import {
  classifyTradeMarker,
  layoutTradeMarkers,
  movingAverage,
  resolveKlineWindow,
} from "./kline";

function trade(id: string, datetime: string, action: string, riskReason?: string): Trade {
  return {
    id,
    datetime,
    symbol: "000001.SZ",
    action,
    price: 10,
    quantity: 100,
    riskReason,
  };
}

function bar(close: number, index: number): KlineBar {
  return {
    datetime: `2026-01-${String(index + 1).padStart(2, "0")}T00:00:00`,
    symbol: "000001.SZ",
    open: close,
    high: close,
    low: close,
    close,
  };
}

describe("kline helpers", () => {
  test("classifies normal, T+1 and risk markers", () => {
    expect(classifyTradeMarker(trade("a", "2026-01-01", "BUY"))).toEqual({ side: "buy", category: "trade" });
    expect(classifyTradeMarker(trade("b", "2026-01-01", "T_SELL"))).toEqual({ side: "sell", category: "t_trade" });
    expect(classifyTradeMarker(trade("c", "2026-01-01", "SELL", "drawdown gate"))).toEqual({ side: "risk", category: "risk" });
  });

  test("assigns deterministic lanes to same-day signals", () => {
    const markers = layoutTradeMarkers(
      [
        trade("b", "2026-01-02T10:00:00", "BUY"),
        trade("a", "2026-01-02T09:30:00", "BUY"),
        trade("c", "2026-01-02T11:00:00", "SELL"),
      ],
      new Set(["2026-01-02"]),
    );

    expect(markers.map((marker) => [marker.trade.id, marker.side, marker.lane])).toEqual([
      ["a", "buy", 0],
      ["b", "buy", 1],
      ["c", "sell", 0],
    ]);
  });

  test("calculates moving averages without look-ahead", () => {
    expect(movingAverage([1, 2, 3, 4].map(bar), 3)).toEqual([null, null, 2, 3]);
  });

  test("centres finite windows around the selected trade", () => {
    expect(resolveKlineWindow(300, "60D", 100)).toEqual({ startIndex: 53, endIndex: 112 });
    expect(resolveKlineWindow(20, "ALL", 5)).toEqual({ startIndex: 0, endIndex: 19 });
  });
});
