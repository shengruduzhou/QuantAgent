import type { KlineBar, Trade } from "../api/types";

export type KlineViewRange = "60D" | "120D" | "1Y" | "ALL";
export type TradeMarkerSide = "buy" | "sell" | "risk";
export type TradeMarkerCategory = "trade" | "t_trade" | "risk";

export interface TradeMarkerLayout {
  trade: Trade;
  date: string;
  side: TradeMarkerSide;
  category: TradeMarkerCategory;
  lane: number;
}

export interface KlineWindow {
  startIndex: number;
  endIndex: number;
}

const RANGE_SIZE: Record<Exclude<KlineViewRange, "ALL">, number> = {
  "60D": 60,
  "120D": 120,
  "1Y": 250,
};

export function classifyTradeMarker(trade: Trade): Pick<TradeMarkerLayout, "side" | "category"> {
  const action = trade.action.toUpperCase();
  const isRisk = Boolean(trade.riskReason)
    || action.includes("RISK")
    || action.includes("STOP")
    || action.includes("FAIL");

  if (isRisk) return { side: "risk", category: "risk" };

  const category: TradeMarkerCategory = action.startsWith("T_") ? "t_trade" : "trade";
  if (action.includes("BUY")) return { side: "buy", category };
  if (action.includes("SELL")) return { side: "sell", category };
  return { side: "risk", category: "risk" };
}

export function layoutTradeMarkers(trades: Trade[], validDates: ReadonlySet<string>): TradeMarkerLayout[] {
  const laneCounts = new Map<string, number>();

  return [...trades]
    .sort((left, right) => left.datetime.localeCompare(right.datetime) || left.id.localeCompare(right.id))
    .flatMap((trade) => {
      const date = trade.datetime.slice(0, 10);
      if (!validDates.has(date)) return [];

      const classification = classifyTradeMarker(trade);
      const laneKey = `${date}:${classification.side}`;
      const lane = laneCounts.get(laneKey) ?? 0;
      laneCounts.set(laneKey, lane + 1);

      return [{ trade, date, lane, ...classification }];
    });
}

export function movingAverage(bars: KlineBar[], period: number): Array<number | null> {
  if (!Number.isInteger(period) || period <= 0) {
    throw new Error("moving-average period must be a positive integer");
  }

  let rollingSum = 0;
  return bars.map((bar, index) => {
    rollingSum += bar.close;
    if (index >= period) rollingSum -= bars[index - period].close;
    if (index < period - 1) return null;
    return Number((rollingSum / period).toFixed(4));
  });
}

export function resolveKlineWindow(length: number, range: KlineViewRange, anchorIndex = -1): KlineWindow {
  if (length <= 0) return { startIndex: 0, endIndex: 0 };
  if (range === "ALL") return { startIndex: 0, endIndex: length - 1 };

  const windowSize = Math.min(RANGE_SIZE[range], length);
  const anchor = anchorIndex >= 0 && anchorIndex < length ? anchorIndex : length - 1;
  const lookAhead = Math.floor(windowSize * 0.2);
  const endIndex = Math.min(length - 1, anchor + lookAhead);
  const startIndex = Math.max(0, endIndex - windowSize + 1);

  return { startIndex, endIndex };
}
