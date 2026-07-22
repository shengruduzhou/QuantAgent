import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import type { EChartsOption } from "echarts";
import type { KeyboardEventHandler } from "react";
import { describe, expect, test, vi } from "vitest";
import type { KlineBar, Trade } from "../api/types";
import { CandlestickChart } from "./CandlestickChart";

let latestOption: EChartsOption = {};

vi.mock("./EChart", () => ({
  EChart: ({
    option,
    ariaLabel,
    onKeyDown,
  }: {
    option: EChartsOption;
    ariaLabel?: string;
    onKeyDown?: KeyboardEventHandler<HTMLDivElement>;
  }) => {
    latestOption = option;
    return <div role="application" aria-label={ariaLabel} tabIndex={0} onKeyDown={onKeyDown} />;
  },
}));

function createBars(count: number): KlineBar[] {
  const start = new Date("2025-01-01T00:00:00Z");
  return Array.from({ length: count }, (_, index) => {
    const date = new Date(start);
    date.setUTCDate(start.getUTCDate() + index);
    const close = 10 + index / 100;
    return {
      datetime: date.toISOString(),
      symbol: "000001.SZ",
      open: close - 0.05,
      high: close + 0.1,
      low: close - 0.1,
      close,
      volume: 1_000_000 + index,
    };
  });
}

function getZoom(): { startValue?: string; endValue?: string } {
  const zoom = Array.isArray(latestOption.dataZoom) ? latestOption.dataZoom[0] : latestOption.dataZoom;
  return (zoom ?? {}) as { startValue?: string; endValue?: string };
}

describe("CandlestickChart keyboard parity", () => {
  test("moves to latest and expands to all history", async () => {
    const bars = createBars(300);
    const trade: Trade = {
      id: "trade-1",
      datetime: bars[100].datetime,
      symbol: "000001.SZ",
      action: "BUY",
      price: bars[100].close,
      quantity: 100,
    };

    render(<CandlestickChart bars={bars} trades={[trade]} selectedTradeId={trade.id} />);

    const chart = screen.getByRole("application", { name: /000001.SZ K 线/ });
    const lastDate = bars.at(-1)?.datetime.slice(0, 10);
    const firstDate = bars[0].datetime.slice(0, 10);

    fireEvent.keyDown(chart, { key: "End" });
    await waitFor(() => expect(getZoom().endValue).toBe(lastDate));

    fireEvent.keyDown(chart, { key: "Home" });
    await waitFor(() => {
      expect(getZoom().startValue).toBe(firstDate);
      expect(getZoom().endValue).toBe(lastDate);
    });
  });
});
